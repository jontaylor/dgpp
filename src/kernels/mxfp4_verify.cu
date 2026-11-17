#include <cuda_bf16.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/mxfp4_verify.hpp"

namespace dgpp {
namespace {
constexpr int H = 5120, I = 1152, E = 384, slots = 24, B = 8, capacity = 64 * 6;
__global__ void partition(const int32_t* routes, const int32_t* ids, int32_t* reused) {
  const int lane = threadIdx.x;
  const auto* experts = routes + capacity * B;
  const int count = experts[capacity] / B;
  // Masked groups still occupy the same b12x reduction position in each
  // miss wave. Their sentinel routes suppress stores, as in the full packet.
  const int expert = lane < count ? experts[lane] : -1;
  int assignments = 0;
#pragma unroll
  for (int i = 0; i < slots; ++i) assignments += ids[i] == expert;
  const bool keep = lane < count && assignments > 1;
  const unsigned mask = __ballot_sync(0xffffffff, keep);
  if (keep) {
    const int dst = __popc(mask & ((1u << lane) - 1));
    reinterpret_cast<int4*>(reused)[2 * dst] = reinterpret_cast<const int4*>(routes)[2 * lane];
    reinterpret_cast<int4*>(reused)[2 * dst + 1] =
        reinterpret_cast<const int4*>(routes)[2 * lane + 1];
    reused[capacity * B + dst] = experts[lane];
  }
  if (lane == 0) reused[capacity * (B + 1)] = __popc(mask) * B;
}

// The prepared K16/N64 word packs the low nibbles of four pairs in
// bits 0..15 and their high nibbles in bits 16..31.
__device__ uint32_t pair(uint32_t word, int shift, uint8_t scale) {
  const unsigned c = word >> shift;
  const unsigned indices = (c & 7) | ((c >> 12) & 0x70);
  const unsigned lo = __byte_perm(0xc0800000, 0xc0804000, indices);
  const unsigned hi = __byte_perm(0x3f3f3f00, 0x40404040, indices);
  const unsigned bits = __byte_perm(lo, hi, 0x5140) | ((c & 0x00080008) << 12);
  const unsigned exponent = scale == 0 ? 0x0040u : scale == 255 ? 0x7fc0u : unsigned(scale) << 7;
  const unsigned scale_bits = exponent | (exponent << 16);
  unsigned out;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(out) : "r"(bits), "r"(scale_bits));
  return out;
}

// Four warps read distinct K32 groups. Each 16-byte load/thread reads a
// contiguous N64 slab per warp and supplies four independent MMA fragments.
template <bool Gate>
__global__ __launch_bounds__(256) void singleton_projection(const uint16_t* input, uint16_t* act,
                                                            float* out, const uint64_t* table,
                                                            const int32_t* routes) {
  constexpr int N = Gate ? 2 * I : H, K = Gate ? H : I, partitions = Gate ? 4 : 8;
  const int group = blockIdx.y;
  const int* experts = routes + capacity * B;
  if (group * B >= experts[capacity]) return;
  const int route = routes[group * B];
  if (route >= slots || routes[group * B + 1] < slots) return;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, e = experts[group];
  const int n0 = blockIdx.x * 64 + (Gate && warp >= 4 ? I : 0);
  const int row = lane / 4, two_k = 2 * (lane % 4);
  const int base = (Gate ? 0 : 3 * E) + 3 * e;
  const auto* w = reinterpret_cast<const uint4*>(table[base]);
  const auto* scales = reinterpret_cast<const uint8_t*>(table[base + 1]);
  const auto* x = input + (Gate ? route / 6 : route) * K;
  float acc[4][4] = {};
  for (int k = (warp % partitions) * 32; k < K; k += partitions * 32) {
    // The packed scales of the four m16 fragments occupy eight contiguous
    // bytes: (low0, low1, high0, high1) for each pair of fragments.
    const uint2 scales8 = *reinterpret_cast<const uint2*>(scales + (k / 32) * N + n0 + row * 8);
    const int next = k + partitions * 64;
    if (next < K && lane % 8 == 0) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int slab = (next / 16 + half) * (N / 64) + n0 / 64;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(w + slab * 32 + lane));
      }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int slab = (k / 16 + half) * (N / 64) + n0 / 64;
      const uint4 lines = w[slab * 32 + lane];
      const uint32_t words[4] = {lines.x, lines.y, lines.z, lines.w};
      const int at = k + half * 16 + two_k;
      const unsigned b0 = *reinterpret_cast<const unsigned*>(x + at);
      const unsigned b1 = *reinterpret_cast<const unsigned*>(x + at + 8);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const unsigned sc = (t < 2 ? scales8.x : scales8.y) >> ((t % 2) * 8);
        const uint8_t s0 = sc, s1 = sc >> 16;
        const unsigned a0 = pair(words[t], 0, s0), a1 = pair(words[t], 8, s1);
        const unsigned a2 = pair(words[t], 4, s0), a3 = pair(words[t], 12, s1);
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[t][0]), "+f"(acc[t][1]), "+f"(acc[t][2]), "+f"(acc[t][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
  }
  __shared__ float partial[8][64];
  if (lane % 4 == 0) {
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      partial[warp][t * 16 + row] = acc[t][0];
      partial[warp][t * 16 + row + 8] = acc[t][2];
    }
  }
  __syncthreads();
  if (tid >= 64) return;
  float sum = 0.f;
#pragma unroll
  for (int p = 0; p < partitions; ++p) sum += partial[p][tid];
  const int col = blockIdx.x * 64 + tid;
  if constexpr (Gate) {
    float u = 0.f;
#pragma unroll
    for (int p = 4; p < 8; ++p) u += partial[p][tid];
    const float gate = fminf(10.f, bf16_bits_to_float(float_to_bf16_bits(sum)));
    u = fminf(10.f, fmaxf(-10.f, bf16_bits_to_float(float_to_bf16_bits(u))));
    const auto silu = float_to_bf16_bits(gate / (1.f + expf(-gate)));
    act[route * I + col] = float_to_bf16_bits(bf16_bits_to_float(silu) * u);
  } else {
    out[(route + route / 6) * H + col] = sum;
  }
}
}  // namespace

void launch_mxfp4_verify_singletons(const uint16_t* hidden, uint16_t* act, float* out,
                                    const uint64_t* table, const int32_t* ids,
                                    const int32_t* routes, int32_t* reused, cudaStream_t stream) {
  partition<<<1, 32, 0, stream>>>(routes, ids, reused);
  singleton_projection<true>
      <<<dim3(I / 64, slots), 256, 0, stream>>>(hidden, act, out, table, routes);
  singleton_projection<false>
      <<<dim3(H / 64, slots), 256, 0, stream>>>(act, nullptr, out, table, routes);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
