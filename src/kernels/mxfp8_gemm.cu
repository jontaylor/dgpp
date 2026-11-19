#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/cuda_check.hpp"
#include "kernels/mxfp8_gemm.hpp"
#include "kernels/mxfp8_quant.cuh"
#include "kernels/pdl.cuh"

namespace dgpp {
namespace {
constexpr int kThreads = 128, kStages = 3;

template <int BM, int BN, bool Wide = false>
struct Tile {
  static constexpr int warps = BM == 16 ? (BN < 16 ? 1 : BN / 16) : (Wide ? 8 : 4);
  static constexpr int threads = warps * 32;
  static constexpr int bk = BM == 16 && BN == 8 ? 512 : BM == 16 && BN == 16 ? 256 : 64;
  static constexpr int stride = bk + 16;
};

template <bool SwiGLU>
__global__ void mxfp8_quantize_kernel(const uint16_t* x, size_t stride, int k, uint8_t* codes,
                                      uint8_t* scales, const uint16_t* up = nullptr,
                                      float alpha = 0.f, float limit = 0.f, float minimum = 0.f) {
  dgpp_pdl_wait();
  // Eight lanes cooperate on each group; each lane loads four adjacent BF16s.
  const int row = blockIdx.y;
  const int column = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (column >= k) return;  // k % 32 guarantees complete participating groups.
  const uint2 bits =
      *reinterpret_cast<const uint2*>(x + static_cast<size_t>(row) * stride + column);
  float values[4] = {__uint_as_float(bits.x << 16), __uint_as_float(bits.x & 0xffff0000u),
                     __uint_as_float(bits.y << 16), __uint_as_float(bits.y & 0xffff0000u)};
  if constexpr (SwiGLU) {
    // Pinned SGLang fused producer: gate * sigmoid(alpha * gate), followed
    // by (up + 1), BF16 rounding, then MXFP8. Keep Triton's exp2/div.full
    // evaluation, rather than substituting g / (1 + expf(-alpha * g)).
    const uint2 ub = *reinterpret_cast<const uint2*>(up + static_cast<size_t>(row) * stride + column);
    const float ups[4] = {__uint_as_float(ub.x << 16), __uint_as_float(ub.x & 0xffff0000u),
                          __uint_as_float(ub.y << 16), __uint_as_float(ub.y & 0xffff0000u)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float g = fminf(values[i], limit), u = fminf(fmaxf(ups[i], -limit), limit);
      const float exponent = __fmul_rn(__fmul_rn(g, -alpha), __uint_as_float(0x3fb8aa3bu));
      float e, sigmoid;
      asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(exponent));
      const float denominator = __fadd_rn(1.f, e);
      asm("div.full.f32 %0, %1, %2;" : "=f"(sigmoid) : "f"(1.f), "f"(denominator));
      values[i] = __bfloat162float(__float2bfloat16_rn(
          __fmul_rn(__fmul_rn(g, sigmoid), __fadd_rn(u, 1.f))));
    }
  }
  const size_t offset = static_cast<size_t>(row) * k + column;
  mxfp8_quantize_four(values, minimum, codes + offset, scales + offset / 32);
}

__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(src),
               "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() {
  asm volatile("cp.async.commit_group;");
}
template <int Count>
__device__ __forceinline__ void wait_pending() {
  asm volatile("cp.async.wait_group %0;" ::"n"(Count));
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                    uint32_t as, uint32_t bs) {
  // PTX/CUTLASS SM120 scale layout: A scale row = lane/4 + 8*(lane%2),
  // B scale row = lane/4; the byte/thread selectors are both zero.
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
      "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11,%12}, {%13}, {%14,%15};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(as),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(bs),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
}

// A warp streams 16 weight rows over half of K. B is the same
// four (up to eight) activation rows for every output tile in the group.
// All four TP2 groups share the grid; the two warp partials meet within the CTA.
constexpr int woa_warps = 2, woa_step = 512;
// Eight live token rows use the complete B fragment. Bring their activation
// window into registers while the asynchronous weight copies are outstanding.
struct WoaActivationWindow {
  uint2 codes[woa_step / 32];
  uint32_t scales[woa_step / 128];
};

__device__ __forceinline__ uint2 woa_token_pair(const uint8_t* token) {
  return make_uint2(*reinterpret_cast<const uint32_t*>(token), *reinterpret_cast<const uint32_t*>(token + 16));
}

template <bool AlignedScales, int FixedRows>
__global__ __launch_bounds__(woa_warps * 32) void woa_decode_kernel(
    const uint8_t* codes, const uint8_t* scales, const uint8_t* weight,
    const float* weight_scale, uint16_t* out, int rows, int groups, int n, bool early_weight_copy) {
  dgpp_pdl_preamble();
  if (!early_weight_copy) dgpp_pdl_wait();
  constexpr int k = 4096;
  if constexpr (FixedRows != 0) rows = FixedRows;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int row8 = lane / 4, part = lane % 4;
  const int column = blockIdx.x * 16;
  const int group = column / n;
  const int span = k / woa_warps;
  const int k0 = warp * span;
  // 16-byte padding separates the row banks for the fragment reads.
  constexpr int pitch = woa_step + 16;
  __shared__ __align__(16) uint8_t stage[woa_warps][2][16][pitch];
  __shared__ float partial[woa_warps][16][8];
  __shared__ uint64_t ready[woa_warps][2];
  auto barrier_address = [&](int slot) {
    return static_cast<unsigned>(__cvta_generic_to_shared(&ready[warp][slot]));
  };
  // Each warp owns two independent stages. One arrival and 16 completed
  // row copies advance the barrier by one phase on each use of a stage.
  if (lane < 2)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barrier_address(lane)) : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncwarp();

  auto issue = [&](int base, int slot) {
    const unsigned bar = barrier_address(slot);
    if (lane == 0)
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                   :: "r"(bar), "r"(16 * woa_step) : "memory");
    __syncwarp();
    if (lane < 16) {
      const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(&stage[warp][slot][lane][0]));
      const uint8_t* src = weight + size_t(column + lane) * k + base;
      asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(dst), "l"(src), "r"(woa_step), "r"(bar) : "memory");
    }
    // On the last window, point the L2 lookahead at the first window of a
    // later output tile. GB10 admits two of these blocks on each of 48 SMs.
    constexpr int ahead = 96;
    const bool next_window = base + woa_step < k0 + span;
    if (next_window || blockIdx.x + ahead < gridDim.x) {
      const int next_column = next_window ? column : column + 16 * ahead;
      const int next_k = next_window ? base + woa_step : k0;
#pragma unroll
      for (int v = lane; v < 16 * (woa_step / 128); v += 32) {
        const int row = v / (woa_step / 128), off = (v % (woa_step / 128)) * 128;
        const uint8_t* src = weight + size_t(next_column + row) * k + next_k + off;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(src));
      }
    }
  };
  const bool live = row8 < rows;
  const size_t token_row = (size_t(live ? row8 : 0) * groups + group) * k;
  float acc[4]{};
  issue(k0, 0);
  // The first compulsory copy reads only the resident projection. Finish must
  // publish every activation code and scale before this grid consumes them.
  if (early_weight_copy) dgpp_pdl_wait();
  for (int step = 0; step < span / woa_step; ++step) {
    const int base = k0 + step * woa_step, slot = step & 1;
    if (step + 1 < span / woa_step) issue(base + woa_step, slot ^ 1);
    const size_t token_base = token_row + base;
    WoaActivationWindow activation;
    if constexpr (FixedRows == 8) {
#pragma unroll
      for (int q = 0; q < woa_step / 32; ++q) {
        const size_t off = token_base + q * 32;
        activation.codes[q] = woa_token_pair(codes + off + part * 4);
        if (q % 4 == 0) activation.scales[q / 4] = *reinterpret_cast<const uint32_t*>(scales + off / 32);
      }
      // Keep these independent global reads ahead of the weight-ready wait.
      asm volatile("" ::: "memory");
    }
    unsigned done;
    do {
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                   : "=r"(done) : "r"(barrier_address(slot)), "r"((step / 2) & 1) : "memory");
    } while (!done);
    __syncwarp();
#pragma unroll
    for (int chunk = 0; chunk < woa_step; chunk += 128) {
      uint32_t token_scales = 0;
      if constexpr (FixedRows == 8) token_scales = activation.scales[chunk / 128];
      else if (live) token_scales = *reinterpret_cast<const uint32_t*>(scales + (token_base + chunk) / 32);
      const float* ws = weight_scale + size_t(column / 32) * (k / 32) + (base + chunk) / 32;
      float4 four;
      if constexpr (AlignedScales) four = *reinterpret_cast<const float4*>(ws);
      else four = make_float4(ws[0], ws[1], ws[2], ws[3]);
      const float weight_scales[4] = {four.x, four.y, four.z, four.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int kk = chunk + j * 32;
        const uint8_t* source = &stage[warp][slot][lane % 16][kk + (lane / 16) * 16];
        const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(source));
        uint32_t a[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(address));
        uint2 token{};
        if constexpr (FixedRows == 8) token = activation.codes[kk / 32];
        else if (live) token = woa_token_pair(codes + token_base + kk + part * 4);
        mma(acc, a, token.x, token.y, (__float_as_uint(weight_scales[j]) >> 23) & 255, (token_scales >> (8 * j)) & 255);
      }
    }
    __syncwarp();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) partial[warp][row8 + (i / 2) * 8][part * 2 + i % 2] = acc[i];
  __syncthreads();
  for (int cell = threadIdx.x; cell < rows * 16; cell += woa_warps * 32) {
    const int r = cell / 16, col = cell % 16;
    float sum = partial[0][col][r];
#pragma unroll
    for (int w = 1; w < woa_warps; ++w) sum += partial[w][col][r];
    const size_t offset = size_t(r) * groups * n + column + col;
    out[offset] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
  }
}

template <typename Out, int BM, int BN, bool Grid32 = false, typename Scale = uint8_t,
          bool Wide = false, bool Prefetch = false>
__global__ __launch_bounds__((Tile<BM, BN, Wide>::threads)) void mxfp8_gemm_kernel(
    const uint8_t* a, const uint8_t* a_scales, const uint8_t* w, const Scale* w_scales, Out* out,
    int m, int n, int k, size_t out_stride) {
  dgpp_pdl_wait();
  constexpr int kStages = Wide ? 2 : 3;
  constexpr int threads = Tile<BM, BN, Wide>::threads;
  constexpr int BK = Tile<BM, BN, Wide>::bk, stride = Tile<BM, BN, Wide>::stride;
  constexpr int warp_m = BM == 16 ? 1 : 2, warp_n = Tile<BM, BN, Wide>::warps / warp_m;
  constexpr int MT = BM / warp_m / 16, NT = BN / warp_n / 8;
  constexpr int slot_bytes = (BM + BN) * stride;
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int m_tiles = (m + BM - 1) / BM;
  const int m0 = (blockIdx.x % m_tiles) * BM, n0 = (blockIdx.x / m_tiles) * BN;
  const int row_base = (warp / warp_n) * (BM / warp_m);
  const int col_base = (warp % warp_n) * (BN / warp_n);
  const int group = lane / 4, part = lane % 4;
  const int steps = (k + BK - 1) / BK, scale_stride = k / 32;
  auto issue = [&](int step, int slot) {
#pragma unroll
    for (int chunk = tid; chunk < (BM + BN) * (BK / 16); chunk += threads) {
      const int row = chunk / (BK / 16), column = (chunk % (BK / 16)) * 16;
      const bool is_a = row < BM;
      const int global_row = is_a ? m0 + row : n0 + row - BM;
      const bool valid = global_row < (is_a ? m : n) && step * BK + column < k;
      const uint8_t* src = (is_a ? a : w) + static_cast<size_t>(valid ? global_row : 0) * k;
      copy16(smem + slot * slot_bytes + row * stride + column,
               valid ? src + step * BK + column : src, valid);
      if constexpr (Prefetch) {
        if (!is_a && valid && column == 0 && step % 2 == 0 && (step + 2) * BK < k) {
          const auto* next = src + (step + 2) * BK;
          asm volatile("prefetch.global.L2 [%0];" ::"l"(next));
        }
      }
    }
  };
  float accum[MT][NT][4] = {};
#pragma unroll
  for (int stage = 0; stage < kStages - 1; ++stage) {
    if (stage < steps) issue(stage, stage);
    commit();
  }
  for (int step = 0; step < steps; ++step) {
    wait_pending<kStages - 2>();
    __syncthreads();
    if (step + kStages - 1 < steps) issue(step + kStages - 1, (step + kStages - 1) % kStages);
    commit();
    const uint8_t* sa = smem + (step % kStages) * slot_bytes;
    const uint8_t* sw = sa + BM * stride;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 32) {
      if constexpr (BK > 64)
        if (step * BK + kk >= k) break;
      uint32_t af[MT][4], bf[NT][2], as[MT], bs[NT];
      const int scale_column = (step * BK + kk) / 32;
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int row = row_base + i * 16 + group;
        const uint8_t* p = sa + row * stride + kk + part * 4;
        af[i][0] = *reinterpret_cast<const uint32_t*>(p);
        af[i][1] = *reinterpret_cast<const uint32_t*>(p + 8 * stride);
        af[i][2] = *reinterpret_cast<const uint32_t*>(p + 16);
        af[i][3] = *reinterpret_cast<const uint32_t*>(p + 8 * stride + 16);
        const int scale_row = m0 + row + (lane & 1) * 8;
        as[i] = scale_row < m
                    ? a_scales[static_cast<size_t>(scale_row) * scale_stride + scale_column]
                    : 127;
      }
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int row = col_base + j * 8 + group;
        const uint8_t* p = sw + row * stride + kk + part * 4;
        bf[j][0] = *reinterpret_cast<const uint32_t*>(p);
        bf[j][1] = *reinterpret_cast<const uint32_t*>(p + 16);
        const int scale_row = n0 + row;
        bs[j] = 127;
        if (scale_row < n) {
          const auto scale =
              w_scales[static_cast<size_t>(Grid32 ? scale_row / 32 : scale_row) * scale_stride +
                       scale_column];
          if constexpr (std::is_same_v<Scale, float>)
            // DSV loader expands E8M0 to exact powers of two, including byte 0.
            bs[j] = (__float_as_uint(scale) >> 23) & 255;
          else
            bs[j] = scale;
        }
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) mma(accum[i][j], af[i], bf[j][0], bf[j][1], as[i], bs[j]);
    }
    // No warp may reuse a ring slot until every warp has consumed it.
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        const int row = m0 + row_base + i * 16 + group + (value / 2) * 8;
        const int column = n0 + col_base + j * 8 + part * 2 + value % 2;
        if (row < m && column < n) {
          if constexpr (std::is_same_v<Out, float>)
            out[static_cast<size_t>(row) * out_stride + column] = accum[i][j][value];
          else
            out[static_cast<size_t>(row) * out_stride + column] =
                __bfloat16_as_ushort(__float2bfloat16_rn(accum[i][j][value]));
        }
      }
}

template <typename Out, int BM, int BN>
void launch(const uint8_t* a, const uint8_t* as, const uint8_t* w, const uint8_t* ws, Out* out,
            int m, int n, int k, cudaStream_t stream, size_t stride) {
  constexpr int smem = kStages * (BM + BN) * Tile<BM, BN>::stride;
  const unsigned blocks = static_cast<unsigned>(((m + BM - 1) / BM) * ((n + BN - 1) / BN));
  mxfp8_gemm_kernel<Out, BM, BN>
      <<<blocks, Tile<BM, BN>::threads, smem, stream>>>(a, as, w, ws, out, m, n, k, stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <typename Out>
void dispatch(const uint8_t* a, const uint8_t* as, const uint8_t* w, const uint8_t* ws, Out* out,
              int m, int n, int k, cudaStream_t stream, size_t stride, int output_tile) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0 || k % 64)
    throw std::invalid_argument("MXFP8 GEMM: k must be a positive multiple of 64");
  if (stride == 0) stride = static_cast<size_t>(n);
  if (stride < static_cast<size_t>(n))
    throw std::invalid_argument("MXFP8 GEMM: output stride is too small");
  if (output_tile != 0 && output_tile != 8 && output_tile != 16 && output_tile != 32 && output_tile != 64)
    throw std::invalid_argument("MXFP8 GEMM: output tile must be 0, 8, 16, 32 or 64");
  // Rotating-weight measurements at MiniMax's projection shapes: narrow
  // column tiles spread short batches across more SMs, while longer K
  // stages amortize their pipeline waits. Short-K shared-down and the
  // 2048-output query projection favor the sixteen-column variant.
  if (output_tile == 0 && m <= 16) output_tile = k < 1024 || n == 2048 ? 16 : 8;
  if (output_tile == 8)
    launch<Out, 16, 8>(a, as, w, ws, out, m, n, k, stream, stride);
  else if (output_tile == 16)
    launch<Out, 16, 16>(a, as, w, ws, out, m, n, k, stream, stride);
  else if (output_tile == 32)
    launch<Out, 16, 32>(a, as, w, ws, out, m, n, k, stream, stride);
  else if (m <= 16 || output_tile == 64)
    launch<Out, 16, 64>(a, as, w, ws, out, m, n, k, stream, stride);
  else
    launch<Out, 64, 64>(a, as, w, ws, out, m, n, k, stream, stride);
}
}  // namespace

void launch_mxfp8_woa_decode(const uint8_t* packed, const uint8_t* w, const float* ws,
                             uint16_t* out, int rows, int groups, int n, int k, cudaStream_t stream, bool early_weight_copy) {
  if (rows < 1 || rows > 8 || groups < 1 || n <= 0 || n % 32 || k != 4096)
    throw std::invalid_argument("wo_a MXFP8 decode: unsupported projection");
  auto launch = [&]<bool AlignedScales>() {
    auto emit = [&]<int FixedRows>() {
      woa_decode_kernel<AlignedScales, FixedRows><<<groups * n / 16, woa_warps * 32, 0, stream>>>(
          packed, packed + size_t(rows) * groups * k, w, ws, out, rows, groups, n, early_weight_copy);
    };
    switch (rows) {
      case 1: return emit.template operator()<1>();
      case 4: return emit.template operator()<4>();
      case 6: return emit.template operator()<6>();
      case 8: return emit.template operator()<8>();
      default: return emit.template operator()<0>();
    }
  };
  if (reinterpret_cast<uintptr_t>(ws) % alignof(float4) == 0) launch.template operator()<true>();
  else launch.template operator()<false>();
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_mxfp8_quantize_rows(const uint16_t* x, size_t stride, int rows, int k, uint8_t* codes,
                                uint8_t* scales, cudaStream_t stream) {
  if (rows <= 0) return;
  if (k <= 0 || k % 32 || stride < static_cast<size_t>(k) || stride % 4)
    throw std::invalid_argument("MXFP8 quantize: requires 32-wide groups and aligned input rows");
  const dim3 grid(static_cast<unsigned>((k + 4 * kThreads - 1) / (4 * kThreads)),
                  static_cast<unsigned>(rows));
  mxfp8_quantize_kernel<false><<<grid, kThreads, 0, stream>>>(x, stride, k, codes, scales);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_minimax_swiglu_mxfp8_rows(const uint16_t* gate, const uint16_t* up, size_t stride,
                                      int rows, int k, float alpha, float limit, uint8_t* codes,
                                      uint8_t* scales, cudaStream_t stream) {
  if (rows <= 0) return;
  if (k <= 0 || k % 32 || stride < static_cast<size_t>(k) || stride % 4)
    throw std::invalid_argument("MiniMax SwiGLU MXFP8: requires 32-wide groups and aligned input rows");
  const dim3 grid(static_cast<unsigned>((k + 4 * kThreads - 1) / (4 * kThreads)),
                  static_cast<unsigned>(rows));
  mxfp8_quantize_kernel<true><<<grid, kThreads, 0, stream>>>(gate, stride, k, codes, scales, up,
                                                          alpha, limit);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_mxfp8_gemm_bf16(const uint8_t* a, const uint8_t* as, const uint8_t* w,
                            const uint8_t* ws, uint16_t* out, int m, int n, int k,
                            cudaStream_t stream, size_t stride, int output_tile) {
  dispatch(a, as, w, ws, out, m, n, k, stream, stride, output_tile);
}
void launch_mxfp8_gemm_f32(const uint8_t* a, const uint8_t* as, const uint8_t* w, const uint8_t* ws,
                           float* out, int m, int n, int k, cudaStream_t stream, size_t stride,
                           int output_tile) {
  dispatch(a, as, w, ws, out, m, n, k, stream, stride, output_tile);
}
// Native dense path for DSV4.1's 32x32 E8M0 scale grid. Quantization
// consumes the caller's existing GEMM workspace and stays on its stream.
template <class Out, class Scale>
bool native_grid32(const uint16_t* x, size_t x_stride, const uint8_t* w, const Scale* ws, Out* out,
                   int m, int n, int k, cudaStream_t stream, size_t stride, void* workspace,
                   size_t bytes) {
  if (m <= 16 || m > 65535 || n <= 0 || k <= 0 || k % 64 || x_stride < size_t(k) || x_stride % 4 ||
      (stride && stride < size_t(n)) || !workspace || (reinterpret_cast<uintptr_t>(x) % 8) ||
      (reinterpret_cast<uintptr_t>(workspace) % 16))
    return false;
  const size_t codes = size_t(m) * k, scales = codes / 32;
  if (bytes < codes + scales) return false;
  auto* a = static_cast<uint8_t*>(workspace);
  auto* as = a + codes;
  const dim3 quant_grid(unsigned((k + 4 * kThreads - 1) / (4 * kThreads)), unsigned(m));
  mxfp8_quantize_kernel<false>
      <<<quant_grid, kThreads, 0, stream>>>(x, x_stride, k, a, as, nullptr, 0.f, 0.f, 1e-4f);
  constexpr int BM = 64, BN = 128;
  const unsigned blocks = unsigned(((m + BM - 1) / BM) * ((n + BN - 1) / BN));
  static const bool wide = [] {
    const char* v = std::getenv("DGPP_DSV41_NATIVE_FP8_WIDE");
    return v && v[0] == '1';
  }();
  static const bool prefetch = [] {
    const char* v = std::getenv("DGPP_DSV41_NATIVE_FP8_PREFETCH");
    return v && v[0] == '1';
  }();
  auto run = [&]<bool Wide, bool Prefetch>() {
    constexpr int smem = (Wide ? 2 : kStages) * (BM + BN) * Tile<BM, BN, Wide>::stride;
    mxfp8_gemm_kernel<Out, BM, BN, true, Scale, Wide, Prefetch>
        <<<blocks, Tile<BM, BN, Wide>::threads, smem, stream>>>(a, as, w, ws, out, m, n, k,
                                                                stride ? stride : size_t(n));
  };
  if (wide) {
    if (prefetch) run.template operator()<true,true>();
    else run.template operator()<true,false>();
  } else {
    if (prefetch) run.template operator()<false,true>();
    else run.template operator()<false,false>();
  }
  DGPP_CUDA_OK(cudaGetLastError());
  return true;
}
bool launch_mxfp8_grid32_bf16(const uint16_t* x, size_t xs, const uint8_t* w, const uint8_t* ws,
                              uint16_t* out, int m, int n, int k, cudaStream_t stream, size_t os,
                              void* scratch, size_t bytes) {
  return native_grid32(x, xs, w, ws, out, m, n, k, stream, os, scratch, bytes);
}
bool launch_mxfp8_grid32_f32(const uint16_t* x, size_t xs, const uint8_t* w, const uint8_t* ws,
                             float* out, int m, int n, int k, cudaStream_t stream, size_t os,
                             void* scratch, size_t bytes) {
  return native_grid32(x, xs, w, ws, out, m, n, k, stream, os, scratch, bytes);
}

bool launch_mxfp8_grid32_expanded_bf16(const uint16_t* x, size_t xs, const uint8_t* w,
                                       const float* ws, uint16_t* out, int m, int n, int k,
                                       cudaStream_t stream, size_t os, void* scratch,
                                       size_t bytes) {
  return native_grid32(x, xs, w, ws, out, m, n, k, stream, os, scratch, bytes);
}

}  // namespace dgpp
