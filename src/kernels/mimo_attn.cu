#include <math_constants.h>
#include <mma.h>
#include <stdexcept>

#include <cuda_bf16.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/mimo_attn.hpp"

namespace dgpp {
namespace {
// Keep short-context trajectories on the established scalar path; the
// coalesced/split path pays off as history grows. This is a device-side
// choice so mixed-length requests share one allocation-free graph.
constexpr int decode_split_min_keys = 512;
__device__ int64_t cache_slot(MimoAttentionShape s, int64_t p) {
  if (!s.window) return p;
  return (s.capacity & (s.capacity - 1)) == 0 ? p & (s.capacity - 1) : p % s.capacity;
}
__device__ float rb(float v) {
  return bf16_bits_to_float(float_to_bf16_bits(v));
}
__device__ bool invalid(MimoAttentionShape s, int64_t p) {
  return p >= 1048576 || (s.window == 0 && p >= s.capacity);
}

__global__ void append_kernel(MimoAttentionShape s, const uint16_t* fused, const float* freq,
                              const int64_t* positions, uint16_t* q, uint16_t* kc, uint16_t* vc,
                              int32_t* status, bool shared_cache, const int32_t* request_ids) {
  const int r = blockIdx.y, head = blockIdx.x, d = threadIdx.x;
  const int64_t p = positions[r];
  const bool bad = invalid(s, p);
  if (head == 0 && d == 0) status[r] = bad ? 1 : 0;
  if (p < 0 || bad) return;
  const int q_width = s.q_heads * 192, k_width = s.kv_heads * 192, v_width = s.kv_heads * 128;
  const auto* source = fused + int64_t(r) * (q_width + k_width + v_width);
  const int64_t slot = int64_t(shared_cache ? 0 : (request_ids ? request_ids[r] : r)) * s.capacity +
                       cache_slot(s, p);
  if (head >= s.q_heads + s.kv_heads) {
    if (d < 128) {
      const int h = head - s.q_heads - s.kv_heads;
      vc[slot * v_width + h * 128 + d] = float_to_bf16_bits(
          __fmul_rn(bf16_bits_to_float(source[q_width + k_width + h * 128 + d]), 0.707f));
    }
    return;
  }
  if (d >= 192) return;
  const bool is_q = head < s.q_heads;
  const int h = is_q ? head : head - s.q_heads;
  source += (is_q ? 0 : q_width) + h * 192;
  auto* dest = is_q ? q + int64_t(r) * q_width + h * 192 : kc + slot * k_width + h * 192;
  float value = bf16_bits_to_float(source[d]);
  if (d < 64) {
    const float angle = __fmul_rn(static_cast<float>(p), freq[d % 32]);
    const float cos = rb(cosf(angle)), sin = rb(sinf(angle));
    const float mate = bf16_bits_to_float(source[d < 32 ? d + 32 : d - 32]);
    value = rb(__fadd_rn(rb(__fmul_rn(value, cos)), rb(__fmul_rn(d < 32 ? -mate : mate, sin))));
  }
  dest[d] = float_to_bf16_bits(value);
}

__device__ float score(MimoAttentionShape s, const uint16_t* q, const uint16_t* kc, int r, int h,
                       int kh, int64_t pos, bool shared_cache, const int32_t* request_ids) {
  const int64_t slot = int64_t(shared_cache ? 0 : (request_ids ? request_ids[r] : r)) * s.capacity +
                       cache_slot(s, pos);
  float dot = 0;
  for (int d = 0; d < 192; ++d)
    dot = __fmaf_rn(bf16_bits_to_float(q[(int64_t(r) * s.q_heads + h) * 192 + d]),
                    bf16_bits_to_float(kc[slot * s.kv_heads * 192 + kh * 192 + d]), dot);
  // 192^-0.5 narrowed once, as in the reference Python scalar multiplier.
  return rb(__fmul_rn(rb(dot), 0.07216878364870322f));
}

// One warp computes a 16-query by 16-key QK tile with BF16 tensor cores.
// Absolute positions retain the same causal/ring mapping as the scalar path.
__global__ void score_tiles(MimoAttentionShape s, const uint16_t* q, const uint16_t* kc,
                            const int64_t* positions, float* scores, int first_key, int end_key) {
  using namespace nvcuda;
  const int row0 = blockIdx.y * 16, head = blockIdx.z;
  const int key0 = first_key + blockIdx.x * 16;
  const int lane = threadIdx.x;
  const int last_row = min(row0 + 15, s.requests - 1);
  const int64_t last = positions[last_row];
  const int64_t first = positions[row0];
  if (key0 > last || (s.window && key0 + 15 < first - s.window + 1)) return;
  __shared__ __align__(32) __nv_bfloat16 a[16 * 192], b[16 * 192];
  __shared__ __align__(32) float c[16 * 16];
  const int kh = head / (s.q_heads / s.kv_heads);
  for (int i = lane; i < 16 * 192; i += 32) {
    const int row = row0 + i / 192, dim = i % 192;
    const int key = key0 + i / 192;
    reinterpret_cast<uint16_t*>(a)[i] =
        row < s.requests ? q[(int64_t(row) * s.q_heads + head) * 192 + dim] : 0;
    reinterpret_cast<uint16_t*>(b)[i] =
        key < end_key ? kc[(int64_t(cache_slot(s, key)) * s.kv_heads + kh) * 192 + dim] : 0;
  }
  __syncwarp();
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
  wmma::fill_fragment(cf, 0.0f);
  for (int k = 0; k < 192; k += 16) {
    wmma::load_matrix_sync(af, a + k, 192);
    wmma::load_matrix_sync(bf, b + k, 192);
    wmma::mma_sync(cf, af, bf, cf);
  }
  wmma::store_matrix_sync(c, cf, 16, wmma::mem_row_major);
  __syncwarp();
  for (int i = lane; i < 256; i += 32) {
    const int row = row0 + i / 16, key = key0 + i % 16;
    if (row < s.requests && key < end_key) {
      const int64_t p = positions[row];
      if (key <= p && (!s.window || key >= p - s.window + 1))
        scores[(int64_t(row) * s.q_heads + head) * s.capacity + cache_slot(s, key)] =
            rb(__fmul_rn(rb(c[i]), 0.07216878364870322f));
    }
  }
}

// Decode tiles share coalesced Q/K loads while preserving each score's
// scalar FP32 FMA order. Tensor QK changes BF16 rounding at rare ties and
// can switch routed experts in real layers, so it is not used for decode.
__global__ void decode_score_tiles(MimoAttentionShape s, const uint16_t* q, const uint16_t* kc,
                                   const int64_t* positions, const int32_t* request_ids,
                                   float* scores) {
  const int row = blockIdx.z, kh = blockIdx.y, tid = threadIdx.x;
  const int64_t pos = positions[row];
  if (pos < decode_split_min_keys - 1 || invalid(s, pos)) return;
  const int group = s.q_heads / s.kv_heads;
  const int first = s.window ? max(0, int(pos) - s.window + 1) : 0;
  const int initial_key = first + blockIdx.x * 16;
  if (initial_key > pos) return;
  const int request = request_ids ? request_ids[row] : row;
  __shared__ float a[16 * 192], b[192 * 16];
  for (int key0 = initial_key; key0 <= pos; key0 += gridDim.x * 16) {
    for (int i = tid; i < 16 * 192; i += 256) {
      const int h = i / 192, dim = i % 192, key = key0 + h;
      a[i] = h < group
                 ? bf16_bits_to_float(q[(int64_t(row) * s.q_heads + kh * group + h) * 192 + dim])
                 : 0;
      const int64_t slot = int64_t(request) * s.capacity + cache_slot(s, key);
      b[dim * 16 + h] =
          key <= pos ? bf16_bits_to_float(kc[(slot * s.kv_heads + kh) * 192 + dim]) : 0;
    }
    __syncthreads();
    const int h = tid / 16, key = key0 + tid % 16;
    if (h < group && key <= pos) {
      float dot = 0;
      for (int dim = 0; dim < 192; ++dim)
        dot = __fmaf_rn(a[h * 192 + dim], b[dim * 16 + tid % 16], dot);
      scores[(int64_t(row) * s.q_heads + kh * group + h) * s.capacity + cache_slot(s, key)] =
          rb(__fmul_rn(rb(dot), 0.07216878364870322f));
    }
    __syncthreads();
  }
}

__global__ void attention_kernel(MimoAttentionShape s, const uint16_t* q, const uint16_t* kc,
                                 const uint16_t* vc, const int64_t* positions,
                                 const uint16_t* sinks, uint16_t* out, bool shared_cache,
                                 float* scores, bool precomputed = false,
                                 uint16_t* probabilities = nullptr,
                                 const int32_t* request_ids = nullptr,
                                 bool parallel_softmax = false, bool adaptive_decode = false) {
  const int r = blockIdx.y, h = blockIdx.x, lane = threadIdx.x;
  const int64_t pos = positions[r];
  float* saved = scores ? scores + (int64_t(r) * s.q_heads + h) * s.capacity : nullptr;
  auto* dest = out + (int64_t(r) * s.q_heads + h) * 128;
  if (pos < 0 || invalid(s, pos)) {
    dest[lane] = 0;
    return;
  }
  if (adaptive_decode && pos < decode_split_min_keys - 1) {
    precomputed = false;
    probabilities = nullptr;
  }
  const int64_t begin = s.window && pos >= s.window ? pos - s.window + 1 : 0;
  const int kh = h / (s.q_heads / s.kv_heads);
  __shared__ float values[128];
  __shared__ float max_score, denominator;
  // Max is exact under reassociation. Each lane walks its scores, then
  // four warps reduce once; retain the original FP32 denominator order.
  float local_max = -CUDART_INF_F;
  for (int64_t t = begin + lane; t <= pos; t += 128) {
    const float value = precomputed ? saved[cache_slot(s, t)]
                                    : score(s, q, kc, r, h, kh, t, shared_cache, request_ids);
    if (saved) saved[cache_slot(s, t)] = value;
    local_max = fmaxf(local_max, value);
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
  if ((lane & 31) == 0) values[lane >> 5] = local_max;
  __syncthreads();
  if (lane == 0) {
    max_score = sinks ? bf16_bits_to_float(sinks[h]) : -CUDART_INF_F;
    for (int warp = 0; warp < 4; ++warp) max_score = fmaxf(max_score, values[warp]);
    denominator = 0;
  }
  __syncthreads();
  if (precomputed && parallel_softmax) {
    // Prefill softmax sums positive FP32 exponentials across lanes. This
    // reassociates only the denominator; scores, differences, probabilities
    // and final output keep their BF16 rounding points. Decode stays serial.
    float local_sum = 0;
    for (int64_t t = begin + lane; t <= pos; t += 128)
      local_sum = __fadd_rn(local_sum, expf(rb(saved[cache_slot(s, t)] - max_score)));
    for (int offset = 16; offset > 0; offset >>= 1)
      local_sum = __fadd_rn(local_sum, __shfl_down_sync(0xffffffffu, local_sum, offset));
    if ((lane & 31) == 0) values[lane >> 5] = local_sum;
    __syncthreads();
    if (lane == 0)
      for (int warp = 0; warp < 4; ++warp) denominator = __fadd_rn(denominator, values[warp]);
    __syncthreads();
  } else {
    for (int64_t t = begin; t <= pos; t += 128) {
      values[lane] =
          t + lane <= pos
              ? expf(rb((saved ? saved[cache_slot(s, t + lane)]
                               : score(s, q, kc, r, h, kh, t + lane, shared_cache, request_ids)) -
                        max_score))
              : 0;
      __syncthreads();
      if (lane == 0)
        for (int j = 0; j < 128 && t + j <= pos; ++j)
          denominator = __fadd_rn(denominator, values[j]);
      __syncthreads();
    }
  }
  if (lane == 0 && sinks)
    denominator = __fadd_rn(denominator, expf(rb(bf16_bits_to_float(sinks[h]) - max_score)));
  __syncthreads();
  float sum = 0;
  for (int64_t t = begin; t <= pos; t += 128) {
    values[lane] =
        t + lane <= pos
            ? rb(expf(rb((saved ? saved[cache_slot(s, t + lane)]
                                : score(s, q, kc, r, h, kh, t + lane, shared_cache, request_ids)) -
                         max_score)) /
                 denominator)
            : 0;
    __syncthreads();
    if (probabilities) {
      if (t + lane <= pos)
        probabilities[(int64_t(r) * s.q_heads + h) * s.capacity + cache_slot(s, t + lane)] =
            float_to_bf16_bits(values[lane]);
      __syncthreads();
      continue;
    }
    for (int j = 0; j < 128 && t + j <= pos; ++j) {
      const int64_t slot =
          int64_t(shared_cache ? 0 : (request_ids ? request_ids[r] : r)) * s.capacity +
          cache_slot(s, t + j);
      sum = __fmaf_rn(values[j], bf16_bits_to_float(vc[slot * s.kv_heads * 128 + kh * 128 + lane]),
                      sum);
    }
    __syncthreads();
  }
  if (!probabilities) dest[lane] = float_to_bf16_bits(sum);
}
__global__ void value_tiles(MimoAttentionShape s, const uint16_t* probabilities, const uint16_t* vc,
                            const int64_t* positions, uint16_t* out, int first_key, int end_key) {
  using namespace nvcuda;
  const int row0 = blockIdx.y * 16, head = blockIdx.z, dim0 = blockIdx.x * 16;
  const int lane = threadIdx.x, kh = head / (s.q_heads / s.kv_heads);
  const int last_row = min(row0 + 15, s.requests - 1);
  const int begin = s.window ? max(first_key, int(positions[row0]) - s.window + 1) : 0;
  const int end = min(end_key, int(positions[last_row]) + 1);
  __shared__ __align__(32) __nv_bfloat16 a[256], b[256];
  __shared__ __align__(32) float c[256];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
  wmma::fill_fragment(cf, 0.0f);
  for (int key0 = begin; key0 < end; key0 += 16) {
    for (int i = lane; i < 256; i += 32) {
      const int row = row0 + i / 16, key = key0 + i % 16;
      uint16_t prob = 0;
      if (row < s.requests && key < end) {
        const int64_t p = positions[row];
        if (key <= p && (!s.window || key >= p - s.window + 1))
          prob = probabilities[(int64_t(row) * s.q_heads + head) * s.capacity + cache_slot(s, key)];
      }
      reinterpret_cast<uint16_t*>(a)[i] = prob;
      const int vkey = key0 + i / 16;
      reinterpret_cast<uint16_t*>(b)[i] =
          vkey < end ? vc[(int64_t(cache_slot(s, vkey)) * s.kv_heads + kh) * 128 + dim0 + i % 16]
                     : 0;
    }
    __syncwarp();
    wmma::load_matrix_sync(af, a, 16);
    wmma::load_matrix_sync(bf, b, 16);
    wmma::mma_sync(cf, af, bf, cf);
    __syncwarp();
  }
  wmma::store_matrix_sync(c, cf, 16, wmma::mem_row_major);
  __syncwarp();
  for (int i = lane; i < 256; i += 32) {
    const int row = row0 + i / 16;
    if (row < s.requests)
      out[(int64_t(row) * s.q_heads + head) * 128 + dim0 + i % 16] = float_to_bf16_bits(c[i]);
  }
}
// Four warps share one probability tile across all 128 value columns.
// Each output fragment retains the same ordered MMA chain as value_tiles.
__global__ void value_tiles_wide(MimoAttentionShape s, const uint16_t* probabilities,
                                 const uint16_t* vc, const int64_t* positions, uint16_t* out,
                                 int first_key, int end_key) {
  using namespace nvcuda;
  const int row0 = blockIdx.x * 16, head = blockIdx.y, tid = threadIdx.x;
  const int warp = tid / 32, kh = head / (s.q_heads / s.kv_heads);
  const int last_row = min(row0 + 15, s.requests - 1);
  const int begin = s.window ? max(first_key, int(positions[row0]) - s.window + 1) : 0;
  const int end = min(end_key, int(positions[last_row]) + 1);
  __shared__ __align__(32) __nv_bfloat16 a[256], b[16 * 128];
  __shared__ __align__(32) float c[16 * 128];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf[2];
#pragma unroll
  for (int tile = 0; tile < 2; ++tile) wmma::fill_fragment(cf[tile], 0.0f);
  for (int key0 = begin; key0 < end; key0 += 16) {
    for (int i = tid; i < 256; i += 128) {
      const int row = row0 + i / 16, key = key0 + i % 16;
      uint16_t probability = 0;
      if (row < s.requests && key < end) {
        const int64_t pos = positions[row];
        if (key <= pos && (!s.window || key >= pos - s.window + 1))
          probability =
              probabilities[(int64_t(row) * s.q_heads + head) * s.capacity + cache_slot(s, key)];
      }
      reinterpret_cast<uint16_t*>(a)[i] = probability;
    }
    for (int i = tid; i < 16 * 128; i += 128) {
      const int key = key0 + i / 128, dim = i % 128;
      reinterpret_cast<uint16_t*>(b)[i] =
          key < end ? vc[(int64_t(cache_slot(s, key)) * s.kv_heads + kh) * 128 + dim] : 0;
    }
    __syncthreads();
    wmma::load_matrix_sync(af, a, 16);
#pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
      wmma::load_matrix_sync(bf, b + (warp * 2 + tile) * 16, 128);
      wmma::mma_sync(cf[tile], af, bf, cf[tile]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int tile = 0; tile < 2; ++tile)
    wmma::store_matrix_sync(c + (warp * 2 + tile) * 16, cf[tile], 128, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < 16 * 128; i += 128) {
    const int row = row0 + i / 128;
    if (row < s.requests)
      out[(int64_t(row) * s.q_heads + head) * 128 + i % 128] = float_to_bf16_bits(c[i]);
  }
}

constexpr int decode_value_split = 512;
__global__ void decode_value_parts(MimoAttentionShape s, const uint16_t* probabilities,
                                   const uint16_t* vc, const int64_t* positions,
                                   const int32_t* request_ids, float* parts, int splits) {
  using namespace nvcuda;
  const int split = blockIdx.x, kh = blockIdx.y, row = blockIdx.z, tid = threadIdx.x;
  const int64_t pos = positions[row];
  if (pos < decode_split_min_keys - 1 || invalid(s, pos)) return;
  const int first = s.window ? max(0, int(pos) - s.window + 1) : 0;
  const int begin = first + split * decode_value_split;
  if (begin > pos) return;
  const int end = min(begin + decode_value_split, int(pos) + 1);
  const int request = request_ids ? request_ids[row] : row;
  const int group = s.q_heads / s.kv_heads, warp = tid / 32;
  __shared__ __align__(32) __nv_bfloat16 a[256], b[16 * 128];
  __shared__ __align__(32) float c[16 * 128];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf[2];
#pragma unroll
  for (int tile = 0; tile < 2; ++tile) wmma::fill_fragment(cf[tile], 0.0f);
  for (int key0 = begin; key0 < end; key0 += 16) {
    for (int i = tid; i < 256; i += 128) {
      const int h = i / 16, key = key0 + i % 16;
      reinterpret_cast<uint16_t*>(a)[i] =
          h < group && key < end
              ? probabilities[(int64_t(row) * s.q_heads + kh * group + h) * s.capacity +
                              cache_slot(s, key)]
              : 0;
    }
    for (int i = tid; i < 16 * 128; i += 128) {
      const int key = key0 + i / 128, dim = i % 128;
      const int64_t slot = int64_t(request) * s.capacity + cache_slot(s, key);
      reinterpret_cast<uint16_t*>(b)[i] = key < end ? vc[(slot * s.kv_heads + kh) * 128 + dim] : 0;
    }
    __syncthreads();
    wmma::load_matrix_sync(af, a, 16);
#pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
      wmma::load_matrix_sync(bf, b + (warp * 2 + tile) * 16, 128);
      wmma::mma_sync(cf[tile], af, bf, cf[tile]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int tile = 0; tile < 2; ++tile)
    wmma::store_matrix_sync(c + (warp * 2 + tile) * 16, cf[tile], 128, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < group * 128; i += 128)
    parts[((int64_t(row) * s.q_heads + kh * group + i / 128) * splits + split) * 128 + i % 128] =
        c[i];
}
__global__ void decode_value_reduce(MimoAttentionShape s, const float* parts,
                                    const int64_t* positions, uint16_t* out, int splits) {
  const int row = blockIdx.y, head = blockIdx.x, dim = threadIdx.x;
  const int64_t pos = positions[row];
  if (pos >= 0 && pos < decode_split_min_keys - 1) return;
  float sum = 0;
  if (pos >= 0 && !invalid(s, pos)) {
    const int keys = s.window ? min(int(pos) + 1, s.window) : int(pos) + 1;
    const int active = (keys + decode_value_split - 1) / decode_value_split;
    for (int split = 0; split < active; ++split)
      sum = __fadd_rn(sum, parts[((int64_t(row) * s.q_heads + head) * splits + split) * 128 + dim]);
  }
  out[(int64_t(row) * s.q_heads + head) * 128 + dim] = float_to_bf16_bits(sum);
}

}  // namespace

void mimo_qkv_append(const MimoAttentionShape& s, const uint16_t* fused, const float* freq,
                     const int64_t* positions, uint16_t* q, uint16_t* kc, uint16_t* vc,
                     int32_t* status, cudaStream_t stream, bool shared_cache,
                     const int32_t* request_ids) {
  s.validate();
  if (shared_cache && s.window && s.capacity < s.window + s.requests - 1)
    throw std::invalid_argument("MiMo chunk exceeds retained ring history");
  if (!fused || !freq || !positions || !q || !kc || !vc || !status)
    throw std::invalid_argument("MiMo append: null device pointer");
  append_kernel<<<dim3(s.q_heads + 2 * s.kv_heads, s.requests), 256, 0, stream>>>(
      s, fused, freq, positions, q, kc, vc, status, shared_cache, request_ids);
  DGPP_CUDA_OK(cudaGetLastError());
}
void mimo_attention(const MimoAttentionShape& s, const uint16_t* q, const uint16_t* kc,
                    const uint16_t* vc, const int64_t* positions, const uint16_t* sinks,
                    uint16_t* out, cudaStream_t stream, bool shared_cache, float* scores,
                    const int32_t* request_ids) {
  s.validate();
  if (shared_cache && s.window && s.capacity < s.window + s.requests - 1)
    throw std::invalid_argument("MiMo chunk exceeds retained ring history");
  if (!q || !kc || !vc || !positions || !out)
    throw std::invalid_argument("MiMo attention: null device pointer");
  attention_kernel<<<dim3(s.q_heads, s.requests), 128, 0, stream>>>(
      s, q, kc, vc, positions, sinks, out, shared_cache, scores, false, nullptr, request_ids);
  DGPP_CUDA_OK(cudaGetLastError());
}
void mimo_attention_decode(const MimoAttentionShape& s, const uint16_t* q, const uint16_t* kc,
                           const uint16_t* vc, const int64_t* positions, const uint16_t* sinks,
                           uint16_t* out, float* scores, cudaStream_t stream,
                           const int32_t* request_ids) {
  s.validate();
  if (!q || !kc || !vc || !positions || !out || !scores || s.q_heads / s.kv_heads > 16)
    throw std::invalid_argument("MiMo tensor decode: invalid buffers or GQA ratio");
  if (s.capacity < 128) {
    mimo_attention(s, q, kc, vc, positions, sinks, out, stream, false, scores, request_ids);
    return;
  }
  const int keys = s.window ? s.window : s.capacity;
  // Bound empty CTA launches for short histories; each CTA strides over
  // further tiles without changing any score's ordered accumulation.
  decode_score_tiles<<<dim3(std::min((keys + 15) / 16, 128), s.kv_heads, s.requests), 256, 0,
                       stream>>>(s, q, kc, positions, request_ids, scores);
  DGPP_CUDA_OK(cudaGetLastError());
  auto* probabilities =
      reinterpret_cast<uint16_t*>(scores + int64_t(s.requests) * s.q_heads * s.capacity);
  attention_kernel<<<dim3(s.q_heads, s.requests), 128, 0, stream>>>(
      s, q, kc, vc, positions, sinks, out, false, scores, true, probabilities, request_ids, false,
      true);
  DGPP_CUDA_OK(cudaGetLastError());
  // Scores are dead after normalization. Their storage holds the smaller
  // split-PV partials, which never overlap the retained probabilities.
  const int splits = (keys + decode_value_split - 1) / decode_value_split;
  decode_value_parts<<<dim3(splits, s.kv_heads, s.requests), 128, 0, stream>>>(
      s, probabilities, vc, positions, request_ids, scores, splits);
  DGPP_CUDA_OK(cudaGetLastError());
  decode_value_reduce<<<dim3(s.q_heads, s.requests), 128, 0, stream>>>(s, scores, positions, out,
                                                                       splits);
  DGPP_CUDA_OK(cudaGetLastError());
}

void mimo_attention_prefill(const MimoAttentionShape& s, const uint16_t* q, const uint16_t* kc,
                            const uint16_t* vc, const int64_t* positions, const uint16_t* sinks,
                            uint16_t* out, float* scores, int end_key, cudaStream_t stream,
                            bool parallel_softmax, bool wide_values) {
  s.validate();
  if (!q || !kc || !vc || !positions || !out || !scores || end_key < s.requests ||
      end_key > 1048576 || (!s.window && end_key > s.capacity) ||
      (s.window && s.capacity < s.window + s.requests - 1))
    throw std::invalid_argument("MiMo tensor-core prefill: invalid buffers or chunk bounds");
  const int first = s.window ? std::max(0, end_key - s.requests - s.window + 1) : 0;
  score_tiles<<<dim3((end_key - first + 15) / 16, (s.requests + 15) / 16, s.q_heads), 32, 0,
                stream>>>(s, q, kc, positions, scores, first, end_key);
  DGPP_CUDA_OK(cudaGetLastError());
  auto* probabilities =
      reinterpret_cast<uint16_t*>(scores + int64_t(s.requests) * s.q_heads * s.capacity);
  attention_kernel<<<dim3(s.q_heads, s.requests), 128, 0, stream>>>(
      s, q, kc, vc, positions, sinks, out, true, scores, true, probabilities, nullptr,
      parallel_softmax);
  DGPP_CUDA_OK(cudaGetLastError());
  if (wide_values)
    value_tiles_wide<<<dim3((s.requests + 15) / 16, s.q_heads), 128, 0, stream>>>(
        s, probabilities, vc, positions, out, first, end_key);
  else
    value_tiles<<<dim3(8, (s.requests + 15) / 16, s.q_heads), 32, 0, stream>>>(
        s, probabilities, vc, positions, out, first, end_key);
  DGPP_CUDA_OK(cudaGetLastError());
}
namespace {
__global__ void mtp_history_gather(const uint16_t* history, uint16_t* hidden,
                                   const int64_t* positions, int64_t* translated,
                                   const int32_t* ids, int width, int capacity, int layer) {
  const int row = blockIdx.x;
  const int req = ids ? ids[row] : 0;
  const int64_t p = positions[row];
  const bool valid = req >= 0 && p >= layer;
  if (threadIdx.x == 0) translated[row] = valid ? p - layer : -1;
  if (!history) return;
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    hidden[size_t(row) * width + i] =
        valid ? history[(size_t(req) * capacity + (p - layer) % capacity) * width + i] : 0;
}
__global__ void mtp_history_store(uint16_t* history, const uint16_t* hidden,
                                  const int64_t* positions, const int32_t* ids, int width,
                                  int capacity, int layer) {
  const int row = blockIdx.x;
  const int req = ids ? ids[row] : 0;
  const int64_t p = positions[row];
  if (req < 0 || p < layer) return;
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    history[(size_t(req) * capacity + p % capacity) * width + i] = hidden[size_t(row) * width + i];
}
}  // namespace
void mimo_mtp_history_gather(const uint16_t* history, uint16_t* hidden, const int64_t* positions,
                             int64_t* translated, const int32_t* ids, int rows, int width,
                             int capacity, int layer, cudaStream_t stream) {
  if (rows < 1 || width < 1 || capacity <= rows || layer < 0 || layer > 2)
    throw std::invalid_argument("MiMo MTP history gather shape");
  mtp_history_gather<<<rows, 256, 0, stream>>>(history, hidden, positions, translated, ids, width,
                                               capacity, layer);
  DGPP_CUDA_OK(cudaGetLastError());
}
void mimo_mtp_history_store(uint16_t* history, const uint16_t* hidden, const int64_t* positions,
                            const int32_t* ids, int rows, int width, int capacity, int layer,
                            cudaStream_t stream) {
  if (rows < 1 || width < 1 || capacity <= rows || layer < 0 || layer > 2)
    throw std::invalid_argument("MiMo MTP history store shape");
  mtp_history_store<<<rows, 256, 0, stream>>>(history, hidden, positions, ids, width, capacity,
                                              layer);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
