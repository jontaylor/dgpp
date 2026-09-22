#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/mimo_dflash.hpp"
namespace dgpp {
namespace {
__global__ void norm_rope(uint16_t* x, const uint16_t* w, const int64_t* pos, int heads) {
  const int row = blockIdx.x / heads, h = blockIdx.x % heads, d = threadIdx.x;
  uint16_t* a = x + (size_t(row) * heads + h) * 128;
  __shared__ float v[128], red[128];
  v[d] = bf16_bits_to_float(a[d]);
  red[d] = v[d] * v[d];
  __syncthreads();
  for (int n = 64; n; n /= 2) {
    if (d < n) red[d] += red[d + n];
    __syncthreads();
  }
  // Qwen RMSNorm casts the normalized activation before multiplying weight.
  v[d] = bf16_bits_to_float(float_to_bf16_bits(v[d] * rsqrtf(red[0] / 128.f + 1e-6f))) *
         bf16_bits_to_float(w[d]);
  v[d] = bf16_bits_to_float(float_to_bf16_bits(v[d]));
  __syncthreads();
  float out = v[d];
  if (d < 64) {
    const int pair = d % 32;
    float angle = float(pos[row]) * powf(10000.f, -float(pair) / 32.f);
    const float c = bf16_bits_to_float(float_to_bf16_bits(cosf(angle)));
    const float s = bf16_bits_to_float(float_to_bf16_bits(sinf(angle)));
    // Match separate BF16 mul/add operations in the reference RoPE.
    float a0 = bf16_bits_to_float(float_to_bf16_bits(v[d] * c));
    float a1 = bf16_bits_to_float(float_to_bf16_bits((d < 32 ? -v[d + 32] : v[d - 32]) * s));
    out = a0 + a1;
  }
  a[d] = pos[row] < 0 ? 0 : float_to_bf16_bits(out);
}
__global__ void append(const uint16_t* k, const uint16_t* v, uint16_t* kc, uint16_t* vc,
                       const int64_t* pos, const int32_t* req, int cap, int kvheads) {
  int row = blockIdx.x;
  int64_t p = pos[row];
  if (p < 0) return;
  int width = kvheads * 128;
  size_t dst = (size_t(req[row]) * cap + p % cap) * width;
  for (int d = threadIdx.x; d < width; d += blockDim.x) {
    kc[dst + d] = k[size_t(row) * width + d];
    vc[dst + d] = float_to_bf16_bits(bf16_bits_to_float(v[size_t(row) * width + d]) * 0.612f);
  }
}
__global__ void attention(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                          const uint16_t* kc, const uint16_t* vc, const uint16_t* sinks,
                          const int64_t* pos, const int32_t* req, uint16_t* out, int cap, int heads,
                          int kvheads) {
  int row = blockIdx.x, head = blockIdx.y, d = threadIdx.x, group = row / 8;
  int64_t p = pos[row], start = pos[group * 8];
  int kvhead = head / (heads / kvheads);
  int width = kvheads * 128;
  __shared__ float red[128], score, maxscore, denom, prob;
  float sum = 0;
  const float query = bf16_bits_to_float(q[(size_t(row) * heads + head) * 128 + d]);
  if (d == 0) {
    maxscore = bf16_bits_to_float(sinks[head]);
    denom = 1.f;
  }
  __syncthreads();
  // Online softmax includes the zero-value attention sink. Context strictly
  // precedes the block; block-local K/V are read from separate scratch.
  int64_t low = max(int64_t(0), p - 1023);
  for (int64_t t = low; t < start + 8 && p >= 0; ++t) {
    bool context = t < start;
    size_t off = context ? (size_t(req[row]) * cap + t % cap) * width + kvhead * 128 + d
                         : (size_t(group) * 8 + t - start) * width + kvhead * 128 + d;
    float kval = bf16_bits_to_float(context ? kc[off] : k[off]);
    red[d] = query * kval;
    __syncthreads();
    for (int n = 64; n; n /= 2) {
      if (d < n) red[d] += red[d + n];
      __syncthreads();
    }
    if (d == 0) {
      score = red[0] * 0.08838834764831845f;
      float next = fmaxf(maxscore, score);
      prob = expf(score - next);
      score = expf(maxscore - next);
      denom = denom * score + prob;
      maxscore = next;
    }
    __syncthreads();
    float val = bf16_bits_to_float(
        context ? vc[off] : float_to_bf16_bits(bf16_bits_to_float(v[off]) * 0.612f));
    sum = sum * score + prob * val;
    __syncthreads();
  }
  out[(size_t(row) * heads + head) * 128 + d] = p < 0 ? 0 : float_to_bf16_bits(sum / denom);
}
__global__ void mask_embedding(uint16_t* x, const uint16_t* mask) {
  int row = blockIdx.x;
  if (row % 8 == 0) return;
  for (int d = threadIdx.x; d < 4096; d += blockDim.x) x[size_t(row) * 4096 + d] = mask[d];
}
__global__ void features(const uint16_t* x, uint16_t* out, int f) {
  int row = blockIdx.x;
  for (int d = threadIdx.x; d < 4096; d += blockDim.x)
    out[size_t(row) * 20480 + f * 4096 + d] = x[size_t(row) * 4096 + d];
}
__global__ void select(const float* in, float* out, int n, int v, int index) {
  int g = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= v) return;
  for (int r = 0; r < n; r++)
    out[(size_t(g) * n + r) * v + i] = in[(size_t(g) * 8 + index + 1) * v + i];
}
}  // namespace
void dflash_norm_rope(uint16_t* x, const uint16_t* w, const int64_t* p, int n, int h,
                      cudaStream_t s) {
  norm_rope<<<n * h, 128, 0, s>>>(x, w, p, h);
  DGPP_CUDA_OK(cudaGetLastError());
}
void dflash_append(const uint16_t* k, const uint16_t* v, uint16_t* kc, uint16_t* vc,
                   const int64_t* p, const int32_t* r, int n, int c, int h, cudaStream_t s) {
  append<<<n, 256, 0, s>>>(k, v, kc, vc, p, r, c, h);
  DGPP_CUDA_OK(cudaGetLastError());
}
void dflash_attention(const uint16_t* q, const uint16_t* k, const uint16_t* v, const uint16_t* kc,
                      const uint16_t* vc, const uint16_t* sink, const int64_t* p, const int32_t* r,
                      uint16_t* o, int g, int c, int h, int kv, cudaStream_t s) {
  attention<<<dim3(g * 8, h), 128, 0, s>>>(q, k, v, kc, vc, sink, p, r, o, c, h, kv);
  DGPP_CUDA_OK(cudaGetLastError());
}
void dflash_mask(uint16_t* x, const uint16_t* mask, int n, cudaStream_t stream) {
  mask_embedding<<<n, 256, 0, stream>>>(x, mask);
  DGPP_CUDA_OK(cudaGetLastError());
}
void dflash_features(const uint16_t* x, uint16_t* y, int n, int f, cudaStream_t s) {
  features<<<n, 256, 0, s>>>(x, y, f);
  DGPP_CUDA_OK(cudaGetLastError());
}
void dflash_select(const float* x, float* y, int g, int n, int v, int i, cudaStream_t s) {
  select<<<dim3((v + 255) / 256, g), 256, 0, s>>>(x, y, n, v, i);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
