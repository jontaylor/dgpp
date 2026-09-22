#include <stdexcept>

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/fp8_dequant.hpp"

namespace dgpp {
namespace {

constexpr int kThreads = 256;

// One thread per output element: memory-bound by construction (one u8 and
// one bf16 per element plus a shared 512-byte scale row per block), so the
// scalar addressing costs nothing that matters. Vectorize only if a
// production path starts calling this in a hot loop.
__global__ void fp8_dequant_blocks_kernel(const uint8_t* __restrict__ payload,
                                          const float* __restrict__ scales,
                                          uint16_t* __restrict__ out, int64_t rows, int64_t cols,
                                          int64_t scale_cols, int block_log2) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  const int64_t total = rows * cols;
  if (i >= total) return;
  const int64_t n = i / cols;
  const int64_t k = i - n * cols;
  const float v = fp8_e4m3_bits_to_float(payload[i]) *
                  scales[(n >> block_log2) * scale_cols + (k >> block_log2)];
  out[i] = float_to_bf16_bits(v);
}

// Aligned MiMo 64x64 grid: two source codes per native conversion and one
// packed BF16 store. Scale application/rounding matches the scalar bridge.
__global__ void fp8_dequant_grid64_kernel(const uint8_t* payload, const float* scales,
                                          uint16_t* out, int64_t rows, int64_t cols) {
  const int64_t pair = int64_t(blockIdx.x) * kThreads + threadIdx.x;
  if (pair >= rows * cols / 2) return;
  const int64_t i = pair * 2, row = i / cols, col = i % cols;
  const auto codes = reinterpret_cast<const uint16_t*>(payload)[pair];
  const float2 values = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(codes, __NV_E4M3)));
  const float scale = scales[(row / 64) * (cols / 64) + col / 64];
  reinterpret_cast<uint32_t*>(out)[pair] = uint32_t(float_to_bf16_bits(values.x * scale)) |
                                           (uint32_t(float_to_bf16_bits(values.y * scale)) << 16);
}

}  // namespace

void launch_fp8_dequant_blocks(const uint8_t* payload, const float* scales, uint16_t* out_bf16,
                               int64_t rows, int64_t cols, cudaStream_t stream) {
  launch_fp8_dequant_grid(payload, scales, out_bf16, rows, cols, 7, stream);
}

void launch_fp8_dequant_grid(const uint8_t* payload, const float* scales, uint16_t* out_bf16,
                             int64_t rows, int64_t cols, int block_log2, cudaStream_t stream) {
  if (block_log2 != 6 && block_log2 != 7)
    throw std::invalid_argument("fp8_dequant: block_log2 must be 6 or 7");
  if (rows <= 0 || cols <= 0)
    throw std::invalid_argument("fp8_dequant: rows and cols must be positive");
  if (!payload || !scales || !out_bf16) throw std::invalid_argument("fp8_dequant: null pointer");
  if (block_log2 == 6 && rows % 64 == 0 && cols % 64 == 0 &&
      reinterpret_cast<uintptr_t>(payload) % 2 == 0 &&
      reinterpret_cast<uintptr_t>(out_bf16) % 4 == 0) {
    const int64_t blocks = (rows * cols / 2 + kThreads - 1) / kThreads;
    if (blocks > 2147483647LL) throw std::runtime_error("fp8_dequant: grid too large");
    fp8_dequant_grid64_kernel<<<static_cast<int>(blocks), kThreads, 0, stream>>>(
        payload, scales, out_bf16, rows, cols);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  const int64_t total = rows * cols;
  const int64_t scale_cols = (cols + (1LL << block_log2) - 1) >> block_log2;
  const int64_t blocks = (total + kThreads - 1) / kThreads;
  if (blocks > 2147483647LL) throw std::runtime_error("fp8_dequant: grid too large");
  fp8_dequant_blocks_kernel<<<static_cast<int>(blocks), kThreads, 0, stream>>>(
      payload, scales, out_bf16, rows, cols, scale_cols, block_log2);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
