#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

#include "common/cuda_check.hpp"

// Read each selected expert's contiguous prepared payload and scales once.
// The checksum keeps all vector loads live and writes only one word per warp.
namespace {
constexpr int threads = 256;
constexpr int vectors_per_thread = 8;

__global__ void scan_experts(const uint4* storage, const int* experts, int vectors,
                             unsigned* checksums) {
  const uint4* expert = storage + size_t(experts[blockIdx.y]) * vectors;
  const int begin = blockIdx.x * threads * vectors_per_thread + threadIdx.x;
  unsigned value = 0;
#pragma unroll
  for (int i = 0; i < vectors_per_thread; ++i) {
    const int index = begin + i * threads;
    if (index < vectors) {
      const uint4 v = expert[index];
      value ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
#pragma unroll
  for (int delta = 16; delta; delta /= 2) value ^= __shfl_xor_sync(0xffffffff, value, delta);
  if (threadIdx.x % 32 == 0) {
    const int block = blockIdx.y * gridDim.x + blockIdx.x;
    checksums[block * (threads / 32) + threadIdx.x / 32] = value;
  }
}
}  // namespace

void launch_mxfp4_weight_scan(const uint8_t* storage, size_t bytes, const int* experts,
                              int count, unsigned* checksums, size_t capacity, cudaStream_t stream) {
  const int vectors = static_cast<int>(bytes / sizeof(uint4));
  constexpr int tile = threads * vectors_per_thread;
  const dim3 grid((vectors + tile - 1) / tile, count);
  if (bytes % sizeof(uint4) || size_t(grid.x) * grid.y * (threads / 32) > capacity)
    throw std::invalid_argument("MXFP4 scan requires aligned records and sufficient checksum space");
  scan_experts<<<grid, threads, 0, stream>>>(reinterpret_cast<const uint4*>(storage), experts,
                                             vectors, checksums);
  DGPP_CUDA_OK(cudaGetLastError());
}
