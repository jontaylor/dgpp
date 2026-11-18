#include "kernels/expert_demand.hpp"
#include "common/cuda_check.hpp"
#include <cub/block/block_scan.cuh>

namespace dgpp {
namespace {
constexpr int E = 384, K = 6, M = 64, B = 8;
__global__ void layer_stamp_kernel(uint64_t* stamp) {
  unsigned long long now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  *stamp = now;
}
__device__ uint32_t load_system(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ void store_system(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
// The graph and its miss service reuse the packet. On a miss the service stream
// owns it until the host releases the mailbox, including its live route count.
constexpr int threads = 128;
__global__ void packet_kernel(int32_t* routes, const uint64_t* table,
                              const int32_t* ids, ExpertDemand* demand, int rows) {
  using Scan = cub::BlockScan<int, threads>;
  static_assert(E == threads * 3);
  __shared__ typename Scan::TempStorage scan;
  __shared__ int histogram[E], selected[M * K];
  const int tid = threadIdx.x;
  const int assignments = rows * K;
  int32_t* experts = routes + M * K * B;
  int32_t* count = experts + M * K;
  for (int e = tid; e < E; e += threads) histogram[e] = 0;
  __syncthreads();
  bool missing = false;
  for (int j = tid; j < assignments; j += threads) {
    const int e = ids[j];
    if (e < 0 || e >= E) asm volatile("trap;");
    selected[j] = e;
    demand->ids[j] = e;
    missing |= table[e * 3] == 0;
    atomicAdd(&histogram[e], 1);
  }
  if (__syncthreads_or(missing)) {
    if (tid != 0) return;
    demand->serviced = 1;
    store_system(&demand->request, 1);
    unsigned long long started;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(started));
    for (;;) {
      const auto state = load_system(&demand->request);
      if (state == 2) break;
      unsigned long long now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      if (state == 3 || now - started > 60'000'000'000ull) asm volatile("trap;");
      __nanosleep(1000);
    }
    *count = 0;
    return;
  }

  // Preserve b12x's ascending expert and slot order, including padding to B.
  int blocks[3], starts[3], total;
  for (int i = 0; i < 3; ++i) blocks[i] = (histogram[3 * tid + i] + B - 1) / B;
  Scan(scan).ExclusiveSum(blocks, starts, total);
  for (int i = 0; i < 3; ++i) histogram[3 * tid + i] = starts[i];
  if (!tid) {
    demand->serviced = 0;
    *count = total * B;
  }
  for (int j = tid; j < total * B; j += threads) routes[j] = assignments;
  __syncthreads();
  for (int j = tid; j < assignments; j += threads) {
    const int e = selected[j];
    int ordinal = 0;
    for (int prev = 0; prev < j; ++prev) ordinal += selected[prev] == e;
    const int slot = histogram[e] * B + ordinal;
    routes[slot] = j;
    if (ordinal % B == 0) experts[slot / B] = e;
  }
}

}
void launch_expert_layer_stamp(uint64_t* stamp, cudaStream_t stream) {
  layer_stamp_kernel<<<1,1,0,stream>>>(stamp);
  DGPP_CUDA_OK(cudaGetLastError());
}
void launch_expert_route_packet(int32_t* routes, const uint64_t* table,
                               const int32_t* ids, ExpertDemand* demand,
                               int rows, cudaStream_t stream) {
  packet_kernel<<<1, threads, 0, stream>>>(routes, table, ids, demand, rows);
  DGPP_CUDA_OK(cudaGetLastError());
}
}
