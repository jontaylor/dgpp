#include "kernels/expert_demand.hpp"
#include "common/cuda_check.hpp"
#include <cub/block/block_scan.cuh>
#include <cstdlib>

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
__device__ void service_miss(ExpertDemand* demand, int32_t* count) {
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
}

// At most one warp of assignments. Sorting (expert, original slot) gives
// the same tile and accumulation order as the full histogram path.
__global__ void warp_packet_kernel(int32_t* routes, const uint64_t* table,
                                   const int32_t* ids, ExpertDemand* demand, int rows) {
  constexpr unsigned mask = 0xffffffff;
  const int lane = threadIdx.x, assignments = rows * K;
  const int expert = lane < assignments ? ids[lane] : E;
  if (lane < assignments) {
    if (expert < 0 || expert >= E) asm volatile("trap;");
    demand->ids[lane] = expert;
  }
  const bool missing = lane < assignments && !table[expert * 3];
  int32_t* experts = routes + M * K * B;
  int32_t* count = experts + M * K;
  if (__any_sync(mask, missing)) {
    __syncwarp(mask);  // publish all lanes' IDs before lane zero wakes the host
    if (lane == 0) service_miss(demand, count);
    return;
  }

  int key = (expert << 5) | lane;
#pragma unroll
  for (int width = 2; width <= 32; width *= 2) {
#pragma unroll
    for (int distance = width / 2; distance; distance /= 2) {
      const int peer = __shfl_xor_sync(mask, key, distance);
      const bool take_min = ((lane & width) == 0) == ((lane & distance) == 0);
      key = take_min ? min(key, peer) : max(key, peer);
    }
  }
  const int sorted_expert = key >> 5;
  const unsigned peers = __match_any_sync(mask, sorted_expert);
  const int within = lane - (__ffs(peers) - 1);
  const bool live = sorted_expert != E;
  const unsigned starts = __ballot_sync(mask, live && within % B == 0);
  if (lane == 0) {
    demand->serviced = 0;
    *count = __popc(starts) * B;
  }
  if (!live) return;
  const int tile = __popc(starts & (mask >> (31 - lane))) - 1;
  const int pos = within % B;
  routes[tile * B + pos] = key & 31;
  if (pos == 0) experts[tile] = sorted_expert;
  if (lane == 31 - __clz(peers))
    for (int j = pos + 1; j < B; ++j) routes[tile * B + j] = assignments;
}

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
    service_miss(demand, count);
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
  static const bool warp_packet = [] {
    const char* flag = std::getenv("DGPP_EXPERT_WARP_PACKET");
    return flag && flag[0] == '1';
  }();
  if (warp_packet && rows * K <= 32)
    warp_packet_kernel<<<1, 32, 0, stream>>>(routes, table, ids, demand, rows);
  else
    packet_kernel<<<1, threads, 0, stream>>>(routes, table, ids, demand, rows);
  DGPP_CUDA_OK(cudaGetLastError());
}
}
