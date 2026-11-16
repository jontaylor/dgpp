#include "kernels/expert_demand.hpp"
#include "common/cuda_check.hpp"

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
__global__ void demand_kernel(const int32_t* ids, const uint64_t* table,
                              ExpertDemand* demand, int count) {
  if (threadIdx.x) return;
  bool missing = false;
  for (int j = 0; j < count; ++j) {
    const int e = ids[j];
    if (e < 0 || e >= E) asm volatile("trap;");
    demand->ids[j] = e;
    missing |= table[e * 3] == 0;
  }
  demand->serviced = missing;
  if (!missing) return;
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
}
// Same ascending-expert/ascending-slot, eight-row grouping as the CPU
// packet builder. One CTA, no host routing or shape-dependent allocations.
__global__ void packet_kernel(uint8_t* packet, const uint64_t* table,
                              const int32_t* ids, const ExpertDemand* demand, int rows) {
  auto* dst = reinterpret_cast<uint64_t*>(packet);
  for (int j = threadIdx.x; j < E * 6; j += blockDim.x) dst[j] = table[j];
  auto* routes = reinterpret_cast<int32_t*>(packet + E * 6 * sizeof(uint64_t));
  auto* experts = routes + M*K*B;
  auto* count = experts + M*K;
  for (int j = threadIdx.x; j < M*K*B; j += blockDim.x) routes[j] = rows*K;
  for (int j = threadIdx.x; j < M*K; j += blockDim.x) experts[j] = 0;
  __shared__ int sizes[E], begins[E];
  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    int n = 0;
    for (int j = 0; j < rows*K; ++j) n += ids[j] == e;
    sizes[e] = (n + B - 1) / B;
  }
  __syncthreads();
  if (!threadIdx.x) {
    int blocks = 0;
    for (int e = 0; e < E; ++e) { begins[e] = blocks; blocks += sizes[e]; }
    *count = demand->serviced ? 0 : blocks*B;
  }
  __syncthreads();
  if (demand->serviced) return;
  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    for (int b = 0; b < sizes[e]; ++b) experts[begins[e] + b] = e;
    int at = begins[e]*B;
    for (int j = 0; j < rows*K; ++j) if (ids[j] == e) routes[at++] = j;
  }
}
}
void launch_expert_layer_stamp(uint64_t* stamp, cudaStream_t stream) {
  layer_stamp_kernel<<<1,1,0,stream>>>(stamp);
  DGPP_CUDA_OK(cudaGetLastError());
}
void launch_expert_demand(const int32_t* ids, const uint64_t* table,
                         ExpertDemand* demand, int rows, cudaStream_t stream) {
  demand_kernel<<<1,32,0,stream>>>(ids, table, demand, rows*K);
  DGPP_CUDA_OK(cudaGetLastError());
}
void launch_expert_route_packet(uint8_t* packet, const uint64_t* table,
                               const int32_t* ids, const ExpertDemand* demand,
                               int rows, cudaStream_t stream) {
  packet_kernel<<<1,256,0,stream>>>(packet, table, ids, demand, rows);
  DGPP_CUDA_OK(cudaGetLastError());
}
}
