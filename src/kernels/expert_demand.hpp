#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace dgpp {
// Mapped, system-coherent mailbox. Only the GPU publishes request=1; only
// the engine thread publishes completion=2 (or failure=3). Reset when idle.
struct alignas(64) ExpertDemand {
  uint32_t request = 0;
  uint32_t serviced = 0;
  int32_t ids[64 * 6]{};
};
void launch_expert_demand(const int32_t* ids, const uint64_t* table,
                         ExpertDemand* demand, int rows, cudaStream_t stream);
// Opt-in graph-compatible diagnostic; timestamps use the GPU nanosecond clock.
void launch_expert_layer_stamp(uint64_t* stamp, cudaStream_t stream);
void launch_expert_route_packet(uint8_t* packet, const uint64_t* table,
                               const int32_t* ids, const ExpertDemand* demand,
                               int rows, cudaStream_t stream);
}  // namespace dgpp
