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
  // A serial paged walk may publish misses from every layer to one word.
  // The layer-local request still carries the host's completion or abort.
  uint32_t* pending_layer = nullptr;
  uint32_t layer_plus_one = 0;
};
// Opt-in graph-compatible diagnostic; timestamps use the GPU nanosecond clock.
void launch_expert_layer_stamp(uint64_t* stamp, cudaStream_t stream);
void launch_expert_route_packet(int32_t* routes, const uint64_t* table,
                               const int32_t* ids, ExpertDemand* demand,
                               int rows, cudaStream_t stream);
}  // namespace dgpp
