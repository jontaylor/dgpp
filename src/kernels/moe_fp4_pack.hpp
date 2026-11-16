#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <span>

#include "loaders/expert_residency.hpp"

namespace dgpp {

// Bitwise scatter of an expert chunk from device or mapped host memory.
// Plans are built at startup; launches use no scratch or device allocation.
struct ExpertChunkCopyPlan {
  struct Copy {
    uint64_t source_offset, destination_offset, source_pitch, destination_pitch;
    uint64_t row_units, units;
    uint32_t first_block, blocks, unit_bytes;
  } copies[9]{};
  uint32_t count = 0, blocks = 0;
};
ExpertChunkCopyPlan expert_chunk_copy_plan(std::span<const expert::ScatterCopy> copies);
void launch_expert_chunk_scatter(uint8_t* destination, const uint8_t* source,
    const ExpertChunkCopyPlan& plan, cudaStream_t stream);

// Multiple completed chunks, possibly from different experts. Sources and
// destinations obey the single-scatter contract and stay alive through stream
// completion. Payload/scale regions must be disjoint; repeated scalar globals
// retain the final copy in submission order. No staging metadata is retained
// beyond the launch.
struct ExpertChunkScatter {
  uint8_t* destination;
  const uint8_t* source;
  ExpertChunkCopyPlan plan;
};
inline constexpr size_t kExpertScatterBatch = 32;
void launch_expert_chunk_scatter_batch(std::span<const ExpertChunkScatter> chunks, cudaStream_t stream);

// NVFP4 source matrices, with the ordinary row-major payload/scales. Globals
// retain DGPP's inverse-scale convention. This conversion moves bits only.
struct Fp4ExpertSource {
  const uint8_t* payload[3]{};  // gate, up, down
  const uint8_t* scales[3]{};
  const float* globals[3]{};
};
struct Fp4ExpertDestination {
  uint8_t* payload[3]{};
  uint8_t* scales[3]{};
  float* globals[3]{};
};

// SM12x W4A16 MMA tiles: K16/N64 packed codes, and K16/N64 permuted raw
// E4M3 scales. Unlike the b12x prepared scale encoding, raw scale bytes are
// reversible, including subnormals and NaNs. The consuming kernel must use
// raw-scale decoding. W13 is gate then up; its two globals stay independent.
struct Fp4PackedExpert {
  uint8_t* w13 = nullptr;
  uint8_t* s13 = nullptr;
  float* g13 = nullptr;  // two inverse globals: gate, up
  uint8_t* w2 = nullptr;
  uint8_t* s2 = nullptr;
  float* g2 = nullptr;
};
// One expert's compact prepared representation. It fits inside the existing
// canonical full-expert allocation; all matrix starts are 16-byte aligned.
size_t moe_fp4_packed_bytes(int hidden, int intermediate);
Fp4PackedExpert moe_fp4_packed_view(uint8_t* data, int hidden, int intermediate);

// Prepared weights/globals with nibble-encoded scales. This format is valid
// only after every source scale byte has been checked to satisfy (x & 0x87)==0.
struct Fp4CompactExpert {
  uint8_t *w13 = nullptr, *s13 = nullptr;
  float* g13 = nullptr;
  uint8_t *w2 = nullptr, *s2 = nullptr;
  float* g2 = nullptr;
};
size_t moe_fp4_compact_bytes(int hidden, int intermediate);
size_t moe_fp4_expanded_scale_bytes(int hidden, int intermediate);
Fp4CompactExpert moe_fp4_compact_view(uint8_t* data, int hidden, int intermediate);
// Sources, compact destination, and expanded scale buffer must be disjoint.
// Their owners must keep them alive until the stream's last consumer finishes.
// Calling compaction for unchecked or unsupported scales is an error: the
// operation itself intentionally does not validate the input.
void launch_moe_fp4_compact(Fp4CompactExpert dst, Fp4PackedExpert src,
                            int hidden, int intermediate, bool scales_verified,
                            cudaStream_t stream);
// Returns independent prepared weight/scale pointers suitable for the device
// binding table; the scale expansion is enqueued on the given stream.
Fp4PackedExpert launch_moe_fp4_expand_scales(Fp4CompactExpert src, uint8_t* scales,
                                             int hidden, int intermediate, cudaStream_t stream);

// Resolve lazily loaded packing kernels before a persistent demand gate can
// wait for them. CUDA module loading may otherwise synchronize the device.
void prepare_moe_fp4_pack();

// Source and destination must be disjoint. All allocations belong to the
// caller; the launchers allocate nothing and are CUDA-graph capturable.
void launch_moe_fp4_pack(Fp4PackedExpert dst, Fp4ExpertSource src,
                         int hidden, int intermediate, cudaStream_t stream);
void launch_moe_fp4_unpack(Fp4ExpertDestination dst, Fp4PackedExpert src,
                           int hidden, int intermediate, cudaStream_t stream);

}  // namespace dgpp
