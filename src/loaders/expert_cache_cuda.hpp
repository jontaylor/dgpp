#pragma once
#include <cuda_runtime.h>

#include "loaders/expert_cache.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp::expert {
// Consumers use the engine-owned stream; transfers use a backend-owned
// nonblocking stream with storage lifetime fences. No worker CUDA contexts.
// Construct/register at boot, before fabric collective execution, and keep
// the engine stream/context alive until all cache/execution leases are gone.
CacheStorageFactory cuda_cache_storage(const StoreIdentity&, cudaStream_t, bool packed_fp4 = false);

// Converts a complete reconstruction into the exact existing kernel views.
// The returned pointers are valid only while the execution lease is held.
std::array<GlmFp4Matrix, 3> execution_matrices(const ExpertExecutionLease&,
    const ChunkLayout& full, Format format);
}  // namespace dgpp::expert
