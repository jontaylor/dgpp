#pragma once

#include <cuda_runtime.h>

#include "models/mimo/attention.hpp"

namespace dgpp {
// Native MTP head d consumes backbone hidden at absolute row p-d.
// Attention positions are translated by -d so each layer's populated history
// starts at zero. Padding never reads or writes a history plane.
void mimo_mtp_history_gather(const uint16_t* history, uint16_t* hidden, const int64_t* positions,
                             int64_t* layer_positions, const int32_t* request_ids, int rows,
                             int width, int history_rows, int layer, cudaStream_t stream);
void mimo_mtp_history_store(uint16_t* history, const uint16_t* hidden, const int64_t* positions,
                            const int32_t* request_ids, int rows, int width, int history_rows,
                            int layer, cudaStream_t stream);
// request_ids optionally maps decode rows to distinct valid cache planes;
// the caller owns bounds checking and must not map active rows to the same slot.
// shared_cache=true treats rows as consecutive positions of one sequence.
// For SWA the capacity must hold window + rows - 1 positions, preventing
// the full chunk append from evicting history needed by its earliest query.
// All pointers are device addresses; input/output/cache buffers must not
// overlap. Each request contributes at most one
// token. The caller supplies consecutive absolute positions per active slot;
// negative positions are padding. No allocation or host/device copy occurs.
// status[request] is overwritten with 0 on valid/padded rows, 1 for a position
// beyond the context/cache bounds. Invalid rows do not mutate the cache and
// produce zero attention output; callers must surface status before use.
void mimo_qkv_append(const MimoAttentionShape& shape, const uint16_t* fused, const float* inv_freq,
                     const int64_t* positions, uint16_t* q, uint16_t* k_cache, uint16_t* v_cache,
                     int32_t* status, cudaStream_t stream, bool shared_cache = false,
                     const int32_t* request_ids = nullptr);

// Scalar decode numerics, bounded shared workspace, three score
// passes with optional score reuse. One CTA per
// request/Q head; asymmetric QK=192 and V=128, optional zero-value sink.
// Optional scores scratch is [rows, q_heads, capacity] FP32 and avoids
// recomputing QK scores in the second/third passes. It may be reused across
// layers on the same stream, but must not alias other inputs or outputs.
void mimo_attention(const MimoAttentionShape& shape, const uint16_t* q, const uint16_t* k_cache,
                    const uint16_t* v_cache, const int64_t* positions, const uint16_t* sinks,
                    uint16_t* out, cudaStream_t stream, bool shared_cache = false,
                    float* scores = nullptr, const int32_t* request_ids = nullptr);
// One decode token per mapped request. Coalesced QK tiles retain scalar score
// accumulation and normalization order, followed by split tensor-core PV.
// Histories shorter than 512 keys retain the complete scalar arithmetic.
// scores holds rows * q_heads * capacity * 6 bytes; scores are reused for
// partial PV sums after normalization. No allocation or host sync.
void mimo_attention_decode(const MimoAttentionShape& shape, const uint16_t* q,
                           const uint16_t* k_cache, const uint16_t* v_cache,
                           const int64_t* positions, const uint16_t* sinks, uint16_t* out,
                           float* scores, cudaStream_t stream,
                           const int32_t* request_ids = nullptr);
// Consecutive, nonnegative rows of one sequence ending at end_key - 1.
// Tensor-core QK and PV, preserving BF16 score/probability roundings.
// Workspace at scores must hold rows * q_heads * capacity * 6 bytes:
// FP32 scores followed by BF16 probabilities.
void mimo_attention_prefill(const MimoAttentionShape& shape, const uint16_t* q,
                            const uint16_t* k_cache, const uint16_t* v_cache,
                            const int64_t* positions, const uint16_t* sinks, uint16_t* out,
                            float* scores, int end_key, cudaStream_t stream,
                            bool parallel_softmax = true, bool wide_values = true);
}  // namespace dgpp
