#pragma once

#include <cuda_runtime.h>

#include "models/mimo/attention.hpp"
#include "models/mimo/cache_audit.hpp"

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
// K/V storage is BF16 unless shape.fp8_cache selects unit-scale E4M3.
// Allocate capacity * heads * dimensions * shape.cache_element_bytes() bytes
// per cache plane. Q and output always remain BF16.
// All pointers are device addresses; input/output/cache buffers must not
// overlap. Each request contributes at most one
// token. The caller supplies consecutive absolute positions per active slot;
// negative positions are padding. No allocation or host/device copy occurs.
// status[request] is overwritten with 0 on valid/padded rows, 1 for a position
// beyond the context/cache bounds. Invalid rows do not mutate the cache and
// produce zero attention output; callers must surface status before use.
// Optional audit points to two zero-initialized device counters (K,V).
// It enables a separate reduction kernel; caller owns counter lifetime/reset.
// No audit work is launched when null. Requires FP8 cache format.
void mimo_qkv_append(const MimoAttentionShape& shape, const uint16_t* fused, const float* inv_freq,
                     const int64_t* positions, uint16_t* q, void* k_cache, void* v_cache,
                     int32_t* status, cudaStream_t stream, bool shared_cache = false,
                     const int32_t* request_ids = nullptr, MimoFp8AuditStats* audit = nullptr);

// Scalar decode numerics, bounded shared workspace, three score
// passes with optional score reuse. One CTA per
// request/Q head; asymmetric QK=192 and V=128, optional zero-value sink.
// Optional scores scratch is [rows, q_heads, capacity] FP32 and avoids
// recomputing QK scores in the second/third passes. It may be reused across
// layers on the same stream, but must not alias other inputs or outputs.
void mimo_attention(const MimoAttentionShape& shape, const uint16_t* q, const void* k_cache,
                    const void* v_cache, const int64_t* positions, const uint16_t* sinks,
                    uint16_t* out, cudaStream_t stream, bool shared_cache = false,
                    float* scores = nullptr, const int32_t* request_ids = nullptr);
// One decode token per mapped request. Coalesced QK tiles retain scalar score
// accumulation and normalization order, followed by split tensor-core PV.
// Histories shorter than 512 keys retain the complete scalar arithmetic.
// scores holds rows * q_heads * capacity * 6 bytes; scores are reused for
// partial PV sums after normalization. No allocation or host sync.
void mimo_attention_decode(const MimoAttentionShape& shape, const uint16_t* q,
                           const void* k_cache, const void* v_cache,
                           const int64_t* positions, const uint16_t* sinks, uint16_t* out,
                           float* scores, cudaStream_t stream,
                           const int32_t* request_ids = nullptr);
// Consecutive, nonnegative rows of one sequence ending at end_key - 1.
// Tensor-core QK and PV, preserving BF16 score/probability roundings.
// Workspace at scores must hold rows * q_heads * capacity * 6 bytes:
// FP32 scores followed by BF16 probabilities. Experimental fused_probabilities
// keeps QK/normalization/PV accumulation unchanged but builds probabilities in
// shared PV tiles. Uses the same workspace allocation, with only two FP32
// normalizers per query/head in its probability region; falls back unless
// wide_values is enabled and capacity >= 4. Decode is unaffected.
void mimo_attention_prefill(const MimoAttentionShape& shape, const uint16_t* q,
                            const void* k_cache, const void* v_cache,
                            const int64_t* positions, const uint16_t* sinks, uint16_t* out,
                            float* scores, int end_key, cudaStream_t stream,
                            bool parallel_softmax = true, bool wide_values = true,
                            bool fused_probabilities = false);
// Split-key online attention. Workspace: mimo_online_partial_bytes(rows, heads,
// capacity). Reusable on one stream; allocation/capture safe. Global queries
// may include padding/invalid positions; shared-cache valid rows are causal.
// SWA delegates to unsplit online, retaining its consecutive-prefill contract.
void mimo_attention_split_online(const MimoAttentionShape& shape, const uint16_t* q,
    const void* k_cache, const void* v_cache, const int64_t* positions, const uint16_t* sinks,
    uint16_t* out, float* partials, cudaStream_t stream, bool shared_cache = false,
    const int32_t* request_ids = nullptr);
// Experimental online decode, including mapped requests and padding.
// Uses tensor-core QK and online softmax: differs from scalar decode numerics.
void mimo_attention_online_decode(const MimoAttentionShape& shape, const uint16_t* q,
    const void* k_cache, const void* v_cache, const int64_t* positions,
    const uint16_t* sinks, uint16_t* out, cudaStream_t stream,
    const int32_t* request_ids = nullptr);
// Consecutive nonnegative shared-cache rows, ending at end_key - 1.
// Three fused tiled passes, no global workspace or capture-time allocation.
// Retains BF16 score/subtraction/probability roundings; tensor-core QK/PV
// and sequential denominator summation can differ from parallel prefill.
// online=true uses one-pass recurrence with BF16 unnormalized PV weights;
// it changes subtraction/probability rounding and FP32 accumulation order.
void mimo_attention_bounded_prefill(const MimoAttentionShape& shape, const uint16_t* q,
    const void* k_cache, const void* v_cache, const int64_t* positions,
    const uint16_t* sinks, uint16_t* out, int end_key, cudaStream_t stream, bool online = false);
}  // namespace dgpp
