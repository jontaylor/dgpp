#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace dgpp {

inline bool mimo_materialized_tile_enabled() {
  static const bool enabled = std::getenv("DGPP_MIMO_ATTN_TILE_ROWS") != nullptr;
  return enabled;
}
inline int mimo_materialized_tile_rows() {
  static const int rows = [] {
    const char* value = std::getenv("DGPP_MIMO_ATTN_TILE_ROWS");
    if (!value || !std::strcmp(value, "128")) return 128;
    if (!std::strcmp(value, "16")) return 16;
    if (!std::strcmp(value, "32")) return 32;
    if (!std::strcmp(value, "64")) return 64;
    throw std::invalid_argument("MiMo attention tile rows: expected 16, 32, 64 or 128");
  }();
  return rows;
}
inline size_t mimo_materialized_workspace_bytes(int rows, int heads, int global_capacity,
                                                int ring_capacity, int tile_rows) {
  // SWA keeps its original 128-row tiling/arithmetic even at tiny contexts.
  return std::max(size_t(std::min(rows, tile_rows)) * global_capacity,
                  size_t(std::min(rows, 128)) * ring_capacity) * heads * 6;
}

inline bool mimo_split_online_attention_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("DGPP_MIMO_SPLIT_ONLINE_ATTN");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}
inline int mimo_online_split_keys() {
  static const int keys = [] {
    const char* value = std::getenv("DGPP_MIMO_ONLINE_SPLIT_KEYS");
    if (!value) return 1024;
    if (!std::strcmp(value, "512")) return 512;
    if (!std::strcmp(value, "1024")) return 1024;
    if (!std::strcmp(value, "2048")) return 2048;
    throw std::invalid_argument("MiMo split keys: expected 512, 1024 or 2048");
  }();
  return keys;
}
constexpr int mimo_split_tile_rows = 32;
inline int mimo_online_splits(int capacity) {
  return std::min(256, (capacity + mimo_online_split_keys() - 1) / mimo_online_split_keys());
}
inline size_t mimo_online_partial_bytes(int rows, int heads, int capacity) {
  return size_t(rows) * heads * mimo_online_splits(capacity) * 130 * sizeof(float);
}

inline bool mimo_online_attention_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("DGPP_MIMO_ONLINE_ATTN");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

// Fixed per-process option: set before memory planning/model construction.
inline bool mimo_bounded_attention_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("DGPP_MIMO_BOUNDED_ATTN");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled || mimo_online_attention_enabled() || mimo_split_online_attention_enabled();
}

// One token per request/slot per invocation. SWA uses a bounded ring;
// global attention uses a linear cache. The GPU shared-cache mode supports
// consecutive rows of one sequence with extra ring capacity for the chunk.
struct MimoAttentionShape {
  int requests = 1;
  int q_heads = 64;
  int kv_heads = 4;
  int capacity = 0;
  int window = 0;  // 0: global, 128: SWA; capacity >=128 retains the window plus chunk history
  bool fp8_cache = false;  // unit-scale E4M3, base model only
  int cache_element_bytes() const { return fp8_cache ? 1 : 2; }
  int q_width() const { return q_heads * 192; }
  int k_width() const { return kv_heads * 192; }
  int v_width() const { return kv_heads * 128; }
  int fused_width() const { return q_width() + k_width() + v_width(); }
  void validate() const {
    if (requests <= 0 || requests > 65535)
      throw std::invalid_argument("MiMo attention requests: expected 1..65535");
    if (q_heads <= 0 || q_heads > 64 || kv_heads <= 0 || kv_heads > 8 || q_heads % kv_heads)
      throw std::invalid_argument("MiMo attention heads: invalid GQA geometry");
    if (capacity <= 0 || capacity > 1048576)
      throw std::invalid_argument("MiMo attention capacity: outside context range");
    if (window != 0 && (window != 128 || capacity < 128))
      throw std::invalid_argument("MiMo attention window: expected global or a 128-token ring");
  }
};

// Snapshot position is the number of committed tokens. Only these physical
// spans can be read by the next causal append/attention step. The allocation
// retains its fixed layout; unwritten future entries are never restored.
struct MimoCacheSpan {
  int64_t first = 0;
  int64_t count = 0;
};
inline std::array<MimoCacheSpan, 2> mimo_snapshot_spans(const MimoAttentionShape& s,
                                                        int64_t position) {
  s.validate();
  if (position < 1 || position > 1048576 || (!s.window && position > s.capacity))
    throw std::invalid_argument("MiMo snapshot: invalid committed position");
  if (!s.window) return {{{0, position}, {0, 0}}};
  const int64_t count = std::min<int64_t>(position, s.window);
  const int64_t first = (position - count) % s.capacity;
  const int64_t head = std::min<int64_t>(count, s.capacity - first);
  return {{{first, head}, {0, count - head}}};
}

namespace mimo_ref {
// FP32 inverse frequencies for the first 64 Q/K dimensions.
std::vector<float> inv_freq(double theta);
// Input is the BF16 output of the fused projection in canonical Q/K/V
// order. Q and K get half-split RoPE; V is scaled by 0.707 before caching.
// A negative position is padding. Positive positions must be consecutive
// per slot, and below capacity for global attention. The caller owns reset.
void qkv_append(const MimoAttentionShape& shape, const uint16_t* fused, const float* inv_freq,
                const int64_t* positions, uint16_t* q, uint16_t* k_cache, uint16_t* v_cache);
// Eager-reference rounding: BF16 QK score, BF16 scaled score, BF16
// max-subtracted score, FP32 softmax, BF16 probabilities and output.
// Sink logits are BF16 per Q head; they add softmax mass with zero V.
// Caches are [request,capacity,kv_head,dim], Q/output [request,q_head,dim].
void attention(const MimoAttentionShape& shape, const uint16_t* q, const uint16_t* k_cache,
               const uint16_t* v_cache, const int64_t* positions, const uint16_t* sinks,
               uint16_t* out);
}  // namespace mimo_ref
}  // namespace dgpp
