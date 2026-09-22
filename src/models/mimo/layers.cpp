#include "models/mimo/layers.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>

#include "kernels/glm_norm.hpp"
#include "kernels/kernels.hpp"

namespace dgpp {
MimoDecoderLayer::MimoDecoderLayer(const MimoLayerResident& w, const MimoTextConfig& c,
                                   int requests, int global_capacity, IGemm& gemm, void* workspace,
                                   size_t workspace_bytes, int max_tokens, float* attention_scores,
                                   GlmMoeLayer* shared_moe, LayerBump* shared_scratch)
    : w_(w),
      cfg_(c),
      shape_{requests, w.q_heads, w.kv_heads,
             c.sliding(w.layer) ? mimo_ring_capacity(max_tokens) : global_capacity,
             c.sliding(w.layer) ? 128 : 0},
      gemm_(gemm),
      gemm_ws_(workspace),
      gemm_ws_bytes_(workspace_bytes),
      max_tokens_(max_tokens),
      attention_scores_(attention_scores) {
  shape_.validate();
  if (requests < 1 || max_tokens < 1 || requests > kGemmDecodeLoweringRows || !w.qkv || !w.output ||
      !w.input_norm || !w.post_norm || (shape_.window && !w.sinks))
    throw std::invalid_argument("MiMo decoder: incomplete weights or too many rows");
  // One aligned allocation, with an identical count and address pass.
  auto layout = [&](LayerBump& buffer) {
    auto bf = [&](size_t cols) {
      return static_cast<uint16_t*>(buffer.alloc(size_t(max_tokens_) * cols * 2));
    };
    x_ = bf(c.hidden_size);
    y_ = bf(c.hidden_size);
    fused_ = bf(shape_.fused_width());
    q_ = bf(shape_.q_width());
    attn_ = bf(shape_.q_heads * 128);
    gate_ = up_ = act_ = nullptr;
    moe_sum_ = nullptr;
    if (c.moe(w.layer)) {
      moe_sum_ = static_cast<float*>(buffer.alloc(size_t(max_tokens_) * c.hidden_size * 4));
    } else {
      if (w.dense_inter <= 0 || !w.gate || !w.up || !w.down)
        throw std::invalid_argument("MiMo decoder: incomplete dense weights");
      gate_ = bf(w.dense_inter);
      up_ = bf(w.dense_inter);
      act_ = bf(w.dense_inter);
    }
  };
  scratch_.counting = true;
  scratch_.capacity = std::numeric_limits<size_t>::max();
  layout(scratch_);
  const auto bytes = scratch_.cursor;
  scratch_.counting = false;
  if (shared_scratch) {
    scratch_.capacity = scratch_.cursor = 0;
    shared_scratch->reset();
    layout(*shared_scratch);
  } else {
    scratch_.init(bytes);
    layout(scratch_);
  }
  // Global and sliding layers have different RoPE constants: these must
  // stay layer-owned even when all execution workspace is shared.
  frequencies_.init(256);
  freq_ = static_cast<float*>(frequencies_.alloc(32 * sizeof(float)));
  const auto freq = mimo_ref::inv_freq(shape_.window ? c.swa_rope_theta : c.rope_theta);
  DGPP_CUDA_OK(cudaMemcpy(freq_, freq.data(), freq.size() * 4, cudaMemcpyHostToDevice));
  if (c.moe(w.layer)) {
    if (shared_moe) {
      moe_ = shared_moe;
      graph_slot_ = w.layer - 1;
    } else {
      owned_moe_ = std::make_unique<GlmMoeLayer>(
          w.moe_view(), mimo_moe_config(c, c.num_attention_heads / w.q_heads), max_tokens_,
          requests, 1);
      moe_ = owned_moe_.get();
    }
  }
}
size_t MimoDecoderLayer::workspace_bytes(const MimoTextConfig& c, int layer, int rows, int world) {
  auto aligned = [](size_t n) { return (n + 255) & ~size_t(255); };
  size_t bytes = 2 * aligned(size_t(rows) * c.hidden_size * 2) +
                 aligned(size_t(rows) * c.qkv_rows(layer) / world * 2) +
                 aligned(size_t(rows) * c.num_attention_heads / world * 192 * 2) +
                 aligned(size_t(rows) * c.num_attention_heads / world * 128 * 2);
  bytes += c.moe(layer) ? aligned(size_t(rows) * c.hidden_size * 4)
                        : 3 * aligned(size_t(rows) * c.intermediate_size / world * 2);
  return bytes;
}
void MimoDecoderLayer::project(const uint16_t* x, const uint16_t* w, uint16_t* y, int n, int k,
                               cudaStream_t stream, int tokens) {
  gemm_.matmul(x, w, y, tokens, n, k, DType::BF16, GemmOut::BF16, k, gemm_ws_, gemm_ws_bytes_,
               stream);
}
void MimoDecoderLayer::fold(uint16_t* partial, BoundaryReducer* boundary, cudaStream_t stream,
                            int tokens, bool capture) {
  if (!boundary) return;
  if (!capture && !boundary->stream_ordered()) DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  boundary->reduce(partial, tokens, cfg_.hidden_size);
}
void MimoDecoderLayer::enqueue(uint16_t* residual, const int64_t* positions, uint16_t* k_cache,
                               uint16_t* v_cache, int32_t* status, BoundaryReducer* boundary,
                               cudaStream_t stream, int tokens, int end_key, bool capture,
                               const int32_t* request_ids, bool cache_only) {
  if (tokens < 1 || tokens > max_tokens_)
    throw std::invalid_argument("MiMo layer: invalid chunk size");
  if (!residual || !positions || !k_cache || !v_cache || !status)
    throw std::invalid_argument("MiMo decoder: null buffer");
  if ((w_.q_heads != cfg_.num_attention_heads) != (boundary != nullptr))
    throw std::invalid_argument("MiMo decoder: TP slices require a boundary reducer");
  const auto n = int64_t(tokens) * cfg_.hidden_size;
  glm_rmsnorm_bf16(residual, w_.input_norm, x_, tokens, cfg_.hidden_size, cfg_.rms_norm_eps,
                   stream);
  project(x_, w_.qkv, fused_, shape_.fused_width(), cfg_.hidden_size, stream, tokens);
  // Expanded rings retain the oldest window plus this entire chunk. All
  // queries can therefore execute concurrently with causal position bounds.
  auto chunk = shape_;
  chunk.requests = tokens;
  mimo_qkv_append(chunk, fused_, freq_, positions, q_, k_cache, v_cache, status, stream,
                  request_ids == nullptr, request_ids);
  if (cache_only) return;
  if (!request_ids && tokens > 1 && attention_scores_ && end_key > 0) {
    // Projection/MoE chunks can be large while attention's temporary score
    // matrix stays bounded. The expanded ring preserves the whole chunk.
    // Read once outside the captured kernel path; explicit opt-in until
    // real-weight and whole-service validation establishes a benefit.
    static const bool fused_prefill = [] {
      const char* value = std::getenv("DGPP_MIMO_FUSED_PREFILL");
      return value && std::strcmp(value, "1") == 0;
    }();
    for (int first = 0; first < tokens; first += attention_tile_rows) {
      auto tile = chunk;
      tile.requests = std::min(attention_tile_rows, tokens - first);
      mimo_attention_prefill(tile, q_ + size_t(first) * shape_.q_width(), k_cache, v_cache,
                             positions + first, w_.sinks,
                             attn_ + size_t(first) * shape_.q_heads * 128, attention_scores_,
                             end_key - tokens + first + tile.requests, stream, true, true,
                             fused_prefill);
    }
  } else if (request_ids && !shape_.window && attention_scores_) {
    mimo_attention_decode(chunk, q_, k_cache, v_cache, positions, w_.sinks, attn_,
                          attention_scores_, stream, request_ids);
  } else {
    mimo_attention(chunk, q_, k_cache, v_cache, positions, w_.sinks, attn_, stream,
                   request_ids == nullptr, attention_scores_, request_ids);
  }
  auto stage = [&]() {
    uint16_t* result = boundary ? boundary->stage(tokens, cfg_.hidden_size) : nullptr;
    if (capture && boundary && !result)
      throw std::logic_error("MiMo graph boundary staging unavailable");
    return result ? result : y_;
  };
  auto* partial = stage();
  project(attn_, w_.output, partial, cfg_.hidden_size, shape_.q_heads * 128, stream, tokens);
  fold(partial, boundary, stream, tokens, capture);
  add_inplace_bf16(residual, partial, n, stream);
  glm_rmsnorm_bf16(residual, w_.post_norm, x_, tokens, cfg_.hidden_size, cfg_.rms_norm_eps, stream);
  partial = stage();
  if (moe_) {
    moe_->rebind(w_.moe_view());
    // The routed-only path leaves FP32 partial sums; round exactly once
    // before the shared TP boundary, as in the existing DGPP MoE engine.
    if (request_ids || tokens == 1)
      moe_->enqueue_decode_f32(x_, moe_sum_, tokens, nullptr, stream, capture ? graph_slot_ : -1);
    else
      moe_->enqueue_prefill_f32(x_, moe_sum_, tokens, nullptr, stream);
    launch_moe_round_bf16(partial, moe_sum_, n, stream);
  } else {
    project(x_, w_.gate, gate_, w_.dense_inter, cfg_.hidden_size, stream, tokens);
    project(x_, w_.up, up_, w_.dense_inter, cfg_.hidden_size, stream, tokens);
    launch_moe_swiglu_clamp(gate_, up_, act_, int64_t(tokens) * w_.dense_inter,
                            std::numeric_limits<float>::infinity(), stream);
    project(act_, w_.down, partial, cfg_.hidden_size, w_.dense_inter, stream, tokens);
  }
  fold(partial, boundary, stream, tokens, capture);
  add_inplace_bf16(residual, partial, n, stream);
}
void MimoDecoderLayer::prepare_graph(cudaStream_t stream) {
  if (moe_) {
    moe_->rebind(w_.moe_view());
    moe_->prepare_graph_table(graph_slot_, stream);
  }
}
}  // namespace dgpp
