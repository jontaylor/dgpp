#include "models/mimo/model.hpp"
#include "models/mimo/cache_format.hpp"
#include "models/mimo/head.hpp"

#include "models/mimo/snapshot_copy.hpp"

#include "kernels/glm_norm.hpp"
#include "kernels/kernels.hpp"

namespace dgpp {
namespace {
size_t align256(size_t n) {
  return (n + 255) & ~size_t(255);
}
void validate(const MimoTextConfig& c, int64_t context, int rank, int world) {
  c.validate_tp(world);
  if (rank < 0 || rank >= world || context <= 0 || context > c.max_position_embeddings)
    throw std::invalid_argument("MiMo: invalid rank/context");
}
int64_t global_capacity(int64_t context) {
  return (context + 127) / 128 * 128;
}
size_t cache_bytes(const MimoTextConfig& c, int64_t context, int world, int rows) {
  size_t bytes = 0;
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    const size_t slots = c.sliding(l) ? mimo_ring_capacity(rows) : global_capacity(context);
    bytes += slots * (c.kv_heads(l) / world) * (192 + 128) * (mimo_fp8_cache_enabled() ? 1 : 2);
  }
  return bytes;  // every request plane is 256-byte aligned, even at tiny contexts
}

}  // namespace
size_t MimoModel::snapshot_bytes(const MimoTextConfig& c, int64_t context, int world,
                                 int /*rows*/) {
  // A snapshot keeps the live sliding window, not the expanded prefill ring.
  return cache_bytes(c, context, world, 1);
}
MemoryPlan MimoModel::plan_memory(const MimoTextConfig& c, int forward_rows, int64_t context,
                                  int rank, int world, int requests, bool mtp) {
  validate(c, context, rank, world);
  if (requests < 1 || requests > kDecodeRows)
    throw std::invalid_argument("MiMo concurrency must be in [1, 8]");
  const int rows = std::max(kDecodeRows, forward_rows);
  MemoryPlan plan;
  plan.context_tokens = context;
  size_t weights = MimoDeviceLoader::globals_bytes(c, world), device = 0, pinned = 0;
  size_t block_scratch = 0, moe_scratch = 0, moe_pinned = 0;
  moe_scratch = GlmMoeLayer::scratch_bytes(mimo_moe_config(c, world), rows, kDecodeRows,
                                           c.num_hidden_layers - 1, &moe_pinned);
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    weights += MimoDeviceLoader::layer_bytes(c, l, world);
    block_scratch = std::max(block_scratch, MimoDecoderLayer::workspace_bytes(c, l, rows, world));
  }
  if (mtp)
    block_scratch =
        std::max(block_scratch, MimoDecoderLayer::workspace_bytes(c, c.mtp_layer(), rows, world));
  session_core_plan_bytes(std::max(kDecodeRows, forward_rows), requests, c.hidden_size,
                          c.vocab_size / world, mtp, c.hidden_size, &device, &pinned);
  plan.add("MiMo resident text weights", weights);
  plan.add(mimo_fp8_cache_enabled() ? "MiMo flat global and ring K/V (unit E4M3)" : "MiMo flat global and ring K/V (BF16)",
           requests * cache_bytes(c, context, world, std::max(kDecodeRows, forward_rows)));
  plan.add("MiMo layer scratch", block_scratch + moe_scratch + size_t(c.num_hidden_layers) * 256,
           moe_pinned);
  plan.add("MiMo session core",
           device + align256(rows * c.hidden_size * 2) + align256(rows * c.num_hidden_layers * 4),
           pinned);
  plan.add("MiMo shared attention scores",
           mimo_bounded_attention_enabled() ? 0 :
           size_t(std::min(rows, MimoDecoderLayer::attention_tile_rows)) * c.num_attention_heads /
               world * std::max<int64_t>(global_capacity(context), mimo_ring_capacity(rows)) *
               (sizeof(float) + sizeof(uint16_t)));
  if (mtp) {
    const size_t plane =
        size_t(mimo_ring_capacity(rows)) * (c.swa_num_key_value_heads / world) * 320 * 2;
    plan.add("MiMo native MTP weights",
             c.mtp_blocks * MimoDeviceLoader::layer_bytes(c, c.mtp_layer(), world));
    plan.add(
        "MiMo native MTP scratch and cache",
        6 * align256(size_t(rows) * c.hidden_size * 2) + align256(size_t(rows) * 8) +
            3 * requests *
                (c.mtp_blocks * plane + 1 * size_t(mimo_ring_capacity(rows)) * c.hidden_size * 2) +
            c.mtp_blocks * 256);
  }
  plan.add("MiMo GEMM setup allowance", 4u << 20);
  return plan;
}
MimoModel::MimoModel(const MimoTextConfig& c, const std::string& checkpoint, int forward_rows,
                     int64_t context, BoundaryReducer* boundary, int rank, int world, int requests,
                     bool mtp)
    : cfg_(c) {
  validate(c, context, rank, world);
  if ((world > 1) != (boundary != nullptr))
    throw std::invalid_argument("MiMo: TP requires a boundary reducer");
  init_stream();
  SessionParams sp;
  sp.max_tokens = std::max(kDecodeRows, forward_rows);
  sp.max_cache_tokens = context;
  sp.rank = rank;
  sp.world = world;
  sp.boundary = boundary;
  if (requests < 1 || requests > kDecodeRows)
    throw std::invalid_argument("MiMo concurrency must be in [1, 8]");
  sp.max_requests = requests;
  sp.mtp = mtp;
  sp.draft_width = c.hidden_size;
  sp.vocab_size = c.vocab_size;
  sp.hidden = c.hidden_size;
  sp.lm_vocab_count = c.vocab_size / world;
  sp.lm_vocab_begin = rank * sp.lm_vocab_count;
  sp.max_position_embeddings = c.max_position_embeddings;
  sp.eos = 151645;
  init_session(sp);
  if (boundary_) boundary_->bind_stream(stream_);
  MimoDeviceLoader loader(checkpoint, rank, world);
  globals_ = loader.load_globals();
  DGPP_LOG_INFO("MiMo rank {}: base K/V and prefix snapshots use {}", rank,
                mimo_fp8_cache_enabled() ? "unit-scale E4M3 FP8" : "BF16");
  cache_.init(requests * cache_bytes(c, context, world, std::max(kDecodeRows, forward_rows)));
  DGPP_CUDA_OK(cudaMemsetAsync(cache_.base, 0, cache_.capacity, stream_));
  const int rows = sp.max_tokens;
  chunk_capacity_ = rows;
  scratch_.init(align256(rows * c.hidden_size * 2) + align256(rows * c.num_hidden_layers * 4));
  residual_ = static_cast<uint16_t*>(scratch_.alloc(rows * c.hidden_size * 2));
  status_ = static_cast<int32_t*>(scratch_.alloc(rows * c.num_hidden_layers * 4));
  if (!mimo_bounded_attention_enabled())
    attention_scores_.init(size_t(std::min(rows, MimoDecoderLayer::attention_tile_rows)) *
                         c.num_attention_heads / world *
                         std::max<int64_t>(global_capacity(context), mimo_ring_capacity(rows)) *
                         (sizeof(float) + sizeof(uint16_t)));
  size_t layer_bytes = 0;
  for (int l = 0; l < c.num_hidden_layers; ++l)
    layer_bytes = std::max(layer_bytes, MimoDecoderLayer::workspace_bytes(c, l, rows, world));
  if (mtp)
    layer_bytes =
        std::max(layer_bytes, MimoDecoderLayer::workspace_bytes(c, c.mtp_layer(), rows, world));
  layer_workspace_.init(layer_bytes);
  weights_.reserve(c.num_hidden_layers);  // layer objects retain references
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    weights_.push_back(loader.load_layer(l));
    const auto& w = weights_.back();
    const size_t slots = c.sliding(l) ? mimo_ring_capacity(rows) : global_capacity(context);
    key_plane_.push_back(slots * w.kv_heads * 192 * (mimo_fp8_cache_enabled() ? 1 : 2));
    value_plane_.push_back(slots * w.kv_heads * 128 * (mimo_fp8_cache_enabled() ? 1 : 2));
    keys_.push_back(static_cast<uint8_t*>(cache_.alloc(requests * key_plane_.back())));
    values_.push_back(static_cast<uint8_t*>(cache_.alloc(requests * value_plane_.back())));
    if (c.moe(l) && !shared_moe_)
      shared_moe_ = std::make_unique<GlmMoeLayer>(w.moe_view(), mimo_moe_config(c, world), rows,
                                                  kDecodeRows, c.num_hidden_layers - 1);
    layers_.push_back(std::make_unique<MimoDecoderLayer>(
        w, c, requests, global_capacity(context), gemm_, nullptr, 0, rows,
        static_cast<float*>(attention_scores_.base), shared_moe_.get(), &layer_workspace_));
    DGPP_LOG_INFO("MiMo rank {}: loaded layer {}/{} ({} weight bytes)", rank, l + 1,
                  c.num_hidden_layers, w.storage->capacity);
  }
  if (mtp) {
    draft_weights_.reserve(c.mtp_blocks);
    for (int d = 0; d < c.mtp_blocks; ++d) {
      draft_weights_.push_back(loader.load_layer(c.mtp_layer(d)));
      DGPP_LOG_INFO("MiMo rank {}: loaded native MTP block {} ({} weight bytes)", rank, d,
                    draft_weights_.back().storage->capacity);
    }
    draft_history_rows_ = mimo_ring_capacity(rows);
    draft_key_plane_ = size_t(draft_history_rows_) * draft_weights_[0].kv_heads * 192;
    draft_value_plane_ = size_t(draft_history_rows_) * draft_weights_[0].kv_heads * 128;
    draft_storage_.init(6 * align256(size_t(rows) * c.hidden_size * 2) +
                        align256(size_t(rows) * 8) + 3 * requests * draft_state_bytes());
    auto bf = [&](size_t n) { return static_cast<uint16_t*>(draft_storage_.alloc(n * 2)); };
    draft_hidden_ = bf(size_t(rows) * c.hidden_size);
    draft_input_ = bf(size_t(rows) * c.hidden_size * 2);
    draft_residual_ = bf(size_t(rows) * c.hidden_size);
    draft_output_ = bf(size_t(rows) * c.hidden_size);
    draft_layer_output_ = bf(size_t(rows) * c.hidden_size);
    draft_positions_ = static_cast<int64_t*>(draft_storage_.alloc(size_t(rows) * 8));
    for (int d = 0; d < c.mtp_blocks; ++d) {
      draft_keys_.push_back(bf(requests * draft_key_plane_));
      draft_values_.push_back(bf(requests * draft_value_plane_));
      if (d == 0)
        draft_history_.push_back(bf(size_t(requests) * draft_history_rows_ * c.hidden_size));
      draft_layers_.push_back(std::make_unique<MimoDecoderLayer>(
          draft_weights_[d], c, requests, global_capacity(context), gemm_, nullptr, 0, rows,
          static_cast<float*>(attention_scores_.base), nullptr, &layer_workspace_));
    }
    draft_backup_ = static_cast<uint8_t*>(draft_storage_.alloc(requests * draft_state_bytes()));
    chain_backup_ = static_cast<uint8_t*>(draft_storage_.alloc(requests * draft_state_bytes()));
    if (draft_storage_.cursor != draft_storage_.capacity)
      throw std::logic_error("MiMo draft byte plan mismatch");
    DGPP_CUDA_OK(cudaMemsetAsync(draft_storage_.base, 0, draft_storage_.capacity, stream_));
  }

  if (cache_.cursor != cache_.capacity) throw std::logic_error("MiMo cache byte plan mismatch");
  DGPP_LOG_INFO("MiMo rank {}: base live K/V {} bytes, prefix snapshot {} bytes including draft state",
                rank, cache_.capacity, session_snapshot_bytes());
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}
MimoModel::~MimoModel() {
  // Device views and buffers die before the session core destroys its stream.
  if (stream_) cudaStreamSynchronize(stream_);
}
MimoModel::Outputs MimoModel::run_rows(const RowRun& run) {
  check_req(run.req, "MiMo run");
  const int groups = std::max(1, run.batch_requests);
  if (run.T < 1 || run.T > chunk_capacity_ || run.num_spans || run.batch_requests > max_requests_ ||
      (run.decode &&
       (run.T > max_decode_rows_ || run.T % groups || run.T / groups > (mtp_ ? 4 : 1))))
    throw std::invalid_argument(
        "MiMo requires single-sequence prefill or at most four decode rows per request");
  const auto in = begin_run(run);
  embed_gather_bf16(globals_.embed, in.tokens, residual_, run.T, cfg_.hidden_size, stream_);
  Outputs out;
  for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
    layers_[l]->enqueue(residual_, in.pos, keys_[l] + (run.decode ? 0 : run.req * key_plane_[l]),
                        values_[l] + (run.decode ? 0 : run.req * value_plane_[l]),
                        status_ + l * run.T, boundary_, stream_, run.T,
                        run.decode ? 0 : int(run.pos0 + run.T), run.capture,
                        run.decode ? in.req_ids : nullptr);
    if (run.capture_layers) {
      std::vector<uint16_t> row(size_t(run.T) * cfg_.hidden_size);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      DGPP_CUDA_OK(cudaMemcpy(row.data(), residual_, row.size() * 2, cudaMemcpyDeviceToHost));
      out.layer_states.push_back(std::move(row));
    }
  }
  glm_rmsnorm_bf16(residual_, globals_.norm, h_, run.T, cfg_.hidden_size, cfg_.rms_norm_eps,
                   stream_);
  // MiMo's native MTP consumes the target's post-final-norm hidden state
  // (vLLM MiMoV2Model and SGLang's default hidden-state capture contract).
  if (mtp_) {
    const int n = std::min(run.T, max_decode_rows_);
    store_draft_hidden(h_ + size_t(run.T - n) * cfg_.hidden_size, in.req_ids + run.T - n,
                       in.pos + run.T - n, n);
  }
  mimo_project_head(gemm_, h_, globals_.head, logits_, run.T, globals_.vocab_count,
                     cfg_.hidden_size, run.decode, run.all_rows, stream_);
  if (run.capture) return finish_run(run, std::move(out));
  if (boundary_) boundary_->settle();
  // Session bounds validate positions before entry; status still gates commit.
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  std::vector<int32_t> status(cfg_.num_hidden_layers * run.T);
  DGPP_CUDA_OK(cudaMemcpy(status.data(), status_, status.size() * 4, cudaMemcpyDeviceToHost));
  if (std::any_of(status.begin(), status.end(), [](int32_t s) { return s != 0; }))
    throw std::runtime_error("MiMo attention rejected a position");
  return finish_run(run, std::move(out));
}
void MimoModel::graph_prepare() {
  for (auto& layer : layers_) layer->prepare_graph(stream_);
}
void MimoModel::reset_slot_state(int req) {
  check_req(req, "MiMo reset");
  snapshot_history_.rewind(req);
  // Position zero hides old global/ring tails; no full-cache memset per request.
}
void MimoModel::write_state_snapshot(int req, uint8_t* dst, int spec_row) {
  check_req(req, "MiMo snapshot");
  // All verified rows remain in the expanded ring. A rejected tail is masked
  // by position; packing the earlier prefix does not need a full cache copy.
  const int64_t position =
      session_pos_[static_cast<size_t>(req)] - (spec_row >= 0 ? rows_after_for_snapshot_ : 0);
  uint8_t* const base = dst;
  const int64_t previous = snapshot_history_.begin(req, base, position);
  for (size_t l = 0; l < layers_.size(); ++l) {
    const auto& shape = layers_[l]->shape();
    const size_t slots = shape.window ? shape.window : shape.capacity;
    snapshot_copied_bytes_ += mimo_write_snapshot_layer(
        shape, position, previous, keys_[l] + req * key_plane_[l],
        values_[l] + req * value_plane_[l], dst, stream_);
    if (!shape.window)
      snapshot_saved_bytes_ += size_t(previous) * (shape.k_width() + shape.v_width()) * shape.cache_element_bytes();
    dst += slots * (shape.k_width() + shape.v_width()) * shape.cache_element_bytes();
  }
  snapshot_history_.commit(req, base, position);
}
void MimoModel::read_state_snapshot(int req, const uint8_t* src, int64_t position) {
  check_req(req, "MiMo restore");
  snapshot_history_.rewind(req);
  for (size_t l = 0; l < layers_.size(); ++l) {
    const auto& shape = layers_[l]->shape();
    const size_t slots = shape.window ? shape.window : shape.capacity;
    mimo_read_snapshot_layer(shape, position, src, keys_[l] + req * key_plane_[l],
                              values_[l] + req * value_plane_[l], stream_);
    src += slots * (shape.k_width() + shape.v_width()) * shape.cache_element_bytes();
  }
}

size_t MimoModel::draft_state_bytes() const {
  return mtp_ ? cfg_.mtp_blocks * (draft_key_plane_ + draft_value_plane_) * 2 +
                    1 * size_t(draft_history_rows_) * cfg_.hidden_size * 2
              : 0;
}
void MimoModel::write_draft_snapshot(int req, uint8_t* dst, bool live, int64_t position) {
  if (!live && mtp_pos_[req] > position) {
    glm_device_copy(dst, draft_backup_ + req * draft_state_bytes(), draft_state_bytes(), stream_);
    return;
  }
  for (int d = 0; d < cfg_.mtp_blocks; ++d) {
    glm_device_copy(dst, draft_keys_[d] + req * draft_key_plane_, draft_key_plane_ * 2, stream_);
    dst += draft_key_plane_ * 2;
    glm_device_copy(dst, draft_values_[d] + req * draft_value_plane_, draft_value_plane_ * 2,
                    stream_);
    dst += draft_value_plane_ * 2;
    if (d == 0) {
      const size_t n = size_t(draft_history_rows_) * cfg_.hidden_size;
      glm_device_copy(dst, draft_history_[d] + req * n, n * 2, stream_);
      dst += n * 2;
    }
  }
}
void MimoModel::read_draft_snapshot(int req, const uint8_t* src) {
  for (int d = 0; d < cfg_.mtp_blocks; ++d) {
    glm_device_copy(draft_keys_[d] + req * draft_key_plane_, src, draft_key_plane_ * 2, stream_);
    src += draft_key_plane_ * 2;
    glm_device_copy(draft_values_[d] + req * draft_value_plane_, src, draft_value_plane_ * 2,
                    stream_);
    src += draft_value_plane_ * 2;
    if (d == 0) {
      const size_t n = size_t(draft_history_rows_) * cfg_.hidden_size;
      glm_device_copy(draft_history_[d] + req * n, src, n * 2, stream_);
      src += n * 2;
    }
  }
}
void MimoModel::snapshot_draft_state(int req) {
  write_draft_snapshot(req, draft_backup_ + req * draft_state_bytes(), true, 0);
}
void MimoModel::restore_draft_state(int req) {
  read_draft_snapshot(req, draft_backup_ + req * draft_state_bytes());
}
// Chain guesses must not become committed draft history. Keep a separate
// post-catch-up backup so restoring the chain does not destroy the earlier
// pre-draft rollback point used by scalar fallback and prefix snapshots.
void MimoModel::snapshot_chain_state(int req) {
  check_req(req, "MiMo chain snapshot");
  write_draft_snapshot(req, chain_backup_ + req * draft_state_bytes(), true, 0);
}
void MimoModel::restore_chain_state(int req) {
  check_req(req, "MiMo chain restore");
  read_draft_snapshot(req, chain_backup_ + req * draft_state_bytes());
}
void MimoModel::mtp_select_block(int block) {
  if (block < 0 || block >= cfg_.mtp_blocks)
    throw std::invalid_argument("MiMo MTP block outside [0, 2]");
  draft_block_ = block;
}
void MimoModel::mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode,
                             bool capture, int head_rows, int batch_requests) {
  if (!mtp_) refuse_mtp("MiMo draft");
  check_req(req, "MiMo draft");
  if (T < 1 || T > max_tokens_ || head_rows < 0 || head_rows > T ||
      (decode && T > max_decode_rows_) || batch_requests > max_requests_)
    throw std::invalid_argument("MiMo draft rows");
  const int H = cfg_.hidden_size;
  if (!decode) stage_prefill_meta(req, first_pos, T);
  const auto* pos = decode ? d_step_pos_ : d_prefill_pos_;
  const auto* ids = decode ? d_req_ids_ : nullptr;
  const size_t history_offset = decode ? 0 : size_t(req) * draft_history_rows_ * H;
  // Every native head is conditioned on backbone hidden states (MiMo-V2-Flash
  // section 2.3.2), not another MTP block's output. Head d at absolute row p
  // uses backbone row p-d and the token at p+1. Preserve the backbone rows
  // across verification, prefill chunks, recursive proposal setup and restores.
  for (int d = draft_block_; d < cfg_.mtp_blocks; ++d) {
    const uint16_t* hidden = h_;
    if (d == 0 && decode) {
      gather_draft_hidden(ids, pos, draft_hidden_, T);
      hidden = draft_hidden_;
    }
    if (d == 0)
      mimo_mtp_history_store(draft_history_[0] + history_offset, hidden, pos, ids, T, H,
                             draft_history_rows_, 0, stream_);
    const auto* history = d ? draft_history_[0] + history_offset : nullptr;
    mimo_mtp_history_gather(history, draft_hidden_, pos, draft_positions_, ids, T, H,
                            draft_history_rows_, d, stream_);
    if (d) hidden = draft_hidden_;
    const auto& w = draft_weights_[d];
    glm_mtp_input_bf16(globals_.embed, tokens, hidden, nullptr, 0, w.enorm, w.hnorm, draft_input_,
                       T, H, cfg_.rms_norm_eps, stream_);
    gemm_.matmul(draft_input_, w.eh_proj, draft_residual_, T, H, 2 * H, DType::BF16, GemmOut::BF16,
                 2 * H, nullptr, 0, stream_);
    // At the start of a prompt deeper predictors have no preceding hidden.
    // Skip those rows in the prefill kernel; decode uses -1 padding instead.
    const int skip = decode ? 0 : int(std::clamp<int64_t>(d - first_pos, 0, T));
    // Native heads depend directly on backbone hidden rows. Unselected heads
    // need only K/V history; none of their attention/MLP outputs feed another
    // head. Prefill requests no logits, so all heads use the cache-only path.
    const bool cache_only = !head_rows || d != draft_block_;
    if (skip < T)
      draft_layers_[d]->enqueue(draft_residual_ + size_t(skip) * H, draft_positions_ + skip,
                                draft_keys_[d] + (decode ? 0 : req * draft_key_plane_),
                                draft_values_[d] + (decode ? 0 : req * draft_value_plane_), status_,
                                boundary_, stream_, T - skip, decode ? 0 : int(first_pos + T - d),
                                capture, ids, cache_only);
    if (!cache_only) {
      glm_rmsnorm_bf16(draft_residual_, w.final_norm, draft_layer_output_, T, H, cfg_.rms_norm_eps,
                       stream_);
      glm_device_copy(draft_output_, draft_layer_output_, size_t(T) * H * 2, stream_);
    }
  }
  if (!head_rows) return;
  gemm_.matmul(draft_output_ + size_t(T - head_rows) * H, globals_.head, logits_, head_rows,
               globals_.vocab_count, H, DType::BF16, GemmOut::F32, H, nullptr, 0, stream_);
  if (decode && (!capture || decode_tail_mirrors_))
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_,
                                 size_t(head_rows) * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
}
}  // namespace dgpp
