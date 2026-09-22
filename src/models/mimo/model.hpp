#pragma once

#include "engine/memory_plan.hpp"
#include "engine/paged_blocks.hpp"
#include "engine/session_model.hpp"
#include "models/mimo/dflash.hpp"
#include "models/mimo/layers.hpp"
#include "models/mimo/snapshot.hpp"

namespace dgpp {
// Independent flat-cache sessions with chunked projections, grouped MoE,
// bounded attention workspace and up to three distinct native MTP heads.
// Expanded rings preserve history during prefill and speculative verification.
class MimoModel : public SessionModel<MimoModel> {
 public:
  using Base = SessionModel<MimoModel>;
  using Base::Outputs;
  using Base::RowRun;
  MimoModel(const MimoTextConfig& cfg, const std::string& checkpoint, int forward_rows,
            int64_t context, BoundaryReducer* boundary, int rank, int world, int requests = 1,
            bool mtp = false);
  ~MimoModel();
  static constexpr bool kResumablePrefill = true;
  static constexpr int prefill_chunk_tokens() { return 2048; }
  static MemoryPlan plan_memory(const MimoTextConfig& cfg, int forward_rows, int64_t context,
                                int rank, int world, int requests = 1, bool mtp = false);
  static size_t snapshot_bytes(const MimoTextConfig& cfg, int64_t context, int world, int rows);
  Outputs run_rows(const RowRun& run);
  // Admission units are single tokens in the fixed-capacity flat cache.
  int64_t kv_blocks_total() const { return max_context_ * max_requests_; }
  int64_t kv_blocks_in_use() const {
    int64_t used = 0;
    for (auto pos : session_pos_) used += pos;
    return used;
  }
  int64_t kv_blocks_for_tokens(int64_t tokens) const { return tokens; }
  int64_t kv_block_tokens() const { return 1; }
  void reset_slot_state(int req);
  // Global K/V retains fixed capacity; sliding windows are packed to 128 entries.
  // Copy only populated history. No block sharing.
  size_t snapshot_state_bytes() const {
    return snapshot_bytes(cfg_, max_context_, world_, chunk_capacity_);
  }
  size_t draft_state_bytes() const;
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void register_state_snapshot(const void* dst) { snapshot_history_.register_destination(dst); }
  void unregister_state_snapshot(const void* dst) { snapshot_history_.unregister_destination(dst); }
  void invalidate_state_snapshot(const void* dst) { snapshot_history_.release(dst); }
  void rollback_state_snapshots(int req, int64_t position) {
    snapshot_history_.rewind(req, position);
  }
  // Base K/V only: draft/MTP snapshot bytes are unchanged and excluded.
  uint64_t snapshot_copied_bytes() const { return snapshot_copied_bytes_; }
  uint64_t snapshot_saved_bytes() const { return snapshot_saved_bytes_; }
  void read_state_snapshot(int req, const uint8_t* src, int64_t position);
  GlmSpecSegments spec_segments(int, int = 0) const {
    // Expanded rings retain the live window before all verification rows.
    // Rollback only changes position; the next walk overwrites rejected tails.
    return {};
  }
  bool has_pool() const { return false; }
  // Compile-time pool protocol for the shared session core; never initialized
  // or used because flat K/V is captured by write/read_state_snapshot.
  struct UnusedPool : PagedBlockTable {
    void copy_block_contents(int32_t, int32_t, cudaStream_t) {
      throw std::logic_error("MiMo has no paged cache");
    }
  };
  UnusedPool& pool() { return unused_pool_; }
  const UnusedPool& pool() const { return unused_pool_; }
  void graph_prepare();
  static constexpr bool kDraftChain = true;
  static constexpr bool kBatchedDraftChain = true;
  void mtp_select_block(int block);
  void mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode,
                    bool capture, int head_rows, int batch_requests);
  void write_draft_snapshot(int req, uint8_t* dst, bool live, int64_t position);
  void read_draft_snapshot(int req, const uint8_t* src);
  void snapshot_draft_state(int req);
  void restore_draft_state(int req);
  const uint16_t* draft_hidden_rows() const {
    return dflash_ ? dflash_->dummy_hidden() : draft_output_;
  }
  void snapshot_chain_state(int req);
  void restore_chain_state(int req);

 private:
  std::unique_ptr<MimoDFlash> dflash_;
  MimoSnapshotHistory snapshot_history_;
  uint64_t snapshot_copied_bytes_ = 0, snapshot_saved_bytes_ = 0;
  MimoTextConfig cfg_;
  std::vector<MimoLayerResident> draft_weights_;
  std::vector<std::unique_ptr<MimoDecoderLayer>> draft_layers_;
  int draft_block_ = 0;
  int draft_history_rows_ = 0;
  int64_t* draft_positions_ = nullptr;
  std::vector<uint16_t*> draft_history_;
  uint16_t* draft_layer_output_ = nullptr;
  LayerBump draft_storage_;
  uint16_t *draft_hidden_ = nullptr, *draft_input_ = nullptr, *draft_residual_ = nullptr;
  uint16_t* draft_output_ = nullptr;
  std::vector<uint16_t*> draft_keys_, draft_values_;
  uint8_t *draft_backup_ = nullptr, *chain_backup_ = nullptr;
  size_t draft_key_plane_ = 0, draft_value_plane_ = 0;
  CublasLtGemm gemm_;
  MimoGlobalsResident globals_;
  std::vector<MimoLayerResident> weights_;
  LayerBump cache_audit_;
  LayerBump cache_, scratch_, attention_scores_, layer_workspace_;
  std::vector<uint8_t*> keys_, values_;
  std::vector<size_t> key_plane_, value_plane_;  // bytes per request, unlike BF16 draft planes
  std::unique_ptr<GlmMoeLayer> shared_moe_;
  std::vector<std::unique_ptr<MimoDecoderLayer>> layers_;
  uint16_t* residual_ = nullptr;
  int32_t* status_ = nullptr;
  int chunk_capacity_ = 1;
  UnusedPool unused_pool_;
};
}  // namespace dgpp
