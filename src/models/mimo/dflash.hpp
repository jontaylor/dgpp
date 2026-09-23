#pragma once
#include <array>

#include "engine/boundary_reducer.hpp"
#include "kernels/gemm.hpp"
#include "loaders/safetensors.hpp"
#include "loaders/weight_build.hpp"
namespace dgpp {
bool mimo_dflash_enabled();
// Separate tensor-parallel BF16 drafter. Target embedding/head are borrowed. Q/K/V and MLP are
// sharded; output projections reduce across ranks. FC and norms are replicated, and each rank
// borrows its head vocabulary shard. All allocations are cold-path, kernels capture-safe.
class MimoDFlash {
 public:
  static constexpr int hidden = 4096, width = 20480, block = 8, drafts = 7;
  static constexpr std::array<int, 5> target_layers{0, 11, 23, 35, 47};
  MimoDFlash(const std::string& directory, int rows, int requests, int vocab,
             const uint16_t* embedding, const uint16_t* head, cudaStream_t stream, int rank = 0,
             int world = 1);
  static size_t state_bytes(int rows, int world = 1);
  static size_t memory_bytes(int rows, int requests, int vocab, int world = 1);
  const uint16_t* output_hidden() const { return x_; }
  const uint16_t* mask_embedding() const { return mask_; }
  uint16_t* features() { return features_; }
  uint16_t* gather() { return gather_; }
  const uint16_t* dummy_hidden() const { return dummy_; }
  // Context FC/norm is replicated and context K/V projections are local shards;
  // this path has no all-reduce and retains no reducer pointer.
  void context(const uint16_t* features, const int64_t* pos, const int32_t* req, int rows,
               cudaStream_t stream);
  void propose(const int64_t* tokens, const int64_t* pos, const int32_t* req, int groups,
               int rows_per_group, cudaStream_t stream, bool capture = false,
               BoundaryReducer* current_boundary = nullptr);
  void select(float* logits, int groups, int rows_out, int index, cudaStream_t stream);
  void snapshot(int req, uint8_t* out, cudaStream_t stream);
  void restore(int req, const uint8_t* in, cudaStream_t stream);
  uint8_t* backup(int req, bool chain) {
    return (chain ? chain_ : backup_) + size_t(req) * state_bytes(rows_, world_);
  }
  void prepare();

 private:
  struct Layer {
    const uint16_t *input, *post, *q, *k, *v, *o, *qn, *kn, *sink, *gate, *up, *down;
  };
  int rows_, requests_, vocab_, capacity_, world_;
  CublasLtGemm gemm_;
  LayerBump weights_, scratch_, cache_;
  std::array<Layer, 5> layers_;
  const uint16_t *fc_, *hidden_norm_, *norm_, *embedding_, *head_, *mask_;
  uint16_t *features_, *gather_, *dummy_, *context_, *x_, *y_, *residual_, *q_, *k_, *v_, *attn_,
      *gate_, *up_, *act_;
  float* logits_;
  int64_t *pos_, *tokens_;
  int32_t *req_, *spans_;
  std::array<uint16_t*, 5> keys_, values_;
  uint8_t *backup_, *chain_;
  void project(const uint16_t* x, const uint16_t* w, uint16_t* out, int n, int k, int rows,
               cudaStream_t stream);
};
}  // namespace dgpp
