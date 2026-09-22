#pragma once
#include <bit>

#include "engine/boundary_reducer.hpp"
#include "kernels/gemm.hpp"
#include "kernels/mimo_attn.hpp"
#include "models/glm/moe_layer.hpp"
#include "models/mimo/loader.hpp"

namespace dgpp {
inline int mimo_ring_capacity(int rows) {
  return std::bit_ceil(unsigned(128 + rows - 1));
}
// One sequence per chunk, consecutive positions. Expanded rings retain the
// window plus the chunk, allowing causal attention queries to run in parallel.
// The owning model supplies the caches and handles snapshot/rollback.
class MimoDecoderLayer {
 public:
  static constexpr int attention_tile_rows = 128;
  MimoDecoderLayer(const MimoLayerResident& weights, const MimoTextConfig& config, int requests,
                   int global_capacity, IGemm& gemm, void* gemm_workspace,
                   size_t gemm_workspace_bytes, int max_tokens = 1,
                   float* attention_scores = nullptr, GlmMoeLayer* shared_moe = nullptr,
                   LayerBump* shared_scratch = nullptr);
  // Weights, GEMM and workspace must outlive this layer. All collectives
  // happen after the corresponding local projection, before residual add.
  // Status is device [tokens], and must be checked by the model before
  // committing a step. Invalid rows do not mutate K/V (residuals are scratch).
  // cache_only appends exactly the same K/V, leaves residual unchanged and
  // skips attention, output/MLP projections and all collectives. Use it only
  // when no hidden output is needed from this layer.
  void enqueue(uint16_t* residual, const int64_t* positions, uint16_t* k_cache, uint16_t* v_cache,
               int32_t* status, BoundaryReducer* boundary, cudaStream_t stream, int tokens = 1,
               int end_key = 0, bool capture = false, const int32_t* request_ids = nullptr,
               bool cache_only = false);
  static size_t workspace_bytes(const MimoTextConfig& cfg, int layer, int rows, int world);
  void prepare_graph(cudaStream_t stream);
  const MimoAttentionShape& shape() const { return shape_; }
  size_t scratch_bytes() const { return scratch_.capacity; }

 private:
  friend struct MimoLayerProbe;
  void project(const uint16_t* x, const uint16_t* w, uint16_t* y, int n, int k, cudaStream_t stream,
               int tokens);
  void fold(uint16_t* partial, BoundaryReducer* boundary, cudaStream_t stream, int tokens,
            bool capture);
  const MimoLayerResident& w_;
  MimoTextConfig cfg_;
  MimoAttentionShape shape_;
  IGemm& gemm_;
  void* gemm_ws_;
  size_t gemm_ws_bytes_;
  int max_tokens_;
  float* attention_scores_;
  LayerBump scratch_, frequencies_;
  uint16_t *x_, *y_, *fused_, *q_, *attn_, *gate_, *up_, *act_;
  float *freq_, *moe_sum_;
  std::unique_ptr<GlmMoeLayer> owned_moe_;
  GlmMoeLayer* moe_ = nullptr;
  int graph_slot_ = 0;
};
}  // namespace dgpp
