#pragma once

#include <memory>
#include <vector>

#include "loaders/weight_build.hpp"
#include "models/glm/moe.hpp"
#include "models/mimo/weights.hpp"

namespace dgpp {
// Owned per-layer allocation; views remain stable until this object dies.
// Cold loading uses bounded host matrices and synchronous copies. It never
// allocates a full host image of the model or expands MXFP4 experts.
bool mimo_fp8_dense_enabled();
bool mimo_fp8_dense_prefill_bf16_enabled();
struct MimoFp8Resident {
  const uint8_t* payload = nullptr;
  const float* scales = nullptr;
};
struct MimoLayerResident {
  MimoFp8Resident qkv_fp8, gate_fp8, up_fp8, down_fp8;
  std::unique_ptr<LayerBump> storage;
  int layer = -1, q_heads = 0, kv_heads = 0, dense_inter = 0;
  const uint16_t *input_norm = nullptr, *post_norm = nullptr;
  const uint16_t *qkv = nullptr, *output = nullptr, *sinks = nullptr;
  const uint16_t *gate = nullptr, *up = nullptr, *down = nullptr;
  const uint16_t *enorm = nullptr, *hnorm = nullptr, *eh_proj = nullptr, *final_norm = nullptr;
  const uint16_t* router = nullptr;
  const float* router_bias = nullptr;
  std::vector<GlmFp4Matrix> experts;
  GlmMoeWeights moe_view() const;
};
struct MimoGlobalsResident {
  std::unique_ptr<LayerBump> storage;
  const uint16_t *embed = nullptr, *norm = nullptr, *head = nullptr;
  int vocab_begin = 0, vocab_count = 0;
};
class MimoDeviceLoader {
 public:
  MimoDeviceLoader(const std::string& checkpoint, int rank, int world);
  const MimoTextConfig& config() const { return reader_.config(); }
  MimoLayerResident load_layer(int layer);
  MimoGlobalsResident load_globals();
  static size_t layer_bytes(const MimoTextConfig& cfg, int layer, int world,
                            bool fp8_dense = mimo_fp8_dense_enabled());
  static size_t globals_bytes(const MimoTextConfig& cfg, int world);

 private:
  MimoCheckpointWeights reader_;
  int rank_, world_;
  bool fp8_dense_ = false;
};
GlmMoeConfig mimo_moe_config(const MimoTextConfig& cfg, int world = 1);
}  // namespace dgpp
