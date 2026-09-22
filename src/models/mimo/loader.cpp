#include "models/mimo/loader.hpp"

#include <cstdlib>
#include <limits>
#include <string>

namespace dgpp {
namespace {
size_t aligned(size_t n) {
  return (n + 255) & ~size_t(255);
}
struct Upload {
  LayerBump& arena;
  template <class T>
  const T* copy(const std::vector<T>& v) {
    auto* p = static_cast<T*>(arena.alloc(v.size() * sizeof(T)));
    DGPP_CUDA_OK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
  }
};
}  // namespace
GlmMoeConfig mimo_moe_config(const MimoTextConfig& cfg, int world) {
  cfg.validate_tp(world);
  GlmMoeConfig m;
  m.hidden = cfg.hidden_size;
  m.inter = cfg.moe_intermediate_size / world;
  m.n_experts = cfg.n_routed_experts;
  m.top_k = cfg.num_experts_per_tok;
  m.n_shared_experts = 0;
  m.routed_scaling_factor = cfg.routed_scaling_factor;
  m.norm_topk_prob = true;
  m.swiglu_limit = std::numeric_limits<float>::infinity();
  m.router_mode = MoeRouterMode::SigmoidBias;
  GlmMoeConfig::validate_config(m);
  return m;
}
GlmMoeWeights MimoLayerResident::moe_view() const {
  GlmMoeWeights w;
  w.router_gate = router;
  w.router_bias = router_bias;
  w.experts_fp4 = experts.data();
  return w;
}
bool mimo_fp8_dense_enabled() {
  const char* mode = std::getenv("DGPP_MIMO_FP8_DENSE");
  if (mode && std::string(mode) != "0" && std::string(mode) != "1")
    throw std::invalid_argument("DGPP_MIMO_FP8_DENSE must be 0 or 1");
  return mode && std::string(mode) == "1";
}
bool mimo_fp8_dense_prefill_bf16_enabled() {
  const char* mode = std::getenv("DGPP_MIMO_FP8_DENSE_PREFILL_BF16");
  if (mode && std::string(mode) != "0" && std::string(mode) != "1")
    throw std::invalid_argument("DGPP_MIMO_FP8_DENSE_PREFILL_BF16 must be 0 or 1");
  const bool enabled = mode && std::string(mode) == "1";
  if (enabled && !mimo_fp8_dense_enabled())
    throw std::invalid_argument("DGPP_MIMO_FP8_DENSE_PREFILL_BF16 requires DGPP_MIMO_FP8_DENSE=1");
  return enabled;
}
MimoDeviceLoader::MimoDeviceLoader(const std::string& checkpoint, int rank, int world)
    : reader_(checkpoint), rank_(rank), world_(world) {
  fp8_dense_ = mimo_fp8_dense_enabled();
  config().validate_tp(world);
  if (rank < 0 || rank >= world) throw std::invalid_argument("MiMo loader: invalid rank");
}
size_t MimoDeviceLoader::layer_bytes(const MimoTextConfig& c, int layer, int world,
                                     bool fp8_dense) {
  c.validate_tp(world);
  const size_t h = c.hidden_size;
  auto dense = [&](size_t n, size_t k) {
    return fp8_dense ? aligned(n * k) + aligned((n / 64) * (k / 64) * 4) : aligned(n * k * 2);
  };
  size_t bytes = 2 * aligned(h * 2) + dense(c.qkv_rows(layer) / world, h) +
                 aligned(h * (c.num_attention_heads / world) * c.v_head_dim * 2);
  if (c.sliding(layer)) bytes += aligned((c.num_attention_heads / world) * 2);
  if (c.moe(layer)) {
    bytes += aligned(size_t(c.n_routed_experts) * h * 2) + aligned(c.n_routed_experts * 4);
    const size_t n = h * (c.moe_intermediate_size / world);
    bytes += size_t(c.n_routed_experts) * 3 * (aligned(n / 2) + aligned(n / 32));
  } else {
    bytes += 3 * dense(h, c.intermediate_size / world);
  }
  if (c.is_mtp(layer)) bytes += 3 * aligned(h * 2) + aligned(h * h * 4);
  return bytes;
}
size_t MimoDeviceLoader::globals_bytes(const MimoTextConfig& c, int world) {
  c.validate_tp(world);
  return aligned(size_t(c.vocab_size) * c.hidden_size * 2) + aligned(c.hidden_size * 2) +
         aligned(size_t(c.vocab_size / world) * c.hidden_size * 2);
}
MimoLayerResident MimoDeviceLoader::load_layer(int layer) {
  const auto& c = config();
  MimoLayerResident r;
  r.storage = std::make_unique<LayerBump>();
  r.storage->init(layer_bytes(c, layer, world_, fp8_dense_));
  Upload u{*r.storage};
  r.layer = layer;
  r.q_heads = c.num_attention_heads / world_;
  r.kv_heads = c.kv_heads(layer) / world_;
  const bool draft = c.is_mtp(layer);
  const auto p = draft ? "model.mtp.layers." + std::to_string(layer - c.mtp_layer()) + "."
                       : "model.layers." + std::to_string(layer) + ".";
  auto ordinary = [&](const std::string& name) {
    return u.copy(reader_.bf16(p + name, rank_, world_).values);
  };
  r.input_norm = ordinary("input_layernorm.weight");
  r.post_norm = ordinary(draft ? "pre_mlp_layernorm.weight" : "post_attention_layernorm.weight");
  auto packed = [&](const MimoFp8Matrix& m) {
    return MimoFp8Resident{u.copy(m.payload), u.copy(m.scales)};
  };
  if (fp8_dense_) {
    r.qkv_fp8 = packed(reader_.qkv_fp8(layer, rank_, world_));
  } else {
    r.qkv = u.copy(reader_.qkv(layer, rank_, world_, 0, c.hidden_size).values);
  }
  r.output = ordinary("self_attn.o_proj.weight");
  if (c.sliding(layer)) r.sinks = ordinary("self_attn.attention_sink_bias");
  if (c.moe(layer)) {
    r.router = ordinary("mlp.gate.weight");
    r.router_bias = u.copy(reader_.router_bias(layer));
    r.experts.reserve(c.n_routed_experts * 3);
    for (int e = 0; e < c.n_routed_experts; ++e) {
      for (const auto* proj : {"gate_proj", "up_proj", "down_proj"}) {
        const auto host = reader_.expert(layer, e, proj, rank_, world_);
        GlmFp4Matrix m;
        m.rows = host.rows;
        m.cols = host.cols;
        m.payload = u.copy(host.payload);
        m.scales = u.copy(host.scales);
        m.scale_group = kMxfp4Group;
        r.experts.push_back(m);
      }
    }
  } else {
    r.dense_inter = c.intermediate_size / world_;
    if (fp8_dense_) {
      r.gate_fp8 = packed(reader_.fp8(p + "mlp.gate_proj.weight", rank_, world_));
      r.up_fp8 = packed(reader_.fp8(p + "mlp.up_proj.weight", rank_, world_));
      r.down_fp8 = packed(reader_.fp8(p + "mlp.down_proj.weight", rank_, world_));
    } else {
      r.gate = ordinary("mlp.gate_proj.weight");
      r.up = ordinary("mlp.up_proj.weight");
      r.down = ordinary("mlp.down_proj.weight");
    }
  }
  if (draft) {
    r.enorm = ordinary("enorm.weight");
    r.hnorm = ordinary("hnorm.weight");
    r.eh_proj = ordinary("eh_proj.weight");
    r.final_norm = ordinary("final_layernorm.weight");
  }
  if (r.storage->cursor != r.storage->capacity)
    throw std::logic_error("MiMo layer byte plan mismatch");
  return r;
}
MimoGlobalsResident MimoDeviceLoader::load_globals() {
  MimoGlobalsResident g;
  const auto& c = config();
  g.storage = std::make_unique<LayerBump>();
  g.storage->init(globals_bytes(c, world_));
  Upload u{*g.storage};
  // Upload embedding/head by rows to cap the transient host footprint.
  auto matrix = [&](const std::string& name) {
    const auto shape = reader_.placement(name, rank_, world_);
    auto* dst = static_cast<uint16_t*>(g.storage->alloc(shape.rows * shape.cols * 2));
    for (int64_t row = 0; row < shape.rows; row += 1024) {
      const auto part =
          reader_.bf16(name, rank_, world_, row, std::min<int64_t>(1024, shape.rows - row));
      DGPP_CUDA_OK(cudaMemcpy(dst + row * shape.cols, part.values.data(), part.values.size() * 2,
                              cudaMemcpyHostToDevice));
    }
    return dst;
  };
  g.embed = matrix("model.embed_tokens.weight");
  g.norm = u.copy(reader_.bf16("model.norm.weight", rank_, world_).values);
  g.head = matrix("lm_head.weight");
  g.vocab_count = c.vocab_size / world_;
  g.vocab_begin = rank_ * g.vocab_count;
  if (g.storage->cursor != g.storage->capacity)
    throw std::logic_error("MiMo global byte plan mismatch");
  return g;
}
}  // namespace dgpp
