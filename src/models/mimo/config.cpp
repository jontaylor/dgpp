#include "models/mimo/config.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace dgpp {
namespace {
[[noreturn]] void reject(const std::string& field, const std::string& reason) {
  throw std::runtime_error("MiMo config." + field + ": " + reason);
}
const minijson::Value& field(const minijson::Value& root, std::string_view name) {
  const auto* v = root.find(name);
  if (!v) reject(std::string(name), "missing");
  return *v;
}
void integer(const minijson::Value& root, const std::string& name, int expected) {
  const auto& v = field(root, name);
  if (!v.is_number() || !std::isfinite(v.as_double()) || v.as_double() != expected)
    reject(name, "expected release value " + std::to_string(expected));
}
void number(const minijson::Value& root, const std::string& name, double expected) {
  const auto& v = field(root, name);
  if (!v.is_number() || !std::isfinite(v.as_double()) || v.as_double() != expected)
    reject(name, "unsupported release value");
}
void string(const minijson::Value& root, const std::string& name, const std::string& expected) {
  const auto& v = field(root, name);
  if (!v.is_string() || v.as_string() != expected) reject(name, "expected " + expected);
}
void boolean(const minijson::Value& root, const std::string& name, bool expected) {
  const auto& v = field(root, name);
  if (!v.is_bool() || v.as_bool() != expected) reject(name, "unsupported release value");
}
std::vector<int> pattern(const minijson::Value& root, const std::string& name, int layers) {
  const auto& v = field(root, name);
  if (!v.is_array() || v.items().size() != static_cast<size_t>(layers))
    reject(name, "expected one entry per layer");
  std::vector<int> out;
  for (const auto& item : v.items()) {
    if (!item.is_number() || (item.as_double() != 0 && item.as_double() != 1))
      reject(name, "entries must be 0 or 1");
    out.push_back(static_cast<int>(item.as_double()));
  }
  return out;
}
}  // namespace

MimoTextConfig MimoTextConfig::parse(const minijson::Value& root) {
  if (!root.is_object()) reject("root", "expected object");
  MimoTextConfig c;
  string(root, "model_type", "mimo_v2");
  const auto& arch = field(root, "architectures");
  if (!arch.is_array() || arch.items().size() != 1 || !arch.items()[0].is_string() ||
      arch.items()[0].as_string() != "MiMoV2ForCausalLM")
    reject("architectures", "expected MiMoV2ForCausalLM");
  // Intentionally narrow initial contract: other MiMo releases must not
  // silently inherit this checkpoint's shapes or quantization layout.
  integer(root, "eos_token_id", 151645);
  integer(root, "hidden_size", c.hidden_size);
  integer(root, "vocab_size", c.vocab_size);
  integer(root, "num_hidden_layers", c.num_hidden_layers);
  integer(root, "num_attention_heads", c.num_attention_heads);
  integer(root, "swa_num_attention_heads", c.num_attention_heads);
  integer(root, "num_key_value_heads", c.num_key_value_heads);
  integer(root, "swa_num_key_value_heads", c.swa_num_key_value_heads);
  integer(root, "head_dim", c.head_dim);
  integer(root, "swa_head_dim", c.head_dim);
  integer(root, "v_head_dim", c.v_head_dim);
  integer(root, "swa_v_head_dim", c.v_head_dim);
  integer(root, "sliding_window", c.sliding_window);
  integer(root, "sliding_window_size", c.sliding_window);
  integer(root, "intermediate_size", c.intermediate_size);
  integer(root, "moe_intermediate_size", c.moe_intermediate_size);
  integer(root, "n_routed_experts", c.n_routed_experts);
  integer(root, "num_experts_per_tok", c.num_experts_per_tok);
  integer(root, "max_position_embeddings", c.max_position_embeddings);
  integer(root, "n_group", 1);
  integer(root, "topk_group", 1);
  number(root, "partial_rotary_factor", 0.334);
  // The reference truncates this product, rather than requiring it integral.
  c.rotary_dim = static_cast<int>(c.head_dim * field(root, "partial_rotary_factor").as_double());
  number(root, "rope_theta", c.rope_theta);
  number(root, "swa_rope_theta", c.swa_rope_theta);
  number(root, "layernorm_epsilon", c.rms_norm_eps);
  number(root, "attention_value_scale", c.attention_value_scale);
  number(root, "attention_dropout", 0);
  string(root, "attention_projection_layout", "fused_qkv");
  string(root, "hidden_act", "silu");
  string(root, "scoring_func", "sigmoid");
  string(root, "topk_method", "noaux_tc");
  string(root, "moe_router_dtype", "bfloat16");
  string(root, "dtype", "bfloat16");
  boolean(root, "attention_bias", false);
  boolean(root, "add_full_attention_sink_bias", false);
  boolean(root, "add_swa_attention_sink_bias", true);
  boolean(root, "norm_topk_prob", true);
  boolean(root, "tie_word_embeddings", false);
  if (!field(root, "n_shared_experts").is_null()) reject("n_shared_experts", "expected null");
  const auto& scale = field(root, "routed_scaling_factor");
  if (!scale.is_null()) number(root, "routed_scaling_factor", 1.0);
  const auto& rope = field(root, "rope_parameters");
  if (!rope.is_object()) reject("rope_parameters", "expected object");
  string(rope, "rope_type", "default");
  number(rope, "rope_theta", c.rope_theta);
  number(rope, "partial_rotary_factor", 0.334);
  if (const auto* old = root.find("rope_scaling"); old && !old->is_null())
    reject("rope_scaling", "legacy override is not supported");
  c.hybrid_layer_pattern = pattern(root, "hybrid_layer_pattern", c.num_hidden_layers);
  c.moe_layer_freq = pattern(root, "moe_layer_freq", c.num_hidden_layers);
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    const bool global = l == 0 || l % 6 == 5;
    if (c.sliding(l) == global)
      reject("hybrid_layer_pattern", "expected release's 9 global / 39 SWA layers");
    if (c.moe(l) != (l != 0))
      reject("moe_layer_freq", "expected dense first layer and 47 MoE layers");
  }
  const auto& q = field(root, "quantization_config");
  if (!q.is_object()) reject("quantization_config", "expected object");
  string(q, "quant_method", "fp8");
  string(q, "store_dtype", "mxfp4");
  string(q, "fmt", "e4m3");
  string(q, "activation_scheme", "dynamic");
  integer(q, "mxfp4_block_size", 32);
  const auto& block = field(q, "weight_block_size");
  if (!block.is_array() || block.items().size() != 2)
    reject("weight_block_size", "expected [128,128]");
  for (const auto& v : block.items())
    if (!v.is_number() || v.as_double() != 128) reject("weight_block_size", "expected [128,128]");
  // Three conventional MTP heads exist under model.mtp.layers and have a
  // separate binding from the backbone and the five-layer dflash/ checkpoint.
  integer(root, "num_nextn_predict_layers", 3);
  return c;
}

MimoTextConfig MimoTextConfig::from_json_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open MiMo config " + path);
  const std::string text{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
  if (f.bad()) throw std::runtime_error("cannot read MiMo config " + path);
  const auto doc = minijson::parse(text);
  return parse(doc.root);
}

bool MimoTextConfig::sliding(int layer) const {
  if (is_mtp(layer)) return true;
  return hybrid_layer_pattern.at(layer) == 1;
}
bool MimoTextConfig::moe(int layer) const {
  if (is_mtp(layer)) return false;
  return moe_layer_freq.at(layer) == 1;
}
int MimoTextConfig::kv_heads(int layer) const {
  return sliding(layer) ? swa_num_key_value_heads : num_key_value_heads;
}
int MimoTextConfig::qkv_rows(int layer) const {
  return num_attention_heads * head_dim + kv_heads(layer) * (head_dim + v_head_dim);
}
void MimoTextConfig::validate_tp(int world) const {
  if (world != 1 && world != 2 && world != 4) reject("world", "expected 1, 2 or 4");
}
uint64_t MimoTextConfig::kv_bytes(int context_tokens, int world) const {
  validate_tp(world);
  if (context_tokens <= 0 || context_tokens > max_position_embeddings)
    reject("context_tokens", "outside model context range");
  uint64_t total = 0;
  for (int l = 0; l < num_hidden_layers; ++l) {
    const int tokens = sliding(l) ? std::min(context_tokens, sliding_window) : context_tokens;
    total += uint64_t(tokens) * (kv_heads(l) / world) * (head_dim + v_head_dim) * 2;
  }
  return total;
}
}  // namespace dgpp
