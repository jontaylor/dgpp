#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "models/mimo/binding.hpp"

namespace {
void check(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
std::string fixture(const std::string& name) {
  std::ifstream f(std::string(DGPP_SOURCE_DIR) + "/tests/data/mimo/" + name);
  check(bool(f), "cannot read " + name);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
dgpp::MimoTextConfig config() {
  const auto text = fixture("config.json");
  const auto doc = dgpp::minijson::parse(text);
  return dgpp::MimoTextConfig::parse(doc.root);
}
template <class F>
void rejects(F f, const std::string& field) {
  try {
    f();
  } catch (const std::exception& e) {
    check(std::string(e.what()).find(field) != std::string::npos,
          "wrong diagnostic: " + std::string(e.what()));
    return;
  }
  throw std::runtime_error("accepted invalid " + field);
}
}  // namespace

DGPP_TEST(mimo_config_release_geometry_and_kv_budget) {
  const auto c = config();
  int swa = 0, moe = 0;
  for (int l = 0; l < 48; ++l) {
    swa += c.sliding(l);
    moe += c.moe(l);
  }
  check(swa == 39 && moe == 47, "layer counts");
  check(c.rotary_dim == 64, "fractional RoPE product must truncate to 64");
  check(c.qkv_rows(0) == 13568 && c.qkv_rows(1) == 14848, "QKV widths");
  // Independent storage arithmetic: nine global layers, four KV heads,
  // plus 39 rings of 128 entries with eight KV heads. K=192, V=128.
  const uint64_t expected = (uint64_t(9) * 4096 * 4 + uint64_t(39) * 128 * 8) * 640;
  check(c.kv_bytes(4096, 1) == expected, "KV memory");
  check(c.kv_bytes(4096, 2) == expected / 2 && c.kv_bytes(4096, 4) == expected / 4,
        "TP KV sharding");
  check(c.kv_bytes(1, 1) == uint64_t(9 * 4 + 39 * 8) * 640, "short context rings");
  rejects([&] { c.kv_bytes(0, 2); }, "context_tokens");
  rejects([&] { c.kv_bytes(1048577, 2); }, "context_tokens");
  rejects([&] { c.validate_tp(8); }, "world");
}

DGPP_TEST(mimo_config_rejects_incompatible_fields) {
  struct Patch {
    const char* from;
    const char* to;
    const char* field;
  };
  const Patch patches[] = {
      {"\"head_dim\": 192", "\"head_dim\": 192.5", "head_dim"},
      {"\"head_dim\": 192", "\"head_dim\": 1e100", "head_dim"},
      {"\"head_dim\": 192", "\"head_dim\": true", "head_dim"},
      {"\"swa_num_key_value_heads\": 8", "\"swa_num_key_value_heads\": 0",
       "swa_num_key_value_heads"},
      {"\"partial_rotary_factor\": 0.334", "\"partial_rotary_factor\": 0.5",
       "partial_rotary_factor"},
      {"\"sliding_window\": 128", "\"sliding_window\": null", "sliding_window"},
      {"\"n_shared_experts\": null", "\"n_shared_experts\": 1", "n_shared_experts"},
      {"\"store_dtype\": \"mxfp4\"", "\"store_dtype\": \"nvfp4\"", "store_dtype"},
      {"\"add_swa_attention_sink_bias\": true", "\"add_swa_attention_sink_bias\": false",
       "add_swa_attention_sink_bias"},
      {"\"attention_projection_layout\": \"fused_qkv\"",
       "\"attention_projection_layout\": \"split\"", "attention_projection_layout"},
      {"\"hybrid_layer_pattern\": [", "\"hybrid_layer_pattern\": [0,", "hybrid_layer_pattern"},
      {"\"moe_layer_freq\": [\n    0", "\"moe_layer_freq\": [\n    1", "moe_layer_freq"},
      {"\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"", "rope_type"},
  };
  for (const auto& p : patches) {
    auto text = fixture("config.json");
    const auto at = text.find(p.from);
    check(at != std::string::npos, "missing patch anchor");
    text.replace(at, std::string(p.from).size(), p.to);
    rejects(
        [&] {
          const auto doc = dgpp::minijson::parse(text);
          dgpp::MimoTextConfig::parse(doc.root);
        },
        p.field);
  }
}

DGPP_TEST(mimo_binding_matches_observed_release_headers) {
  const auto text = fixture("tensor_headers.json");
  const auto doc = dgpp::minijson::parse(text);
  std::unordered_map<std::string, dgpp::MimoTensorDesc> expected;
  for (const auto& t : dgpp::mimo_expected_text_tensors(config())) expected.emplace(t.name, t);
  for (const auto& item : doc.root.members()) {
    const std::string name(item.key);
    const auto it = expected.find(name);
    check(it != expected.end(), "unrecognized fixture tensor " + name);
    const auto dtype = dgpp::dtype_from_string(item.value.at("dtype").as_string());
    check(dtype && *dtype == it->second.dtype, "dtype " + name);
    std::vector<int64_t> shape;
    for (const auto& d : item.value.at("shape").items()) shape.push_back(d.as_int());
    check(shape == it->second.shape, "shape " + name);
  }
  check(expected.size() == 72574, "main text tensor count from published index");
  check(!expected.contains("model.layers.48.input_layernorm.weight"),
        "must not infer MTP layers from draft metadata");
}

DGPP_TEST(mimo_binding_rejects_missing_wrong_and_extra_tensors) {
  const auto c = config();
  std::unordered_map<std::string, dgpp::MimoTensorDesc> tensors;
  for (const auto& t : dgpp::mimo_expected_text_tensors(c)) tensors.emplace(t.name, t);
  check(dgpp::mimo_validate_text_binding(c, tensors).mtp == 0, "complete binding");
  const std::string name = "model.layers.0.self_attn.qkv_proj.weight_scale_inv";
  const auto good = tensors.at(name);
  tensors.at(name).shape[0] = 106;
  rejects([&] { dgpp::mimo_validate_text_binding(c, tensors); }, name);
  tensors.at(name) = good;
  tensors.at(name).dtype = dgpp::DType::BF16;
  rejects([&] { dgpp::mimo_validate_text_binding(c, tensors); }, name);
  tensors.erase(name);
  rejects([&] { dgpp::mimo_validate_text_binding(c, tensors); }, name);
  tensors.emplace(name, good);
  tensors.emplace("visual.example", good);
  check(dgpp::mimo_validate_text_binding(c, tensors).multimodal == 1,
        "multimodal explicitly unvalidated");
  tensors.emplace("model.mtp.layers.0.eh_proj.weight", good);
  check(dgpp::mimo_validate_text_binding(c, tensors).mtp == 1, "MTP explicitly unvalidated");
  tensors.emplace("model.layers.48.input_layernorm.weight", good);
  rejects([&] { dgpp::mimo_validate_text_binding(c, tensors); }, "unexpected");
}
