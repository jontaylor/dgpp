#include "models/mimo/binding.hpp"

#include <stdexcept>
#include <unordered_set>

namespace dgpp {
std::vector<MimoExpectedTensor> mimo_expected_text_tensors(const MimoTextConfig& c) {
  std::vector<MimoExpectedTensor> out;
  auto add = [&](const std::string& name, DType dtype, std::vector<int64_t> shape) {
    out.push_back({{dtype, std::move(shape)}, name});
  };
  auto fp8 = [&](const std::string& base, int64_t rows, int64_t cols, int64_t scale_rows = 0) {
    add(base + ".weight", DType::F8_E4M3, {rows, cols});
    add(base + ".weight_scale_inv", DType::F32,
        {scale_rows ? scale_rows : (rows + 127) / 128, (cols + 127) / 128});
  };
  auto fp4 = [&](const std::string& base, int64_t rows, int64_t cols) {
    add(base + ".weight", DType::U8, {rows, cols / 2});
    add(base + ".weight_scale", DType::U8, {rows, cols / 32});
  };
  const int64_t h = c.hidden_size;
  add("model.embed_tokens.weight", DType::BF16, {c.vocab_size, h});
  add("lm_head.weight", DType::BF16, {c.vocab_size, h});
  add("model.norm.weight", DType::BF16, {h});
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    add(p + "input_layernorm.weight", DType::BF16, {h});
    add(p + "post_attention_layernorm.weight", DType::BF16, {h});
    // Observed release storage, independent of serving world size. The
    // index declares tp_size=4; global QKV's 108 scale rows are NOT the
    // ordinary ceil(13568/128)=106 grid. MimoQkvLayout maps payload rows
    // to chunk-local scales before gathering canonical Q/K/V heads.
    fp8(p + "self_attn.qkv_proj", c.qkv_rows(l), h, c.sliding(l) ? 116 : 108);
    add(p + "self_attn.o_proj.weight", DType::BF16, {h, c.num_attention_heads * c.v_head_dim});
    if (c.sliding(l))
      add(p + "self_attn.attention_sink_bias", DType::BF16, {c.num_attention_heads});
    if (!c.moe(l)) {
      fp8(p + "mlp.gate_proj", c.intermediate_size, h);
      fp8(p + "mlp.up_proj", c.intermediate_size, h);
      fp8(p + "mlp.down_proj", h, c.intermediate_size);
    } else {
      add(p + "mlp.gate.weight", DType::BF16, {c.n_routed_experts, h});
      add(p + "mlp.gate.e_score_correction_bias", DType::F32, {c.n_routed_experts});
      for (int e = 0; e < c.n_routed_experts; ++e) {
        const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
        fp4(ep + "gate_proj", c.moe_intermediate_size, h);
        fp4(ep + "up_proj", c.moe_intermediate_size, h);
        fp4(ep + "down_proj", h, c.moe_intermediate_size);
      }
    }
  }
  return out;
}

// Bind each native predictor independently.
std::vector<MimoExpectedTensor> mimo_expected_mtp_tensors(const MimoTextConfig& c) {
  std::vector<MimoExpectedTensor> out;
  for (int block = 0; block < c.mtp_blocks; ++block) {
    const std::string p = "model.mtp.layers." + std::to_string(block) + ".";
    auto add = [&](const std::string& n, DType dtype, std::vector<int64_t> shape) {
      out.push_back({{dtype, std::move(shape)}, p + n});
    };
    const int64_t h = c.hidden_size;
    for (auto n : {"enorm", "hnorm", "final_layernorm", "input_layernorm", "pre_mlp_layernorm"})
      add(std::string(n) + ".weight", DType::BF16, {h});
    add("eh_proj.weight", DType::BF16, {h, 2 * h});
    add("self_attn.o_proj.weight", DType::BF16, {h, c.num_attention_heads * c.v_head_dim});
    add("self_attn.attention_sink_bias", DType::BF16, {c.num_attention_heads});
    auto fp8 = [&](const std::string& n, int64_t rows, int64_t cols, int64_t scales = 0) {
      add(n + ".weight", DType::F8_E4M3, {rows, cols});
      add(n + ".weight_scale_inv", DType::F32,
          {scales ? scales : (rows + 127) / 128, (cols + 127) / 128});
    };
    fp8("self_attn.qkv_proj", c.qkv_rows(c.mtp_layer()), h, 116);
    fp8("mlp.gate_proj", c.intermediate_size, h);
    fp8("mlp.up_proj", c.intermediate_size, h);
    fp8("mlp.down_proj", h, c.intermediate_size);
  }
  return out;
}

MimoBindReport mimo_validate_text_binding(
    const MimoTextConfig& cfg, const std::unordered_map<std::string, MimoTensorDesc>& tensors) {
  std::unordered_set<std::string> expected;
  for (const auto& t : mimo_expected_text_tensors(cfg)) {
    expected.insert(t.name);
    const auto it = tensors.find(t.name);
    if (it == tensors.end()) throw std::runtime_error("MiMo binding: missing " + t.name);
    if (it->second.dtype != t.dtype)
      throw std::runtime_error("MiMo binding: dtype mismatch " + t.name);
    if (it->second.shape != t.shape)
      throw std::runtime_error("MiMo binding: shape mismatch " + t.name);
  }
  MimoBindReport report;
  report.text = expected.size();
  for (const auto& [name, desc] : tensors) {
    if (expected.contains(name)) continue;
    if (name.starts_with("visual.") || name.starts_with("audio_encoder.") ||
        name.starts_with("speech_embeddings.")) {
      ++report.multimodal;
      continue;
    }
    if (name.starts_with("model.mtp.layers.")) {
      ++report.mtp;
      continue;
    }
    throw std::runtime_error("MiMo binding: unexpected " + name);
  }
  return report;
}
}  // namespace dgpp
