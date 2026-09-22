#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"

namespace dgpp {

// Description of the released MiMo-V2.6-Flash-RL text backbone. Parsing
// this contract does not imply an executable model or a compatible drafter.
struct MimoTextConfig {
  int hidden_size = 4096;
  int vocab_size = 152576;
  int num_hidden_layers = 48;
  int num_attention_heads = 64;
  int num_key_value_heads = 4;
  int swa_num_key_value_heads = 8;
  int head_dim = 192;
  int v_head_dim = 128;
  int rotary_dim = 64;
  int sliding_window = 128;
  int intermediate_size = 16384;
  int moe_intermediate_size = 2048;
  int n_routed_experts = 256;
  int num_experts_per_tok = 8;
  int max_position_embeddings = 1048576;
  double rope_theta = 1e7;
  double swa_rope_theta = 1e4;
  double rms_norm_eps = 1e-6;
  double attention_value_scale = 0.707;
  double routed_scaling_factor = 1.0;
  std::vector<int64_t> eos_token_ids{151645};
  std::vector<int> hybrid_layer_pattern;
  std::vector<int> moe_layer_freq;

  // Native dense SWA predictors are addressed immediately after the backbone.
  static constexpr int mtp_blocks = 3;
  int mtp_layer(int block = 0) const { return num_hidden_layers + block; }
  bool is_mtp(int layer) const { return layer >= mtp_layer() && layer < mtp_layer(mtp_blocks); }
  bool sliding(int layer) const;
  bool moe(int layer) const;
  int kv_heads(int layer) const;
  int qkv_rows(int layer) const;
  // BF16 K/V storage only, with bounded SWA rings. Excludes allocator
  // rounding, prefix copies, speculative state and execution workspaces.
  uint64_t kv_bytes(int context_tokens, int world) const;
  void validate_tp(int world) const;

  static MimoTextConfig parse(const minijson::Value& root);
  static MimoTextConfig from_json_file(const std::string& path);
};

}  // namespace dgpp
