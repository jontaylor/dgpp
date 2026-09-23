#include "models/mimo/dflash.hpp"

#include <bit>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>

#include "kernels/dsv41_dspark.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/glm_spec.hpp"
#include "kernels/kernels.hpp"
#include "kernels/mimo_dflash.hpp"
#include "models/mimo/dflash_boundary.hpp"
namespace dgpp {
bool mimo_dflash_enabled() {
  const char* p = std::getenv("DGPP_MIMO_DFLASH");
  return p && std::strcmp(p, "1") == 0;
}
namespace {
size_t aligned(size_t n) {
  return (n + 255) & ~size_t(255);
}
// Read the published torch ZIP's raw BF16 storage without evaluating pickle.
// Accept exactly the released metadata schema (4096 contiguous BF16 elements,
// mask_token_id=151675). Unsupported archive changes fail closed.
std::vector<uint16_t> read_mask(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("DFlash requires learned mask_embedding.pt");
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});
  auto u16 = [&](size_t p) {
    if (p + 2 > bytes.size()) throw std::runtime_error("DFlash mask ZIP bounds");
    return uint32_t(bytes[p]) | (uint32_t(bytes[p + 1]) << 8);
  };
  auto u32 = [&](size_t p) { return u16(p) | (u16(p + 2) << 16); };
  if (bytes.size() > 65536) throw std::runtime_error("DFlash mask ZIP too large");
  std::vector<uint16_t> result;
  bool metadata = false, little = false;
  for (size_t p = 0; p + 46 <= bytes.size(); p++) {
    if (u32(p) != 0x02014b50u) continue;
    const size_t len = u16(p + 28), extra = u16(p + 30), comment = u16(p + 32), n = u32(p + 24),
                 local = u32(p + 42);
    if (p + 46 + len + extra + comment > bytes.size() || u16(p + 10) != 0 || u32(p + 20) != n ||
        u32(local) != 0x04034b50u)
      throw std::runtime_error("DFlash mask ZIP schema");
    std::string name(reinterpret_cast<const char*>(bytes.data() + p + 46), len);
    size_t off = local + 30 + u16(local + 26) + u16(local + 28);
    if (off + n > bytes.size()) throw std::runtime_error("DFlash mask storage bounds");
    if (name.ends_with("/data.pkl")) {
      uint64_t h = 1469598103934665603ull;
      for (size_t i = 0; i < n; i++) h = (h ^ bytes[off + i]) * 1099511628211ull;
      metadata = h == 0x35d13844c64acbadull;
    }
    if (name.ends_with("/byteorder"))
      little = std::string(reinterpret_cast<const char*>(bytes.data() + off), n) == "little";
    if (name.ends_with("/data/0")) {
      if (n != 8192 || !result.empty()) throw std::runtime_error("DFlash mask dimensions");
      result.resize(4096);
      std::memcpy(result.data(), bytes.data() + off, n);
    }
    p += 45 + len + extra + comment;
  }
  if (!metadata || !little || result.size() != 4096)
    throw std::runtime_error("Unsupported DFlash learned mask schema");
  return result;
}
void validate_config(const std::string& directory) {
  std::ifstream f(directory + "/config.json");
  if (!f) throw std::runtime_error("DFlash missing config.json");
  std::string json((std::istreambuf_iterator<char>(f)), {});
  // Published MiMo config has a trailing comma. Remove only commas followed
  // by whitespace and a closing container, outside JSON strings.
  bool quoted = false, escaped = false;
  std::string clean;
  for (size_t i = 0; i < json.size(); i++) {
    char ch = json[i];
    if (!quoted && ch == ',') {
      size_t j = i + 1;
      while (j < json.size() && std::isspace(static_cast<unsigned char>(json[j]))) j++;
      if (j < json.size() && (json[j] == '}' || json[j] == ']')) continue;
    }
    clean += ch;
    if (ch == '"' && !escaped) quoted = !quoted;
    escaped = quoted && ch == '\\' && !escaped;
  }
  const auto root = minijson::parse(clean).root;
  auto number = [&](const minijson::Value& r, const char* k, double v) {
    if (r.at(k).as_double() != v)
      throw std::runtime_error(std::string("Unsupported DFlash config: ") + k);
  };
  for (auto [key, val] : {std::pair{"hidden_size", 4096},
                          {"intermediate_size", 16384},
                          {"num_hidden_layers", 5},
                          {"num_attention_heads", 64},
                          {"num_key_value_heads", 8},
                          {"head_dim", 128},
                          {"block_size", 8},
                          {"sliding_window", 1024},
                          {"vocab_size", 152576}})
    number(root, key, val);
  number(root, "partial_rotary_factor", 0.5);
  number(root, "rope_theta", 10000);
  number(root, "rms_norm_eps", 1e-6);
  if (root.at("is_causal").as_bool() || root.at("attention_bias").as_bool())
    throw std::runtime_error("Unsupported causal/bias DFlash");
  auto& dc = root.at("dflash_config");
  number(dc, "attention_value_scale", 0.612);
  number(dc, "mask_token_id", 151675);
  const auto& targets = dc.at("target_layer_ids").items();
  if (targets.size() != 5) throw std::runtime_error("DFlash targets");
  for (int i = 0; i < 5; i++)
    if (targets[i].as_int() != MimoDFlash::target_layers[i])
      throw std::runtime_error("DFlash target layers");
  const auto& types = root.at("layer_types").items();
  if (types.size() != 5) throw std::runtime_error("DFlash layer types");
  for (const auto& t : types)
    if (t.as_string() != "sliding_attention")
      throw std::runtime_error("DFlash requires sliding layers");
}
}  // namespace
size_t MimoDFlash::state_bytes(int rows, int world) {
  return size_t(5) * 2 * std::bit_ceil(unsigned(1024 + rows)) * (1024 / world) * 2;
}
size_t MimoDFlash::memory_bytes(int rows, int requests, int vocab, int world) {
  // Exact resident weights, rounded tensors plus a conservative scratch plan.
  return 167796736ull + 2768318208ull / world +
         size_t(rows) * (3 * width + 4 * hidden + (8192 * 2 + 1024 * 2 + 16384 * 3) / world) * 2 +
         size_t(requests) * 8 * vocab * 4 + 3 * size_t(requests) * state_bytes(rows, world) +
         (1u << 20);
}
MimoDFlash::MimoDFlash(const std::string& dir, int rows, int requests, int vocab,
                       const uint16_t* embedding, const uint16_t* head, cudaStream_t stream,
                       int rank, int world)
    : rows_(rows),
      requests_(requests),
      vocab_(vocab),
      capacity_(std::bit_ceil(unsigned(1024 + rows))),
      world_(world),
      embedding_(embedding),
      head_(head) {
  if ((world != 1 && world != 2 && world != 4 && world != 8) || rank < 0 || rank >= world)
    throw std::invalid_argument("DFlash TP geometry");
  validate_config(dir);
  auto file = SafetensorsFile::open(dir + "/dflash_draft_model.safetensors");
  auto axis = [](const std::string& name) {
    if (name.ends_with("q_proj.weight") || name.ends_with("k_proj.weight") ||
        name.ends_with("v_proj.weight") || name.ends_with("gate_proj.weight") ||
        name.ends_with("up_proj.weight") || name.ends_with("attention_sink_bias"))
      return 0;
    if (name.ends_with("o_proj.weight") || name.ends_with("down_proj.weight")) return 1;
    return -1;
  };
  auto mask = read_mask(dir + "/mask_embedding.pt");
  size_t weight_bytes = 8192;
  file->for_each([&](const TensorInfo& t) {
    weight_bytes += aligned(t.nbytes() / (axis(t.name) < 0 ? 1 : world));
  });
  weights_.init(weight_bytes);
  auto load = [&](const std::string& name, std::vector<int64_t> shape) {
    const auto& t = file->at(name);
    if (t.dtype != DType::BF16 || t.shape != shape || t.nbytes() != t.numel() * 2)
      throw std::runtime_error("DFlash tensor shape/dtype mismatch: " + name);
    int a = axis(name);
    size_t bytes = t.nbytes() / (a < 0 ? 1 : world);
    auto* dst = static_cast<uint16_t*>(weights_.alloc(bytes));
    if (a == 1) {
      size_t cols = shape[1] / world;
      DGPP_CUDA_OK(cudaMemcpy2D(dst, cols * 2, static_cast<const uint16_t*>(t.data) + rank * cols,
                                shape[1] * 2, cols * 2, shape[0], cudaMemcpyHostToDevice));
    } else
      DGPP_CUDA_OK(cudaMemcpy(dst,
                              static_cast<const uint8_t*>(t.data) + (a == 0 ? rank * bytes : 0),
                              bytes, cudaMemcpyHostToDevice));
    return dst;
  };
  fc_ = load("fc.weight", {4096, 20480});
  hidden_norm_ = load("hidden_norm.weight", {4096});
  norm_ = load("norm.weight", {4096});
  for (int i = 0; i < 5; i++) {
    auto& l = layers_[i];
    std::string p = "layers." + std::to_string(i) + ".";
    l.input = load(p + "input_layernorm.weight", {4096});
    l.post = load(p + "post_attention_layernorm.weight", {4096});
    l.q = load(p + "self_attn.q_proj.weight", {8192, 4096});
    l.k = load(p + "self_attn.k_proj.weight", {1024, 4096});
    l.v = load(p + "self_attn.v_proj.weight", {1024, 4096});
    l.o = load(p + "self_attn.o_proj.weight", {4096, 8192});
    l.qn = load(p + "self_attn.q_norm.weight", {128});
    l.kn = load(p + "self_attn.k_norm.weight", {128});
    l.sink = load(p + "self_attn.attention_sink_bias", {64});
    l.gate = load(p + "mlp.gate_proj.weight", {16384, 4096});
    l.up = load(p + "mlp.up_proj.weight", {16384, 4096});
    l.down = load(p + "mlp.down_proj.weight", {4096, 16384});
  }
  auto* mask_device = static_cast<uint16_t*>(weights_.alloc(8192));
  DGPP_CUDA_OK(cudaMemcpy(mask_device, mask.data(), 8192, cudaMemcpyHostToDevice));
  mask_ = mask_device;
  if (weights_.cursor != weights_.capacity)
    throw std::runtime_error("DFlash unexpected checkpoint tensors");
  auto layout = [&](LayerBump& s) {
    auto bf = [&](int width) { return static_cast<uint16_t*>(s.alloc(size_t(rows) * width * 2)); };
    features_ = bf(width);
    gather_ = bf(width);
    dummy_ = bf(width);
    context_ = bf(hidden);
    x_ = bf(hidden);
    y_ = bf(hidden);
    residual_ = bf(hidden);
    q_ = bf(8192 / world);
    k_ = bf(1024 / world);
    v_ = bf(1024 / world);
    attn_ = bf(8192 / world);
    gate_ = bf(16384 / world);
    up_ = bf(16384 / world);
    act_ = bf(16384 / world);
    logits_ = static_cast<float*>(s.alloc(size_t(requests) * 8 * vocab * 4));
    pos_ = static_cast<int64_t*>(s.alloc(size_t(rows) * 8));
    tokens_ = static_cast<int64_t*>(s.alloc(size_t(rows) * 8));
    req_ = static_cast<int32_t*>(s.alloc(size_t(rows) * 4));
    spans_ = static_cast<int32_t*>(s.alloc(size_t(rows + 1) * 4));
  };
  scratch_.counting = true;
  scratch_.capacity = std::numeric_limits<size_t>::max();
  layout(scratch_);
  size_t bytes = scratch_.cursor;
  scratch_.counting = false;
  scratch_.init(bytes);
  layout(scratch_);
  DGPP_CUDA_OK(cudaMemsetAsync(dummy_, 0, size_t(rows) * width * 2, stream));
  cache_.init(size_t(requests) * state_bytes(rows, world) * 3);
  for (int i = 0; i < 5; i++) {
    keys_[i] =
        static_cast<uint16_t*>(cache_.alloc(size_t(requests) * capacity_ * (1024 / world_) * 2));
    values_[i] =
        static_cast<uint16_t*>(cache_.alloc(size_t(requests) * capacity_ * (1024 / world_) * 2));
  }
  backup_ = static_cast<uint8_t*>(cache_.alloc(size_t(requests) * state_bytes(rows, world)));
  chain_ = static_cast<uint8_t*>(cache_.alloc(size_t(requests) * state_bytes(rows, world)));
  DGPP_CUDA_OK(cudaMemsetAsync(cache_.base, 0, cache_.capacity, stream));
  gemm_.set_decode_rows(64);
}
void MimoDFlash::project(const uint16_t* x, const uint16_t* w, uint16_t* out, int n, int k,
                         int rows, cudaStream_t stream) {
  gemm_.matmul(x, w, out, rows, n, k, DType::BF16, GemmOut::BF16, k, nullptr, 0, stream);
}
void MimoDFlash::context(const uint16_t* features, const int64_t* pos, const int32_t* req, int rows,
                         cudaStream_t stream) {
  project(features, fc_, context_, 4096, 20480, rows, stream);
  glm_rmsnorm_bf16(context_, hidden_norm_, x_, rows, 4096, 1e-6f, stream);
  for (int i = 0; i < 5; i++) {
    auto& l = layers_[i];
    project(x_, l.k, k_, 1024 / world_, 4096, rows, stream);
    project(x_, l.v, v_, 1024 / world_, 4096, rows, stream);
    dflash_norm_rope(k_, l.kn, pos, rows, 8 / world_, stream);
    dflash_append(k_, v_, keys_[i], values_[i], pos, req, rows, capacity_, 8 / world_, stream);
  }
}
void MimoDFlash::propose(const int64_t* tokens, const int64_t* pos, const int32_t* req, int groups,
                         int rpg, cudaStream_t stream, bool capture,
                         BoundaryReducer* current_boundary) {
  const int n = groups * 8;
  if (n > rows_ || groups > requests_) throw std::invalid_argument("DFlash block exceeds capacity");
  dsv41_dspark_block_rows(pos, tokens, req, groups, rpg, 8, 151675, pos_, tokens_, req_, spans_,
                          stream);
  embed_gather_bf16(embedding_, tokens_, residual_, n, 4096, stream);
  dflash_mask(residual_, mask_, n, stream);
  for (int i = 0; i < 5; i++) {
    auto& l = layers_[i];
    glm_rmsnorm_bf16(residual_, l.input, x_, n, 4096, 1e-6f, stream);
    project(x_, l.q, q_, 8192 / world_, 4096, n, stream);
    project(x_, l.k, k_, 1024 / world_, 4096, n, stream);
    project(x_, l.v, v_, 1024 / world_, 4096, n, stream);
    dflash_norm_rope(q_, l.qn, pos_, n, 64 / world_, stream);
    dflash_norm_rope(k_, l.kn, pos_, n, 8 / world_, stream);
    dflash_attention(q_, k_, v_, keys_[i], values_[i], l.sink, pos_, req_, attn_, groups, capacity_,
                     64 / world_, 8 / world_, stream);
    auto fold = [&](const uint16_t* input, const uint16_t* weight, int in) {
      mimo_dflash_fold(
          current_boundary, world_, y_, n, 4096, capture,
          [&](uint16_t* partial) { project(input, weight, partial, 4096, in, n, stream); },
          [&]() { DGPP_CUDA_OK(cudaStreamSynchronize(stream)); },
          [&](uint16_t* partial) {
            add_inplace_bf16(residual_, partial, size_t(n) * 4096, stream);
          });
    };
    fold(attn_, l.o, 8192 / world_);
    glm_rmsnorm_bf16(residual_, l.post, x_, n, 4096, 1e-6f, stream);
    project(x_, l.gate, gate_, 16384 / world_, 4096, n, stream);
    project(x_, l.up, up_, 16384 / world_, 4096, n, stream);
    swiglu_limit_bf16(gate_, up_, act_, size_t(n) * (16384 / world_),
                      std::numeric_limits<float>::infinity(), stream);
    fold(act_, l.down, 16384 / world_);
  }
  glm_rmsnorm_bf16(residual_, norm_, x_, n, 4096, 1e-6f, stream);
  gemm_.matmul(x_, head_, logits_, n, vocab_, 4096, DType::BF16, GemmOut::F32, 4096, nullptr, 0,
               stream);
}
void MimoDFlash::select(float* logits, int groups, int rows_out, int index, cudaStream_t stream) {
  if (index < 0 || index >= 7) throw std::invalid_argument("DFlash index");
  dflash_select(logits_, logits, groups, rows_out, vocab_, index, stream);
}
void MimoDFlash::snapshot(int req, uint8_t* dst, cudaStream_t stream) {
  size_t plane = size_t(capacity_) * (1024 / world_) * 2;
  for (int i = 0; i < 5; i++) {
    glm_device_copy(dst, keys_[i] + size_t(req) * capacity_ * (1024 / world_), plane, stream);
    dst += plane;
    glm_device_copy(dst, values_[i] + size_t(req) * capacity_ * (1024 / world_), plane, stream);
    dst += plane;
  }
}
void MimoDFlash::restore(int req, const uint8_t* src, cudaStream_t stream) {
  size_t plane = size_t(capacity_) * (1024 / world_) * 2;
  for (int i = 0; i < 5; i++) {
    glm_device_copy(keys_[i] + size_t(req) * capacity_ * (1024 / world_), src, plane, stream);
    src += plane;
    glm_device_copy(values_[i] + size_t(req) * capacity_ * (1024 / world_), src, plane, stream);
    src += plane;
  }
}
void MimoDFlash::prepare() {
  for (int n = 1; n <= 64; n++) {
    for (auto [out, in] : {std::pair{4096, 20480},
                           {1024 / world_, 4096},
                           {8192 / world_, 4096},
                           {4096, 8192 / world_},
                           {16384 / world_, 4096},
                           {4096, 16384 / world_}})
      if (!gemm_.ensure_plan(n, out, in, DType::BF16, GemmOut::BF16, in))
        throw std::runtime_error("DFlash GEMM plan unavailable");
    if (!gemm_.ensure_plan(n, vocab_, 4096, DType::BF16, GemmOut::F32, 4096))
      throw std::runtime_error("DFlash head plan unavailable");
  }
}
}  // namespace dgpp
