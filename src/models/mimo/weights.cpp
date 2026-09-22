#include "models/mimo/weights.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

#include "kernels/latent_format.hpp"

namespace dgpp {
namespace {
[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument("MiMo weights: " + message);
}
void rank_check(int rank, int world) {
  if (world != 1 && world != 2 && world != 4) fail("world must be 1, 2 or 4");
  if (rank < 0 || rank >= world) fail("rank outside world");
}
void range(int64_t begin, int64_t count, int64_t size, const char* axis) {
  if (begin < 0 || count <= 0 || count > size || begin > size - count)
    fail(std::string(axis) + " slice out of bounds");
}
size_t elements(int64_t rows, int64_t cols) {
  if (rows <= 0 || cols <= 0 ||
      uint64_t(rows) > uint64_t(std::numeric_limits<int64_t>::max()) / uint64_t(cols) ||
      uint64_t(rows) > std::numeric_limits<size_t>::max() / uint64_t(cols))
    fail("invalid matrix size");
  return static_cast<size_t>(rows) * static_cast<size_t>(cols);
}
void matrix(const TensorInfo& t, DType dtype) {
  if (t.dtype != dtype) fail(t.name + ": wrong dtype");
  if (t.shape.size() != 2) fail(t.name + ": expected matrix");
  const auto n = elements(t.shape[0], t.shape[1]);
  if (n > std::numeric_limits<size_t>::max() / dtype_size(dtype) || !t.data ||
      t.data_end < t.data_begin || t.nbytes() != n * dtype_size(dtype))
    fail(t.name + ": invalid payload byte count or pointer");
}
void fp8_pair(const TensorInfo& w, const TensorInfo& s, int64_t scale_rows) {
  matrix(w, DType::F8_E4M3);
  matrix(s, DType::F32);
  if (s.shape[0] != scale_rows || s.shape[1] != (w.shape[1] - 1) / 128 + 1)
    fail(s.name + ": wrong scale shape");
}
uint16_t fp8_value(const TensorInfo& w, const TensorInfo& s, int64_t row, int64_t scale_row,
                   int64_t col) {
  const auto code = static_cast<const uint8_t*>(w.data)[row * w.shape[1] + col];
  float scale;
  // Safetensors offsets need not provide native float alignment.
  std::memcpy(
      &scale,
      static_cast<const uint8_t*>(s.data) + (scale_row * s.shape[1] + col / 128) * sizeof(float),
      sizeof(float));
  return float_to_bf16_bits(fp8_e4m3_bits_to_float(code) * scale);
}
}  // namespace

MimoQkvLayout::MimoQkvLayout(const MimoTextConfig& c, int layer) {
  // These are checkpoint TP4 chunks, regardless of the requested world.
  c.validate_tp(4);
  if (c.num_attention_heads != 64 || (c.kv_heads(layer) != 4 && c.kv_heads(layer) != 8) ||
      c.head_dim != 192 || c.v_head_dim != 128)
    fail("invalid QKV geometry");
  q_rows_ = c.num_attention_heads / 4 * c.head_dim;
  k_rows_ = c.kv_heads(layer) / 4 * c.head_dim;
  v_rows_ = c.kv_heads(layer) / 4 * c.v_head_dim;
  chunk_rows_ = q_rows_ + k_rows_ + v_rows_;
}
int MimoQkvLayout::source_row(int local_row, int rank, int world) const {
  rank_check(rank, world);
  if (local_row < 0 || local_row >= rows() / world) fail("QKV row out of bounds");
  const int chunks = 4 / world;
  int part_rows = q_rows_, part_offset = 0;
  if (local_row >= chunks * q_rows_) {
    local_row -= chunks * q_rows_;
    part_rows = k_rows_;
    part_offset = q_rows_;
    if (local_row >= chunks * k_rows_) {
      local_row -= chunks * k_rows_;
      part_rows = v_rows_;
      part_offset += k_rows_;
    }
  }
  const int chunk = rank * chunks + local_row / part_rows;
  return chunk * chunk_rows_ + part_offset + local_row % part_rows;
}
int MimoQkvLayout::source_scale_row(int row) const {
  if (row < 0 || row >= rows()) fail("source QKV row out of bounds");
  return (row / chunk_rows_) * ((chunk_rows_ + 127) / 128) + (row % chunk_rows_) / 128;
}

MimoBf16Matrix mimo_load_qkv_bf16(const TensorInfo& w, const TensorInfo& s,
                                  const MimoQkvLayout& layout, int rank, int world,
                                  int64_t col_begin, int64_t cols) {
  rank_check(rank, world);
  fp8_pair(w, s, layout.scale_rows());
  if (w.shape[0] != layout.rows()) fail(w.name + ": wrong QKV row count");
  range(col_begin, cols, w.shape[1], "column");
  MimoBf16Matrix out{layout.rows() / world, cols, {}};
  out.values.resize(elements(out.rows, cols));
  for (int row = 0; row < out.rows; ++row) {
    const auto source = layout.source_row(row, rank, world);
    const auto scale = layout.source_scale_row(source);
    for (int64_t col = 0; col < cols; ++col)
      out.values[row * cols + col] = fp8_value(w, s, source, scale, col_begin + col);
  }
  return out;
}

MimoBf16Matrix mimo_load_fp8_bf16(const TensorInfo& w, const TensorInfo& s, int64_t row_begin,
                                  int64_t rows, int64_t col_begin, int64_t cols) {
  matrix(w, DType::F8_E4M3);
  fp8_pair(w, s, (w.shape[0] - 1) / 128 + 1);
  range(row_begin, rows, w.shape[0], "row");
  range(col_begin, cols, w.shape[1], "column");
  MimoBf16Matrix out{rows, cols, {}};
  out.values.resize(elements(rows, cols));
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t k = 0; k < cols; ++k)
      out.values[r * cols + k] =
          fp8_value(w, s, row_begin + r, (row_begin + r) / 128, col_begin + k);
  return out;
}

MimoBf16Matrix mimo_slice_bf16(const TensorInfo& weight, int64_t row_begin, int64_t rows,
                               int64_t col_begin, int64_t cols) {
  TensorInfo w = weight;
  if (w.shape.size() == 1) w.shape.push_back(1);
  matrix(w, DType::BF16);
  range(row_begin, rows, w.shape[0], "row");
  range(col_begin, cols, w.shape[1], "column");
  MimoBf16Matrix out{rows, cols, {}};
  out.values.resize(elements(rows, cols));
  for (int64_t r = 0; r < rows; ++r)
    std::memcpy(
        out.values.data() + r * cols,
        static_cast<const uint8_t*>(w.data) + ((row_begin + r) * w.shape[1] + col_begin) * 2,
        cols * 2);
  return out;
}

MimoMatrixSlice mimo_matrix_placement(const MimoTextConfig& c, const MimoExpectedTensor& t,
                                      int rank, int world) {
  rank_check(rank, world);
  c.validate_tp(world);
  const auto& n = t.name;
  if ((t.dtype != DType::BF16 && t.dtype != DType::F8_E4M3) || t.shape.empty() ||
      t.shape.size() > 2 || n.find(".qkv_proj.") != std::string::npos ||
      n.find(".experts.") != std::string::npos)
    fail("not an ordinary BF16/FP8 weight: " + n);
  MimoMatrixSlice out{0, t.shape[0], 0, t.shape.size() == 1 ? 1 : t.shape[1]};
  elements(out.rows, out.cols);
  if (n == "lm_head.weight" || n.ends_with(".attention_sink_bias") ||
      n.ends_with(".mlp.gate_proj.weight") || n.ends_with(".mlp.up_proj.weight")) {
    if (out.rows % world) fail("row partition not divisible");
    out.rows /= world;
    out.row_begin = rank * out.rows;
  } else if (n.ends_with(".self_attn.o_proj.weight") || n.ends_with(".mlp.down_proj.weight")) {
    if (out.cols % world) fail("column partition not divisible");
    out.cols /= world;
    out.col_begin = rank * out.cols;
  } else if (n != "model.embed_tokens.weight" && n != "model.norm.weight" &&
             !n.ends_with(".input_layernorm.weight") &&
             !n.ends_with(".post_attention_layernorm.weight") && !n.ends_with(".mlp.gate.weight") &&
             !(n.starts_with("model.mtp.layers.") &&
               (n.ends_with("norm.weight") || n.ends_with("eh_proj.weight")))) {
    fail("unknown ordinary weight role: " + n);
  }
  return out;
}

MimoMxfp4Matrix mimo_slice_mxfp4(const TensorInfo& w, const TensorInfo& s, int64_t row_begin,
                                 int64_t rows, int64_t col_begin, int64_t cols) {
  matrix(w, DType::U8);
  matrix(s, DType::U8);
  if (w.shape[1] > std::numeric_limits<int64_t>::max() / 2 || w.shape[1] % 16 ||
      s.shape[0] != w.shape[0] || s.shape[1] != w.shape[1] / 16)
    fail(s.name + ": wrong MXFP4 scale shape");
  range(row_begin, rows, w.shape[0], "row");
  range(col_begin, cols, w.shape[1] * 2, "column");
  if (col_begin % 32 || cols % 32) fail("MXFP4 column slice must align to 32 elements");
  MimoMxfp4Matrix out{rows, cols, {}, {}};
  out.payload.resize(elements(rows, cols / 2));
  out.scales.resize(elements(rows, cols / 32));
  for (int64_t r = 0; r < rows; ++r) {
    std::memcpy(out.payload.data() + r * (cols / 2),
                static_cast<const uint8_t*>(w.data) + (row_begin + r) * w.shape[1] + col_begin / 2,
                cols / 2);
    std::memcpy(out.scales.data() + r * (cols / 32),
                static_cast<const uint8_t*>(s.data) + (row_begin + r) * s.shape[1] + col_begin / 32,
                cols / 32);
  }
  return out;
}

MimoBf16Matrix mimo_dequant_mxfp4(const MimoMxfp4Matrix& m) {
  if (m.cols <= 0 || m.cols % 32 || m.payload.size() != elements(m.rows, m.cols / 2) ||
      m.scales.size() != elements(m.rows, m.cols / 32))
    fail("invalid MXFP4 buffers");
  MimoBf16Matrix out{m.rows, m.cols, {}};
  out.values.resize(elements(m.rows, m.cols));
  for (int64_t r = 0; r < m.rows; ++r)
    for (int64_t c = 0; c < m.cols; ++c) {
      const auto packed = m.payload[r * (m.cols / 2) + c / 2];
      const auto code = (packed >> (4 * (c % 2))) & 15;
      const auto scale = m.scales[r * (m.cols / 32) + c / 32];
      out.values[r * m.cols + c] =
          float_to_bf16_bits(fp4_e2m1_bits_to_float(code) * e8m0_byte_to_float(scale));
    }
  return out;
}

MimoWeightBudget mimo_weight_budget(const MimoTextConfig& c, int world) {
  c.validate_tp(world);
  const uint64_t h = c.hidden_size, v = c.vocab_size;
  MimoWeightBudget out;
  out.bf16_bytes = 2 * (v * h + (v / world) * h + h);
  for (int layer = 0; layer < c.num_hidden_layers; ++layer) {
    out.bf16_bytes += 2 * (2 * h + uint64_t(c.qkv_rows(layer) / world) * h +
                           h * (c.num_attention_heads / world) * c.v_head_dim);
    if (c.sliding(layer)) out.bf16_bytes += 2 * (c.num_attention_heads / world);
    if (c.moe(layer)) {
      out.bf16_bytes += 2 * uint64_t(c.n_routed_experts) * h;
      out.fp32_bytes += 4 * c.n_routed_experts;
      // Each of three matrices is H*I/world elements: one nibble and
      // one scale byte per 32 elements. No BF16 expert expansion.
      const uint64_t n = uint64_t(c.n_routed_experts) * 3 * h * (c.moe_intermediate_size / world);
      out.mxfp4_bytes += n / 2 + n / 32;
    } else {
      out.bf16_bytes += 2 * 3 * h * (c.intermediate_size / world);
    }
  }
  return out;
}

MimoCheckpointWeights::MimoCheckpointWeights(const std::string& directory)
    : directory_(directory),
      cfg_(MimoTextConfig::from_json_file(
          (std::filesystem::path(directory) / "config.json").string())) {
  const auto path = std::filesystem::path(directory) / "model.safetensors.index.json";
  std::ifstream f(path);
  if (!f) fail("cannot open " + path.string());
  const std::string text{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
  if (f.bad()) fail("cannot read " + path.string());
  const auto doc = minijson::parse(text);
  const auto& metadata = doc.root.at("metadata");
  const auto& tp = metadata.at("tp_size");
  if (!tp.is_number() || tp.as_double() != 4 || metadata.at("save_format").as_string() != "mxfp4")
    fail("index metadata must declare source TP4 MXFP4");
  const auto& weights = doc.root.at("weight_map");
  if (!weights.is_object()) fail("weight_map must be an object");
  for (const auto& entry : weights.members()) {
    if (!entry.value.is_string()) fail("shard name must be a string");
    const std::filesystem::path name(entry.value.as_string());
    if (name.empty() || name != name.filename() || name.extension() != ".safetensors")
      fail("invalid shard path");
    if (!index_.emplace(std::string(entry.key), name.string()).second)
      fail("duplicate tensor in index");
  }
  for (auto& t : mimo_expected_text_tensors(cfg_)) expected_.emplace(t.name, t);
  for (auto& t : mimo_expected_mtp_tensors(cfg_)) expected_.emplace(t.name, t);
}

MimoMatrixSlice MimoCheckpointWeights::placement(const std::string& name, int rank,
                                                 int world) const {
  const auto it = expected_.find(name);
  if (it == expected_.end()) fail("unknown text tensor: " + name);
  return mimo_matrix_placement(cfg_, it->second, rank, world);
}

MimoBf16Matrix MimoCheckpointWeights::bf16(const std::string& name, int rank, int world,
                                           int64_t row_begin, int64_t rows, int64_t col_begin,
                                           int64_t cols) {
  const auto local = placement(name, rank, world);
  // Check starts before subtracting to avoid overflow on hostile windows.
  if (row_begin < 0 || row_begin >= local.rows) fail("row slice out of bounds");
  if (col_begin < 0 || col_begin >= local.cols) fail("column slice out of bounds");
  if (rows == -1) rows = local.rows - row_begin;
  if (cols == -1) cols = local.cols - col_begin;
  range(row_begin, rows, local.rows, "row");
  range(col_begin, cols, local.cols, "column");
  const auto wf = shard(name);
  const auto& w = wf->at(name);
  const auto& expected = expected_.at(name);
  if (w.dtype != expected.dtype || w.shape != expected.shape)
    fail("tensor contract mismatch: " + name);
  if (w.dtype == DType::BF16)
    return mimo_slice_bf16(w, local.row_begin + row_begin, rows, local.col_begin + col_begin, cols);
  const auto scale_name = name.substr(0, name.size() - 7) + ".weight_scale_inv";
  const auto sf = shard(scale_name);
  return mimo_load_fp8_bf16(w, sf->at(scale_name), local.row_begin + row_begin, rows,
                            local.col_begin + col_begin, cols);
}

std::vector<float> MimoCheckpointWeights::router_bias(int layer) {
  if (layer < 0 || layer >= cfg_.num_hidden_layers || !cfg_.moe(layer))
    fail("layer is not a backbone MoE");
  const auto name = "model.layers." + std::to_string(layer) + ".mlp.gate.e_score_correction_bias";
  const auto f = shard(name);
  const auto& t = f->at(name);
  if (t.dtype != DType::F32 || t.shape != std::vector<int64_t>{cfg_.n_routed_experts} || !t.data ||
      t.data_end < t.data_begin || t.nbytes() != size_t(cfg_.n_routed_experts) * 4)
    fail("router bias contract mismatch");
  std::vector<float> out(cfg_.n_routed_experts);
  std::memcpy(out.data(), t.data, out.size() * sizeof(float));
  return out;
}

std::shared_ptr<SafetensorsFile> MimoCheckpointWeights::shard(const std::string& tensor) {
  const auto it = index_.find(tensor);
  if (it == index_.end()) fail("index missing " + tensor);
  if (!cached_ || cached_name_ != it->second) {
    auto opened = SafetensorsFile::open((std::filesystem::path(directory_) / it->second).string());
    cached_ = std::move(opened);
    cached_name_ = it->second;
  }
  return cached_;
}

MimoBf16Matrix MimoCheckpointWeights::qkv(int layer, int rank, int world, int64_t col_begin,
                                          int64_t cols) {
  rank_check(rank, world);
  if (layer < 0 || layer >= cfg_.mtp_layer(cfg_.mtp_blocks)) fail("layer outside backbone/draft");
  const std::string base =
      cfg_.is_mtp(layer)
          ? "model.mtp.layers." + std::to_string(layer - cfg_.mtp_layer()) + ".self_attn.qkv_proj"
          : "model.layers." + std::to_string(layer) + ".self_attn.qkv_proj";
  const auto wf = shard(base + ".weight"), sf = shard(base + ".weight_scale_inv");
  const auto& w = wf->at(base + ".weight");
  if (w.shape.size() != 2 || w.shape[1] != cfg_.hidden_size) fail("QKV hidden dimension mismatch");
  return mimo_load_qkv_bf16(w, sf->at(base + ".weight_scale_inv"), MimoQkvLayout(cfg_, layer), rank,
                            world, col_begin, cols);
}

MimoMxfp4Matrix MimoCheckpointWeights::expert(int layer, int expert, const std::string& projection,
                                              int rank, int world) {
  rank_check(rank, world);
  if (layer < 0 || layer >= cfg_.num_hidden_layers || !cfg_.moe(layer))
    fail("layer is not a backbone MoE");
  if (expert < 0 || expert >= cfg_.n_routed_experts) fail("expert outside router");
  const bool down = projection == "down_proj";
  if (!down && projection != "gate_proj" && projection != "up_proj")
    fail("unknown expert projection");
  const std::string base = "model.layers." + std::to_string(layer) + ".mlp.experts." +
                           std::to_string(expert) + "." + projection;
  const auto wf = shard(base + ".weight"), sf = shard(base + ".weight_scale");
  const auto& w = wf->at(base + ".weight");
  const int64_t rows = down ? cfg_.hidden_size : cfg_.moe_intermediate_size;
  const int64_t cols = down ? cfg_.moe_intermediate_size : cfg_.hidden_size;
  if (w.shape != std::vector<int64_t>{rows, cols / 2}) fail("expert payload shape mismatch");
  return mimo_slice_mxfp4(w, sf->at(base + ".weight_scale"), down ? 0 : rank * (rows / world),
                          down ? rows : rows / world, down ? rank * (cols / world) : 0,
                          down ? cols / world : cols);
}
}  // namespace dgpp
