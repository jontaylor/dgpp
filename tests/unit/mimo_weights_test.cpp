#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "common/test.hpp"
#include "models/mimo/binding.hpp"
#include "models/mimo/weights.hpp"

namespace {
void check(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}
template <class F>
void rejects(F fn, const std::string& part) {
  try {
    fn();
  } catch (const std::invalid_argument& e) {
    check(std::string(e.what()).find(part) != std::string::npos, e.what());
    return;
  }
  throw std::runtime_error("accepted invalid " + part);
}
std::string fixture(const char* name) {
  std::ifstream f(std::string(DGPP_SOURCE_DIR) + "/tests/data/mimo/" + name);
  check(bool(f), "fixture missing");
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
dgpp::MimoTextConfig config() {
  return dgpp::MimoTextConfig::from_json_file(std::string(DGPP_SOURCE_DIR) +
                                              "/tests/data/mimo/config.json");
}
template <class T>
dgpp::TensorInfo tensor(const std::vector<T>& bytes, dgpp::DType dtype, int64_t rows,
                        int64_t cols) {
  dgpp::TensorInfo out;
  out.name = "fixture";
  out.dtype = dtype;
  out.shape = {rows, cols};
  out.data = bytes.data();
  out.data_end = bytes.size() * sizeof(T);
  return out;
}
}  // namespace

DGPP_TEST(mimo_qkv_source_mapping_is_bijective_for_all_worlds) {
  const auto c = config();
  for (int layer : {0, 1}) {
    const dgpp::MimoQkvLayout layout(c, layer);
    for (int world : {1, 2, 4}) {
      std::unordered_set<int> seen;
      for (int rank = 0; rank < world; ++rank) {
        // Independent semantic ordering: each rank owns consecutive heads
        // from each of Q, K and V, not consecutive checkpoint rows.
        int dest = 0, offset = 0;
        const int heads[] = {64, c.kv_heads(layer), c.kv_heads(layer)};
        const int dims[] = {192, 192, 128};
        const int chunk_rows = layout.rows() / 4;
        for (int kind = 0; kind < 3; ++kind) {
          for (int h = rank * heads[kind] / world; h < (rank + 1) * heads[kind] / world; ++h)
            for (int d = 0; d < dims[kind]; ++d) {
              const int chunk = h / (heads[kind] / 4);
              const int row =
                  chunk * chunk_rows + offset + (h % (heads[kind] / 4)) * dims[kind] + d;
              check(layout.source_row(dest++, rank, world) == row, "head order");
              check(seen.insert(row).second, "duplicate row across ranks");
            }
          offset += heads[kind] / 4 * dims[kind];
        }
      }
      check(seen.size() == static_cast<size_t>(layout.rows()), "all checkpoint rows covered");
    }
  }
  const dgpp::MimoQkvLayout global(c, 0);
  check(global.source_scale_row(3391) == 26 && global.source_scale_row(3392) == 27,
        "scale grid restarts at a non-128-aligned chunk boundary");
  rejects([&] { global.source_row(0, 2, 2); }, "rank");
  rejects([&] { global.source_row(0, 0, 3); }, "world");
  rejects([&] { global.source_row(global.rows(), 0, 1); }, "row");
  rejects([&] { global.source_scale_row(-1); }, "row");
}

DGPP_TEST(mimo_qkv_matches_real_checkpoint_bf16_samples) {
  const auto text = fixture("weight_samples.json");
  const auto doc = dgpp::minijson::parse(text);
  for (int layer : {0, 1}) {
    const dgpp::MimoQkvLayout layout(config(), layer);
    // Unobserved entries are zero; only samples with independent goldens
    // are asserted. No claim of full-matrix numerical parity follows.
    std::vector<uint8_t> payload(static_cast<size_t>(layout.rows()) * 4096);
    std::vector<float> scales(layout.scale_rows() * 32);
    for (const auto& sample : doc.root.at("qkv").items()) {
      if (sample.at("layer").as_int() != layer) continue;
      const auto row = sample.at("source_row").as_int();
      const auto sr = sample.at("scale_row").as_int();
      const auto col = sample.at("col_begin").as_int();
      check(layout.source_scale_row(row) == sr, "reference scale row");
      for (size_t j = 0; j < sample.at("payload").items().size(); ++j)
        payload[row * 4096 + col + j] = sample.at("payload").items()[j].as_int();
      for (size_t j = 0; j < sample.at("scales").items().size(); ++j)
        scales[sr * 32 + j] = static_cast<float>(sample.at("scales").items()[j].as_double());
    }
    const auto w = tensor(payload, dgpp::DType::F8_E4M3, layout.rows(), 4096);
    const auto s = tensor(scales, dgpp::DType::F32, layout.scale_rows(), 32);
    for (int world : {1, 2, 4}) {
      size_t matched = 0;
      for (int rank = 0; rank < world; ++rank) {
        const auto result = dgpp::mimo_load_qkv_bf16(w, s, layout, rank, world, 120, 16);
        for (const auto& sample : doc.root.at("qkv").items()) {
          if (sample.at("layer").as_int() != layer) continue;
          const auto kind = sample.at("kind").as_string();
          const int heads = kind == "q" ? 64 : config().kv_heads(layer);
          const int dim = kind == "v" ? 128 : 192;
          const int head = sample.at("head").as_int();
          if (head < rank * heads / world || head >= (rank + 1) * heads / world) continue;
          const int offset = kind == "q" ? 0 : (64 * 192 + (kind == "v" ? heads * 192 : 0)) / world;
          const int row =
              offset + (head - rank * heads / world) * dim + sample.at("element").as_int();
          check(layout.source_row(row, rank, world) == sample.at("source_row").as_int(),
                "sample source");
          for (int j = 0; j < 16; ++j)
            check(result.values[row * 16 + j] == sample.at("bf16").items()[j].as_int(),
                  "QKV BF16 parity");
          ++matched;
        }
      }
      const size_t expected =
          std::count_if(doc.root.at("qkv").items().begin(), doc.root.at("qkv").items().end(),
                        [&](const auto& sample) { return sample.at("layer").as_int() == layer; });
      check(matched == expected, "every reference sample checked exactly once per world");
    }
    auto bad = s;
    bad.shape[0] = 106;
    rejects([&] { dgpp::mimo_load_qkv_bf16(w, bad, layout, 0, 2, 0, 16); }, "payload");
    rejects([&] { dgpp::mimo_load_qkv_bf16(w, s, layout, 0, 2, 4090, 16); }, "column");
  }
}

DGPP_TEST(mimo_mxfp4_slices_match_real_checkpoint_samples) {
  const auto text = fixture("weight_samples.json");
  const auto doc = dgpp::minijson::parse(text);
  for (const auto& sample : doc.root.at("experts").items()) {
    const auto rows = sample.at("rows").as_int(), cols = sample.at("cols").as_int();
    const auto row = sample.at("row").as_int(), col = sample.at("col_begin").as_int();
    std::vector<uint8_t> payload(rows * cols / 2), scales(rows * cols / 32);
    for (int j = 0; j < 32; ++j)
      payload[row * cols / 2 + col / 2 + j] = sample.at("payload").items()[j].as_int();
    for (int j = 0; j < 2; ++j)
      scales[row * cols / 32 + col / 32 + j] = sample.at("scales").items()[j].as_int();
    const auto w = tensor(payload, dgpp::DType::U8, rows, cols / 2);
    const auto s = tensor(scales, dgpp::DType::U8, rows, cols / 32);
    const auto packed = dgpp::mimo_slice_mxfp4(w, s, row, 1, col, 64);
    const auto result = dgpp::mimo_dequant_mxfp4(packed);
    for (int j = 0; j < 64; ++j)
      check(result.values[j] == sample.at("bf16").items()[j].as_int(),
            "MXFP4 BF16 parity at column " + std::to_string(j) +
                " actual=" + std::to_string(result.values[j]) +
                " expected=" + std::to_string(sample.at("bf16").items()[j].as_int()));
    rejects([&] { dgpp::mimo_slice_mxfp4(w, s, row, 1, col + 1, 64); }, "align");
    rejects([&] { dgpp::mimo_slice_mxfp4(w, s, rows, 1, col, 64); }, "row");
    auto truncated = packed;
    truncated.scales.pop_back();
    rejects([&] { dgpp::mimo_dequant_mxfp4(truncated); }, "buffers");
  }
}

DGPP_TEST(mimo_dense_fp8_slices_cross_both_scale_axes) {
  std::vector<uint8_t> payload(130 * 130, 0x38);  // FP8 1.0
  std::vector<float> scales{1, 2, 4, 8};
  const auto w = tensor(payload, dgpp::DType::F8_E4M3, 130, 130);
  const auto s = tensor(scales, dgpp::DType::F32, 2, 2);
  const auto out = dgpp::mimo_load_fp8_bf16(w, s, 127, 2, 127, 2);
  check(out.values == std::vector<uint16_t>({0x3f80, 0x4000, 0x4080, 0x4100}),
        "FP8 2D scale blocks");
  rejects([&] { dgpp::mimo_load_fp8_bf16(w, s, 0, -1, 0, 1); }, "row");
  rejects([&] { dgpp::mimo_load_fp8_bf16(w, s, 0, 1, std::numeric_limits<int64_t>::max(), 1); },
          "column");
  auto bad = w;
  bad.data = nullptr;
  rejects([&] { dgpp::mimo_load_fp8_bf16(bad, s, 0, 1, 0, 1); }, "pointer");
  bad = w;
  bad.dtype = dgpp::DType::BF16;
  rejects([&] { dgpp::mimo_load_fp8_bf16(bad, s, 0, 1, 0, 1); }, "dtype");
}

DGPP_TEST(mimo_initial_weight_budget_matches_tensor_placement) {
  const auto c = config();
  for (int world : {1, 2, 4}) {
    uint64_t expected = 0;
    for (const auto& t : dgpp::mimo_expected_text_tensors(c)) {
      // Derive independently from the checkpoint manifest. The first
      // resident plan widens FP8 payloads to BF16 and drops their scales.
      if (t.name.ends_with("weight_scale_inv")) continue;
      uint64_t count = 1;
      for (auto d : t.shape) count *= d;
      uint64_t bytes = count * (t.dtype == dgpp::DType::F8_E4M3 ? 2 : dgpp::dtype_size(t.dtype));
      const bool replicated = t.name == "model.embed_tokens.weight" ||
                              t.name == "model.norm.weight" ||
                              t.name.find("layernorm.weight") != std::string::npos ||
                              t.name.find(".mlp.gate.") != std::string::npos;
      expected += replicated ? bytes : bytes / world;
    }
    check(dgpp::mimo_weight_budget(c, world).total() == expected,
          "weight bytes vs tensor placement");
  }
}

DGPP_TEST(mimo_checkpoint_reader_owns_slices_across_shard_changes) {
  struct Temp {
    std::filesystem::path path;
    Temp() {
      auto pattern = (std::filesystem::temp_directory_path() / "dgpp-mimo-weights-XXXXXX").string();
      std::vector<char> name(pattern.begin(), pattern.end());
      name.push_back(0);
      const auto p = mkdtemp(name.data());
      if (!p) throw std::runtime_error("mkdtemp failed");
      path = p;
    }
    ~Temp() { std::filesystem::remove_all(path); }
  } temp;
  std::ofstream(temp.path / "config.json") << fixture("config.json");
  std::string entries;
  // Sparse synthetic shards preserve the real geometry but allocate only
  // header/scale pages and the marked payload bytes on disk.
  auto write = [&](const std::string& file, const std::string& name, const std::string& dtype,
                   int64_t rows, int64_t cols, int bytes, const std::vector<uint8_t>& initial,
                   uint64_t marker, uint8_t value) {
    if (!entries.empty()) entries += ',';
    entries += "\"" + name + "\":\"" + file + "\"";
    const uint64_t count = rows * (cols ? cols : 1) * bytes;
    std::string header = "{\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[" +
                         std::to_string(rows) + (cols ? "," + std::to_string(cols) : "") +
                         "],\"data_offsets\":[0," + std::to_string(count) + "]}}";
    while (header.size() % 8) header += ' ';
    const uint64_t length = header.size();
    std::ofstream f(temp.path / file, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&length), 8);
    f << header;
    f.seekp(8 + length + count - 1);
    f.put(0);
    f.seekp(8 + length);
    if (!initial.empty()) f.write(reinterpret_cast<const char*>(initial.data()), initial.size());
    if (marker < count) {
      f.seekp(8 + length + marker);
      f.put(static_cast<char>(value));
    }
    check(bool(f), "write sparse shard");
  };
  const std::string q = "model.layers.0.self_attn.qkv_proj";
  std::vector<float> ones(108 * 32, 1.0f);
  std::vector<uint8_t> scale_bytes(ones.size() * 4);
  std::memcpy(scale_bytes.data(), ones.data(), scale_bytes.size());
  // Rank 1 at TP2 starts at checkpoint chunk 2 (row 6784).
  write("q.safetensors", q + ".weight", "F8_E4M3", 13568, 4096, 1, {}, 6784ull * 4096, 0x38);
  write("s.safetensors", q + ".weight_scale_inv", "F32", 108, 32, 4, scale_bytes,
        scale_bytes.size(), 0);
  const std::string e = "model.layers.1.mlp.experts.0.";
  for (const auto& projection : {std::string("gate_proj"), std::string("down_proj")}) {
    const bool down = projection == "down_proj";
    const int rows = down ? 4096 : 2048, cols = down ? 2048 : 4096;
    write(projection + ".safetensors", e + projection + ".weight", "U8", rows, cols / 2, 1, {},
          down ? 512 : 1024ull * 2048, down ? 0x54 : 0x32);
    const std::vector<uint8_t> scales(rows * cols / 32, 127);
    write(projection + "_scale.safetensors", e + projection + ".weight_scale", "U8", rows,
          cols / 32, 1, scales, scales.size() - 1, 127);
  }
  const std::string o = "model.layers.0.self_attn.o_proj.weight";
  write("o.safetensors", o, "BF16", 4096, 8192, 2, {}, 4096 * 2 + 1, 0x40);
  const std::string norm = "model.layers.0.input_layernorm.weight";
  write("norm.safetensors", norm, "BF16", 4096, 0, 2, {0x80, 0x3f}, 8192, 0);
  const std::string sink = "model.layers.1.self_attn.attention_sink_bias";
  write("sink.safetensors", sink, "BF16", 64, 0, 2, {}, 32 * 2 + 1, 0x40);
  const std::string dense = "model.layers.0.mlp.down_proj";
  write("dense.safetensors", dense + ".weight", "F8_E4M3", 4096, 16384, 1, {}, 8192, 0x38);
  std::vector<float> dense_scales(32 * 128, 3.0f);
  std::vector<uint8_t> dense_scale_bytes(dense_scales.size() * 4);
  std::memcpy(dense_scale_bytes.data(), dense_scales.data(), dense_scale_bytes.size());
  write("ds.safetensors", dense + ".weight_scale_inv", "F32", 32, 128, 4, dense_scale_bytes,
        dense_scale_bytes.size(), 0);
  const std::string bias = "model.layers.1.mlp.gate.e_score_correction_bias";
  const float correction = -0.25f;
  std::vector<uint8_t> bias_bytes(4);
  std::memcpy(bias_bytes.data(), &correction, 4);
  write("bias.safetensors", bias, "F32", 256, 0, 4, bias_bytes, 1024, 0);
  std::ofstream(temp.path / "model.safetensors.index.json")
      << "{\"metadata\":{\"tp_size\":4,\"save_format\":\"mxfp4\"},\"weight_map\":{" << entries
      << "}}";
  dgpp::MimoCheckpointWeights reader(temp.path.string());
  check(reader.bf16(o, 1, 2, 0, 1, 0, 1).values[0] == 0x4000, "output projection column partition");
  check(reader.bf16(norm, 1, 2, 0, 1).values[0] == 0x3f80, "norm replication");
  check(reader.bf16(sink, 1, 2, 0, 1).values[0] == 0x4000, "sink head partition");
  check(reader.bf16(dense + ".weight", 1, 2, 0, 1, 0, 1).values[0] == 0x4040,
        "dense column partition and separate scale shard");
  check(reader.router_bias(1)[0] == -0.25f, "FP32 router correction");
  rejects([&] { reader.bf16(o, 0, 2, 0, 1, 4096, 1); }, "column");
  rejects([&] { reader.bf16(o, 0, 2, -1, 1); }, "row");
  rejects([&] { reader.bf16(q + ".weight", 0, 2); }, "ordinary");
  rejects([&] { reader.bf16(bias, 0, 2); }, "ordinary");
  rejects([&] { reader.bf16("visual.unknown", 0, 2); }, "unknown");
  rejects([&] { reader.router_bias(0); }, "MoE");
  const auto qkv = reader.qkv(0, 1, 2, 0, 1);
  check(qkv.rows == 6784 && qkv.values[0] == 0x3f80, "reader QKV mapping and F32 scales");
  const auto gate = reader.expert(1, 0, "gate_proj", 1, 2);
  check(gate.rows == 1024 && gate.cols == 4096 && gate.payload[0] == 0x32 && gate.scales[0] == 127,
        "expert intermediate row sharding");
  const auto down = reader.expert(1, 0, "down_proj", 1, 2);
  check(down.rows == 4096 && down.cols == 1024 && down.payload[0] == 0x54 && down.scales[0] == 127,
        "expert intermediate column sharding");
  check(qkv.values[0] == 0x3f80 && gate.payload[0] == 0x32,
        "returned buffers survive mmap eviction");
  rejects([&] { reader.qkv(-1, 0, 2, 0, 1); }, "layer");
  rejects([&] { reader.expert(0, 0, "gate_proj", 0, 2); }, "MoE");
  rejects([&] { reader.expert(1, 256, "gate_proj", 0, 2); }, "expert");
  rejects([&] { reader.expert(1, 0, "other", 0, 2); }, "projection");
  rejects([&] { reader.expert(1, 1, "up_proj", 0, 2); }, "index missing");
}

DGPP_TEST(mimo_ordinary_placements_cover_partitions_and_replicas) {
  const auto cfg = config();
  for (const auto& t : dgpp::mimo_expected_text_tensors(cfg)) {
    if (t.name.find(".experts.") != std::string::npos ||
        t.name.find(".qkv_proj.") != std::string::npos ||
        (t.dtype != dgpp::DType::BF16 && t.dtype != dgpp::DType::F8_E4M3))
      continue;
    const bool row = t.name == "lm_head.weight" || t.name.ends_with("attention_sink_bias") ||
                     t.name.ends_with("mlp.gate_proj.weight") ||
                     t.name.ends_with("mlp.up_proj.weight");
    const bool col = t.name.ends_with("o_proj.weight") || t.name.ends_with("mlp.down_proj.weight");
    for (int world : {1, 2, 4}) {
      int64_t area = 0;
      for (int rank = 0; rank < world; ++rank) {
        const auto p = dgpp::mimo_matrix_placement(cfg, t, rank, world);
        check(p.row_begin == (row ? rank * p.rows : 0), "row origin");
        check(p.col_begin == (col ? rank * p.cols : 0), "column origin");
        area += p.rows * p.cols;
      }
      const int64_t full = t.shape[0] * (t.shape.size() == 1 ? 1 : t.shape[1]);
      check(area == full * (row || col ? 1 : world), "complete placement coverage");
    }
  }
}

DGPP_TEST(mimo_bf16_slice_preserves_bits_and_rejects_malformed_payloads) {
  // Deliberately unaligned source containing signed zero, NaN and infinity.
  const std::vector<uint16_t> bits{1, 0x8000, 0x7fc1, 2, 0x7f80, 0xff80};
  std::vector<uint8_t> raw(1 + bits.size() * 2);
  std::memcpy(raw.data() + 1, bits.data(), bits.size() * 2);
  auto t = tensor(bits, dgpp::DType::BF16, 2, 3);
  t.data = raw.data() + 1;
  check(dgpp::mimo_slice_bf16(t, 0, 2, 1, 2).values ==
            std::vector<uint16_t>({0x8000, 0x7fc1, 0x7f80, 0xff80}),
        "bit-preserving strided copy");
  t.data_end -= 1;
  rejects([&] { dgpp::mimo_slice_bf16(t, 0, 1, 0, 1); }, "payload");
}

DGPP_TEST(mimo_native_mtp_contract_and_tp_placement) {
  const auto c = config();
  const auto tensors = dgpp::mimo_expected_mtp_tensors(c);
  check(tensors.size() == 48, "three native blocks each have 16 tensors");
  check(c.sliding(c.mtp_layer()) && !c.moe(c.mtp_layer()), "draft is dense SWA");
  check(c.qkv_rows(c.mtp_layer()) == 14848, "draft QKV geometry");
  for (int world : {1, 2, 4}) {
    for (const auto& t : tensors) {
      if (t.name.find("scale_inv") != std::string::npos ||
          t.name.find("qkv_proj") != std::string::npos) continue;
      for (int rank = 0; rank < world; ++rank) {
        const auto slice = dgpp::mimo_matrix_placement(c, t, rank, world);
        if (t.name.ends_with("eh_proj.weight"))
          check(slice.rows == 4096 && slice.cols == 8192 && !slice.row_begin && !slice.col_begin,
                "fusion must be replicated, embedding half first");
        if (t.name.ends_with("mlp.down_proj.weight"))
          check(slice.cols == 16384 / world && slice.col_begin == rank * slice.cols,
                "draft dense output is column-sharded");
      }
    }
  }
}

DGPP_TEST(mimo_fp8_resident_preserves_qkv_chunk_scales_and_rank_slices) {
  const auto cfg = config();
  for (int layer : {0, 1}) {
    dgpp::MimoQkvLayout layout(cfg, layer);
    constexpr int cols = 256;
    std::vector<uint8_t> codes(size_t(layout.rows()) * cols);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = uint8_t(1 + i % 120);
    std::vector<float> scales(size_t(layout.scale_rows()) * 2);
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = float(i + 1) * .001f;
    auto w = tensor(codes, dgpp::DType::F8_E4M3, layout.rows(), cols);
    auto s = tensor(scales, dgpp::DType::F32, layout.scale_rows(), 2);
    for (int world : {1, 2, 4})
      for (int rank = 0; rank < world; ++rank) {
        auto p = dgpp::mimo_load_fp8(w, s, 0, layout.rows() / world, 64, 128, &layout, rank, world);
        auto ref = dgpp::mimo_load_qkv_bf16(w, s, layout, rank, world, 64, 128);
        for (int64_t r = 0; r < p.rows; ++r)
          for (int64_t c = 0; c < p.cols; ++c)
            check(dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(p.payload[r * p.cols + c]) *
                                           p.scales[(r / 64) * (p.cols / 64) + c / 64]) ==
                      ref.values[r * p.cols + c],
                  "resident QKV mismatch");
      }
    rejects([&] { dgpp::mimo_load_fp8(w, s, 0, layout.rows(), 1, 128, &layout); }, "align");
  }
}

DGPP_TEST(mimo_fp8_resident_preserves_ordinary_tp_column_offsets) {
  constexpr int rows = 256, cols = 384;
  std::vector<uint8_t> codes(rows * cols);
  for (size_t i = 0; i < codes.size(); ++i) codes[i] = uint8_t(1 + i % 120);
  const std::vector<float> scales{.01f, .02f, .03f, .04f, .05f, .06f};
  const auto w = tensor(codes, dgpp::DType::F8_E4M3, rows, cols);
  const auto s = tensor(scales, dgpp::DType::F32, 2, 3);
  for (int rb : {0, 64, 128})
    for (int cb : {0, 64, 192}) {
      auto p = dgpp::mimo_load_fp8(w, s, rb, 128, cb, 192);
      auto ref = dgpp::mimo_load_fp8_bf16(w, s, rb, 128, cb, 192);
      for (int64_t r = 0; r < p.rows; ++r)
        for (int64_t c = 0; c < p.cols; ++c)
          check(dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(p.payload[r * p.cols + c]) *
                                         p.scales[(r / 64) * (p.cols / 64) + c / 64]) ==
                    ref.values[r * p.cols + c],
                "resident ordinary mismatch");
    }
}

DGPP_TEST(mimo_fp8_prefill_bridge_excludes_decode_and_every_capture) {
  for (bool enabled : {false, true})
    for (int rows : {1, 4, 8, 32, 64, 65, 128, 512})
      for (bool capture : {false, true})
        for (bool decode : {false, true})
          for (int end_key : {0, 1024}) {
            const bool actual =
                dgpp::mimo_fp8_dense_use_bridge(enabled, rows, capture, decode, end_key);
            check(actual == (enabled && rows >= 65 && !capture && !decode && end_key == 1024),
                  "incorrect prefill bridge execution kind");
          }
}
