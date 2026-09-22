// CPU-only sampled parity check through the real checkpoint reader. This
// checks host unpacking at each TP geometry, not distributed GPU execution.
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "models/mimo/weights.hpp"

int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4)
      throw std::runtime_error(
          "usage: mimo_weight_check SNAPSHOT_DIR GOLDEN_JSON [LAYER_GOLDEN_JSON]");
    std::ifstream f(argv[2]);
    if (!f) throw std::runtime_error("cannot open goldens");
    const std::string text{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    if (f.bad()) throw std::runtime_error("cannot read goldens");
    const auto doc = dgpp::minijson::parse(text);
    const auto& qkv_samples = doc.root.at("qkv").items();
    const auto& expert_samples = doc.root.at("experts").items();
    if (qkv_samples.empty() || expert_samples.empty()) throw std::runtime_error("empty goldens");
    dgpp::MimoCheckpointWeights reader(argv[1]);
    const auto& cfg = reader.config();
    for (int world : {1, 2, 4}) {
      size_t matched_qkv = 0, matched_experts = 0;
      for (int rank = 0; rank < world; ++rank) {
        for (int layer : {0, 1}) {
          const auto qkv = reader.qkv(layer, rank, world, 120, 16);
          for (const auto& sample : qkv_samples) {
            if (sample.at("layer").as_int() != layer) continue;
            const auto kind = sample.at("kind").as_string();
            if (kind != "q" && kind != "k" && kind != "v")
              throw std::runtime_error("unknown QKV kind");
            const int heads = kind == "q" ? 64 : cfg.kv_heads(layer);
            const int dim = kind == "v" ? 128 : 192;
            const int head = sample.at("head").as_int();
            if (head < rank * heads / world || head >= (rank + 1) * heads / world) continue;
            const int offset =
                kind == "q" ? 0 : (64 * 192 + (kind == "v" ? heads * 192 : 0)) / world;
            const int row =
                offset + (head - rank * heads / world) * dim + sample.at("element").as_int();
            if (sample.at("col_begin").as_int() != 120 || sample.at("bf16").items().size() != 16)
              throw std::runtime_error("unsupported golden column range");
            for (int j = 0; j < 16; ++j)
              if (qkv.values.at(static_cast<size_t>(row) * 16 + j) !=
                  sample.at("bf16").items()[j].as_int())
                throw std::runtime_error("real QKV sample mismatch");
            ++matched_qkv;
          }
        }
        for (const auto& projection : {std::string("gate_proj"), std::string("down_proj")}) {
          const auto packed = reader.expert(1, 0, projection, rank, world);
          const auto dense = dgpp::mimo_dequant_mxfp4(packed);
          for (const auto& sample : expert_samples) {
            if (sample.at("projection").as_string() != projection) continue;
            const int64_t row_start = projection == "down_proj" ? 0 : rank * dense.rows;
            const int64_t col_start = projection == "down_proj" ? rank * dense.cols : 0;
            const int64_t row = sample.at("row").as_int() - row_start;
            const int64_t col = sample.at("col_begin").as_int() - col_start;
            const auto& expected = sample.at("bf16").items();
            if (row < 0 || row >= dense.rows || col < 0 || col >= dense.cols) continue;
            if (expected.empty() || expected.size() > static_cast<size_t>(dense.cols - col))
              throw std::runtime_error("golden crosses rank boundary");
            for (size_t j = 0; j < expected.size(); ++j)
              if (dense.values.at(row * dense.cols + col + j) != expected[j].as_int())
                throw std::runtime_error("real MXFP4 sample mismatch");
            ++matched_experts;
          }
        }
      }
      if (matched_qkv != qkv_samples.size() || matched_experts != expert_samples.size())
        throw std::runtime_error("not all golden samples checked");
      std::cout << "Host TP" << world << " unpack parity: " << matched_qkv << " QKV rows, "
                << matched_experts << " expert slices passed\n";
    }
    if (argc == 4) {
      std::ifstream layer_file(argv[3]);
      if (!layer_file) throw std::runtime_error("cannot open layer goldens");
      const std::string layer_text{std::istreambuf_iterator<char>(layer_file),
                                   std::istreambuf_iterator<char>()};
      if (layer_file.bad()) throw std::runtime_error("cannot read layer goldens");
      const auto layer_doc = dgpp::minijson::parse(layer_text);
      const auto& samples = layer_doc.root.at("samples").items();
      if (samples.empty()) throw std::runtime_error("empty layer goldens");
      for (int world : {1, 2, 4}) {
        size_t comparisons = 0;
        for (const auto& sample : samples) {
          const std::string name(sample.at("name").as_string());
          const auto row = sample.at("row").as_int(), col = sample.at("col").as_int();
          const auto& values = sample.at("values").items();
          if (values.empty()) throw std::runtime_error("empty sample");
          if (sample.at("dtype").as_string() == "F32") {
            if (name != "model.layers.1.mlp.gate.e_score_correction_bias" || col != 0 ||
                values.size() != 1)
              throw std::runtime_error("unexpected router bias sample");
            if (reader.router_bias(1).at(row) != static_cast<float>(values[0].as_double()))
              throw std::runtime_error("router bias mismatch");
            ++comparisons;
            continue;
          }
          int matches = 0;
          for (int rank = 0; rank < world; ++rank) {
            const auto p = reader.placement(name, rank, world);
            if (row < p.row_begin || row >= p.row_begin + p.rows || col < p.col_begin ||
                col >= p.col_begin + p.cols)
              continue;
            const auto got = reader.bf16(name, rank, world, row - p.row_begin, 1, col - p.col_begin,
                                         values.size());
            for (size_t j = 0; j < values.size(); ++j)
              if (got.values[j] != values[j].as_int())
                throw std::runtime_error("ordinary weight mismatch: " + name);
            ++matches;
            ++comparisons;
          }
          const int expected = sample.at("axis").as_string() == "replicated" ? world : 1;
          if (matches != expected)
            throw std::runtime_error("sample placement multiplicity mismatch");
        }
        std::cout << "Host TP" << world << " ordinary weights: " << samples.size() << " samples, "
                  << comparisons << " rank comparisons passed\n";
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
