#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "loaders/safetensors.hpp"
#include "models/mimo/binding.hpp"
#include "models/mimo/config.hpp"

namespace dgpp {
inline constexpr int kMimoFp8DenseBridgeMinRows = 512;
// Explicit execution-kind guard: CUDA graph capture and decode never bridge.
inline bool mimo_fp8_dense_use_bridge(bool enabled, int rows, bool capture, bool has_request_ids,
                                      int end_key) {
  return enabled && rows >= kMimoFp8DenseBridgeMinRows && !capture && !has_request_ids && end_key > 0;
}

// Source checkpoint chunks are [Q0,K0,V0,Q1,K1,V1,...] at TP4.
// A serving rank consumes [its Q heads, its K heads, its V heads].
class MimoQkvLayout {
 public:
  explicit MimoQkvLayout(const MimoTextConfig& cfg, int layer);
  int rows() const { return 4 * chunk_rows_; }
  int scale_rows() const { return 4 * ((chunk_rows_ + 127) / 128); }
  int source_row(int local_row, int rank, int world) const;
  int source_scale_row(int source_row) const;

 private:
  int q_rows_, k_rows_, v_rows_, chunk_rows_;
};

struct MimoBf16Matrix {
  int64_t rows = 0, cols = 0;
  std::vector<uint16_t> values;
};
// Exact source E4M3 codes, reblocked scales on 64x64 boundaries. QKV
// source chunks restart their 128-row scale grid, including partial blocks.
struct MimoFp8Matrix {
  int64_t rows = 0, cols = 0;
  std::vector<uint8_t> payload;
  std::vector<float> scales;
};
MimoFp8Matrix mimo_load_fp8(const TensorInfo& weight, const TensorInfo& scales, int64_t row_begin,
                            int64_t rows, int64_t col_begin, int64_t cols,
                            const MimoQkvLayout* qkv = nullptr, int rank = 0, int world = 1);
struct MimoMxfp4Matrix {
  int64_t rows = 0, cols = 0;
  std::vector<uint8_t> payload;  // packed low-nibble-first e2m1
  std::vector<uint8_t> scales;   // e8m0, one byte per 32 logical columns
};

// Host-side cold-path unpacking. BF16 conversion happens once after FP32
// dequantization; these paths do not introduce an FP8 requantization.
// Column/row windows bound memory and allow small real-checkpoint probes.
MimoBf16Matrix mimo_load_qkv_bf16(const TensorInfo& weight, const TensorInfo& scales,
                                  const MimoQkvLayout& layout, int rank, int world,
                                  int64_t col_begin, int64_t cols);
MimoBf16Matrix mimo_load_fp8_bf16(const TensorInfo& weight, const TensorInfo& scales,
                                  int64_t row_begin, int64_t rows, int64_t col_begin, int64_t cols);
MimoBf16Matrix mimo_slice_bf16(const TensorInfo& weight, int64_t row_begin, int64_t rows,
                               int64_t col_begin, int64_t cols);
MimoMxfp4Matrix mimo_slice_mxfp4(const TensorInfo& weight, const TensorInfo& scales,
                                 int64_t row_begin, int64_t rows, int64_t col_begin, int64_t cols);
MimoBf16Matrix mimo_dequant_mxfp4(const MimoMxfp4Matrix& matrix);

struct MimoWeightBudget {
  uint64_t bf16_bytes = 0;
  uint64_t fp32_bytes = 0;
  uint64_t mxfp4_bytes = 0;
  uint64_t total() const { return bf16_bytes + fp32_bytes + mxfp4_bytes; }
};
// Proposed initial residency: replicated embeddings, norms and routers;
// vocab-sharded output head; head-sharded BF16 attention; intermediate-
// sharded BF16 dense MLP and packed MXFP4 experts. Weights only, excluding
// GPU workspaces, caches, alignment and temporary host unpack buffers.
MimoWeightBudget mimo_weight_budget(const MimoTextConfig& cfg, int world);

// Ordinary matrix placement in checkpoint coordinates. Vectors are [N,1].
// QKV and packed experts use their specialized readers, not this layout.
struct MimoMatrixSlice {
  int64_t row_begin = 0, rows = 0, col_begin = 0, cols = 0;
};
MimoMatrixSlice mimo_matrix_placement(const MimoTextConfig& cfg, const MimoExpectedTensor& tensor,
                                      int rank, int world);

// A host-only checkpoint reader. It retains at most one cached shard and
// copies requested data into owned buffers; returned matrices never borrow
// pointers from an mmap. The caller may upload them in a later GPU loader.
class MimoCheckpointWeights {
 public:
  explicit MimoCheckpointWeights(const std::string& directory);
  const MimoTextConfig& config() const { return cfg_; }
  MimoBf16Matrix qkv(int layer, int rank, int world, int64_t col_begin, int64_t cols);
  MimoFp8Matrix fp8(const std::string& name, int rank, int world);
  MimoFp8Matrix qkv_fp8(int layer, int rank, int world);
  MimoMxfp4Matrix expert(int layer, int expert, const std::string& projection, int rank, int world);
  MimoMatrixSlice placement(const std::string& name, int rank, int world) const;
  // Norms, sinks, output projections, dense MLP, router, embedding and head.
  // Windows are relative to the rank-local placement; -1 means its remainder.
  // FP8 dense matrices are converted once to BF16 using source scale blocks.
  // Router weights stay BF16 here; execution must accumulate their dot in FP32.
  MimoBf16Matrix bf16(const std::string& name, int rank, int world, int64_t row_begin = 0,
                      int64_t rows = -1, int64_t col_begin = 0, int64_t cols = -1);
  std::vector<float> router_bias(int layer);

 private:
  std::shared_ptr<SafetensorsFile> shard(const std::string& tensor);
  std::string directory_, cached_name_;
  MimoTextConfig cfg_;
  std::unordered_map<std::string, std::string> index_;
  std::unordered_map<std::string, MimoExpectedTensor> expected_;
  std::shared_ptr<SafetensorsFile> cached_;
};

}  // namespace dgpp
