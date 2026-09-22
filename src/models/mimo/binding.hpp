#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/mimo/config.hpp"

namespace dgpp {
struct MimoTensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};
struct MimoExpectedTensor : MimoTensorDesc {
  std::string name;
};
// Main text stack only. The released U8 expert scales carry MXFP4 scale
// bytes, not NVFP4 e4m3 scales. No dequantization is implemented here.
std::vector<MimoExpectedTensor> mimo_expected_text_tensors(const MimoTextConfig& cfg);
std::vector<MimoExpectedTensor> mimo_expected_mtp_tensors(const MimoTextConfig& cfg);
struct MimoBindReport {
  size_t text = 0;
  size_t multimodal = 0;
  size_t mtp = 0;
};
// Throws on missing/mismatched text tensors or unknown names. Recognized
// multimodal and MTP prefixes are counted but their contents are not validated.
MimoBindReport mimo_validate_text_binding(
    const MimoTextConfig& cfg, const std::unordered_map<std::string, MimoTensorDesc>& tensors);
}  // namespace dgpp
