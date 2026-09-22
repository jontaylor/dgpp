#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "models/mimo/cache_format.hpp"
#include "models/mimo/config.hpp"

DGPP_TEST(mimo_cache_scale_metadata_is_never_silently_applied) {
  std::ifstream file(std::string(DGPP_SOURCE_DIR) + "/tests/data/mimo/config.json");
  const std::string original{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (original.empty()) throw std::runtime_error("missing MiMo config fixture");
  for (const auto field : {"\"k_scale\": 2", "\"v_scale\": 0.5", "\"calculate_kv_scales\": true"}) {
    std::string json = original;
    json.insert(json.find('{') + 1, std::string(field) + ",");
    bool rejected = false;
    try {
      const auto doc = dgpp::minijson::parse(json);
      dgpp::MimoTextConfig::parse(doc.root);
    } catch (const std::exception& e) {
      const std::string message = e.what();
      rejected = message.find("FP8 cache") != std::string::npos;
    }
    if (dgpp::mimo_fp8_cache_enabled() && !rejected)
      throw std::runtime_error("FP8 cache silently accepted non-unit/dynamic scale metadata");
  }
  if (dgpp::mimo_fp8_cache_enabled()) {
    std::string nested = original;
    nested.insert(nested.find('{', nested.find("\"quantization_config\"")) + 1,
                  "\"kv_scale\": 2,");
    bool rejected = false;
    try {
      const auto doc = dgpp::minijson::parse(nested);
      dgpp::MimoTextConfig::parse(doc.root);
    } catch (const std::exception& e) {
      rejected = std::string(e.what()).find("unit scales") != std::string::npos;
    }
    if (!rejected) throw std::runtime_error("nested non-unit cache scale accepted");
  }
  std::string json = original;
  json.insert(json.find('{') + 1, "\"k_scale\": 1, \"v_scale\": 1, \"calculate_kv_scales\": false,");
  const auto doc = dgpp::minijson::parse(json);
  dgpp::MimoTextConfig::parse(doc.root);
}
