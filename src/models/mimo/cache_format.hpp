#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string_view>
namespace dgpp {
inline bool mimo_fp8_cache_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("DGPP_MIMO_FP8_KV");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("DGPP_MIMO_FP8_KV must be 0 or 1 (unit-scale E4M3)");
  }();
  return enabled;
}
inline bool mimo_fp8_cache_audit_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("DGPP_MIMO_FP8_KV_AUDIT");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) != "1")
      throw std::invalid_argument("DGPP_MIMO_FP8_KV_AUDIT must be 0 or 1");
    if (!mimo_fp8_cache_enabled())
      throw std::invalid_argument("DGPP_MIMO_FP8_KV_AUDIT requires DGPP_MIMO_FP8_KV=1");
    return true;
  }();
  return enabled;
}
}  // namespace dgpp
