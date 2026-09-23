#pragma once
#include "common/dtypes.hpp"

namespace dgpp {
// Precondition: value is expanded E4M3 (or NaN), hence every finite bit pattern
// is exactly representable in BF16. This is not a general FP32->BF16 conversion.
// Keep the legacy signed canonical-NaN policy and both signed zeros.
DGPP_HD inline uint16_t mimo_fp8_expanded_to_bf16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint16_t top = static_cast<uint16_t>(bits >> 16);
  return (bits & 0x7fffffffu) > 0x7f800000u
      ? static_cast<uint16_t>((top & 0x8000u) | 0x7fc0u) : top;
}
}  // namespace dgpp
