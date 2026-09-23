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
// Direct E4M3-to-BF16 expansion: every finite input is exact, including
// subnormals and signed zero. Keep the signed canonical quiet-NaN contract.
DGPP_HD inline uint16_t mimo_fp8_bits_to_bf16(uint8_t code) {
  const uint16_t sign = uint16_t(code & 128u) << 8;
  const unsigned magnitude = code & 127u;
  if (magnitude == 127u) return sign | 0x7fc0u;
  if (magnitude >= 8u) return sign | uint16_t((magnitude << 4) + 0x3c00u);
  // Eight-entry subnormal table expressed as branches to avoid device storage.
  const uint16_t low = magnitude == 0 ? 0 : magnitude == 1 ? 0x3b00 :
      magnitude < 4 ? uint16_t(0x3b80 + (magnitude - 2) * 0x40) :
      uint16_t(0x3c00 + (magnitude - 4) * 0x20);
  return sign | low;
}
}  // namespace dgpp
