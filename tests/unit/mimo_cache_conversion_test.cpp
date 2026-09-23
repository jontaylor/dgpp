#include "common/test.hpp"
#include "models/mimo/cache_conversion.hpp"
#include <stdexcept>

DGPP_TEST(mimo_fp8_exact_expansion_all_codes_and_signed_nan_payloads) {
  for (int code = 0; code < 256; ++code) {
    const float value = dgpp::fp8_e4m3_bits_to_float(static_cast<uint8_t>(code));
    if (dgpp::mimo_fp8_expanded_to_bf16(value) != dgpp::float_to_bf16_bits(value))
      throw std::runtime_error("FP8 exact expansion differs from legacy RNE");
  }
  for (uint32_t bits : {0x7f800001u, 0x7fc00000u, 0x7fffffffu,
                        0xff800001u, 0xffc00000u, 0xffffffffu}) {
    const float value = std::bit_cast<float>(bits);
    if (dgpp::mimo_fp8_expanded_to_bf16(value) != dgpp::float_to_bf16_bits(value))
      throw std::runtime_error("FP8 exact expansion changed signed NaN canonicalization");
  }
}
