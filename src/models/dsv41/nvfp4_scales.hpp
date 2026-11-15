#pragma once
// ModelOpt's weight_scale_2 MULTIPLIES the dequantized NVFP4 weight. DGPP's
// existing NVFP4 kernels DIVIDE the final dot by a descriptor global. The
// target checkpoint uses exactly invertible powers of two, so translating
// that descriptor preserves the factor exactly, without touching block scales
// or packed weights. Other factors require a multiplication epilogue rather
// than a rounded reciprocal and are refused here.
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace dgpp {
inline float dsv41_nvfp4_inverse_global(float weight_scale_2, float input_scale) {
  if (!std::isfinite(input_scale) || input_scale <= 0)
    throw std::invalid_argument("dsv41 NVFP4: input_scale must be finite and positive");
  const auto bits = std::bit_cast<uint32_t>(weight_scale_2);
  const auto exponent = (bits >> 23) & 255u;
  if ((bits & 0x807fffffu) || exponent < 1 || exponent > 253)
    throw std::invalid_argument("dsv41 NVFP4: weight_scale_2 must be a positive normal power of two with a normal reciprocal");
  return std::bit_cast<float>((254u - exponent) << 23);
}
}  // namespace dgpp
