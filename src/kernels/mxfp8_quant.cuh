#pragma once

#include <cstdint>

#include <cuda_fp8.h>

namespace dgpp {
// Eight adjacent lanes each contribute four consecutive values to a K32 group.
template <bool ReturnResidual = false>
__device__ __forceinline__ void mxfp8_quantize_four(const float (&values)[4], float minimum,
                                                    uint8_t* codes, uint8_t* scale, float* residual = nullptr) {
  float maximum =
      fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])), fmaxf(fabsf(values[2]), fabsf(values[3])));
  const unsigned active = __activemask();
#pragma unroll
  for (int offset = 4; offset; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, offset, 8));
  maximum = fmaxf(maximum, minimum);
  // Exactly ceil(log2(amax / 448)) from IEEE bits, without log2 rounding
  // at the exponent boundary. A zero group uses the caller's minimum.
  const int biased = static_cast<int>((__float_as_uint(maximum) >> 23) & 255);
  const float threshold = __uint_as_float((static_cast<unsigned>(biased) << 23) | 0x600000u);
  const int exponent = max(-127, biased - 135 + (maximum > threshold));
  const float inverse = __uint_as_float(static_cast<unsigned>(127 - exponent) << 23);
  const uint16_t lo = __nv_cvt_float2_to_fp8x2(
      make_float2(values[0] * inverse, values[1] * inverse), __NV_SATFINITE, __NV_E4M3);
  const uint16_t hi = __nv_cvt_float2_to_fp8x2(
      make_float2(values[2] * inverse, values[3] * inverse), __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<uint32_t*>(codes) =
      static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
  if constexpr (ReturnResidual) {
    const float coefficient = exponent == -127 ? __uint_as_float(0x00400000u)
                                               : __uint_as_float(static_cast<unsigned>(exponent + 127) << 23);
    const float2 low = __half22float2(__nv_cvt_fp8x2_to_halfraw2(lo, __NV_E4M3));
    const float2 high = __half22float2(__nv_cvt_fp8x2_to_halfraw2(hi, __NV_E4M3));
    residual[0] = values[0] - low.x * coefficient;
    residual[1] = values[1] - low.y * coefficient;
    residual[2] = values[2] - high.x * coefficient;
    residual[3] = values[3] - high.y * coefficient;
  }
  if ((threadIdx.x & 7) == 0) *scale = static_cast<uint8_t>(exponent + 127);
}
}  // namespace dgpp
