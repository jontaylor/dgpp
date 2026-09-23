#pragma once
#include <cuda_fp8.h>
#include "common/dtypes.hpp"
#include "models/mimo/cache_conversion.hpp"
#ifndef DGPP_MIMO_FP8_KV_FAST_LOAD
#define DGPP_MIMO_FP8_KV_FAST_LOAD 0
#endif
namespace dgpp {
// E4M3 has no scale metadata: K and V always use scale 1. Saturation is explicit.
__device__ inline uint16_t mimo_cache_load(const void* cache, int64_t index, bool fp8) {
  if (!fp8) return static_cast<const uint16_t*>(cache)[index];
  __nv_fp8_e4m3 value;
  value.__x = static_cast<const uint8_t*>(cache)[index];
#if DGPP_MIMO_FP8_KV_FAST_LOAD
  return mimo_fp8_expanded_to_bf16(static_cast<float>(value));
#else
  return float_to_bf16_bits(static_cast<float>(value));
#endif
}
__device__ inline void mimo_cache_store(void* cache, int64_t index, float value, bool fp8) {
  if (fp8) static_cast<uint8_t*>(cache)[index] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  else static_cast<uint16_t*>(cache)[index] = float_to_bf16_bits(value);
}
}  // namespace dgpp
