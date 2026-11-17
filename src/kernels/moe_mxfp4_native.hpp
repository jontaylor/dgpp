#pragma once
#include "kernels/glm_moe_launch.hpp"
namespace dgpp {
void launch_moe_mxfp8_quantize_tiles(const uint16_t*, size_t, int, uint8_t*, uint8_t*,
                                     const MoeSegment*, const MoeTile*, const int32_t*,
                                     cudaStream_t);
void launch_moe_mxfp8_swiglu_tiles(const uint16_t* gate, const uint16_t* up, float limit, size_t stride, int k,
                                   uint8_t* codes, uint8_t* scales, const MoeSegment* segs,
                                   const MoeTile* tiles, const int32_t* count, cudaStream_t stream, int blocks = 96);
// Quantized E4M3/E8M0 activations and raw MXFP4 expert views. Supports canonical,
// prepared K16/N64, and exact chunk-direct views. Output strides are elements.
void launch_moe_mxfp4_native_bf16(const uint8_t*, const uint8_t*, const MoeSegment*, int, int,
                                  const MoeExpertView*, int, uint16_t*, size_t, int, int,
                                  cudaStream_t, const int32_t* = nullptr);
void launch_moe_mxfp4_native_f32(const uint8_t*, const uint8_t*, const MoeSegment*, int, int,
                                 const MoeExpertView*, int, float*, size_t, int, int, cudaStream_t,
                                 const int32_t* = nullptr);
}  // namespace dgpp
