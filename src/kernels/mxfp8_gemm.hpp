#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {
// Fused attention-finish layout: contiguous [rows,groups,k] codes then their K32 scales.
// Weight [groups,n,k] has one expanded E8M0 scale per 32x32 block.
// Supports 1..8 decode rows, k=4096 and n divisible by 32.
// An early weight copy requires immutable weights throughout graph replay.
void launch_mxfp8_woa_decode(const uint8_t* packed, const uint8_t* w, const float* ws,
                             uint16_t* out, int rows, int groups, int n, int k, cudaStream_t,
                             bool early_weight_copy = false);

// TP2 wo_b: BF16 [rows,4096], optional second MXFP8 term for activation error.
// Scratch holds one or two [rows,4096] code planes and their K32 scales.
void launch_mxfp8_wob_decode(const uint16_t* x, const uint8_t* w, const float* ws, uint16_t* out,
                             int rows, int n, uint8_t* scratch, bool residual, cudaStream_t stream);

// Dynamic MXFP8: BF16 [rows,k] -> E4M3 [rows,k], E8M0 [rows,k/32].
// Each 32-value group's scale is ceil-rounded to a power of two, matching
// the pinned SGLang MiniMax producer. x_stride is in BF16 elements.
void launch_mxfp8_quantize_rows(const uint16_t* x, size_t x_stride, int rows, int k, uint8_t* codes,
                                uint8_t* scales, cudaStream_t);

// Pinned SGLang dense/shared SwiGLU-OAI producer, including its sigmoid
// evaluation and BF16 boundary. Gate and up have the same BF16 row stride.
void launch_minimax_swiglu_mxfp8_rows(const uint16_t* gate, const uint16_t* up, size_t stride,
                                      int rows, int k, float alpha, float limit, uint8_t* codes,
                                      uint8_t* scales, cudaStream_t);

// Native SM121 block-scaled E4M3 x E4M3, FP32 accumulation. Both operands
// are row-major with k-contiguous payloads and k/32-contiguous E8M0 scales.
// out[m,n] = A[m,k] * W[n,k]^T; out_stride is in output elements (0 -> n).
// All supported row geometries retain the same ordered k32 MMA chain.
// output_tile overrides the column tile for kernel diagnostics (8/16/32/64);
// zero selects the production policy. Overrides use sixteen-row tiles.
void launch_mxfp8_gemm_bf16(const uint8_t* a, const uint8_t* a_scales, const uint8_t* w,
                            const uint8_t* w_scales, uint16_t* out, int m, int n, int k,
                            cudaStream_t, size_t out_stride = 0, int output_tile = 0);
void launch_mxfp8_gemm_f32(const uint8_t* a, const uint8_t* a_scales, const uint8_t* w,
                           const uint8_t* w_scales, float* out, int m, int n, int k, cudaStream_t,
                           size_t out_stride = 0, int output_tile = 0);
// Optional prefill path: FP8 weights with one E8M0 scale per 32x32 tile,
// dynamically quantized activations. Returns false if scratch/shape is unsuitable.
bool launch_mxfp8_grid32_bf16(const uint16_t*, size_t, const uint8_t*, const uint8_t*, uint16_t*,
                              int, int, int, cudaStream_t, size_t, void*, size_t);
bool launch_mxfp8_grid32_f32(const uint16_t*, size_t, const uint8_t*, const uint8_t*, float*, int,
                             int, int, cudaStream_t, size_t, void*, size_t);
// DSV loader's exact E8M0-to-float expansion, NOT arbitrary FP32 scales.
bool launch_mxfp8_grid32_expanded_bf16(const uint16_t*, size_t, const uint8_t*, const float*,
                                       uint16_t*, int, int, int, cudaStream_t, size_t, void*,
                                       size_t);
}  // namespace dgpp
