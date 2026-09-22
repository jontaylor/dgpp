#pragma once
#include <cstdint>

#include <cuda_runtime.h>
namespace dgpp {
// Full-head RMSNorm followed by half-rotary NeoX RoPE. Rows with p<0 are padding.
void dflash_norm_rope(uint16_t* x, const uint16_t* norm, const int64_t* pos, int rows, int heads,
                      cudaStream_t stream);
void dflash_append(const uint16_t* k, const uint16_t* v, uint16_t* kc, uint16_t* vc,
                   const int64_t* pos, const int32_t* req, int rows, int capacity, int kv_heads,
                   cudaStream_t stream);
// Block-local K/V never enters the committed context ring. Window is symmetric
// for a noncausal SWA query; all eight block positions are visible.
void dflash_attention(const uint16_t* q, const uint16_t* k, const uint16_t* v, const uint16_t* kc,
                      const uint16_t* vc, const uint16_t* sink, const int64_t* pos,
                      const int32_t* req, uint16_t* out, int groups, int capacity, int q_heads,
                      int kv_heads, cudaStream_t stream);
void dflash_mask(uint16_t* embeddings, const uint16_t* mask, int rows, cudaStream_t stream);
void dflash_features(const uint16_t* src, uint16_t* dst, int rows, int feature,
                     cudaStream_t stream);
void dflash_select(const float* src, float* dst, int groups, int rows_out, int vocab,
                   int draft_index, cudaStream_t stream);
}  // namespace dgpp
