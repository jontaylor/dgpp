#pragma once

#include <cstdint>

namespace dgpp::fp4_layout {
// Eight permuted E8M0 bytes become one word. In each byte, the low
// nibble belongs to channel i, the high nibble to channel i+4.
__device__ __forceinline__ uint8_t mx_scale(const uint8_t* scales, int i, bool nibbles) {
  if (!nibbles) return scales[i];
  return 0x70 | ((scales[(i / 8) * 4 + (i % 4)] >> (i & 4)) & 15);
}

// The packed word containing source (n,k). Its eight nibbles follow the MMA
// lane ordering, rather than source K order. K is a code index, not a byte.
__device__ __forceinline__ int packed_word(int n, int k, int rows) {
  const int tc_col = n % 8;
  const int tc_row = (k % 8) / 2;
  const int lane_word = (tc_col * 4 + tc_row) * 4 + (n % 64) / 16;
  return ((k / 16) * (rows / 64) + n / 64) * 128 + lane_word;
}
__device__ __forceinline__ int packed_slot(int n, int k) {
  const int offset = (k % 16 >= 8 ? 2 : 0) + k % 2;
  const int concat = (n % 16 >= 8 ? 4 : 0) + offset;
  return (concat % 2) * 4 + concat / 2;
}
__device__ __forceinline__ int swap_scale_pair(int n) {
  const int r = n % 4;
  return n - r + (r == 1 ? 2 : r == 2 ? 1 : r);
}
__device__ __forceinline__ int transpose_scale_block(int n) {
  return n / 64 * 64 + (n % 8) * 8 + (n % 64) / 8;
}

}  // namespace dgpp::fp4_layout
