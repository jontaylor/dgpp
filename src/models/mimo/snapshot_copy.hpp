#pragma once

#include "common/cuda_check.hpp"
#include "models/mimo/attention.hpp"

namespace dgpp {
// The layout stays fixed: capacity-sized global K then V, or window-sized
// packed K then V. Only global history can retain its previous packed offsets.
inline size_t mimo_write_snapshot_layer(const MimoAttentionShape& shape, int64_t position,
                                        int64_t previous, const void* keys,
                                        const void* values, uint8_t* dst,
                                        cudaStream_t stream) {
  const auto* k = static_cast<const uint8_t*>(keys);
  const auto* v = static_cast<const uint8_t*>(values);
  auto spans = mimo_snapshot_spans(shape, position);
  size_t packed = 0;
  if (!shape.window && previous > 0 && previous <= position) {
    spans[0] = {previous, position - previous};
    packed = static_cast<size_t>(previous);
  }
  const size_t slots = shape.window ? shape.window : shape.capacity;
  const size_t snapshot_keys = slots * shape.k_width();
  size_t bytes = 0;
  for (const auto span : spans) {
    if (!span.count) continue;
    const size_t kb = size_t(span.count) * shape.k_width() * shape.cache_element_bytes();
    const size_t vb = size_t(span.count) * shape.v_width() * shape.cache_element_bytes();
    DGPP_CUDA_OK(cudaMemcpyAsync(dst + packed * shape.k_width() * shape.cache_element_bytes(),
                                 k + size_t(span.first) * shape.k_width() * shape.cache_element_bytes(), kb,
                                 cudaMemcpyDeviceToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(dst + snapshot_keys * shape.cache_element_bytes() + packed * shape.v_width() * shape.cache_element_bytes(),
                                 v + size_t(span.first) * shape.v_width() * shape.cache_element_bytes(), vb,
                                 cudaMemcpyDeviceToDevice, stream));
    packed += span.count;
    bytes += kb + vb;
  }
  return bytes;
}
inline void mimo_read_snapshot_layer(const MimoAttentionShape& shape, int64_t position,
                                      const uint8_t* src, void* keys,
                                      void* values, cudaStream_t stream) {
  auto* k = static_cast<uint8_t*>(keys);
  auto* v = static_cast<uint8_t*>(values);
  const size_t slots = shape.window ? shape.window : shape.capacity;
  const size_t snapshot_keys = slots * shape.k_width();
  size_t packed = 0;
  for (const auto span : mimo_snapshot_spans(shape, position)) {
    if (!span.count) continue;
    DGPP_CUDA_OK(cudaMemcpyAsync(k + size_t(span.first) * shape.k_width() * shape.cache_element_bytes(),
                                 src + packed * shape.k_width() * shape.cache_element_bytes(),
                                 size_t(span.count) * shape.k_width() * shape.cache_element_bytes(),
                                 cudaMemcpyDeviceToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(v + size_t(span.first) * shape.v_width() * shape.cache_element_bytes(),
                                 src + snapshot_keys * shape.cache_element_bytes() + packed * shape.v_width() * shape.cache_element_bytes(),
                                 size_t(span.count) * shape.v_width() * shape.cache_element_bytes(),
                                 cudaMemcpyDeviceToDevice, stream));
    packed += span.count;
  }
}
}  // namespace dgpp
