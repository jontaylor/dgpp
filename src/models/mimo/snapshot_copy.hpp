#pragma once

#include "common/cuda_check.hpp"
#include "models/mimo/attention.hpp"

namespace dgpp {
// The layout stays fixed: capacity-sized global K then V, or window-sized
// packed K then V. Only global history can retain its previous packed offsets.
inline size_t mimo_write_snapshot_layer(const MimoAttentionShape& shape, int64_t position,
                                        int64_t previous, const uint16_t* keys,
                                        const uint16_t* values, uint8_t* dst,
                                        cudaStream_t stream) {
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
    const size_t kb = size_t(span.count) * shape.k_width() * 2;
    const size_t vb = size_t(span.count) * shape.v_width() * 2;
    DGPP_CUDA_OK(cudaMemcpyAsync(dst + packed * shape.k_width() * 2,
                                 keys + size_t(span.first) * shape.k_width(), kb,
                                 cudaMemcpyDeviceToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(dst + snapshot_keys * 2 + packed * shape.v_width() * 2,
                                 values + size_t(span.first) * shape.v_width(), vb,
                                 cudaMemcpyDeviceToDevice, stream));
    packed += span.count;
    bytes += kb + vb;
  }
  return bytes;
}
inline void mimo_read_snapshot_layer(const MimoAttentionShape& shape, int64_t position,
                                      const uint8_t* src, uint16_t* keys,
                                      uint16_t* values, cudaStream_t stream) {
  const size_t slots = shape.window ? shape.window : shape.capacity;
  const size_t snapshot_keys = slots * shape.k_width();
  size_t packed = 0;
  for (const auto span : mimo_snapshot_spans(shape, position)) {
    if (!span.count) continue;
    DGPP_CUDA_OK(cudaMemcpyAsync(keys + size_t(span.first) * shape.k_width(),
                                 src + packed * shape.k_width() * 2,
                                 size_t(span.count) * shape.k_width() * 2,
                                 cudaMemcpyDeviceToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(values + size_t(span.first) * shape.v_width(),
                                 src + snapshot_keys * 2 + packed * shape.v_width() * 2,
                                 size_t(span.count) * shape.v_width() * 2,
                                 cudaMemcpyDeviceToDevice, stream));
    packed += span.count;
  }
}
}  // namespace dgpp
