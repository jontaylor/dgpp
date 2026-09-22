#pragma once

#include "kernels/gemm.hpp"

namespace dgpp {
// Keep the session core's [T, vocab] layout even when only its final row is
// consumed. Hidden rows are read-only: native MTP still needs the whole chunk.
// Decode verification and diagnostic all_rows calls retain every projection.
inline void mimo_project_head(IGemm& gemm, const uint16_t* hidden, const uint16_t* weight,
                              float* logits, int rows, int vocab, int width,
                              bool decode, bool all_rows, cudaStream_t stream) {
  const int first = decode || all_rows ? 0 : rows - 1;
  gemm.matmul(hidden + size_t(first) * width, weight, logits + size_t(first) * vocab,
              rows - first, vocab, width, DType::BF16, GemmOut::F32, width,
              nullptr, 0, stream);
}
}  // namespace dgpp
