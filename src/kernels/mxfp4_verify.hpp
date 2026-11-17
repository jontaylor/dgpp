#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {
// Specialization for the TP2 target, four rows with six routes each.
// Split out singleton experts and leave reused groups in b12x packet format.
void launch_mxfp4_verify_singletons(const uint16_t* hidden, uint16_t* activation, float* output,
                                    const uint64_t* table, const int32_t* ids,
                                    const int32_t* routes, int32_t* reused, cudaStream_t stream);
}  // namespace dgpp
