#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace dgpp {
struct ExpertDemand;
// Optional b12x W4A16 specialization for the TP2 NVFP4 target. All preparation
// precedes graph capture; execution has no Python or JIT dependency.
bool sm12x_moe_enabled(int hidden, int intermediate, int experts, int top_k);
class Sm12xMoe {
 public:
  static constexpr int kCapacity = 64;
  static size_t device_bytes();
  static size_t pinned_bytes();
  Sm12xMoe();
  ~Sm12xMoe();
  Sm12xMoe(const Sm12xMoe&) = delete;
  Sm12xMoe& operator=(const Sm12xMoe&) = delete;
  // addresses holds prepared experts indexed by router ID. A wave may bind
  // inactive entries to any live expert. execute_experts masks route blocks
  // while retaining their scheduling positions and reduction domain; empty
  // executes the complete routed set.
  void prepare(std::span<const int32_t> ids, std::span<const uint8_t* const> addresses,
               int rows, cudaStream_t stream, std::span<const int> execute_experts = {});
  // Unweighted FP32 output [rows,top_k+1,H]; shared slots are untouched.
  void run(const uint16_t* input, const int32_t* ids, const float* weights,
           float* slots, int rows, cudaStream_t stream);
  void prepare_device(const uint64_t* table, const int32_t* ids,
                      const ExpertDemand* demand, int rows, cudaStream_t stream);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dgpp
