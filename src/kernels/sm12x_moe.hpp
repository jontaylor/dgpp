#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include "loaders/expert_residency.hpp"

namespace dgpp {
struct ExpertDemand;
int sm12x_projection_mode();
// Optional b12x W4A16 specialization for the TP2 native FP4 target. All preparation
// precedes graph capture; execution has no Python or JIT dependency.
bool sm12x_moe_enabled(int hidden, int intermediate, int experts, int top_k);
class Sm12xMoe {
 public:
  static constexpr int kCapacity = 64;
  static size_t device_bytes();
  static size_t pinned_bytes();
  explicit Sm12xMoe(int scale_group = 16, bool mx_nibbles = false);
  ~Sm12xMoe();
  Sm12xMoe(const Sm12xMoe&) = delete;
  Sm12xMoe& operator=(const Sm12xMoe&) = delete;
  // addresses holds prepared experts indexed by router ID. A wave may bind
  // inactive entries to any live expert. execute_experts masks route blocks
  // while retaining their scheduling positions and reduction domain; empty
  // executes the complete routed set.
  void prepare(std::span<const int32_t> ids, std::span<const uint8_t* const> addresses,
               int rows, cudaStream_t stream, std::span<const int> execute_experts = {});
  // Quantize once before the demand packet; every residency wave uses this row.
  const uint8_t* prepare_verify_input(const uint16_t* hidden, int rows, cudaStream_t stream);
  const uint8_t* prepared_verify_input(int rows) const;
  // Unweighted FP32 output [rows,top_k+1,H]; shared slots are untouched.
  // A captured device route uses its layer's published binding table.
  // Host-prepared waves use the private bindings uploaded by prepare().
  void run(const uint16_t* input, const int32_t* ids, const float* weights,
           float* slots, int rows, cudaStream_t stream, const uint64_t* bindings = nullptr,
           expert::ExpertProjection part = expert::ExpertProjection::kAll,
           const uint8_t* verify_input = nullptr);
  void prepare_device(const uint64_t* table, const int32_t* ids,
                      ExpertDemand* demand, int rows, cudaStream_t stream, bool resident_wave = false);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dgpp
