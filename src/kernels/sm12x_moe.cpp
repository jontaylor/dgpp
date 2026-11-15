#include "kernels/sm12x_moe.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include "common/cuda_check.hpp"
#include "kernels/moe_fp4_pack.hpp"
#include "kernels/expert_demand.hpp"
#ifdef DGPP_HAVE_B12X_AOT
#define CUTE_DSL_CUDA_ERROR_CHECK(err) DGPP_CUDA_OK(err)
#include "dgpp_b12x_raw_fp32.h"
static_assert(DGPP_B12X_CAPACITY == dgpp::Sm12xMoe::kCapacity, "regenerate the b12x AOT backend");
#endif

namespace dgpp {
namespace {
constexpr int H = 5120, I = 1152, E = 384, K = 6, M = Sm12xMoe::kCapacity, B = 8;
constexpr size_t kTemp = 4 * 256 * B * 256;
constexpr size_t kPacket = E * 6 * sizeof(uint64_t) + M * K * (B + 1) * sizeof(int32_t) + sizeof(int32_t);
}
bool sm12x_moe_enabled(int hidden, int intermediate, int experts, int top_k) {
  const char* option = std::getenv("DGPP_MOE_B12X");
  if (!option || option[0] != '1') return false;
#ifndef DGPP_HAVE_B12X_AOT
  throw std::runtime_error("DGPP_MOE_B12X requires a build with DGPP_B12X_AOT_DIR");
#endif
  return hidden == H && intermediate == I && experts == E && top_k == K;
}
size_t Sm12xMoe::device_bytes() {
  return M * K * 3 * I * sizeof(uint16_t) + 2 * kTemp * sizeof(float) +
      (4 * 256 + 2) * sizeof(int32_t) + kPacket;
}
size_t Sm12xMoe::pinned_bytes() { return kPacket; }
#ifdef DGPP_HAVE_B12X_AOT
struct Sm12xMoe::Impl {
  dgpp_b12x_raw_fp32_Kernel_Module_t module{};
  uint16_t *fc1 = nullptr, *act = nullptr;
  float *tmp1 = nullptr, *tmp2 = nullptr;
  int32_t* locks = nullptr;
  uint8_t *packet = nullptr, *host = nullptr;
  cudaEvent_t upload = nullptr;
  bool uploaded = false;
  template<class T> void allocate(T*& p, size_t n) {
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
  }
  void initialize() {
    dgpp_b12x_raw_fp32_Kernel_Module_Load(&module);
    allocate(fc1, M*K*2*I); allocate(act, M*K*I);
    allocate(tmp1, kTemp); allocate(tmp2, kTemp); allocate(locks, 4*256+2);
    allocate(packet, kPacket);
    DGPP_CUDA_OK(cudaMemset(locks, 0, (4*256+2)*sizeof(int32_t)));
    DGPP_CUDA_OK(cudaHostAlloc(&host, kPacket, cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&upload, cudaEventDisableTiming));
  }
  ~Impl() {
    if (uploaded) (void)cudaEventSynchronize(upload);
    for (void* p : std::array<void*,6>{fc1, act, tmp1, tmp2, locks, packet})
      if (p) (void)cudaFree(p);
    if (host) (void)cudaFreeHost(host);
    if (upload) (void)cudaEventDestroy(upload);
    if (module.module) (void)cudaLibraryUnload(module.module);
  }
};
Sm12xMoe::Sm12xMoe() : impl_(std::make_unique<Impl>()) { impl_->initialize(); }
Sm12xMoe::~Sm12xMoe() = default;
void Sm12xMoe::prepare_device(const uint64_t* table, const int32_t* ids,
                            const ExpertDemand* demand, int rows, cudaStream_t stream) {
  if (rows < 1 || rows > M) throw std::invalid_argument("b12x device route row count");
  launch_expert_route_packet(impl_->packet, table, ids, demand, rows, stream);
}
void Sm12xMoe::prepare(std::span<const int32_t> ids,
    std::span<const uint8_t* const> addresses, int rows, cudaStream_t stream,
    std::span<const int> execute_experts) {
  if (rows < 1 || rows > M || ids.size() != static_cast<size_t>(rows*K) || addresses.size() != E)
    throw std::invalid_argument("b12x MoE: unsupported routing geometry");
  auto& c = *impl_;
  if (c.uploaded) DGPP_CUDA_OK(cudaEventSynchronize(c.upload));
  auto* table = reinterpret_cast<uint64_t*>(c.host);
  for (int e = 0; e < E; ++e) {
    if (!addresses[e]) throw std::invalid_argument("b12x MoE: missing prepared expert binding");
    const auto p = moe_fp4_packed_view(const_cast<uint8_t*>(addresses[e]), H, I);
    table[e*3] = reinterpret_cast<uintptr_t>(p.w13);
    table[e*3+1] = reinterpret_cast<uintptr_t>(p.s13);
    table[e*3+2] = reinterpret_cast<uintptr_t>(p.g13);
    table[E*3+e*3] = reinterpret_cast<uintptr_t>(p.w2);
    table[E*3+e*3+1] = reinterpret_cast<uintptr_t>(p.s2);
    table[E*3+e*3+2] = reinterpret_cast<uintptr_t>(p.g2);
  }
  auto* routes = reinterpret_cast<int32_t*>(c.host + E*6*sizeof(uint64_t));
  auto* experts = routes + M*K*B;
  auto* count = experts + M*K;
  std::fill_n(routes, M*K*B, rows*K);
  std::fill_n(experts, M*K, 0);
  std::array<std::vector<int32_t>,E> groups;
  std::array<bool,E> execute;
  execute.fill(execute_experts.empty());
  for (const int e : execute_experts) {
    if (e < 0 || e >= E) throw std::invalid_argument("b12x MoE: invalid wave expert ID");
    execute[e] = true;
  }
  for (int j = 0; j < rows*K; ++j) {
    if (ids[j] < 0 || ids[j] >= E) throw std::invalid_argument("b12x MoE: invalid expert ID");
    groups[ids[j]].push_back(j);
  }
  int blocks = 0;
  for (int e = 0; e < E; ++e)
    for (size_t begin = 0; begin < groups[e].size(); begin += B) {
      experts[blocks] = e;
      if (execute[e])
        for (int j = 0; j < B && begin+j < groups[e].size(); ++j)
          routes[blocks*B+j] = groups[e][begin+j];
      ++blocks;
    }
  *count = blocks*B;
  DGPP_CUDA_OK(cudaMemcpyAsync(c.packet, c.host, kPacket, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaEventRecord(c.upload, stream));
  c.uploaded = true;
}
void Sm12xMoe::run(const uint16_t* hidden, const int32_t* ids, const float* weights,
                   float* out, int rows, cudaStream_t stream) {
  if (rows < 1 || rows > M) throw std::invalid_argument("b12x MoE: row count exceeds capacity");
  auto& c = *impl_;
  dgpp_b12x_raw_fp32_Tensor_fc1_bf16_flat_t fc1{c.fc1};
  dgpp_b12x_raw_fp32_Tensor_activated_bf16_flat_t act{c.act};
  dgpp_b12x_raw_fp32_Tensor_fc2_bf16_flat_t fc2{out};
  auto* packed = reinterpret_cast<int32_t*>(c.packet + E*6*sizeof(uint64_t));
  dgpp_b12x_raw_fp32_Tensor_packed_route_indices_t routes{packed};
  dgpp_b12x_raw_fp32_Tensor_block_expert_ids_t experts{packed+M*K*B};
  dgpp_b12x_raw_fp32_Tensor_packed_route_count_t count{packed+M*K*(B+1)};
  dgpp_b12x_raw_fp32_Tensor_activation_amax_flat_t amax{c.tmp1};
  dgpp_b12x_raw_fp32_Tensor_fc1_c_tmp_f32_flat_t tmp1{c.tmp1};
  dgpp_b12x_raw_fp32_Tensor_fc2_c_tmp_f32_flat_t tmp2{c.tmp2};
  dgpp_b12x_raw_fp32_Tensor_locks_i32_flat_t locks{c.locks};
  // The generated C ABI does not express read-only pointer qualifiers.
  auto* input = const_cast<uint16_t*>(hidden);
  const int status = cute_dsl_dgpp_b12x_raw_fp32_wrapper(&c.module, input, input, input,
      c.packet, c.packet+E*3*sizeof(uint64_t), int64_t(E)*2*I*H/8, int64_t(E)*H*I/8,
      &fc1, &act, &fc2, c.tmp1, c.tmp1, c.tmp1, c.tmp1, &routes, &experts, &count, &amax,
      0, const_cast<float*>(weights), &tmp1, &tmp2, &locks, c.tmp1, c.tmp1, c.tmp1,
      const_cast<int32_t*>(ids), c.tmp1, c.tmp1,
      E, 0, rows, 32, stream);
  if (status) throw std::runtime_error("b12x MoE AOT launch failed: " + std::to_string(status));
}
#else
struct Sm12xMoe::Impl {};
void Sm12xMoe::prepare_device(const uint64_t*, const int32_t*, const ExpertDemand*, int, cudaStream_t) {
  throw std::logic_error("b12x AOT is unavailable");
}
Sm12xMoe::Sm12xMoe() { throw std::logic_error("b12x AOT is unavailable"); }
Sm12xMoe::~Sm12xMoe() = default;
void Sm12xMoe::prepare(std::span<const int32_t>, std::span<const uint8_t* const>, int, cudaStream_t,
                       std::span<const int>) {
  throw std::logic_error("b12x AOT is unavailable");
}
void Sm12xMoe::run(const uint16_t*, const int32_t*, const float*, float*, int, cudaStream_t) {
  throw std::logic_error("b12x AOT is unavailable");
}
#endif
}  // namespace dgpp
