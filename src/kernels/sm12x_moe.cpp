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
#include "kernels/mxfp4_verify.hpp"
#ifdef DGPP_HAVE_B12X_AOT
#define CUTE_DSL_CUDA_ERROR_CHECK(err) DGPP_CUDA_OK(err)
#include "dgpp_b12x_raw_fp32.h"
#include "dgpp_b12x_mxfp4_fp32.h"
#ifdef DGPP_HAVE_B12X_MX_NIBBLES
#include "dgpp_b12x_mxfp4_nibble_fp32.h"
#endif
static_assert(DGPP_B12X_CAPACITY == dgpp::Sm12xMoe::kCapacity, "regenerate the b12x AOT backend");
#endif

namespace dgpp {
namespace {
bool verify_split_enabled() {
  const char* value = std::getenv("DGPP_MX_VERIFY_SINGLETONS");
  return value && value[0] == '1';
}
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
      (4 * 256 + 2) * sizeof(int32_t) + kPacket + (verify_split_enabled() ? (M*K*(B+1)+1)*sizeof(int32_t) : 0);
}
size_t Sm12xMoe::pinned_bytes() { return kPacket; }
#ifdef DGPP_HAVE_B12X_AOT
struct NvBackend {
  dgpp_b12x_raw_fp32_Kernel_Module_t module{};
  using Fc1 = dgpp_b12x_raw_fp32_Tensor_fc1_bf16_flat_t;
  using Act = dgpp_b12x_raw_fp32_Tensor_activated_bf16_flat_t;
  using Fc2 = dgpp_b12x_raw_fp32_Tensor_fc2_bf16_flat_t;
  using Routes = dgpp_b12x_raw_fp32_Tensor_packed_route_indices_t;
  using Experts = dgpp_b12x_raw_fp32_Tensor_block_expert_ids_t;
  using Count = dgpp_b12x_raw_fp32_Tensor_packed_route_count_t;
  using Amax = dgpp_b12x_raw_fp32_Tensor_activation_amax_flat_t;
  using Tmp1 = dgpp_b12x_raw_fp32_Tensor_fc1_c_tmp_f32_flat_t;
  using Tmp2 = dgpp_b12x_raw_fp32_Tensor_fc2_c_tmp_f32_flat_t;
  using Locks = dgpp_b12x_raw_fp32_Tensor_locks_i32_flat_t;
  static constexpr auto launch = cute_dsl_dgpp_b12x_raw_fp32_wrapper;
  void load() { dgpp_b12x_raw_fp32_Kernel_Module_Load(&module); }
};
struct MxBackend {
  dgpp_b12x_mxfp4_fp32_Kernel_Module_t module{};
  using Fc1 = dgpp_b12x_mxfp4_fp32_Tensor_fc1_bf16_flat_t;
  using Act = dgpp_b12x_mxfp4_fp32_Tensor_activated_bf16_flat_t;
  using Fc2 = dgpp_b12x_mxfp4_fp32_Tensor_fc2_bf16_flat_t;
  using Routes = dgpp_b12x_mxfp4_fp32_Tensor_packed_route_indices_t;
  using Experts = dgpp_b12x_mxfp4_fp32_Tensor_block_expert_ids_t;
  using Count = dgpp_b12x_mxfp4_fp32_Tensor_packed_route_count_t;
  using Amax = dgpp_b12x_mxfp4_fp32_Tensor_activation_amax_flat_t;
  using Tmp1 = dgpp_b12x_mxfp4_fp32_Tensor_fc1_c_tmp_f32_flat_t;
  using Tmp2 = dgpp_b12x_mxfp4_fp32_Tensor_fc2_c_tmp_f32_flat_t;
  using Locks = dgpp_b12x_mxfp4_fp32_Tensor_locks_i32_flat_t;
  static constexpr auto launch = cute_dsl_dgpp_b12x_mxfp4_fp32_wrapper;
  void load() { dgpp_b12x_mxfp4_fp32_Kernel_Module_Load(&module); }
};
#ifdef DGPP_HAVE_B12X_MX_NIBBLES
struct MxNibbleBackend {
  dgpp_b12x_mxfp4_nibble_fp32_Kernel_Module_t module{};
  using Fc1 = dgpp_b12x_mxfp4_nibble_fp32_Tensor_fc1_bf16_flat_t;
  using Act = dgpp_b12x_mxfp4_nibble_fp32_Tensor_activated_bf16_flat_t;
  using Fc2 = dgpp_b12x_mxfp4_nibble_fp32_Tensor_fc2_bf16_flat_t;
  using Routes = dgpp_b12x_mxfp4_nibble_fp32_Tensor_packed_route_indices_t;
  using Experts = dgpp_b12x_mxfp4_nibble_fp32_Tensor_block_expert_ids_t;
  using Count = dgpp_b12x_mxfp4_nibble_fp32_Tensor_packed_route_count_t;
  using Amax = dgpp_b12x_mxfp4_nibble_fp32_Tensor_activation_amax_flat_t;
  using Tmp1 = dgpp_b12x_mxfp4_nibble_fp32_Tensor_fc1_c_tmp_f32_flat_t;
  using Tmp2 = dgpp_b12x_mxfp4_nibble_fp32_Tensor_fc2_c_tmp_f32_flat_t;
  using Locks = dgpp_b12x_mxfp4_nibble_fp32_Tensor_locks_i32_flat_t;
  static constexpr auto launch = cute_dsl_dgpp_b12x_mxfp4_nibble_fp32_wrapper;
  void load() { dgpp_b12x_mxfp4_nibble_fp32_Kernel_Module_Load(&module); }
};
#endif
struct Sm12xMoe::Impl {
  int32_t* reused = nullptr;
  NvBackend nv;
  MxBackend mx;
#ifdef DGPP_HAVE_B12X_MX_NIBBLES
  MxNibbleBackend nibble;
#endif
  int group;
  bool mx_nibbles;
  Impl(int scale_group, bool compact) : group(scale_group), mx_nibbles(compact) {
    if (compact && (group != 32 || verify_split_enabled()))
      throw std::invalid_argument("MX scale packing requires MXFP4 with unsplit verification");
#ifndef DGPP_HAVE_B12X_MX_NIBBLES
    if (compact) throw std::invalid_argument("rebuild with the mxfp4-nibble b12x export");
#endif
    if (group != 16 && group != 32) throw std::invalid_argument("b12x FP4 scale group");
  }
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
#ifdef DGPP_HAVE_B12X_MX_NIBBLES
    if (mx_nibbles) nibble.load();
    else
#endif
    if (group == 32) mx.load(); else nv.load();
    allocate(fc1, M*K*2*I); allocate(act, M*K*I);
    allocate(tmp1, kTemp); allocate(tmp2, kTemp); allocate(locks, 4*256+2);
    allocate(packet, kPacket);
    if (group == 32 && verify_split_enabled()) allocate(reused, M*K*(B+1)+1);
    DGPP_CUDA_OK(cudaMemset(locks, 0, (4*256+2)*sizeof(int32_t)));
    DGPP_CUDA_OK(cudaHostAlloc(&host, kPacket, cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&upload, cudaEventDisableTiming));
  }
  ~Impl() {
    if (uploaded) (void)cudaEventSynchronize(upload);
    for (void* p : std::array<void*,6>{fc1, act, tmp1, tmp2, locks, packet})
      if (p) (void)cudaFree(p);
    if (reused) (void)cudaFree(reused);
    if (host) (void)cudaFreeHost(host);
    if (upload) (void)cudaEventDestroy(upload);
#ifdef DGPP_HAVE_B12X_MX_NIBBLES
    if (nibble.module.module) (void)cudaLibraryUnload(nibble.module.module);
#endif
    if (nv.module.module) (void)cudaLibraryUnload(nv.module.module);
    if (mx.module.module) (void)cudaLibraryUnload(mx.module.module);
  }
};
Sm12xMoe::Sm12xMoe(int group, bool mx_nibbles) : impl_(std::make_unique<Impl>(group, mx_nibbles)) { impl_->initialize(); }
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
    const auto p = moe_fp4_packed_view(const_cast<uint8_t*>(addresses[e]), H, I, c.group, c.mx_nibbles);
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
  auto invoke = [&]<class Backend>(Backend& backend) {
    typename Backend::Fc1 fc1{c.fc1};
    typename Backend::Act act{c.act};
    typename Backend::Fc2 fc2{out};
    auto* packed = reinterpret_cast<int32_t*>(c.packet + E*6*sizeof(uint64_t));
    if (c.reused && rows == 4) {
      launch_mxfp4_verify_singletons(hidden, c.act, out, reinterpret_cast<const uint64_t*>(c.packet),
                                     ids, packed, c.reused, stream);
      packed = c.reused;
    }
    typename Backend::Routes routes{packed};
    typename Backend::Experts experts{packed+M*K*B};
    typename Backend::Count count{packed+M*K*(B+1)};
    typename Backend::Amax amax{c.tmp1};
    typename Backend::Tmp1 tmp1{c.tmp1};
    typename Backend::Tmp2 tmp2{c.tmp2};
    typename Backend::Locks locks{c.locks};
    // The generated C ABI does not express read-only pointer qualifiers.
    auto* input = const_cast<uint16_t*>(hidden);
    return Backend::launch(&backend.module, input, input, input,
        c.packet, c.packet+E*3*sizeof(uint64_t), int64_t(E)*2*I*H/8, int64_t(E)*H*I/8,
        &fc1, &act, &fc2, c.tmp1, c.tmp1, c.tmp1, c.tmp1, &routes, &experts, &count, &amax,
        0, const_cast<float*>(weights), &tmp1, &tmp2, &locks, c.tmp1, c.tmp1, c.tmp1,
        const_cast<int32_t*>(ids), c.tmp1, c.tmp1,
        E, 0, rows, 32, stream);
  };
  const int status =
#ifdef DGPP_HAVE_B12X_MX_NIBBLES
      c.mx_nibbles ? invoke(c.nibble) :
#endif
      c.group == 32 ? invoke(c.mx) : invoke(c.nv);
  if (status) throw std::runtime_error("b12x MoE AOT launch failed: " + std::to_string(status));
}
#else
struct Sm12xMoe::Impl {};
void Sm12xMoe::prepare_device(const uint64_t*, const int32_t*, const ExpertDemand*, int, cudaStream_t) {
  throw std::logic_error("b12x AOT is unavailable");
}
Sm12xMoe::Sm12xMoe(int, bool) { throw std::logic_error("b12x AOT is unavailable"); }
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
