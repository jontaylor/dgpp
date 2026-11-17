// Isolate the shipped b12x backend at equal row counts and expert reuse.
// Prepared weights exceed L2; preparation is outside captured execution.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/moe_fp4_pack.hpp"
#include "kernels/sm12x_moe.hpp"
using namespace dgpp;
template <class T>
T* alloc(size_t n) {
  T* p;
  DGPP_CUDA_OK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}
// Each input line is one real four-row target pass at one layer (24 IDs).
static int replay(const char* path) {
  constexpr int H = 5120, I = 1152, E = 384, routes = 24, values = 4 * 7 * H;
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open routes");
  std::vector<std::array<int32_t, routes>> records;
  std::array<int32_t, routes> row;
  while (file >> row[0]) {
    for (int j = 1; j < routes; ++j)
      if (!(file >> row[j])) throw std::runtime_error("incomplete route group");
    for (int e : row)
      if (e < 0 || e >= E) throw std::runtime_error("invalid route");
    records.push_back(row);
  }
  if (records.empty()) throw std::runtime_error("empty route file");
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  auto* input = alloc<uint16_t>(4 * H);
  auto* ids = alloc<int32_t>(routes);
  auto* weights = alloc<float>(routes);
  auto* out = alloc<float>(values);
  std::mt19937 gen(17);
  std::vector<uint16_t> x(4 * H);
  for (auto& v : x) v = ((gen() & 1) << 15) | (0x3b80 + gen() % 1280);
  DGPP_CUDA_OK(cudaMemcpy(input, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  std::array<float, routes> prob;
  prob.fill(1.f / 6);
  DGPP_CUDA_OK(cudaMemcpy(weights, prob.data(), sizeof(prob), cudaMemcpyHostToDevice));
  const size_t bytes = moe_fp4_packed_bytes(H, I, 32);
  auto* storage = alloc<uint8_t>(bytes * E);
  std::vector<const uint8_t*> addresses;
  std::vector<uint8_t> data(bytes);
  for (auto& v : data) v = gen();
  // Distinct weights and scales at every matrix position. Rotate each expert's
  // source so mixing expert addresses or the order of K groups is observable.
  for (int e = 0; e < E; ++e) {
    auto* block = storage + bytes * e;
    addresses.push_back(block);
    auto v = moe_fp4_packed_view(block, H, I, 32);
    DGPP_CUDA_OK(cudaMemcpy(block, data.data(), bytes, cudaMemcpyHostToDevice));
    auto scales = [&](uint8_t* to, size_t n) {
      std::vector<uint8_t> exponents(n);
      for (auto& sc : exponents) sc = 115 + gen() % 11;
      DGPP_CUDA_OK(cudaMemcpy(to, exponents.data(), n, cudaMemcpyHostToDevice));
    };
    scales(v.s13, size_t(2 * I) * H / 32);
    scales(v.s2, size_t(H) * I / 32);
    std::rotate(data.begin(), data.begin() + 1024, data.end());
  }
  setenv("DGPP_MX_VERIFY_SINGLETONS", "0", 1);
  Sm12xMoe baseline(32);
  setenv("DGPP_MX_VERIFY_SINGLETONS", "1", 1);
  Sm12xMoe hybrid(32);
  cudaEvent_t a, b;
  DGPP_CUDA_OK(cudaEventCreate(&a));
  DGPP_CUDA_OK(cudaEventCreate(&b));
  std::puts("record,singletons,reused,baseline_us,split_us,singleton_nrmse,reused_nrmse");
  const int samples = std::min<int>(records.size(), 32);
  std::vector<size_t> selected;
  for (int i = 0; i < samples; ++i)
    selected.push_back(samples > 1 ? size_t(i) * (records.size() - 1) / (samples - 1) : 0);
  // Include the least and most overlapping real calls even if uniform
  // temporal subsampling missed both ends of the distribution.
  auto unique = [](const auto& ids) {
    std::array<bool, E> present{};
    for (int e : ids) present[e] = true;
    return std::count(present.begin(), present.end(), true);
  };
  const auto extremes =
      std::minmax_element(records.begin(), records.end(),
                          [&](const auto& a, const auto& b) { return unique(a) < unique(b); });
  for (auto it : {extremes.first, extremes.second}) {
    const size_t index = it - records.begin();
    if (std::find(selected.begin(), selected.end(), index) == selected.end())
      selected.push_back(index);
  }
  for (size_t i = 0; i < selected.size(); ++i) {
    const size_t record = selected[i];
    const auto& group = records[record];
    DGPP_CUDA_OK(cudaMemcpy(ids, group.data(), sizeof(group), cudaMemcpyHostToDevice));
    std::array<int, E> counts{};
    for (int e : group) ++counts[e];
    std::vector<float> ref(values), candidate(values);
    auto measure = [&](Sm12xMoe& moe, std::vector<float>& result) {
      moe.prepare(group, addresses, 4, stream);
      DGPP_CUDA_OK(cudaMemsetAsync(out, 0xff, values * sizeof(float), stream));
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      moe.run(input, ids, weights, out, 4, stream);
      DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
      DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, 0, 0, 0));
      for (int k = 0; k < 4; ++k) DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
      DGPP_CUDA_OK(cudaEventRecord(a, stream));
      for (int k = 0; k < 32; ++k) DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
      DGPP_CUDA_OK(cudaEventRecord(b, stream));
      DGPP_CUDA_OK(cudaEventSynchronize(b));
      float ms;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, a, b));
      DGPP_CUDA_OK(cudaMemcpy(result.data(), out, values * sizeof(float), cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaGraphExecDestroy(exec));
      DGPP_CUDA_OK(cudaGraphDestroy(graph));
      return ms * 1000 / 32;
    };
    float base_us, split_us;
    if (i % 2) {
      split_us = measure(hybrid, candidate);
      base_us = measure(baseline, ref);
    } else {
      base_us = measure(baseline, ref);
      split_us = measure(hybrid, candidate);
    }
    // Splitting the same groups across residency waves must preserve every
    // output bit of the hybrid backend, including its shared-slot sentinel.
    std::vector<int> waves[2];
    int ordinal = 0;
    for (int e = 0; e < E; ++e)
      if (counts[e]) waves[(ordinal++) % 2].push_back(e);
    DGPP_CUDA_OK(cudaMemsetAsync(out, 0xff, values*sizeof(float), stream));
    for (const auto& wave : waves) {
      if (wave.empty()) continue;
      hybrid.prepare(group, addresses, 4, stream, wave);
      hybrid.run(input, ids, weights, out, 4, stream);
    }
    std::vector<float> combined(values);
    DGPP_CUDA_OK(cudaMemcpyAsync(combined.data(), out, values*sizeof(float), cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    if (std::memcmp(combined.data(),candidate.data(),values*sizeof(float)) != 0)
      throw std::runtime_error("masked waves differ from one resident invocation");
    double error[2]{}, norm[2]{};
    for (int slot = 0; slot < 28; ++slot) {
      for (int col = 0; col < H; ++col) {
        const int j = slot * H + col;
        if (slot % 7 == 6) {
          if (!std::isnan(candidate[j])) throw std::runtime_error("shared slot overwritten");
          continue;
        }
        if (!std::isfinite(candidate[j]) || !std::isfinite(ref[j]))
          throw std::runtime_error("missing routed output");
        const int multiple = counts[group[(slot / 7) * 6 + slot % 7]] > 1;
        const double diff = candidate[j] - ref[j];
        error[multiple] += diff * diff;
        norm[multiple] += double(ref[j]) * ref[j];
      }
    }
    double rms[2];
    for (int g = 0; g < 2; ++g) rms[g] = std::sqrt(error[g] / std::max(norm[g], 1.e-30));
    std::printf("%zu,%ld,%ld,%.2f,%.2f,%.6f,%.6f\n", record,
                std::count(counts.begin(), counts.end(), 1),
                std::count_if(counts.begin(), counts.end(), [](int n) { return n > 1; }), base_us,
                split_us, rms[0], rms[1]);
    std::fflush(stdout);
    if (std::max(rms[0], rms[1]) > .02)
      throw std::runtime_error("split verification numerical mismatch");
  }
  DGPP_CUDA_OK(cudaFree(input));
  DGPP_CUDA_OK(cudaFree(ids));
  DGPP_CUDA_OK(cudaFree(out));
  DGPP_CUDA_OK(cudaFree(weights));
  DGPP_CUDA_OK(cudaFree(storage));
  DGPP_CUDA_OK(cudaEventDestroy(a));
  DGPP_CUDA_OK(cudaEventDestroy(b));
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
  return 0;
}
int main(int argc, char** argv) {
  if (argc == 2) return replay(argv[1]);
  if (argc != 1) throw std::invalid_argument("usage: mxfp4_backend_bench [four-row-route-file]");
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  constexpr int H = 5120, I = 1152, E = 384, K = 6;
  auto* input = alloc<uint16_t>(6 * H);
  auto* ids = alloc<int32_t>(36);
  auto* weights = alloc<float>(36);
  auto* slots = alloc<float>(6 * 7 * H);
  std::vector<uint16_t> x(6 * H, 0x3d80);
  std::vector<float> w(36, 1.f / 6);
  DGPP_CUDA_OK(cudaMemcpy(input, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(weights, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
  std::puts("group,rows,unique,median_us");
  for (int group : {16, 32}) {
    const size_t bytes = moe_fp4_packed_bytes(H, I, group);
    auto* storage = alloc<uint8_t>(bytes * E);
    std::vector<const uint8_t*> addresses(E);
    for (int e = 0; e < E; e++) {
      addresses[e] = storage + e * bytes;
      auto v = moe_fp4_packed_view(storage + e * bytes, H, I, group);
      DGPP_CUDA_OK(cudaMemset(v.w13, 0x12, size_t(I) * H));
      DGPP_CUDA_OK(cudaMemset(v.w2, 0x12, size_t(H) * I / 2));
      DGPP_CUDA_OK(cudaMemset(v.s13, group == 32 ? 121 : 24, size_t(2 * I) * H / group));
      DGPP_CUDA_OK(cudaMemset(v.s2, group == 32 ? 121 : 24, size_t(H) * I / group));
      if (group == 16) {
        float g[2] = {1, 1};
        DGPP_CUDA_OK(cudaMemcpy(v.g13, g, 8, cudaMemcpyHostToDevice));
        DGPP_CUDA_OK(cudaMemcpy(v.g2, g, 4, cudaMemcpyHostToDevice));
      }
    }
    Sm12xMoe moe(group);
    for (int rows : {4, 6})
      for (int unique : {6, 7, 10, 14, 19, 22, 24, 29, 36}) {
        if (unique > rows * K) continue;
        std::vector<int32_t> routes(rows * K);
        // Unique within each row, repeat across rows only. Spread IDs across storage.
        for (int t = 0; t < rows; t++)
          for (int j = 0; j < K; j++) routes[t * K + j] = ((t * K + j) % unique) * 7;
        DGPP_CUDA_OK(cudaMemcpy(ids, routes.data(), routes.size() * 4, cudaMemcpyHostToDevice));
        moe.prepare(routes, addresses, rows, stream);
        auto run = [&] { moe.run(input, ids, weights, slots, rows, stream); };
        run();
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int r = 0; r < 8; r++) run();
        DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
        DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, 0, 0, 0));
        for (int r = 0; r < 3; r++) DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        std::vector<float> times;
        for (int r = 0; r < 5; r++) {
          cudaEventRecord(a, stream);
          for (int j = 0; j < 8; j++) cudaGraphLaunch(exec, stream);
          cudaEventRecord(b, stream);
          cudaEventSynchronize(b);
          float ms;
          cudaEventElapsedTime(&ms, a, b);
          times.push_back(ms * 1000 / 64);
        }
        std::sort(times.begin(), times.end());
        std::printf("%d,%d,%d,%.3f\n", group, rows, unique, times[2]);
        std::fflush(stdout);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
      }
    cudaFree(storage);
  }
  cudaFree(input);
  cudaFree(ids);
  cudaFree(weights);
  cudaFree(slots);
  cudaStreamDestroy(stream);
}
