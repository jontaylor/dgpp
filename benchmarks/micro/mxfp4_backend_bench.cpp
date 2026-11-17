// Isolate the shipped b12x backend at equal row counts and expert reuse.
// Prepared weights exceed L2; preparation is outside captured execution.
#include <algorithm>
#include <cstdio>
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
int main() {
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
