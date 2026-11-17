#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/mxfp8_gemm.hpp"
#include "mxfp4_native_prototype.cuh"
using namespace dgpp;
template <class T>
struct Buffer {
  T* p;
  explicit Buffer(size_t n) { DGPP_CUDA_OK(cudaMalloc(&p, n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
};
template <class F>
float timing(F f, cudaStream_t s) {
  f();
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  std::vector<float> ts;
  for (int r = 0; r < 5; r++) {
    cudaEventRecord(a, s);
    for (int j = 0; j < 3; j++) f();
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    ts.push_back(ms / 3);
  }
  std::sort(ts.begin(), ts.end());
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ts[2];
}
void measure(int tokens, int n, int k, cudaStream_t stream) {
  constexpr int E = 384, K = 6;
  const int rows = (tokens * K + E - 1) / E, total = rows * E;
  Buffer<uint16_t> x(size_t(total) * k);
  Buffer<uint8_t> a(size_t(total) * k), as(size_t(total) * k / 32), w(size_t(E) * n * k / 2),
      ws(size_t(E) * n * k / 32);
  Buffer<float> ref(size_t(total) * n), out(size_t(total) * n);
  Buffer<MoeSegment> segs(E);
  Buffer<MoeExpertView> views(E * 3);
  std::mt19937 rng(42);
  std::vector<uint16_t> hx(size_t(total) * k);
  for (auto& v : hx) v = float_to_bf16_bits((int(rng() % 2001) - 1000) / 1024.f);
  DGPP_CUDA_OK(cudaMemcpy(x.p, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  std::vector<uint8_t> hw(size_t(n) * k / 2);
  for (auto& v : hw) v = rng();
  std::vector<MoeSegment> hs(E);
  std::vector<MoeExpertView> hv(E * 3);
  for (int e = 0; e < E; e++) {
    DGPP_CUDA_OK(
        cudaMemcpy(w.p + size_t(e) * hw.size(), hw.data(), hw.size(), cudaMemcpyHostToDevice));
    hs[e] = {e * rows, rows, e};
    hv[e * 3].payload = w.p + size_t(e) * hw.size();
    hv[e * 3].fp4_scales = ws.p + size_t(e) * n * k / 32;
  }
  DGPP_CUDA_OK(cudaMemset(ws.p, 122, size_t(E) * n * k / 32));
  DGPP_CUDA_OK(
      cudaMemcpy(segs.p, hs.data(), hs.size() * sizeof(MoeSegment), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(
      cudaMemcpy(views.p, hv.data(), hv.size() * sizeof(MoeExpertView), cudaMemcpyHostToDevice));
  auto legacy = [&](int tile, float* dst) {
    launch_moe_grouped_mma_fp4_f32(x.p, k, segs.p, E, rows, 0, views.p, 0, dst, n, n, k, stream,
                                   nullptr, 32, tile);
  };
  legacy(64, ref.p);
  legacy(128, out.p);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  std::vector<float> h0(size_t(total) * n), h1(h0.size());
  cudaMemcpy(h0.data(), ref.p, h0.size() * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h1.data(), out.p, h1.size() * 4, cudaMemcpyDeviceToHost);
  if (h0 != h1) throw std::runtime_error("128-row output differs");
  auto quant = [&] {
    for (int first = 0; first < total; first += 65535) {
      const int count = std::min(65535, total - first);
      launch_mxfp8_quantize_rows(x.p + size_t(first) * k, k, count, k, a.p + size_t(first) * k,
                                 as.p + size_t(first) * k / 32, stream);
    }
  };
  auto native = [&] {
    mx4bench::run<float, 64, 128>(a.p, as.p, out.p, rows, E, n, k, stream, segs.p, views.p, 0);
  };
  quant();
  native();
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  cudaMemcpy(h1.data(), out.p, h1.size() * 4, cudaMemcpyDeviceToHost);
  double err = 0, norm = 0;
  for (size_t i = 0; i < h0.size(); i++) {
    double d = h1[i] - h0[i];
    err += d * d;
    norm += double(h0[i]) * h0[i];
  }
  double rel = sqrt(err / norm);
  if (!std::isfinite(rel) || rel > .06)
    throw std::runtime_error("native result outside quantization budget");
  float t64 = timing([&] { legacy(64, out.p); }, stream),
        t128 = timing([&] { legacy(128, out.p); }, stream),
        tn = timing(
            [&] {
              quant();
              native();
            },
            stream);
  printf("%d,%d,%d,%d,%.4f,%.4f,%.4f,%.6f\n", tokens, rows, n, k, t64, t128, tn, rel);
  fflush(stdout);
}
int main() {
  try {
    cudaStream_t s;
    cudaStreamCreate(&s);
    puts("tokens,rows_per_expert,n,k,bf16_64_ms,bf16_128_ms,native_with_quant_ms,relative_l2");
    for (int t : {4096, 24317}) {
      measure(t, 1152, 5120, s);
      measure(t, 5120, 1152, s);
    }
    cudaStreamDestroy(s);
  } catch (const std::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
