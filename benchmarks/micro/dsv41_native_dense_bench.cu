#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/fp8w_gemm.hpp"
#include "kernels/mxfp8_gemm.hpp"
using namespace dgpp;
template <class T>
struct Buffer {
  T* p;
  explicit Buffer(size_t n) { DGPP_CUDA_OK(cudaMalloc(&p, n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
};
template <class F>
float timing(F f, cudaStream_t s) {
  for (int i = 0; i < 3; i++) f();
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  std::vector<float> ts;
  for (int r = 0; r < 5; r++) {
    cudaEventRecord(a, s);
    for (int i = 0; i < 10; i++) f();
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    ts.push_back(ms / 10);
  }
  std::sort(ts.begin(), ts.end());
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ts[2];
}
void measure(int m, int n, int k, cudaStream_t s) {
  Buffer<uint16_t> x(size_t(m) * k);
  Buffer<uint8_t> w(size_t(n) * k), ws(size_t((n + 31) / 32) * (k / 32)),
      expanded(size_t(n) * k / 32), scratch(size_t(m) * k * 2);
  Buffer<float> ref(size_t(m) * n), out(size_t(m) * n), baseline(size_t(m) * n);
  std::mt19937 rng(43);
  std::vector<uint16_t> hx(size_t(m) * k);
  std::vector<uint8_t> hw(size_t(n) * k), hs(size_t((n + 31) / 32) * k / 32),
      he(size_t(n) * k / 32);
  for (auto& v : hx) v = float_to_bf16_bits((int(rng() % 2001) - 1000) / 1024.f);
  for (auto& v : hw) v = (rng() % 80) | ((rng() % 2) * 128);
  for (auto& v : hs) v = 120 + rng() % 5;
  for (int r = 0; r < n; r++)
    for (int c = 0; c < k / 32; c++)
      he[size_t(r) * (k / 32) + c] = hs[size_t(r / 32) * (k / 32) + c];
  cudaMemcpy(x.p, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(w.p, hw.data(), hw.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(ws.p, hs.data(), hs.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(expanded.p, he.data(), he.size(), cudaMemcpyHostToDevice);
  auto native = [&] {
    if (!launch_mxfp8_grid32_f32(x.p, k, w.p, ws.p, out.p, m, n, k, s, n, scratch.p,
                                 size_t(m) * k * 2))
      throw std::runtime_error("native path not selected");
  };
  auto old = [&] { launch_fp8w_gemm_e8m0_f32(x.p, k, w.p, ws.p, baseline.p, m, n, k, s, n, 5, 5); };
  native();
  launch_mxfp8_gemm_f32(scratch.p, scratch.p + size_t(m) * k, w.p, expanded.p, ref.p, m, n, k, s,
                        n);
  old();
  cudaStreamSynchronize(s);
  std::vector<float> ho(size_t(m) * n), hr(ho.size()), hb(ho.size());
  cudaMemcpy(ho.data(), out.p, ho.size() * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hr.data(), ref.p, hr.size() * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), baseline.p, hb.size() * 4, cudaMemcpyDeviceToHost);
  if (ho != hr) throw std::runtime_error("grid32 differs from expanded-scale native oracle");
  double e = 0, v = 0;
  for (size_t i = 0; i < ho.size(); i++) {
    double d = ho[i] - hb[i];
    e += d * d;
    v += double(hb[i]) * hb[i];
  }
  double rel = sqrt(e / v);
  if (!std::isfinite(rel) || rel > .06)
    throw std::runtime_error("native quantization error exceeds budget");
  printf("%d,%d,%d,%.4f,%.4f,%.6f\n", m, n, k, timing(old, s), timing(native, s), rel);
  fflush(stdout);
}
int main() {
  try {
    cudaStream_t s;
    cudaStreamCreate(&s);
    puts("rows,n,k,bf16_ms,native_including_quant_ms,relative_l2");
    for (int m : {128, 4096})
      for (auto shape : std::vector<std::pair<int, int>>{
               {512, 5120}, {1024, 5120}, {4096, 5120}, {5120, 1024}, {5120, 1152}})
        measure(m, shape.first, shape.second, s);
    cudaStreamDestroy(s);
  } catch (const std::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
