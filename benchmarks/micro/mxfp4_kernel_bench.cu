#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/moe_fp4_pack.hpp"
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
void measure(int tokens, int n, int k, cudaStream_t stream, const std::vector<int>& routes = {}) {
  constexpr int E = 384, K = 6;
  std::vector<std::vector<int32_t>> by_expert(E);
  if (routes.empty()) {
    const int per = (tokens * K + E - 1) / E;
    for (int e = 0; e < E; e++)
      for (int r = 0; r < per; r++) by_expert[e].push_back((e * per + r) % tokens);
  } else {
    for (size_t i = 0; i < routes.size(); i++) by_expert.at(routes[i]).push_back(i / K);
  }
  int rows = 0;
  std::vector<int32_t> rowmap;
  for (const auto& v : by_expert) {
    rows = std::max(rows, int(v.size()));
    rowmap.insert(rowmap.end(), v.begin(), v.end());
  }
  const int total = rowmap.size();
  // Gate/up share token activations; down consumes distinct expert SwiGLU rows.
  const int input_rows = k == 1152 ? total : tokens;
  if (k == 1152)
    for (int r = 0; r < total; ++r) rowmap[r] = r;
  Buffer<int32_t> actrows(total);
  DGPP_CUDA_OK(cudaMemcpy(actrows.p, rowmap.data(), total * 4, cudaMemcpyHostToDevice));
  Buffer<uint16_t> x(size_t(input_rows) * k);
  Buffer<uint8_t> a(size_t(input_rows) * k), as(size_t(input_rows) * k / 32),
      w(size_t(E) * n * k / 2), ws(size_t(E) * n * k / 32);
  Buffer<float> ref(size_t(total) * n), out(size_t(total) * n);
  Buffer<MoeSegment> segs(E);
  Buffer<MoeExpertView> views(E * 3), column_views(E * 3);
  Buffer<uint8_t> column_ws(size_t(E) * n * k / 32);
  std::mt19937 rng(42);
  std::vector<uint16_t> hx(size_t(input_rows) * k);
  for (auto& v : hx) v = float_to_bf16_bits((int(rng() % 2001) - 1000) / 1024.f);
  DGPP_CUDA_OK(cudaMemcpy(x.p, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  std::vector<uint8_t> hw(size_t(n) * k / 2);
  for (auto& v : hw) v = rng();
  std::vector<MoeSegment> hs(E);
  std::vector<MoeExpertView> hv(E * 3);
  for (int e = 0; e < E; e++) {
    DGPP_CUDA_OK(
        cudaMemcpy(w.p + size_t(e) * hw.size(), hw.data(), hw.size(), cudaMemcpyHostToDevice));
    hs[e] = {e ? hs[e - 1].row0 + hs[e - 1].rows : 0, int(by_expert[e].size()), e};
    hv[e * 3].payload = w.p + size_t(e) * hw.size();
    hv[e * 3].fp4_scales = ws.p + size_t(e) * n * k / 32;
  }
  // Nonuniform scales also exercise staged/transposed indexing correctness.
  std::vector<uint8_t> hsrow(size_t(n) * k / 32), hscol(hsrow.size());
  for (int r = 0; r < n; ++r)
    for (int g = 0; g < k / 32; ++g) {
      const uint8_t value = 120 + rng() % 5;
      hsrow[size_t(r) * (k / 32) + g] = value;
      hscol[size_t(g) * n + r] = value;
    }
  auto hvc = hv;
  for (int e = 0; e < E; ++e) {
    DGPP_CUDA_OK(cudaMemcpy(ws.p + size_t(e) * hsrow.size(), hsrow.data(), hsrow.size(),
                            cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(column_ws.p + size_t(e) * hscol.size(), hscol.data(), hscol.size(),
                            cudaMemcpyHostToDevice));
    hvc[e * 3].fp4_scales = column_ws.p + size_t(e) * hscol.size();
  }
  DGPP_CUDA_OK(cudaMemcpy(column_views.p, hvc.data(), hvc.size() * sizeof(MoeExpertView),
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(
      cudaMemcpy(segs.p, hs.data(), hs.size() * sizeof(MoeSegment), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(
      cudaMemcpy(views.p, hv.data(), hv.size() * sizeof(MoeExpertView), cudaMemcpyHostToDevice));
  auto legacy = [&](int tile, float* dst) {
    launch_moe_grouped_mma_fp4_f32(x.p, k, segs.p, E, rows, 0, views.p, 0, dst, n, n, k, stream,
                                   actrows.p, 32, tile);
  };
  legacy(64, ref.p);
  legacy(128, out.p);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  std::vector<float> h0(size_t(total) * n), h1(h0.size());
  cudaMemcpy(h0.data(), ref.p, h0.size() * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h1.data(), out.p, h1.size() * 4, cudaMemcpyDeviceToHost);
  if (h0 != h1) throw std::runtime_error("128-row output differs");
  const char* prepared = std::getenv("DGPP_MXFP4_BENCH_PREPARED");
  if (prepared && prepared[0] == '1') {
    constexpr int hidden = 5120, intermediate = 1152;
    const size_t bytes = moe_fp4_packed_bytes(hidden, intermediate, 32);
    Buffer<uint8_t> pw(bytes * E);
    Buffer<MoeExpertView> pv(E * 3);
    auto hpv = hv;
    for (int e = 0; e < E; ++e) {
      auto dst = moe_fp4_packed_view(pw.p + bytes * e, hidden, intermediate, 32);
      Fp4ExpertSource source;
      for (int q = 0; q < 3; ++q) {
        source.payload[q] = hv[e * 3].payload;
        source.scales[q] = hv[e * 3].fp4_scales;
      }
      launch_moe_fp4_pack(dst, source, hidden, intermediate, stream, 32);
      hpv[e * 3].payload = k == intermediate ? dst.w2 : dst.w13;
      hpv[e * 3].fp4_scales = k == intermediate ? dst.s2 : dst.s13;
      hpv[e * 3].fp4_packed_rows = k == intermediate ? hidden : 2 * intermediate;
    }
    DGPP_CUDA_OK(cudaMemcpyAsync(pv.p, hpv.data(), hpv.size() * sizeof(MoeExpertView),
                                 cudaMemcpyHostToDevice, stream));
    auto prepared_run = [&] {
      launch_moe_grouped_mma_fp4_f32(x.p, k, segs.p, E, rows, 0, pv.p, 0, out.p, n, n, k, stream,
                                     actrows.p, 32, 64);
    };
    prepared_run();
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    DGPP_CUDA_OK(cudaMemcpy(h1.data(), out.p, h1.size() * 4, cudaMemcpyDeviceToHost));
    if (h1 != h0) throw std::runtime_error("prepared layout changed BF16 accumulators");
    printf("prepared-bf16,%d,%d,%d,%.4f\n", tokens, n, k, timing(prepared_run, stream));
    fflush(stdout);
    launch_mxfp8_quantize_rows(x.p, k, input_rows, k, a.p, as.p, stream);
    mx4bench::run<float, 64, 128, true, true, true, 2, true, false, 24>(
        a.p, as.p, out.p, rows, E, n, k, stream, segs.p, views.p, 0, actrows.p);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    std::vector<float> native_ref(h0.size()), native_got(h0.size());
    DGPP_CUDA_OK(cudaMemcpy(native_ref.data(), out.p, h0.size() * 4, cudaMemcpyDeviceToHost));
    auto prepared_native = [&]<int M, int S, int Schedule = 24>() {
      auto run = [&] {
        mx4bench::run<float, M, 128, true, true, true, S, true, false, Schedule, true>(
            a.p, as.p, out.p, rows, E, n, k, stream, segs.p, pv.p, 0, actrows.p);
      };
      run();
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      DGPP_CUDA_OK(cudaMemcpy(native_got.data(), out.p, h0.size() * 4, cudaMemcpyDeviceToHost));
      if (native_got != native_ref) throw std::runtime_error("prepared native arithmetic mismatch");
      printf("prepared-native,%d,%d,%d,%d,%d,%d,%.4f\n", tokens, n, k, M, S, Schedule,
             timing(run, stream));
      fflush(stdout);
    };
    prepared_native.template operator()<64, 2>();
    prepared_native.template operator()<64, 3>();
    prepared_native.template operator()<128, 2>();
    prepared_native.template operator()<128, 3>();
    prepared_native.template operator()<64, 2, 27>();
    prepared_native.template operator()<128, 2, 27>();
  }
  const int tile_capacity = moe_tile_list_capacity(E, total, 64);
  Buffer<MoeTile> work(moe_tile_list_capacity(E, total, 32));
  Buffer<int32_t> work_count(1);
  auto compact_legacy = [&] {
    launch_moe_tile_list(segs.p, E, 64, work.p, work_count.p, stream);
    launch_moe_grouped_mma_fp4_f32(x.p, k, segs.p, E, rows, 0, views.p, 0, out.p, n, n, k, stream,
                                   actrows.p, 32, 64, work.p, work_count.p, tile_capacity);
  };
  compact_legacy();
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  DGPP_CUDA_OK(cudaMemcpy(h1.data(), out.p, h1.size() * 4, cudaMemcpyDeviceToHost));
  if (h1 != h0) throw std::runtime_error("compact BF16 changed accumulators");
  printf("compact-bf16,%d,%d,%d,%.4f\n", tokens, n, k, timing(compact_legacy, stream));
  fflush(stdout);
  auto quant = [&] {
    for (int first = 0; first < input_rows; first += 65535) {
      const int count = std::min(65535, input_rows - first);
      launch_mxfp8_quantize_rows(x.p + size_t(first) * k, k, count, k, a.p + size_t(first) * k,
                                 as.p + size_t(first) * k / 32, stream);
    }
  };
  printf("quant-only,%d,%d,%d,%.4f\n", tokens, n, k, timing(quant, stream));
  fflush(stdout);
  auto native = [&] {
    mx4bench::run<float, 64, 128>(a.p, as.p, out.p, rows, E, n, k, stream, segs.p, views.p, 0,
                                  actrows.p);
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
  auto packed = [&]<int BM, int BN, bool MatrixLoads = false, bool WideWarps = false,
                    int Stages = 3, bool StageScales = false, bool Compact = false,
                    bool ColumnScales = false, int Schedule = 0>() {
    auto run = [&] {
      if constexpr (Compact) launch_moe_tile_list(segs.p, E, BM, work.p, work_count.p, stream);
      mx4bench::run<float, BM, BN, true, MatrixLoads, WideWarps, Stages, StageScales, ColumnScales,
                    Schedule>(a.p, as.p, out.p, rows, E, n, k, stream, segs.p,
                              ColumnScales ? column_views.p : views.p, 0, actrows.p,
                              Compact ? work.p : nullptr, work_count.p,
                              moe_tile_list_capacity(E, total, BM));
    };
    run();
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    std::vector<float> hp(h1.size());
    DGPP_CUDA_OK(cudaMemcpy(hp.data(), out.p, hp.size() * 4, cudaMemcpyDeviceToHost));
    if (hp != h1) throw std::runtime_error("packed pipeline changed native accumulators");
    float kernel = timing(run, stream);
    float all = timing(
        [&] {
          quant();
          run();
        },
        stream);
    printf("packed,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f\n", tokens, n, k, BM, BN,
           int(MatrixLoads), int(WideWarps), Stages, int(StageScales), int(Compact),
           int(ColumnScales), Schedule, kernel, all);
    fflush(stdout);
  };
  const char* ablations = std::getenv("DGPP_MXFP4_BENCH_ABLATIONS");
  if (ablations && ablations[0] == '1') {
    // Causal ablations: one change at a time, then cumulative combinations.
    packed.template operator()<64, 128, true, true, 2, true>();
    packed.template operator()<64, 128, true, true, 2, true, false, false, 1>();
    packed.template operator()<64, 128, true, true, 2, true, false, false, 2>();
    packed.template operator()<64, 128, true, true, 2, true, false, false, 4>();
    packed.template operator()<64, 128, true, true, 2, true, false, false, 8>();
    packed.template operator()<64, 128, true, true, 2, true, false, false, 24>();
    packed.template operator()<64, 128, true, true, 2, true, false, false, 3>();
    packed.template operator()<64, 128, true, true, 2, true, false, false, 7>();
    packed.template operator()<64, 128, true, true, 2, true, false, false, 15>();
  } else {
    packed.template operator()<64, 128>();

    packed.template operator()<64, 128, true, true>();
    packed.template operator()<64, 128, true, true, 2>();
    packed.template operator()<64, 128, true, false, 2>();
    packed.template operator()<64, 128, true, true, 2, true>();
    packed.template operator()<64, 128, true, true, 3, true>();
    packed.template operator()<64, 128, true, true, 2, true, true>();
    packed.template operator()<64, 128, true, true, 3, true, true>();
    packed.template operator()<32, 256, true, true, 2, true, true>();
    packed.template operator()<32, 256, true, true, 3, true, true>();
    // Smaller tiles reduce per-block register pressure and raise occupancy.
    packed.template operator()<64, 64, true, true, 2, true, true>();
    packed.template operator()<64, 64, true, true, 3, true, true>();
    packed.template operator()<32, 128, true, true, 2, true, true>();
    packed.template operator()<32, 128, true, false, 2, true, true>();
    packed.template operator()<64, 128, true, true, 2, true, true, true>();
    packed.template operator()<64, 128, true, true, 3, true, true, true>();
  }
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
int main(int argc, char** argv) {
  try {
    cudaStream_t s;
    cudaStreamCreate(&s);
    puts("tokens,rows_per_expert,n,k,bf16_64_ms,bf16_128_ms,native_with_quant_ms,relative_l2");
    std::vector<int> routes;
    if (argc >= 2) {
      std::ifstream f(argv[1]);
      int e;
      while (f >> e) routes.push_back(e);
      if (routes.empty() || routes.size() % 6) throw std::runtime_error("bad routes file");
    }
    for (int t : (routes.empty() ? std::vector<int>{4096, 24317}
                                 : std::vector<int>{int(routes.size() / 6)})) {
      const std::string projection = argc >= 3 ? argv[2] : "both";
      if (projection != "both" && projection != "gate" && projection != "down")
        throw std::runtime_error("projection must be both, gate or down");
      if (projection != "down") measure(t, 1152, 5120, s, routes);
      if (projection != "gate") measure(t, 5120, 1152, s, routes);
    }
    cudaStreamDestroy(s);
  } catch (const std::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
