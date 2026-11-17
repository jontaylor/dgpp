// TP2 target verification, four independent rows and synthetic grid32 FP8 weights.
// Compare graph replays on a rotating working set larger than the device L2.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/scale_gemm.hpp"

namespace {
using namespace dgpp;
constexpr int rows = 4;
constexpr size_t workspace_bytes = 16U << 20;

template <class T>
struct Device {
  T* p = nullptr;
  explicit Device(size_t count) { DGPP_CUDA_OK(cudaMalloc(&p, count * sizeof(T))); }
  Device(const Device&) = delete;
  ~Device() { cudaFree(p); }
};

__device__ unsigned mix(unsigned i) {
  i ^= i >> 16;
  i *= 0x21f0aaadU;
  i ^= i >> 15;
  return i * 0x735a2d97U;
}
__global__ void fill(uint8_t* w, int count, float* scales, int ns, uint16_t* act, int nx) {
  int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    unsigned v = mix(i);
    w[i] = (v >> 24 & 128U) | (v % 96U);
  }
  if (i < ns) scales[i] = ldexpf(1.f, int(mix(i + 1) % 7U) - 5);
  if (i < nx) act[i] = float_to_bf16_bits((int(mix(i + 2) % 2049U) - 1024) / 1024.f);
}

struct Shape {
  const char* name;
  int groups, n, k;
  bool joined_input, incumbent_pair, incumbent_ws;
  int second_n = 0;
};

struct Arm {
  const char* name;
  bool pair, workspace, gemv;
  int warps;
};

void measure(const Shape& shape, int l2, cudaStream_t stream) {
  int total_n = shape.second_n ? shape.n + shape.second_n : shape.groups * shape.n;
  int xs = shape.joined_input ? shape.k : shape.groups * shape.k;
  int os = total_n;
  size_t matrix = size_t(total_n) * shape.k;
  size_t scale_matrix = matrix / 1024;
  int copies = std::max(3, int((size_t(l2) * 4 + matrix - 1) / matrix));
  size_t elements = matrix * copies;
  Device<uint8_t> weights(elements), scratch(workspace_bytes);
  Device<float> scales(scale_matrix * copies);
  Device<uint16_t> x(rows * xs), y(rows * os);
  fill<<<(elements + 255) / 256, 256, 0, stream>>>(weights.p, int(elements), scales.p,
                                                   int(scale_matrix * copies), x.p, rows * xs);
  DGPP_CUDA_OK(cudaGetLastError());
  const int control_width = shape.n == 25600 ? 8 : 0;
  std::vector<Arm> arms{{"control", shape.incumbent_pair, shape.incumbent_ws, false, control_width},
                        {"gemv", false, false, true, 0}};
  for (int w : {2, 4, 8}) {
    for (bool ws : {false, true}) {
      static const char* labels[3][2] = {{"w2", "w2-ws"}, {"w4", "w4-ws"}, {"w8", "w8-ws"}};
      arms.push_back({labels[w / 4][ws], false, ws, false, w});
    }
  }
  if (shape.groups > 1) arms.push_back({"pair", true, true, false, 0});

  auto launch = [&](const Arm& a, int copy, int first, int m) {
    void* ws = a.workspace ? scratch.p : nullptr;
    const auto* input = x.p + first * xs;
    auto* w = weights.p + copy * matrix;
    auto* sc = scales.p + copy * scale_matrix;
    for (int g = 0; g < shape.groups;) {
      auto* in0 = input + (shape.joined_input ? 0 : g * shape.k);
      size_t offset = size_t(g) * shape.n * shape.k;
      int n0 = g && shape.second_n ? shape.second_n : shape.n;
      // Query and KV use separate compact outputs. wo_a writes columns of
      // the same four-group row in all groups.
      int stride0 = shape.second_n ? n0 : os;
      auto* out0 = y.p + (shape.second_n ? g * rows * shape.n : g * shape.n) + first * stride0;
      if (a.pair) {
        int n1 = shape.second_n ? shape.second_n : n0;
        size_t next = offset + size_t(n0) * shape.k;
        auto* in1 = in0 + (shape.joined_input ? 0 : shape.k);
        if (!mma_gemv_pair_ok(w + offset, w + next, in0, in1, xs, m, n0, n1, shape.k, ws,
                              workspace_bytes))
          throw std::runtime_error("unsupported projection pair");
        int stride1 = shape.second_n ? n1 : os;
        auto* out1 = shape.second_n ? y.p + rows * n0 + first * stride1 : out0 + n0;
        launch_mma_gemv_fp8_bf16_pair(in0, w + offset, sc + offset / 1024, out0, n0, stride0, in1,
                                      w + next, sc + next / 1024, out1, n1, stride1, xs, m, shape.k,
                                      5, 5, stream, ws, workspace_bytes);
        g += 2;
      } else {
        launch_scale_gemm_grid_bf16(in0, xs, w + offset, sc + offset / 1024, out0, m, n0, shape.k,
                                    stream, stride0, 5, 5, !a.gemv, ws, workspace_bytes, false);
        ++g;
      }
    }
  };
  auto result = [&] {
    std::vector<uint16_t> host(rows * os);
    DGPP_CUDA_OK(
        cudaMemcpyAsync(host.data(), y.p, host.size() * 2, cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    return host;
  };
  std::vector<uint16_t> reference;
  std::vector<cudaGraphExec_t> graphs(arms.size());
  std::vector<double> errors(arms.size());
  for (size_t i = 0; i < arms.size(); ++i) {
    const auto& arm = arms[i];
    mma_gemv_set_decode_width(arm.warps);
    launch(arm, 0, 0, rows);
    auto output = result();
    if (!i) reference = output;
    double err = 0, norm = 0;
    for (size_t j = 0; j < output.size(); ++j) {
      double b = bf16_bits_to_float(reference[j]), d = bf16_bits_to_float(output[j]) - b;
      err += d * d;
      norm += b * b;
    }
    errors[i] = std::sqrt(err / norm);
    if (!(errors[i] < 0.01)) throw std::runtime_error("projection outside reference tolerance");
    DGPP_CUDA_OK(cudaMemsetAsync(y.p, 0xfe, rows * os * 2, stream));
    for (int r = 0; r < rows; ++r) launch(arm, 0, r, 1);
    if (output != result()) throw std::runtime_error("batched and standalone rows diverged");
    cudaGraph_t graph;
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int c = 0; c < copies; ++c) launch(arm, c, 0, rows);
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&graphs[i], graph, 0));
    DGPP_CUDA_OK(cudaGraphDestroy(graph));
    DGPP_CUDA_OK(cudaGraphLaunch(graphs[i], stream));
  }
  cudaEvent_t start, end;
  DGPP_CUDA_OK(cudaEventCreate(&start));
  DGPP_CUDA_OK(cudaEventCreate(&end));
  std::vector<std::vector<float>> samples(arms.size());
  for (size_t r = 0; r < 9; ++r) {
    for (size_t j = 0; j < arms.size(); ++j) {
      size_t i = (j + r) % arms.size();
      DGPP_CUDA_OK(cudaEventRecord(start, stream));
      for (int rep = 0; rep < 5; ++rep) DGPP_CUDA_OK(cudaGraphLaunch(graphs[i], stream));
      DGPP_CUDA_OK(cudaEventRecord(end, stream));
      DGPP_CUDA_OK(cudaEventSynchronize(end));
      float ms = 0;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, start, end));
      samples[i].push_back(ms * 1000 / (copies * 5));
    }
  }
  for (size_t i = 0; i < arms.size(); ++i) {
    auto& v = samples[i];
    std::sort(v.begin(), v.end());
    std::printf("%s,%s,%d,%.2f,%.2f,%.2f,%.7f\n", shape.name, arms[i].name, copies, v[2], v[4],
                v[6], errors[i]);
    DGPP_CUDA_OK(cudaGraphExecDestroy(graphs[i]));
  }
  std::fflush(stdout);
  DGPP_CUDA_OK(cudaEventDestroy(start));
  DGPP_CUDA_OK(cudaEventDestroy(end));
}
}  // namespace

int main() {
  try {
    cudaDeviceProp prop{};
    DGPP_CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::puts("projection,variant,weight_copies,p25_us,median_us,p75_us,rel_l2");
    for (const auto& shape : std::vector<Shape>{{"wo_a", 4, 1024, 4096, false, false, true},
                                                {"wo_b", 1, 5120, 4096, true, false, false},
                                                {"q_b", 1, 16384, 1280, true, false, false},
                                                {"q_a_kv", 2, 1280, 5120, true, true, true, 512},
                                                {"idx_q", 1, 4096, 1280, true, false, false},
                                                {"engram", 1, 25600, 3072, true, false, false}})
      measure(shape, prop.l2CacheSize, stream);
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
