// TP2 target verification, four independent rows and synthetic grid32 FP8 weights.
// Compare graph replays on a rotating working set larger than the device L2.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/csa2.hpp"
#include "kernels/dsv41_dspark.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/mxfp8_gemm.hpp"
#include "kernels/scale_gemm.hpp"

namespace {
using namespace dgpp;
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
  bool mxfp8_input = false;
  bool residual = false;
  bool exact_wob = false;
};

void measure(const Shape& shape, int l2, cudaStream_t stream, bool native_wob = false, int rows = 4) {
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

  if (native_wob) {
    arms.resize(1);
    arms.push_back({"exact-split", false, true, false, 0, false, false, true});
    arms.push_back({"mxfp8-input", false, false, false, 0, true});
    arms.push_back({"mxfp8-residual", false, false, false, 0, true, true});
  }
  auto launch = [&](const Arm& a, int copy, int first, int m) {
    void* ws = a.workspace ? scratch.p : nullptr;
    const auto* input = x.p + first * xs;
    auto* w = weights.p + copy * matrix;
    auto* sc = scales.p + copy * scale_matrix;
    if (a.exact_wob) {
      if (!try_launch_wo_b_split_bf16(input, w, sc, y.p + first * os, m, shape.n, shape.k,
                                      ws, workspace_bytes, stream))
        throw std::runtime_error("exact wo_b requires the two-split TP2 shape");
      return;
    }
    if (a.mxfp8_input) {
      launch_mxfp8_wob_decode(input, w, sc, y.p + first * os, m, shape.n, scratch.p, a.residual, stream);
      return;
    }
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
    if (arm.exact_wob && output != reference) throw std::runtime_error("exact wo_b differs from split-K");
    double err = 0, norm = 0;
    for (size_t j = 0; j < output.size(); ++j) {
      double b = bf16_bits_to_float(reference[j]), d = bf16_bits_to_float(output[j]) - b;
      err += d * d;
      norm += b * b;
    }
    errors[i] = std::sqrt(err / norm);
    if (!(errors[i] < (arm.residual ? 0.005 : arm.mxfp8_input ? 0.05 : 0.01))) throw std::runtime_error("projection outside reference tolerance");
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

__global__ void fill_partials(float* c, float* m, float* l, float* sink, float* freq, int64_t* pos,
                              int entries, int heads, int splits) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < entries * 512) c[i] = float(int(mix(i + 13) % 4097) - 2048) / 1024.f;
  if (i < entries) {
    m[i] = float(int(mix(i + 3) % 17) - 8) / 8.f;
    l[i] = 1.f + float(i % 5);
  }
  if (i < heads) sink[i] = 0.1f;
  if (i < 32) freq[i] = powf(10000.f, -i / 32.f);
  if (i < entries / heads / splits) pos[i] = i == 2 ? -1 : 24317 + i;
}

void woa_handoff(int l2, cudaStream_t stream, int rows, int main_splits, int window_splits) {
  if (rows < 1 || rows > 8 || main_splits < 0 || main_splits > 127 ||
      window_splits < 1 || window_splits > 128 - main_splits)
    throw std::invalid_argument("wo_a handoff requires 1..8 rows and at most 128 main and window splits");
  constexpr int groups = 4, n = 1024, k = 4096, heads = 32;
  const int splits = main_splits + window_splits;
  const int stride = groups * k, values = rows * stride, entries = rows * heads * splits;
  constexpr size_t matrix = size_t(groups) * n * k;
  const int copies = std::max(3, int(4 * size_t(l2) / matrix) + 1);
  Device<uint8_t> weights(matrix * copies), packed(values + values / 32), scratch(workspace_bytes);
  Device<float> scales(matrix / 1024 * copies), c(entries * 512), m(entries), l(entries),
      sink(heads), freq(32);
  Device<int64_t> pos(rows);
  Device<uint16_t> attention(values), y(rows * groups * n);
  fill<<<(matrix * copies + 255) / 256, 256, 0, stream>>>(
      weights.p, matrix * copies, scales.p, matrix / 1024 * copies, attention.p, values);
  fill_partials<<<(entries * 512 + 255) / 256, 256, 0, stream>>>(c.p, m.p, l.p, sink.p, freq.p,
                                                                 pos.p, entries, heads, splits);
  auto finish = [&](uint16_t* out, uint8_t* codes) {
    const int window = rows * heads * main_splits;
    csa2_attn_finish(m.p, l.p, c.p, main_splits, m.p + window, l.p + window, c.p + window * 512,
                     window_splits, sink.p, rows, heads, pos.p, freq.p, out, stream, codes);
  };
  auto launch = [&](int variant, int copy) {
    finish(variant ? nullptr : attention.p, variant ? packed.p : nullptr);
    auto* w = weights.p + copy * matrix;
    auto* ws = scales.p + copy * matrix / 1024;
    if (variant) {
      launch_mxfp8_woa_decode(packed.p, w, ws, y.p, rows, groups, n, k, stream);
    } else {
      for (int g = 0; g < groups; g += 2) {
        const size_t off = size_t(g) * n * k;
        launch_mma_gemv_fp8_bf16_pair(attention.p + g * k, w + off, ws + off / 1024, y.p + g * n, n,
                                      groups * n, attention.p + (g + 1) * k, w + off + n * k,
                                      ws + (off + n * k) / 1024, y.p + (g + 1) * n, n, groups * n,
                                      stride, rows, k, 5, 5, stream, scratch.p, workspace_bytes);
      }
    }
  };
  auto read = [&](auto* p, size_t count) {
    std::vector<std::remove_pointer_t<decltype(p)>> data(count);
    DGPP_CUDA_OK(
        cudaMemcpyAsync(data.data(), p, count * sizeof(*p), cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    return data;
  };
  launch(0, 0);
  auto ref = read(attention.p, values);
  finish(attention.p, packed.p);
  if (ref != read(attention.p, values)) throw std::runtime_error("attention finish changed");
  auto q = read(packed.p, values + values / 32);
  DGPP_CUDA_OK(cudaMemsetAsync(packed.p, 0xa5, q.size(), stream));
  launch(1, 0);
  if (q != read(packed.p, q.size())) throw std::runtime_error("omitting the BF16 output changed MXFP8");
  auto weight = read(weights.p, matrix);
  auto ws = read(scales.p, matrix / 1024);
  auto out = read(y.p, rows * groups * n);
  // The public finish and projection interfaces also accept sliced output and scale buffers.
  {
    Device<uint16_t> sliced_attention(values + 1);
    Device<float> sliced_scales(matrix / 1024 + 1);
    finish(sliced_attention.p + 1, packed.p);
    DGPP_CUDA_OK(cudaMemcpyAsync(sliced_scales.p + 1, scales.p, ws.size() * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream));
    launch_mxfp8_woa_decode(packed.p, weights.p, sliced_scales.p + 1, y.p, rows, groups, n, k, stream);
    if (read(sliced_attention.p + 1, values) != ref || read(y.p, out.size()) != out)
      throw std::runtime_error("sliced buffers changed wo_a handoff");
  }
  for (int i = 0; i < values; i += 32) {
    float amax = 1e-4f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::abs(bf16_bits_to_float(ref[i + j])));
    int exponent = int(std::ceil(std::log2(amax / 448.f)));
    if (q[values + i / 32] != exponent + 127) throw std::runtime_error("K32 scale mismatch");
    for (int j = 0; j < 32; ++j)
      if (q[i + j] != float_to_fp8_e4m3_bits(std::ldexp(bf16_bits_to_float(ref[i + j]), -exponent)))
        throw std::runtime_error("E4M3 mismatch");
  }
  double err = 0, norm = 0;
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < groups * n; col += 13) {
      double sum = 0;
      for (int kk = 0; kk < k; ++kk) {
        size_t i = (size_t(row) * groups + col / n) * k + kk;
        sum += std::ldexp(fp8_e4m3_bits_to_float(q[i]), int(q[values + i / 32]) - 127) *
               fp8_e4m3_bits_to_float(weight[size_t(col) * k + kk]) *
               ws[size_t(col / 32) * (k / 32) + kk / 32];
      }
      const double diff = bf16_bits_to_float(out[row * groups * n + col]) - sum;
      err += diff * diff;
      norm += sum * sum;
    }
  }
  double rel = std::sqrt(err / norm);
  if (!(rel < 0.003))
    throw std::runtime_error("native wo_a differs from quantized reference: " +
                             std::to_string(rel));
  // A DSpark verifier must produce the same result for a token alone and
  // beside other tokens, including padding. Exercise both ends of dispatch.
  for (int count : {1, 8}) {
    Device<uint8_t> batch(count * stride * 33 / 32);
    Device<uint16_t> result(count * groups * n);
    for (int r = 0; r < count; ++r) {
      const int source = (r + 1) % rows;
      DGPP_CUDA_OK(cudaMemcpyAsync(batch.p + r * stride, packed.p + source * stride, stride,
                                   cudaMemcpyDeviceToDevice, stream));
      DGPP_CUDA_OK(cudaMemcpyAsync(batch.p + count * stride + r * stride / 32,
                                   packed.p + values + source * stride / 32, stride / 32,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    launch_mxfp8_woa_decode(batch.p, weights.p, scales.p, result.p, count, groups, n, k, stream);
    auto actual = read(result.p, count * groups * n);
    for (int r = 0; r < count; ++r) {
      const int source = (r + 1) % rows;
      if (!std::equal(actual.begin() + r * groups * n, actual.begin() + (r + 1) * groups * n,
                      out.begin() + source * groups * n))
        throw std::runtime_error("MXFP8 wo_a output depends on batch companions");
    }
  }
  const int64_t live = 24319;
  if (rows > 2) DGPP_CUDA_OK(cudaMemcpyAsync(pos.p + 2, &live, sizeof(live), cudaMemcpyHostToDevice, stream));
  cudaGraphExec_t graphs[2]{};
  cudaEvent_t start, end;
  DGPP_CUDA_OK(cudaEventCreate(&start));
  DGPP_CUDA_OK(cudaEventCreate(&end));
  for (int v = 0; v < 2; ++v) {
    cudaGraph_t graph;
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < copies; ++i) launch(v, i);
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&graphs[v], graph, 0));
    DGPP_CUDA_OK(cudaGraphDestroy(graph));
    DGPP_CUDA_OK(cudaGraphLaunch(graphs[v], stream));
  }
  std::printf("quantized reference relative RMS %.6f, %d weight copies\n", rel, copies);
  for (int r = 0; r < 12; ++r) {
    for (int j = 0; j < 2; ++j) {
      const int v = (j + r) % 2;
      DGPP_CUDA_OK(cudaEventRecord(start, stream));
      for (int rep = 0; rep < 12; ++rep) DGPP_CUDA_OK(cudaGraphLaunch(graphs[v], stream));
      DGPP_CUDA_OK(cudaEventRecord(end, stream));
      DGPP_CUDA_OK(cudaEventSynchronize(end));
      float ms;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, start, end));
      std::printf("handoff,%d,%s,%.3f\n", r, v ? "native" : "paired", ms * 1000 / (copies * 12));
    }
  }
  for (auto graph : graphs) DGPP_CUDA_OK(cudaGraphExecDestroy(graph));
  DGPP_CUDA_OK(cudaEventDestroy(start));
  DGPP_CUDA_OK(cudaEventDestroy(end));
}
// Read one rank-local head three times, as in fixed-depth-three DSpark.
// Rotate whole heads between chains so the first row starts beyond L2.
void markov_chain(int vocab, int rank, size_t l2, cudaStream_t stream) {
  const size_t weights = size_t(vocab) * rank;
  const size_t packed_size = dsv41_markov_packed_bytes(vocab, rank) / sizeof(uint16_t);
  const int copies = std::max(3, int((4 * l2 + 2 * weights - 1) / (2 * weights)));
  Device<uint16_t> head(weights * copies), packed(packed_size * copies), embedding(3 * rank);
  Device<int64_t> tokens(3);
  Device<float> base(3 * vocab), result(3 * vocab);
  const int64_t ids[] = {2, 1, 0};
  DGPP_CUDA_OK(cudaMemcpyAsync(tokens.p, ids, sizeof(ids), cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(head.p, 0x3d, 2 * weights * copies, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(embedding.p, 0x3c, 3 * rank * 2, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(base.p, 0, size_t(3) * vocab * 4, stream));
  for (int i = 0; i < copies; ++i)
    dsv41_pack_markov_head(head.p + i * weights, packed.p + i * packed_size, vocab, rank, stream);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  cudaGraphExec_t graphs[4]{};
  for (int v = 0; v < 4; ++v) {
    cudaGraph_t graph;
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < copies; ++i) {
      auto* w = v < 2 ? head.p + i * weights : packed.p + i * packed_size;
      for (int r = 0; r < 3; ++r)
        dsv41_dspark_markov_bias(base.p, 3 * vocab, r, embedding.p, w, rank, 0, vocab, tokens.p + r,
            1, 1, result.p + r * vocab, vocab, 1, stream, nullptr, static_cast<DsparkMarkovOrder>(v));
    }
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&graphs[v], graph, 0));
    DGPP_CUDA_OK(cudaGraphDestroy(graph));
    DGPP_CUDA_OK(cudaGraphLaunch(graphs[v], stream));
  }
  cudaEvent_t a, b;
  DGPP_CUDA_OK(cudaEventCreate(&a));
  DGPP_CUDA_OK(cudaEventCreate(&b));
  std::puts("layout,round,microseconds_per_three_biases");
  for (int round = 0; round < 12; ++round) {
    for (int k = 0; k < 4; ++k) {
      int v = (k + round) % 4;
      DGPP_CUDA_OK(cudaEventRecord(a, stream));
      for (int i = 0; i < 8; ++i) DGPP_CUDA_OK(cudaGraphLaunch(graphs[v], stream));
      DGPP_CUDA_OK(cudaEventRecord(b, stream));
      DGPP_CUDA_OK(cudaEventSynchronize(b));
      float ms;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, a, b));
      std::printf("%d,%d,%.2f\n", v, round, ms * 1000 / (copies * 8));
    }
  }
  for (auto g : graphs) DGPP_CUDA_OK(cudaGraphExecDestroy(g));
  DGPP_CUDA_OK(cudaEventDestroy(a));
  DGPP_CUDA_OK(cudaEventDestroy(b));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    cudaDeviceProp prop{};
    DGPP_CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (argc == 4 && std::string(argv[1]) == "--markov-chain") {
      const int vocab = std::stoi(argv[2]), rank = std::stoi(argv[3]);
      if (vocab < 32 || vocab > 262144 || rank < 8 || rank > 512 || rank % 8)
        throw std::invalid_argument("--markov-chain local_vocab markov_rank");
      markov_chain(vocab, rank, prop.l2CacheSize, stream);
      DGPP_CUDA_OK(cudaStreamDestroy(stream));
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--wo-b-mxfp8") {
      int count = argc == 3 ? std::stoi(argv[2]) : 4;
      if (argc > 3 || count < 1 || count > 8) throw std::invalid_argument("--wo-b-mxfp8 [rows 1..8]");
      std::puts("projection,variant,weight_copies,p25_us,median_us,p75_us,rel_l2");
      measure({"wo_b", 1, 5120, 4096, true, false, true}, prop.l2CacheSize, stream, true, count);
      DGPP_CUDA_OK(cudaStreamDestroy(stream));
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--woa-handoff") {
      if (argc != 2 && argc != 5)
        throw std::invalid_argument("--woa-handoff [rows main_splits window_splits]");
      woa_handoff(prop.l2CacheSize, stream, argc == 5 ? std::stoi(argv[2]) : 4,
                  argc == 5 ? std::stoi(argv[3]) : 0, argc == 5 ? std::stoi(argv[4]) : 8);
      DGPP_CUDA_OK(cudaStreamDestroy(stream));
      return 0;
    }
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
