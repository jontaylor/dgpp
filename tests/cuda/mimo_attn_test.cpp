#include "kernels/mimo_attn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"

namespace {
using dgpp::kda_test::DevBuf;
struct Fixture {
  dgpp::MimoAttentionShape shape;
  std::vector<uint16_t> fused, q, k, v, output, sinks;
  std::vector<int64_t> positions;
  std::vector<float> freq;
  DevBuf df, dq, dk, dv, dout, dsinks, dpos, dfreq, dstatus;
  cudaStream_t stream = dgpp::kda_test::test_stream();

  explicit Fixture(dgpp::MimoAttentionShape s, bool shared_cache = false)
      : shape(s),
        fused(s.requests * s.fused_width()),
        q(s.requests * s.q_width(), 0x1234),
        k(size_t(shared_cache ? 1 : s.requests) * s.capacity * s.k_width(), 0x1234),
        v(size_t(shared_cache ? 1 : s.requests) * s.capacity * s.v_width(), 0x1234),
        output(s.requests * s.q_heads * 128),
        sinks(s.q_heads),
        positions(s.requests, -1),
        freq(dgpp::mimo_ref::inv_freq(s.window ? 1e4 : 1e7)),
        df(fused.size() * 2),
        dq(q.size() * 2),
        dk(k.size() * 2),
        dv(v.size() * 2),
        dout(output.size() * 2),
        dsinks(sinks.size() * 2),
        dpos(positions.size() * 8),
        dfreq(freq.size() * 4),
        dstatus(s.requests * 4) {
    dq.upload(q.data(), q.size() * 2);
    dk.upload(k.data(), k.size() * 2);
    dv.upload(v.data(), v.size() * 2);
    dfreq.upload(freq.data(), freq.size() * 4);
    for (int h = 0; h < s.q_heads; ++h) sinks[h] = dgpp::float_to_bf16_bits(float(h % 5 - 2) / 4);
    dsinks.upload(sinks.data(), sinks.size() * 2);
  }
  void prepare() {
    for (int r = 0; r < shape.requests; ++r)
      for (int i = 0; i < shape.fused_width(); ++i)
        fused[r * shape.fused_width() + i] = dgpp::float_to_bf16_bits(
            float((i * 7 + std::max<int64_t>(0, positions[r]) * 13 + r * 17) % 101 - 50) / 64);
    df.upload(fused.data(), fused.size() * 2);
    dpos.upload(positions.data(), positions.size() * 8);
  }
  void launch() {
    dgpp::mimo_qkv_append(shape, df.as<uint16_t>(), dfreq.as<float>(), dpos.as<int64_t>(),
                          dq.as<uint16_t>(), dk.as<uint16_t>(), dv.as<uint16_t>(),
                          dstatus.as<int32_t>(), stream);
    dgpp::mimo_attention(shape, dq.as<uint16_t>(), dk.as<uint16_t>(), dv.as<uint16_t>(),
                         dpos.as<int64_t>(), shape.window ? dsinks.as<uint16_t>() : nullptr,
                         dout.as<uint16_t>(), stream);
  }
  void reference(bool attention) {
    dgpp::mimo_ref::qkv_append(shape, fused.data(), freq.data(), positions.data(), q.data(),
                               k.data(), v.data());
    if (attention)
      dgpp::mimo_ref::attention(shape, q.data(), k.data(), v.data(), positions.data(),
                                shape.window ? sinks.data() : nullptr, output.data());
  }
  void compare(bool tensor_core = false) {
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    std::vector<uint16_t> got(output.size());
    dout.download(got.data(), got.size() * 2);
    auto stats = dgpp::kda_test::compare_bf16(got, output, 1);
    if (tensor_core) {
      // Tensor cores reassociate the FP32 dot product. Keep the one-ulp
      // relative gate but permit cancellation noise below 1e-4 absolute;
      // the aggregate relative-L2 gate remains 0.2% for either path.
      std::vector<float> a(got.size()), b(output.size());
      for (size_t i = 0; i < got.size(); ++i) {
        a[i] = dgpp::bf16_bits_to_float(got[i]);
        b[i] = dgpp::bf16_bits_to_float(output[i]);
      }
      stats = dgpp::kda_test::compare_abs_rel(a.data(), b.data(), a.size(), 1.0 / 128, 1e-4);
    }
    dgpp::kda_test::require_bf16("MiMo attention", stats, 0.002, 0.0);
    std::vector<int32_t> status(shape.requests, -1);
    dstatus.download(status.data(), status.size() * 4);
    if (std::any_of(status.begin(), status.end(), [](auto x) { return x != 0; }))
      throw std::runtime_error("valid positions reported errors");
  }
};
}  // namespace

// Exact A/B gate: this candidate must not introduce a new numerical tolerance.
DGPP_TEST(mimo_cuda_fused_prefill_is_bitwise_equal_and_kernel_only) {
  // Bound race-instrumentation cost; the default retains full 8K/64K coverage.
  const bool small = std::getenv("DGPP_MIMO_ATTN_SANITIZER_SMALL") != nullptr;
  for (int window : {0, 128}) {
    for (int scenario = 0; scenario < 4; ++scenario) {
      const int rows = scenario == 1 ? 17 : (scenario == 2 ? 128 : 1);
      const int capacity = window ? 263 : (scenario == 0 ? 3 : (small ? 512 : (scenario == 3 ? 65536 : 8192)));
      const int end_key = window ? capacity * 3 + 7 : capacity;  // wrapped non-power-of-two ring
      Fixture f({rows, 32, window ? 4 : 2, capacity, window}, true);
      DevBuf scores(size_t(rows) * 32 * capacity * 6);
      uint32_t rng = 0x12345678;
      auto fill = [&](std::vector<uint16_t>& data) {
        for (auto& x : data) {
          rng = rng * 1664525u + 1013904223u;
          x = dgpp::float_to_bf16_bits(float(int(rng >> 16) - 32768) / 16384.f);
        }
      };
      for (int h = 0; h < f.shape.q_heads; ++h)
        f.sinks[h] = dgpp::float_to_bf16_bits(h % 2 ? 20.f : -20.f);
      f.dsinks.upload(f.sinks.data(), f.sinks.size() * 2);
      fill(f.q);
      fill(f.k);
      fill(f.v);
      f.dq.upload(f.q.data(), f.q.size() * 2);
      f.dk.upload(f.k.data(), f.k.size() * 2);
      f.dv.upload(f.v.data(), f.v.size() * 2);
      for (bool parallel : {false, true}) {
        for (bool sink : {false, true}) {
          auto run = [&](bool fused) {
            dgpp::mimo_attention_prefill(
                f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(), f.dv.as<uint16_t>(),
                f.dpos.as<int64_t>(), sink ? f.dsinks.as<uint16_t>() : nullptr,
                f.dout.as<uint16_t>(), scores.as<float>(), end_key, f.stream,
                parallel, true, fused);
          };
          // end_key is fixed across replay; positions follow its public contract.
          for (int r = 0; r < rows; ++r) f.positions[r] = end_key - rows + r;
          f.dpos.upload(f.positions.data(), rows * 8);
          cudaGraph_t graph;
          DGPP_CUDA_OK(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeGlobal));
          run(true);
          DGPP_CUDA_OK(cudaStreamEndCapture(f.stream, &graph));
          size_t count = 0;
          DGPP_CUDA_OK(cudaGraphGetNodes(graph, nullptr, &count));
          std::vector<cudaGraphNode_t> nodes(count);
          DGPP_CUDA_OK(cudaGraphGetNodes(graph, nodes.data(), &count));
          for (auto node : nodes) {
            cudaGraphNodeType type;
            DGPP_CUDA_OK(cudaGraphNodeGetType(node, &type));
            if (type != cudaGraphNodeTypeKernel)
              throw std::runtime_error("fused prefill graph contains non-kernel node");
          }
          cudaGraphExec_t executable;
          DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
          for (int replay = 0; replay < 2; ++replay) {
            // Mutated values detect stale scratch/output on graph replay.
            fill(f.v);
            f.dv.upload(f.v.data(), f.v.size() * 2);
            run(false);
            DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
            std::vector<uint16_t> expected(f.output.size()), got(f.output.size());
            f.dout.download(expected.data(), expected.size() * 2);
            DGPP_CUDA_OK(cudaGraphLaunch(executable, f.stream));
            DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
            f.dout.download(got.data(), got.size() * 2);
            if (expected != got)
              throw std::runtime_error("fused prefill changed BF16 output bits");
          }
          DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
          DGPP_CUDA_OK(cudaGraphDestroy(graph));
        }
      }
    }
  }
}

DGPP_TEST(mimo_cuda_global_and_ring_attention_match_reference) {
  for (int window : {0, 128}) {
    // Actual TP4 head shapes: 16 Q heads, 1 global / 2 sliding KV heads.
    Fixture f({2, 16, window ? 2 : 1, window ? 128 : 257, window});
    int64_t second = 0;
    for (int64_t p = 0; p <= 256; ++p) {
      f.positions = {p, p % 7 == 0 ? -1 : second++};
      f.prepare();
      f.launch();
      const bool compare = p < 2 || (p >= 126 && p <= 129) || p >= 255;
      f.reference(compare);
      if (compare) f.compare();
    }
    // Reusing a cache slot at position zero must not read an old ring tail.
    f.positions = {0, -1};
    f.prepare();
    f.launch();
    f.reference(true);
    f.compare();
  }
}

DGPP_TEST(mimo_cuda_graph_replay_covers_all_rows_and_padding) {
  Fixture f({65, 2, 1, 128, 128});
  f.prepare();
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  DGPP_CUDA_OK(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeGlobal));
  f.launch();
  DGPP_CUDA_OK(cudaStreamEndCapture(f.stream, &graph));
  size_t nodes = 0;
  DGPP_CUDA_OK(cudaGraphGetNodes(graph, nullptr, &nodes));
  if (nodes != 2) throw std::runtime_error("expected two allocation-free kernel graph nodes");
  DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (int64_t p = 0; p < 3; ++p) {
    for (int r = 0; r < f.shape.requests; ++r) f.positions[r] = r % 3 == 0 ? -1 : p;
    f.prepare();
    DGPP_CUDA_OK(cudaGraphLaunch(executable, f.stream));
    f.reference(true);
    f.compare();
  }
  std::vector<uint16_t> actual(f.k.size());
  f.dk.download(actual.data(), actual.size() * 2);
  for (int r = 0; r < f.shape.requests; r += 3) {
    const size_t start = size_t(r) * f.shape.capacity * f.shape.k_width();
    const size_t end = start + f.shape.capacity * f.shape.k_width();
    if (!std::all_of(actual.begin() + start, actual.begin() + end,
                     [](auto v) { return v == 0x1234; }))
      throw std::runtime_error("padding mutated cache");
  }
  DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
  DGPP_CUDA_OK(cudaGraphDestroy(graph));
}

DGPP_TEST(mimo_cuda_mapped_graph_rows_preserve_independent_histories) {
  for (int window : {0, 128}) {
    Fixture f({4, 4, 2, window ? 128 : 257, window});
    DevBuf mapping(4 * sizeof(int32_t));
    DevBuf control_k(f.k.size() * 2), control_v(f.v.size() * 2), control_q(f.q.size() * 2),
        control_out(f.output.size() * 2), control_status(4 * 4);
    control_k.upload(f.k.data(), f.k.size() * 2);
    control_v.upload(f.v.data(), f.v.size() * 2);
    auto scalar = f.shape;
    scalar.requests = 1;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    DGPP_CUDA_OK(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeGlobal));
    dgpp::mimo_qkv_append(f.shape, f.df.as<uint16_t>(), f.dfreq.as<float>(), f.dpos.as<int64_t>(),
                          f.dq.as<uint16_t>(), f.dk.as<uint16_t>(), f.dv.as<uint16_t>(),
                          f.dstatus.as<int32_t>(), f.stream, false, mapping.as<int32_t>());
    dgpp::mimo_attention(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(), f.dv.as<uint16_t>(),
                         f.dpos.as<int64_t>(), window ? f.dsinks.as<uint16_t>() : nullptr,
                         f.dout.as<uint16_t>(), f.stream, false, nullptr, mapping.as<int32_t>());
    DGPP_CUDA_OK(cudaStreamEndCapture(f.stream, &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    int64_t next[4] = {};
    for (int step = 0; step < 260; ++step) {
      int32_t ids[4];
      for (int row = 0; row < 4; ++row) {
        ids[row] = (row + step) % 4;
        if (step == 200 && ids[row] == 2) next[2] = 0;
        f.positions[row] = (step + row) % 7 == 0 ? -1 : next[ids[row]]++;
      }
      mapping.upload(ids, sizeof(ids));
      f.prepare();
      DGPP_CUDA_OK(cudaGraphLaunch(executable, f.stream));
      for (int row = 0; row < 4; ++row) {
        auto* k = control_k.as<uint16_t>() + size_t(ids[row]) * scalar.capacity * scalar.k_width();
        auto* v = control_v.as<uint16_t>() + size_t(ids[row]) * scalar.capacity * scalar.v_width();
        dgpp::mimo_qkv_append(scalar, f.df.as<uint16_t>() + row * scalar.fused_width(),
                              f.dfreq.as<float>(), f.dpos.as<int64_t>() + row,
                              control_q.as<uint16_t>() + row * scalar.q_width(), k, v,
                              control_status.as<int32_t>() + row, f.stream);
        dgpp::mimo_attention(scalar, control_q.as<uint16_t>() + row * scalar.q_width(), k, v,
                             f.dpos.as<int64_t>() + row, window ? f.dsinks.as<uint16_t>() : nullptr,
                             control_out.as<uint16_t>() + row * scalar.q_heads * 128, f.stream);
      }
      DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
      std::vector<uint16_t> actual(f.output.size()), expected(f.output.size());
      f.dout.download(actual.data(), actual.size() * 2);
      control_out.download(expected.data(), expected.size() * 2);
      if (actual != expected)
        throw std::runtime_error("mapped graph differs from isolated GPU rows");
    }
    std::vector<uint16_t> k(f.k.size()), v(f.v.size());
    f.dk.download(k.data(), k.size() * 2);
    f.dv.download(v.data(), v.size() * 2);
    control_k.download(f.k.data(), f.k.size() * 2);
    control_v.download(f.v.data(), f.v.size() * 2);
    if (k != f.k || v != f.v) throw std::runtime_error("mapped cache histories differ");
    DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
    DGPP_CUDA_OK(cudaGraphDestroy(graph));
  }
}

DGPP_TEST(mimo_cuda_invalid_position_does_not_write_cache) {
  Fixture f({2, 2, 1, 4, 0});
  f.positions = {4, -1};
  f.prepare();
  f.launch();
  DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
  int32_t status[2];
  f.dstatus.download(status, sizeof(status));
  if (status[0] != 1 || status[1] != 0) throw std::runtime_error("invalid position status");
  std::vector<uint16_t> k(f.k.size()), v(f.v.size()), out(f.output.size());
  f.dk.download(k.data(), k.size() * 2);
  f.dv.download(v.data(), v.size() * 2);
  f.dout.download(out.data(), out.size() * 2);
  if (k != f.k || v != f.v || std::any_of(out.begin(), out.end(), [](auto x) { return x != 0; }))
    throw std::runtime_error("invalid/padded rows touched state or returned nonzero output");
}

DGPP_TEST(mimo_cuda_parallel_chunk_preserves_causal_history_and_reset) {
  for (int window : {0, 128}) {
    Fixture f({128, 4, 2, window ? 256 : 512, window});
    DevBuf scores(size_t(f.shape.requests) * f.shape.q_heads * f.shape.capacity *
                  (sizeof(float) + sizeof(uint16_t)));
    auto scalar = f.shape;
    scalar.requests = 1;
    // Cross the physical ring boundary as well as the logical window, end
    // with a short chunk, then reuse the same cache for a new sequence.
    for (int start : {0, 128, 256, 384, 0}) {
      const int count = start == 384 ? 3 : 128;
      f.shape.requests = count;
      for (int t = 0; t < count; ++t) f.positions[t] = start + t;
      f.prepare();
      dgpp::mimo_qkv_append(f.shape, f.df.as<uint16_t>(), f.dfreq.as<float>(), f.dpos.as<int64_t>(),
                            f.dq.as<uint16_t>(), f.dk.as<uint16_t>(), f.dv.as<uint16_t>(),
                            f.dstatus.as<int32_t>(), f.stream, true);
      dgpp::mimo_attention(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(), f.dv.as<uint16_t>(),
                           f.dpos.as<int64_t>(), window ? f.dsinks.as<uint16_t>() : nullptr,
                           f.dout.as<uint16_t>(), f.stream, true, scores.as<float>());
      dgpp::mimo_attention_prefill(
          f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(), f.dv.as<uint16_t>(),
          f.dpos.as<int64_t>(), window ? f.dsinks.as<uint16_t>() : nullptr, f.dout.as<uint16_t>(),
          scores.as<float>(), start + count, f.stream, true, false);
      DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
      std::vector<uint16_t> narrow(f.output.size()), wide(f.output.size());
      f.dout.download(narrow.data(), narrow.size() * 2);
      dgpp::mimo_attention_prefill(
          f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(), f.dv.as<uint16_t>(),
          f.dpos.as<int64_t>(), window ? f.dsinks.as<uint16_t>() : nullptr, f.dout.as<uint16_t>(),
          scores.as<float>(), start + count, f.stream);
      DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
      f.dout.download(wide.data(), wide.size() * 2);
      if (wide != narrow) throw std::runtime_error("wide PV changed the ordered MMA output");
      for (bool online : {false, true}) {
        dgpp::mimo_attention_bounded_prefill(
            f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(), f.dv.as<uint16_t>(),
            f.dpos.as<int64_t>(), window ? f.dsinks.as<uint16_t>() : nullptr,
            f.dout.as<uint16_t>(), start + count, f.stream, online);
        DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
        std::vector<uint16_t> bounded(f.output.size());
        f.dout.download(bounded.data(), bounded.size() * 2);
        double error = 0, norm = 0, maxabs = 0;
        int changed = 0;
        for (int i = 0; i < count * f.shape.q_heads * 128; ++i) {
          const double ref = dgpp::bf16_bits_to_float(wide[i]);
          const double got = dgpp::bf16_bits_to_float(bounded[i]);
          if (!std::isfinite(got)) throw std::runtime_error("nonfinite bounded attention");
          error += (got - ref) * (got - ref);
          norm += ref * ref;
          maxabs = std::max(maxabs, std::abs(got - ref));
          changed += bounded[i] != wide[i];
        }
        const double rel = std::sqrt(error / std::max(norm, 1e-30));
        std::cout << "bounded window=" << window << " start=" << start << " count=" << count
                  << " online=" << online << " rel_l2=" << rel << " maxabs=" << maxabs
                  << " changed=" << changed << '\n';
        if (rel > (online ? 0.025 : 0.005))
          throw std::runtime_error("bounded attention exceeds synthetic relative L2 gate");
      }
      // Existing CPU-reference comparison below remains on baseline output.
      DGPP_CUDA_OK(cudaMemcpyAsync(f.dout.as<uint16_t>(), wide.data(), wide.size() * 2,
                                   cudaMemcpyHostToDevice, f.stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
      for (int t = 0; t < count; ++t) {
        dgpp::mimo_ref::qkv_append(scalar, f.fused.data() + t * scalar.fused_width(), f.freq.data(),
                                   f.positions.data() + t, f.q.data() + t * scalar.q_width(),
                                   f.k.data(), f.v.data());
        dgpp::mimo_ref::attention(scalar, f.q.data() + t * scalar.q_width(), f.k.data(), f.v.data(),
                                  f.positions.data() + t, window ? f.sinks.data() : nullptr,
                                  f.output.data() + t * scalar.q_heads * 128);
      }
      f.compare(true);
    }
  }
}

DGPP_TEST(mimo_cuda_long_prefill_parallel_softmax_checks_high_precision_normalizer) {
  Fixture f({16, 4, 2, 65536, 0}, true);
  for (size_t i = 0; i < f.k.size(); ++i)
    f.k[i] = dgpp::float_to_bf16_bits(float(int((i * 17 + i / 193) % 101) - 50) / 64);
  for (size_t i = 0; i < f.v.size(); ++i)
    f.v[i] = dgpp::float_to_bf16_bits(float(int((i * 13 + i / 127) % 97) - 48) / 64);
  f.dk.upload(f.k.data(), f.k.size() * 2);
  f.dv.upload(f.v.data(), f.v.size() * 2);
  for (int row = 0; row < 16; ++row) f.positions[row] = 65520 + row;
  f.prepare();
  dgpp::mimo_qkv_append(f.shape, f.df.as<uint16_t>(), f.dfreq.as<float>(), f.dpos.as<int64_t>(),
                        f.dq.as<uint16_t>(), f.dk.as<uint16_t>(), f.dv.as<uint16_t>(),
                        f.dstatus.as<int32_t>(), f.stream, true);
  DevBuf scores(size_t(16) * 4 * 65536 * 6);
  std::vector<uint16_t> serial(f.output.size()), parallel_output(f.output.size());
  for (bool parallel : {false, true}) {
    dgpp::mimo_attention_prefill(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(),
                                 f.dv.as<uint16_t>(), f.dpos.as<int64_t>(), nullptr,
                                 f.dout.as<uint16_t>(), scores.as<float>(), 65536, f.stream,
                                 parallel);
    DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
    auto& output = parallel ? parallel_output : serial;
    f.dout.download(output.data(), output.size() * 2);
  }
  // An independent FP64 denominator for query 0 / head 0 distinguishes
  // parallel-rounding drift from the long serial FP32 sum's own error.
  std::vector<float> score(65536);
  scores.download(score.data(), score.size() * sizeof(float));
  f.dv.download(f.v.data(), f.v.size() * 2);
  const int count = int(f.positions[0]) + 1;
  const float maximum = *std::max_element(score.begin(), score.begin() + count);
  std::vector<float> exponentials(count);
  double denominator = 0;
  for (int key = 0; key < count; ++key) {
    const float difference =
        dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(score[key] - maximum));
    exponentials[key] = std::exp(difference);
    denominator += exponentials[key];
  }
  double serial_error = 0, parallel_error = 0;
  float maximum_error = 0;
  for (int dim = 0; dim < 128; ++dim) {
    float reference = 0;
    for (int key = 0; key < count; ++key) {
      const float probability = dgpp::bf16_bits_to_float(
          dgpp::float_to_bf16_bits(float(double(exponentials[key]) / denominator)));
      const float value = dgpp::bf16_bits_to_float(f.v[size_t(key) * f.shape.v_width() + dim]);
      reference = std::fma(probability, value, reference);
    }
    reference = dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(reference));
    const float a = dgpp::bf16_bits_to_float(serial[dim]) - reference;
    const float b = dgpp::bf16_bits_to_float(parallel_output[dim]) - reference;
    serial_error += double(a) * a;
    parallel_error += double(b) * b;
    maximum_error = std::max(maximum_error, std::abs(b));
  }
  std::cout << "long softmax: serial SSE=" << serial_error << " parallel SSE=" << parallel_error
            << " parallel maxabs=" << maximum_error << '\n';
  if (maximum_error > 1e-4f || parallel_error > serial_error * 1.1 + 1e-12)
    throw std::runtime_error("parallel softmax exceeds high-precision normalization error gate");
  for (size_t i = 0; i < serial.size(); ++i) {
    const float a = dgpp::bf16_bits_to_float(serial[i]);
    const float b = dgpp::bf16_bits_to_float(parallel_output[i]);
    if (!std::isfinite(b) || std::abs(a - b) > 1e-4f + std::abs(a) / 128)
      throw std::runtime_error("parallel softmax exceeds absolute/relative output gate");
  }
}

DGPP_TEST(mimo_cuda_tensor_decode_preserves_mapped_slots_and_padding) {
  Fixture f({4, 32, 2, 65536, 0});
  for (size_t i = 0; i < f.k.size(); ++i)
    f.k[i] = dgpp::float_to_bf16_bits(float(int((i * 17 + i / 193) % 101) - 50) / 64);
  for (size_t i = 0; i < f.v.size(); ++i)
    f.v[i] = dgpp::float_to_bf16_bits(float((i * 13 + i / 127) % 97) / 128 + .125f);
  f.dk.upload(f.k.data(), f.k.size() * 2);
  f.dv.upload(f.v.data(), f.v.size() * 2);
  DevBuf scores(size_t(4) * 32 * 65536 * 6), mapping(4 * 4), reference(f.output.size() * 2);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  DGPP_CUDA_OK(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeGlobal));
  dgpp::mimo_attention_decode(
      f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(), f.dv.as<uint16_t>(), f.dpos.as<int64_t>(),
      nullptr, f.dout.as<uint16_t>(), scores.as<float>(), f.stream, mapping.as<int32_t>());
  DGPP_CUDA_OK(cudaStreamEndCapture(f.stream, &graph));
  DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  auto one = f.shape;
  one.requests = 1;
  for (int trial = 0; trial < 3; ++trial) {
    int32_t ids[4];
    for (int row = 0; row < 4; ++row) ids[row] = (row + trial) % 4;
    f.positions = {trial == 0 ? 17 : 509 + trial, 8191 + trial, 65533 + trial, -1};
    mapping.upload(ids, sizeof(ids));
    f.prepare();
    dgpp::mimo_qkv_append(f.shape, f.df.as<uint16_t>(), f.dfreq.as<float>(), f.dpos.as<int64_t>(),
                          f.dq.as<uint16_t>(), f.dk.as<uint16_t>(), f.dv.as<uint16_t>(),
                          f.dstatus.as<int32_t>(), f.stream, false, mapping.as<int32_t>());
    for (int row = 0; row < 4; ++row)
      dgpp::mimo_attention(one, f.dq.as<uint16_t>() + size_t(row) * one.q_width(),
                           f.dk.as<uint16_t>() + size_t(ids[row]) * one.capacity * one.k_width(),
                           f.dv.as<uint16_t>() + size_t(ids[row]) * one.capacity * one.v_width(),
                           f.dpos.as<int64_t>() + row, nullptr,
                           reference.as<uint16_t>() + size_t(row) * one.q_heads * 128, f.stream);
    DGPP_CUDA_OK(cudaGraphLaunch(executable, f.stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
    reference.download(f.output.data(), f.output.size() * 2);
    std::vector<uint16_t> actual(f.output.size());
    f.dout.download(actual.data(), actual.size() * 2);
    for (int row = 0; row < f.shape.requests; ++row)
      if (f.positions[row] >= 0 && f.positions[row] < 511)
        for (int i = 0; i < f.shape.q_heads * 128; ++i) {
          const size_t offset = size_t(row) * f.shape.q_heads * 128 + i;
          if (actual[offset] != f.output[offset])
            throw std::runtime_error("short decode changed scalar arithmetic");
        }
    f.compare(true);
  }
  DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
  DGPP_CUDA_OK(cudaGraphDestroy(graph));
}

DGPP_TEST(mimo_cuda_partial_snapshot_restores_causal_chunk_after_poison_and_wrap) {
  for (int window : {0, 128}) {
    Fixture control({128, 4, 2, window ? 256 : 512, window}, true);
    Fixture restored(control.shape, true);
    for (size_t i = 0; i < control.k.size(); ++i)
      control.k[i] = dgpp::float_to_bf16_bits(float(int(i % 97) - 48) / 64);
    for (size_t i = 0; i < control.v.size(); ++i)
      control.v[i] = dgpp::float_to_bf16_bits(float(int(i % 89) - 44) / 64);
    std::vector<uint16_t> poison_k(control.k.size(), 0x7fc0), poison_v(control.v.size(), 0x7fc0);
    const size_t snapshot_slots = window ? window : control.shape.capacity;
    const size_t snapshot_keys = snapshot_slots * control.shape.k_width();
    const size_t snapshot_values = snapshot_slots * control.shape.v_width();
    DevBuf saved_k(snapshot_keys * 2), saved_v(snapshot_values * 2);
    DevBuf scores(size_t(128) * 4 * control.shape.capacity * 6);
    for (int position : {1, 127, 128, 255, 256, 257, 383}) {
      control.dk.upload(control.k.data(), control.k.size() * 2);
      control.dv.upload(control.v.data(), control.v.size() * 2);
      saved_k.upload(poison_k.data(), snapshot_keys * 2);
      saved_v.upload(poison_v.data(), snapshot_values * 2);
      restored.dk.upload(poison_k.data(), poison_k.size() * 2);
      restored.dv.upload(poison_v.data(), poison_v.size() * 2);
      size_t packed_tokens = 0;
      for (const auto span : dgpp::mimo_snapshot_spans(control.shape, position)) {
        if (!span.count) continue;
        const size_t ko = size_t(span.first) * control.shape.k_width();
        const size_t vo = size_t(span.first) * control.shape.v_width();
        const size_t kb = size_t(span.count) * control.shape.k_width() * 2;
        const size_t vb = size_t(span.count) * control.shape.v_width() * 2;
        DGPP_CUDA_OK(cudaMemcpyAsync(
            saved_k.as<uint16_t>() + packed_tokens * control.shape.k_width(),
            control.dk.as<uint16_t>() + ko, kb, cudaMemcpyDeviceToDevice, control.stream));
        DGPP_CUDA_OK(cudaMemcpyAsync(
            saved_v.as<uint16_t>() + packed_tokens * control.shape.v_width(),
            control.dv.as<uint16_t>() + vo, vb, cudaMemcpyDeviceToDevice, control.stream));
        DGPP_CUDA_OK(
            cudaMemcpyAsync(restored.dk.as<uint16_t>() + ko,
                            saved_k.as<uint16_t>() + packed_tokens * control.shape.k_width(), kb,
                            cudaMemcpyDeviceToDevice, control.stream));
        DGPP_CUDA_OK(
            cudaMemcpyAsync(restored.dv.as<uint16_t>() + vo,
                            saved_v.as<uint16_t>() + packed_tokens * control.shape.v_width(), vb,
                            cudaMemcpyDeviceToDevice, control.stream));
        packed_tokens += span.count;
      }
      DGPP_CUDA_OK(cudaStreamSynchronize(control.stream));
      for (Fixture* f : {&control, &restored}) {
        for (int row = 0; row < 128; ++row) f->positions[row] = position + row;
        f->prepare();
        dgpp::mimo_qkv_append(f->shape, f->df.as<uint16_t>(), f->dfreq.as<float>(),
                              f->dpos.as<int64_t>(), f->dq.as<uint16_t>(), f->dk.as<uint16_t>(),
                              f->dv.as<uint16_t>(), f->dstatus.as<int32_t>(), f->stream, true);
        dgpp::mimo_attention_prefill(
            f->shape, f->dq.as<uint16_t>(), f->dk.as<uint16_t>(), f->dv.as<uint16_t>(),
            f->dpos.as<int64_t>(), window ? f->dsinks.as<uint16_t>() : nullptr,
            f->dout.as<uint16_t>(), scores.as<float>(), position + 128, f->stream);
      }
      DGPP_CUDA_OK(cudaStreamSynchronize(control.stream));
      control.dout.download(control.output.data(), control.output.size() * 2);
      restored.dout.download(restored.output.data(), restored.output.size() * 2);
      if (control.output != restored.output)
        throw std::runtime_error("partial snapshot changed continuation after poisoned restore");
      for (auto value : restored.output)
        if (!std::isfinite(dgpp::bf16_bits_to_float(value)))
          throw std::runtime_error("partial snapshot exposed poisoned cache history");
    }
  }
}

void benchmark_decode() {
  for (int context : {128, 512, 2048, 8192, 65536}) {
    Fixture f({1, 32, 2, 65536, 0});
    f.positions[0] = context - 1;
    f.prepare();
    dgpp::mimo_qkv_append(f.shape, f.df.as<uint16_t>(), f.dfreq.as<float>(), f.dpos.as<int64_t>(),
                          f.dq.as<uint16_t>(), f.dk.as<uint16_t>(), f.dv.as<uint16_t>(),
                          f.dstatus.as<int32_t>(), f.stream);
    DevBuf scores(size_t(32) * 65536 * 6);
    for (int tensor : {0, 1, 2, 3}) {
      auto run = [&] {
        if (tensor == 3)
          dgpp::mimo_attention_online_decode(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(),
              f.dv.as<uint16_t>(), f.dpos.as<int64_t>(), nullptr, f.dout.as<uint16_t>(), f.stream);
        else if (tensor == 1)
          dgpp::mimo_attention_decode(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(),
                                      f.dv.as<uint16_t>(), f.dpos.as<int64_t>(), nullptr,
                                      f.dout.as<uint16_t>(), scores.as<float>(), f.stream);
        else
          dgpp::mimo_attention(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(),
                               f.dv.as<uint16_t>(), f.dpos.as<int64_t>(), nullptr,
                               f.dout.as<uint16_t>(), f.stream, false, tensor == 2 ? nullptr : scores.as<float>());
      };
      cudaEvent_t start, end;
      DGPP_CUDA_OK(cudaEventCreate(&start));
      DGPP_CUDA_OK(cudaEventCreate(&end));
      run();
      DGPP_CUDA_OK(cudaEventRecord(start, f.stream));
      for (int i = 0; i < 5; ++i) run();
      DGPP_CUDA_OK(cudaEventRecord(end, f.stream));
      DGPP_CUDA_OK(cudaEventSynchronize(end));
      float ms = 0;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, start, end));
      std::cout << "decode context=" << context << " tensor=" << tensor << " ms=" << ms / 5 << '\n';
      DGPP_CUDA_OK(cudaEventDestroy(start));
      DGPP_CUDA_OK(cudaEventDestroy(end));
    }
  }
}

void benchmark() {
  const int context = std::getenv("DGPP_MIMO_ATTN_BENCH_CONTEXT")
                          ? std::stoi(std::getenv("DGPP_MIMO_ATTN_BENCH_CONTEXT"))
                          : 8192;
  if (context < 128 || context > 1048576) throw std::invalid_argument("benchmark context");
  const int first_mode = std::getenv("DGPP_MIMO_ATTN_BENCH_FIRST_MODE")
      ? std::stoi(std::getenv("DGPP_MIMO_ATTN_BENCH_FIRST_MODE")) : 3;
  if (first_mode < 0 || first_mode > 6) throw std::invalid_argument("benchmark first mode");
  for (int window : {0, 128}) {
    Fixture f({128, 32, window ? 4 : 2, window ? 256 : context, window}, true);
    DevBuf scores(first_mode <= 4 ? size_t(128) * 32 * f.shape.capacity * 6 : 1);
    uint32_t rng = 37;
    for (auto* cache : {&f.k, &f.v}) {
      for (auto& x : *cache) {
        rng = rng * 1664525u + 1013904223u;
        x = dgpp::float_to_bf16_bits(float(int(rng >> 16) - 32768) / 16384.f);
      }
    }
    f.dk.upload(f.k.data(), f.k.size() * 2);
    f.dv.upload(f.v.data(), f.v.size() * 2);
    for (int t = 0; t < 128; ++t) f.positions[t] = context - 128 + t;
    f.prepare();
    dgpp::mimo_qkv_append(f.shape, f.df.as<uint16_t>(), f.dfreq.as<float>(), f.dpos.as<int64_t>(),
                          f.dq.as<uint16_t>(), f.dk.as<uint16_t>(), f.dv.as<uint16_t>(),
                          f.dstatus.as<int32_t>(), f.stream, true);
    for (int tensor = first_mode; tensor < 7; ++tensor) {
      cudaEvent_t start, end;
      DGPP_CUDA_OK(cudaEventCreate(&start));
      DGPP_CUDA_OK(cudaEventCreate(&end));
      auto run = [&] {
        if (tensor >= 5)
          dgpp::mimo_attention_bounded_prefill(f.shape, f.dq.as<uint16_t>(),
              f.dk.as<uint16_t>(), f.dv.as<uint16_t>(), f.dpos.as<int64_t>(),
              window ? f.dsinks.as<uint16_t>() : nullptr, f.dout.as<uint16_t>(),
              context, f.stream, tensor == 6);
        else if (tensor)
          dgpp::mimo_attention_prefill(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(),
                                       f.dv.as<uint16_t>(), f.dpos.as<int64_t>(),
                                       window ? f.dsinks.as<uint16_t>() : nullptr,
                                       f.dout.as<uint16_t>(), scores.as<float>(), context, f.stream,
                                       tensor >= 2, tensor >= 3, tensor == 4);
        else
          dgpp::mimo_attention(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(),
                               f.dv.as<uint16_t>(), f.dpos.as<int64_t>(),
                               window ? f.dsinks.as<uint16_t>() : nullptr, f.dout.as<uint16_t>(),
                               f.stream, true, scores.as<float>());
      };
      run();
      DGPP_CUDA_OK(cudaEventRecord(start, f.stream));
      for (int i = 0; i < 5; ++i) run();
      DGPP_CUDA_OK(cudaEventRecord(end, f.stream));
      DGPP_CUDA_OK(cudaEventSynchronize(end));
      float ms = 0;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, start, end));
      std::cout << "window=" << window << " tensor=" << tensor << " ms=" << ms / 5 << '\n';
      DGPP_CUDA_OK(cudaEventDestroy(start));
      DGPP_CUDA_OK(cudaEventDestroy(end));
    }
  }
}

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "SKIP: MiMo attention requires an idle CUDA GPU\n";
    return 2;
  }
  if (std::getenv("DGPP_MIMO_DECODE_BENCH")) {
    benchmark_decode();
    return 0;
  }
  if (std::getenv("DGPP_MIMO_ATTN_BENCH")) {
    benchmark();
    return 0;
  }
  return dgpp::test::run_all();
}

DGPP_TEST(mimo_cuda_speculative_rows_match_sequential_with_rejection_and_wrap) {
  for (int width : {2, 4}) {
    for (int window : {0, 128}) {
      Fixture f({2 * width, 16, window ? 2 : 1, window ? 256 : 1024, window});
      DevBuf mapping(2 * width * sizeof(int32_t)),
          scores(size_t(2 * width) * 16 * f.shape.capacity * 6);
      DevBuf refk(f.k.size() * 2), refv(f.v.size() * 2), refq(f.q.size() * 2);
      DevBuf refout(f.output.size() * 2);
      std::vector<int32_t> ids(2 * width);
      for (int r = 0; r < 2 * width; ++r) ids[r] = r < width ? 1 : 0;
      mapping.upload(ids.data(), ids.size() * sizeof(int32_t));
      refk.upload(f.k.data(), f.k.size() * 2);
      refv.upload(f.v.data(), f.v.size() * 2);
      auto one = f.shape;
      one.requests = 1;
      // Deliberately reject each speculative tail, overwrite it on the next step,
      // and cross both the live-window and expanded-ring boundaries.
      for (int pos = 0; pos < 520; ++pos) {
        for (int row = 0; row < 2 * width; ++row)
          f.positions[row] = row == 2 * width - 1 && pos % 3 == 0 ? -1 : pos + row % width;
        f.prepare();
        dgpp::mimo_qkv_append(f.shape, f.df.as<uint16_t>(), f.dfreq.as<float>(),
                              f.dpos.as<int64_t>(), f.dq.as<uint16_t>(), f.dk.as<uint16_t>(),
                              f.dv.as<uint16_t>(), f.dstatus.as<int32_t>(), f.stream, false,
                              mapping.as<int32_t>());
        if (window)
          dgpp::mimo_attention(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(),
                               f.dv.as<uint16_t>(), f.dpos.as<int64_t>(), f.dsinks.as<uint16_t>(),
                               f.dout.as<uint16_t>(), f.stream, false, nullptr,
                               mapping.as<int32_t>());
        else
          dgpp::mimo_attention_decode(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(),
                                      f.dv.as<uint16_t>(), f.dpos.as<int64_t>(), nullptr,
                                      f.dout.as<uint16_t>(), scores.as<float>(), f.stream,
                                      mapping.as<int32_t>());
        for (int row = 0; row < 2 * width; ++row) {
          auto* k = refk.as<uint16_t>() + size_t(ids[row]) * one.capacity * one.k_width();
          auto* v = refv.as<uint16_t>() + size_t(ids[row]) * one.capacity * one.v_width();
          auto* q = refq.as<uint16_t>() + size_t(row) * one.q_width();
          dgpp::mimo_qkv_append(one, f.df.as<uint16_t>() + size_t(row) * one.fused_width(),
                                f.dfreq.as<float>(), f.dpos.as<int64_t>() + row, q, k, v,
                                f.dstatus.as<int32_t>() + row, f.stream);
          dgpp::mimo_attention(one, q, k, v, f.dpos.as<int64_t>() + row,
                               window ? f.dsinks.as<uint16_t>() : nullptr,
                               refout.as<uint16_t>() + size_t(row) * one.q_heads * 128, f.stream);
        }
        if (pos < 2 || pos % 127 == 0 || pos >= 510) {
          DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
          refout.download(f.output.data(), f.output.size() * 2);
          f.compare(!window);
        }
      }
    }
  }
}

DGPP_TEST(mimo_native_mtp_history_shift_padding_reorder_and_wrap) {
  constexpr int width = 16, capacity = 16, rows = 8;
  DevBuf history(2 * capacity * width * 2), hidden(rows * width * 2),
      output(rows * width * 2), positions(rows * 8), translated(rows * 8), ids(rows * 4);
  auto stream = dgpp::kda_test::test_stream();
  std::vector<uint16_t> expected(2 * capacity * width, 0x1234), source(rows * width), got(rows * width);
  history.upload(expected.data(), expected.size() * 2);
  std::vector<int32_t> mapping{1, 1, 1, 1, 0, 0, 0, -1};
  ids.upload(mapping.data(), rows * 4);
  // Replay rejected tails at the same positions and cross the ring boundary.
  for (int start : {0, 1, 1, 13, 16, 16}) {
    for (int layer : {0, 1}) {
      std::vector<int64_t> pos{start, start+1, start+2, -1, start, start+1, -1, -1};
      for (int r = 0; r < rows; ++r)
        for (int i = 0; i < width; ++i) source[r * width + i] = uint16_t(100 * start + r * width + i + layer);
      hidden.upload(source.data(), source.size() * 2);
      positions.upload(pos.data(), rows * 8);
      dgpp::mimo_mtp_history_store(history.as<uint16_t>(), hidden.as<uint16_t>(), positions.as<int64_t>(),
                                   ids.as<int32_t>(), rows, width, capacity, layer, stream);
      for (int r = 0; r < rows; ++r)
        if (mapping[r] >= 0 && pos[r] >= layer)
          for (int i = 0; i < width; ++i)
            expected[(mapping[r] * capacity + pos[r] % capacity) * width + i] = source[r * width + i];
      // Head d consumes the backbone row shifted by d, with its own
      // attention origin shifted by depth. Invalid early rows stay padded.
      dgpp::mimo_mtp_history_gather(history.as<uint16_t>(), output.as<uint16_t>(), positions.as<int64_t>(),
                                    translated.as<int64_t>(), ids.as<int32_t>(), rows, width, capacity, layer+1, stream);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      output.download(got.data(), got.size() * 2);
      std::vector<int64_t> physical(rows);
      translated.download(physical.data(), rows * 8);
      for (int r = 0; r < rows; ++r) {
        const bool valid = mapping[r] >= 0 && pos[r] >= layer+1;
        if (physical[r] != (valid ? pos[r] - layer-1 : -1))
          throw std::runtime_error("native MTP attention origin/padding mismatch");
        for (int i = 0; i < width; ++i) {
          const uint16_t want = valid ? expected[(mapping[r] * capacity + (pos[r]-layer-1) % capacity) * width + i] : 0;
          if (got[r * width + i] != want) throw std::runtime_error("native MTP hidden alignment mismatch");
        }
      }
      std::vector<uint16_t> actual(expected.size());
      history.download(actual.data(), actual.size() * 2);
      if (actual != expected) throw std::runtime_error("native MTP history padding wrote stale state");
    }
  }
}

DGPP_TEST(mimo_cuda_online_mapped_decode_graph_and_padding) {
  Fixture f({4, 4, 2, 263, 128});
  uint32_t rng = 23;
  auto fill = [&](std::vector<uint16_t>& data) {
    for (auto& x : data) {
      rng = rng * 1664525u + 1013904223u;
      x = dgpp::float_to_bf16_bits(float(int(rng >> 16) - 32768) / 16384.f);
    }
  };
  fill(f.q); fill(f.k); fill(f.v);
  f.dq.upload(f.q.data(), f.q.size() * 2);
  f.dk.upload(f.k.data(), f.k.size() * 2);
  f.dv.upload(f.v.data(), f.v.size() * 2);
  DevBuf ids(4 * sizeof(int32_t));
  const int32_t mapping[] = {2, 0, 3, 1};
  ids.upload(mapping, sizeof(mapping));
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  DGPP_CUDA_OK(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeGlobal));
  dgpp::mimo_attention_online_decode(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(),
      f.dv.as<uint16_t>(), f.dpos.as<int64_t>(), f.dsinks.as<uint16_t>(),
      f.dout.as<uint16_t>(), f.stream, ids.as<int32_t>());
  DGPP_CUDA_OK(cudaStreamEndCapture(f.stream, &graph));
  size_t nodes = 0;
  DGPP_CUDA_OK(cudaGraphGetNodes(graph, nullptr, &nodes));
  if (nodes != 1) throw std::runtime_error("online decode must capture one kernel");
  DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (int step : {0, 1}) {
    f.positions = {526 + step, -1, 129 + step, 0};
    f.dpos.upload(f.positions.data(), f.positions.size() * 8);
    dgpp::mimo_attention(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(),
        f.dv.as<uint16_t>(), f.dpos.as<int64_t>(), f.dsinks.as<uint16_t>(),
        f.dout.as<uint16_t>(), f.stream, false, nullptr, ids.as<int32_t>());
    DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
    std::vector<uint16_t> ref(f.output.size()), got(ref.size());
    f.dout.download(ref.data(), ref.size() * 2);
    DGPP_CUDA_OK(cudaGraphLaunch(executable, f.stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
    f.dout.download(got.data(), got.size() * 2);
    double error = 0, norm = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      const double a = dgpp::bf16_bits_to_float(got[i]), b = dgpp::bf16_bits_to_float(ref[i]);
      if (!std::isfinite(a)) throw std::runtime_error("nonfinite online decode");
      if (i / (4 * 128) == 1 && got[i]) throw std::runtime_error("online padding nonzero");
      error += (a - b) * (a - b); norm += b * b;
    }
    const double rel = std::sqrt(error / std::max(norm, 1e-30));
    std::cout << "online mapped decode step=" << step << " rel_l2=" << rel << '\n';
    if (rel > 0.025) throw std::runtime_error("online decode synthetic L2 gate");
  }
  DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
  DGPP_CUDA_OK(cudaGraphDestroy(graph));
}

DGPP_TEST(mimo_cuda_bounded_random_serial_parallel_oracles) {
  const bool small = std::getenv("DGPP_MIMO_ATTN_SANITIZER_SMALL") != nullptr;
  for (int window : {0, 128}) {
    const int capacity = window ? 263 : (small ? 512 : 8192);
    const int end = window ? 800 : capacity;
    Fixture f({17, 4, 2, capacity, window}, true);
    DevBuf scores(size_t(17) * 4 * capacity * 6);
    uint32_t rng = 47;
    for (auto* data : {&f.q, &f.k, &f.v}) {
      for (auto& x : *data) {
        rng = rng * 1664525u + 1013904223u;
        x = dgpp::float_to_bf16_bits(float(int(rng >> 16) - 32768) / 16384.f);
      }
    }
    for (int r = 0; r < 17; ++r) f.positions[r] = end - 17 + r;
    f.dq.upload(f.q.data(), f.q.size() * 2);
    f.dk.upload(f.k.data(), f.k.size() * 2);
    f.dv.upload(f.v.data(), f.v.size() * 2);
    f.dpos.upload(f.positions.data(), f.positions.size() * 8);
    for (bool sink : {false, true}) {
      std::vector<uint16_t> serial(f.output.size()), parallel(serial.size());
      for (bool par : {false, true}) {
        dgpp::mimo_attention_prefill(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(),
            f.dv.as<uint16_t>(), f.dpos.as<int64_t>(), sink ? f.dsinks.as<uint16_t>() : nullptr,
            f.dout.as<uint16_t>(), scores.as<float>(), end, f.stream, par, true, true);
        DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
        f.dout.download((par ? parallel : serial).data(), serial.size() * 2);
      }
      for (bool online : {false, true}) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        DGPP_CUDA_OK(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeGlobal));
        dgpp::mimo_attention_bounded_prefill(f.shape, f.dq.as<uint16_t>(), f.dk.as<uint16_t>(),
            f.dv.as<uint16_t>(), f.dpos.as<int64_t>(), sink ? f.dsinks.as<uint16_t>() : nullptr,
            f.dout.as<uint16_t>(), end, f.stream, online);
        DGPP_CUDA_OK(cudaStreamEndCapture(f.stream, &graph));
        size_t nodes = 0;
        DGPP_CUDA_OK(cudaGraphGetNodes(graph, nullptr, &nodes));
        if (nodes != 1) throw std::runtime_error("bounded prefill must capture one kernel");
        DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        DGPP_CUDA_OK(cudaGraphLaunch(executable, f.stream));
        DGPP_CUDA_OK(cudaStreamSynchronize(f.stream));
        std::vector<uint16_t> got(serial.size());
        f.dout.download(got.data(), got.size() * 2);
        for (bool par : {false, true}) {
          const auto& ref = par ? parallel : serial;
          double error = 0, norm = 0, maxabs = 0;
          int changed = 0;
          for (size_t i = 0; i < ref.size(); ++i) {
            const double a = dgpp::bf16_bits_to_float(got[i]), b = dgpp::bf16_bits_to_float(ref[i]);
            if (!std::isfinite(a)) throw std::runtime_error("nonfinite bounded prefill");
            error += (a - b) * (a - b); norm += b * b;
            maxabs = std::max(maxabs, std::abs(a - b)); changed += got[i] != ref[i];
          }
          const double rel = std::sqrt(error / std::max(norm, 1e-30));
          std::cout << "bounded random capacity=" << capacity << " window=" << window
                    << " sink=" << sink << " online=" << online << " parallel_ref=" << par
                    << " rel_l2=" << rel << " maxabs=" << maxabs << " changed=" << changed << '\n';
          if (rel > (online ? 0.025 : 0.005)) throw std::runtime_error("bounded random L2 gate");
        }
        DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
        DGPP_CUDA_OK(cudaGraphDestroy(graph));
      }
    }
  }
}
