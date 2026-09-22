// Real-checkpoint FP8 residency / graph / numerical / timing probe.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "common/dtypes.hpp"
#include "kernels/fp8_dequant.hpp"
#include "kernels/gemm.hpp"
#include "kernels/scale_gemm.hpp"
#include "models/mimo/weights.hpp"

static void ok(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
  void* p = nullptr;
  explicit Buffer(size_t bytes) { ok(cudaMalloc(&p, bytes)); }
  ~Buffer() { cudaFree(p); }
  template <class T>
  T* as() {
    return static_cast<T*>(p);
  }
  template <class T>
  void put(const std::vector<T>& v) {
    ok(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
};
int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("usage: mimo_fp8_dense_check CHECKPOINT RANK WORLD");
    const int rank = std::stoi(argv[2]), world = std::stoi(argv[3]);
    dgpp::MimoCheckpointWeights reader(argv[1]);
    dgpp::CublasLtGemm gemm;
    cudaStream_t stream;
    ok(cudaStreamCreate(&stream));
    for (int layer : {0, 1})
      for (const std::string site : {"qkv", "gate", "up", "down"}) {
        if (layer == 1 && site != "qkv") continue;
        const auto name = "model.layers." + std::to_string(layer) + ".mlp." + site + "_proj.weight";
        auto packed =
            site == "qkv" ? reader.qkv_fp8(layer, rank, world) : reader.fp8(name, rank, world);
        auto ref = site == "qkv" ? reader.qkv(layer, rank, world, 0, reader.config().hidden_size)
                                 : reader.bf16(name, rank, world);
        const int n = packed.rows, k = packed.cols;
        // Exhaustive host oracle verifies chunk permutations and scale restarts.
        for (int r = 0; r < n; ++r)
          for (int c = 0; c < k; ++c) {
            const auto v = dgpp::float_to_bf16_bits(
                dgpp::fp8_e4m3_bits_to_float(packed.payload[size_t(r) * k + c]) *
                packed.scales[size_t(r / 64) * (k / 64) + c / 64]);
            if (v != ref.values[size_t(r) * k + c])
              throw std::runtime_error("host scale/layout mismatch");
          }
        Buffer w(packed.payload.size()), scales(packed.scales.size() * 4),
            bf(ref.values.size() * 2);
        w.put(packed.payload);
        scales.put(packed.scales);
        bf.put(ref.values);
        Buffer transient(ref.values.size() * 2);
        for (int m : {1, 4, 8, 32, 128, 512}) {
          std::vector<uint16_t> x(size_t(m) * k);
          for (size_t i = 0; i < x.size(); ++i)
            x[i] = dgpp::float_to_bf16_bits(std::sin(float(i % 10007) * .017f));
          Buffer act(x.size() * 2), out(size_t(m) * n * 2), baseline(size_t(m) * n * 2);
          act.put(x);
          auto run = [&] {
            dgpp::launch_scale_gemm_grid_bf16(act.as<uint16_t>(), k, w.as<uint8_t>(),
                                              scales.as<float>(), out.as<uint16_t>(), m, n, k,
                                              stream, 0, 6, 6, m > dgpp::dense_gemv_rows());
          };
          run();
          gemm.matmul(act.p, bf.p, baseline.p, m, n, k, dgpp::DType::BF16, dgpp::GemmOut::BF16, k,
                      nullptr, 0, stream);
          ok(cudaStreamSynchronize(stream));
          std::vector<uint16_t> y(size_t(m) * n), z(y.size());
          ok(cudaMemcpy(y.data(), out.p, y.size() * 2, cudaMemcpyDeviceToHost));
          ok(cudaMemcpy(z.data(), baseline.p, z.size() * 2, cudaMemcpyDeviceToHost));
          double sq = 0, refsq = 0, maxerr = 0, maxref = 0;
          for (size_t i = 0; i < y.size(); ++i) {
            double a = dgpp::bf16_bits_to_float(y[i]), b = dgpp::bf16_bits_to_float(z[i]);
            if (!std::isfinite(a) || !std::isfinite(b))
              throw std::runtime_error("nonfinite output");
            sq += (a - b) * (a - b);
            refsq += b * b;
            maxerr = std::max(maxerr, std::abs(a - b));
            maxref = std::max(maxref, std::abs(b));
          }
          double rel = std::sqrt(sq / std::max(refsq, 1e-30));
          if (rel > .01 || maxerr > .025 * maxref + 1e-3)
            throw std::runtime_error(
                "GPU projection tolerance exceeded: layer=" + std::to_string(layer) +
                " site=" + site + " m=" + std::to_string(m) + " rel_rms=" + std::to_string(rel) +
                " max_abs=" + std::to_string(maxerr));
          cudaGraph_t graph;
          ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
          run();
          ok(cudaStreamEndCapture(stream, &graph));
          size_t count = 0;
          ok(cudaGraphGetNodes(graph, nullptr, &count));
          std::vector<cudaGraphNode_t> nodes(count);
          ok(cudaGraphGetNodes(graph, nodes.data(), &count));
          if (count == 0) throw std::runtime_error("empty graph");
          for (auto node : nodes) {
            cudaGraphNodeType type;
            ok(cudaGraphNodeGetType(node, &type));
            if (type != cudaGraphNodeTypeKernel) throw std::runtime_error("non-kernel graph node");
          }
          cudaGraphExec_t exec;
          ok(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
          ok(cudaGraphLaunch(exec, stream));
          ok(cudaStreamSynchronize(stream));
          std::vector<uint16_t> replay(y.size());
          ok(cudaMemcpy(replay.data(), out.p, y.size() * 2, cudaMemcpyDeviceToHost));
          if (replay != y) throw std::runtime_error("graph replay mismatch");
          cudaEvent_t a, b;
          ok(cudaEventCreate(&a));
          ok(cudaEventCreate(&b));
          ok(cudaEventRecord(a, stream));
          for (int i = 0; i < 10; ++i) ok(cudaGraphLaunch(exec, stream));
          ok(cudaEventRecord(b, stream));
          ok(cudaEventSynchronize(b));
          float ms;
          ok(cudaEventElapsedTime(&ms, a, b));
          cudaGraph_t bf_graph;
          ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
          gemm.matmul(act.p, bf.p, baseline.p, m, n, k, dgpp::DType::BF16, dgpp::GemmOut::BF16, k,
                      nullptr, 0, stream);
          ok(cudaStreamEndCapture(stream, &bf_graph));
          cudaGraphExec_t bf_exec;
          ok(cudaGraphInstantiate(&bf_exec, bf_graph, nullptr, nullptr, 0));
          ok(cudaGraphLaunch(bf_exec, stream));
          ok(cudaEventRecord(a, stream));
          for (int i = 0; i < 10; ++i) ok(cudaGraphLaunch(bf_exec, stream));
          ok(cudaEventRecord(b, stream));
          ok(cudaEventSynchronize(b));
          float bf_ms;
          ok(cudaEventElapsedTime(&bf_ms, a, b));
          std::cout << "rank=" << rank << " layer=" << layer << " site=" << site << " m=" << m
                    << " rel_rms=" << rel << " max_abs=" << maxerr << " kernel_nodes=" << count
                    << " fp8_us=" << ms * 100 << " bf16_us=" << bf_ms * 100 << "\n";
          // Include fresh dequantization on EVERY timed invocation. No cached
          // BF16 shadow: the same bounded buffer is overwritten each time.
          if (m > 64) {
            auto hybrid = [&] {
              dgpp::launch_fp8_dequant_grid(w.as<uint8_t>(), scales.as<float>(),
                                            transient.as<uint16_t>(), n, k, 6, stream);
              gemm.matmul(act.p, transient.p, out.p, m, n, k, dgpp::DType::BF16,
                          dgpp::GemmOut::BF16, k, nullptr, 0, stream);
            };
            hybrid();
            ok(cudaStreamSynchronize(stream));
            std::vector<uint16_t> restored(ref.values.size());
            ok(cudaMemcpy(restored.data(), transient.p, restored.size() * 2,
                          cudaMemcpyDeviceToHost));
            if (restored != ref.values) throw std::runtime_error("GPU bridge dequant mismatch");
            ok(cudaMemcpy(replay.data(), out.p, replay.size() * 2, cudaMemcpyDeviceToHost));
            // Compare to the SAME zero-workspace GEMM as the serving model.
            gemm.matmul(act.p, bf.p, baseline.p, m, n, k, dgpp::DType::BF16, dgpp::GemmOut::BF16, k,
                        nullptr, 0, stream);
            ok(cudaStreamSynchronize(stream));
            ok(cudaMemcpy(z.data(), baseline.p, z.size() * 2, cudaMemcpyDeviceToHost));
            if (replay != z) throw std::runtime_error("hybrid BF16 baseline mismatch");
            ok(cudaEventRecord(a, stream));
            for (int i = 0; i < 10; ++i) hybrid();
            ok(cudaEventRecord(b, stream));
            ok(cudaEventSynchronize(b));
            float hybrid_ms;
            ok(cudaEventElapsedTime(&hybrid_ms, a, b));
            ok(cudaEventRecord(a, stream));
            for (int i = 0; i < 10; ++i)
              gemm.matmul(act.p, bf.p, baseline.p, m, n, k, dgpp::DType::BF16, dgpp::GemmOut::BF16,
                          k, nullptr, 0, stream);
            ok(cudaEventRecord(b, stream));
            ok(cudaEventSynchronize(b));
            float eager_bf_ms;
            ok(cudaEventElapsedTime(&eager_bf_ms, a, b));
            std::cout << "hybrid rank=" << rank << " layer=" << layer << " site=" << site
                      << " m=" << m << " dequant_plus_gemm_us=" << hybrid_ms * 100
                      << " eager_bf16_us=" << eager_bf_ms * 100
                      << " scratch_bytes=" << ref.values.size() * 2 << "\n";
          }
          cudaGraphExecDestroy(bf_exec);
          cudaGraphDestroy(bf_graph);
          cudaEventDestroy(a);
          cudaEventDestroy(b);
          cudaGraphExecDestroy(exec);
          cudaGraphDestroy(graph);
        }
      }
    cudaStreamDestroy(stream);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
