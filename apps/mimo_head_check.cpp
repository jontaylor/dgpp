// Real-checkpoint vocabulary head comparison. No backbone or fabric required.
// Optional input is the complete post-final-norm [rows, hidden] BF16 buffer.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "models/mimo/head.hpp"
#include "models/mimo/loader.hpp"

int main(int argc, char** argv) {
  try {
    if (argc < 6 || argc > 7)
      throw std::invalid_argument(
          "usage: mimo_head_check CHECKPOINT ROWS RANK WORLD OUTPUT_PREFIX [HIDDEN_BF16]");
    const int rows = std::stoi(argv[2]), rank = std::stoi(argv[3]), world = std::stoi(argv[4]);
    if (rows < 1 || rows > 4096 || world < 1 || rank < 0 || rank >= world)
      throw std::invalid_argument("invalid rows/rank/world");
    dgpp::MimoCheckpointWeights reader(argv[1]);
    reader.config().validate_tp(world);
    auto weight = reader.bf16("lm_head.weight", rank, world);
    const int H = static_cast<int>(weight.cols), V = static_cast<int>(weight.rows);
    std::cout << "rows=" << rows << " rank=" << rank << " world=" << world
              << " hidden=" << H << " vocab=" << V << '\n';
    std::vector<uint16_t> hidden(size_t(rows) * H);
    if (argc == 7) {
      std::ifstream input(argv[6], std::ios::binary);
      if (!input.read(reinterpret_cast<char*>(hidden.data()), hidden.size() * 2) ||
          input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("hidden input must contain exactly rows * hidden BF16 values");
    } else {
      for (size_t i = 0; i < hidden.size(); ++i)
        hidden[i] = dgpp::float_to_bf16_bits(std::sin(float(i % 8191) * 0.017f) * 1.4f);
    }
    dgpp::LayerBump storage;
    const size_t logits_bytes = size_t(rows) * V * sizeof(float);
    storage.init(weight.values.size() * 2 + hidden.size() * 2 + 2 * logits_bytes + 1024);
    auto* w = static_cast<uint16_t*>(storage.alloc(weight.values.size() * 2));
    auto* h = static_cast<uint16_t*>(storage.alloc(hidden.size() * 2));
    auto* ref = static_cast<float*>(storage.alloc(logits_bytes));
    auto* out = static_cast<float*>(storage.alloc(logits_bytes));
    DGPP_CUDA_OK(cudaMemcpy(w, weight.values.data(), weight.values.size() * 2, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(h, hidden.data(), hidden.size() * 2, cudaMemcpyHostToDevice));
    weight.values.clear();
    dgpp::CublasLtGemm gemm;
    gemm.matmul(h, w, ref, rows, V, H, dgpp::DType::BF16, dgpp::GemmOut::F32,
                 H, nullptr, 0, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> baseline(size_t(rows) * V), candidate(baseline.size());
    DGPP_CUDA_OK(cudaMemcpy(baseline.data(), ref, logits_bytes, cudaMemcpyDeviceToHost));
    bool exact = true;
    for (int mode = 0; mode < 3; ++mode) {
      // NaN sentinels catch overwrites of unused logit rows.
      DGPP_CUDA_OK(cudaMemset(out, 0xff, logits_bytes));
      dgpp::mimo_project_head(gemm, h, w, out, rows, V, H, mode == 2, mode == 1, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaMemcpy(candidate.data(), out, logits_bytes, cudaMemcpyDeviceToHost));
      std::vector<uint16_t> after(hidden.size());
      DGPP_CUDA_OK(cudaMemcpy(after.data(), h, after.size() * 2, cudaMemcpyDeviceToHost));
      if (after != hidden) throw std::runtime_error("projection changed hidden rows");
      const size_t first = mode == 0 ? size_t(rows - 1) * V : 0;
      for (size_t i = 0; i < first; ++i)
        if (std::bit_cast<uint32_t>(candidate[i]) != 0xffffffffu)
          throw std::runtime_error("projection overwrote an unused logit row");
      double diff2 = 0, ref2 = 0, max_abs = 0;
      size_t different = 0;
      for (size_t i = first; i < candidate.size(); ++i) {
        if (!std::isfinite(candidate[i]) || !std::isfinite(baseline[i]))
          throw std::runtime_error("nonfinite logits");
        const double d = double(candidate[i]) - baseline[i];
        diff2 += d * d;
        ref2 += double(baseline[i]) * baseline[i];
        max_abs = std::max(max_abs, std::abs(d));
        different += std::bit_cast<uint32_t>(candidate[i]) != std::bit_cast<uint32_t>(baseline[i]);
      }
      std::cout << "mode=" << (mode == 0 ? "final" : mode == 1 ? "all_rows" : "decode")
                << " hidden_exact=true unused_rows_untouched=true different=" << different
                << " max_abs=" << max_abs << " rel_l2=" << std::sqrt(diff2 / std::max(ref2, 1e-300))
                << '\n';
      if (mode != 0 && different) throw std::runtime_error("full-row path changed logits");
      exact &= different == 0;
      if (mode == 0) {
        const auto argmax = [first](const std::vector<float>& x) {
          return std::max_element(x.begin() + first, x.end()) - (x.begin() + first);
        };
        std::cout << "baseline_argmax=" << argmax(baseline) << " candidate_argmax=" << argmax(candidate) << '\n';
        for (bool original : {true, false}) {
          std::ofstream file(std::string(argv[5]) + (original ? ".baseline.f32" : ".candidate.f32"),
                               std::ios::binary);
          const auto& data = original ? baseline : candidate;
          file.write(reinterpret_cast<const char*>(data.data() + first), size_t(V) * sizeof(float));
          if (!file) throw std::runtime_error("cannot write logits output");
        }
      }
    }
    if (const char* bench = std::getenv("DGPP_MIMO_HEAD_BENCH"); bench && std::string(bench) == "1") {
      // Plans are already populated by the correctness checks. Warm both
      // shapes explicitly, then alternate pair order to expose repeat noise.
      auto project = [&](bool full) {
        dgpp::mimo_project_head(gemm, h, w, out, rows, V, H, false, full, nullptr);
      };
      for (int i = 0; i < 2; ++i) {
        project(true);
        project(false);
      }
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      cudaEvent_t start, stop;
      DGPP_CUDA_OK(cudaEventCreate(&start));
      DGPP_CUDA_OK(cudaEventCreate(&stop));
      auto timed = [&](bool full) {
        DGPP_CUDA_OK(cudaEventRecord(start, nullptr));
        project(full);
        DGPP_CUDA_OK(cudaEventRecord(stop, nullptr));
        DGPP_CUDA_OK(cudaEventSynchronize(stop));
        float ms = 0;
        DGPP_CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
      };
      float full_sum = 0, final_sum = 0;
      for (int i = 0; i < 5; ++i) {
        float full_ms, final_ms;
        if (i % 2 == 0) {
          full_ms = timed(true);
          final_ms = timed(false);
        } else {
          final_ms = timed(false);
          full_ms = timed(true);
        }
        full_sum += full_ms;
        final_sum += final_ms;
        std::cout << "head_bench repetition=" << i + 1 << " full_ms=" << full_ms
                  << " final_ms=" << final_ms << '\n';
      }
      DGPP_CUDA_OK(cudaEventDestroy(start));
      DGPP_CUDA_OK(cudaEventDestroy(stop));
      std::cout << "head_bench repetitions=5 warmup=2 full_mean_ms=" << full_sum / 5
                << " final_mean_ms=" << final_sum / 5 << '\n';
    }
    // No numerical tolerance is silently introduced by this probe. Exit 2
    // means a shape-induced difference requiring the existing numerical gate.
    return exact ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
