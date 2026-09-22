#include "kernels/mimo_dflash.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "loaders/weight_build.hpp"
using namespace dgpp;
int main() {
  try {
    constexpr int rows = 24, capacity = 2048, width = 128;
    LayerBump arena;
    arena.init(4u << 20);
    auto upload = [&]<class T>(const std::vector<T>& v) {
      auto* p = static_cast<T*>(arena.alloc(v.size() * sizeof(T)));
      DGPP_CUDA_OK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
      return p;
    };
    auto q = upload(std::vector<uint16_t>(rows * width, 0));
    auto k = upload(std::vector<uint16_t>(rows * width, 0));
    std::vector<uint16_t> hv(rows * width);
    for (int r = 0; r < rows; r++)
      for (int d = 0; d < width; d++) hv[r * width + d] = float_to_bf16_bits(float(r % 8 + 1) / 8);
    auto v = upload(hv);
    auto kc = upload(std::vector<uint16_t>(2 * capacity * width, 0));
    std::vector<uint16_t> hvc(2 * capacity * width);
    for (int req = 0; req < 2; req++)
      for (int j = 0; j < capacity * width; j++)
        hvc[req * capacity * width + j] = float_to_bf16_bits((req + 1) * 0.25f);
    auto vc = upload(hvc);
    auto sink = upload(std::vector<uint16_t>(1, 0));
    auto out = upload(std::vector<uint16_t>(rows * width, 0xffff));
    std::vector<int64_t> hp(rows);
    std::vector<int32_t> hr(rows);
    for (int i = 0; i < rows; i++) {
      hp[i] = i >= 16 ? -1 : 4094 + i % 8;
      hr[i] = i >= 16 ? -1 : 1 - i / 8;
    }
    auto pos = upload(hp);
    auto ids = upload(hr);
    // Request map swaps planes. Blocks cross the physical ring boundary. The
    // third group is padding with an invalid request ID, and must never read it.
    dflash_attention(q, k, v, kc, vc, sink, pos, ids, out, 3, capacity, 1, 1, nullptr);
    std::vector<uint16_t> actual(rows * width);
    DGPP_CUDA_OK(cudaMemcpy(actual.data(), out, actual.size() * 2, cudaMemcpyDeviceToHost));
    for (int r = 0; r < rows; r++) {
      float expected = 0;
      if (r < 16) {
        int context = 1023 - r % 8;
        float sum = context * (hr[r] + 1) * 0.25f;
        for (int j = 0; j < 8; j++)
          sum += bf16_bits_to_float(float_to_bf16_bits(float(j + 1) / 8 * 0.612f));
        expected = sum / (context + 9);
      }
      expected = bf16_bits_to_float(float_to_bf16_bits(expected));
      for (int d = 0; d < width; d++)
        if (std::abs(bf16_bits_to_float(actual[r * width + d]) - expected) > 0.002f)
          throw std::runtime_error("block attention window/sink/padding mismatch");
    }
    // All-negative appends leave both physical cache planes byte-for-byte intact.
    auto negative = upload(std::vector<int64_t>(rows, -1));
    dflash_append(k, v, kc, vc, negative, ids, rows, capacity, 1, nullptr);
    std::vector<uint16_t> after(hvc.size());
    DGPP_CUDA_OK(cudaMemcpy(after.data(), vc, after.size() * 2, cudaMemcpyDeviceToHost));
    if (after != hvc) throw std::runtime_error("padding modified context cache");
    std::cout << "PASS DFlash block attention: noncausal block, window/ring boundary, swapped "
                 "slots, sink denominator, padding isolation\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
