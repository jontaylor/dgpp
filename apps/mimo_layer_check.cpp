// Focused real-checkpoint layer probe. Requires exclusive/idle GPU access.
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "common/dtypes.hpp"
#include "kernels/glm_norm.hpp"
#include "models/mimo/layers.hpp"

namespace dgpp {
struct MimoLayerProbe {
  static void write_attention(const MimoDecoderLayer& layer, int rows, std::ofstream& out) {
    std::vector<uint16_t> values(size_t(rows) * layer.shape_.q_heads * 128);
    DGPP_CUDA_OK(cudaMemcpy(values.data(), layer.attn_, values.size() * 2, cudaMemcpyDeviceToHost));
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * 2);
  }
};
}  // namespace dgpp

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 6)
      throw std::invalid_argument(
          "usage: mimo_layer_check SNAPSHOT LAYER OUTPUT_BF16 [CHUNK] [TOKENS]");
    const int chunk = argc > 4 ? std::stoi(argv[4]) : 1;
    const int tokens = argc > 5 ? std::stoi(argv[5]) : 3;
    if (chunk < 1 || tokens < 1 || chunk > 4096)
      throw std::invalid_argument("invalid chunk/tokens");
    const bool tensor_decode = std::getenv("DGPP_MIMO_LAYER_DECODE") != nullptr;
    if (tensor_decode && chunk != 1) throw std::invalid_argument("decode probe requires chunk 1");
    dgpp::LayerBump request_map;
    if (tensor_decode) {
      request_map.init(256);
      DGPP_CUDA_OK(cudaMemset(request_map.base, 0, 256));
    }
    const int layer = std::stoi(argv[2]);
    dgpp::MimoDeviceLoader loader(argv[1], 0, 1);
    auto w = loader.load_layer(layer);
    dgpp::CublasLtGemm gemm;
    if (const char* rows = std::getenv("DGPP_MIMO_LAYER_DENSE_ROWS"))
      gemm.set_decode_rows(std::stoi(rows));
    if (std::getenv("DGPP_MIMO_LAYER_MMA")) gemm.set_decode_mma(true, 8);
    dgpp::LayerBump attention_scores;
    if (dgpp::mimo_split_online_attention_enabled())
      attention_scores.init(dgpp::mimo_online_partial_bytes(std::min(chunk, dgpp::mimo_split_tile_rows),
                             w.q_heads, tokens));
    else if (!dgpp::mimo_bounded_attention_enabled())
      attention_scores.init(size_t(std::min(chunk, dgpp::MimoDecoderLayer::attention_tile_rows)) *
                          w.q_heads * std::max(tokens, dgpp::mimo_ring_capacity(chunk)) * 6);
    dgpp::MimoDecoderLayer block(w, loader.config(), 1, tokens, gemm, nullptr, 0, chunk,
                                 static_cast<float*>(attention_scores.base));
    const bool draft = loader.config().is_mtp(layer);
    dgpp::LayerBump fusion;
    uint16_t *embed = nullptr, *fused = nullptr, *hidden = nullptr;
    int64_t* token_ids = nullptr;
    if (draft) {
      fusion.init(size_t(chunk) * (4 * 8192 + 256));
      embed = static_cast<uint16_t*>(fusion.alloc(chunk * 8192));
      fused = static_cast<uint16_t*>(fusion.alloc(chunk * 16384));
      hidden = static_cast<uint16_t*>(fusion.alloc(chunk * 8192));
      token_ids = static_cast<int64_t*>(fusion.alloc(chunk * 8));
    }
    const auto shape = block.shape();
    dgpp::LayerBump state;
    state.init(size_t(chunk) * (256 + 8192 + 256) +
               size_t(shape.capacity) * (shape.k_width() + shape.v_width()) * shape.cache_element_bytes());
    auto* pos = static_cast<int64_t*>(state.alloc(chunk * 8));
    auto* x = static_cast<uint16_t*>(state.alloc(chunk * 8192));
    auto* status = static_cast<int32_t*>(state.alloc(chunk * 4));
    auto* k = state.alloc(size_t(shape.capacity) * shape.k_width() * shape.cache_element_bytes());
    auto* v = state.alloc(size_t(shape.capacity) * shape.v_width() * shape.cache_element_bytes());
    DGPP_CUDA_OK(cudaMemset(k, 0, size_t(shape.capacity) * shape.k_width() * shape.cache_element_bytes()));
    DGPP_CUDA_OK(cudaMemset(v, 0, size_t(shape.capacity) * shape.v_width() * shape.cache_element_bytes()));
    const bool cache_prefix = std::getenv("DGPP_MIMO_LAYER_CACHE_ONLY_PREFIX") != nullptr;
    if (cache_prefix && !draft)
      throw std::invalid_argument("cache-only prefix probe requires an MTP head");
    std::ofstream out(argv[3], std::ios::binary);
    std::ofstream attention_out;
    if (std::getenv("DGPP_MIMO_LAYER_ATTN_DUMP"))
      attention_out.open(std::string(argv[3]) + ".attention", std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output");
    for (int64_t p = 0; p < tokens; p += chunk) {
      const int count = std::min<int64_t>(chunk, tokens - p);
      std::vector<uint16_t> input(size_t(count) * 4096);
      std::vector<int64_t> positions(count);
      for (int t = 0; t < count; ++t) {
        positions[t] = p + t;
        for (int i = 0; i < 4096; ++i)
          input[t * 4096 + i] =
              dgpp::float_to_bf16_bits(float((i * 7 + (p + t) * 13) % 101 - 50) / 64);
      }
      DGPP_CUDA_OK(cudaMemcpy(x, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
      DGPP_CUDA_OK(cudaMemcpy(pos, positions.data(), count * 8, cudaMemcpyHostToDevice));
      if (draft) {
        // Independent deterministic embedding/hidden fixture, using real fusion weights.
        std::vector<uint16_t> embeddings(input.size());
        std::vector<int64_t> ids(count);
        for (int t = 0; t < count; ++t) {
          ids[t] = t;
          for (int i = 0; i < 4096; ++i)
            embeddings[t * 4096 + i] =
                dgpp::float_to_bf16_bits(float((i * 11 + (p + t) * 19) % 97 - 48) / 64);
        }
        DGPP_CUDA_OK(
            cudaMemcpy(embed, embeddings.data(), embeddings.size() * 2, cudaMemcpyHostToDevice));
        DGPP_CUDA_OK(cudaMemcpy(token_ids, ids.data(), count * 8, cudaMemcpyHostToDevice));
        dgpp::glm_mtp_input_bf16(embed, token_ids, x, nullptr, 0, w.enorm, w.hnorm, fused, count,
                                 4096, 1e-6, nullptr);
        gemm.matmul(fused, w.eh_proj, hidden, count, 4096, 8192, dgpp::DType::BF16,
                    dgpp::GemmOut::BF16, 8192, nullptr, 0, nullptr);
        DGPP_CUDA_OK(cudaMemcpy(x, hidden, input.size() * 2, cudaMemcpyDeviceToDevice));
      }
      const bool cache_only = cache_prefix && p + count < tokens;
      std::vector<uint16_t> before;
      if (cache_only) {
        before.resize(input.size());
        DGPP_CUDA_OK(cudaMemcpy(before.data(), x, input.size() * 2, cudaMemcpyDeviceToHost));
      }
      block.enqueue(x, pos, k, v, status, nullptr, nullptr, count, p + count, false,
                    tensor_decode ? static_cast<int32_t*>(request_map.base) : nullptr, cache_only);
      if (cache_only) {
        std::vector<uint16_t> after(input.size());
        DGPP_CUDA_OK(cudaMemcpy(after.data(), x, input.size() * 2, cudaMemcpyDeviceToHost));
        if (before != after) throw std::runtime_error("cache-only path mutated residual");
      }
      if (draft) {
        dgpp::glm_rmsnorm_bf16(x, w.final_norm, hidden, count, 4096, 1e-6, nullptr);
        DGPP_CUDA_OK(cudaMemcpy(x, hidden, input.size() * 2, cudaMemcpyDeviceToDevice));
      }
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      if (attention_out.is_open())
        dgpp::MimoLayerProbe::write_attention(block, count, attention_out);
      std::vector<int32_t> results(count);
      DGPP_CUDA_OK(cudaMemcpy(results.data(), status, count * 4, cudaMemcpyDeviceToHost));
      for (auto result : results)
        if (result) throw std::runtime_error("layer reported invalid position");
      DGPP_CUDA_OK(cudaMemcpy(input.data(), x, input.size() * 2, cudaMemcpyDeviceToHost));
      out.write(reinterpret_cast<const char*>(input.data()), input.size() * 2);
    }
    if (!out) throw std::runtime_error("cannot write output");
    if (std::getenv("DGPP_MIMO_LAYER_KV_DUMP")) {
      std::ofstream cache_out(std::string(argv[3]) + ".kv", std::ios::binary);
      for (auto [data, width] : {std::pair{k, shape.k_width()}, std::pair{v, shape.v_width()}}) {
        std::vector<uint16_t> cache(size_t(shape.capacity) * width);
        if (shape.fp8_cache) {
          std::vector<uint8_t> bytes(cache.size());
          DGPP_CUDA_OK(cudaMemcpy(bytes.data(), data, bytes.size(), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < bytes.size(); ++i)
            cache[i] = dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(bytes[i]));
        } else {
          DGPP_CUDA_OK(cudaMemcpy(cache.data(), data, cache.size() * 2, cudaMemcpyDeviceToHost));
        }
        cache_out.write(reinterpret_cast<const char*>(cache.data()), cache.size() * 2);
      }
      if (!cache_out) throw std::runtime_error("cannot write K/V output");
    }
    std::cout << "MiMo layer " << layer << ": " << tokens << " rows, chunk " << chunk << "; "
              << w.storage->capacity << " resident weight bytes, " << block.scratch_bytes()
              << " scratch bytes\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
