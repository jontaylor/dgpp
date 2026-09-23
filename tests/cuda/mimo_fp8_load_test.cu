#include "common/cuda_check.hpp"
#include "kernels/mimo_attn.hpp"
#include "kernels/mimo_cache.cuh"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

// Kept as separate kernels so PTX/ptxas evidence is reproducible from this test.
__global__ void fp8_load_original(const uint8_t* input, uint16_t* output) {
  const int i = threadIdx.x;
  __nv_fp8_e4m3 value;
  value.__x = input[i];
  output[i] = dgpp::float_to_bf16_bits(static_cast<float>(value));
}
__global__ void fp8_load_exact(const uint8_t* input, uint16_t* output) {
  const int i = threadIdx.x;
  __nv_fp8_e4m3 value;
  value.__x = input[i];
  output[i] = dgpp::mimo_fp8_expanded_to_bf16(static_cast<float>(value));
}
__global__ void fp8_load_selected(const uint8_t* input, const uint16_t* bf16, uint16_t* output) {
  const int i = threadIdx.x;
  output[i] = dgpp::mimo_cache_load(input, i, true);
  output[256 + i] = dgpp::mimo_cache_load(bf16, i, false);
}
int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 2;
  try {
    if (dgpp::mimo_fp8_fast_load_build() != bool(DGPP_MIMO_FP8_KV_FAST_LOAD))
      throw std::runtime_error("probe and linked attention library have different fast-load builds");
    std::vector<uint8_t> input(256);
    std::vector<uint16_t> bf16(256), output(1024);
    for (int code = 0; code < 256; ++code) {
      input[code] = uint8_t(code);
      bf16[code] = uint16_t(code * 257);  // includes signed-zero/NaN payload passthrough
    }
    bf16[1] = 0x8000;
    bf16[2] = 0xffc1;
    bf16[3] = 0x7fff;
    uint8_t* source;
    uint16_t *bf, *dest;
    DGPP_CUDA_OK(cudaMalloc(&source, 256));
    DGPP_CUDA_OK(cudaMalloc(&bf, 512));
    DGPP_CUDA_OK(cudaMalloc(&dest, 2048));
    DGPP_CUDA_OK(cudaMemcpy(source, input.data(), 256, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(bf, bf16.data(), 512, cudaMemcpyHostToDevice));
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    DGPP_CUDA_OK(cudaStreamCreate(&stream));
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    fp8_load_original<<<1, 256, 0, stream>>>(source, dest);
    fp8_load_exact<<<1, 256, 0, stream>>>(source, dest + 256);
    fp8_load_selected<<<1, 256, 0, stream>>>(source, bf, dest + 512);
    DGPP_CUDA_OK(cudaGetLastError());
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) {
      if (replay) {
        std::reverse(input.begin(), input.end());
        DGPP_CUDA_OK(cudaMemcpy(source, input.data(), 256, cudaMemcpyHostToDevice));
      }
      DGPP_CUDA_OK(cudaGraphLaunch(executable, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      DGPP_CUDA_OK(cudaMemcpy(output.data(), dest, 2048, cudaMemcpyDeviceToHost));
      for (int i = 0; i < 256; ++i) {
        if (output[i] != output[256 + i] || output[i] != output[512 + i])
          throw std::runtime_error("GPU load differs at E4M3 code " + std::to_string(input[i]));
        if (output[768 + i] != bf16[i]) throw std::runtime_error("BF16 passthrough changed");
        if ((input[i] & 127) != 127) {
          const auto expected = dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(input[i]));
          if (output[i] != expected) throw std::runtime_error("GPU load differs from finite-code CPU reference");
        } else if ((output[i] & 0x7fff) != 0x7fc0) {
          throw std::runtime_error("GPU load changed canonical quiet NaN");
        }
      }
    }
    DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
    DGPP_CUDA_OK(cudaGraphDestroy(graph));
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
    DGPP_CUDA_OK(cudaFree(dest));
    DGPP_CUDA_OK(cudaFree(bf));
    DGPP_CUDA_OK(cudaFree(source));
    std::cout << "PASS all256 E4M3 codes, NaN-sign parity, signed zero, BF16 passthrough, graph replay; fast_load="
              << DGPP_MIMO_FP8_KV_FAST_LOAD << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
