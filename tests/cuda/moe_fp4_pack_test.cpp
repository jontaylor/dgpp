#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "kernels/moe_fp4_pack.hpp"

namespace {
void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}
struct Buffer {
  uint8_t* allocation = nullptr;
  size_t bytes;
  explicit Buffer(size_t n) : bytes(n) {
    DGPP_CUDA_OK(cudaMalloc(&allocation, n + 32));
    DGPP_CUDA_OK(cudaMemset(allocation, 0xa5, n + 32));
  }
  ~Buffer() { cudaFree(allocation); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  uint8_t* data() const { return allocation + 16; }
  std::vector<uint8_t> read() const {
    std::vector<uint8_t> all(bytes + 32);
    DGPP_CUDA_OK(cudaMemcpy(all.data(), allocation, all.size(), cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < 16; ++j)
      require(all[j] == 0xa5 && all[16 + bytes + j] == 0xa5, "packing buffer guard overwritten");
    return {all.begin() + 16, all.begin() + 16 + bytes};
  }
};
void roundtrip(int h, int i) {
  std::array<std::unique_ptr<Buffer>, 9> source, restored;
  std::array<std::vector<uint8_t>, 9> original;
  const float globals[3] = {4.f, .25f, 8.f};
  for (int m = 0; m < 3; ++m) {
    const int n = m == 2 ? h : i, k = m == 2 ? i : h;
    original[m * 3].resize(static_cast<size_t>(n) * k / 2);
    original[m * 3 + 1].resize(static_cast<size_t>(n) * k / 16);
    for (int row = 0; row < n; ++row) {
      for (int col = 0; col < k / 2; ++col)
        original[m * 3][static_cast<size_t>(row) * (k / 2) + col] =
            static_cast<uint8_t>(row * 131 + col * 17 + 17 + m * 29);
      for (int col = 0; col < k / 16; ++col)
        original[m * 3 + 1][static_cast<size_t>(row) * (k / 16) + col] =
            static_cast<uint8_t>(row * 37 + col * 13 + m * 17);
    }
    original[m * 3 + 2].resize(4);
    std::memcpy(original[m * 3 + 2].data(), &globals[m], 4);
    for (int part = 0; part < 3; ++part) {
      const auto idx = m * 3 + part;
      source[idx] = std::make_unique<Buffer>(original[idx].size());
      restored[idx] = std::make_unique<Buffer>(original[idx].size());
      DGPP_CUDA_OK(cudaMemcpy(source[idx]->data(), original[idx].data(), original[idx].size(), cudaMemcpyHostToDevice));
    }
  }
  std::array<std::unique_ptr<Buffer>, 6> packed;
  const size_t sizes[] = {static_cast<size_t>(i) * h, static_cast<size_t>(i) * h / 8, 8,
                           static_cast<size_t>(h) * i / 2, static_cast<size_t>(h) * i / 16, 4};
  for (int j = 0; j < 6; ++j) packed[j] = std::make_unique<Buffer>(sizes[j]);
  dgpp::Fp4ExpertSource src;
  dgpp::Fp4ExpertDestination dst;
  for (int m = 0; m < 3; ++m) {
    src.payload[m] = source[m * 3]->data(); src.scales[m] = source[m * 3 + 1]->data();
    src.globals[m] = reinterpret_cast<float*>(source[m * 3 + 2]->data());
    dst.payload[m] = restored[m * 3]->data(); dst.scales[m] = restored[m * 3 + 1]->data();
    dst.globals[m] = reinterpret_cast<float*>(restored[m * 3 + 2]->data());
  }
  dgpp::Fp4PackedExpert view{packed[0]->data(), packed[1]->data(), reinterpret_cast<float*>(packed[2]->data()),
                             packed[3]->data(), packed[4]->data(), reinterpret_cast<float*>(packed[5]->data())};
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto launch = [&] {
    dgpp::launch_moe_fp4_pack(view, src, h, i, stream);
    dgpp::launch_moe_fp4_unpack(dst, view, h, i, stream);
  };
  launch();
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  for (int j = 0; j < 9; ++j) {
    require(restored[j]->read() == original[j], "FP4 payload/scales/globals failed byte-exact roundtrip");
    require(source[j]->read() == original[j], "FP4 packing changed its source");
  }
  for (const auto& p : packed) (void)p->read();
  // Frozen K16/N64 tile from FlashInfer 0.7.4's independent b12x repacker,
  // using the gate input formula above. Pins lane order beyond reversibility.
  constexpr uint32_t golden[] = {0x6251d951, 0x9584d951, 0xc8b7d951, 0xfbead951,
      0x7362ea62, 0xa695ea62, 0xd9c8ea62, 0x0cfbea62, 0x8473fb73, 0xb7a6fb73,
      0xead9fb73, 0x1d0cfb73, 0xa5840c84, 0xd8b70c84, 0x0bea0c84, 0x3e1d0c84};
  const auto words = packed[0]->read();
  require(std::memcmp(words.data(), golden, sizeof(golden)) == 0, "packed FP4 MMA lane order changed");
  DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch();
  cudaGraph_t graph;
  DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t exec;
  DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  for (auto& b : original[0]) b ^= 0xff;
  for (auto& b : original[1]) b ^= 0xff;
  DGPP_CUDA_OK(cudaMemcpyAsync(source[0]->data(), original[0].data(), original[0].size(), cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(source[1]->data(), original[1].data(), original[1].size(), cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  for (int j = 0; j < 9; ++j)
    require(restored[j]->read() == original[j], "FP4 graph replay ignored changed source bytes");
  DGPP_CUDA_OK(cudaGraphExecDestroy(exec));
  DGPP_CUDA_OK(cudaGraphDestroy(graph));
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
}
}  // namespace

DGPP_TEST(moe_fp4_pack_preserves_scales_globals_and_graph_replay) {
  roundtrip(128, 64);
  roundtrip(5120, 1152);
}
DGPP_TEST(expert_chunk_scatter_preserves_strides_padding_and_mapped_host_bytes) {
  constexpr size_t bytes = 8192;
  const std::array<dgpp::expert::ScatterCopy, 3> copies{{
      {0, 512, 64, 24, 80, 96},
      {2048, 4096, 12, 31, 20, 24},
      {4099, 6147, 3, 19, 7, 9}}};
  std::vector<uint8_t> input(bytes), expected(bytes, 0xa5);
  for (size_t j = 0; j < bytes; ++j) input[j] = static_cast<uint8_t>(j * 37 + j / 17);
  for (const auto& c : copies)
    for (size_t row = 0; row < c.rows; ++row)
      std::memcpy(expected.data() + c.destination_offset + row * c.destination_pitch,
                  input.data() + c.source_offset + row * c.source_pitch, c.row_bytes);
  const auto plan = dgpp::expert_chunk_copy_plan(copies);
  Buffer source(bytes), destination(bytes);
  DGPP_CUDA_OK(cudaMemcpy(source.data(), input.data(), bytes, cudaMemcpyHostToDevice));
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  dgpp::launch_expert_chunk_scatter(destination.data(), source.data(), plan, stream);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  require(destination.read() == expected, "expert scatter changed data or padding");
  uint8_t* host = nullptr;
  DGPP_CUDA_OK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
  void* mapped = nullptr;
  DGPP_CUDA_OK(cudaHostGetDevicePointer(&mapped, host, 0));
  std::memcpy(host, input.data(), bytes);
  DGPP_CUDA_OK(cudaMemsetAsync(destination.data(), 0xa5, bytes, stream));
  DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  dgpp::launch_expert_chunk_scatter(destination.data(), static_cast<uint8_t*>(mapped), plan, stream);
  cudaGraph_t graph;
  DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t exec;
  DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  require(destination.read() == expected, "mapped expert scatter or graph changed bytes");
  DGPP_CUDA_OK(cudaGraphExecDestroy(exec));
  DGPP_CUDA_OK(cudaGraphDestroy(graph));
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
  DGPP_CUDA_OK(cudaFreeHost(host));
}
int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 2;
  return dgpp::test::run_all();
}
