#include <filesystem>
#include <fstream>
#include <iostream>

#include "kernels/glm_norm.hpp"
#include "models/mimo/dflash.hpp"
#include "models/mimo/loader.hpp"
using namespace dgpp;
template <class T>
void dump(const std::string& path, const T* p, size_t n) {
  std::vector<T> h(n);
  DGPP_CUDA_OK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(h.data()), n * sizeof(T));
}
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: mimo_dflash_check CHECKPOINT OUTPUT_DIR");
    std::string ckpt = argv[1], out = argv[2];
    std::filesystem::create_directories(out);
    MimoDeviceLoader loader(ckpt, 0, 1);
    auto globals = loader.load_globals();
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreate(&stream));
    MimoDFlash draft(ckpt + "/dflash", 16, 2, globals.vocab_count, globals.embed, globals.head,
                     stream);
    std::vector<uint16_t> feature(4 * 20480);
    for (size_t i = 0; i < feature.size(); i++)
      feature[i] = float_to_bf16_bits(float(int(i * 17 % 101) - 50) * 0.002f);
    DGPP_CUDA_OK(
        cudaMemcpy(draft.features(), feature.data(), feature.size() * 2, cudaMemcpyHostToDevice));
    LayerBump scratch;
    scratch.init((1u << 20) + MimoDFlash::state_bytes(16));
    auto* pos = static_cast<int64_t*>(scratch.alloc(4 * 8));
    auto* ids = static_cast<int32_t*>(scratch.alloc(4 * 4));
    auto* tok = static_cast<int64_t*>(scratch.alloc(4 * 8));
    auto* state = static_cast<uint8_t*>(scratch.alloc(MimoDFlash::state_bytes(16)));
    int64_t hp[4] = {0, 1, 0, 1}, ht[4] = {42, 43, 44, 45};
    int32_t hi[4] = {0, 0, 1, 1};
    DGPP_CUDA_OK(cudaMemcpy(pos, hp, sizeof(hp), cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(ids, hi, sizeof(hi), cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(tok, ht, sizeof(ht), cudaMemcpyHostToDevice));
    draft.context(draft.features(), pos, ids, 4, stream);
    draft.snapshot(0, state, stream);
    draft.propose(tok, pos, ids, 2, 2, stream);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    dump(out + "/features.bf16", draft.features(), 4 * 20480);
    dump(out + "/hidden.bf16", draft.output_hidden(), 16 * 4096);
    dump(out + "/embedding.bf16", globals.embed + 43 * 4096, 4096);
    dump(out + "/embedding2.bf16", globals.embed + 45 * 4096, 4096);
    dump(out + "/mask.bf16", draft.mask_embedding(), 4096);
    std::vector<uint16_t> before(16 * 4096), after(before.size());
    DGPP_CUDA_OK(cudaMemcpy(before.data(), draft.output_hidden(), before.size() * 2,
                            cudaMemcpyDeviceToHost));
    draft.prepare();
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    draft.propose(tok, pos, ids, 2, 2, stream);
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    DGPP_CUDA_OK(
        cudaMemcpy(after.data(), draft.output_hidden(), after.size() * 2, cudaMemcpyDeviceToHost));
    if (before != after) throw std::runtime_error("graph/eager hidden mismatch");
    DGPP_CUDA_OK(cudaMemsetAsync(draft.features(), 0, feature.size() * 2, stream));
    draft.context(draft.features(), pos, ids, 2, stream);
    draft.restore(0, state, stream);
    draft.propose(tok, pos, ids, 2, 2, stream);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    DGPP_CUDA_OK(
        cudaMemcpy(after.data(), draft.output_hidden(), after.size() * 2, cudaMemcpyDeviceToHost));
    if (before != after) throw std::runtime_error("snapshot restore mismatch");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    std::cout << "PASS: two-request real-weight DFlash block; eager/graph and context snapshot "
                 "restore exact. Reference files: "
              << out << "\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
