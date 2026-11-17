#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/moe_fp4_pack.hpp"
#include "kernels/moe_mxfp4_native.hpp"
#include "kernels/moe_mxfp4_native_impl.cuh"
#include "kernels/mxfp8_gemm.hpp"
using namespace dgpp;
template <class T>
struct Buffer {
  T* p;
  explicit Buffer(size_t n) { DGPP_CUDA_OK(cudaMalloc(&p, n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
  void write(const std::vector<T>& v) {
    DGPP_CUDA_OK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> read(size_t n) {
    std::vector<T> v(n);
    DGPP_CUDA_OK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
int main() {
  try {
    constexpr int H = 128, I = 128, C = 64, E = 2, R = 169, K = 128, S = 144;
    constexpr size_t payload = H * I / 2, scales = H * I / 32;
    std::mt19937 rng(7162);
    std::vector<uint8_t> hw(E * 3 * payload), hs(E * 3 * scales);
    for (auto& v : hw) v = rng();
    for (auto& v : hs) v = 120 + rng() % 5;
    Buffer<uint8_t> w(hw.size()), ws(hs.size());
    w.write(hw);
    ws.write(hs);
    std::vector<uint16_t> hx(R * K);
    for (auto& v : hx) v = float_to_bf16_bits((int(rng() % 2001) - 1000) / 1024.f);
    Buffer<uint16_t> x(hx.size());
    x.write(hx);
    Buffer<uint8_t> a(R * K), as(R * K / 32), qa(R * K), qas(R * K / 32);
    Buffer<float> out(R * S), ref(R * S);
    Buffer<uint16_t> bfout(R * S);
    std::vector<MoeSegment> hseg{{3, 7, 0}, {17, 19, 1}};
    Buffer<MoeSegment> seg(2);
    seg.write(hseg);
    std::vector<int32_t> hm(R);
    for (int r = 0; r < R; ++r) hm[r] = (r * 7) % R;
    Buffer<int32_t> map(R);
    map.write(hm);
    std::vector<MoeExpertView> cv(E * 3), pv(E * 3), dv(E * 3);
    const size_t pb = moe_fp4_packed_bytes(H, I, 32);
    Buffer<uint8_t> packed(E * pb);
    constexpr size_t cp = C * H / 2, cs = C * H / 32, record = 3 * (cp + cs);
    std::vector<uint8_t> hc(E * 2 * record);
    Buffer<uint8_t> chunks(hc.size());
    std::vector<const uint8_t*> pointers(E * 2);
    Buffer<const uint8_t*> ptr(E * 2);
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreate(&stream));
    for (int e = 0; e < E; ++e) {
      Fp4ExpertSource src;
      auto packed_view = moe_fp4_packed_view(packed.p + e * pb, H, I, 32);
      for (int m = 0; m < 3; ++m) {
        auto& v = cv[e * 3 + m];
        v.payload = w.p + (e * 3 + m) * payload;
        v.fp4_scales = ws.p + (e * 3 + m) * scales;
        src.payload[m] = v.payload;
        src.scales[m] = v.fp4_scales;
        auto& p = pv[e * 3 + m];
        p.payload = m == 2 ? packed_view.w2 : packed_view.w13;
        p.fp4_scales = m == 2 ? packed_view.s2 : packed_view.s13;
        p.fp4_packed_rows = m == 2 ? H : 2 * I;
        p.fp4_packed_row0 = m == 1 ? I : 0;
        auto& d = dv[e * 3 + m];
        d.fp4_chunks = ptr.p + e * 2;
        d.fp4_chunk_width = C;
        d.fp4_chunk_columns = m == 2;
        d.fp4_chunk_payload = m * (cp + cs);
        d.fp4_chunk_scales = m * (cp + cs) + cp;
        for (int c = 0; c < 2; ++c) {
          auto* dest = hc.data() + (e * 2 + c) * record + m * (cp + cs);
          if (m < 2) {
            std::copy_n(hw.data() + (e * 3 + m) * payload + c * cp, cp, dest);
            std::copy_n(hs.data() + (e * 3 + m) * scales + c * cs, cs, dest + cp);
          } else
            for (int r = 0; r < H; ++r) {
              std::copy_n(hw.data() + (e * 3 + m) * payload + r * (I / 2) + c * (C / 2), C / 2,
                          dest + r * (C / 2));
              std::copy_n(hs.data() + (e * 3 + m) * scales + r * (I / 32) + c * (C / 32), C / 32,
                          dest + cp + r * (C / 32));
            }
        }
      }
      launch_moe_fp4_pack(packed_view, src, H, I, stream, 32);
      for (int c = 0; c < 2; ++c) pointers[e * 2 + c] = chunks.p + (e * 2 + c) * record;
    }
    chunks.write(hc);
    ptr.write(pointers);
    Buffer<MoeExpertView> canon(E * 3), prepared(E * 3), direct(E * 3);
    canon.write(cv);
    prepared.write(pv);
    direct.write(dv);
    launch_mxfp8_quantize_rows(x.p, K, R, K, a.p, as.p, stream);
    Buffer<MoeTile> tiles(4);
    Buffer<int32_t> count(1);
    launch_moe_tile_list(seg.p, 2, 64, tiles.p, count.p, stream);
    launch_moe_mxfp8_quantize_tiles(x.p, K, K, qa.p, qas.p, seg.p, tiles.p, count.p, stream);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    auto ar = a.read(R * K), qr = qa.read(R * K), sr = as.read(R * K / 32),
         qsr = qas.read(R * K / 32);
    for (auto sg : hseg)
      for (int r = sg.row0; r < sg.row0 + sg.rows; ++r) {
        if (!std::equal(ar.begin() + r * K, ar.begin() + (r + 1) * K, qr.begin() + r * K) ||
            !std::equal(sr.begin() + r * (K / 32), sr.begin() + (r + 1) * (K / 32),
                        qsr.begin() + r * (K / 32)))
          throw std::runtime_error("segmented quantization differs from row quantization");
      }
    // Compare all bytes, including untouched gaps between routed segments.
    Buffer<uint16_t> activation(R*K), up(R*K);
    std::reverse(hx.begin(), hx.end());
    up.write(hx);
    for (float limit : {0.5f, 8.f}) {
      for (auto* buf : {a.p, qa.p}) DGPP_CUDA_OK(cudaMemsetAsync(buf, 0xcc, R*K, stream));
      for (auto* buf : {as.p, qas.p}) DGPP_CUDA_OK(cudaMemsetAsync(buf, 0xcc, R*K/32, stream));
      launch_moe_swiglu_clamp(x.p, up.p, activation.p, R*K, limit, stream);
      launch_moe_mxfp8_quantize_tiles(activation.p, K, K, a.p, as.p, seg.p, tiles.p, count.p, stream);
      launch_moe_mxfp8_swiglu_tiles(x.p, up.p, limit, K, K, qa.p, qas.p, seg.p, tiles.p, count.p, stream);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      if (a.read(R*K) != qa.read(R*K) || as.read(R*K/32) != qas.read(R*K/32))
        throw std::runtime_error("fused SwiGLU quantization differs from the two-stage chain");
    }
    launch_mxfp8_quantize_rows(x.p, K, R, K, a.p, as.p, stream);
    // Compare every output byte, including row gaps, against separate gate,
    // up and activation kernels. Include a second M tile and the small-wave path.
    Buffer<uint8_t> expected_codes(R * I), expected_scales(R * I / 32);
    for (int rows : {137, 7}) {
      hseg[1].rows = rows;
      seg.write(hseg);
      launch_moe_tile_list(seg.p, E, 64, tiles.p, count.p, stream);
      for (auto* buf : {expected_codes.p, qa.p})
        DGPP_CUDA_OK(cudaMemsetAsync(buf, 0xcc, R * I, stream));
      for (auto* buf : {expected_scales.p, qas.p})
        DGPP_CUDA_OK(cudaMemsetAsync(buf, 0xcc, R * I / 32, stream));
      launch_moe_mxfp4_native_bf16(a.p, as.p, seg.p, E, rows, prepared.p, 0, activation.p, I, I, K,
                                   stream, map.p);
      launch_moe_mxfp4_native_bf16(a.p, as.p, seg.p, E, rows, prepared.p, 1, up.p, I, I, K, stream,
                                   map.p);
      launch_moe_mxfp8_swiglu_tiles(activation.p, up.p, 0.5f, I, I, expected_codes.p,
                                    expected_scales.p, seg.p, tiles.p, count.p, stream);
      launch_moe_mxfp4_packed_gate_up(a.p, as.p, seg.p, E, rows, prepared.p, I, K, map.p, 0.5f,
                                      qa.p, qas.p, stream);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      if (qa.read(R * I) != expected_codes.read(R * I) ||
          qas.read(R * I / 32) != expected_scales.read(R * I / 32))
        throw std::runtime_error("paired gate/up differs from separate projections");
    }
    for (int active_rows : {19, 7}) {
      hseg[1].rows = active_rows;
      seg.write(hseg);
      for (int m = 0; m < 3; ++m) {
        const int32_t* rows = m == 2 ? nullptr : map.p;
        DGPP_CUDA_OK(cudaMemset(ref.p, 0, R * S * 4));
        mx4native::run<float, 64, 128, true, true, true, 2, true, false, 24>(
            a.p, as.p, ref.p, 19, E, K, K, stream, seg.p, canon.p, m, rows, nullptr, nullptr, 0, S);
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        auto expected = ref.read(R * S);
        for (auto* views : {canon.p, prepared.p, direct.p}) {
          DGPP_CUDA_OK(cudaMemset(out.p, 0, R * S * 4));
          launch_moe_mxfp4_native_f32(a.p, as.p, seg.p, E, active_rows, views, m, out.p, S, K, K,
                                      stream, rows);
          DGPP_CUDA_OK(cudaStreamSynchronize(stream));
          if (out.read(R * S) != expected) throw std::runtime_error("native view mismatch");
          launch_moe_mxfp4_native_bf16(a.p, as.p, seg.p, E, active_rows, views, m, bfout.p, S, K, K,
                                       stream, rows);
          DGPP_CUDA_OK(cudaStreamSynchronize(stream));
          auto got = bfout.read(R * S);
          for (auto sg : hseg)
            for (int r = sg.row0; r < sg.row0 + sg.rows; ++r)
              for (int c = 0; c < K; ++c)
                if (got[r * S + c] != float_to_bf16_bits(expected[r * S + c]))
                  throw std::runtime_error("BF16 native output mismatch");
        }
      }
    }
    Buffer<uint8_t> replay_codes(R * I), replay_scales(R * I / 32);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_moe_mxfp8_quantize_tiles(x.p, K, K, qa.p, qas.p, seg.p, tiles.p, count.p, stream);
    DGPP_CUDA_OK(cudaMemsetAsync(replay_codes.p, 0xcc, R * I, stream));
    DGPP_CUDA_OK(cudaMemsetAsync(replay_scales.p, 0xcc, R * I / 32, stream));
    launch_moe_mxfp4_packed_gate_up(a.p, as.p, seg.p, E, 7, prepared.p, I, K, map.p, 0.5f,
                                    replay_codes.p, replay_scales.p, stream);
    launch_moe_mxfp4_native_f32(a.p, as.p, seg.p, E, 7, direct.p, 2, out.p, S, K, K, stream);
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, 0));
    DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    if (out.read(R * S) != ref.read(R * S)) throw std::runtime_error("native graph mismatch");
    if (replay_codes.read(R * I) != expected_codes.read(R * I) ||
        replay_scales.read(R * I / 32) != expected_scales.read(R * I / 32))
      throw std::runtime_error("paired gate/up graph mismatch");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    puts(
        "PASS canonical/prepared/chunk views, gate/up/down, output strides, quantization and graph "
        "replay");
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
