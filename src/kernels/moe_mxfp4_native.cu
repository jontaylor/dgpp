#include <cstdlib>
#include "common/dtypes.hpp"

#include "kernels/moe_mxfp4_native.hpp"
#include "kernels/moe_mxfp4_native_impl.cuh"
namespace dgpp {
namespace {
template <bool Activate>
__global__ void quantize_tiles(const uint16_t* x, size_t stride, int k, uint8_t* codes,
                               uint8_t* scales, const MoeSegment* segs, const MoeTile* tiles,
                               const int32_t* count, const uint16_t* up, float limit) {
  dgpp_pdl_wait();
  const int columns = (k + 511) / 512;
  const int work_items = *count * 64 * columns;
  for (int work = blockIdx.x; work < work_items; work += gridDim.x) {
    const auto tile = tiles[work / (64 * columns)];
    const auto seg = segs[tile.seg];
    const int local = tile.m0 + (work / columns) % 64;
    if (local >= seg.rows) continue;
    const int row = seg.row0 + local;
    const int column = ((work % columns) * blockDim.x + threadIdx.x) * 4;
    if (column >= k) continue;
    const uint2 bits =
        *reinterpret_cast<const uint2*>(x + static_cast<size_t>(row) * stride + column);
    float values[4] = {__uint_as_float(bits.x << 16), __uint_as_float(bits.x & 0xffff0000u),
                       __uint_as_float(bits.y << 16), __uint_as_float(bits.y & 0xffff0000u)};
    if constexpr (Activate) {
      const uint2 u = *reinterpret_cast<const uint2*>(up + static_cast<size_t>(row) * stride + column);
      const uint16_t words[4] = {uint16_t(u.x), uint16_t(u.x >> 16), uint16_t(u.y), uint16_t(u.y >> 16)};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        values[j] = mx4native::swiglu_bf16(values[j], bf16_bits_to_float(words[j]), limit);
      }
    }
    mx4native::quantize_four(values, codes + static_cast<size_t>(row) * k + column,
                             scales + static_cast<size_t>(row) * (k / 32) + column / 32);
  }
}
template <bool Paired = false, class Out>
void launch(const uint8_t* a, const uint8_t* as, const MoeSegment* segs, int count, int rows,
            const MoeExpertView* views, int which, Out* out, size_t stride, int n, int k,
            cudaStream_t stream, const int32_t* map, uint8_t* scales = nullptr, float limit = 0.f) {
  if (!a || !as || !segs || !views || !out || count < 1 || rows < 1 || n < 64 || n % 64 || k < 64 ||
      k % 64 || stride < size_t(n) || which < 0 || which > 2)
    throw std::invalid_argument("native MXFP4 expert geometry");
  static const int tile = [] {
    const char* v = std::getenv("DGPP_MOE_MX_NATIVE_TILE_ROWS");
    return v ? std::atoi(v) : 64;
  }();
  static const int stages = [] {
    const char* v = std::getenv("DGPP_MOE_MX_NATIVE_STAGES");
    return v ? std::atoi(v) : 2;
  }();
  static const bool skip = [] {
    const char* v = std::getenv("DGPP_MOE_MX_NATIVE_TAIL_SKIP");
    return !v || v[0] != '0';
  }();
  auto run = [&]<int M, int S>() {
    if (skip)
      mx4native::run<Out, M, 128, true, true, true, S, true, false, 27, true, Paired>(
          a, as, out, rows, count, n, k, stream, segs, views, which, map, nullptr, nullptr, 0,
          stride, scales, limit);
    else
      mx4native::run<Out, M, 128, true, true, true, S, true, false, 24, true, Paired>(
          a, as, out, rows, count, n, k, stream, segs, views, which, map, nullptr, nullptr, 0,
          stride, scales, limit);
  };
  if (rows <= 16) {
    run.template operator()<16, 2>();
  } else if (tile == 128) {
    if (stages == 3)
      run.template operator()<128, 3>();
    else
      run.template operator()<128, 2>();
  } else {
    if (stages == 3)
      run.template operator()<64, 3>();
    else
      run.template operator()<64, 2>();
  }
}
}  // namespace
void launch_moe_mxfp8_quantize_tiles(const uint16_t* x, size_t stride, int k, uint8_t* codes,
                                     uint8_t* scales, const MoeSegment* segs, const MoeTile* tiles,
                                     const int32_t* count, cudaStream_t stream) {
  quantize_tiles<false><<<96, 128, 0, stream>>>(x, stride, k, codes, scales, segs, tiles, count, nullptr, 0.f);
  DGPP_CUDA_OK(cudaGetLastError());
}
void launch_moe_mxfp8_swiglu_tiles(const uint16_t* gate, const uint16_t* up, float limit, size_t stride, int k,
                                   uint8_t* codes, uint8_t* scales, const MoeSegment* segs,
                                   const MoeTile* tiles, const int32_t* count, cudaStream_t stream, int blocks) {
  quantize_tiles<true><<<blocks, 128, 0, stream>>>(gate, stride, k, codes, scales, segs, tiles, count, up, limit);
  DGPP_CUDA_OK(cudaGetLastError());
}
void launch_moe_mxfp4_packed_gate_up(const uint8_t* a, const uint8_t* as, const MoeSegment* segs,
                                     int count, int rows, const MoeExpertView* views, int n, int k,
                                     const int32_t* map, float limit, uint8_t* codes,
                                     uint8_t* scales, cudaStream_t stream) {
  if (!scales) throw std::invalid_argument("missing paired MX activation scales");
  launch<true>(a, as, segs, count, rows, views, 0, codes, n, n, k, stream, map, scales, limit);
}
void launch_moe_mxfp4_native_bf16(const uint8_t* a, const uint8_t* as, const MoeSegment* s,
                                  int count, int rows, const MoeExpertView* v, int which,
                                  uint16_t* out, size_t stride, int n, int k, cudaStream_t stream,
                                  const int32_t* map) {
  launch(a, as, s, count, rows, v, which, out, stride, n, k, stream, map);
}
void launch_moe_mxfp4_native_f32(const uint8_t* a, const uint8_t* as, const MoeSegment* s,
                                 int count, int rows, const MoeExpertView* v, int which, float* out,
                                 size_t stride, int n, int k, cudaStream_t stream,
                                 const int32_t* map) {
  launch(a, as, s, count, rows, v, which, out, stride, n, k, stream, map);
}
}  // namespace dgpp
