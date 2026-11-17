#include <cstdlib>

#include "kernels/moe_mxfp4_native.hpp"
#include "kernels/moe_mxfp4_native_impl.cuh"
namespace dgpp {
namespace {
__global__ void quantize_tiles(const uint16_t* x, size_t stride, int k, uint8_t* codes,
                               uint8_t* scales, const MoeSegment* segs, const MoeTile* tiles,
                               const int32_t* count) {
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
    float maximum =
        fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])), fmaxf(fabsf(values[2]), fabsf(values[3])));
    const unsigned active = __activemask();
#pragma unroll
    for (int offset = 4; offset; offset >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, offset, 8));

    // Exactly ceil(log2(amax / 448)) from IEEE bits, without log2 rounding
    // at the exponent boundary. Zero groups use scale byte zero and zero codes.
    const int biased = static_cast<int>((__float_as_uint(maximum) >> 23) & 255);
    const float threshold = __uint_as_float((static_cast<unsigned>(biased) << 23) | 0x600000u);
    const int exponent = max(-127, biased - 135 + (maximum > threshold));
    const float inverse = __uint_as_float(static_cast<unsigned>(127 - exponent) << 23);
    const uint16_t lo = __nv_cvt_float2_to_fp8x2(
        make_float2(values[0] * inverse, values[1] * inverse), __NV_SATFINITE, __NV_E4M3);
    const uint16_t hi = __nv_cvt_float2_to_fp8x2(
        make_float2(values[2] * inverse, values[3] * inverse), __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint32_t*>(codes + static_cast<size_t>(row) * k + column) =
        static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    if ((threadIdx.x & 7) == 0)
      scales[static_cast<size_t>(row) * (k / 32) + column / 32] =
          static_cast<uint8_t>(exponent + 127);
  }
}
template <class Out>
void launch(const uint8_t* a, const uint8_t* as, const MoeSegment* segs, int count, int rows,
            const MoeExpertView* views, int which, Out* out, size_t stride, int n, int k,
            cudaStream_t stream, const int32_t* map) {
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
      mx4native::run<Out, M, 128, true, true, true, S, true, false, 27, true>(
          a, as, out, rows, count, n, k, stream, segs, views, which, map, nullptr, nullptr, 0,
          stride);
    else
      mx4native::run<Out, M, 128, true, true, true, S, true, false, 24, true>(
          a, as, out, rows, count, n, k, stream, segs, views, which, map, nullptr, nullptr, 0,
          stride);
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
  quantize_tiles<<<96, 128, 0, stream>>>(x, stride, k, codes, scales, segs, tiles, count);
  DGPP_CUDA_OK(cudaGetLastError());
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
