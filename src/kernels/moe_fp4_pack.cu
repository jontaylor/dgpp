#include "kernels/moe_fp4_pack.hpp"

#include <climits>
#include <stdexcept>

#include "common/cuda_check.hpp"

namespace dgpp {
namespace {
constexpr int kThreads = 256;

__global__ void scatter_expert_chunk(uint8_t* destination, const uint8_t* source,
                                     ExpertChunkCopyPlan plan) {
  for (uint32_t m = 0; m < plan.count; ++m) {
    const auto& c = plan.copies[m];
    if (blockIdx.x < c.first_block || blockIdx.x - c.first_block >= c.blocks) continue;
    const uint64_t idx = uint64_t(blockIdx.x - c.first_block) * kThreads + threadIdx.x;
    if (idx >= c.units) return;
    const uint64_t row = idx / c.row_units;
    const uint64_t col = (idx % c.row_units) * c.unit_bytes;
    const auto* in = source + c.source_offset + row * c.source_pitch + col;
    auto* out = destination + c.destination_offset + row * c.destination_pitch + col;
    if (c.unit_bytes == 16)
      *reinterpret_cast<uint4*>(out) = *reinterpret_cast<const uint4*>(in);
    else if (c.unit_bytes == 4)
      *reinterpret_cast<uint32_t*>(out) = *reinterpret_cast<const uint32_t*>(in);
    else
      *out = *in;
    return;
  }
}

// The packed word containing source (n,k). Its eight nibbles follow the MMA
// lane ordering, rather than source K order. K is a code index, not a byte.
__device__ __forceinline__ int packed_word(int n, int k, int rows) {
  const int tc_col = n % 8;
  const int tc_row = (k % 8) / 2;
  const int lane_word = (tc_col * 4 + tc_row) * 4 + (n % 64) / 16;
  return ((k / 16) * (rows / 64) + n / 64) * 128 + lane_word;
}
__device__ __forceinline__ int packed_slot(int n, int k) {
  const int offset = (k % 16 >= 8 ? 2 : 0) + k % 2;
  const int concat = (n % 16 >= 8 ? 4 : 0) + offset;
  return (concat % 2) * 4 + concat / 2;
}
__device__ __forceinline__ int swap_scale_pair(int n) {
  const int r = n % 4;
  return n - r + (r == 1 ? 2 : r == 2 ? 1 : r);
}
__device__ __forceinline__ int transpose_scale_block(int n) {
  return n / 64 * 64 + (n % 8) * 8 + (n % 64) / 8;
}

template <bool Merge>
__global__ void pack_payload(uint32_t* out, const uint8_t* first,
                              const uint8_t* second, int rows, int k) {
  const int idx = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  if (idx >= rows * k / 8) return;
  const int kt = idx / ((rows / 64) * 128);
  const int nt = (idx / 128) % (rows / 64);
  const int pos = idx % 128;
  const int th = pos / 4;
  const int warp = pos % 4;
  const int col = th / 4;
  const int kr = (th % 4) * 2;
  uint32_t result = 0;
#pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    const int concat = slot < 4 ? slot * 2 : (slot - 4) * 2 + 1;
    const int offset = (concat % 4) / 2 * 8 + concat % 2;
    const int n = nt * 64 + warp * 16 + col + (concat >= 4 ? 8 : 0);
    const int code = kt * 16 + kr + offset;
    const bool up = Merge && n >= rows / 2;
    const int source_n = up ? n - rows / 2 : n;
    const uint8_t q = (up ? second : first)[source_n * (k / 2) + code / 2];
    result |= static_cast<uint32_t>((q >> ((code % 2) * 4)) & 15) << (slot * 4);
  }
  out[idx] = result;
}

template <bool Merge>
__global__ void unpack_payload(uint8_t* first, uint8_t* second,
                                const uint32_t* in, int rows, int k) {
  const int idx = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  if (idx >= rows * k / 2) return;
  const int n = idx / (k / 2), code = idx % (k / 2) * 2;
  const uint32_t q = in[packed_word(n, code, rows)];
  const uint8_t lo = (q >> (packed_slot(n, code) * 4)) & 15;
  const uint8_t hi = (q >> (packed_slot(n, code + 1) * 4)) & 15;
  const bool up = Merge && n >= rows / 2;
  const int source_n = up ? n - rows / 2 : n;
  (up ? second : first)[source_n * (k / 2) + code / 2] = lo | (hi << 4);
}

template <bool Merge, bool Unpack>
__global__ void move_scales(uint8_t* first, uint8_t* second, uint8_t* packed,
                             const uint8_t* source_first, const uint8_t* source_second,
                             const uint8_t* source_packed, int rows, int k) {
  const int idx = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  if (idx >= rows * (k / 16)) return;
  if constexpr (Unpack) {
    const int n = idx / (k / 16), kg = idx % (k / 16);
    const int packed_n = swap_scale_pair(transpose_scale_block(n));
    const bool up = Merge && n >= rows / 2;
    const int source_n = up ? n - rows / 2 : n;
    (up ? second : first)[source_n * (k / 16) + kg] = source_packed[kg * rows + packed_n];
  } else {
    const int kg = idx / rows;
    const int n = transpose_scale_block(swap_scale_pair(idx % rows));
    const bool up = Merge && n >= rows / 2;
    const int source_n = up ? n - rows / 2 : n;
    packed[idx] = (up ? source_second : source_first)[source_n * (k / 16) + kg];
  }
}

int blocks(int count) { return (count + kThreads - 1) / kThreads; }
void check_geometry(int h, int i) {
  // Keep all device indexing and grid arithmetic representable.
  if (h <= 0 || i <= 0 || h % 64 || i % 64 ||
      static_cast<int64_t>(h) * i > INT_MAX / 2)
    throw std::invalid_argument("MoE FP4 packing: widths must be positive multiples of 64 within index range");
}
void check_packed(Fp4PackedExpert p) {
  if (!p.w13 || !p.s13 || !p.g13 || !p.w2 || !p.s2 || !p.g2 ||
      (reinterpret_cast<uintptr_t>(p.w13) & 15) || (reinterpret_cast<uintptr_t>(p.w2) & 15))
    throw std::invalid_argument("MoE FP4 packing: null or unaligned packed pointers");
}
}  // namespace

ExpertChunkCopyPlan expert_chunk_copy_plan(std::span<const expert::ScatterCopy> copies) {
  ExpertChunkCopyPlan plan;
  if (copies.empty() || copies.size() > 9)
    throw std::invalid_argument("expert scatter: invalid copy count");
  for (const auto& c : copies) {
    if (!c.row_bytes || !c.rows || c.source_pitch < c.row_bytes || c.destination_pitch < c.row_bytes)
      throw std::invalid_argument("expert scatter: invalid pitched copy");
    const uint64_t alignment = c.source_offset | c.destination_offset | c.source_pitch |
        c.destination_pitch | c.row_bytes;
    const uint32_t unit = !(alignment & 15) ? 16 : !(alignment & 3) ? 4 : 1;
    const uint64_t row_units = c.row_bytes / unit;
    if (c.rows > UINT64_MAX / row_units)
      throw std::overflow_error("expert scatter: copy size overflow");
    const uint64_t units = row_units * c.rows;
    const uint64_t blocks = (units - 1) / kThreads + 1;
    if (blocks > INT_MAX - plan.blocks)
      throw std::overflow_error("expert scatter: grid size overflow");
    plan.copies[plan.count++] = {c.source_offset, c.destination_offset, c.source_pitch,
        c.destination_pitch, row_units, units, plan.blocks, static_cast<uint32_t>(blocks), unit};
    plan.blocks += static_cast<uint32_t>(blocks);
  }
  return plan;
}

void launch_expert_chunk_scatter(uint8_t* destination, const uint8_t* source,
    const ExpertChunkCopyPlan& plan, cudaStream_t stream) {
  if (!destination || !source || !plan.count || !plan.blocks || plan.count > 9 ||
      (reinterpret_cast<uintptr_t>(destination) & 15) || (reinterpret_cast<uintptr_t>(source) & 15))
    throw std::invalid_argument("expert scatter: null, unaligned or empty copy");
  scatter_expert_chunk<<<plan.blocks, kThreads, 0, stream>>>(destination, source, plan);
  DGPP_CUDA_OK(cudaGetLastError());
}

size_t moe_fp4_packed_bytes(int hidden, int intermediate) {
  check_geometry(hidden, intermediate);
  const size_t matrix = static_cast<size_t>(hidden) * intermediate;
  // W13 codes/scales/globals, alignment, then W2 codes/scales/globals.
  return matrix + matrix / 8 + 16 + matrix / 2 + matrix / 16 + 16;
}
Fp4PackedExpert moe_fp4_packed_view(uint8_t* data, int hidden, int intermediate) {
  (void)moe_fp4_packed_bytes(hidden, intermediate);
  const size_t matrix = static_cast<size_t>(hidden) * intermediate;
  auto* s13 = data + matrix;
  auto* g13 = s13 + matrix / 8;
  auto* w2 = g13 + 16;
  auto* s2 = w2 + matrix / 2;
  auto* g2 = s2 + matrix / 16;
  return {data, s13, reinterpret_cast<float*>(g13), w2, s2, reinterpret_cast<float*>(g2)};
}

void launch_moe_fp4_pack(Fp4PackedExpert dst, Fp4ExpertSource src,
                         int hidden, int intermediate, cudaStream_t stream) {
  check_geometry(hidden, intermediate);
  check_packed(dst);
  for (int m = 0; m < 3; ++m)
    if (!src.payload[m] || !src.scales[m] || !src.globals[m])
      throw std::invalid_argument("MoE FP4 packing: null source matrix");
  pack_payload<true><<<blocks(2 * intermediate * hidden / 8), kThreads, 0, stream>>>(
      reinterpret_cast<uint32_t*>(dst.w13), src.payload[0], src.payload[1], 2 * intermediate, hidden);
  pack_payload<false><<<blocks(hidden * intermediate / 8), kThreads, 0, stream>>>(
      reinterpret_cast<uint32_t*>(dst.w2), src.payload[2], nullptr, hidden, intermediate);
  move_scales<true, false><<<blocks(2 * intermediate * (hidden / 16)), kThreads, 0, stream>>>(
      nullptr, nullptr, dst.s13, src.scales[0], src.scales[1], nullptr, 2 * intermediate, hidden);
  move_scales<false, false><<<blocks(hidden * (intermediate / 16)), kThreads, 0, stream>>>(
      nullptr, nullptr, dst.s2, src.scales[2], nullptr, nullptr, hidden, intermediate);
  DGPP_CUDA_OK(cudaGetLastError());
  DGPP_CUDA_OK(cudaMemcpyAsync(dst.g13, src.globals[0], sizeof(float), cudaMemcpyDeviceToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(dst.g13 + 1, src.globals[1], sizeof(float), cudaMemcpyDeviceToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(dst.g2, src.globals[2], sizeof(float), cudaMemcpyDeviceToDevice, stream));
}

void launch_moe_fp4_unpack(Fp4ExpertDestination dst, Fp4PackedExpert src,
                           int hidden, int intermediate, cudaStream_t stream) {
  check_geometry(hidden, intermediate);
  check_packed(src);
  for (int m = 0; m < 3; ++m)
    if (!dst.payload[m] || !dst.scales[m] || !dst.globals[m])
      throw std::invalid_argument("MoE FP4 packing: null destination matrix");
  unpack_payload<true><<<blocks(2 * intermediate * hidden / 2), kThreads, 0, stream>>>(
      dst.payload[0], dst.payload[1], reinterpret_cast<const uint32_t*>(src.w13), 2 * intermediate, hidden);
  unpack_payload<false><<<blocks(hidden * intermediate / 2), kThreads, 0, stream>>>(
      dst.payload[2], nullptr, reinterpret_cast<const uint32_t*>(src.w2), hidden, intermediate);
  move_scales<true, true><<<blocks(2 * intermediate * (hidden / 16)), kThreads, 0, stream>>>(
      dst.scales[0], dst.scales[1], nullptr, nullptr, nullptr, src.s13, 2 * intermediate, hidden);
  move_scales<false, true><<<blocks(hidden * (intermediate / 16)), kThreads, 0, stream>>>(
      dst.scales[2], nullptr, nullptr, nullptr, nullptr, src.s2, hidden, intermediate);
  DGPP_CUDA_OK(cudaGetLastError());
  DGPP_CUDA_OK(cudaMemcpyAsync(dst.globals[0], src.g13, sizeof(float), cudaMemcpyDeviceToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(dst.globals[1], src.g13 + 1, sizeof(float), cudaMemcpyDeviceToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(dst.globals[2], src.g2, sizeof(float), cudaMemcpyDeviceToDevice, stream));
}
}  // namespace dgpp
