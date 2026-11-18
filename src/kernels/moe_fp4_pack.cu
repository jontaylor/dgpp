#include "kernels/moe_fp4_pack.hpp"
#include "kernels/moe_fp4_layout.cuh"

#include <algorithm>
#include <array>
#include <climits>
#include <stdexcept>

#include "common/cuda_check.hpp"

namespace dgpp {
namespace {
constexpr int kThreads = 256;

__global__ void compact_scales(const uint32_t* source, uint16_t* destination, int words) {
  const int j = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  if (j >= words) return;
  const uint32_t x = source[j];
  destination[j] = ((x >> 3) & 0xf) | ((x >> 7) & 0xf0) |
                   ((x >> 11) & 0xf00) | ((x >> 15) & 0xf000);
}
__global__ void expand_compact_scales(const uint16_t* source, uint32_t* destination, int words) {
  const int j = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  if (j >= words) return;
  const uint32_t x = source[j];
  destination[j] = ((x & 0xf) << 3) | ((x & 0xf0) << 7) |
                   ((x & 0xf00) << 11) | ((x & 0xf000) << 15);
}

// Three scalar scales belong to the compute dependency chain. Avoid a
// copy-engine operation between packing and its consumer.
__global__ void copy_globals(float* d0, float* d1, float* d2,
                             const float* s0, const float* s1, const float* s2) {
  if (threadIdx.x == 0) { *d0 = *s0; *d1 = *s1; *d2 = *s2; }
}

__device__ __forceinline__ void scatter_chunk_bytes(uint8_t* destination, const uint8_t* source,
                                                   const ExpertChunkCopyPlan& plan) {
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

__global__ void scatter_expert_chunk(uint8_t* destination, const uint8_t* source,
                                     ExpertChunkCopyPlan plan) {
  scatter_chunk_bytes(destination, source, plan);
}
struct ExpertScatterBatch { ExpertChunkScatter chunks[kExpertScatterBatch]; };
__global__ void scatter_expert_chunks(const __grid_constant__ ExpertScatterBatch batch) {
  const auto& chunk = batch.chunks[blockIdx.y];
  scatter_chunk_bytes(chunk.destination, chunk.source, chunk.plan);
}

using namespace fp4_layout;

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

template <bool Merge, bool Unpack, bool Nibbles = false>
__global__ void move_scales(uint8_t* first, uint8_t* second, uint8_t* packed,
                             const uint8_t* source_first, const uint8_t* source_second,
                             const uint8_t* source_packed, int rows, int k, int group) {
  const int idx = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  if (idx >= rows * (k / group)) return;
  if constexpr (Unpack) {
    const int n = idx / (k / group), kg = idx % (k / group);
    const int packed_n = swap_scale_pair(transpose_scale_block(n));
    const bool up = Merge && n >= rows / 2;
    const int source_n = up ? n - rows / 2 : n;
    (up ? second : first)[source_n * (k / group) + kg] = mx_scale(source_packed, kg * rows + packed_n, Nibbles);
  } else {
    const int kg = idx / rows;
    const int n = transpose_scale_block(swap_scale_pair(idx % rows));
    const bool up = Merge && n >= rows / 2;
    const int source_n = up ? n - rows / 2 : n;
    packed[idx] = (up ? source_second : source_first)[source_n * (k / group) + kg];
  }
}

template<bool Merge>
__global__ void pack_mx_scale_words(uint32_t* packed, const uint8_t* first,
                                    const uint8_t* second, int rows, int k) {
  const int word = blockIdx.x * blockDim.x + threadIdx.x;
  if (word >= rows * (k / 32) / 8) return;
  const int kg = word * 8 / rows;
  uint32_t result = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int n = transpose_scale_block(swap_scale_pair((word * 8 + j) % rows));
    const bool up = Merge && n >= rows / 2;
    if (up) n -= rows / 2;
    result |= uint32_t((up ? second : first)[n * (k / 32) + kg] & 15) << ((j % 4) * 8 + (j & 4));
  }
  packed[word] = result;
}

template<bool Expand>
__global__ void mx_chunk_codec(uint8_t* destination, const uint8_t* source, int payload) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int scales = payload / 16;
  const auto* src = source + blockIdx.y * (payload + (Expand ? scales / 2 : scales));
  auto* dst = destination + blockIdx.y * (payload + (Expand ? scales : scales / 2));
  if (j < payload / 16) {
    reinterpret_cast<uint4*>(dst)[j] = reinterpret_cast<const uint4*>(src)[j];
    return;
  }
  const int i = j - payload / 16;
  if (i >= scales / 8) return;
  auto* out = reinterpret_cast<uint32_t*>(dst + payload);
  const auto* in = reinterpret_cast<const uint32_t*>(src + payload);
  constexpr uint32_t mask = 0x0f0f0f0f, base = 0x70707070;
  if constexpr (Expand) {
    out[2*i] = (in[i] & mask) | base;
    out[2*i+1] = ((in[i] >> 4) & mask) | base;
  } else {
    out[i] = (in[2*i] & mask) | ((in[2*i+1] & mask) << 4);
  }
}

int blocks(int count) { return (count + kThreads - 1) / kThreads; }
void check_geometry(int h, int i) {
  // Keep all device indexing and grid arithmetic representable.
  if (h <= 0 || i <= 0 || h % 64 || i % 64 ||
      static_cast<int64_t>(h) * i > INT_MAX / 2)
    throw std::invalid_argument("MoE FP4 packing: widths must be positive multiples of 64 within index range");
}
void check_packed(Fp4PackedExpert p, int group = 16) {
  if (group != 16 && group != 32) throw std::invalid_argument("MoE FP4 packing: invalid scale group");
  if (p.mx_nibbles && group != 32) throw std::invalid_argument("MX nibble scales require group 32");
  if (!p.w13 || !p.s13 || !p.w2 || !p.s2 || (group == 16 && (!p.g13 || !p.g2)) ||
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

void launch_expert_chunk_scatter_batch(std::span<const ExpertChunkScatter> chunks, cudaStream_t stream) {
  if (chunks.empty()) return;
  if (chunks.size() > kExpertScatterBatch) throw std::invalid_argument("expert scatter: batch too large");
  if (chunks.size() == 1) {
    const auto& c = chunks.front();
    launch_expert_chunk_scatter(c.destination, c.source, c.plan, stream);
    return;
  }
  ExpertScatterBatch batch{};
  uint32_t blocks = 0;
  for (size_t i = 0; i < chunks.size(); ++i) {
    const auto& c = chunks[i];
    if (!c.destination || !c.source || !c.plan.count || !c.plan.blocks || c.plan.count > 9 ||
        (reinterpret_cast<uintptr_t>(c.destination) & 15) || (reinterpret_cast<uintptr_t>(c.source) & 15))
      throw std::invalid_argument("expert scatter: null, unaligned or empty batch entry");
    batch.chunks[i] = c;
    blocks = std::max(blocks, c.plan.blocks);
  }
  // NVFP4 chunks repeat each projection's scalar global at the same output
  // address. Keep the final copy, matching sequential assembly without
  // concurrent stores to the shared scalar. Payload/scales are disjoint.
  std::array<uintptr_t, kExpertScatterBatch * 9> scalars;
  size_t scalar_count = 0;
  for (size_t i = chunks.size(); i-- > 0;) {
    auto& chunk = batch.chunks[i];
    for (uint32_t j = 0; j < chunk.plan.count; ++j) {
      auto& c = chunk.plan.copies[j];
      if (c.units != 1 || c.unit_bytes != 4) continue;
      const auto address = reinterpret_cast<uintptr_t>(chunk.destination) + c.destination_offset;
      if (std::find(scalars.begin(), scalars.begin() + scalar_count, address) != scalars.begin() + scalar_count)
        c.blocks = 0;
      else scalars[scalar_count++] = address;
    }
  }
  scatter_expert_chunks<<<dim3(blocks, chunks.size()), kThreads, 0, stream>>>(batch);
  DGPP_CUDA_OK(cudaGetLastError());
}

size_t moe_fp4_packed_bytes(int hidden, int intermediate, int scale_group, bool mx_nibbles) {
  if (mx_nibbles && scale_group != 32) throw std::invalid_argument("MX nibble scales require group 32");
  if (scale_group != 16 && scale_group != 32) throw std::invalid_argument("MoE FP4 packing: invalid scale group");
  check_geometry(hidden, intermediate);
  const size_t matrix = static_cast<size_t>(hidden) * intermediate;
  // W13 codes/scales/globals, alignment, then W2 codes/scales/globals.
  const int density = scale_group * (mx_nibbles ? 2 : 1);
  return matrix + 2*matrix / density + matrix / 2 + matrix / density + (scale_group == 16 ? 32 : 0);
}
Fp4PackedExpert moe_fp4_packed_view(uint8_t* data, int hidden, int intermediate, int scale_group, bool mx_nibbles) {
  (void)moe_fp4_packed_bytes(hidden, intermediate, scale_group, mx_nibbles);
  const int density = scale_group * (mx_nibbles ? 2 : 1);
  const size_t matrix = static_cast<size_t>(hidden) * intermediate;
  auto* s13 = data + matrix;
  auto* g13 = s13 + 2*matrix / density;
  auto* w2 = g13 + (scale_group == 16 ? 16 : 0);
  auto* s2 = w2 + matrix / 2;
  auto* g2 = s2 + matrix / density;
  return {data, s13, scale_group == 16 ? reinterpret_cast<float*>(g13) : nullptr,
          w2, s2, scale_group == 16 ? reinterpret_cast<float*>(g2) : nullptr, mx_nibbles};
}

size_t moe_fp4_compact_bytes(int hidden, int intermediate) {
  check_geometry(hidden, intermediate);
  const size_t matrix = static_cast<size_t>(hidden) * intermediate;
  return matrix + matrix / 16 + 16 + matrix / 2 + matrix / 32 + 16;
}
size_t moe_fp4_expanded_scale_bytes(int hidden, int intermediate) {
  check_geometry(hidden, intermediate);
  const size_t matrix = static_cast<size_t>(hidden) * intermediate;
  return matrix / 8 + matrix / 16;
}
Fp4CompactExpert moe_fp4_compact_view(uint8_t* data, int hidden, int intermediate) {
  (void)moe_fp4_compact_bytes(hidden, intermediate);
  if (!data || (reinterpret_cast<uintptr_t>(data) & 15))
    throw std::invalid_argument("MoE FP4 compact: null or unaligned base");
  const size_t matrix = static_cast<size_t>(hidden) * intermediate;
  auto* s13 = data + matrix;
  auto* g13 = s13 + matrix / 16;
  auto* w2 = g13 + 16;
  auto* s2 = w2 + matrix / 2;
  auto* g2 = s2 + matrix / 32;
  return {data, s13, reinterpret_cast<float*>(g13), w2, s2, reinterpret_cast<float*>(g2)};
}

void launch_moe_fp4_compact(Fp4CompactExpert dst, Fp4PackedExpert src,
                            int hidden, int intermediate, bool scales_verified, cudaStream_t stream) {
  check_geometry(hidden, intermediate);
  check_packed(src);
  if (!scales_verified || !dst.w13 || !dst.s13 || !dst.g13 || !dst.w2 || !dst.s2 || !dst.g2 ||
      (reinterpret_cast<uintptr_t>(dst.w13) & 15) || (reinterpret_cast<uintptr_t>(dst.w2) & 15) ||
      (reinterpret_cast<uintptr_t>(src.s13) & 3) || (reinterpret_cast<uintptr_t>(src.s2) & 3) ||
      (reinterpret_cast<uintptr_t>(dst.s13) & 1) || (reinterpret_cast<uintptr_t>(dst.s2) & 1))
    throw std::invalid_argument("MoE FP4 compact: unchecked scales, null or unaligned pointers");
  const size_t matrix = static_cast<size_t>(hidden) * intermediate;
  DGPP_CUDA_OK(cudaMemcpyAsync(dst.w13, src.w13, matrix, cudaMemcpyDeviceToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(dst.w2, src.w2, matrix / 2, cudaMemcpyDeviceToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(dst.g13, src.g13, 2*sizeof(float), cudaMemcpyDeviceToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(dst.g2, src.g2, sizeof(float), cudaMemcpyDeviceToDevice, stream));
  const int a = static_cast<int>(matrix / 8 / 4), b = static_cast<int>(matrix / 16 / 4);
  compact_scales<<<blocks(a), kThreads, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(src.s13), reinterpret_cast<uint16_t*>(dst.s13), a);
  compact_scales<<<blocks(b), kThreads, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(src.s2), reinterpret_cast<uint16_t*>(dst.s2), b);
  DGPP_CUDA_OK(cudaGetLastError());
}
Fp4PackedExpert launch_moe_fp4_expand_scales(Fp4CompactExpert src, uint8_t* scales,
                                             int hidden, int intermediate, cudaStream_t stream) {
  check_geometry(hidden, intermediate);
  if (!src.w13 || !src.s13 || !src.g13 || !src.w2 || !src.s2 || !src.g2 || !scales ||
      (reinterpret_cast<uintptr_t>(src.w13) & 15) || (reinterpret_cast<uintptr_t>(src.w2) & 15) ||
      (reinterpret_cast<uintptr_t>(src.s13) & 1) || (reinterpret_cast<uintptr_t>(src.s2) & 1) ||
      (reinterpret_cast<uintptr_t>(scales) & 3))
    throw std::invalid_argument("MoE FP4 expand: null or unaligned pointers");
  const size_t matrix = static_cast<size_t>(hidden) * intermediate;
  const int a = static_cast<int>(matrix / 8 / 4), b = static_cast<int>(matrix / 16 / 4);
  expand_compact_scales<<<blocks(a), kThreads, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(src.s13), reinterpret_cast<uint32_t*>(scales), a);
  expand_compact_scales<<<blocks(b), kThreads, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(src.s2), reinterpret_cast<uint32_t*>(scales + matrix / 8), b);
  DGPP_CUDA_OK(cudaGetLastError());
  return {src.w13, scales, src.g13, src.w2, scales + matrix / 8, src.g2};
}

void launch_mx_chunk_codec(uint8_t* destination, const uint8_t* source,
                           int payload, bool expand, cudaStream_t stream) {
  if (payload <= 0 || payload % 256) throw std::invalid_argument("MX chunk codec alignment");
  const dim3 grid(blocks(payload / 16 + payload / 128), 3);
  if (expand) mx_chunk_codec<true><<<grid, kThreads, 0, stream>>>(destination, source, payload);
  else mx_chunk_codec<false><<<grid, kThreads, 0, stream>>>(destination, source, payload);
  DGPP_CUDA_OK(cudaGetLastError());
}

void prepare_moe_fp4_pack() {
  cudaFuncAttributes attributes{};
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, mx_chunk_codec<true>));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, mx_chunk_codec<false>));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, pack_mx_scale_words<true>));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, pack_mx_scale_words<false>));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, move_scales<true, true, true>));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, move_scales<false, true, true>));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, scatter_expert_chunk));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, scatter_expert_chunks));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, compact_scales));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, expand_compact_scales));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, copy_globals));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, pack_payload<true>));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, pack_payload<false>));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, unpack_payload<true>));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, unpack_payload<false>));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, move_scales<true, false>));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, move_scales<false, false>));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, move_scales<true, true>));
  DGPP_CUDA_OK(cudaFuncGetAttributes(&attributes, move_scales<false, true>));
}

void launch_moe_fp4_pack(Fp4PackedExpert dst, Fp4ExpertSource src,
                         int hidden, int intermediate, cudaStream_t stream, int scale_group,
                         expert::ExpertProjection projection) {
  check_geometry(hidden, intermediate);
  check_packed(dst, scale_group);
  for (int m = 0; m < 3; ++m)
    if (!src.payload[m] || !src.scales[m] || (scale_group == 16 && !src.globals[m]))
      throw std::invalid_argument("MoE FP4 packing: null source matrix");
  if (projection != expert::ExpertProjection::kDown)
    pack_payload<true><<<blocks(2 * intermediate * hidden / 8), kThreads, 0, stream>>>(
      reinterpret_cast<uint32_t*>(dst.w13), src.payload[0], src.payload[1], 2 * intermediate, hidden);
  if (projection != expert::ExpertProjection::kGateUp)
    pack_payload<false><<<blocks(hidden * intermediate / 8), kThreads, 0, stream>>>(
      reinterpret_cast<uint32_t*>(dst.w2), src.payload[2], nullptr, hidden, intermediate);
  if (dst.mx_nibbles) {
    if (projection != expert::ExpertProjection::kDown)
      pack_mx_scale_words<true><<<blocks(2 * intermediate * (hidden / 32) / 8), kThreads, 0, stream>>>(
        reinterpret_cast<uint32_t*>(dst.s13), src.scales[0], src.scales[1], 2 * intermediate, hidden);
    if (projection != expert::ExpertProjection::kGateUp)
      pack_mx_scale_words<false><<<blocks(hidden * (intermediate / 32) / 8), kThreads, 0, stream>>>(
        reinterpret_cast<uint32_t*>(dst.s2), src.scales[2], nullptr, hidden, intermediate);
  } else {
    if (projection != expert::ExpertProjection::kDown)
      move_scales<true, false><<<blocks(2 * intermediate * (hidden / scale_group)), kThreads, 0, stream>>>(
        nullptr, nullptr, dst.s13, src.scales[0], src.scales[1], nullptr, 2 * intermediate, hidden, scale_group);
    if (projection != expert::ExpertProjection::kGateUp)
      move_scales<false, false><<<blocks(hidden * (intermediate / scale_group)), kThreads, 0, stream>>>(
        nullptr, nullptr, dst.s2, src.scales[2], nullptr, nullptr, hidden, intermediate, scale_group);
  }
  DGPP_CUDA_OK(cudaGetLastError());
  if (scale_group == 16) copy_globals<<<1, 32, 0, stream>>>(dst.g13, dst.g13 + 1, dst.g2,
                                    src.globals[0], src.globals[1], src.globals[2]);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_fp4_unpack(Fp4ExpertDestination dst, Fp4PackedExpert src,
                           int hidden, int intermediate, cudaStream_t stream, int scale_group) {
  check_geometry(hidden, intermediate);
  check_packed(src, scale_group);
  for (int m = 0; m < 3; ++m)
    if (!dst.payload[m] || !dst.scales[m] || (scale_group == 16 && !dst.globals[m]))
      throw std::invalid_argument("MoE FP4 packing: null destination matrix");
  unpack_payload<true><<<blocks(2 * intermediate * hidden / 2), kThreads, 0, stream>>>(
      dst.payload[0], dst.payload[1], reinterpret_cast<const uint32_t*>(src.w13), 2 * intermediate, hidden);
  unpack_payload<false><<<blocks(hidden * intermediate / 2), kThreads, 0, stream>>>(
      dst.payload[2], nullptr, reinterpret_cast<const uint32_t*>(src.w2), hidden, intermediate);
  if (src.mx_nibbles) {
    move_scales<true, true, true><<<blocks(2 * intermediate * (hidden / scale_group)), kThreads, 0, stream>>>(
        dst.scales[0], dst.scales[1], nullptr, nullptr, nullptr, src.s13, 2 * intermediate, hidden, scale_group);
    move_scales<false, true, true><<<blocks(hidden * (intermediate / scale_group)), kThreads, 0, stream>>>(
        dst.scales[2], nullptr, nullptr, nullptr, nullptr, src.s2, hidden, intermediate, scale_group);
  } else {
    move_scales<true, true><<<blocks(2 * intermediate * (hidden / scale_group)), kThreads, 0, stream>>>(
        dst.scales[0], dst.scales[1], nullptr, nullptr, nullptr, src.s13, 2 * intermediate, hidden, scale_group);
    move_scales<false, true><<<blocks(hidden * (intermediate / scale_group)), kThreads, 0, stream>>>(
        dst.scales[2], nullptr, nullptr, nullptr, nullptr, src.s2, hidden, intermediate, scale_group);
  }
  DGPP_CUDA_OK(cudaGetLastError());
  if (scale_group == 16) copy_globals<<<1, 32, 0, stream>>>(dst.globals[0], dst.globals[1], dst.globals[2],
                                    src.g13, src.g13 + 1, src.g2);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
