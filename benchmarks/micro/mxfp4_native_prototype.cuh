#pragma once

// Benchmark-only canonical-weight prototype; not a serving backend.
#include <stdexcept>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/cuda_check.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/mxfp8_gemm.hpp"
#include "kernels/pdl.cuh"

namespace dgpp::mx4bench {
namespace {
constexpr int kThreads = 128, kStages = 3;

template <int BM, int BN, bool WideWarps = false>
struct Tile {
  static constexpr int warps = BM == 16 ? (BN < 16 ? 1 : BN / 16) : (WideWarps ? 8 : 4);
  static constexpr int threads = warps * 32;
  static constexpr int bk = BM == 16 && BN == 8 ? 512 : BM == 16 && BN == 16 ? 256 : 64;
  static constexpr int stride = bk + 16;
};

__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(src),
               "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() {
  asm volatile("cp.async.commit_group;");
}
template <int Count>
__device__ __forceinline__ void wait_pending() {
  asm volatile("cp.async.wait_group %0;" ::"n"(Count));
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                    uint32_t as, uint32_t bs) {
  // PTX/CUTLASS SM120 scale layout: A scale row = lane/4 + 8*(lane%2),
  // B scale row = lane/4; the byte/thread selectors are both zero.
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
      "m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11,%12}, {%13}, {%14,%15};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(as),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(bs),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
}

template <typename Out, int BM, int BN, bool PackedPipeline = false, bool MatrixLoads = false,
          bool WideWarps = false, int Stages = 3, bool StageScales = false, bool ColumnScales = false>
__global__ __launch_bounds__((Tile<BM, BN, WideWarps>::threads)) void mxfp8_gemm_kernel(
    const uint8_t* a, const uint8_t* a_scales, const uint8_t* w, const uint8_t* w_scales, Out* out,
    int m, int n, int k, size_t out_stride, const MoeSegment* segs, const MoeExpertView* views,
    int which, const int32_t* act_rows = nullptr, const MoeTile* tiles = nullptr,
    const int32_t* tile_count = nullptr, int tile_capacity = 0) {
  dgpp_pdl_wait();
  constexpr int kStages = Stages;
  constexpr int threads = Tile<BM, BN, WideWarps>::threads;
  constexpr int BK = Tile<BM, BN, WideWarps>::bk, stride = Tile<BM, BN, WideWarps>::stride;
  constexpr int warp_m = BM == 16 ? 1 : 2, warp_n = Tile<BM, BN, WideWarps>::warps / warp_m;
  constexpr int MT = BM / warp_m / 16, NT = BN / warp_n / 8;
  static_assert(!StageScales || BK == 64, "staged scales require two K32 groups");
  static_assert(!ColumnScales || StageScales, "column scales require staged loads");
  constexpr int wstride = PackedPipeline ? BK / 2 + 16 : stride;
  constexpr int data_bytes = BM * stride + BN * wstride;
  constexpr int slot_bytes = data_bytes + (StageScales ? (BM + BN) * 4 : 0);
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int m_tiles = (m + BM - 1) / BM;
  const int nt = (n + BN - 1) / BN;
  const int work = tiles ? int(blockIdx.x) / nt : 0;
  if (tiles && work >= *tile_count) return;
  const auto seg = segs[tiles ? tiles[work].seg : blockIdx.y];
  const int segment_tiles = (seg.rows + BM - 1) / BM;
  const int local = tiles ? (tiles[work].m0 / BM) * nt + int(blockIdx.x) % nt : 0;
  const int m0 = (tiles ? local % segment_tiles : int(blockIdx.x) % m_tiles) * BM;
  const int n0 = (tiles ? local / segment_tiles : int(blockIdx.x) / m_tiles) * BN;
  if (m0 >= seg.rows) return;
  m = seg.rows;
  const auto view = views[seg.expert * 3 + which];
  w = view.payload;
  w_scales = view.fp4_scales;
  out += size_t(seg.row0) * out_stride;
  const int row_base = (warp / warp_n) * (BM / warp_m);
  const int col_base = (warp % warp_n) * (BN / warp_n);
  const int group = lane / 4, part = lane % 4;
  const int steps = (k + BK - 1) / BK, scale_stride = k / 32;
  auto issue = [&](int step, int slot) {
    if constexpr (StageScales) {
      for (int r = tid; r < BM + BN; r += threads) {
        const bool is_a = r < BM;
        const int gr = is_a ? m0 + r : n0 + r - BM;
        uint16_t pair = 0x7f7f;
        if (gr < (is_a ? m : n)) {
          const int sr = is_a ? (act_rows ? act_rows[seg.row0 + gr] : seg.row0 + gr) : gr;
          const uint8_t* src =
              (is_a ? a_scales : w_scales) + size_t(sr) * scale_stride + step * (BK / 32);
          if constexpr (ColumnScales) {
            if (!is_a)
              pair = uint16_t(w_scales[size_t(step * 2) * n + sr]) |
                     (uint16_t(w_scales[size_t(step * 2 + 1) * n + sr]) << 8);
            else
              pair = *reinterpret_cast<const uint16_t*>(src);
          } else {
            pair = *reinterpret_cast<const uint16_t*>(src);
          }
        }
        *reinterpret_cast<uint16_t*>(smem + slot * slot_bytes + data_bytes + r * 4) = pair;
      }
    }
    if constexpr (PackedPipeline) {
      // Both operands travel through cp.async. B remains packed in shared
      // memory; expand only the register fragments consumed by MMA.
      constexpr int ac = BM * (BK / 16), bc = BN * (BK / 32);
      for (int chunk = tid; chunk < ac + bc; chunk += threads) {
        const bool is_a = chunk < ac;
        const int at = is_a ? chunk : chunk - ac;
        const int row = at / (is_a ? BK / 16 : BK / 32);
        const int col = at % (is_a ? BK / 16 : BK / 32) * 16;
        const int global_row = is_a ? m0 + row : n0 + row;
        const bool valid = global_row < (is_a ? m : n) && step * BK + col * (is_a ? 1 : 2) < k;
        uint8_t* dst =
            smem + slot * slot_bytes + (is_a ? row * stride : BM * stride + row * wstride) + col;
        const int ar = valid && is_a
                           ? (act_rows ? act_rows[seg.row0 + global_row] : seg.row0 + global_row)
                           : 0;
        const uint8_t* src =
            is_a ? a + size_t(ar) * k + step * BK + col
                 : w + size_t(valid ? global_row : 0) * (k / 2) + step * (BK / 2) + col;
        copy16(dst, valid ? src : (is_a ? a : w), valid);
      }
    } else {

#pragma unroll
      for (int chunk = tid; chunk < (BM + BN) * (BK / 16); chunk += threads) {
        const int row = chunk / (BK / 16), column = (chunk % (BK / 16)) * 16;
        const bool is_a = row < BM;
        const int global_row = is_a ? m0 + row : n0 + row - BM;
        const bool valid = global_row < (is_a ? m : n) && step * BK + column < k;
        auto* dst = smem + slot * slot_bytes + row * stride + column;
        if (is_a) {
          const int ar =
              valid ? (act_rows ? act_rows[seg.row0 + global_row] : seg.row0 + global_row) : 0;
          const uint8_t* src = a + size_t(ar) * k;
          copy16(dst, valid ? src + step * BK + column : src, valid);
        } else {
          // The native mixed instruction expects E2M1 in bits 2..5 of each byte.
          // Keep global weights packed at four bits; expand only in shared memory.
          uint64_t bits = valid ? *reinterpret_cast<const uint64_t*>(
                                      w + size_t(global_row) * (k / 2) + (step * BK + column) / 2)
                                : 0;
          uint4 v{};
          auto* words = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
          for (int q = 0; q < 4; q++) {
            uint32_t word = 0;
#pragma unroll
            for (int t = 0; t < 4; t++)
              word |= uint32_t((bits >> (4 * (q * 4 + t))) & 15) << (8 * t + 2);
            words[q] = word;
          }
          *reinterpret_cast<uint4*>(dst) = v;
        }
      }
    }
  };
  float accum[MT][NT][4] = {};
#pragma unroll
  for (int stage = 0; stage < kStages - 1; ++stage) {
    if (stage < steps) issue(stage, stage);
    commit();
  }
  for (int step = 0; step < steps; ++step) {
    wait_pending<kStages - 2>();
    __syncthreads();
    if (step + kStages - 1 < steps) issue(step + kStages - 1, (step + kStages - 1) % kStages);
    commit();
    const uint8_t* sa = smem + (step % kStages) * slot_bytes;
    const uint8_t* sw = sa + BM * stride;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 32) {
      if constexpr (BK > 64)
        if (step * BK + kk >= k) break;
      uint32_t af[MT][4], bf[NT][2], as[MT], bs[NT];
      const int scale_column = (step * BK + kk) / 32;
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int row = row_base + i * 16 + group;
        const uint8_t* p = sa + row * stride + kk + part * 4;
        if constexpr (MatrixLoads) {
          const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(
              sa + (row_base + i * 16 + lane % 16) * stride + kk + (lane / 16) * 16));
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                       : "=r"(af[i][0]), "=r"(af[i][1]), "=r"(af[i][2]), "=r"(af[i][3])
                       : "r"(address));
        } else {
          af[i][0] = *reinterpret_cast<const uint32_t*>(p);
          af[i][1] = *reinterpret_cast<const uint32_t*>(p + 8 * stride);
          af[i][2] = *reinterpret_cast<const uint32_t*>(p + 16);
          af[i][3] = *reinterpret_cast<const uint32_t*>(p + 8 * stride + 16);
        }
        const int scale_row = m0 + row + (lane & 1) * 8;
        if constexpr (StageScales)
          as[i] = smem[(step % kStages) * slot_bytes + data_bytes + (row + (lane & 1) * 8) * 4 +
                       kk / 32];
        else {
          as[i] = scale_row < m
                      ? a_scales[static_cast<size_t>(act_rows ? act_rows[seg.row0 + scale_row]
                                                              : seg.row0 + scale_row) *
                                     scale_stride +
                                 scale_column]
                      : 127;
        }
      }
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int row = col_base + j * 8 + group;
        if constexpr (PackedPipeline) {
          const uint8_t* p = sw + row * wstride + kk / 2 + part * 2;
          const uint32_t lo = *reinterpret_cast<const uint16_t*>(p);
          const uint32_t hi = *reinterpret_cast<const uint16_t*>(p + 8);
          bf[j][0] = (__byte_perm(lo, lo >> 4, 0x5140) & 0x0f0f0f0fU) << 2;
          bf[j][1] = (__byte_perm(hi, hi >> 4, 0x5140) & 0x0f0f0f0fU) << 2;
        } else {
          const uint8_t* p = sw + row * stride + kk + part * 4;
          bf[j][0] = *reinterpret_cast<const uint32_t*>(p);
          bf[j][1] = *reinterpret_cast<const uint32_t*>(p + 16);
        }
        const int scale_row = n0 + row;
        if constexpr (StageScales)
          bs[j] = smem[(step % kStages) * slot_bytes + data_bytes + (BM + row) * 4 + kk / 32];
        else {
          bs[j] = scale_row < n
                      ? w_scales[static_cast<size_t>(scale_row) * scale_stride + scale_column]
                      : 127;
        }
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) mma(accum[i][j], af[i], bf[j][0], bf[j][1], as[i], bs[j]);
    }
    // No warp may reuse a ring slot until every warp has consumed it.
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        const int row = m0 + row_base + i * 16 + group + (value / 2) * 8;
        const int column = n0 + col_base + j * 8 + part * 2 + value % 2;
        if (row < m && column < n) {
          if constexpr (std::is_same_v<Out, float>)
            out[static_cast<size_t>(row) * out_stride + column] = accum[i][j][value];
          else
            out[static_cast<size_t>(row) * out_stride + column] =
                __bfloat16_as_ushort(__float2bfloat16_rn(accum[i][j][value]));
        }
      }
}

template <class Out, int BM, int BN, bool PackedPipeline = false, bool MatrixLoads = false,
          bool WideWarps = false, int Stages = 3, bool StageScales = false, bool ColumnScales = false>
void run(const uint8_t* a, const uint8_t* as, Out* out, int maxrows, int experts, int n, int k,
         cudaStream_t stream, const MoeSegment* segs, const MoeExpertView* views, int which,
         const int32_t* act_rows = nullptr, const MoeTile* tiles = nullptr,
         const int32_t* tile_count = nullptr, int tile_capacity = 0) {
  constexpr int smem = Stages * (BM * Tile<BM, BN, WideWarps>::stride +
                                 BN * (PackedPipeline ? Tile<BM, BN, WideWarps>::bk / 2 + 16
                                                      : Tile<BM, BN, WideWarps>::stride) +
                                 (StageScales ? (BM + BN) * 4 : 0));
  dim3 grid(unsigned((tiles ? tile_capacity : (maxrows + BM - 1) / BM) * ((n + BN - 1) / BN)),
            tiles ? 1u : unsigned(experts));
  mxfp8_gemm_kernel<Out, BM, BN, PackedPipeline, MatrixLoads, WideWarps, Stages, StageScales,
                      ColumnScales>
      <<<grid, Tile<BM, BN, WideWarps>::threads, smem, stream>>>(
          a, as, nullptr, nullptr, out, maxrows, n, k, n, segs, views, which, act_rows, tiles,
          tile_count, tile_capacity);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace
}  // namespace dgpp::mx4bench
