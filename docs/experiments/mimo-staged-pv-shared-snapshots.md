# MiMo MTP attention and snapshot candidates

Baseline source: `149bc6f`. Experiments and build logs:
`/mnt/benchmarks/dgpp-mimo-attention-20260923`.

These candidates are not enabled by default and have no measured Spark speedup
at this stage. Retain native MTP3, C2, 262144 context/request and four prefix
slots. Do not enable DFlash as part of this work.

## Staged PV tiles

`DGPP_MIMO_PV_KEY_TILE=16|32|64` (default 16) controls the fused materialized
prefill PV kernel. Stage 32/64 keys per CTA iteration, with paired contiguous
cache loads. All output fragments retain the established ascending 16-key MMA
chain, including the tail. Larger stages reduce barriers and make score loads
more contiguous; the tradeoff is shared memory/occupancy. This remains
materialized attention, not a FlashAttention replacement. Decode is unchanged.

## Direct FP8 expansion

CMake `DGPP_MIMO_FP8_KV_INTEGER_LOAD=ON` expands E4M3 bytes directly into BF16
bits instead of converting via FP32. Finite values are exact. Signed zero and
CUDA-canonical positive quiet NaN are explicit. Larger PV stages also load pairs of
adjacent cache elements. Existing fast-load and scalar paths remain controls.
Run the all-256-code GPU/graph test against CUDA's original conversion before
using the candidate with real weights. CPU equivalence alone is insufficient.

## Shared snapshot storage

`DGPP_MIMO_SHARED_SNAPSHOTS=1` changes registered prefix-arena snapshots only.
Complete 256-token global-attention blocks can be shared by snapshots on the
same verified request lineage, including a request attached from a snapshot.
Partial blocks and sliding-window state are private. MTP state remains in each
slot's private allocation with its existing copy/restore semantics. Raw external
snapshot callers retain the original layout.

Live attention caches are still contiguous and independent per request. This
is not a paged live KV implementation and does not raise concurrency. Request
reset drops lineage; rollback drops blocks crossing the rollback boundary.
Only snapshots own blocks; weak lineage cannot pin evicted memory. Copies,
async allocations and frees use the model stream. These operations are outside
decode graph capture. CUDA memory-pool support and cost need hardware validation.

Admission still derives four slots from the original worst-case byte budget.
The arena allocates only private/MTP storage; populated global blocks are
allocated on demand. `prefix_cache.snapshot_bytes` remains the worst-case
per-slot budget. `snapshot_storage_bytes` reports owned storage across slots,
including the private arena. It excludes memory retained by CUDA's allocator
and must not be equated with process/device resident memory.

## Validation sequence

1. CPU all-code expansion and snapshot ownership tests, including ASan/UBSan.
2. Both-rank GPU conversion, fused/materialized parity and graph replay tests
   at tile sizes 16, 32 and 64, plus small memcheck/racecheck runs.
3. Attention microbenchmarks with BF16 and FP8, full and sliding attention,
   at 8K/64K/128K; fixed shapes and repeated controls, no service timing overlap.
4. New uncontended serving baseline, then attention-only, cache-only and
   snapshots-only comparisons on the same source/config. Record hashes, flags,
   metrics, raw requests/SSE and outputs. Include cold and cached long prompts,
   C1/C2 native MTP, real harness tools, cancellation and branch/restore tests.
5. Combine only candidates passing correctness and end-to-end performance
   gates. Preserve a slower candidate's evidence; leave it disabled.

The first both-rank GPU gate rejected the negative NaN encoding (0xff): CUDA
expands it to positive NaN. The direct converter was corrected; both encodings
now follow the GPU reference. This failure is retained in the experiment logs.

At preparation time four focused host tests passed, including sanitizers.
The attention/cache-only cross-build passed. Integrated cross-build and all
Spark tests are tracked separately in the experiment state.

## Hardware follow-up (in progress)

Both ranks passed finite BF16/FP8 tiled parity, graph replay and snapshot tests.
All256 FP8 codes pass after correcting CUDA's NaN sign behavior. The first
larger-PV and integer-load candidates were slower and are not selected.

Additional opt-in paths:
- `DGPP_MIMO_FP8_PREFILL_BRIDGE=1` expands full-attention FP8 KV once per tile
  into the otherwise-unused probability workspace, after reserved normalizers.
  It runs only if the expanded planes fit; no additional persistent allocation.
- `DGPP_MIMO_SWA_QK_KEY_TILE=32|64` groups adjacent score tiles and shares Q
  loads for sliding-window layers. Full layers keep the established kernel by
  default; `DGPP_MIMO_QK_KEY_TILE` controls separate experimental full-layer
  grouping, which was slower in the first microbenchmarks.
- `DGPP_MIMO_QK_GEMM=1` uses zero-workspace cuBLASLt for full-attention QK,
  followed by the original score-rounding operations. Tiles containing a raw
  dot product within32 FP32 ULPs of a BF16 halfway point are recomputed using
  the original WMMA kernel. This is a targeted numerical repair, not a proof
  of bitwise equivalence for all possible activations. Small synthetic cases
  currently match exactly with repair. Raw GEMM without repair missed the
  independent FP64 pointwise gate and is not the implementation being tested
  for serving. The repair gives back most of raw GEMM's speed improvement.

The CPU FP64 oracle computes dot products and value sums independently while
retaining the model's BF16 score/difference/probability rounding points.
Its numerical mode is diagnostic; bitwise mode remains the default gate.
All service comparisons must retain native MTP3 and use the same frozen source.
