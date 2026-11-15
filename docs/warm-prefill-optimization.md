# Warm prefill optimization

This work separates warm expert execution from prefix-cache reuse and cold
storage costs. The diagnostic replays a tool-enabled request with
`prefix_cache=false`: every request computes all 2,209 prompt tokens. After
warming, both ranks report 100% expert hits, zero expert reads, zero
reconstruction/layout preparation, and zero swap. The deployment retains
TP2 NVFP4, 1M context, the 100 GiB expert reservation per rank, four expert
read workers, direct Engram caching and fixed speculative depth three.

| Configuration | Warm prefill (ms), two successive requests |
|---|---:|
| Previous production | 2,080 / 2,086 |
| Resident-first grouping | 1,951 / 1,952 |
| Resident-first + contiguous prepared-weight copies | 1,418 / 1,412 |

These are focused serving diagnostics, not a broad throughput/concurrency
campaign. Responses, tool arguments, usage and speculative acceptance match:
50 output tokens, 13 target passes and 36 accepted predictions. Decode remains
roughly 0.92–0.93 seconds; the changes target prefill. The first resident-first
request and the copy candidate each read 95.317 GiB per rank and took 32.715
and 31.806 seconds of prefill respectively. Those cold results are reported
separately, rather than used to dismiss the warm improvement.

The prepared NVFP4 layout follows MMA lane ordering. The old prefill kernel
gathered it byte by byte into canonical row order, then decoded it into MMA
fragments again. The new path copies contiguous 512-byte n64/k16 slabs with
`cp.async`, retaining the prepared layout in shared memory and decoding the
fragments directly. It preserves the ascending-k16 accumulation chain, scale
arithmetic, BF16 intermediate and FP32 down outputs. The existing fixture
compares prepared and canonical results bitwise, including mixed misses and
cache pressure. `DGPP_MOE_PACKED_PREFILL_COPY=0` retains the previous reader
for diagnosis; the contiguous copy is the default.

Resident-first grouping is controlled by `DGPP_MOE_RESIDENT_FIRST=1`.
Its previous cold-heavy pair did not demonstrate a gain, but the zero-read
warm case does. This illustrates why cold, warm and prefix-cache performance
must be assessed separately.

A short two-rank warm capture of the copy candidate found about 0.59 seconds
in routed-expert prefill, 0.21 in attention, 0.19 in dense projections and
0.13 in communication. GPU-idle gaps were about 21 ms per rank. Categories
are attribution within a diagnostic trace, not additive benchmarks; the
prefill-to-first-target-graph interval also includes the transition into
generation. Host synchronization API durations include waiting for GPU work
and must not be interpreted as additional idle time.

A larger 4,559-token request continues to read approximately 37 GiB per rank
on repeated passes. It is a warm-cache, miss-heavy workload, not an all-resident
execution measurement. A fourfold improvement has not been demonstrated;
remaining costs span expert kernels, attention, dense projections and
communication, rather than a single CPU gap or an established SSD limit.

The compact-tile follow-up passed bitwise fixtures but recorded 1,424 / 1,421 ms
warm prefill, versus 1,418 / 1,412 ms for contiguous copying alone. Although it
removed empty blocks from highly skewed launch grids, it did not improve serving.
That candidate was removed rather than enabled or retained as another runtime
option. The final deployment keeps resident-first grouping and contiguous copies.

The baseline runtime source is `8eea0cfc`. The retained candidate was built from
that source plus the prepared-copy kernel change in this delivery; the version
stamp predates the final commit. Both ranks had matching binary hashes:

| Runtime | SHA-256 |
|---|---|
| Baseline / resident-first | `d8398763ccb2006e1c0c7d768cf22e6b5c404a6557654330576974eb87e9b47c` |
| Retained prepared-copy implementation | `d6f15940e8e458194135ceef6d478331eb5ff129decec276bd32508f00bafa6e` |

## Bounded demand read-ahead follow-up

The next pass queues already-selected experts as demanded reads, independently
of background-prefetch pressure gates. It uses the existing staging pool,
limits outstanding look-ahead to half that pool, and looks ahead at most two
selected experts. It neither admits nor evicts weights early. Demand timing
includes time spent queued. `DGPP_EXPERT_DEMAND_LOOKAHEAD=0` restores the prior
speculative-prefetch behavior; demand look-ahead is enabled by default.

Attention row metadata is now generated on the GPU, removing its host staging
and stream synchronization. Wave-local activation skips unnecessary full-buffer
clears; the W4A4 full-batch path retains them. Native chunk unpacking no longer
clears padding that the matrix kernels do not consume.

The retained NVFP4 configuration increases staging buffers and queue depth from
16 to 64, retaining four workers, 24 execution slots and the same 100 GiB cache
reservation per rank. Staging is charged within that reservation.

| Focused diagnostic | Previous production | Demand, 16 buffers | Demand, 64 buffers |
|---|---:|---:|---:|
| 2,209-token cold prefill, seconds | 31.8274 | 31.0823 | 25.3854 |
| 2,209-token steady warm prefill, seconds | 1.4229 / 1.4239 | 1.4135 / 1.4108 | 1.4112 / 1.4150 |
| 4,559-token first prefill after small workload, seconds | 9.8376 | 8.7631 | 7.4103 |
| 4,559-token repeated miss-heavy prefill, seconds | 16.4010 / 15.6908 | 13.7376 / 13.6382 | 10.8163 / 10.3771 |

These are focused diagnostics, not published deployment campaign throughput
rows. Every request recomputes its prompt (`prefix_cache=false`). Outputs,
normalized tool arguments, usage and speculative acceptance match across all
arms; both ranks have zero swap. Cold reads are exactly 95.316833496 GiB per
rank in every arm. Steady warm requests have zero expert reads and 100% hits.
Repeated large requests read approximately 36.5–37.8 GiB per rank, with small
arm-dependent differences, so the miss-heavy gain cannot be attributed solely
to overlap. Warm decode remains approximately 0.92 seconds; this pass establishes
no material warm speedup or broader decode/concurrency claim.

The measured source is `d0504bac` plus this follow-up's implementation changes.
Baseline SHA-256 is `d6f15940e8e458194135ceef6d478331eb5ff129decec276bd32508f00bafa6e`;
candidate SHA-256 is `3be30c66f80d007ef538eb62a151ead2d782a778f0cad0630b80a3928de9c6f8`.
Both candidate arms explicitly enabled demand look-ahead. The final source
makes that same path the default; the deployed rebuild is distinct from the
measured binary. No profiling or concurrent builds ran during these requests.

Validation: 27 expert-cache tests and two attention-layer tests passed on Spark.
The prepared/mixed/pressure B12x paging fixture and generic NVFP4/MXFP4 paged-wave
fixture also passed, preserving bitwise results. The new host regression covers
both cache policies, pressure gates, deduplication, bounded staging and unchanged
residency before acquisition.

### Remaining review work

- Dedicated copy streams and per-slot completion fences are not implemented.
  Acquisition/admission still drains the compute stream. Cache writes, eviction,
  relocation and scratch reuse need explicit reader/write dependencies before
  removing those drains; demand read-ahead does not provide full copy/compute overlap.
- Large prefill still launches whole reconstructed waves. Independent expert
  readiness needs the same buffer-lifetime and execution-order integration.
- Chunk-direct projection kernels are not implemented. They must preserve FP32
  accumulation across chunks and final scaling; summing independently rounded
  partial outputs is not equivalent. Reconstruction/scatter therefore remains.
- W4A4 activation quantization is unchanged: its quality and correctness gate
  has not been performed. Smaller BF16 tile designs remain possible; the earlier
  rejected compact grid does not exhaust them.
- No worker-count or bandwidth sweep was run. The bounded capacity comparison
  already demonstrates a useful gain with four workers. No fourfold warm gain
  or full concurrency campaign is claimed.
