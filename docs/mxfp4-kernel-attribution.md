# Why the existing MXFP4 prefill kernel beat the native prototype

Follow-up: [native serving integration and final comparison](mxfp4-native-serving.md).
The results and deployment statements below describe the earlier screen.

This is a causal follow-up to the [kernel follow-through](mxfp4-kernel-followthrough.md).
The comparison concerns grouped expert **prefill**, not the b12x verification
backend. It replays one real 4,096-token, top-six layer routing pattern on
GB10, with identical synthetic activations, weights and nonuniform scales.
All experts are already in GPU-accessible memory. There is no SSD I/O,
CPU routing decision or network transfer inside these kernel timings.
These are diagnostics, not deployment campaign rows or checkpoint-quality
measurements.

## The missing weight-line lookahead

Both kernels read packed four-bit weights from global memory. Native FP4
arithmetic does not reduce weight bytes relative to the BF16-activation
kernel: the latter expands only register fragments immediately before
MMA, rather than writing a full BF16 weight matrix to memory.

The existing canonical kernel explicitly prefetches upcoming weight rows
into L2 beyond the shared-memory ring's lookahead. The native prototype
had asynchronous copies but no equivalent cache-line lookahead. Its
32-byte weight-row slice per K64 stage could still arrive too late for
the consumer. Asynchronous copy alone does not guarantee hidden latency.

Adding weight-line prefetch to the native prototype preserves its exact
accumulators. The first isolated comparison, including activation
quantization, changed gate/up from 10.70 to 7.97 ms and down from 11.82 to
9.30 ms. Current canonical BF16 arithmetic took 9.50 and 10.16 ms in that
run. The prototype was therefore not intrinsically slower because it used
native FP4 instructions.

Nsight's separate gate/up captures support that mechanism:

| Counter | Native without lookahead | Native with lookahead |
|---|---:|---:|
| Registers/thread | 71 | 71 |
| L2 throughput, percent of peak | 60.39 | 85.45 |
| Compute throughput, percent of peak | 30.10 | 40.30 |
| Eligible warps/scheduler | 0.41 | 0.61 |
| Cycles per issued instruction | 20.77 | 15.44 |
| Executed warp instructions | 1.317 billion | 1.406 billion |

The counter captures use the two-line native lookahead variant.
The faster version executes more instructions and does not increase
register-limited theoretical occupancy. The unprefetched capture reports
7.7 warp cycles waiting on L1TEX scoreboard dependencies and 6.9 at CTA
barriers; these warp-level stall counts are not additive wall-clock costs. Prefetch increases the amount of work ready to issue.
This is GPU-memory prefetch of known addresses, not speculative expert
selection or SSD prefetch.

## Counterfactual and repeated results

The existing kernel was tested with weight-line prefetch on/off/off/on in
separate processes. Each process used the same binary and route replay;
results below are medians of the per-process median timings. Native totals
include quantization; preparation is excluded from prepared-layout timing.

| Kernel / layout | Gate/up projection, ms | Down projection, ms |
|---|---:|---:|
| Existing canonical, normal lookahead | 9.284 | 10.145 |
| Existing canonical, weight lookahead disabled | 11.099 | 12.293 |
| Existing prepared, direct slab copies | 9.216 | 10.859 |
| Native canonical, no weight lookahead | 10.365 | 11.786 |
| Native canonical, one upcoming line | **7.711** | **9.130** |
| Native canonical, two upcoming lines | 7.911 | 9.271 |

Disabling just the existing weight prefetch makes its canonical path about
20–21% slower. Adding the same one-line lookahead to native arithmetic cuts
its total time by about 26% for gate/up and 23% for down. The native result
is also about 16% lower latency than the prepared baseline in both
projections. This is a per-projection microbenchmark result, not a model
speedup. No quality change is introduced by prefetch itself: all native
schedule variants match the original native accumulators exactly.

## Other differences tested separately

The existing kernel skips padded m16 slabs, reuses B fragments across four
token slabs per warp, and has one barrier per steady-state K64 iteration.
The original native prototype computed all padded rows, split token rows
across two warp groups, and had an additional trailing barrier.

For this routing pattern, native full tiles cover 39,296 rows while the
existing live m16 slabs cover 27,424, for 24,576 real assignments. Skipping
empty slabs removes about 30% of the full-tile row work. Nevertheless,
porting this skip, changing warp layout, or removing the redundant barrier
individually barely changes the native timings. Even their combination
remains near the original runtime without weight prefetch. They are valid
implementation differences, but were not the dominant performance gap.

## Prepared weights are a separate advantage

Production resident experts use a prepared K16/N64 layout. The existing
kernel copies contiguous 512-byte slabs directly into shared memory and
extracts MMA fragments there. Raw scale bytes are also arranged for the
consumer. Preparation moves bits, does not change weight precision, and
happens outside expert execution. Canonical weight-row prefetch is not used
for these prepared views.

Disabling direct slab copies while keeping the same prepared weights and
arithmetic forces byte-wise gathering back to canonical shared-memory
order. That takes **18.695 ms gate/up and 20.363 ms down**, compared with
9.216 and 10.859 ms for direct copies. Those fallback figures are one
additional process's median timings, not the four-process aggregation.
Prepared layout is therefore useful because the consumer reads it directly;
the layout alone is not enough. It is not universally faster than an
optimized canonical path, as the down-projection results above illustrate.

The native prototype still uses canonical views. A faster canonical
projection is not a demonstrated full-model improvement: it must support
prepared/chunk-direct sources, retain shared-expert overlap and graph
execution, and be measured with actual checkpoint activations and serving
acceptance. The standalone native equality checks only prove that each
optimization preserves the prototype's already-quantized arithmetic.

## Reproducing the focused comparisons

`mxfp4_kernel_bench ROUTE_IDS [both|gate|down]` accepts whitespace-separated
expert IDs, six per input token. `DGPP_MXFP4_BENCH_ABLATIONS=1` selects the
causal schedule comparisons; the default retains the earlier tile/layout
sweep. `DGPP_MXFP4_BENCH_PREPARED=1` also prepares and checks the production
layout outside timing.

The `packed` rows contain tokens, N, K, tile M, tile N, matrix-load flag,
eight-warp flag, pipeline stages, staged-scale flag, compact-list flag,
column-scale flag, schedule bitmask, kernel milliseconds, and milliseconds
including quantization. Schedule bits are: 1 = one barrier, 2 = skip empty
m16 slabs, 4 = reuse weights across all token slabs in a warp, 8 = weight
prefetch, 16 = limit that prefetch to one upcoming line rather than two.

`DGPP_MOE_FP4_WEIGHT_PREFETCH=0` disables only the existing kernel's canonical
weight-line prefetch for the counterfactual. Scale prefetch and arithmetic
remain unchanged. The default is enabled. `DGPP_MOE_PACKED_PREFILL_COPY=0`
selects the existing prepared-layout gather path instead of contiguous
slab copies. Both controls are diagnostic; production settings are retained.

The counterfactual benchmark binary SHA-256 is
`1860b3d2496267ba89d0b03c19e5fbc7ec150074b98bc69df71c23934c6e198b`,
built from `1bacd8bc` plus this change. The saved implementation diff and
profiling logs remain untracked. The two-node service was stopped during
single-GB10 kernel measurements. Production is restored with the prior
MXFP4 server binary and enlarged pool; no native serving backend or new
quality tradeoff is enabled by this investigation.
