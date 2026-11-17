# MXFP4 kernel alternatives on GB10

This is the focused follow-up to the [warm execution profile](mxfp4-warm-execution-profile.md)
and [execution changes](mxfp4-execution-optimizations.md), on two Sparks on
17 November 2026. These are diagnostic comparisons, not deployment benchmark
rows or a concurrency campaign. The production cache budget and four-row
DSpark verification shape remain unchanged.

## Geometry and hardware corrections

GB10 reports 48 SMs, 1,536 threads per SM, 102,400 bytes of shared memory per
SM and 101,376 bytes of opt-in shared memory per block. The proposed 128-row
expert tile uses about 75 KiB, so only one such block fits per SM. The
64-row tile uses about 46.5 KiB and allows two. The review's 228 KiB premise
does not apply to this device.

Serving splits the 24,317-token prompt into 4,096-token prefill chunks.
Uniform routing over 384 experts at top-6 gives about 64 rows per expert
for one chunk, rather than 380. Both shapes were screened, but serving
performance determines the configuration. Verification uses four rows and
24 routed assignments at depth three; a six-row synthetic result is not its
baseline.

## Wider expert tile

`DGPP_MOE_MX_PREFILL_ROWS128=1` selects a 128-row tile for MXFP4 prefill
segments exceeding 64 rows. The default remains 64. The arithmetic retains
the ascending k16 accumulation chain. The native MXFP4 paging fixture
passed its bitwise canonical-output checks with the switch enabled.

On the same binary, the final warmed 24k prefill changed from **12.092 to
13.498 seconds**, an 11.6% regression. Resident 2,209-token prefill changed
from 1.358 to 1.608 seconds. Long-prompt decode remained about 47 tokens/s.
The compared requests had the same acceptance and expert read volume
(5.159 GiB per rank). This option remains disabled.

## Matched b12x formats

`mxfp4_backend_bench` uses the shipped AOT backend, 384 prepared experts,
H=5120, local intermediate size 1152, top-6, CUDA graph replay, and equal
routing patterns across formats. Preparation is outside timing. The full
weight allocation exceeds L2; repeated selected experts may still benefit
from cache. These are routed-kernel times, excluding shared experts,
routing, paging, and communication. Constant finite synthetic weights do
not establish checkpoint quality or end-to-end performance.

| Rows | Unique experts | NVFP4, microseconds | MXFP4, microseconds |
|---:|---:|---:|---:|
| 4 | 7 | 370.832 | 288.913 |
| 4 | 14 | 725.663 | 564.332 |
| 4 | 22 | 1123.449 | 875.551 |
| 6 | 7 | 374.773 | 292.124 |
| 6 | 14 | 730.414 | 567.219 |
| 6 | 22 | 1128.627 | 879.206 |
| 6 | 29 | 1466.957 | 1154.066 |
| 6 | 36 | 1807.296 | 1426.155 |

MXFP4 is about 22% faster at matched four-row shapes. This does not support
the proposed MXFP4 scale-path regression. It does not establish optimal
absolute efficiency or rule out a better small-batch kernel.

## Native MXFP4 weights with MXFP8 activations

`mxfp4_kernel_bench` contains a benchmark-only W4A8 prototype. The exact
`mma.sync.kind::mxf8f6f4` E4M3-by-E2M1 instruction assembles and runs for
SM121a. E2M1 codes occupy bits 2–5 of the instruction's byte representation.
Weights remain packed in global memory and are expanded into shared memory;
activations are quantized per 32 values. Timing includes that quantization.
The prototype reads canonical expert views, not production prepared or
chunk-direct views. It is not a serving backend.

| Tokens | Rows/expert | Projection N × K | Current W4A16, ms | Native W4A8, ms |
|---:|---:|---:|---:|---:|
| 4096 | 64 | 1152 × 5120 | 8.464 | 13.610 |
| 4096 | 64 | 5120 × 1152 | 8.803 | 14.306 |
| 24317 | 380 | 1152 × 5120 | 33.925 | 56.856 |
| 24317 | 380 | 5120 × 1152 | 35.628 | 53.788 |

The prototype is 51–68% slower. Random-input relative L2 difference from
unquantized BF16 arithmetic was about 2.44%; this is a kernel sanity check,
not a model-quality evaluation. No serving integration is justified by
these results. Prepared native layouts, improved weight staging, small-row
tiles and fused quantization remain possible future work; this prototype
does not exhaust native W4A8's potential.

## Native dense FP8

`DGPP_DSV41_NATIVE_FP8_PREFILL=1` enables native FP8 CSA2 projections for
compatible 32×32 E8M0 weight-scale grids. Mode `2` limits the change to
small (at most 128-row) batches or projections with at least 5,120 outputs.
The loader expands E8M0 scales into floats; the serving kernel recovers the
original exponent exactly. Generic FP32-scale model paths are unchanged. It uses the existing GEMM workspace,
quantizes activations per 32 values with the DSV reference's 1e-4 amax floor,
and uses a 64×128 tile. Unsupported shapes or insufficient workspace fall
back. Small-row decode is unchanged. Global weight storage was already FP8:
this does **not** halve weight bytes.

The isolated `dsv41_native_dense_bench` includes activation quantization.
Its grid-scale output exactly matched the existing native kernel with
expanded per-row scales; relative L2 difference from BF16-activation
arithmetic was about 2.44% on random inputs. At 128 rows, the tested shapes
were faster. At 4,096 rows, narrow 512/1024-output projections were slower,
while 5120-output projections were faster. Those weights may fit L2; these
numbers are not DRAM bandwidth measurements. The first serving arm connected only the byte-scale entry point and did
not select the intended native kernel; its timings are excluded. The
corrected arm targets CSA2 directly and supplies the existing workspace to
previously unsplit prefill projections, while preserving decode dispatch.
Shared-expert and Engram projections are not converted by this option.

The corrected full serving screen selected the native path on both ranks.
The new grid-scale fixture passed FP64-reference checks, exact activation
codes/scales including the minimum floor, byte versus expanded-float scales,
BF16 and FP32 outputs, padded rows, workspace guards and graph replay.

| CSA2 dense mode | Resident 2,209-token prefill, s | Warm 24,317-token prefill, s | Short decode, tokens/s | Long decode, tokens/s | Long expert reads, GiB/rank |
|---|---:|---:|---:|---:|---:|
| Existing BF16 arithmetic | 1.364 | 12.157 | 55.64 | 47.32 | 5.159 |
| All compatible native FP8 | 1.352 | 13.101 | 55.61 | 42.91 | 10.564 |
| Selective native FP8 | 1.354 | 13.091 | 55.60 | 41.81 | 11.145 |

These are the final warmed samples of each arm, not campaign aggregates.
Decode rates are the engine's decode steps, excluding prefill and queueing.
All three short probes were resident with zero expert reads. Short and long
responses matched after normalizing generated tool-call IDs, as did
speculative acceptance (36/13 accepted predictions/draft passes for short;
38/14 for long). Activation quantization nonetheless changed internal
routing and roughly doubled long-request expert reads. The small resident
prefill improvement does not offset the long-request regression, so both
native modes remain disabled. This does not isolate native-kernel speed
from the changed cache workload, nor establish a quality loss: a broader
quality test was not needed to reject a slower serving configuration.

## Four-way attention merge

`DGPP_CSA2_PREFILL_MERGE4=1` merges the four main-source partials in the
listed attention kernel, using the existing ticket mechanism. The window
source stays separate. The finish kernel still applies window combination,
sink and inverse rotation: this is not complete finish fusion.

The split-group test now covers groups four and eight, repeatability,
row-subset invariance and comparison against the independent combine.
The DSA suite passed 53 tests and the CSA2 merge-enabled suite passed three.

With the same binary, final warm 24k prefill was **12.148 seconds without
merging versus 12.573 seconds with merging**, 3.5% slower. Short prefill was
essentially unchanged (1.361 versus 1.360 seconds). Generated wording and
acceptance changed, so decode rates are not an isolated speed comparison.
Long-request expert reads also rose from 5.159 to 8.200 GiB per rank: the
end-to-end regression cannot be attributed solely to merge-kernel overhead.
The resident short probe required no expert reads in either arm and showed
no material prefill gain. The merge remains disabled.

## Measurement boundaries

Serving arms use identical tool requests, warm-up order and the same binary
within each comparison, with prefix reuse disabled for full-prefill probes.
Zero swap and matching operation streams are checked on both ranks. Source
revision, working diff and binary hashes are recorded. Generated results and
logs remain untracked. No other GPU test or serving world overlaps timing.
Cold-start responses are retained but do not determine warm-path selection.

These screens do not establish broad model quality, concurrency scaling or
the optimum across all kernel designs. A negative prototype is retained as
an explicit result rather than described as an unmeasured serving win.

The wider-tile serving binary was
`ec1da307a3df6b6f79f269f78658c2662d7734cb4b5a98647e3af9a984417d98`;
the attention screen used
`f4599e8d0025659c92aa9ab9a1b5675a72b5e99e9663615200a020cf121d0447`.
The corrected dense screen uses
`66702e1f496036d8c75c74cbcb5d8c50ebba5395715c083d1f17efd44071cee6`.
All were built from `5f1c7b5f` plus this change's working diff.

## Retained configuration and remaining work

None of the new alternatives improved the deployed workload overall.
Production retains MXFP4, four attention splits, early shared-expert prefill,
12 execution slots, the existing cache budget and depth-three speculation.
The original production binary is retained:
`f2fd9a2406e2f46efb01af8bc91fecacbb0afbe5d68c03f18b826201939fac18`.
The service port, credentials and client model alias are unchanged. The
new switches remain opt-in and disabled.

Not implemented: a prepared/chunk-direct native W4A8 serving backend (the
canonical prototype lost), full attention finish fusion (the four-way merge
alone lost), or native conversion of Engram/shared projections (the tested
CSA2 conversion did not establish a serving win). Wider native tile tuning,
TMA and distributed shared-memory designs remain future work. These results
do not claim that the underlying arithmetic ideas cannot eventually win.
