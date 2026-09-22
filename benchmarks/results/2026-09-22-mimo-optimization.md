# MiMo optimization campaign, 2026-09-22

Objective: continue until measured gains diminish or optimization coverage
matches Qwen, preserving correctness and port 30001. Base remains
`3ee00e2641dd`; development changes are uncommitted. Qwen's established paths
include chunked prefill, concurrent sessions, graphs, prefix caching and MTP.
MiMo began this campaign with 128-row prefill, concurrency one, 32K context,
no graphs/cache/MTP, and scalar attention with retained QK scores.

Baseline deployment (preserved):
`/home/jon/dgpp/experimental/mimo-prefill-scores-20260921/`, binary SHA256
`f7076a87c929da8e8b0a554dec466e521053544b5f0c412aad4d4d87e0c95269`.
Prior verified time to first content: 1.14 s / 480 tokens, 5.50 s / 2030,
34.98 s / 8030, no prefix hits. The maintenance lock is used for this campaign.

## Final result

MiMo remains serving on both Sparks at port 30001: 65,536 tokens per request,
eight concurrent requests, CUDA graphs, 16 GiB prefix cache (22 slots), and
256/2,048 busy/idle prefill budgets. MTP is disabled. Cold 8,030-token TTFT
fell from 34.98 to 8.058 seconds (4.34x); aggregate end-to-end throughput is
52.02 tokens/s at four requests and 56.27 at eight. All eight tested next-turn
conversations reused their preceding turn. Local optimization has reached the
measured stopping point described below; full Qwen parity is not claimed.

## Tensor-core attention

A 16-query x 16-key WMMA BF16 tile replaces scalar QK. A second WMMA path
multiplies BF16 probabilities by V. FP32 softmax retains the published
BF16 score and probability roundings and the sink; only dot-product
association changes. Workspace is FP32 scores plus BF16 probabilities,
shared across layers. Bounds use absolute positions and the existing
expanded-ring retention contract. Decode retains the scalar control path.

The scalar reference comparison remains strict. The tensor-core comparison
uses the same one-BF16-ulp relative gate and 0.2% relative L2, with an explicit
1e-4 absolute floor for cancellation near zero. The first strict run exposed
one near-zero mismatch out of 65536 values: max absolute 0.000031, relative
L2 0.000001. This is not full-model numerical-equivalence evidence.

Real-weight 259-row dense/MoE probes pass the original layer gate. Maximum
per-row relative L2 versus single-row execution: 0.000212233 / 0.001077815.
Four attention groups pass; Compute Sanitizer memcheck and racecheck report
zero errors/hazards on Spark-1 for the first complete tensor-core path.

Isolated CUDA-event benchmark: TP2 attention geometry, 128 queries ending
at position 8191, five measured calls after warm-up. These are kernel-only
measurements, not end-to-end service claims.

| Variant | Global attention ms | Sliding attention ms |
|---|---:|---:|
| Scalar score reuse, original | 62.26 | 0.980 |
| Tensor-core QK only | 41.42 | 0.477 |
| Tensor-core QK and PV | 19.63 | 0.311 |
| QK/PV plus direct global indexing | 12.53 | 0.455 |

Nsight Systems on Spark-2 confirmed the complete tensor-core path's timings
(~19.82 ms global, 0.314 ms sliding) and separated score/value/softmax cost.
The report is `/tmp/mimo-tensor-attention.nsys-rep` on Spark-2. Profiling is
isolated from full-model startup; its timings are supporting evidence only.
The sliding regression in the direct-global-index variant motivates separate
power-of-two ring indexing validation rather than assuming a universal win.

Power-of-two ring indexing (256 slots for a 128-row chunk) passes the
259-row real-layer gates again. Isolated timings: 12.59 ms global, 0.395 ms
sliding. End-to-end deployment `88301757824359d0` with 128-row chunks then
returned correct ORCHID answers at 1.017/1.017 s TTFT for 480 tokens,
4.262 s for 2030, and 19.385 s for 8030 (zero cached tokens). The prior
8030-token result was 34.980 s. Raw results are
`/tmp/dgpp-mimo-opt-20260922/tensor-prefill.json`.

## Chunk size

512-row chunks (Qwen's current chunk size) pass dense/MoE comparisons over
1027 tokens, covering multiple logical-window wraps, a physical 1024-slot
ring wrap, and a three-token final chunk. Maximum per-row relative L2:
0.000234151 dense / 0.001142244 MoE. No tolerance change.

At deployment `9a3bdbc12a0523d2`, TTFT becomes 0.486/0.488 s at 480 tokens,
2.049 s at 2030, and 10.686 s at 8030; all return ORCHID with no cache hit.
Raw data: `/tmp/dgpp-mimo-opt-20260922/chunk512-prefill.json`. Durable build:
`/home/jon/dgpp/experimental/mimo-chunk512-20260922/`, SHA256
`770aeea71efc2a19d4f6bb530726aa333173ebbdd3ca65a21c69b84b39d2f8ad`.

The fixed merge-sort decode prompt at temperature zero and 256 output tokens
measured 12.188/12.221 s from first to last content on the 128-row tensor
build (~20.9 tokens/s). These are two trials of the same prompt; generated
content is retained in `decode-eager128.json` for trajectory comparison.

## Decode graphs and shared MoE workspace

Single-slot decode capture on both ranks contains 736 kernel nodes, including
97 collective nodes, with no memcpy, memset, host, or event nodes. The fixed
256-token merge-sort response takes 10.641/10.658 seconds first-to-last content,
versus 12.188/12.221 seconds eager. Both generated texts are exactly equal to
the eager control. Warm TTFT is 0.180 seconds. Deployment: `7e50dac6b761dedc`.

Sharing the sequential MoE execution workspace across 47 layers, while retaining
47 distinct stable graph weight tables, gives 10.613/10.636 seconds with exactly
the same text. This is a memory improvement, not a demonstrated speed gain.
At listening, rank-0 node available memory rises from 21.70 to 28.29 GiB.
Corrected memory planning counts both FP32 attention scores and BF16 probabilities;
shared-workspace plan is 86.49 GiB plus 4 GiB headroom at 32K context.
Deployment: `360b613f4266e907`. Reasoning, streaming usage, forced tool calls,
two queued requests, and oversized-context rejection all pass on this build.

## Prefix cache and request-plane kernels

Context-aware snapshot planning now accounts for MiMo's flat K/V geometry.
Single-slot graph deployment `eb9de67bd2bfbc08` enables 4 GiB of prefix cache.
Alternating 8030-token ORCHID/TULIP prompts retain the correct histories and
restore 8025 tokens. After the separate sanitizer process completed, four
warm requests take 0.420/0.415/0.411/0.412 seconds end-to-end, including four
completion tokens. Earlier overlapping-sanitizer timings are excluded.

Request-plane mapping adds an optional device slot-ID vector to append and
scalar attention. A graph replay test rotates four rows among four independent
histories, includes padding and request reset, and crosses ring boundaries.
Mapped results and complete K/V buffers match isolated GPU calls bit-for-bit.
An initial CPU comparison encountered one BF16 rounding-threshold mismatch
(max abs .001953, relative L2 .000417); the mapping-specific gate therefore
compares exact GPU paths, retaining the existing independent CPU-reference gates.
All five CUDA groups pass; final attention-kernel memcheck reports zero errors
and racecheck zero hazards on Spark-2.

## Eight request slots

Deployment `930da22a79162f8f` supports eight independent flat-cache sessions,
scalar graphs for every slot, and compact batched graphs with one row per slot.
32K is the per-request context; aggregate admission capacity is 262144 tokens.
MTP remains rejected. Fixed 128-token merge-sort prompts yield these first-sweep
end-to-end aggregate rates: one request 19.20 tok/s (cold), two 30.84, four 41.21,
eight 45.05. The request-0 transcript is exactly identical across all four widths.
Raw results: `/tmp/dgpp-mimo-opt-20260922/concurrency8.json`. Four-to-eight growth
shows diminishing aggregate gain and a substantial latency tradeoff.

Alternating 8030-token secret-word prompts still return the right word: cold
11.38/11.37 seconds, cached 0.440/0.441 seconds with 8025 tokens restored.
The API acceptance probe exposed an aggregate-versus-per-request capacity bug:
`max_tokens=32768` on a ten-token prompt was accepted with an eight-slot pool.
A distinct `SchedulerEngine::max_request_tokens()` limit now protects chat and
legacy completions; its host regression test passes. The fix must be verified
in the next deployment before considering this candidate complete.

Attention scratch now tiles 128 query rows independently of projection/MoE
chunk size, reducing the 32K score/probability allocation from 3 GiB to .75 GiB.
Real dense and MoE outputs at 1027 tokens/chunk512 match the previous full-score
implementation bit-for-bit. Sequential layer execution workspace is also shared,
while RoPE frequencies stay layer-owned (global/sliding theta differ).

An isolated full-width real layer-1 decode Nsight profile on Spark-2 shows 58.9%
of GPU kernel time in BF16 projections, 23.0% expert gate/up, 11.9% expert down,
and 3.5% attention. Report: `/tmp/mimo-layer1-decode.nsys-rep`. This is a world-1
layer profile, not a whole-service TP2 latency decomposition.

## Budgeted prefill and bounded scratch

Deployment `52dd05cb25d6902d` enables 256-token busy / 512-token idle prefill
budgets, shared layer workspace and 128-row attention-score tiles. All 40 host
serving tests pass, and the live oversized-context request is correctly rejected
with HTTP 400. Reasoning, SSE usage, tool calls and queued arithmetic pass again.

During a 512-token red-black-tree response, an arriving 8030-token LAVENDER
prompt completes in 15.96 seconds; the existing response's largest content
interarrival gap is .590 seconds. Its total time is 38.72 seconds. Raw arrival
timestamps are in `fairness-probe.json`. This demonstrates interleaving, not an
isolated throughput improvement.

Cold, cache-disabled TTFT on this eight-slot build is .490/.490 seconds at 480,
2.260 at 2030 and 11.201 at 8030 tokens. The 256-token decode probe with prefix
caching enabled takes 11.810/11.828 seconds. Its cold/warm texts agree, but differ
from the initial cache-disabled control after285 characters. Cache-boundary
prefill cuts change GEMM row shapes; a cache-disabled follow-up is needed to
separate this numerical effect from shared workspace. Do not treat these as
identical-trajectory timings against the earlier single-slot control.

2048-token chunk validation covers4103 real-weight layer rows, including the
physical4096-slot ring wrap and a seven-token tail. Versus scalar GPU execution,
max per-row relative L2 is .000340815 dense / .001142244 MoE; peak-rel ative error
is .00497512 / .00930233. Both pass the existing1% L2 /5% peak gates. Raw outputs
and `chunk2048-layer-comparison.json` are under the local experiment directory.

## Larger chunks and final attention pass

At `eb163d08fa35ba84`, idle prefill chunks increase to 2048 while busy chunks
remain 256. Cold cache-disabled TTFT is .484/.484 seconds at 480 tokens, 1.705
at 2030 and 9.607 at 8030 (versus .490, 2.260 and 11.201 with 512-token chunks).
Cache-disabled decode takes 10.665/10.690 seconds with exactly the original
256-token text. This separates the prefix-cut numerical effect from shared
scratch changes. Four concurrent distinct-word requests retain independent
outputs across slot reuse, but all subsequent cache probes miss: with just five
snapshot slots, active rolling snapshots evict the reusable prefix entries.
Increase the prefix budget before the final cache/concurrency acceptance pass.

Parallelizing only the softmax maximum (not the denominator accumulation) passes
all five GPU groups and memcheck/racecheck with zero errors/hazards on Spark-1.
Isolated 128-query global attention improves from 12.59 to 11.17 ms; sliding
attention is .373 ms. The maximum is exact under reassociation.

The existing tensor-core dense projection alternative is not adopted. Two
alternating idle-Spark layer-profile comparisons show approximately .400 ms per
projection for scalar BF16 GEMV versus .411/.409 ms for tensor-core GEMV, with
expert-kernel times around .31/.16 ms unchanged. The initial overlapping-startup
profile is exploratory only. Reports `/tmp/mimo-projection-{scalar,mma}.nsys-rep`
on Spark-2 retain the last pair.

## 64K capacity and practical cache reuse

`81698b1f4e070f2b` runs 65536 tokens per request, eight sessions, 12 GiB of
prefix snapshots (11 slots), 256/2048 busy/idle prefill budgets. API acceptance
passes, including HTTP 400 at prompt 10 + max_tokens 65536. Four simultaneous
835-token distinct-word prompts each produce 121 tokens without cross-talk;
reversing request order restores 830 tokens for every request and exactly
reproduces every cold output. Raw: `isolation64.json`.

Cold TTFT is .480/.480 seconds at 480, 1.639 at 2030 and 8.960 at 8030 tokens.
The 65030-token prompt returns ORCHID correctly: 271.543 seconds TTFT,
272.651 total; repeat restores 65025 tokens, .385 seconds TTFT, 1.492 total.
Raw: `context64-long.json`. Long cold prefill remains a bottleneck, so this is
not the optimization endpoint. Long-context decode is also slower (about 371 ms
per decode pass for this four-token answer).

## Parallel prefill normalization

A parallel FP32 denominator initially differed from the serial-normalization
control by relative L2 .004592 on a near-cancellation 64K fixture, despite max
absolute error only .000004 and no absolute/relative mismatches. Rather than
silently widening the established CPU gates, a separate long-context test now
checks an independent FP64 denominator (using the actual BF16-rounded scores
and exponentials) for one full 128-dimensional query/head and bounds all rows'
absolute/relative differences. Against that independent reference, serial output
has squared error 7.61323e-10; parallel output has zero error. The five existing
GPU groups retain their original gates. All six groups pass; real MoE-layer
4103-row error remains .001142244 maximum row relative L2.

Same-process production-shape isolated comparison (128 queries): at 8192 context,
serial versus parallel denominator is 11.575 versus 10.236 ms; at 65536 it is 115.875
versus 109.882 ms. Sliding attention is .383 versus .356ms. The bounded gain shows
that normalization is only part of the remaining long-prefill cost.

## Tensor-QK decode candidate and wide prefill PV

The first tensor-QK decode prototype reduces isolated 64K attention from 32.65 to
25.12ms. Splitting tensor-core PV over 512-key groups, then reducing FP32 partials,
removes the remaining long serial value chain: at 128 / 8192 / 65536 context, the
scalar score-reuse control takes .0389 / 2.2477 / 34.808 ms, and the tensor path
.0372 / .1208 / .9941 ms. This candidate targets only global decode layers;
sliding decode retains the established scalar path. Graph kernels map explicit
request planes and handle padding without touching cached histories.

All seven GPU groups pass, including mixed short/8K/64K mapped slots and graph
padding, under the same tensor-core absolute/relative and 0.2% aggregate gates.
Decode scratch now includes BF16 probabilities; dead FP32 score storage is
reused for the smaller split-PV partials.

A four-warp prefill PV tile shares probabilities across all 128 value columns,
retaining the ordered MMA chain. It matches the narrow tile bit-for-bit across
global/sliding, partial chunks, wraps and reset. In the same-process comparison,
global attention at 8192 context is 10.261 ms narrow vs 7.893 ms wide, and at 65536
109.803 ms vs 89.741 ms. These kernel results still require service validation.


### Rejected candidates and decode numerical correction

Changing prefill QK launch order to query/head first worsened the 8K wide-PV
kernel from 7.893 to 10.023 ms and 64K from 89.741 to 94.73 ms. Reverted.

The tensor-QK decode candidate passed all seven GPU groups, memcheck and
racecheck, but failed the real layer-5 gate at two of 4,103 rows: maximum
per-row relative L2 0.0804824 and peak-relative error 0.15. Dense layer 0 passed
at 0.000323. At row 644, attention relative L2 was only 0.0001495, but the
MoE output relative L2 was 0.0334553. Restoring scalar score/normalization
order while retaining split PV made that row's final output identical.
The tensor-QK candidate was not deployed. The replacement shares coalesced
Q/K loads through a tile but preserves the scalar ordered FMA and serial
normalization; split tensor-core PV remains. Acceptance thresholds are unchanged.

The ordered-score replacement passes layer 5 across 4,103 rows: maximum
relative L2 0.00226882 and peak-relative error 0.00649351. Isolated global
decode at 128 / 8192 / 65536 tokens is 0.0397 / 0.3338 / 2.5882 ms,
versus scalar 0.0389 / 2.2567 / 33.0836 ms. The faster rejected tensor-QK
numbers above are not the accepted implementation's performance.


## Accepted ordered-score decode and wide prefill PV deployment

Durable candidate: `experimental/mimo-split-attention-20260922`, namespace
`6bf2e70fd282a943`, SHA256
`ceac4a26a7cd6c1b750b04fd9aeef174a0b8cfcaa6c670e254c89fac43ac0dc7`.
Both ranks warm-captured scalar and 2/3/4/6/8-slot variants. Scalar graphs have
763 kernel nodes, including 97 collective nodes, with no copy, memset, host,
event or other nodes. Dense layer 0 maximum relative L2 is 0.000153186;
MoE layer 5 is 0.00226882. Both pass the unchanged 4,103-row gates. Seven
GPU groups pass under memcheck/racecheck with zero errors/hazards.

API reasoning, streaming usage, forced tools, queued arithmetic and per-request
budget rejection pass. Cold TTFT is 0.489/0.465 seconds for 480 tokens,
1.600 for 2,030 and 8.065 for 8,030. Fixed 256-token merge-sort requests take
10.470/10.465 seconds first-to-last content. Their text is repeatable but differs
from the earlier scalar-decode control after 192 characters; therefore the small
decode timing change is not a matched-trajectory speedup claim. Outputs remain
coherent; full-model equivalence is still an open gate.

A further prefill PV split changes isolated 8K attention from 8.125 to 7.742 ms
and 64K from 89.440 to 87.024 ms (approximately 5% and 3%). It is not adopted:
the small isolated gain does not yet justify another numerically reassociated
production path and its validation burden.


The same 65,030-token ORCHID request passes cold and cached: cold TTFT
226.258 seconds (previous scalar-decode/narrow-PV deployment 271.543), total
226.515. Repeating restores 65,025 tokens, with 0.357 seconds TTFT and 0.589
seconds total (previous total 1.492). The service reports 81 ms per decode pass,
versus about 371 ms previously. These are four-token answers, not a sustained
long-generation throughput benchmark. Raw: `split-long.json`.


A two-warp shared-query QK candidate preserved bitwise output in the chunk
fixtures but was slower. Alternating 64K probes gave 91.34 ms for the accepted
control, 191.70 ms for the candidate, then 90.84 ms for the control. Reverted.

The split-decode service passed four simultaneous distinct-word requests and
reverse-order cache restoration (830 cached tokens each, exact per-word output
match). The fairness probe completed both requests with a 0.510-second maximum
content gap on the existing decode. However its eight-request aggregate throughput
was 36.87 tokens/s, below the earlier 45.05 result; only request 0 retained the
same text across the two implementations. This prompted a bounded decode-score
grid: at most 128 tiles per K/V head/request, looping over remaining keys with
the same ordered arithmetic. Its full 4,103-row MoE output is bit-for-bit identical
to the preceding ordered-score candidate. Isolated 128/8K/64K timings are
0.0311/0.3317/2.5630 ms, versus 0.0397/0.3338/2.5882 ms for the unbounded grid.


Bounded-grid deployment `944fe143c83b5f0f` passed startup/capture and measured
37.14 aggregate tokens/s at eight requests, only a small gain over 36.87. The
earlier throughput difference is therefore not explained by launch count alone.

The existing Qwen policy of using cuBLASLt above four dense rows was separately
checked with 4,103 real-layer rows in chunks of eight. Dense layer 0 passed at
0.000340815 relative L2, but MoE layer 5 failed (0.261507 relative L2 and
0.259804 peak-relative error). It is not adopted; no threshold was relaxed.
The single-row MMA and multi-row Lt experiments are distinct rejected candidates.


The final adaptive decode candidate uses the original scalar arithmetic below
512 keys and the bounded coalesced-score/split-PV path from 512 keys onward.
The choice occurs per row on device, including mixed-length graph batches.
Short mapped rows must match scalar bit-for-bit in the GPU test; the test also
crosses positions 510/511 and exercises 64K histories and padding. Real layer 5
passes all 4,103 rows with maximum relative L2 0.00226882; its first 511 outputs
are bit-for-bit scalar. Isolated scalar/adaptive timings in ms are:

| Keys | Scalar | Adaptive |
|---:|---:|---:|
| 128 | 0.0372 | 0.0499 |
| 512 | 0.1413 | 0.0754 |
| 2048 | 0.5548 | 0.1249 |
| 8192 | 2.2079 | 0.3293 |
| 65536 | 33.1093 | 2.5489 |

The short-path launch overhead is explicit; its purpose is retaining the
established short-history arithmetic while accelerating the longer histories.
The rejected Lt layer-5 result was reproduced on Spark-1 and is bit-for-bit
identical to the Spark-2 candidate, excluding a cross-node mismatch in that gate.


## Snapshot bandwidth and packed sliding windows

The adaptive 64K service's cache metrics exposed 2,854 snapshot copies averaging
11.12 ms each. A flat snapshot copied 1,163,919,360 bytes even for a short
history. That is a material bandwidth cost on every rolling decode snapshot;
it invalidates treating the earlier launch-grid theory as the main diagnosis.
The proposed scalar-decode control redeployment was replaced by a direct
same-configuration snapshot optimization comparison.

Writes/restores now copy only committed global K/V. Sliding snapshots pack only
the live 128-token window, using up to two physical spans at ring wrap. The
shared session core optionally passes the snapshot position to the model's
restore method; existing families keep their two-argument method. Allocation
planning and runtime snapshot size both use the compact window geometry.

All 40 host serve tests pass. The GPU suite has eight passing groups, including
bitwise continuation after restoring into NaN-poisoned cache storage at positions
1, 127, 128, 255, 256, 257 and 383, followed by a 128-token causal chunk. The
new packed snapshot group passes memcheck and racecheck with zero errors or
hazards. The seven unchanged attention groups retain their earlier full sanitizer
passes. Deployment candidate: `experimental/mimo-packed-cache-20260922`, SHA256
`15b613c0e119a3a2dffbc56a2cc7b9133402fd6aa94ec622a8739d5a90ec5e1a`.


Packed-cache deployment `525e3c8f30badf36` runs the same 64K/8-request/12-GiB
settings on both ranks (digest `e7bc8a18a220b970`) with matching binary hashes.
It reports 16 slots of 767,754,240 bytes (732.2 MiB), versus 11 slots of
1,163,919,360 bytes. Short-workload snapshots average 0.565 ms versus 11.12 ms
for full-plane copies. The 12 GiB budget remains unchanged.

| Concurrent requests | Full-plane snapshots, tokens/s | Packed/live snapshots, tokens/s |
|---:|---:|---:|
| 1 | 18.32 | 22.78 |
| 2 | 26.28 | 37.69 |
| 4 | 32.76 | 51.28 |
| 8 | 36.83 | 56.67 |

Every generated response matches its corresponding pre-snapshot-change response
at every width. Cache-disabled 256-token responses also remain identical, with
10.545/10.575-second decode times. Cold prefill is 0.480 seconds at 480 tokens,
1.570 at 2,030 and 8.044 at 8,030. Four-to-eight concurrency adds about 10.5%
aggregate throughput with substantially higher per-request latency.

Eight simultaneous distinct-word requests preserve isolation and exact repeated
outputs. Four of eight exact-prompt repeats restore 828/830 tokens; the others
miss due to cache churn, so 16 slots do not guarantee retention of every prior
cut. Four 4,135-token requests then all restore 4,130 tokens in reverse order,
reproducing their cold outputs. This exercises real-model ring wrap at the
4,096-entry physical boundary. Disconnect/reuse succeeds with no engine failure.


At 12 GiB, the 65,030-token request passes again: 228.005 seconds cold TTFT,
228.233 total, then 65,025 cached tokens with 0.354 seconds TTFT and 0.579 total.
The warm service reports 76 ms per decode pass. The busy-prefill probe's existing
decode has a 0.494-second maximum content gap; the new 8K request finishes in
14.192 seconds. No failed requests or engine failures occurred; the intended
client-disconnect probe increments cancellation once and the next request works.

The eight-conversation next-turn test returns every correct unique word, and six
requests restore their complete prior conversation (913–955 cached tokens).
Two useful close snapshots were evicted before admission, so the strict eight-hit
gate fails at 12 GiB. Raw: `packed-multiturn8.json`. A 16 GiB cache-only candidate
is being checked for actual memory headroom and retention; it uses the identical
packed-cache binary and leaves context, concurrency and inference arithmetic alone.


## Final cache capacity and stopping point

Cache-only deployment `9c96470ef91f8327` uses 16 GiB and 22 snapshot slots.
Both ranks start and warm-capture successfully. Node available memory is 5.85
GiB on Spark-1 and 6.14 GiB on Spark-2; the same binary SHA256 above is retained.
The exact context/concurrency limits stay 65,536 tokens per request and eight
requests. Eight concurrent next-turn conversations all restore their full prior
conversation (913–955 tokens), returning the correct unique word. This passes
the strict eight-hit gate that failed at 12 GiB. Raw: `cache16-multiturn8.json`.

The 1/2/4/8-request sweep is 22.87/37.70/52.02/56.27 aggregate tokens/s. Cold
TTFT is 0.487/0.485 seconds for 480 tokens, 1.597 for 2,030 and 8.058 for 8,030.
The larger cache adds retention capacity, not a demonstrated compute speedup.

The remaining measured local candidates give small gains, regress, or fail the
unchanged numerical gates: split prefill PV adds only 3–5% at kernel level;
shared-query/launch-order alternatives are slower; tensor-QK decode and larger
Lt projection batches fail real MoE gates. Four-to-eight concurrency adds only
about 8% aggregate throughput on the final configuration. Snapshot copy overhead
on short workloads is now below a millisecond. The older 4,096-row chunk experiment
is not promoted: its expanded rings alone add 3.047 GiB per rank, exceeding spare
memory above the four-GiB headroom target in this final configuration, before
additional projection/MoE workspace. No final-head 4K-chunk speed claim is made.

Further substantial work is separate from these local optimizations: full-model
HF logits/NLL equivalence, MTP/DFlash, and a dynamically sized/paged global prefix
cache. MTP and multimodal execution remain unavailable. There is no claim of full
Qwen feature parity, full-model numerical equivalence, or four-node validation.

## Final operational verification

Final 16 GiB deployment's 65,030-token probe returns ORCHID in both trials.
Cold TTFT is 227.622 seconds (227.852 total); the repeat restores 65,025 tokens
and gives 0.315 seconds TTFT (0.544 total). This is a four-output-token probe,
not a sustained long-context generation benchmark. Raw: `cache16-long.json`.

After acceptance, health reports MiMo OK; metrics show zero active/queued
requests, zero failed requests, and `engine_failed=false`. Both rank logs have
no ERROR/FATAL entries and agree on configuration digest `eb4693ebb89cf1e4`.
Both deployed binary hashes match
`15b613c0e119a3a2dffbc56a2cc7b9133402fd6aa94ec622a8739d5a90ec5e1a`.
Rank PIDs are 3016792 / 1819493, namespace `9c96470ef91f8327`.
Near the end of cold long prefill, available node memory remains 5,607 / 5,905
MiB. The final metrics are `cache16-final-metrics.json` in the evidence root
`/tmp/dgpp-mimo-opt-20260922/`. Working changes remain uncommitted.

Restart command on Spark-1 (use the maintenance lock before a future restart):

```bash
DGPP_ENV_FILE=/home/jon/dgpp/releases/dgpp-0.1.0+gf1b05f544293.dirty/.env \
python3 /home/jon/dgpp/releases/dgpp-0.1.0+g116d3c375d87/scripts/dgpp-cluster up \
  --config /home/jon/dgpp/experimental/mimo-cache16-20260922/deployment.json \
  --bin /home/jon/dgpp/experimental/mimo-packed-cache-20260922/dgpp-serve
```

Use the same configuration and binary with `down` for a coordinated stop.
