# Adaptive MoE expert residency

## Recommended NVFP4 deployment

Use the [two-node NVFP4 streaming template](../deploy/cluster_deepseek-v4.1-flash-1017_nvfp4-streaming_w2.example.json)
for aggregate throughput. For single-request speculation, apply the C1 settings
below to that template. These are separate serving recipes, selected from the
measured workloads; neither is a universal optimum.

| Setting | C16 aggregate throughput | C1 speculation |
| --- | --- | --- |
| Expert backend | Canonical; `DGPP_MOE_B12X` unset | b12x W4A16; `DGPP_MOE_B12X=1` on both ranks |
| `engine.mtp` | `false` | `true` |
| `engine.mtp_depth` | Unused | `3` |
| `engine.mtp_schedule` | Unused | `false` |
| `engine.expert_streaming.execution_slots` | `1` | `24` in the latest miss-path comparisons |

The recorded benchmark recipes use `engine.bf16_weights: "bf12+bf16"`, grouped
prefill, an 8,192-token KV capacity, disabled prefix caching, and a 100 GiB
expert budget per rank. Each rank has
16 I/O buffers, four read workers and queue depth 16. Keep `decode_graph`,
`prefill_fold_overlap` and `l2_prefetch` disabled as in the template. Select
either residency policy through `engine.expert_streaming.policy`; use
`minimum_fraction: 0` to let fractional caching fully evict cold experts.

I/O workers are distinct from reconstruction slots. Fixed depth three verifies
four target rows with six routed experts per row: at most 24 distinct target
experts **per layer**, not six layers. The 24 slots reserve reconstruction and
prepared storage inside the expert budget, reducing persistent cache capacity.

For conversational and tool serving, override the short-prompt benchmark
settings. The two-Spark interactive deployment uses these engine settings:

```json
{
  "kv_capacity": 1048576,
  "prefix_cache_gib": 1.0,
  "prefill_initial_tokens": 4096,
  "prefill_idle_budget_tokens": 4096,
  "prefill_idle_floor_tokens": 4096
}
```

The token pool is shared by active requests and retained prefixes. Prefix
caching lets a tool-result continuation reuse the preceding prompt instead of
processing the full conversation again. Larger idle chunks reduce repeated
model walks when loading a long prompt. The prefix arena uses separate memory
from the unchanged expert budget; startup accounts for both. These overrides
do not change the configurations or measurements reported below.

The matched V46 C16 comparison measured 102.90/103.21 aggregate decode tok/s
for whole-expert/fractional caching, with identical responses and zero warm
expert reads. The V46 C1 depth comparison favored depth two for the sampled
prose prompt and depth three for code; depth three minimized combined decode
time. Its BF12-head follow-up measured 30.07/45.30 prose/code decode tok/s with
the same outputs and acceptance as its BF16-head control. These are recorded
build measurements, not a new campaign on the retained V64 implementation.
See the [fixed-depth comparison](#warm-fixed-depth-serving-comparison) and
[C16 results](#c16-throughput-and-read-window) for their scopes.

V64 retains ready-chunk copy batching, heap-based demand admission and indexed
native preparation. Its final warm C1 code request measured 45.01 decode tok/s.
The subsequent separate-transfer-stream prototype did not improve the intended
miss path and was removed. Independent expert readiness remains prospective;
the selected path reconstructs experts in order on the execution stream.
See [implementation status](#scope-and-implementation-status) for those results.
Physical DRAM bandwidth remains unmeasured; payload/time estimates are separate.

## Earlier serving measurements

The following measurements describe the execution work leading to the recipes
above. Settings and recommendations in those measurements apply to their stated
builds; use the deployment guidance above for current settings.

The NVFP4 implementation includes an optional b12x W4A16 backend with
BF16 activations, shared-expert overlap and expert-graph replay. Shared loader
and allocation-planning fixes cut fractional startup from 223 to 108 seconds
and raised its warm C1 prose/code decode from 21.53/36.58 to 25.33/41.44 tok/s.
Both policies converge to full native hits on those repeated prompts with no
expert reads. See [the current backend comparison](#prepared-raw-scale-w4a16-integration)
for the matched controls and the effect of changed speculative acceptance.

Execution speed is the current priority. The longer NVFP4 policy campaign was
stopped to optimize the shared engine. Warm actual-serving measurements with
native cached experts reached 25.00 decode tok/s at C1 and 100.48 aggregate
at C16 after expert-only graph replay, fused routing and disabling L2 weight
prefetch. An earlier build measured 77.54 at C8 on the same prose workload
(64 output tokens/request). Including prefill and request wall time, those
rates were 23.33, 78.12 and 63.51 tok/s respectively. These are third presentations on one running server,
not cold-start or general workload rates.

Those measurements use ordinary autoregressive decode. Eager DSpark support
now batches target verification and the distributed row pick, restores every
request's target state before redrafting, and tracks acceptance per position.
It accelerates eligible greedy requests; other sampling recipes retain ordinary
decoding. NVFP4 main experts remain paged while the smaller MXFP4 draft expert
population stays resident, charged against the same total expert budget.
Both-format/both-policy fixtures match ordinary greedy transcripts through
full acceptance, rejection, rollback and slot reuse. Fixed-depth five-position
DSpark measured 26.51 decode and 24.70 wall tok/s on the third C1 presentation,
versus 25.00 and 23.33 with ordinary decode. The warm output matches the
ordinary greedy response. The first two DSpark presentations reached only
0.79 and 14.77 decode tok/s; the warm gain is about 6%, not a cold-start claim.
Confidence scheduling reads each acceptance score after its chain head produces
it, shortens verification and omits later draft-chain calls. Both ranks adapt
from committed counts and a shared modeled cost rather than rank-local wall
time. A focused fixture covers scheduled shortening, transcript parity and
poisoned future confidence rows. Scheduled DSpark measured 32.97 decode and
29.96 wall tok/s on the third C1 presentation, about 24% above fixed-depth
DSpark and 32% above ordinary decode. Prompt hash, response text, output tokens
and finish reason match the fixed-depth response. Its first presentation took
111.56 seconds (1.08 decode tok/s), and its second reached 17.22 decode tok/s.
A follow-up with the split shared-expert graph measured 34.38 decode and
31.06 wall tok/s at C1, with the same response, about 4% above the preceding
short run. These are short serving measurements. C16 completed one presentation at
11.46 decode tok/s before the next presentation failed because an eager draft
had not retired the stream-ordered transport folds before a host pick. The
draft now settles that boundary, with a focused regression assertion. The
corrected serial-draft build completed C16: its third presentation reached only
16.60 decode and 12.69 wall tok/s, versus 100.31 and 89.80 with ordinary grouped
prefill/decode. All 16 warm responses match the ordinary build by prompt hash,
text, output tokens and finish reason. Thus fixed serial drafting is unsuitable
for C16, despite its C1 gain.

The eager DSpark adapter now batches the draft layers and Markov picks as well
as target verification. Variable accepted lengths use padded rows within the
existing decode-row capacity; each request retains its own confidence cutoff.
Fixtures for both formats/policies match ordinary greedy through full acceptance,
rejection, alternating request order and a transition to scalar drafting.
Batched drafting completed C16: third-presentation decode/wall rates were
17.28/12.80 tok/s, versus 16.60/12.69 with serial drafting. All 16 warm outputs
still match ordinary decoding. This does not approach the ordinary build's
100.31/89.80 tok/s. Keep MTP disabled for the throughput-oriented C16 recipe.
The later fixed-depth comparison supersedes this early scheduled-C1 guidance;
use fixed depth three with confidence scheduling disabled as the C1 starting point.
The two serving configurations have different resident draft allocations, both
charged within the same configured total expert budget.

A longer, fixed-depth diagnostic used 256 output tokens on one prose prompt
and one code prompt, each repeated. Accepted predictions exclude the normal
target token: the warm prose request averaged 1.19 accepted predictions per
pass, and code averaged 2.46. The corresponding total committed counts were
2.18 and 3.45 tokens per pass, with decode rates of 21.31 and 35.13 tok/s.
Opt-in host phase timing measured about 8.12/8.26 ms drafting and
93.90/89.58 ms target verification per pass for prose/code; target picking
and state handling together took about 0.25 ms. These disjoint wall phases
include device waits and are not GPU kernel-time categories. Instrumented
rates are diagnostic observations. The preceding 34.38 tok/s scheduled result
used a shorter 64-token prose response and averaged only 1.10 accepted
predictions (2.10 total committed tokens) per pass. Neither result establishes
acceptance across prose or code generally. Fixed-depth verification also
executes rows after the first rejected prediction. Confidence scheduling is
deferred while the shared execution path is optimized.

With confidence scheduling on those same longer prompts, warm decode measured
25.67 tok/s for prose and 37.84 for code. Accepted predictions fell to 0.90
and 2.10 per pass; mean verify rows fell from six to 2.49 and 3.81. Prose
verification took 66.61 ms and code 73.23 ms, with drafting 7.32/7.75 ms.
Rank-zero reads were zero for the repeated prose request and 47.81 MiB for
code. These are head-rank counters and cannot establish zero peer reads.
The four response texts, prompt hashes, token counts and finish reasons match
the fixed-depth run. Confidence scheduling helps these samples, but shared
expert execution speed is the current optimization priority.

Repeated prompts are not sufficient evidence that a mixed workload fits the
cache. After the fixed-depth prose/code pair had each been repeated, switching
back to prose caused 20.42 GiB of rank-zero expert-image reads and switching
again to code caused 18.76 GiB. Decode fell to 5.31/6.06 tok/s respectively.
Those requests exercise changing locality, rather than a no-I/O warm path.

The server supports up to 64 plain expert-streaming request slots, but 16
remains the recommended setting here. A follow-up at C32 reached 13.69 decode
tok/s (6.71 including prefill) on the first presentation and 25.22 (18.07 wall)
on the second. The run was stopped before completing the third presentation;
C64 was not measured. Raising concurrency did not improve these serving rates.

The eager adapter used by expert streaming now exposes DeepSeek's existing
grouped-prefill implementation. Queued cold prompts can share one model walk,
amortizing weight loading and launches across their token rows. This honors
`engine.prefill_group`; disabling it retains one prompt per walk. A focused
fixture checks first-token and subsequent decode transcripts against isolated
prefills in both exact and bounded modes. Grouped prefill measured 100.31 decode
and 89.80 wall tok/s on the third C16 presentation, compared with 100.48 and
78.12 previously. Warm prefill time fell from 3.04 to 1.31 seconds, and all
16 warm responses match the preceding build for the same prompt hashes.
Grouping from an empty cache took 240.24 seconds on the first presentation
(119.02 seconds in prefill). The preceding C16 run had prior C1 warmup, so its
first presentation is not a matched cold baseline. A follow-up primed the first
request wave separately before enabling grouping. It took 255.07 seconds cold,
with 129.13 seconds in prefill; warm decode/wall rates were 99.47/89.12 tok/s.
That startup heuristic was removed because it did not improve serving speed.

Demand loading keeps a bounded window of missing chunks in flight while
current chunks are copied to the GPU. The window uses the existing configured
staging pool, reserving space for the current device reader and outstanding
speculative tickets. With the default two buffers this remains one chunk ahead;
more buffers and workers allow concurrent reads. Asynchronous read errors are
preserved. Next-expert speculation starts after the current acquire. Native
cache hits retain their direct execution path. The later
[C16 comparison](#c16-throughput-and-read-window) measures the wider read window.

A paired-row FP4 gate/up candidate was removed after serving measurements.
Its six-row GPU diagnostic was 3.6% faster for NVFP4 and 7.2% for MXFP4, but
full-model fixed-depth speculation stayed at 21.25/35.15 decode tok/s for
prose/code versus 21.31/35.13 previously. Ordinary C16 reached 100.56 versus
102.24 tok/s on the final warm presentation. Outputs matched; these short
observations did not justify retaining the extra kernel.

This setup uses 16 active slots, 100 GiB of expert capacity per rank and an
8,192-token KV capacity. The earlier two-slot/96 GiB/1,048,576-token setup
measured about 15.31 decode tok/s at C1. Gains combine execution and
configuration changes. The common native-hit path avoids reconstruction,
retires readers using CUDA stream ordering, keeps per-layer GPU binding tables
and reuses fused resident decode kernels. Native decode also skips GPU segment
construction and its metadata readback. For nonlegacy streams, up to 16 rows
and an FP8 shared expert, the native expert chain replays through a cached
CUDA Graph. The current candidate includes the shared expert's side-stream
fork and join in that graph and allows its separate FP8 kernels at one row,
keeping the routed FP4 kernels free of the mixed shared branch's register cost.
Bitwise MoE fixtures passed this capture/replay path for both formats/policies,
changed routes and cache rebinding. C1 scheduled DSpark measured 34.38 decode
and 31.06 wall tok/s. Ordinary C16 measured 102.24 decode and 91.30 wall
tok/s on the third presentation, versus 100.31/89.80 previously; all 16 warm
responses match. These small differences are short-run observations, not a
claim of a stable percentage gain across workloads. Graphs are specific to the layer, row count,
input/output pointers and shared-weight binding; route IDs, routing weights and
the persistent expert table remain live device inputs. Other recipes retain
ordinary kernel launches. The focused bitwise test covers both routed formats,
both policies, cache-instance rebinding and input changes during replay. All
three measured graph-serving responses match the earlier NVFP4 output.

Segment removal alone measured 23.78 decode tok/s at C1 and 96.93 at C16,
essentially unchanged from the preceding 23.50/97.92 run. The graph path added
about 3% at C1 in this short measurement. Fused routing and disabling L2
prefetch added smaller gains. Paging still requires one host routing decision
fence per MoE layer; whole-model decode graphs remain disabled.

A warm three-request GPU trace showed kernels occupying 7.88 seconds of an
8.53-second GPU interval (92.4%). Prefetch kernels occupied 2.91 seconds but
overlapped other kernels for most of that time: only 0.28 seconds were exclusive
to prefetch. Summed kernel durations are not a latency breakdown. Most CPU
synchronization time waits on device work, so it does not establish CPU dispatch
as the main remaining bottleneck.

The hot native NVFP4 gate/up and down kernels launch 256 threads (eight warps),
and the dominant dense MMA variants launch 128 or 256. The compiled gate/up
uses 93 registers per thread and down uses 121. With the GB10's 65,536-register
SM budget, either native kernel can fit at most two 256-thread blocks, or
16 of its 48 warp slots (33% nominal occupancy), even before other limits.
Block size alone therefore does not establish full occupancy. Reducing register
pressure must preserve useful memory throughput and avoid local-memory spills;
occupancy by itself is not the throughput objective.

Both streaming examples now use 16 active slots, a 100 GiB expert budget and
8,192 aggregate KV tokens, with L2 weight prefetch disabled and a default output
limit of 4,096 tokens. These are throughput-oriented settings for the two-Spark
setup; increase KV capacity for longer aggregate contexts. The measured NVFP4
prefetch-off gains were small, so they do not establish a universal recommendation
for other models or hardware.

The MXFP4 two-node comparison is complete: both strategies serve the full
model from local SSD within the same 96 GiB per-rank expert budget. All 280
serving responses match across A/B/B/A boots. Whole-expert caching is
3.2–7.0% faster on repeated decode cells, while fractional caching reduces
post-startup reads but incurs more initialization work and total reads.
With a zero floor, fractional residency converges to a complete native hot
set on the repeated warm prompt. See the
[completed comparison](#completed-local-ssd-full-model-comparison-2026-11-14)
for latency, throughput, I/O, memory and interpretation limits.

The earlier NVFP4 build passed the full-model TP2 numerical
comparison: both policies match the streaming control byte for byte across
40 main layers and both vocabulary shards on seven seeded tokens, with zero
swap. The earlier NVFP4 policy comparison was stopped after three serving boots
to prioritize execution optimization; numerical parity alone does not
establish serving speed. In both NVFP4 fractional
serving boots, the third identical warm window records 12,390/12,390 native
and full-expert hits on each rank, with zero expert-image reads,
reconstruction time and native-layout copy bytes. This proves convergence
for that fixed prompt; the earlier four-boot policy comparison was not completed.
Physical DRAM bandwidth remains unknown. The held locality replay has eight
complete A/B/B/A groups covering high, low and changing locality; three
64-row changing-locality legs remain unrun. These limits are retained rather
than triggering additional campaigns automatically.

## Warm six-row verification profile

A short Nsight capture on both ranks used the baseline fixed-depth-five
DSpark deployment: six target rows, an exact draft head, 100 GiB expert budget
per rank and the original two-buffer I/O settings. Profiling began during
decode after two presentations of the same 256-token prose request. Both
ranks recorded zero expert-image bytes and zero expert-image read calls across
the captured request. This is a hot expert execution observation, not an SSD
performance measurement.

| Timeline measure | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| GPU activity coverage, including copies | 92.4% | 93.1% |
| Main routed FP4 kernel coverage | 49.5% | 49.7% |
| Dense kernel coverage | 26.1% | 26.0% |
| GPU communication kernel coverage | 5.2% | 5.6% |
| GPU idle gaps | 7.6% | 6.9% |

Kernel-category coverage can overlap; these rows must not be added together.
Shared-expert work mostly overlaps routed-expert execution. Rank 0's main
routed gate/up and down totals were 3.139 and 1.554 seconds over 4,280 calls
of each kernel, about 0.733/0.363 ms per layer. Its 9.471-second GPU span
covered 107 complete sets of 40 main expert layers. Rank 1's capture begins
slightly later and has a 9.161-second span; raw totals are not a matched
rank-speed comparison. CPU CUDA synchronization totals overlap GPU execution
and are not additional wall-time overhead. GPU idle gaps alone do not prove
CPU coordination caused them.

This capture supports specialized small-batch FP4 expert execution as the next
optimization priority; a generic grouped GEMM is one candidate. It does not
establish a DSpark correctness defect, universal workload
acceptance, or that eliminating host paging decisions would remove the dominant
cost. The bounded SSD-window candidate remains separate from this baseline
profile and has no serving performance claim.

The grouped W4A16 decode prototype uses the existing 16-row tensor-core
kernel for routed experts while retaining the concurrent FP8 shared-expert
lane and expert-only graph replay. Paging already downloads routes for cache
admission, so the prototype uses them to upload compact expert segments,
padded to a fixed token-dependent bound for graph replay. It keeps BF16
activations and uses the same expert budget for both residency policies.
`DGPP_MOE_FP4_GROUPED_DECODE=1` opts into this prototype; it is disabled when
W4A4 is enabled.

The first prototype selected MMA only for native hits and retained GEMV for
miss waves. Their different summation and rounding changed continuations as
the cache warmed, making those repeated requests unsuitable as a hot-path
comparison. The revised prototype uses the same W4A16 routed and FP8 shared
arithmetic for both paths. Both-format/both-policy fixtures check miss waves,
native hits and changed-route graph replay against resident grouped execution
and the existing double-precision oracle budget. These kernels are not bitwise
the former GEMV recipe.

The NVFP4 checkpoint selects six target experts per token and three draft
experts. Six-row verification therefore has 36 routed assignments. Earlier
top-8 microbenchmarks used the wrong routing geometry; serving used the
checkpoint's correct routing configuration. Corrected top-6 synthetic chains
including the shared expert measured 539 versus 693 us for grouped versus
slot execution with eight experts, and 1,043 versus 977 us with 32 experts.
Resident grouped microbenchmarks include empty expert segments, whereas
paged serving compacts selected experts, so larger-pool microbenchmarks do
not reproduce the serving launch geometry.

The revised prototype's warm full-model prose result was 24.88 decode tok/s
and 80.68 ms per target pass, versus a fresh slot-path baseline of 25.20 tok/s
and 78.10 ms. It demonstrated no prose improvement. Code reached 39.08 tok/s
and 83.30 ms per target pass, but lacks a contemporary matching baseline.
Both workloads had stable repeated continuations and zero expert reads on
both ranks during the measured warm presentations. The prototype remains
disabled by default. Increasing grouped-kernel staging depth did not show a
useful general improvement and was removed.

### SM121 small-batch backend comparison

A routed-expert-only comparison used FlashInfer 0.7.4's bundled b12x on a
48-SM GB10, BF16 activations, NVFP4 weights, hidden size 5,120 and a TP2
intermediate shard of 1,152. All 384 expert IDs were available. Each call had
six rows and top-6 routing. Synthetic routing covered 0/20/40/60/80 percent
requested reuse; the realized unique counts are below. Preparation, weight
packing and JIT compilation preceded CUDA graph capture; graph timing excludes
Python launch overhead. It includes each backend's routed execution and
reduction, but excludes shared experts, routing logits, paging and TP traffic.

| Unique experts / 36 assignments | DGPP slot GEMV | b12x packed W4A16 |
| --- | ---: | ---: |
| 36 | 1.489 ms | 1.534 ms |
| 29 | 1.247 ms | 1.242 ms |
| 22 | 1.034 ms | 0.945 ms |
| 14 | 0.781 ms | 0.606 ms |
| 7 | 0.585 ms | 0.317 ms |

The existing grouped prototype was slower than slot GEMV at 36/29/22 unique
experts, even with compact routes prepared outside timing. These measurements
used the subsequently abandoned deeper-stage kernel and do not establish the
original grouped implementation's serving speed. b12x's source-native A16
promotion path was slower throughout this comparison, at 2.32–2.72 ms.
Its mixed-precision weight plan forced A16 for all six-row calls; the recorded
execution contract confirmed BF16 activations and FP32 accumulation.

FlashInfer's separate native NVFP4 W4A16 implementation took 0.945 ms at 22
unique experts and 0.436–1.546 ms across the five reuse levels. That unmodified
API lacks the checkpoint's SwiGLU clamp and uses packed FP16 local products;
its timing is an implementation reference, not a numerically compatible
serving replacement. b12x's packed and source-native calls used clamp 10.
Their synthetic output relative RMSE versus the DGPP-rounding reference was
0.37% and 0.40%, respectively; these checks establish layout/scale plumbing,
not full-model numerical or quality qualification.

The result favors reuse-sensitive small-batch dispatch rather than a universal
grouped-kernel replacement. Any integration must retain the concurrent
shared-expert lane and graph replay, and execute the same arithmetic on misses
and resident hits.

A subsequent head-rank capture observed the existing paging D2H copies, adding
no CUDA operations or synchronization. Two 64-token continuations on the slot
backend produced 1,080 prose and 560 code six-row target-layer calls. Mean
unique counts were 21.49 and 21.90 of 36 assignments: 40.3% and 39.2% reuse.
These short continuations establish this deployment's routing geometry, not
universal workload statistics or warm serving throughput.

Replaying route tables selected at the 25th/50th/75th percentiles of unique
expert count, with the same synthetic weights and activations across backends,
gave 6–12% lower routed-chain latency for packed b12x. The median-count prose
case took 0.902 versus 0.996 ms; code took 0.939 versus 1.040 ms. The grouped
staging prototype remained slower than slot GEMV in all six cases. Preparation
and graph capture stayed outside timing. This supports packed small-batch
execution as an integration target; it is not an integrated full-model serving
result. The external backend is not yet connected to DGPP's paged expert
pointer tables, load-time conversion or shared-expert lane.

Enabling b12x's deterministic output mode retained the same latency advantage:
the median-count prose/code route tables took 0.901/0.942 ms. Ten repeated
executions produced identical output. The synthetic relative RMSE remained
0.368%, so selecting deterministic reduction did not resolve the arithmetic
difference from the DGPP reference. This remains a kernel-level observation,
not full-model numerical qualification.

An isolated adapter subsequently changed the packed kernel to resolve payload,
scale and global-scale pointers separately for each expert. It retained the
latency advantage without gathering weights into a contiguous expert array.
Changing the expert pointer tables and corresponding route IDs in place under
the same captured graph preserved output bits. CuTe's C export produced native
objects; a C++ launcher using host-prepared route blocks matched the Python
b12x invocation bit for bit on all six route tables. Its median-count prose/code
latencies were 0.892/0.932 ms. Kernel execution requires no Python runtime.
The adapter still uses b12x-prepared packed weights; conversion of DGPP's cache
allocations and integration into serving remain unfinished.

Adding DGPP's concurrent FP8 shared-expert lane narrowed the advantage.
The median prose/code routes took 1.033/1.084 ms with the C++ b12x adapter,
versus 1.100/1.103 ms with slot execution. This measures both lanes through
their join, excluding final output addition, routing, paging and TP traffic.
It demonstrates why isolated routed-chain gains cannot be used as serving
speedups or assumed to survive shared-expert scheduling unchanged.

Using 32 cooperative blocks instead of 48 improved the combined-lane times
to 0.975/1.021 ms for the two median routes, versus contemporary slot times
of 1.100/1.094 ms, about 11%/7% lower latency. The C++ output remained bitwise
equal to the 48-block b12x invocation for both tables. This is a retained
small-batch prototype configuration; it has not been measured in full-model
serving or across other batch sizes. The experiment includes CUDA graph replay
and the shared-lane join.

The slower source-native result also depends on dispatch. Disabling b12x's
native small-M direct candidate and using its native-layout tensor-core path
reduced median-count prose/code latency to 0.980/1.025 ms. That path was close
to DGPP's slot kernel and slower than the packed path; the earlier 2.32–2.72 ms
range must not be interpreted as a lower bound for native-layout execution.

A separate existing-kernel experiment issued all nine FP4 down-projection
chunks together at the TP2 width of 1,152, replacing three passes of 4/4/1.
It preserved output bits but slowed the complete routed chain by approximately
8–12% across the six captured route tables. The production cutoff was left
unchanged.

### Prepared raw-scale W4A16 integration

The optional `DGPP_MOE_B12X=1` path prepares NVFP4 full experts inside the
existing cache slab. It preserves the original E4M3 scale bytes and separate
gate/up inverse globals. Partial chunks remain in canonical storage format;
demotion reverses the prepared layout without re-quantization. Each execution
slot reserves a second full-expert region for preparing a reconstructed miss,
deducted from the configured expert budget before allocating cache capacity.

The specialization uses BF16 activations and unweighted FP32 down outputs.
DGPP's existing ordered FP32 reduction applies router weights and adds the
shared expert before the final BF16 rounding. Both complete hits and miss waves
use this same arithmetic. Mixed-hit layers execute their resident subset in
one batch before loading missing experts. These waves keep the complete
route-block scheduling domain and mask inactive experts. Missing experts now
share a grouped launch up to `execution_slots`, with independent reconstructed
and prepared weight regions kept alive through that launch. The first missing
expert's bounded file prefetch starts before shared/resident expert execution.
This overlaps file reads with useful resident work, while reconstruction and
CUDA copies remain owned by the engine thread. One slot retains single-expert
waves. Extra slots are deducted from the same total budget, reducing cache
capacity; they do not add unaccounted scratch or loader threads.

A short full-model comparison of single-expert and grouped reconstructed waves
used fixed depth three, the BF12 head, 24 execution slots, 16 staging buffers,
four I/O workers and a 100 GiB total expert budget per rank. Each deployment
served one cold C16 mixed wave of 32-token responses, followed by three
128-token C1 code requests. On the first code request after mixed traffic,
decode measured 15.33 versus 18.67 tok/s and prefill took 2.29 versus 1.80
seconds. Response text and speculative acceptance matched. Rank-zero exposed
file-ticket wait fell from 1.87 to 0.76 seconds, while expert reads increased
from 12.03 to 12.25 GiB. On the last code request, both ranks performed zero
expert reads; decode measured 44.25 versus 45.02 tok/s. The preceding warm
request measured 37.86 versus 37.59 tok/s. These short observations support
the miss-path change without establishing a warm execution improvement.

The mixed requests changed two of sixteen continuations and their acceptance
counts, so their rates do not establish an isolated scheduling speedup. All
three code responses and their acceptance counts match. Both deployments
recorded zero swap and matching operation streams between ranks, and ended
stopped. The focused paging fixture also preserves bitwise output across
single-expert waves, grouped waves, native hits and cache churn.

Independent miss readiness is not yet implemented. `enqueue_paged_sm12x`
acquires every member of a bounded wave synchronously before its grouped
launch. The file queue exposes independent `ReadTicket::ready()` results,
but `ExpertCache::acquire` consumes chunks in order and returns only after
complete reconstruction. Ready chunks now share a copy fence, but copies
still use the same CUDA stream as expert execution and drain before an
unfinished file-ticket wait.
Launching a ready group earlier therefore does not, by itself, allow its GPU
work to overlap reconstruction copies. A useful implementation must preserve
grouping while separating read completion, copy completion and execution-slot
retirement. File workers must continue to leave CUDA work to the engine.

Cache allocation already spans all layers: a common planner scores experts
by decayed routing frequency and estimated loading cost per byte, rather than
reserving equal capacity per layer. The remaining layer-value question is
whether those scores predict exposed serving stalls. The current load estimate
uses file-ticket elapsed time extrapolated to a full expert, including queueing
and potentially overlapped time; it is not exposed milliseconds saved per GiB.
Saved route IDs can estimate hit curves, but need matching layer-level stall
costs to establish that value. A low layer hit rate alone is insufficient
reason to transfer cache capacity.

Integrated GPUs now reconstruct each chunk with a single GPU scatter of its
payloads, scales and globals, including strided down-projection columns.
Registered host staging is mapped directly into the kernel; cache-to-scratch
copies use the same plan. Plans are built at startup and launches allocate no
payload scratch. Native demotion uses the reverse plan. This preserves both
FP4 formats and the exact packed weight bytes. Discrete GPUs retain pitched
copies; `DGPP_EXPERT_GPU_SCATTER=0` selects that control on GB10.

In a same-binary serving diagnostic with the preceding fixed-depth-three
configuration, the first C1 code request after mixed traffic measured
18.60 versus 19.65 decode tok/s. Rank-zero reconstruction time excluding
file-ticket waits fell from 3.00 to 2.57 seconds; file reads were 12.23 versus
12.28 GiB. All three code responses and their acceptance counts matched.
The final zero-I/O warm request measured 45.29 versus 45.15 tok/s, so this
change has no demonstrated warm execution gain. Four of sixteen mixed
continuations differed; mixed rates are excluded as an isolated speedup
claim. The control's peer memory sampler did not start because its script
was missing, although the launcher enforced zero-swap scopes; the scatter
deployment sampled zero swap on both ranks. This is a diagnostic observation
with that control-observability limitation. Byte-exact GPU fixtures cover
mapped host/device sources, untouched padding, graph replay, both residency
policies and cache churn.

A follow-up replaced per-chunk stream fences with preallocated CUDA events
that held each staging lease until its copy completed. It preserved the
two-buffer demand reserve, passed the cache and bitwise paging fixtures, and
used the same GPU scatter and budget. In serving, code after the mixed wave
measured 19.23 versus 19.65 decode tok/s, with identical text and 87 accepted
predictions across 41 passes. Its final zero-I/O request measured 45.18 versus
45.15 tok/s. Both deployments sampled zero swap on both ranks and had matching
rank operation streams. This did not establish a serving benefit, so the event
retirement candidate was removed. Copies still shared the compute stream;
independent expert readiness remains unimplemented.

A second rejected prototype launched groups of at least four reconstructed
experts immediately before an unready file-ticket wait, and retired their
execution leases through CUDA stream ordering. A same-binary off/on comparison
used the same fixed-depth-three configuration and cold mixed/code sequence.
Code after mixed traffic measured 20.43 versus 19.33 decode tok/s; final warm
decode measured 45.16 versus 45.13 tok/s with zero expert reads on both ranks.
All three code responses matched, each with 87 accepted predictions across
41 passes. The first code request read 12.10 versus 12.56 GiB on rank zero,
so this does not isolate launch overhead from the preceding cache state.
Six of sixteen mixed continuations differed; their rates are not an isolated
scheduling comparison. Both deployments sampled zero swap on both ranks and
completed with matching rank operation streams. The prototype showed no
serving benefit and was removed. It still reconstructed experts in order on
the compute stream; this result does not test a fully independent read/copy
pipeline or rule out the SSD-native scheduling approach.

Warm execution captures the routed kernel,
concurrent shared FP8 lane and final accumulation in one expert graph.

The current specialization supports H5120, intermediate width 1152 per rank,
384 experts, top6 and up to 64 token rows. Short prefills execute prepared
experts directly; batches beyond decode capacity use the existing prefill
scratch. Larger prefills use the canonical fallback. It is an opt-in candidate;
full-model measurements are described below. Existing MXFP4 execution remains available.

An initial six-row/top6 check with actual median-count prose/code routes and
synthetic weights and nonuniform scales measured routed-chain latency of 0.879/0.925 ms versus
1.008/1.042 ms for slot GEMV. Shared execution, routing, paging and TP
communication are excluded from those timings. Relative RMS output difference
against slot GEMV was 0.000114/0.000051, compared with the earlier prepared
adapter's approximately 0.00368 on its synthetic reference. These are numerical
observations for those inputs, not full-model quality results. Device pointer
and route rebinding preserved output bits under the same captured graph.

Build the optional backend by running `scripts/export_b12x_moe.py --output DIR`
on the target GPU with FlashInfer 0.7.4 and Cutlass DSL 4.8.1/CUDA 13 installed,
then configure CMake with `-DDGPP_B12X_AOT_DIR=DIR`. The exporter emits the AOT
header, object and small static CUDA dialect runtime archive. The server needs
neither Python nor a JIT compiler. Generated files belong in the build directory.

The current serving comparison uses two GB10 nodes, a 100 GiB expert
budget per rank, BF16 activations and fixed-depth-five DSpark without confidence
scheduling. Each prose/code prompt requests 256 output tokens and is presented
three times. These are the third presentations; prefill is measured separately
and neither rank reads expert weights during these warm requests.

| Execution | Workload | Decode tok/s | Verify ms/pass | Accepted predictions/pass | Prefill seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| Slot GEMV | Prose | 25.46 | 77.26 | 1.19 | 0.203 |
| Prepared b12x | Prose | 25.33 | 74.29 | 1.10 | 0.187 |
| Slot GEMV | Code | 39.67 | 78.48 | 2.46 | 0.239 |
| Prepared b12x | Code | 42.34 | 75.23 | 2.58 | 0.225 |

Verification uses six rows and top6 throughout. Its timer includes the target
forward, communication and host coordination. The different accumulation order
changes generated continuations and acceptance counts, so the decode-rate
difference cannot be attributed entirely to kernel speed. Both nodes recorded
zero swap and matching operation streams. This establishes a working optional
backend and modest verification-time savings on these samples, rather than a
general serving-speed win; it remains disabled by default.

With this b12x backend, adaptive fractional residency measured 25.33/41.44 warm
decode tok/s on prose/code versus 25.33/42.34 for whole-expert caching. All six
responses match across policies, and all twelve match the preceding build's
responses. Both ranks record full native hits and zero expert reads on the
third presentations. Thus fractional allocation converges to the same useful
hot coverage without incurring population-wide planning on every interval.

The shared read/planning changes, with the b12x arithmetic held constant, raised
fractional warm decode from 21.53/36.58 to 25.33/41.44 tok/s and cut its server
startup from 223.3 to 108.4 seconds. Whole-expert first-presentation prose prefill
fell from 56.19 to 24.62 seconds, with decode rising from 1.77 to 2.96 tok/s.
These are serving measurements on the same prompts, not a separate I/O sweep.

### Warm fixed-depth serving comparison

A short C1 comparison holds the b12x backend, whole-expert policy, checkpoint,
100 GiB expert budget per rank and V46 binary constant. Confidence scheduling
is disabled. Each setting starts a new two-GB10 deployment and presents the
same prose and code prompts three times, requesting 256 output tokens each.
The table reports the third presentations, with zero expert-image reads and
100% native/full hits on both ranks. Prefill and queueing are excluded from
decode throughput.

| Draft depth | Target rows | Prose decode tok/s | Code decode tok/s | Prose verify ms/pass | Code verify ms/pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off | 1 | 24.34 | 24.15 | — | — |
| 1 | 2 | 29.68 | 32.46 | 48.65 | 49.37 |
| 2 | 3 | 30.58 | 37.68 | 55.77 | 56.44 |
| 3 | 4 | 29.21 | 44.06 | 62.15 | 62.87 |
| 5 | 6 | 25.31 | 42.39 | 74.25 | 75.10 |

Depth two is fastest for this prose prompt; depth three is fastest for code
and minimizes combined decode time for the equally sized pair. At depth three,
prose/code accept 1.04/2.13 predictions per pass, compared with 1.10/2.58 at
depth five. The additional accepted predictions at five do not repay its
larger target forward. Drafting takes 7.48/7.51 ms at three and 8.09/8.21 ms
at five; verification remains about 89% of the depth-three cycle.

Warm prefill takes 0.184–0.187 seconds for prose and 0.222–0.225 seconds for
code. Prefix caching is disabled. All rank operation streams match, and both
rank launch scopes prohibit swap. Continuous process-memory samples confirm
zero swap on both ranks for depths one through five; the off baseline lacks
peer memory samples because its monitoring helper was absent. Its serving
measurements completed before that collection failure and were retained.

These are short observations on two prompts. Generated continuations differ
across verification widths, so the decode differences include changes in
acceptance and subsequent routing. They do not isolate kernel speed on one
fixed continuation or establish a universal best depth. Use fixed depth three
as the starting point for further C1 execution work, with depth two retained
as the faster prose setting for this sample.

A single follow-up at depth three enables `engine.bf16_weights: "bf12+bf16"`
on the same binary. This existing lossless format packs each rank's vocabulary
head from 0.62 to 0.46 GiB, reducing decode weight traffic. The original BF16
head stays resident for prefill, so the companion adds approximately 0.46 GiB
per rank; it does not reduce total resident memory. Packing takes about 0.1
seconds at startup. Expert budget and quantization remain unchanged.

| Depth-three head storage | Prose decode tok/s | Code decode tok/s | Prose verify ms/pass | Code verify ms/pass |
| --- | ---: | ---: | ---: | ---: |
| BF16 checkpoint | 29.21 | 44.06 | 62.15 | 62.87 |
| BF12 companion + BF16 | 30.07 | 45.30 | 60.96 | 61.75 |

Drafting also falls from 7.48/7.51 to 6.67/6.70 ms per pass because it reads
the same head. Warm prefill takes 0.187/0.224 seconds for prose/code. All six
response texts match the depth-three BF16 control exactly, with unchanged
acceptance counts. Both ranks retain zero expert reads on the reported warm
requests, zero swap and matching operation streams. The deployment ends
stopped, matching its initial state. The roughly 3% decode gain is a short
serving observation; the target forward remains the dominant cost.

A subsequent short Nsight capture uses this fixed-depth-three b12x/BF12
recipe on the code prompt after two warm presentations. Both ranks read zero
expert-image bytes across the captured request, and its output matches the
unprofiled BF12 control. This captures four-row/top6 verification in the
current backend, including the intervening draft work:

| GPU timeline coverage | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Routed target experts | 46.4% | 46.5% |
| Dense kernels | 30.6% | 30.7% |
| Communication kernels | 4.7% | 3.6% |
| GPU gaps | 6.8% | 7.5% |

Kernel coverage may overlap. Rank captures begin slightly apart, so their
raw totals are not a rank-speed comparison. Shared experts occupy about 14%
of each timeline and mostly overlap routed work. The fused b12x kernel
launches 32 blocks of 256 threads with 114 registers/thread and 58,368 bytes
of shared memory per block; rank 0 averages 0.804 ms per routed layer call.
Routed execution remains the largest optimization target, followed by dense
projections. Host synchronization totals mostly wait for this GPU work and
cannot be added as a separate latency cost. Profiled throughput is excluded
from the serving comparison above.

The checkpoint has eight output groups, hence four per TP2 rank. Its costly
attention output projection is therefore 5120 by 4096; the query expansion is
16384 by 1280. Their observed mean kernel times of 109.35 and 99.03
microseconds correspond to roughly 192 and 212 GB/s of unique FP8 payload.
The vocabulary-sharded BF12 head similarly streams about 248 GB/s of packed
payload. These are payload/time estimates, not physical DRAM counter readings;
they explain why the dense share alone does not establish an inefficient
kernel or justify another occupancy sweep.

A bounded 48-block candidate for up to four rows did not reduce target-pass
time: warm prose/code verification took 61.71/62.78 ms versus 60.96/61.75 ms
with 32 blocks. Generated continuations changed; code's slightly higher decode
rate accompanied higher acceptance and does not establish a kernel-speed win.
The candidate was reverted. Its observer flagged a startup transition before
the no-swap scope attached; sampled process/cgroup swap remained zero. This
diagnostic was retained with that limitation and was not repeated to repair
the observer receipt.

A second occupancy candidate uses b12x's three-stage small-batch mode and
64 persistent blocks. It reduces shared memory from 58,368 to 38,016 bytes
per block. In a four-row/top6 kernel comparison with the FP8 shared expert
running concurrently and CUDA Graph replay, prose/code latency is
820.31/851.58 microseconds for the original versus 820.16/848.37 for the
candidate. The routes are the first four rows of saved median-count six-row
serving tables; weights, scales and activations are synthetic. This shows no
useful combined-lane gain, so the candidate was not put into serving. The
original four-stage, 32-block implementation remains selected.

Two focused dense-kernel prototypes also remain excluded. A packed E4M3
conversion/scaling bridge preserves the original BF16 values, with a fallback
for generic FP32 scales, extreme scales and NaN codes. Its conversion check
matches 2,146,304 elements, and four-row projection outputs match bitwise,
but the six tested TP2 attention projection shapes run about 1–10% slower.
A separate prototype allocates activation shared memory for the actual row
count and supplies padded MMA rows as zero registers. It preserves bitwise
outputs and improves the query-down projection from 35.53 to 33.36
microseconds, but the other five shapes show no meaningful improvement.
Both comparisons replay CUDA Graphs and rotate eight weight copies beyond
L2; they use synthetic weights/scales/activations and are kernel measurements,
not serving rates. Neither candidate justifies another full-model run. The
depth-three b12x/BF12 recipe above remains the fastest measured C1 recipe
for combined prose/code decode time; no additional serving gain is claimed.

### C16 throughput and read window

Depth-three DSpark lets sixteen requests use 64 target rows, within the
prepared b12x backend's capacity. This removes the larger-batch fallback but
does not make speculation the fastest C16 recipe. With the same V46 binary,
100 GiB total expert budget per rank, BF12 head companion and sixteen distinct
prose prompts requesting 64 output tokens, third-presentation results were:

| Execution recipe | Decode tok/s | Engine prefill seconds | Expert reads per rank, GiB |
| --- | ---: | ---: | ---: |
| b12x, ordinary decode, 2 buffers / 1 worker | 60.17 | 4.64 | 4.99 / 4.99 |
| b12x, fixed depth 3, 2 buffers / 1 worker | 22.17 | 13.32 | 34.96 / 34.99 |
| b12x, fixed depth 3, 16 buffers / 4 workers | 32.95 | 7.09 | 35.31 / 35.25 |
| Canonical slot, ordinary decode, 16 buffers / 4 workers | 102.90 | 1.23 | 0 / 0 |

The b12x rows use whole-expert caching. Read counters cover each complete
presentation, including its prefill; they are not decode-only I/O counters.
Different execution and batch schedules produced different continuations, so
these are serving-recipe comparisons rather than isolated kernel speedups.

The matched depth-three read-window change uses queue depth 16 in place of
eight, alongside sixteen buffers and four workers. Rank-zero file-ticket wait
falls from 24.68 to 4.29 seconds with about 1% more bytes read. Its
reconstruction time excluding file waits stays near 13 seconds. First-presentation
wall time falls from 152.5 to 73.2 seconds. Both rank buffers remain charged
inside the common expert budget. The wider window therefore addresses an
observed miss-path cost without increasing that budget.

The throughput-oriented recipe disables MTP and the optional b12x backend,
retains `bf16_weights: "bf12+bf16"` and grouped prefill, and uses the wider
read window. The NVFP4 two-node streaming template now carries these engine
settings. Both policies were run on that recipe with three presentations of
the same sixteen prompts:

| Residency policy | Startup seconds | First presentation wall seconds | First prefill seconds | First expert reads per rank, GiB | Third decode tok/s | Third prefill seconds | Third wall tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Whole expert | 16.46 | 64.8 | 21.87 | 98.86 / 98.86 | 102.90 | 1.23 | 92.56 |
| Adaptive fractional | 152.55 | 72.9 | 26.14 | 47.33 / 47.33 | 103.21 | 1.23 | 92.84 |

Startup includes universal fractional initialization; first-presentation read
counts exclude startup. Thus the lower fractional request-time reads do not
establish lower total cold-start I/O or latency. Both warm presentations have
100% native/full hits and zero expert-image reads on both ranks. All 48
responses match across policies. Every request reports zero cached prompt
tokens, both ranks record zero swap, operation streams match, and the servers
end stopped. These are short C16 observations on one prompt set. The follow-up
below extends the serving comparison to changing and mixed workloads;
physical DRAM bandwidth remains unavailable.

### NVFP4 serving under changing locality

A compact follow-up uses the same V46 binary and C16 throughput recipe for
both policies: canonical slot execution, ordinary decoding, BF12 head,
100 GiB expert budget per rank and sixteen read buffers/four workers. One
deployment per policy serves eight consecutive waves of sixteen requests,
64 output tokens each. The sequence changes from prose to code, returns to
prose, then mixes prose/code/JSON/math/chat before returning to prose again.
Each wave has sixteen distinct prompts. These are actual model routes;
the mixed workload is broader traffic, not a uniform expert-selection oracle.

| Wave | Whole decode tok/s | Fractional decode tok/s | Whole reads, GiB/rank 0 | Fractional reads, GiB/rank 0 |
| --- | ---: | ---: | ---: | ---: |
| Prose, first presentation | 23.56 | 21.89 | 98.86 | 47.96 |
| Prose, repeated | 90.14 | 89.87 | 0 | 0 |
| Code, workload shift | 42.57 | 37.49 | 35.64 | 35.56 |
| Code, repeated | 98.50 | 96.75 | 0 | 0 |
| Prose, return | 35.55 | 30.68 | 36.75 | 36.91 |
| Mixed, workload shift | 24.11 | 20.71 | 80.34 | 79.41 |
| Mixed, repeated | 25.59 | 21.61 | 71.79 | 71.97 |
| Prose, final return | 38.86 | 32.46 | 33.68 | 33.74 |

Decode excludes prefill and queueing. Reads cover the complete wave, including
prefill, and count direct expert-image file bytes rather than device-wide
physical NVMe traffic. Rank 1 reads differ by at most 0.12 GiB from the listed
rank 0 values. Fractional initialization reads another 100.58 GiB per rank
before these request counters start, versus zero for whole-expert caching.
Startup takes 16.6 seconds for whole-expert and 136.5 seconds for fractional
caching. Lower first-wave reads therefore do not establish lower total cold
I/O or latency.

Both repeated prose/code waves have 100% native/full hits and zero expert
reads on both ranks. Repeated mixed traffic still misses: whole/fractional
full-hit rates are 95.99%/95.97% on rank 0. Its prefill costs 12.92/15.53
seconds, versus 1.65/1.75 seconds for repeated prose. Fractional execution is
slower through shifts despite similar file bytes; these measurements do not
attribute that gap to one reconstruction, allocation or I/O component.

Server-histogram estimates for repeated prose TTFT p50/p95/p99 are
1.75/2.43/2.49 seconds for both policies. Inter-token latency is approximately
0.174/0.249/0.300 seconds for whole-expert and 0.174/0.264/0.300 for fractional
caching. Repeated mixed TTFT rises to approximately 11.1/19.1/19.8 seconds;
inter-token p99 reaches approximately 4.41 seconds for both. These are linear
interpolations of coarse histogram buckets, with sixteen TTFT observations
per wave, not precise tail estimates.

All 128 response texts match across policies and every request reports zero
cached prompt tokens. Both deployments have matching rank operation streams,
zero swap and stop normally. Minimum sampled node-available memory is
8.31/9.07 GiB across the whole-expert ranks and 8.24/8.87 GiB across the
fractional ranks. The common expert allocation remains inside its 100 GiB
per-rank limit. Process RSS excludes much of the CUDA allocation and cannot
substitute for total node usage. Physical DRAM bandwidth remains unavailable.

## Scope and implementation status

Serving reads now omit the serial byte-by-byte FNV payload checksum. Set
`DGPP_EXPERT_VERIFY_RECORDS=1` to enable it for diagnostics. Image identity,
header/index checksums, record extents, complete reads and direct-I/O alignment
remain checked. The image builder still writes payload checksums; existing
images need no conversion. This change applies to both residency policies and
both weight formats.

Periodic allocation planning also skips intervals in which every observed
expert is already fully resident. Routing frequency/recency observations
continue; a partial selection re-enables planning. Demand admission remains
active, and explicit adaptation without observations retains its prior behavior.
This avoids rescoring the complete chunk population during a zero-I/O hot phase.

Demand admission builds a minimum heap of eligible donors instead of sorting
the complete cache population. It computes each donor's decayed value per
byte once, then selects only as many donors as the admission needs. Score,
expert-ID tie order, fractional floors, cooldown and promotion margin are
preserved. This changes planner CPU cost rather than the allocation policy.
A GB10 CPU comparison with 15,360 experts and 1,200 proposals per scenario
matched complete plan-sequence digests for both policies, with uniform and
unequal expert sizes. Whole-expert proposal time fell from 324.9 to 130.2 ms for uniform
experts and 244.7 to 97.6 ms for unequal experts. Fractional proposal time fell
from 2,254.0 to 238.4 ms and 2,281.9 to 242.1 ms respectively. These are
planner timings, not model throughput measurements. The current 16 planner
and 23 cache tests pass.

A short two-node NVFP4 serving comparison retained this heap change. Both
builds used fractional caching with a zero minimum fraction, fixed depth
three, b12x/BF12, 24 reconstruction slots, 16 staging buffers, four workers
and a 100 GiB expert budget per rank. After one cold C16 mixed wave of
32-token responses, the first 128-token C1 code request improved from 16.03
to 18.25 decode tok/s and prefill fell from 2.317 to 1.882 seconds. All three
code responses matched, each with 87 accepted predictions across 41 passes.
Rank-zero reconstruction time excluding file waits fell from 4.675 to 1.855
seconds, while file-ticket wait increased from 0.610 to 2.035 seconds and
reads changed from 12.41 to 12.21 GiB. Reducing CPU work exposed more storage
wait; file-wait time alone would give the wrong interpretation. Final warm
decode measured 45.12 versus 45.07 tok/s, with zero expert reads on both
ranks. Four mixed continuations differed, so their rates do not isolate the
planner change. Both deployments sampled zero swap on both ranks and
completed with matching rank operation streams. Whole-expert planner speed
is measured above; its end-to-end gain from this change is unmeasured.

Native layout preparation also avoids scanning the complete cache slab. It
examines aligned runs containing the requested expert's own chunks, then uses
the holes left by those chunks for relocation. An index supports constant-time
free-slot insertion and removal. It replaces the old slab-sized hole scratch
list; temporary hole tracking is bounded by one expert's chunk count. No new
weight slab is allocated. Both formats and policies pass randomized cache
churn with exact and incomplete slab tails, and the canonical/b12x bitwise
paging fixtures pass, including native bindings and captured replay.

A short fractional-cache serving run used the same configuration and sequence
as the preceding heap candidate, reusing that completed run as its baseline.
All three code responses and their 87 accepted predictions across 41 passes
matched. On the second code request, both ranks read zero expert bytes;
rank-zero native preparation fell from 0.422 to 0.293 seconds despite 798
versus 785 conversions and slightly more copy bytes. Decode measured 41.19
versus 41.90 tok/s and prefill 0.423 versus 0.380 seconds. Final warm decode
remained 45.07 versus 45.09 tok/s with no conversions or expert reads. Code
after mixed traffic measured 18.25 versus 18.39 tok/s. One mixed continuation
differed, so mixed rates do not isolate the layout change. The candidate had
zero sampled swap on both ranks, matching rank operation streams, and stopped
normally. These observations support retaining the preparation-cost reduction;
they do not establish a stable warm throughput gain.

Demand reconstruction submits its bounded read window before waiting for
the first chunk. It batches copies from already-ready reads behind one stream
fence, retaining every staging lease until that fence succeeds. Before an
unfinished read or a refill with all buffers held, it drains the copies to
release staging capacity. Cache destinations are reserved separately and the
admitted residency is published only after completion. This adds no per-copy
CUDA events or payload buffers. Both-format/both-policy host tests cover ready
batches and failed-fence lifetime; the canonical and b12x paging fixtures pass.

One short fractional-cache serving run reused the preceding completed run as
its baseline, with the same fixed-three b12x/BF12 recipe and mixed/code
sequence. All three code outputs and acceptance counts matched: 87 accepted
predictions over 41 passes per response.

| Code presentation | Previous decode tok/s | Batched-copy decode tok/s |
| --- | ---: | ---: |
| First after mixed traffic | 18.39 | 19.39 |
| Second, zero expert reads | 41.90 | 39.42 |
| Third, no reads or conversions | 45.09 | 45.01 |

First-code prefill remained 1.874/1.872 seconds. Rank-zero reconstruction
excluding file waits fell from 2.039 to 1.921 seconds, and rank one from
1.912 to 1.866. File-ticket waits fell from 2.133/2.266 to 2.006/2.233
seconds across the two ranks. Four mixed responses differed, so their rates
do not isolate the change. Both ranks sampled zero swap and produced matching
operation streams within the candidate deployment. It stopped normally.
These observations support a modest miss-path improvement; the slower second
request and unchanged final warm rate preclude a general throughput claim.
Independent readiness and execution of complete experts remain separate work.

A separate-transfer-stream prototype was tested and removed. It preallocated
reader and copy-ready events, rotated reconstruction slots, and launched groups
of four reconstructed experts while loading later groups. A focused test proved
that another slot could finish reconstruction while the execution stream was
blocked; the paging fixtures preserved bitwise results. In one serving sequence
against the retained batched-copy baseline, code after mixed traffic measured
19.08 versus 19.39 decode tok/s, and final warm code 44.83 versus 45.01.
The intermediate zero-read request improved from 39.42 to 42.31 tok/s, but
three mixed responses differed and the intended miss-path gain was absent.
All three code outputs and their acceptance counts matched. Both ranks sampled
zero swap and matching operation streams; the deployment stopped normally.
These results do not justify retaining the extra stream, events and scheduling
path. The single-stream batched-copy implementation remains selected.

Implement `whole_expert_cache` and `adaptive_fractional_cache` over the same
rank-local storage, memory manager, MoE execution path and telemetry. Neither
policy is selected as the expected winner. The initial model is DeepSeek
V4.1 Flash 1017 on GB10; existing resident and whole-layer streaming modes
remain available as controls.

The branch implements a reconstruction baseline for both policies in the
DeepSeek V4.1 MXFP4 eager serving path. The frozen initial build passed small
GB10 correctness fixtures and full-checkpoint world-one numerical comparisons.
A subsequent whole-expert demand-admission correction has passed host checks,
GB10 kernel/model fixtures and the full-checkpoint world-one comparison. The
full-checkpoint TP2 short serving diagnostic completed all four matched boots,
with equal greedy output and zero swap. It measures a workload-dependent policy
contrast; broader model throughput or latency gains remain unproven. Later
sections distinguish the implemented baseline
from the remaining execution and measurement gates.

The current policy revision additionally permits `minimum_fraction: 0` and
fractional demand admission. Fractional startup still populates at least one
chunk per expert and water-fills the available budget; a zero floor permits
later removal of all chunks from cold experts. Both policies can retain weights
from demand staging immediately, outside the periodic speculative-loading cap.
Fractional admission compares expected saved load cost per byte and removes
lower-value resident tails subject to floors, hysteresis and cooldown. A plan
is published only after transfers complete; failure poisons the cache. Host
planner/cache/config tests pass, including both-format byte reconstruction,
feasible/infeasible hot sets and failed retention. The cross-compiled revision passed all 34 planner/cache/config host tests and
seven GB10 MoE/model fixture tests with zero sampled swap and scope swap disabled.
The MoE test covers both formats, GEMV/MMA, shared expert variants and rows
1/7/64; repeated one-row hot routing performs no expert-file reads for either
policy without periodic adaptation. The combined implementation has passed
full-checkpoint MXFP4 TP2 numerical qualification and completed the
local-SSD serving comparison described below. Historical performance tables
explicitly identify earlier builds and their admission semantics. The third
proposed hybrid policy is deliberately not implemented.

For equal-size experts, making a hot set of weight size `H` fully resident
while keeping fraction `f` of the remaining routed weights `M-H` requires
`H + f*(M-H)` weight capacity. Compare this with the cache payload capacity,
after subtracting execution scratch, staging and allocation padding from the
subsystem budget. Chunk rounding can increase the requirement. A positive
floor can prevent full hot-set promotion even when whole-expert caching fits
that set; a zero floor removes that coverage constraint. Startup remains
universal and partial. Admission retains already-read demand chunks immediately;
periodic adaptation handles speculative allocation and longer-term changes.

The following shared execution revision adds native layout reuse in the same
cache slab. A complete expert is gathered once into the existing bounded
execution scratch and packed into an aligned run of cache slots. Relocated
compact chunks move into holes left by that expert; no extra payload slab is
allocated. Native groups remain in stable runs. Partial demotion first restores
independent compact chunks, preserving source packing, scales and globals.
Direct leases prevent cache eviction/relocation until readers finish. When an
entire routed set is resident, MoE binds all native views, uses the existing
grouped expert chain, and retires the group with one completion fence. Partial
sets retain the bounded wave fallback. Host cache tests pass under ASan/UBSan,
including distinct expert payloads, fragmented layouts, randomized admission and
adaptation, pinned readers, one-fence grouped retirement and failed conversion.
This revision also passed 39 planner/cache/config host tests and seven GB10
MoE/model fixtures, with matching binary hashes, zero sampled swap and scope
swap disabled. The MoE fixture asserts stable repeated hot routing has no
reconstruction copies, layout conversions or file reads, while preserving
bitwise resident-chain output for both formats and policies, GEMV/MMA, shared
expert variants and rows 1/7/64. The combined revision passed a full MXFP4 TP2 numerical check against a
separate current-build whole-layer streaming control: all 40 layers, seven
seed-43 rows, routes and complete-vocabulary logits match byte-exactly on both
ranks. Both policies used a 32 GiB expert budget, chunk width 64 and floor zero;
all ranks had zero sampled swap with scope swap disabled. Control, whole and
fractional legs took approximately 240, 248 and 373 seconds respectively; these
include loading and are correctness timings, not performance comparisons. This
one-forward check exercises cold paging; the repeated native-hit assertions
come from the MoE/model fixtures. The serving measurements below use the
subsequent GPU segmentation revision. That revision uses
the existing GPU segmentation kernels for paged execution too: the host reads
route IDs, weights and segment metadata at one decision fence, while row maps,
slot maps and the fixed expert segment table stay on the device. The native
grouped path includes empty segments; miss waves address the selected expert
directly. Public route getters retain their existing semantics. This change
passed the 39 host tests and seven GB10 MoE/model fixtures with zero sampled
swap and scope swap disabled, including route getter and bitwise output parity.
CUDA Graph compatibility remains unfinished; graph
guards remain enabled.

The subsequent full-model serving attempt exposed a separate storage dependency:
both ranks mapped the two Engram shards from the NAS, although expert-cache
misses used local NVMe. Those Engram shards total 203,073,077,576 bytes
(about 189.1 GiB). The run was interrupted and its partial measurements retained
as diagnostics; it does not establish all-local-SSD serving performance.
The replacement comparison requires local checkpoint files and expert images
on both nodes, plus live process file/mapping checks for NAS dependencies.
Expert-file counters alone cannot establish where the rest of the model reads
its weights. Engram demand paging and operating-system page-cache warmth must
also be disclosed when interpreting cold versus repeated requests.

### Completed local-SSD full-model comparison (2026-11-14)

The replacement comparison completed all four boots in A/B/B/A order on two
GB10 nodes. All checkpoint shards, including Engram, and both expert images
are on local SSD. Each rank has a 96 GiB expert-subsystem budget. The shared
engine permits two active requests; client concurrency is C1/C2/C4, so C4
includes queueing. Each boot runs three identical 32-token warm windows, five
serving classes at all three concurrency levels with first/repeated
presentations capped at 64 output tokens, then 512/2048-target prefill probes.
No quality or long-context campaign is included.

All 70 serving request cells have identical greedy text, completion counts
and finish reasons across all four boots (280 requests total). Every boot's
rank operation streams match. Both memory monitors completed, each observing
all four serving processes, with zero swap violations and scope swap limits
of zero. The frozen server binary is
`21fca38ed97a210da6075abaa0a422cacf640898cd070da0a69cea509baba84b`.

The following ranges cover all five classes and both boots per policy for
repeated presentations. Decode rates exclude prefill; client wall rates
include prefill and queueing. Ranges are observed extrema, not confidence
intervals.

| Clients | Whole decode tok/s | Fractional decode tok/s | Whole client wall tok/s | Fractional client wall tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 15.47–16.63 | 14.93–16.05 | 14.32–15.45 | 13.79–14.74 |
| 2 | 26.50–27.23 | 25.16–25.80 | 23.86–24.44 | 21.95–22.71 |
| 4 | 26.88–27.96 | 25.24–26.41 | 24.68–25.71 | 22.78–23.87 |

Every repeated serving phase has 100% complete expert hits on rank zero,
zero expert-image reads and zero reconstruction for both policies. Some
one-time native-layout work remains within those repeats, so they are not
all stable-layout measurements. Comparing the median of each policy's two
boots within each of the 15 class/concurrency cells, whole-expert repeated
decode is 3.2–7.0% faster. This describes these workloads and the shared eager
execution path; it does not isolate policy overhead or establish a general
winner.

The fractional strategy does converge to a fully native hot set on the
identical warm prompt. In its third warm window, both ranks record
12,403/12,403 complete/native expert hits with zero expert-file bytes,
reconstruction and layout work. The preceding window already has complete
hits but still incurs one-time layout work. Zero minimum residency and
immediate demand retention therefore remove the earlier allocation barrier;
this fixed-prompt observation does not prove convergence for arbitrary routing.

First-presentation results remain separate. Prose C1 favors fractional
caching: its two boots achieve 3.70–3.76 decode tok/s and 6.69–6.92 s TTFT,
versus whole-expert caching's 2.92–2.94 tok/s and 9.24–10.64 s TTFT.
Other classes have mixed results. A first presentation does not establish
empty expert residency or globally cold storage: earlier warmup, previous
classes and Engram's operating-system page cache affect the state. Per-request
TTFT/wall throughput and server p50/p95/p99 histogram brackets are retained;
quantiles are reported as their original buckets, without interpolation.
With one or a few requests per phase, these are not reliable population tails.

Initialization changes the complete cost comparison:

| Boot | Policy | Startup s | Serving command s | Entire boot s | Lifetime expert-image GiB, ranks 0 / 1 |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | Whole | 283.69 | 528.74 | 1079.44 | 217.46 / 217.39 |
| B1 | Fractional | 479.50 | 523.66 | 1216.52 | 263.77 / 263.84 |
| B2 | Fractional | 477.49 | 510.96 | 1195.71 | 264.14 / 264.20 |
| A2 | Whole | 283.49 | 509.33 | 1049.06 | 218.43 / 218.47 |

Entire-boot wall time includes startup, warmup, serving, calibration, prefill
and teardown; it is not serving throughput. Both fractional boots read
96.37 GiB per rank for universal initial population before workload execution,
then 167.40–167.83 GiB per rank afterward. Whole-expert caching starts empty.
Fractional thus saves about 50 GiB per rank after startup, but reads more over
the entire measured boot. Expert-image counters include record padding and
exclude checkpoint source hashing. The frozen binary still performs the full
source scan on each boot; the subsequent local source-stamp optimization is
not part of these timings.

Sampled whole-device reads from boot through teardown, including source scans
and non-expert activity, are 507.82–509.27 GiB per node for whole-expert caching
and 554.56–554.93 GiB for fractional caching. Sampled intervals omit the
margins outside the first and last samples; device traffic is not expert-only
and must not be added to logical expert reads. Within serving repeats,
interior block samples register nearly zero reads, but outer sample brackets
include adjacent traffic and do not prove exactly zero total device I/O.

Across all boots and nodes, minimum sampled MemAvailable is 11.18–11.81 GiB,
process RSS peaks are 12.57–13.73 GiB, and process swap peaks are zero.
Substantial CUDA allocations are outside RSS. RSS, node available memory and
the fixed expert reservation overlap in accounting and cannot be summed into
a total model allocation. The common startup memory plan is 106.15 GiB per
rank, including the 96 GiB expert subsystem and separate backbone, KV, scratch,
Engram staging and runtime allocations. Physical DRAM bandwidth remains unknown.

The prefill probes compute all prompt tokens without prefix-cache reuse.
They use distinct tagged prompts, including calibration, and should not be
interpreted as an exact-prompt policy contrast or a long-context result.
The saved expert reconstruction and layout clocks cover different amounts of
work across policies and are nested within engine time. They must not be
summed into an end-to-end time or used to claim a kernel-only speedup.

### NVFP4 local-SSD policy baseline: first pair before execution tuning

The current optimized serving binary runs the pinned NVIDIA NVFP4 checkpoint
on both Sparks with all weight shards, including Engram, on local SSD.
The first whole/fractional pair has completed all 30 serving phases per boot:
70 request cells, 140 requests, with identical generated text, token counts
and finish reasons. Both boots also completed their prefill probes and have matching rank
operation streams. The second fractional boot also completed serving; its prefill was
interrupted when the user redirected work to execution speed. The final
whole boot was cancelled. These are provisional
single-boot-per-policy observations, not the final A/B/B/A estimate.

| Clients | Whole repeated decode tok/s | Fractional repeated decode tok/s |
| ---: | ---: | ---: |
| 1 | 15.31–16.32 | 14.86–15.68 |
| 2 | 26.02–26.68 | 24.70–25.19 |
| 4 | 26.19–27.09 | 24.90–25.61 |

Four clients queue behind two active engine slots. The table uses engine
decode timing; client wall rates also include prefill and queueing.
Ranges span the five workload classes within the first policy pair.
Across its 15 matched repeated-decode cells, whole-expert caching is
3.0–6.3% faster (median 5.3%). Every repeat under both policies records
100% rank-zero full-expert hits and zero expert-image reads. These facts
separate warm policy overhead from miss-related I/O without establishing
maximum resident-model throughput. A live observation during the second
fractional boot showed only 8%/7% GPU utilization on the two ranks. The
campaign alternates first presentations and short warm repeats, limits the
engine to two active slots, and uses eager execution with host routing
decisions and synchronization. This instantaneous utilization observation
is not a phase average or a measured breakdown of the warm-path limit;
zero expert-image reads do not establish efficient GPU execution.
Source inspection identifies two unconditional stream synchronizations in a
baseline native-hit MoE call: `GlmMoeLayer::enqueue_host` drains the stream after
routing metadata download, then `ExpertCache::release_native_leases` calls
`CudaCacheStorage::finish`, which drains it again after expert execution.
Across 40 routed layers this is approximately 80 stream synchronizations per
decode step, including a two-row step. This establishes the presence of
host/device serialization, not its measured share of latency. Deferred
reader retirement and a GPU routing/binding path are common execution work;
they must preserve cache eviction and lease lifetime safety under misses.
The subsequent tuning build removes the native-retirement drain for the
CUDA backend: all cache writes and kernel readers use the same ordered
engine stream. Staging reuse and destruction still fence. It also retains
separate native-binding device tables per global layer and cache instance,
refreshing selected entries when their addresses change. The next build
routes complete native decode hits through the existing fused slot kernels
instead of the grouped expert chain; its focused both-format parity test
passes at row counts 1/7/64, including cache rebinding. Paging still retains
the host routing decision fence and still refuses CUDA Graph capture.

Both ranks in both fractional boots' third identical warm window also record
12,390/12,390 native/full-expert hits, zero expert-image reads, zero
reconstruction time and zero native-layout copy bytes. This demonstrates
convergence with a zero floor for that fixed prompt. The preceding repeat
still performed layout work despite complete weight hits.

The first whole boot's minimum available memory is 10.85/11.13 GiB,
with peak RSS 12.63/12.71 GiB. The first fractional boot records
11.56/10.97 GiB minimum available and 13.11/13.55 GiB peak RSS.
Sampled process swap is zero on both ranks in both completed boots.
The remaining boots must be included in the final report. Both policies use
96 GiB per-rank expert budgets and the same 106.13 GiB total memory plan.
Source-stamp warmth differs at the first boot as described below; raw
startup differences cannot be attributed solely to residency policy.

The first pair's lifetime expert-image reads are 264.16–264.21 GiB per rank
for whole-expert caching and 314.06–314.19 GiB for fractional caching.
Fractional initialization reads 96.57 GiB per rank, then reads
217.49–217.62 GiB during warmup, serving, calibration and prefill.
Thus its approximately 46.6 GiB reduction after startup is outweighed by
initialization in this first pair. These counters exclude source hashing.
Whole-device sampled reads are 570.66–570.92 GiB per node for the first
whole boot and 333.18–333.43 GiB for the first fractional boot; the first
whole boot additionally refreshes the source stamp. That unequal scan
cost prevents treating this device-read difference as a policy benefit.

Native conversion/unpacking time and copy volume have separate counters, as do
native executions and active direct leases. Conversion clocks are nested within
engine time and must not be added to it. A native execution counter includes
its first layout conversion; only subsequent stable hits have no layout copies.
The replay harness emits these counters alongside reconstruction and I/O.

The working tree additionally contains NVFP4 checkpoint recognition, native
loading, source-image binding and packed MXFP4 Engram mapping/staging/decoding.
The native model path selects an NVFP4 cache identity and rejects mixed-format
main/draft paging before allocation. Host image/config tests, GPU loader and
small-model parity fixtures pass. The frozen pre-optimization build completed
full-checkpoint NVFP4 parity for both policies. The current optimized build
also passes the TP2 seven-token comparison across all 40 main layers and both
vocabulary shards, byte for byte under both policies, with zero swap. For the pinned source,
the pinned checkpoint download is complete and all 55 files (429,008,476,351
bytes) passed size/hash verification at revision
`3ae9c0d3fbbea5370bd06e679d22044db15883f1`. The completed serving diagnostic
used the separate frozen MXFP4 demand-admission build.

The TP2 numerical queue exposed a supervisor observation race: it read a
running receipt immediately before a successful worker exited, then treated
the exit as a missing final receipt. Both whole-expert smoke ranks actually
completed with zero swap and outputs matching the control byte for byte.
Recovery verified the saved artifact sizes/hashes and reused those outputs;
it also qualified the fractional smoke world byte for byte. The full-checkpoint
TP2 whole-expert and fractional cases now match the control byte for byte across
all 40 main layers, seven tokens and both vocabulary shards, with zero swap.
The corrected observer rereads the terminal
receipt after process exit, retries transient SSH observations on the same
handle, and preserves genuine failures. Six offline regression checks cover
these cases. Original failed controller receipts remain preserved separately;
the supervisor failure is not evidence of a model or CUDA failure.

The implementation includes:

- CUDA-free TP geometry and a shared planner with decayed routing frequency,
  observed unprefetched read-wait cost, configurable zero-or-positive fractional floors, cooldowns,
  hysteresis and bounded adaptation traffic.
- Immutable indexed expert images with checksums and explicit direct/buffered
  reads; a bounded asynchronous queue coalesces chunk reads and gives demand
  priority over queued speculative reads.
- A cache manager with fixed chunk slots, complete-expert reconstruction slots,
  aligned pinned staging, engine-thread transfer ownership, completion fences,
  and leases. Partial read/copy failures poison the cache before execution.
- A CUDA backend using the existing model stream, and MoE execution waves using
  the existing complete-expert GEMV/MMA kernels. The shared expert executes once.
- Checkpoint image preparation, descriptor-only routed-expert loading, model
  memory preflight, configuration/journal propagation and per-rank telemetry.

The host suites contain 57 tests covering geometry/planning (16), indexed
storage (5), bounded reads (6), cache/lifetime/failure/resource behavior (20), checkpoint
image preparation and native binding (7) and effective configuration round trips (3). These suites
originally passed AddressSanitizer/UndefinedBehaviorSanitizer at 37 tests;
the extended 14-test planner and 14-test cache suites also pass those sanitizers.
The extended seven-test image suite passes ASan/UBSan, including ModelOpt NVFP4
scalar validation, exact global-factor translation, stale-source rejection and
native main/draft binding against independent source fixtures.
Queue/cache tests passed ThreadSanitizer, including all 14 tests in the current
pressure-feedback cache/monitor suite.
The complete native configuration (28), serving (125), scheduler (152) and
journal/fabric suites also passed. The complete server and CUDA fixtures
cross-compiled for GB10.

On one idle GB10, the paging MoE fixture passed bitwise comparisons for both
policies and both formats, GEMV/MMA, FP8/FP4 shared experts, 1/7/64 rows and
adaptation/slot reuse. Loader fixtures passed descriptor-only loading at
TP1/2/4 (without live collectives). The tiny MXFP4 model matched resident
prefill and three decode steps bitwise after adaptation, including inference
on a worker thread after an explicit, fenced ownership handoff. Ownership
transfer refuses live leases and requires the previous thread to release first. An extended fixture
also passed grouped/mixed prefill, batched verify/rollback, prefix snapshot/
attach and slot reuse in exact and bounded prefill modes. A separate model
fixture passed DSpark draft-block/chain, target verification, fixed-cut rollback
and redraft parity for both policies, with the complete enabled main/draft
population sharing one budget. The initial baseline disabled serving MTP; eager DSpark integration and current
serving evidence are described above. This is fixture
qualification, not real-checkpoint, multi-rank or performance evidence.

Single-node and live TP2 HTTP smoke tests on two GB10 nodes with a synthetic
512-token vocabulary also passed for both policies and a resident control: identical greedy responses, live metrics and clean
shutdown after engine-thread handoff. It is serving integration evidence only;
the fixture timings are not a policy performance comparison. The TP2 test
exercised real collectives and rank-local images, and both ranks drained cleanly.
Separate full-checkpoint TP2 dump checks now establish byte-exact parity for
both policies against a same-world whole-layer streaming control, as described
below.

Paging also opts the eager adapter into the existing native one-row-per-request
batched model walk. The resident/streaming eager defaults remain unchanged,
and a single-slot paging batch retains the scalar path. A GPU adapter regression
passed against resident batched logits/picks with reordered slots, invalid-batch
rejection, position advancement and slot reuse. Live TP2 streaming tests observed
14 decode rows in 7 physical steps for C2 under each paging policy, versus the
resident eager control's 14 serial steps. Both ranks' operation streams matched
within every boot. Configuring two request slots alone therefore must not be
reported as a two-row decode batch without checking these counters.

Current full-checkpoint paging supports MXFP4 and NVFP4 DeepSeek V4.1.
NVFP4 has optimized full-checkpoint TP2 numerical parity under both policies
and actual serving measurements summarized above. Paging requires eager
execution (`decode_graph: false`), `prefill_fold_overlap: false`, no external
draft and no runtime adapters. Native eager DSpark is supported. Unsupported
combinations fail configuration/startup. Host geometry/storage and CUDA MoE
fixtures support both formats. The earlier frozen reconstruction results below
remain evidence for that implementation, rather than timings of current kernels.

### Requirement coverage

| Requirement | Current evidence and boundary |
| --- | --- |
| Interchangeable policies and equal bounded expert budget | Both policies use the same store, planner interface, staging pool, reconstruction slots and production expert kernels. Padding and staging count against the budget. |
| Whole-expert residency | Committed residency is zero or complete. Demand misses reconstruct and admit directly from the same reads when capacity/replacement permits. Full residents can be packed once into native layout and directly reused; stable native hits avoid reconstruction. The current implementation passes host/GB10 fixtures and full MXFP4 TP2 cold-forward parity. |
| Universal fractional coverage and adaptive promotion | Startup universally water-fills the population. Demand reads immediately retain useful chunks within budget; periodic plans handle speculative allocation. A configurable zero or rounded positive floor, promotion, demotion, cooldowns and bounded loading are supported. Host and GPU fixtures exercise locality changes. |
| Asynchronous reads, coalescing and prefetch | Bounded worker queue with demand priority, shared read tickets and reserved demand staging. File workers do not call CUDA. |
| Mathematically valid chunks | Matched gate/up rows and down columns, preserving source codes, scales and globals. Complete reconstruction is byte-exact in both-format fixtures. Direct partial computation is deferred. |
| Real checkpoint correctness | Both policies match all 40 main layers and complete logits of the MXFP4 world-one control on seven seeded tokens. Both also match all 40 layers and both vocabulary shards of a separate TP2 streaming control byte for byte, with zero swap. All four full-checkpoint TP2 serving diagnostic boots have matching greedy transcripts. |
| Both weight formats | MXFP4 model integration; both-format common-backend fixtures. Native NVFP4 host image/config checks, GPU loader TP1/2/4 slices, packed Engram and model parity fixtures pass. Both policies pass full-checkpoint NVFP4 world-one parity on the frozen reconstruction build. The current optimized full-checkpoint TP2 comparison also passes byte-exact 40-layer and full-logit parity under both policies with zero swap. Optimized NVFP4 serving now includes matched C1 warm checks, a C16 warm policy pair and a C16 changing/mixed-locality sequence; all 128 response texts in the locality sequence match across policies. The earlier four-boot NVFP4 baseline was stopped during execution tuning. |
| Locality, cold start and warm cache | Frozen high/low/changing synthetic replay, A/B/B/A, real I/O and expert arithmetic at 1/8/64 rows. Historical long-warmup replay confirms near-full occupancy. Current zero-floor/native-layout replay preserves 33 of 36 completed legs with 64-selection warm windows; eight complete A/B/B/A groups have matching per-phase output digests and zero sampled swap. The three remaining changing-locality legs at 64 rows were held to prioritize the user-requested full-model serving comparison. The optimized NVFP4 C16 serving sequence also exercises prose/code shifts, mixed traffic and return to prior prompts; repeated prose/code have zero reads, while repeated mixed traffic continues missing. A phase name or fixed warmup alone does not establish steady state. |
| Request concurrency, model throughput and latency | The all-local-SSD full-model TP2 A/B/B/A comparison completed: five classes at C1/C2/C4, two presentations each, 70 matched requests per boot, plus prefill probes. All 70 request cells match across four boots. C4 queues behind two engine slots; decode rates and client wall rates are reported separately. Earlier C1/C2 diagnostics also observed native C2 batches. Optimized NVFP4 checks cover C1 fixed-depth speculation and C16 ordinary/speculative serving; the current ordinary C16 warm pair reaches about 103 aggregate decode tok/s. |
| I/O and memory | Logical direct-read bytes/calls, block-device observations, fixed reservations, process/cgroup peaks and zero-swap checks. Block traffic is not expert-only attribution; physical DRAM bandwidth is unknown. |
| Workload-aware replacement | Decayed frequency, recency and observed queue/file wait cost are implemented. The newer working tree samples node MemAvailable and image-device busy time, with hysteresis to pause background adaptation/prefetch; its cache/monitor tests pass ASan/UBSan. Current MoE/model GPU fixtures and full MXFP4 TP2 parity pass; a dedicated full-checkpoint pressure-injection run is unrun. Logical demotion does not return the fixed CUDA slab allocation to the OS. |

The complete Python stream/report suite also passes (40 tests), including
retaining optional per-request server timings. Those first-to-last-token
request timings and client wall rates are distinct from aggregate engine
decode-only throughput and server token-latency histograms.

### Whole-expert demand-admission correction

The initial frozen serving/replay build admitted whole experts only during
periodic adaptation. At 64 MiB per 1024 routes, the first full-checkpoint
diagnostic leg committed only 3.49 GiB of its 32 GiB reservation. This does not
represent a warmed whole-expert cache and is unsuitable for judging practical
use of a 128 GiB node. Historical results retain their original binary identity;
the corrected replay is identified separately below.

The correction fills unused capacity directly from the demand read staging
already needed for execution. Once capacity is exhausted, replacement considers
decayed frequency, recency and observed load cost, with cooldown and hysteresis;
a low-value scan request may bypass valuable resident experts. Victims are
released before loading, complete residency is committed only after every copy
is fenced, and any partial failure poisons the cache. No second storage read
is needed for admission. This demand path is distinct from speculative periodic
adaptation and its byte allowance. Admission copies and planning remain included
in reconstruction wall time. Separate demand-admission count/byte metrics are
subsets of total promotion counts/bytes, not extra traffic to add to them.

The 14 planner and 11 cache host tests pass, including immediate repeat hits,
unequal-size replacement, cooldown/bypass behavior, hard-budget accounting,
one-read admission for both formats and failure without partial publication.
ASan/UBSan and cache TSan pass. The GB10 paged-MoE fixture passes, requiring
bitwise resident-chain parity and a zero-read repeated one-row route before
any periodic adaptation. All four paged-model fixtures also pass, including
native eager batching, rollback, prefix reuse and draft-chain verification.

The historical conservative baseline reconstructs every selected expert,
including full cache hits, into bounded scratch. The newer native-layout path
above avoids rebuilding stable full hits. In the historical path, routing IDs
and weights are copied to the host, and one expert segment executes per wave. Waves currently repeat
whole-row gather/activation work. Whole misses execute from scratch and can
populate persistent residency directly from the same staged bytes. Periodic
adaptation remains available for workload-driven speculative allocation.
Prefetch overlaps file work with existing kernel work; direct fractional
computation and CUDA transfer overlap are not implemented. These costs must
be measured before making performance claims.

Image identity hashes the canonical enabled **expert source set**: configuration
bytes plus canonical expert tensor names, dtypes, shapes and complete source
bytes. It is not a hash of unrelated checkpoint payloads, despite the manifest
field's `checkpoint_sha256` name. The frozen MXFP4 serving binary scans those source
bytes at every startup, including image reuse; report this startup cost
separately. A subsequent local change saves the verified digest in an atomic
`.source-stamp` sidecar. Reopening an unchanged image uses source tensor
descriptors, configuration bytes, mapped shard device/inode/size, nanosecond
mtime/ctime and header identity, together with the image's inode metadata.
It does not read source weight or NVFP4 scalar payloads on a cache hit.
An absent, stale or damaged sidecar triggers the original full source hash;
borrowed tensors without mapped-file ownership always take that path.
The source files must remain immutable during preparation and use. Store
header/record validation remains in place; per-chunk payload checksums are now
diagnostic-only through `DGPP_EXPERT_VERIFY_RECORDS=1`. Existing images with
invalid identity or metadata are refused rather than silently replaced.
This startup change is not part of the completed frozen MXFP4 comparison.
The current NVFP4 serving binary includes it. Its first serving boot opens
weights through the Hugging Face snapshot alias, whereas numerical validation
opened the direct local checkpoint path. Because mapped owner paths participate
in stamp identity, that first serving boot refreshes the source stamp; later
boots reuse the serving alias. Report that difference in source-stamp warmth
separately from policy initialization and expert-image reads. It prevents
attributing all startup differences to residency policy or weight format.

## Controlled expert replay

### Current zero-floor/native-layout replay: completed groups

The current replay uses immediate demand retention, a zero fractional floor,
native layout reuse, chunk width 64 and periodic adaptation every 1024
selections. It retains the synthetic H5120/I2304/E384 MXFP4 image, 2 GiB
expert reservation, fixed high/low/changing route traces, existing expert
arithmetic and fresh-process A/B/B/A ordering. Thirty-three of 36 planned
legs completed before the replay was held to prioritize full-model serving.
Eight groups have both repeats of both policies; only those groups enter
the table. The 64-row changing-locality group is incomplete and excluded.
Every completed group's per-phase output digests match and sampled swap is
zero.

Each entry is the median of two runs for a 512-selection measurement window,
shown as **whole / fractional**. These windows follow the saved warm sequences;
their phase names do not establish convergence. Rows are activation rows
sharing an expert, rather than independent serving clients. Wall time includes
expert arithmetic and output readback/validation.

| Rows | Locality/window | Phase wall seconds | Phase image reads GiB | Native hit fraction |
| --- | --- | ---: | ---: | ---: |
| 1 | High | 1.75 / 1.82 | 0.80 / 0.75 | 91.1% / 91.7% |
| 1 | Low | 12.86 / 14.33 | 5.96 / 6.42 | 33.8% / 28.0% |
| 1 | Changing, before shift | 1.22 / 1.21 | 0.56 / 0.51 | 93.8% / 94.3% |
| 1 | Changing, after shift | 1.07 / 1.16 | 0.47 / 0.49 | 95.3% / 94.5% |
| 8 | High | 2.14 / 2.07 | 0.80 / 0.75 | 91.1% / 91.7% |
| 8 | Low | 16.37 / 13.38 | 6.28 / 6.29 | 30.2% / 29.1% |
| 8 | Changing, before shift | 1.64 / 1.26 | 0.54 / 0.53 | 93.9% / 94.1% |
| 8 | Changing, after shift | 1.27 / 1.13 | 0.48 / 0.46 | 95.2% / 94.9% |
| 64 | High | 3.34 / 3.09 | 0.79 / 0.76 | 91.2% / 91.5% |
| 64 | Low | 18.24 / 14.32 | 6.28 / 6.37 | 30.2% / 28.1% |

Both policies retain the concentrated hot sets effectively and adapt to the
saved locality shift. Broad routing produces substantially more misses for
both policies; fractional phase reads are similar or higher in these low-locality
groups. Phase wall advantages vary by row count and do not consistently follow
the small read-volume differences. Across all phases of each complete group,
including initial population, fractional image reads exceed whole-expert
reads: approximately 5.49–5.51 versus 3.90–3.91 GiB for high locality,
33.96–34.13 versus 31.91–32.58 GiB for low locality, and 7.25–7.27 versus
5.75–5.76 GiB for changing locality. These are finite-trace mechanism results;
they neither measure model tokens/s nor establish an overall policy winner.

### Corrected demand-admission replay

The corrected GB10 binary completed all 36 A/B/B/A legs with the same frozen
high/low/changing traces, H5120/I2304/E384 MXFP4 experts, a 2 GiB reservation,
adaptation every 16 routes and no prefetch. Each leg warmed 2048 routes and
measured 512; changing traces additionally warmed 1024 routes after the shift
before a second 512-route measurement. Premeasurement resident occupancy was
98.93–99.08% of the total expert reservation, including scratch/staging overhead
in the denominator. All legs had zero process swap. All 12 measured
trace/row-count/phase output-digest groups matched across both policies and
repeats. Binary SHA256:
`5ef7930363f1ea37d10df32ed9b9167bc27e502faa4452c144c97cca92b28c37`.

Measured wall milliseconds per routed expert operation, retaining both repeats:

| Rows | Trace/phase | Whole expert | Fractional |
|---:|---|---:|---:|
| 1 | High | 6.670, 6.012 | 7.574, 7.633 |
| 1 | Low | 39.038, 40.171 | 36.044, 26.090 |
| 1 | Changing, before shift | 5.540, 5.551 | 5.806, 5.795 |
| 1 | Changing, after shift/warmup | 4.378, 3.835 | 4.152, 6.823 |
| 8 | High | 6.820, 6.340 | 7.920, 8.348 |
| 8 | Low | 30.865, 29.348 | 35.037, 30.901 |
| 8 | Changing, before shift | 5.310, 5.899 | 6.650, 7.550 |
| 8 | Changing, after shift/warmup | 4.781, 3.854 | 5.194, 4.887 |
| 64 | High | 9.911, 9.558 | 12.862, 9.069 |
| 64 | Low | 37.250, 31.510 | 34.735, 35.625 |
| 64 | Changing, before shift | 8.068, 8.201 | 8.457, 10.399 |
| 64 | Changing, after shift/warmup | 7.185, 6.561 | 8.040, 6.313 |

Whole-expert caching was faster in both repeats for high locality at rows 1/8
and before the distribution shift at all three row counts. Fractional caching
was faster in both low-locality rows-1 repeats; rows-8 favored whole-expert
caching, and rows-64 results overlapped. These observations do not establish a
general winner. For low locality, process physical reads were 7.42–7.72 versus
7.14–7.15 GiB at rows 1, 6.94–7.33 versus 7.32–7.47 GiB at rows 8, and
6.94–7.45 versus 7.43–7.59 GiB at rows 64 (whole versus fractional).
Measured intervals include reconstruction, kernel synchronization and output
validation. Rows share an expert and are not independent HTTP requests.

The first 1,024 selections immediately after the hot-set change also incur
costs; excluding them from the post-warmup rows above would hide the transition.
Summing the two consecutive 512-selection shift-warmup windows gives:

| Rows | Whole wall seconds | Fractional wall seconds | Whole process physical reads (GiB) | Fractional process physical reads (GiB) |
|---:|---:|---:|---:|---:|
| 1 | 6.593, 6.471 | 7.791, 9.576 | 2.496, 2.426 | 3.188, 3.286 |
| 8 | 7.000, 7.056 | 9.846, 9.374 | 2.443, 2.461 | 3.289, 2.948 |
| 64 | 9.660, 8.365 | 11.271, 12.283 | 2.496, 2.303 | 3.220, 3.009 |

Both repeats favored whole-expert caching on elapsed time and process physical
reads during this fixed transition window. These are total costs over the
same 1,024 selections, not incremental costs relative to an unchanged trace,
expert-only device counters or time to a demonstrated recovery threshold.

For this corrected build, reconstruction wall time per routed expert operation
spanned the following ranges across the measured high/low/changing phases and
both repeats:

| Rows | Whole reconstruction (ms/operation) | Fractional reconstruction (ms/operation) |
|---:|---:|---:|
| 1 | 1.206–2.006 | 1.183–1.768 |
| 8 | 1.232–1.699 | 1.197–1.822 |
| 64 | 1.216–1.743 | 1.257–1.587 |

Each value is a phase's accumulated reconstruction wall time divided by its
512 operations, not a request latency percentile. This clock excludes file-ticket
waits but includes CPU work and completion fences. It is nested within the
operation wall times above; adding it again would double count. Both full hits
and misses use reconstruction in this bounded-scratch baseline.

Fixed warmup and near-full occupancy do not prove policy convergence; this
is a controlled reconstruction diagnostic, not full-model serving throughput
or a standard campaign result. The 96 GiB-per-rank TP2 serving comparison is
separate; its complete A/B/B/A results are reported above.

### Replay harness and historical measurements

`expert_replay_bench` exercises the real bounded cache, direct reads and CUDA
reconstruction backend without loading the backbone. By default it runs no
expert arithmetic. It creates an immutable synthetic image if absent, with valid
MXFP4 or NVFP4 geometry and deterministic codes/scales. Image creation occurs
before reported timings. An existing image must match the synthetic recipe's
identity and requested geometry; use a dedicated image path.

Supply a saved text trace with one `PHASE EXPERT_INDEX` pair per line. Adjacent
rows sharing a phase are summarized together; expert indices are zero based.
For example, a trace containing `cold 0`, `steady 0`, `shift 200` selects those
three experts in order. Freeze and hash the trace before comparing policies.

```sh
cmake --build build --target expert_replay_bench
build/expert_replay_bench \
  --hidden 5120 --intermediate 2304 --chunk 64 --experts 384 \
  --format mxfp4 --budget-bytes 2147483648 --adapt-every 16 \
  --policy whole_expert_cache \
  --image /path/on/local-nvme/synthetic-experts.img \
  --trace /path/to/frozen-routes.trace
```

Repeat with `adaptive_fractional_cache`, in A/B/B/A order, using the same
image, trace, shape and budget. Use separate fresh processes for high, low
and changing locality. Include initialization in total cost: the whole policy
starts empty, whereas the fractional policy reads its initial population.
The replay uses no speculative prefetch and no concurrent execution slots.

Add `--compute gemv --rows 1` to execute the existing FP4 gate/up projections,
the existing SwiGLU clamp with limit 7, and the existing FP32 down projection
for every selected expert. Both formats use their original codes and scales.
Use `--rows 8` or `--rows 64` for multiple activation rows sharing the selected
expert; this is expert batch reuse, not independent concurrent requests. The
launcher chunks rows through its GEMV core; this diagnostic does not select
the model's MMA dispatch. Activation and output device bytes are preallocated
and reported separately from the equal expert-weight budget. Each selection's
complete output is copied to the host, checked for finite values and included
in a per-phase FNV-1a digest. Require matching phase digests across policy
legs; the resident-kernel correctness fixtures remain the numerical oracle.
These synthetic weights are a mechanism workload, not a quantized checkpoint
or quality evaluation.

JSONL output includes per-phase wall time and p50/p95/p99 selection latency
(observe, acquire, release and any scheduled adaptation), file read bytes and
calls, cache hits/transitions, pinned/device reservation, process peak RSS,
read-ticket wait and reconstruction wall time. Useful cached/staged copy bytes
and scratch-zero bytes allow reporting effective copy throughput, but do not
measure DRAM bandwidth or GPU-only copy time. Process `read_bytes` includes
all physical reads charged to the process and its threads, including driver
activity; pair it with filesystem/device identity and block counters. It is
not automatically expert-only NVMe attribution. Direct reads bypass the Linux
page cache, without establishing a cold SSD controller cache.

With arithmetic enabled, `acquire_adapt_*` excludes arithmetic and output
validation. `selection_*` and phase wall time include them. Separate
`kernel_launch_sync_wall_ns` and `output_readback_validation_ns` report CPU
launch/synchronization and readback/validation wall costs; the former is not
CUDA-event kernel time. First-use module loading remains in the cold phase.
Output validation is diagnostic overhead, included in total cost rather than
silently subtracted from an apparent model rate. The initial real-kernel
locality/batch campaign below passed its output, budget and zero-swap gates.

This is a mechanism diagnostic. It supplies neither model tokens/sec nor
TTFT/inter-token latency, and cannot replace full-model serving campaigns.

### Initial GB10 mechanism measurements (2026-11-13)

One Spark-2 GB10 ran synthetic MXFP4 H5120/I2304/E384, chunk64, a 2 GiB
expert reservation, adaptation every 16 selections and no prefetch. The fixed
binary SHA256 was
`bf9f0ce19908f442684e720ae98b9840f771ed343ac76c4c676147d493770f46`.
Each workload used the same saved route trace in A/B/B/A order, with a new
process/cache per leg and a common prebuilt local-NVMe image. There were no
other GPU compute processes at each launch. Every sampled process VmSwap was
zero. Block read-byte deltas matched direct file-returned bytes in each leg;
block I/O operation counts were also retained. These are descriptive results
from two runs per policy, without statistical uncertainty estimates.

High locality selected one of 16 hot experts on 90% of selections; low locality
selected uniformly from 384. Both had 64 initial and 256 steady selections.
Changing locality used that high-locality recipe for 320 selections, then
shifted the hot set to disjoint experts for another 320. A seeded generator
(Python `random.Random(43)`) saved all three traces before measurement.
Here and in the short arithmetic replay, `steady` is a saved phase label for
the later window. These short traces do not establish cache or routing-estimator
convergence; report occupancy before treating them as steady-state evidence.

| Workload | Whole total process wall (s), two runs | Fractional total process wall (s), two runs | Whole steady phase wall (s), median | Fractional steady phase wall (s), median | Whole total file reads (GiB), median | Fractional total file reads (GiB), median |
|---|---|---|---|---|---|---|
| High | 5.21, 6.02 | 10.43, 7.42 | 2.72 | 1.88 | 2.13 | 3.40 |
| Low | 17.24, 14.85 | 17.05, 13.44 | 12.13 | 7.84 | 6.23 | 7.29 |
| Changing | 13.03, 9.42 | 10.42, 10.62 | 2.96 after shift | 1.50 after shift | 3.93 | 4.61 |

Fractional initialization read 2037 MiB and took 3.70–5.61 s across these
legs; the whole policy started empty. Fractional residency reduced later
reads and steady-phase wall time, while increasing total reads over these
short traces. High-locality total wall favored whole caching; the other total
wall results overlap. Neither policy has an established end-to-end advantage.

Median per-run steady selection p99 (including scheduled adaptation) was
130/103 ms for high locality, 184/165 ms for low locality, and 156/71 ms after
the locality shift, whole/fractional respectively. These are cache selection
latencies, not token latencies. Reconstruction wall excluding file-ticket
wait still charged about 1.2–2.2 ms per steady selection, including CPU
submission and completion fences. Reconstructing full cache hits therefore
has a measurable cost before any expert arithmetic executes.

The fixed expert reservation stayed within 2 GiB. Process peak RSS was about
0.10–0.11 GiB and excludes CUDA allocations, so it cannot establish total node
peak memory. DRAM bandwidth, overlapped expert computation, concurrency,
full-model startup, quality and serving latency remain separate open gates.

A separate Nsight Systems 2025.3.2 diagnostic of the high-locality whole-policy
replay captured 17,712 host-to-device copies (2.29 GB; 94.7 ms of CUDA activity),
53,136 device-to-device copies (4.62 GB; 179.1 ms), and 320 scratch clears
(6.02 GB; 18.0 ms). Copies include adaptation as well as reconstruction. These
are instrumented activity durations, not phase wall times; do not add them to
the cache's nested reconstruction counters or substitute them into the timing
table. CPU/NAS checkpoint staging was active during this diagnostic, so it is
not an isolated bandwidth comparison. The installed GB20B GPU metric set
exposes copy-engine/SM activity but no DRAM byte-throughput counters. Actual
DRAM bandwidth remains **UNKNOWN**; CUDA copy bytes/time are effective transfer
rates, not physical DRAM traffic. The profile's higher process RSS is profiler
overhead and is excluded from the uninstrumented memory observations above.

### GB10 replay with existing expert arithmetic (2026-11-13)

The same frozen traces, MXFP4 H5120/I2304/E384 image, 2 GiB expert budget,
chunk64 and adaptation every 16 selections were replayed with `--compute gemv`
at 1/8/64 activation rows per selected expert. Each of the nine cases used
fresh processes in A/B/B/A order. The benchmark binary SHA256 was
`a39b92c06124c5ec63681f5c84786d07f9953e62c12a5bb974ae1e5ab9704a66`,
from base `2cd96ed77ed7` plus the uncommitted implementation. All 36 measured
legs completed with finite outputs and matching per-phase output digests
across policies. Twelve preceding small smoke legs covered both formats,
both policies and 1/7/64 rows. This remains a single-expert mechanism
diagnostic, using synthetic weights and the existing GEMV dispatch, without
model routing, attention, TP collectives or HTTP.

| Rows | Locality | Whole process wall (s), two runs | Fractional process wall (s), two runs | Whole total file reads (GiB), median | Fractional total file reads (GiB), median |
|---|---|---|---|---|---|
| 1 | High | 7.32, 6.41 | 11.40, 11.28 | 2.14 | 3.38 |
| 1 | Low | 16.79, 14.60 | 21.23, 18.24 | 6.26 | 7.26 |
| 1 | Changing | 9.83, 10.92 | 12.30, 11.28 | 3.90 | 4.65 |
| 8 | High | 7.55, 7.47 | 10.29, 11.30 | 2.13 | 3.38 |
| 8 | Low | 16.52, 16.94 | 19.61, 22.26 | 6.29 | 7.23 |
| 8 | Changing | 11.04, 10.99 | 13.11, 12.85 | 3.92 | 4.65 |
| 64 | High | 8.02, 7.07 | 7.69, 9.22 | 2.18 | 3.42 |
| 64 | Low | 16.02, 15.80 | 19.40, 17.55 | 6.32 | 7.27 |
| 64 | Changing | 12.66, 12.96 | 13.22, 13.16 | 3.88 | 4.63 |

Process wall includes initialization, CUDA/module startup, arithmetic and
per-selection readback/validation. Fractional initialization alone took
3.46–6.18 s. Whole caching had the lower median total wall in all nine short
cases; this does not establish a long-running serving winner. Fractional
residency often reduced later steady-phase time, while its initial population
and total read volume remained higher. Observed tails were mixed. For example,
the first 64 selections after the disjoint hot-set shift had:

| Rows | Whole selection p99 (ms), two runs | Fractional selection p99 (ms), two runs |
|---|---|---|
| 1 | 154.51, 180.43 | 192.91, 147.61 |
| 64 | 185.23, 146.66 | 169.87, 171.10 |

Selection latency includes adaptation, arithmetic and diagnostic output
validation. It is not inter-token latency. Reconstruction excluding read-ticket
wait still cost 1.18–2.37 ms per steady selection across these cases. Existing
kernel launch/synchronization wall averaged 0.053–0.061 ms at one row,
0.217–0.228 ms at eight, and 1.636–1.679 ms at 64. Thus complete reconstruction
is substantial relative to the expert arithmetic, especially at small batches.
Output readback/validation added roughly 0.024–0.031, 0.138–0.176 and
1.053–1.060 ms respectively, and is retained in reported total costs.

Every process ran in a swap-limited scope (`MemorySwapMax=0`); sampled VmSwap
was zero. Expert reservations stayed below 2 GiB. Separate activation/output
device buffers were 44,544 / 356,352 / 2,850,816 bytes for 1/8/64 rows.
Process peak RSS was 104–285 MiB and excludes CUDA allocations. Source-copy
reads on this node finished before the campaign began. Filesystem block read
bytes exceeded expert-file read bytes by 0–11.28 MiB per measured leg; these
device counters include other reads and do not provide expert-only physical
attribution. The measurements do not establish physical DRAM bandwidth,
multi-request concurrency, checkpoint quality or model throughput.

In the first one-row run of each short case, whole-policy residency ended at
0.84 / 1.05 / 1.45 GiB for high / low / changing locality, while the fractional
policy retained approximately 1.98 GiB throughout. The whole policy admits at
periodic adaptation and charges the same 64 MiB-per-adaptation traffic bound;
the short comparison includes its gradual admission. Equal reserved budgets
therefore do not imply equal useful resident bytes during these traces. A
separate, frozen longer replay uses 2,048 warmup selections and 512 subsequent
measurement selections per case, plus 1,024 warmup and 512 measurement
selections after a hot-set shift. It records pre-measurement occupancy and
keeps the original short-trace results unchanged.

That longer replay completed 12 A/B/B/A legs using the same arithmetic binary
and a new frozen seed-44 trace set. All phase output digests matched across
policies, all sampled swap was zero, and both policies had at least 98.9% of
the total expert reservation resident before every measurement window
(approximately 1.98 GiB; the remainder includes reconstruction and staging).
These are one-row post-warmup mechanism results, with 512 selections per cell:

| Window | Whole wall (s), two runs | Fractional wall (s), two runs | Whole selection p99 (ms), two runs | Fractional selection p99 (ms), two runs |
|---|---|---|---|---|
| High | 3.306, 3.383 | 3.658, 3.561 | 98.12, 96.95 | 76.04, 93.25 |
| Low | 17.885, 14.617 | 15.763, 16.080 | 158.36, 130.80 | 138.26, 131.79 |
| Changing, before shift | 3.188, 3.669 | 3.920, 2.487 | 85.47, 106.19 | 106.45, 60.15 |
| Changing, after shift and warmup | 2.704, 2.544 | 2.469, 2.719 | 80.04, 75.00 | 72.40, 81.21 |

Whole caching's full-hit rate was 91.8–92.0% in the high-locality measurement
window and 30.3–30.5% in the low-locality window. Fractional caching's was
91.0% and 25.4–27.0% respectively, with additional partial coverage. Median
file reads in the low-locality window were 7.68 GiB whole versus 7.25 GiB
fractional, including adaptation. Whole process totals including warmup and
startup were 18.78/18.79 s high, 91.88/89.82 s low, and 36.18/37.43 s changing;
fractional totals were 21.99/23.28, 101.21/96.91 and 41.88/28.68 s. The spread
and mixed tail results support no general policy winner. Full cache occupancy
also does not prove convergence of every decayed routing/cost estimate.

## Existing integration points

The inspected base is upstream master `2cd96ed77ed747e381937d0f498e391f65f8506d`.

| Component | Upstream behavior | Integration responsibility |
| --- | --- | --- |
| `src/models/dsv41/loader.cpp`, `build_moe` | Loads every routed expert's gate/up rows and down columns for this TP rank | Separate permanent backbone loading from expert descriptors and demand reads |
| `src/loaders/resident_stream.hpp` | A complete resident stack or one reusable whole-layer allocation | Add an explicit expert-paged loading plan; do not build all experts before freeing them |
| `src/models/dsv41/model.cpp`, `moe_view`, `enqueue_layer` | Binds a complete expert table before routing/execution | Split route discovery, residency acquisition, execution and release |
| `src/models/glm/moe_layer.cpp` | Device routing and segmentation in production; a host route round trip in the diagnostic path | Export compact selected-expert work, support bounded execution waves, and bind only acquired expert views |
| `src/models/glm/moe_layer.hpp`, `prepare_graph_table` | Graph tables hold stable pointers to a fully resident expert population | Add cache-generation leases and update tables only outside executions using them |
| `src/loaders/resident_image.{hpp,cpp}` | Verified, rank-specific whole-layer startup images, with aligned direct reads | Reuse layout/identity/I/O conventions for a separate indexed expert-chunk image |
| `src/engine/nvme_tier.hpp` | Prefix/KV I/O worker plus engine-owned CUDA transfers | Reuse safe transfer ownership and storage checks, not prefix eviction semantics |
| `src/engine/memory_plan.hpp` | Device and pinned allocation ledger | Replace full expert residency with the bounded expert subsystem's complete reservation |
| `src/serve/{cluster_config,fabric_serve,rank_metrics}.*` | Validated config, shared journal settings, per-rank measurements | Carry policy settings, effective byte budgets and rank-local statistics |

### Preserve the actual quantization

The existing DeepSeek checkpoint is **MXFP4**, not NVFP4. Its expert payload
contains E2M1 pairs and E8M0 scales per 32 values, without a tensor global
scale. `GlmFp4Matrix` also describes NVFP4: E2M1 pairs, E4M3 scales per 16
values, and the existing device global-scale convention. These formats must
remain distinct in the disk manifest, byte formulas, validation and kernels.
Do not convert MXFP4 to NVFP4 to make the target's name match a configuration.
An NVFP4 checkpoint requires its own verified model/binding source.

The public `nvidia/DeepSeek-V4.1-Flash-1017-NVFP4` revision
`3ae9c0d3fbbea5370bd06e679d22044db15883f1` is a separate, currently unsupported
frontend target. Its ModelOpt expert tensors use `.weight_scale`,
`.weight_scale_2` and `.input_scale`; their semantics must be bound and
verified against that release. Its Engram tables are also MXFP4, requiring
table loading/dequantization support beyond changing the expert scale group.
Range-read safetensors headers confirm the first main gate uses U8
`weight[2304,2560]`, E4M3 `weight_scale[2304,320]` and scalar F32
`weight_scale_2`/`input_scale`. The first draft gate instead remains I8
`weight[2304,2560]` with E8M0 `scale[2304,160]`: one global FP4 format switch
would misinterpret the draft. An Engram row has 128 U8 packed-code bytes plus
eight E8M0 scales for 256 values.

A range-read audit of all 40 main-expert shard headers and their scalar stripes
verified all 46,080 main weight globals and positive finite calibrated input
scales. Weight globals are powers of two from 2^-14 through 2^-8. ModelOpt's
dequantization multiplies the E4M3 block scale by `weight_scale_2`; DGPP's
existing NVFP4 kernels divide the completed dot by a descriptor global.
`nvfp4_scales.hpp` translates only exactly invertible positive normal powers
of two, refusing factors that would require a rounded reciprocal. The image
builder now preserves the U8 codes and E4M3 scales, writes this exact descriptor
global, and hashes the original global and input-scale bytes. Input scales
are validated metadata; the reconstruction path still uses BF16 activations,
not calibrated W4A4 activation quantization. NVFP4/MXFP4 mixed draft paging is
refused. The new host image fixtures cover TP slices, image reuse, changed
activation metadata and invalid global factors.

The Engram prerequisite preserves packed source rows through mmap and pinned
staging. Its new decoder places BF16 bits directly, including E8M0 byte zero,
signed zero, overflow and NaN scales. Its GPU oracle crosses all 16 E2M1 codes
with all 256 E8M0 bytes, and model-layer fixtures exercise the host gather,
hashes, TP row selection, projection and gate. The six kernel tests and two
Engram-layer tests passed on GB10 after the frozen measurement chain released
the fabric. All five separately staged prerequisite binaries passed; that build
predates the additional main/draft binding fixture, which passed in the newer
loader/image qualification described below.
The expected-tensor table now describes the NVFP4 main expert quadruples while
retaining MXFP4 draft pairs, with exact scalar metadata shapes and byte counts.
Checkpoint configuration recognition now validates the audited ModelOpt producer,
main-layer population, NVFP4 group size, calibration declarations and MXFP4
Engram layer list. Six config tests pass ASan/UBSan (the optional local-checkpoint
test skips); the public revision's actual config also parses under those
sanitizers. The MXFP4 draft group remains independent of the main expert format.
The native loader now preserves NVFP4 payload/block-scale slices, installs the
exact reciprocal global descriptor, validates input metadata and accounts for
both scalar reads on every TP rank. A cross-compiled fixture checks code/scale
rows and columns, descriptor factors, allocation/source-byte plans and unchanged
MXFP4 drafts at TP1/2/4. This newer loader/config build is staged separately on
both nodes; its supervised GPU loader and seven-test image suites passed after
the frozen serving comparison completed. The earlier packed Engram prerequisite
build also passed its six kernel tests, loader suite, two layer tests, MXFP4
paged-model fixtures and image suite.
The newer model build selects native NVFP4 cache geometry and contains
prefill/decode parity fixtures comparing both policies against native resident
execution, plus a preallocation mixed-draft refusal fixture. It preserves the
same bounded-budget and execution-thread handoff checks used by MXFP4. This
build is staged separately; its six paged-model GPU tests passed, including
native NVFP4 prefill/decode parity under both policies and mixed-draft refusal.
The authorized full NVFP4 checkpoint download started after that model test
queue completed, avoiding I/O interference with the frozen measurements and
GPU qualification. The full-checkpoint NVFP4 whole-layer streaming control has
completed; both policies match it byte-exactly. These completed runs use the
earlier frozen reconstruction build, and do not qualify the later optimized
execution path on that checkpoint.
The supervised full-checkpoint gate started after the download, GPU fixture
qualifications and MXFP4 TP2 numerical gate. It uses the frozen forward-check
binary, a separate checkpoint symlink view on the NAS and an Engram sidecar
derived and checked from that view's tokenizer/configuration. It leaves the
Hugging Face snapshot unchanged. Three world-one runs use the same seven tokens
and seed 43: a whole-layer streaming control, then each policy with a 32 GiB
expert budget and chunk width 64. The checker requires byte-exact finite states,
routes and complete logits across all 40 main layers; execution receipts require
the frozen binary identity and zero swap. GPU fixtures do not substitute for
this gate. The expert image uses explicit buffered NAS I/O, so the gate provides
no NVMe timing, physical-read or bandwidth evidence. Predecessor failures stop
the queue for diagnosis; live workers are observed without automatic replay.
The control completed all 40 layers on seven tokens with zero swap. Its saved
state and complete-vocabulary logit dumps passed finite/shape validation and
their copied artifact hashes match the execution receipt. This validates the
reference artifacts. The whole-expert leg subsequently completed all 40 layers
and seven tokens with zero swap; its states, routes and complete-vocabulary
logits match the native control byte-exactly. Both independently copied dump
hashes equal the reference hashes. Its 308,065,079,296-byte expert image is
published for reuse. The fractional leg also completed with zero sampled swap;
its independently copied states, routes and complete-vocabulary logits match
the control byte-exactly. All three legs are complete. This qualifies the
frozen reconstruction/periodic-adaptation build only, not the later demand
retention, native execution or GPU segmentation revisions.
NVFP4 GEMV dispatch now includes down widths 2304/1152/576 at TP1/2/4.
Actual-shape oracle tests at N5120 with 1/3/8 rows, both epilogues and row-count
invariance pass on GB10, along with all 12 FP4 GEMV tests. Full-checkpoint
numerical gates and frontend integration remain required before claiming this checkpoint runs. The small NVFP4 backend
fixtures and replay smoke do not qualify that frontend.

The current placement is tensor parallel: **every rank holds an intermediate
slice of every expert**, rather than a subset of whole experts. Preserve that
ownership. For the release's intermediate width 2304, TP=4 owns 576 columns
per expert. See [the checkpoint budget](checkpoint_budget_dsv41.md).

## Shared expert store

Use an immutable `ExpertKey` containing model identity, main/draft layer,
expert, TP rank/world and quantization/layout version. `ChunkKey` adds the
intermediate interval. The manager spans all layers and enabled DSpark stages;
384 experts in one layer are not the whole model's cache population.

Partition a SwiGLU expert by aligned intermediate intervals `J`:

```text
g_J = gate[J, :] x
u_J = up[J, :] x
a_J = SwiGLU(g_J, u_J), using the model's existing clamps and roundings
y   = sum_J down[:, J] a_J
```

Store the matching gate rows, up rows, down columns and every corresponding
scale in one independently addressable chunk record. The down slice is
strided in a row-major checkpoint: copying an arbitrary prefix of the source
file is incorrect. Pack it once into a contiguous rank-local record. NVFP4
tensor globals retain the source tensor's values, never recomputed per chunk.

The implemented 64-wide chunk respects both scale grids and divides 576.
The current GEMV compiled sets accept K=64 in both formats, but that alone
does not qualify direct fractional execution: the down-contribution reduction
order, scaling, roundings and dispatch still need numerical/performance gates.
Do not pad intermediate activations into a different expert operation silently.
A different TP width needs a validated aligned tail, or an explicit startup
rejection.

An MXFP4 H=5120, C=64 triple has 522,240 useful weight/scale bytes before
allocation and I/O padding. A TP=4 expert has nine such chunks. A requested
fraction therefore resolves to a discrete count; 40% is not exactly
representable for this geometry. Publish requested and actual fractions.

Build a separate expert image with aligned records, per-record checksums,
tensor geometry, exact source identity, rank/world and a versioned index.
Publish only completely written, synced records; a temporary image becomes
visible atomically. Validate checksums before making weights executable.
Image generation uses bounded staging and never constructs the resident
model. Report source conversion time, bytes, scratch and extra disk space.
Reuse `ResidentImage`'s conventions, not its per-layer ABI or sampled identity
as a substitute for complete record integrity.

The file worker uses bounded asynchronous `pread` jobs into a preallocated
aligned staging ring. Support direct I/O as the controlled benchmark path;
buffered fallback is explicit and reported. Demand and speculative prefetch
share one queue with demand priority, bounded depth and bounded speculative
bytes. Never silently retry failed direct I/O through a different mode.

## Memory and lifetime invariants

`engine.expert_streaming.budget_bytes` is a per-rank **total expert-weight subsystem
budget**. It includes allocated cache capacity (including free reserved
slots), reconstructed expert scratch, in-flight destination slots, pinned
weight staging and record padding. Report the device/pinned split and
useful/padded split. Charge each physical allocation exactly once; in-flight
bytes already in a reserved slot are a state breakdown, not a second charge.

Both policies reserve the same execution and staging capacities before
assigning the remaining bytes to persistent residency. This prevents
fractional caching from gaining unreported temporary full experts. Backbone,
shared experts, routers, KV/prefix cache, Engram, activations, metadata, CUDA
context, collectives and other runtime allocations are separate ledger items.
On GB10, pinned and device memory consume the same physical RAM; the process
and node totals must also fit with explicit headroom and zero swap.

For the validation checkpoints, summing the actual sizes of all 48 indexed
weight shards gives 510,296,708,312 bytes (475.251 GiB) for MXFP4 and
428,986,460,424 bytes (399.525 GiB) for NVFP4. These are complete checkpoint
file sizes, not expert-cache requirements. The prepared NVFP4 TP=1,
chunk=64, main-model expert image is 308,065,079,296 bytes (286.908 GiB),
including its index and record padding. The configured TP=2 comparison
budget is 96 GiB per rank, or 192 GiB combined, including scratch and
staging. It is a configured subsystem limit rather than the hardware's
physical memory capacity. Comparing that budget with a TP=1 image is only
a sizing illustration; rank-specific image geometry and fixed reservations
determine actual cache coverage. The completed serving performance campaign
uses MXFP4. NVFP4 now has current optimized TP2 numerical parity and its own
local-SSD serving campaign in progress; its performance remains unreported.

At startup, reject a budget below the progress reservation. Strategy B also
requires at least its rounded-up minimum chunk fraction in **every** eligible
expert. Never silently fall back to whole-expert caching or zero coverage.
At full residency, bypass reconstruction and bind the complete resident
expert directly where its layout permits it.

Record states are absent, loading, ready, leased and failed. Acquiring a
loading record joins its completion instead of issuing another read. Leases
pin immutable contents through the last GPU reader's completion event.
Generation numbers prevent late I/O completions from publishing into reused
slots. Evict only unleased ready contents. If all capacity is pinned, execute
a bounded work wave or apply backpressure; never allocate an emergency copy.

Cancellation removes a waiter but does not invalidate shared I/O or a GPU
reader. Failed/short/corrupt reads never publish weights; wake every waiter
with the failure, reclaim after actual I/O completion, and propagate a
coordinated fabric failure before any rank enters the next collective.
Shutdown drains or joins the worker before freeing its buffers.

## Policy A: whole expert cache

Treat a rank's complete expert triple as the admission and eviction unit.
Records may be read in chunks internally, but the expert is not executable
or counted as resident until every chunk is ready. The current baseline reconstructs cache hits too; direct binding of stable
complete views remains an optimization requiring qualification. A batch's union of selected experts is deduplicated
before acquisition and split into bounded waves if it exceeds capacity.

Use one shared access estimator: request/row multiplicity, exponentially
decayed frequency, last use and measured load cost. A candidate's retained
benefit is predicted avoided loading time per reserved byte. Use a lazy
priority structure with deterministic ties and a probation/protected split
to resist scans; recency and decay let stale hot experts lose protection.
Offer LRU as a diagnostic comparator, not a separate storage implementation.

## Policy B: adaptive fractional cache

At initialization allocate a positive number of complete chunks to every
eligible expert. Derive the greatest feasible common fraction from the
remaining cache bytes, then allocate the rounding remainder deterministically
without exceeding the byte cap. An explicitly requested initial fraction
must fit; report any unassigned capacity and its cause.

Reconsider allocation after a configurable number of routed work items.
The shared estimator predicts the bytes and service time saved by each
additional chunk, including reuse across concurrent rows, measured queue
pressure and promotion transfer cost. Maintain a strict rounded minimum
coverage, give hot experts additional chunks up to 100%, and reclaim chunks
from colder experts above that minimum. Demote stale promotions after decay.
Already fetched demanded chunks may be retained when their budgeted gain
justifies it; unused free staging is not an invisible extra cache.

For equal-sized experts with budget fraction `b` and `h` full promotions
among `N` experts, the remaining common fraction is
`(b*N - h)/(N - h)`, before chunk rounding and execution reservations.
For N=384, b=0.5, h=48, this is 3/7. Apply the formula across the actual
budgeted population, not independently to each layer with the full budget.

Frequency alone with a linear byte-saving objective favors concentration.
Broad coverage must therefore be an explicit constraint, with additional
tail-latency and churn penalties, rather than an assumed emergent property.
Require predicted saved service time over a horizon to exceed loading cost
and hysteresis margin; use minimum residence/cooldown intervals and a hard
adaptation-byte allowance. Suppress speculative promotion under saturated
I/O or memory pressure while preserving demand progress and minimum coverage.
Log decisions and actual transfers so oscillation can be measured.

## Execution and graph integration

The first implementation reconstructs a missing expert from its resident
chunks plus only the missing disk records into bounded scratch. Both
policies then use the same existing gate/up, SwiGLU, down and routing-weight
accumulation kernels. Reconstruction must reproduce the complete payloads,
scales and globals byte for byte, enabling a bitwise numerical baseline.
Measure resident-chunk copies, reconstruction kernel/transfer time, scratch
occupancy and total memory traffic separately from storage reads.

This baseline does not establish compute/I/O overlap for Strategy B. Add a
second execution stage that computes resident gate/up intervals while file
reads progress, then the remaining intervals. Preserve the activation's exact
rounding at each element. For direct chunk down-projection, add kernels that
accept explicit intermediate origin/width, scale/payload strides and FP32
accumulation state. Reduce chunks in a deterministic order and apply router
weights and the shared expert once. Group rows selecting the same expert and
compatible chunk geometry to retain Tensor Core weight reuse.

Chunk sums are mathematically equivalent but not necessarily bitwise equal
to an unsplit FP32 reduction. Predeclare numerical tolerances and logit/route
gates before GPU validation; compare also against a fixed-chunk arithmetic
control at full residency. Do not change tolerances after a failure. NVFP4
W4A4's dynamic per-row activation global can depend on the **entire**
intermediate activation; independently requantizing each chunk changes the
operation. Keep existing W4A16 arithmetic first; qualify W4A4 separately
with unchanged full-row scale semantics and its existing opt-in quality gate.

Split the MoE API into route/compact work, acquire/lease, execute waves and
finalize. Avoid copying all expert scores to the host as the diagnostic
`enqueue_host` currently does. Copy compact route IDs/counts only, retaining
weights on device. Record this synchronization cost. Cache hits eventually
use a device residency lookup; a miss boundary coordinates host I/O.

Do not globally disable TP or substitute the old whole-layer loader. Add
an expert-paged execution mode orthogonal to `LoaderResidency` and
`decode_graph`; keep backbone weights resident. Begin with eager host
orchestration of identical kernel work for A and B, including on a fabric,
then capture stable attention/router and expert-execution subgraphs around
the miss boundary. Whole-model graph replay cannot invoke host disk I/O or
dereference evictable static tables. Stable table storage does not make its
pointed-to weights stable: lease slots and table generations through replay.

All CUDA operations remain engine-owned. The existing NVMe tier documents
fabric timeouts from auxiliary-stream copies beside device collective spin
loops. Use safe boundaries and the model stream for transfers initially;
enable a transfer stream only after an explicit multi-rank overlap gate.
Keep collective ordering and failure decisions coherent across ranks;
rank-local eviction choices and file-worker completion timing may differ.
Allocate all CUDA objects, scratch and event pools before the fabric starts.

Include decode, exact/bounded prefill, mixed rows, all enabled DSpark stages,
speculative verification/rollback, prefix reuse, cancellation and concurrent
requests in the integration gates. MTP settings must match between A and B.

## Runtime configuration

An optional `engine.expert_streaming` object selects either policy at launch.
Absent this object, existing behavior is unchanged. Effective settings are
strictly parsed, included in the configuration digest and journaled to peers.
The image directory must be usable on each rank's node. For example, merge
these engine settings into an otherwise complete deployment configuration:

```json
{
  "engine": {
    "decode_graph": false,
    "prefill_fold_overlap": false,
    "expert_streaming": {
      "policy": "adaptive_fractional_cache",
      "budget_bytes": 34359738368,
      "image_directory": "/var/cache/dgpp/expert-images",
      "chunk_intermediate": 64,
      "minimum_fraction": 0.05,
      "adapt_every_routes": 1024,
      "adaptation_bytes": 67108864,
      "execution_slots": 1,
      "io_buffers": 2,
      "io_workers": 1,
      "queue_depth": 8,
      "prefetch_bytes": 8388608,
      "io_mode": "direct"
    }
  }
}
```

The example budget is illustrative; preflight rejects budgets that cannot hold
the model's rounded fractional floor, execution slots and staging together.
For whole-expert caching change `policy` to `whole_expert_cache`; omit
`initial_fraction` if supplied. No in-process live policy switching is exposed.

| Field | Meaning |
| --- | --- |
| `policy`, `budget_bytes`, `image_directory` | Required policy, complete per-rank subsystem budget and image location |
| `chunk_intermediate` | Intermediate interval, default 64; must align to 32 and pass TP geometry validation |
| `execution_slots` | Complete reconstructed-expert slots, default 1 |
| `io_buffers`, `io_workers`, `queue_depth` | Defaults 2, 1, 8; require workers <= buffers <= depth |
| `io_mode` | `direct` (default) or explicit `buffered`; no silent fallback |
| `prefetch_bytes` | Maximum queued/active speculative record bytes; default 0 disables prefetch |
| `initial_fraction` | Optional initial fractional coverage; otherwise derived from remaining cache capacity |
| `minimum_fraction` | Lower coverage bound in [0,1], default 0.05; positive values round up to chunks. Zero allows cold experts to become absent after universal initial population. |
| `adapt_every_routes` | Reconsider allocation after this many routed selections, default 1024 |
| `decay_routes`, `horizon_routes` | Routing half-life and benefit horizon, defaults 1024 and 8192 |
| `promotion_margin`, `cooldown_routes` | Hysteresis and minimum change interval, defaults 0.25 and 128 |
| `adaptation_bytes` | Maximum newly retained bytes per periodic adaptation; default unlimited. Both policies retain eligible chunks from existing demand reads separately. |

The shared planner uses decayed frequency and measured unprefetched queue/file
wait cost. The model additionally installs a Linux resource monitor for background
work, sampled at most once per second when adaptation/prefetch is attempted.
Node MemAvailable at or below 4 GiB pauses new background work; at or above 6 GiB
resumes it. Image-device busy time at or above 90% pauses background work; at or
below 70% resumes it. These are fixed implementation defaults, not JSON settings.
Both signals have hysteresis; all demand reconstruction and whole-expert demand
admission continue, and mandatory fractional initialization is unaffected.
Already submitted speculative reads can finish within the existing bounded pool.
The gate neither changes fractional floors nor frees the preallocated CUDA slabs.

The storage signal is Linux block-stat I/O-active milliseconds divided by elapsed
monotonic milliseconds. It includes unrelated device traffic and is a coarse busy
indicator, not physical bandwidth, queue saturation or expert-only attribution.
Network filesystems without a block device report that signal as unknown. Missing,
malformed or reset sources are unknown and clear the corresponding pause rather
than indefinitely blocking background work. Metrics expose source validity,
sample sequence, latest values, pressure flags and deferred operation counts.
Two deterministic monitor tests cover sampling, hysteresis, overflow, unknown and
reset behavior; an injected-pressure cache test proves demand progress, byte-exact
reconstruction, continued whole admission and resumed background loading under
both policies. The 14-test cache suite passes ASan/UBSan and ThreadSanitizer.
This resource-feedback code cross-compiles with the server,
forward-check tool and GPU model fixture. Native configuration (28), serving
(125), scheduler (152) and fabric/metrics (21) tests pass, including a metric
check distinguishing measured zero memory availability from unknown device
utilization. Subsequent shared execution revisions passed the current 39 host
and seven GB10 fixture checks, full MXFP4 TP2 numerical parity, and the
completed local-SSD MXFP4 serving comparison above. Optimized NVFP4 numerical
parity and the completed same-build policy comparison are summarized above.
Later b12x and loader changes have their own shorter serving comparisons;
the 128-response result does not establish their throughput across all
workloads. Older NVFP4 results retain their own frozen build identities.

Dynamic slab resizing, an LRU comparator, a scan-resistant probation/protected
split and direct chunk execution remain design extensions; they are not accepted
configuration settings.

## Telemetry and fair comparison

The target measurement set for both policies is: routed work
and unique selected experts; complete expert hits, partial hits and requested
chunk hits; useful missing bytes, submitted/returned direct and buffered bytes,
aligned/padding bytes and completed file operations; coalesced waiters;
demand/prefetch/adaptation queue waits; I/O completion latency; CUDA transfer,
reconstruction and expert execution time; eviction/promotion/demotion counts
and bytes; active leases, cache/pinned/scratch capacity and high watermarks;
failed reads, cancellations and backpressure. Define cache-hit denominators
at first lookup, before warming or joining a pending read.

Implemented `dgpp_rank_expert_*` metrics expose policy/I/O mode, request and
chunk-hit denominators, cache transitions/bytes, read wait, file bytes/calls/
failures, queue/coalescing/backpressure, staging high watermarks, reservations,
resident chunks and poisoned state. Scheduler JSON includes a compact subset.
Read wait includes queueing; it is not device-only latency. Successful
reconstruction wall time excludes file-ticket waits but includes CPU work and
completion fences; it is not GPU-only time. Useful cached/staged copy bytes and
execution-slot zeroing bytes have separate counters, excluding disk padding.
These are byte volumes, not measured hardware bandwidth. Transfer/kernel
histograms, detailed reconstruction time, physical device attribution, memory
bandwidth and matched campaign results remain outstanding.

File bytes are not proof of physical NVMe traffic. For direct runs report
successful aligned reads as filesystem-bypassing-page-cache bytes and collect
the resolved block device's read bytes/operations concurrently. Device counts
are physical observations at that layer, include competing Engram/KV I/O, and
need an idle-device control and separate attribution. Buffered runs report
logical file bytes plus device bytes; do not infer exact page-cache hits by
subtracting aggregate counters when attribution is unavailable. Mark unknown
attribution explicitly. Keep Engram and prefix-cache I/O separate in engine
counters and avoid adding overlapping memory or timing scopes.

### Hardware bandwidth diagnostic

A bounded capability probe ran separately on idle Spark-2 with the frozen
demand-admission replay binary. Nsight Systems captured actual H5120/I2304
MXFP4 expert reconstruction and arithmetic, eight rows per expert, a 64 MiB
budget, four synthetic experts and 320 selections of expert zero. GPU metrics
were sampled at 1 kHz using the supported `gb20b` set; CUDA tracing captured
2,240 kernel records. All 19 exported metric names cover clocks, compute and
copy activity, with no DRAM byte/bandwidth metric. The device's exported
`memoryBandwidth` attribute is theoretical capacity, not measured traffic.

Nsight Compute 2025.3.1 queried 3,476 base metrics on this GB20B/SM121 device.
The usual `dram__bytes_read` and `dram__bytes_write` metrics are absent. The
exposed metric families include L1/L2, SM, graphics and system counters; the
standard Linux perf PMU inventory exposes CPU/SPE/SMMU events, without a DRAM
controller PMU. This does not establish that every vendor instrumentation
option is unavailable, but neither installed profiler presently provides the
required direct DRAM-byte observation.

A separate one-pass Nsight Compute probe successfully collected one MXFP4 gate
kernel with uncontrolled caches and clocks: 464,849 read sectors and 9,082 write
sectors at L2 in 75,616 ns. At 32 bytes per sector this is 15,165,792 L2 bytes,
or 200.56 decimal GB/s of L2 read-plus-write traffic during that one kernel.
It excludes other kernels and reconstruction copies; it is not whole-expert,
whole-model, peak-device or DRAM bandwidth. Neither these counter probes nor
their instrumented wall times are included in the uninstrumented policy timing
tables. Both probes completed in zero-swap scopes with the expected frozen
binary identity. Physical DRAM bandwidth remains **UNKNOWN**; no theoretical
bandwidth, L2-sector rate or effective reconstruction-copy rate substitutes for
that missing measurement.

A subsequent read-only OS/firmware inventory on Spark-2 found no `tegrastats`
tool or devfreq devices. Exposed PMUs cover CPU cores, software/SPE and SMMU
groups, not a DRAM byte counter. Firmware advertises `DRAM8901:00`, but that
device has no bound driver, telemetry files or matching installed module alias.
This closes the observed platform-device lead on the current kernel; it does
not establish that future firmware/drivers cannot expose such a counter. No
driver was loaded and no device registers were accessed during this inventory.

Follow [the benchmark method](benchmarks.md). The campaign runner accepts
`--compare-experts` to produce paired serving campaigns from one deployment,
changing only the policy. With paging enabled, each leg captures per-rank
Prometheus snapshots and the image filesystem's Linux block/partition read
counters before warmup and after the campaign. Configure `ports.metrics` for
peer telemetry. NFS/unknown devices and unavailable metrics remain explicitly
unknown; device counters include competing I/O and are not expert-only bytes. The default two pairs run A/B/B/A; every leg must
pass the normal zero-swap/binary and operation-stream checks. Source identity
includes current tracked and untracked source bytes, so uncommitted code is
not represented by HEAD alone; changing source or binary invalidates the
comparison. Omit `initial_fraction` from this shared template, because whole
caching does not accept it. For example:

```sh
python3 scripts/serve_campaign.py run --config deploy/cluster_local_paging.json \
  --bin build-spark-cross/dgpp-serve --compare-experts --comparison-repeats 2 \
  --phases serving,prefill,transcripts --out benchmarks/results/expert-policy-pairs
```

This is the regular warmed serving campaign. It does not implement synthetic
locality replay, expert-only physical-device attribution, a cold-cache mechanism benchmark
or automatic production restoration; those remain campaign gates. Run from
the configured rank-0 checkout, after numerical qualification and idle-fabric
checks; restore any prior production world once after the full comparison.
Record source revision, binary and manifest hashes, exact checkpoint,
quantization, TP world, clocks, budgets (including scratch), effective
fractions, MTP, prefill mode, batch sizes, concurrency and I/O mode.

Run two complementary layers of evidence:

1. Identical saved route schedules exercise high locality, broad low locality
   and an abrupt hot-set change. Replay real expert kernels and real I/O;
   distinguish this mechanism benchmark from end-to-end generation. Include
   concurrent reuse, variable expert costs, capacity overflow and cold start.
2. Matched prompts and sampling seeds through the real server at multiple
   concurrency/batch levels. Use recorded routes to explain differences;
   do not force synthetic routes in quality or throughput claims. A changed
   numerical path must pass its quality gate before performance comparisons.

Report decode-only tokens/s by concurrency separately from cold prefill
seconds and the labelled full service wall rate. Also report aggregate and
per-request output rates, TTFT and inter-token p50/p95/p99, physical/logical
I/O bytes and operations, expert/chunk hit rates, adaptation/eviction rates,
peak node/device/pinned memory and measured memory bandwidth. Include startup
preparation and cold baseline population time separately, especially B's
mandatory initial reads. Disclose any page-cache warmth; do not clear global
page cache on a shared host. Compare cold and steady state, repeat matched
A/B order, and include recovery time and extra traffic after locality shifts.

Use the service's TTFT and inter-token histograms for server timing. With MTP
disabled, each token-producing pass contributes one token; validate histogram
count deltas against requests and generated tokens minus first tokens. Report
nearest-rank p50/p95/p99 as bucket brackets without assuming a within-bucket
distribution, and retain right-censoring above the largest finite bucket.
The maintained `timed_load.py --server-timings` client now captures raw
Prometheus snapshots around each measured phase, outside the timed request
interval, and writes per-phase TTFT and inter-token p50/p95/p99 bucket brackets.
Paging campaigns enable it automatically. TTFT combines the cache-hit and
cache-miss series; its count must equal completed requests. Inter-token counts
must equal generated tokens minus first tokens, so this mode requires one token
per decode pass, as in the eager, no-MTP streaming template. Missing series,
changed bounds, counter resets, nonmonotonic deltas and unmatched counts fail
the phase before publication; zero observations have null quantiles and overflow
is explicitly right-censored. Raw snapshots retain expert counters for later
per-phase cache and I/O analysis. Sixteen client tests pass, including an HTTP/SSE
fixture with warmup and C1/C4 traffic; optional instrumentation preserves the
request workload and decode-rate calculation. This qualifies client collection,
not new full-model performance results.
Offline replay also reproduced all 64 TTFT/inter-token percentile checks from
the saved four-boot real-model diagnostic snapshots; no GPUs were launched.

The decode-step/inter-token histogram preserves its original bounds through
10 s and adds finite bounds through 640 s for storage-backed stalls. The
Prometheus unit suite, including long-stall/overflow observations, and all
125 host serving tests passed with this extension.
Client SSE update gaps are a separate delivery metric: HTTP buffering can put
several tokens in one update. A streaming-client fixture reconciled all expected
TTFT/ITL observations at C1 and C2 before the full-model campaign was queued.

The fabric is otherwise idle for measured rows, every rank must have zero
swap, operation streams must match for bitwise controls, cold probes must
have zero cached prompt tokens, and production is restored at the end.
Generated measurements/profiles remain untracked. No winner is declared
from a host policy simulator or an unmeasured hardware path.

## Real-checkpoint parity tool

The experimental [two-node MXFP4 streaming template](../deploy/cluster_deepseek-v4.1-flash-1017_mxfp4-streaming_w2.example.json)
provides the 96 GiB expert budget, eager execution, two slots and disabled
draft/prefix caching. Its peer metrics listener uses port 18082 so the campaign
can retain counters from both ranks. Its 1M-token KV pool supports the maintained campaign's
context probes; it is deliberately larger than the 4096-token diagnostic pool.
The experimental [NVFP4 counterpart](../deploy/cluster_deepseek-v4.1-flash-1017_nvfp4-streaming_w2.example.json)
uses the same engine and residency settings, with the pinned NVIDIA model and
a separate NVFP4 expert-image directory. Its focused serving comparison is
prepared to follow the current full-checkpoint numerical comparison; no NVFP4
serving performance is claimed from preparing this template. All 48 NVFP4
shards, including Engram, are staged on local SSD on both nodes before either
comparison runs.
The template was resolved through the normal launcher configuration code for
the two validation nodes, then checked with `--memory-plan` on each rank without
launching a model or fabric world. Both policies produce the same 106.15 GiB
plan per rank plus 4 GiB required headroom. The whole policy passed on both
nodes at 113.91/120.25 GiB available. The later fractional check passed on rank
one but was correctly refused on rank zero when available memory had fallen
to 108.66 GiB during the active checkpoint download. That failure remains
recorded; it reflects changing node availability, not a larger fractional
allocation. Those old pending queues were held before launch after the policy
and execution scope changed. The revised comparison uses a new source/binary
identity and fresh all-rank preflight after current MXFP4 numerical qualification; it preserves both policies' 96 GiB budgets and will stop
on refusal without changing the configuration or launching a campaign.
Neither these checks nor the short-KV diagnostic establish full-context
execution, measured peak memory or standard-campaign feasibility/performance.
The standard campaign remains unrun. The full campaign queue was superseded
before any serving launch at the user's request to focus on the original
streaming comparison. The replacement completed full MXFP4 TP2 parity and 33 of 36 matched locality
replay legs. At the user's request, the remaining replay legs were held and
focused A/B/B/A full-model serving started directly after fresh all-rank memory
checks. The old waiting controller was intentionally stopped before launch;
its interrupted receipt does not indicate a policy or numerical failure. It uses the zero floor and immediate demand retention, direct
native resident execution and GPU segmentation. That focused local-SSD
comparison has now completed all four boots; its results are reported above.
The focused warmup consists
of three identical 32-token request windows with all-rank counter snapshots;
these fixed windows expose continuing adaptation and do not establish
convergence by themselves. The replay preserves the earlier expert selection
sequences, splits warm/shift-warm phases into 64-selection windows, and uses
the shipped 1024-route adaptation cadence with floor zero. Earlier replay
measurements used a positive floor and adaptation every 16 selections, so
comparisons across revisions are not a controlled execution-only speedup.
The preserved replay receipt covers all four legs for high and low locality at
1/8/64 rows, and changing locality at 1/8 rows. All per-phase output digests
match within each complete group, with zero sampled swap and swap disabled.
High-locality measurement windows have approximately 91–92% complete native
hits under both policies; the trace includes a cold tail, so these are not
pure hot-set tests. Median total file reads, including initialization and
warmup, are about 3.90–3.91 GiB for whole-expert caching versus 5.50–5.51 GiB
for fractional residency. Broad low-locality measurement windows have about
28–34% full hits and 5.96–6.28 GiB whole-expert versus 6.29–6.43 GiB fractional
file reads. These small repeated synthetic groups establish neither a general
winner nor model tokens/sec. The incomplete changing-locality group at 64 rows
is excluded from policy comparisons.
The completed comparison used the exact runtime binary and current template,
including the peer metrics port, with the same preallocation plan on both
nodes. No budget retuning or failed-leg replay was used.

The explicit `--streaming-focused --phases serving,prefill` profile uses the
maintained clients with all five prompt classes at C1/C2/C4, two repetitions
and a 64-token output cap. This is 70 serving requests and at most 4,480 output
tokens per boot, plus cold prefill probes at 512 and 2,048 tokens twice each.
The model, binary, template, 96 GiB expert budget per rank and A/B/B/A ordering
remain fixed. C4 measures queueing above the two active request slots, rather
than four simultaneous decode rows. Standard warmup and pre/post expert
instrumentation remain enabled; fixed warmup does not prove convergence.
The profile excludes quality, long-context, API and transcript suites. The
completed controlled-locality replay remains a separate mechanism experiment.
Profile parameters are recorded in each status and comparison receipt, and
the standard benchmark-row exporter refuses focused results. At an illustrative
2–7 aggregate tokens/s, the serving output cap represents about 11–37 minutes
per boot before startup, prefill and queueing; it is not a completion forecast.
Focused results cannot establish quality, long-context or standard-suite
performance. The default campaign workload is unchanged.

Campaign memory sampling is prepared on both nodes at 0.5-second intervals,
starting before serving launch and ending after teardown. It records node
MemAvailable, identity-checked process RSS/HWM/swap and cgroup current/peak/swap
counters, alongside image-device read counters. Startup is included in this
planned observation window, unlike the completed short diagnostic's sampler.
Offline identity/accounting checks and one-sample readiness checks on both
nodes pass; these are not measured campaign peaks. Node sampled extrema and
process/cgroup high watermarks remain separate quantities and must not be
summed. Device-wide traffic is still not expert-only attribution.

The runner accepts repeatable `--phase-timeout PHASE=SECONDS` overrides and
records every effective deadline in the campaign status, comparison plan and
individual command receipts. Both policies receive the same overrides; omitted
values preserve the previous defaults. Deadlines apply to each client command
(including each transcript/isolation command), not to a whole boot. Invalid,
nonpositive, duplicate or unknown overrides fail before boot. For example,
`--phase-timeout serving=43200` allows twelve hours for each serving sweep
without changing its prompts, repetitions, token limits or concurrency levels.
This is timeout configuration, not evidence that the run will finish within it.

The C1/C2/C4/C8 serving sweep alone contains 225 requests with a 256-token cap,
or at most 57,600 generated tokens per boot and 230,400 across A/B/B/A. At an
illustrative aggregate rate of 2–7 tokens/s, generation alone would take
approximately 2.3–8 hours per boot, before prefill, context and quality phases.
Those rates came from the short C1/C2 diagnostic and do not predict C4/C8
performance or request completion lengths. The one-hour serving default is
therefore unsuitable as an assumed feasibility bound. The quality client's nonstreaming HTTP read has a separate 900-second default,
now configurable with the campaign's `--quality-request-timeout SECONDS`
(`serve_eval.py --request-timeout SECONDS` directly). The same finite positive
value applies to original replies and truncation retries, is recorded in both
campaign and evaluation settings, and is identical across policy legs. It does
not change the 2,048-token standard cap, 512-token extraction cap or 8,192-token
retry cap. Forty report/client tests pass, including configured/default HTTP
deadlines, unchanged retry caps and invalid-deadline rejection before network
or output. Context requests retain their separate 14,400-second deadline.
A longer subprocess deadline does not extend these socket deadlines. At the
diagnostic's fractional C2 aggregate rates, a pair of 2,048-token quality replies
can exceed 900 seconds; output lengths remain workload-dependent.

The full quality phase permits 164 HumanEval, 300 GSM8K and 100 extraction
items. Its standard caps sum to 1,001,472 output tokens per boot. If every item
were truncated and retried, retries permit another 4,620,288 tokens, for a
5,621,760-token cap envelope per boot. This is an upper bound, not a runtime
prediction or an assumption that all items reach their caps. It makes clear
that the short serving diagnostic cannot establish full-campaign duration or
justify a launch without assessing quality-phase cost as well.
Full-campaign launch
still requires the numerical and memory gates, idle fabric, and a performance
and duration assessment covering every phase; increasing a deadline does not
satisfy those gates.

The image directory expands `~` on each rank and must resolve to its local SSD
image, with enough space if preparation is needed. The campaign's block-device
probe now expands that path too; previously it attempted to stat a literal
`~` path and reported unknown physical I/O. A regression checks path expansion
and block-sector conversion, and read-only probes resolve both validation
nodes' image paths to readable `nvme0n1p2` counters. These probes ran during
numerical qualification and are instrumentation checks, not measured I/O rows.
On the validation nodes,
the template paths now alias the existing verified TP2 images through symlinks;
header/index identities and file sizes match the frozen comparison, and no
payload was copied. This avoids attempting another 135 GiB image on rank zero's
approximately 22 GiB free SSD. The dry-run command validates the shared policy
comparison plan without allocating or launching:

```sh
python3 scripts/serve_campaign.py run \
  --config deploy/cluster_deepseek-v4.1-flash-1017_mxfp4-streaming_w2.example.json \
  --out benchmarks/results/expert-streaming --compare-experts --dry-run
```

The plan retains all standard phases and A/B/B/A order, changing only policy.
Removing `--dry-run` launches the maintained campaign after the required
numerical, memory and idle-fabric gates; it is not a bounded short diagnostic.

The newer server's memory refusal message also names
`engine.expert_streaming.budget_bytes` when paging is enabled. The message change
cross-compiles and was exercised with a deliberately oversized 128 GiB expert
budget in memory-plan-only mode: it correctly refused before allocation and
included the expert-budget hint. That negative check did not alter the deployment
template, the 96 GiB comparisons or any frozen validation binary.

`dsv41_forward_check` accepts `--expert-streaming-json FILE`, where FILE
contains only the `engine.expert_streaming` object. Without it the tool retains
the existing world-1 whole-layer streaming reference, allowing a full-model
comparison without fitting every expert at once. The paged runs keep the
backbone resident and use the supplied bounded expert budget.

Use identical prompt IDs, seed, layer limit and prefill mode for the reference,
whole and fractional runs. Existing `--out` dumps layer residuals, collapse
coefficients, routing IDs and index selections. Optional `--logits-out` writes
`DSV41LG1`, three native int32 values (rows, vocabulary origin, vocabulary
count), then the fp32 logits. Require byte equality for this reconstruction
baseline, including complete logits; preserve any failure rather than
adjusting a tolerance. Startup source hashing/image preparation is included
in the tool's reported wall time, so this is a correctness tool rather than
a serving performance benchmark. The CLI passed byte-identical states and
full logits on the tiny MXFP4 fixture against the layer-streaming reference
for both policies. A layer limit still prepares the configured model's complete
image population.

For a TP2/TP4 comparison, the tool accepts `--world`, `--rank`, `--peer`
(rank zero's rendezvous address), `--fabric-port` and
`--rendezvous-timeout-ms`. Launch one process per rank with identical token IDs,
seed and prefill mode, using unique output paths. First run without
`--expert-streaming-json` for the whole-layer streaming control at that same
world size; then run each policy. This isolates residency from TP reduction
order rather than requiring a TP result to equal a world-one result.

Collect each run into a directory containing `rank0.states`, `rank0.logits`,
and the corresponding files for every other rank. The offline gate is:

```sh
python3 scripts/dsv41_dump_check.py --reference CONTROL --candidate PAGED \
  --config CHECKPOINT/config.json --tokens 7 --world 2
```

It validates complete state/collapse/route/index captures, finite states and
logits, token/expert bounds, matching replicated captures across ranks, exact
per-rank equality to the control and complete nonoverlapping vocabulary slices.
Seven offline regression tests pass, including refusal of matching nonfinite
artifacts and incorrect shards. Process/binary identity, zero swap and fabric
isolation still require separate run receipts. The TP CLI cross-compiles; its
three GPU smoke worlds and all three full-checkpoint TP2 worlds pass exact dump
checks with zero swap on both ranks. A supervised
queue stages a tiny MXFP4 checkpoint on both nodes, runs streaming/whole/fractional
TP2 smoke worlds, then repeats those three numerical worlds on the full MXFP4
checkpoint with seven tokens, seed 43, exact prefill and a 32 GiB expert budget
for both paging policies. It requires zero process/cgroup swap, unchanged binary
identity and exact dump checks. This queue waits for the existing serving,
NVFP4 GPU-test and download chain to complete, keeping its I/O outside the
frozen performance measurements.

### Full MXFP4 checkpoint parity (2026-11-13)

The official `deepseek-ai/DeepSeek-V4.1-Flash-1017` revision
`08c8657f02291e11b6f6b63e0e16014f8e748518` passed the world-one comparison
over all 40 main layers, seven seeded random token IDs (seed 43), exact
prefill and complete logits. Every downloaded shard was verified against
its Hugging Face LFS SHA256. The first 46 shards were staged on local NVMe;
the two Engram shards remained on the NAS. DGPP's Engram sidecar was derived
from this config/tokenizer and independently rederived to check equality.
Both policies used a 32 GiB expert subsystem budget and 64-wide chunks.

The frozen forward binary SHA256 was
`090c49695d6a1b24957f6189dfc6091147ada6c74fc389a6a1f8b0a841f6fb6e`,
from base `2cd96ed77ed7` plus the implementation. Both policies matched the
whole-layer streaming control **byte for byte**, including all residual
streams, collapse coefficients, routed IDs, index selections and full logits.
All dumped floating-point values were finite. Every run was in a zero-swap
scope and every sampled process VmSwap was zero.

| Correctness leg | Total process wall (s) | Source/image work included |
|---|---|---|
| Whole-layer streaming reference | 282.243 | Layer reads during the walk |
| Whole-expert cache | 1654.222 | Source identity scan and first 270.03 GiB image build |
| Adaptive fractional cache | 379.881 | Source identity scan, image reuse and initial population |

These unequal preparation histories make the totals unsuitable for a policy
performance comparison. The matched serving campaign prepares the common TP2
images before its measured policy boots. Matching state-file SHA256 was
`56c1d8f1cda2c058c30b7a75178eedfdbd6c6b742467c09946d3c997d8ab3aee`;
matching full-logits SHA256 was
`ef3edb24bb7f551d483c35e6199b584465758aadeb293e009040d74192254fe1`.
This qualifies the reconstruction baseline against DGPP's existing numerical
path on these inputs. It is not a quality evaluation, long-context gate or
full-checkpoint multi-rank qualification.

The subsequent demand-admission build was requalified against those saved
independent reference files with the same inputs, 32 GiB budget and existing
image. Its forward binary SHA256 is
`90f784827305aa13118f959591e0bd45599d5352ebc6be87988b36b71d552d34`.
Both policies again produced finite, byte-identical states and complete
logits, with zero sampled swap and scope swap disabled. Whole/fractional
process wall was 324.366/387.986 s, including source scans and initial
population. Sampled minimum node MemAvailable was 77.142/77.333 GiB;
these are node-wide observations, not isolated CUDA-allocation measurements.
The common reference hashes above remained unchanged. This requalification
does not extend the prompt, TP or quality scope of the original control.

### Full-checkpoint TP2 serving diagnostic

The common preparation boot completed on two GB10 nodes with 32 GiB per-rank
expert budgets, 64-wide chunks, exact checkpoint revision as above, no MTP,
eager execution, bounded prefill, two request slots and a 4096-token KV pool.
Each rank's image is about 135 GiB. Rank zero reached readiness in 1121.7 s,
including its first image build; both ranks then drained cleanly without
generation requests. This preparation cost is separate from the subsequent
A/B/B/A policy boots, which reuse the images but still verify source identity.

The frozen serving binary SHA256 is
`ed8b334bfc448266e6b5fae59764c725c93f3caa85eef8328ac9ab5798e62525`.
The initial A/B/B/A diagnostic was stopped after its first whole-expert leg
exposed slow cache admission: only 3.49 GiB of 32 GiB had been populated after
the workload, with 8.8% full hits. That leg's ranks had matching operation
streams, reconciled counts and zero swap; both C2 phases observed 14 decode
rows in seven physical steps. The fractional boot was stopped cleanly before
requests, and both GPUs were verified idle. This is a preserved diagnostic
failure of the warmup/admission setup, not a completed policy comparison.

The fixed diagnostic contains one first-use
request followed by prose/code at C1 and C2, seven requests and 52 output
tokens per boot. It requires observed native batching, equal greedy transcripts
across boots, matching operation streams within each boot, unchanged image
metadata and zero swap. Its short natural-route workload does not replace the
standard shipped-template campaign or establish sustained serving performance.
The demand-admission correction has passed full-checkpoint requalification.
A live preflight with a 96 GiB
expert budget passes on both Sparks: the model plan is 104.24 GiB per node,
plus a separate 4 GiB headroom requirement, against 119.83/120.23 GiB host
available at the checks. This is allocation feasibility, not measured peak
memory or proof of cache warmup. Subsequent runs must sample node memory and
actual resident-weight bytes, alongside hits, I/O and swap.

A subsequent diagnostic completed in A/B/B/A order with the same 96 GiB
budget per rank, first-use traffic, a fixed 192-output-token warmup and 96
measured output tokens per boot. It passed the current-build controlled
locality replay at 1/8/64 rows. The node sampler covers warmup and measurement
at 0.5 s intervals, recording MemAvailable, process/cgroup memory and swap;
it starts after readiness and does not measure startup peak. Per-phase cache
occupancy and hits distinguish a warmup label from demonstrated stability.
This diagnostic remains separate from the standard shipped-template campaign.

The first whole-expert leg completed with identical rank operation streams,
zero swap and 65.34 GiB of committed expert weights on each rank. Rank-zero
residency grew from 13.49 GiB after first-use traffic to 45.48 GiB after prose
warmup and 65.34 GiB after code warmup. The final prose warmup and all four
measured prose/code C1/C2 phases had 100% full expert hits. That demonstrates
a covered working set for these phases rather than a need to fill the remaining
cache space. The warmup-plus-measurement full-hit fraction was 92.08% on both
ranks; local block reads over that interval were approximately 65.77 GiB per
node. Across 526 node samples per rank, minimum MemAvailable was 12.99/13.57 GiB
of approximately 125.72 GiB total. These are node-wide availability observations,
not isolated CUDA/cache residency.

The first fractional leg also completed, retaining 95.99 GiB per rank and
matching the whole-expert leg's greedy transcripts. Its four measured phases
had approximately 65–71% full expert hits. In this first pair, whole-expert
decode-only rates were approximately 5.32–5.34 tokens/s at C1 and 6.77–6.92 at C2;
fractional rates were 2.28–2.33 and 3.03–3.27 respectively. These are short
custom-diagnostic observations with fixed warmup, not standard campaign rows
or proof of fractional-policy convergence. Boot wall time, including source
scan and cache initialization, was 286.7 s versus 501.8 s. Both fractional ranks
also had zero swap. The repeated policy pair is described below; this short
diagnostic establishes no general winner.

Rank-zero counters help separate the observed costs in that first pair.
Whole-expert measured phases performed zero expert-file reads; reconstruction
still took 3.53–6.94 s over each complete phase (prefill plus decode). Fractional
phases read 5.71–11.64 GiB through 11,699–23,828 file calls, with 89.22–91.46%
chunk hits, 11.64–18.37 s of file-ticket wait and 3.75–7.65 s reconstruction.
These are phase totals, not per-token latency, and they must not be added to
overlapping engine/latency clocks. They support an I/O-wait explanation for
this pair's gap without establishing physical storage or DRAM bandwidth.

The second fractional boot also completed with matching greedy transcripts,
matching operation streams between its two ranks, 95.99 GiB resident per rank
and zero swap. Its measured full-hit fractions were 65.08–70.53%, close to the
first fractional boot. Decode-only rates across the two fractional repeats span
2.235–2.276 tokens/s for prose C1, 2.900–3.271 for prose C2, 1.792–2.333 for
code C1 and 2.635–3.030 for code C2. The second boot read 5.71/9.66/7.28/11.44 GiB
in those four phases on rank zero. Minimum node MemAvailable was 13.37/13.59 GiB.
The final whole-expert boot also completed: its measured phases again had 100%
full expert hits, zero expert-file reads and 65.34 GiB committed per rank.
All four boots have identical greedy transcripts, matching operation streams
between ranks, zero swap, zero prompt-cache tokens and observed native C2
batches. The final whole boot's minimum MemAvailable was 13.05/13.64 GiB.

The complete repeated decode-only measurements are below, in tokens/s. Each
C1 phase produces 16 output tokens and measures 15 decode rows; each C2 phase
produces 32 output tokens and measures 30 decode rows. These short observations
are custom diagnostics, not standard shipped-template benchmark rows.

Expert-image reads in this diagnostic used each Spark's local NVMe SSD,
not the NAS. All four boots recorded `io_mode="direct"` on both ranks,
with the image directory backed by local `/dev/nvme0n1p2` (ext4).
Each rank's prepared image was 144,972,845,056 bytes. This identifies the
storage backing for expert misses; the whole-expert measured warm phases
had no expert-file reads because their selected experts were already cached.
This verifies expert-image placement only. It does not establish local storage
for the Engram tables or exclude network I/O from end-to-end serving latency.

| Phase | Whole first | Whole repeat | Fractional first | Fractional repeat |
| --- | ---: | ---: | ---: | ---: |
| Prose C1 | 5.338 | 5.276 | 2.276 | 2.235 |
| Prose C2 | 6.769 | 6.683 | 3.271 | 2.900 |
| Code C1 | 5.322 | 5.269 | 2.333 | 1.792 |
| Code C2 | 6.919 | 6.832 | 3.030 | 2.635 |

Whole boot times including source scan were 286.7/281.1 s; fractional boot times
including initial population were 501.8/473.3 s. After readiness, the single
first-use request per boot had server TTFT in (20,40] s for whole and (7.5,10] s
for fractional. Those are histogram brackets from one request, not reliable
population p95/p99 estimates, and exclude startup time. Fractional first-use
benefits from work already performed during initialization. The longer startup
must remain visible when comparing cold deployment costs.

The policy implementation also has an important admission asymmetry. Whole
misses can retain the entire expert directly from their demand reads; fractional
misses reconstruct missing chunks for execution, but persistent promotion goes
through periodic adaptation. In this diagnostic, adaptation allows at most
64 MiB of new residency per 1,024 observed expert selections. Fractional startup
water-fills the population, so concentrating its nearly 96 GiB allocation on
the repeatedly selected experts requires moving existing residency. It still
recorded promotions and demotions during measurement. The short fixed warmup
therefore does not distinguish slow adaptation from a limiting allocation or
scoring problem. Minimum fractional coverage alone does not establish that the
selected whole-expert working set could not fit under the same budget.

The saved rank-zero deltas quantify that transient in both fractional repeats:
first-use plus the three warmup phases promoted 5.354 GiB and demoted the same
amount. Full-expert hits rose from about 3.7–3.9% on first use to 28.3%, 39.4–39.6%
and 54.7% in successive warmup phases. Subsequent measured phases promoted
another 4.109 GiB while full-hit fractions remained about 65–71%. Promotions
are cumulative traffic and can revisit chunks, so these totals are not the
size of a uniquely learned hot set. They demonstrate continuing adaptation
rather than a settled allocation at the start of measurement.

Likewise, a full-expert cache hit avoids disk reads but does not bypass this
baseline's reconstruction copies, host routing inspection, sequential expert
waves or synchronization. The measured whole-expert rate is not a resident
engine performance ceiling. C2 throughput is aggregate across two requests.

Whole is faster in every measured warm phase in both repeats of this fixed
workload. Fractional cache coverage and file reads are similar between repeats,
but timings vary. The fixed warmup does not demonstrate fractional convergence;
neither these observations nor the controlled replay establishes a general
policy winner. No physical DRAM bandwidth or full-context feasibility claim is
supported by this short-KV diagnostic.

## Implementation and verification sequence

1. Shared CUDA-free geometry, byte accounting, policy state machines and
   deterministic host tests. Prove minimum coverage, promotion/demotion,
   pinning, no over-budget transitions and request coalescing under failures.
2. Bounded immutable store builder/reader, direct-I/O detection and integrity
   tests. Verify all source slice offsets, down-column packing, odd/corrupt
   records, identities, short reads, cancellation and shutdown.
3. Loader/model/MoE integration using byte-exact reconstruction for both
   policies. Verify small GPU fixtures, real-checkpoint per-layer/logit parity,
   TP ownership/collectives and budget high watermarks. No resident startup
   allocation proportional to all experts may precede paging.
4. Follow-on optimization: direct chunk kernels and safe overlap, preserving scaling and activation
   semantics. Qualify isolated numerical changes, full-residency controls,
   mixed/decode/prefill/MTP cases, captured subgraphs and multi-rank progress.
5. Frozen matched mechanism and server campaigns with all requested metrics,
   locality patterns and cold/steady phases. Publish measured results and
   limitations; keep failures and unrun cells visible.

The requested initial implementation may reconstruct complete experts into
bounded scratch and use the existing kernels, provided its reconstruction
overhead is measured. Step 4 describes later execution optimizations, not an
extra prerequisite for that permitted initial delivery. The initial baseline excluded serving MTP and whole-model CUDA Graph execution.
Current eager DSpark and expert-only graph replay are described above;
whole-model graph execution remains unsupported with paging.

Initial completion requires both policies integrated into the shared serving
engine, correct MXFP4/NVFP4 packing and scaling, bounded and separately accounted
memory, multi-rank ownership, and the requested numerical, locality, cold/warm,
concurrency, latency and I/O comparison evidence. Reconstruction costs must be
reported alongside inference costs. A policy library, synthetic benchmark or
configuration flag alone is partial work. Unavailable physical DRAM counters
remain an explicit measurement gap; cache-copy volumes and L2 rates must not be
substituted for that requested measurement.
