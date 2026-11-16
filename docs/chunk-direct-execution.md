# Chunk-direct expert execution

An opt-in NVFP4 prefill path now consumes missing experts directly from the
registered SSD-read buffers. It removes the full-expert reconstruction copy,
but the initial copy-only comparison did not demonstrate a meaningful serving
improvement. The later memory-reclamation comparison below keeps chunk-direct
enabled with fewer reconstruction slots. BF16 activations remain enabled.

## Implementation and limits

`DGPP_EXPERT_CHUNK_DIRECT=1` enables the path on both ranks. A bounded device
pointer table occupies an existing execution slot; the W4A16 tensor-core tile
follows it for gate/up rows and down-projection columns. The kernel preserves
ascending accumulation and applies the original global scale once per complete
dot product. No weights or activations are requantized.

Read leases stay alive through GPU consumption. Transfer completion and kernel
reader completion have separate events, so a later coalesced transfer cannot
erase an earlier reader's lifetime fence. Completed staging leases are retired
as a prefix rather than repeatedly scanning and erasing the entire live vector.
Admission still copies retained chunks into the persistent cache. Native hits
keep their existing prepared layout and fast execution path.

This implementation handles complete misses in NVFP4 prefill with power-of-two
chunk widths of at least 64, dividing the rank's intermediate dimension.
Partial hits and unsupported shapes use the existing reconstruction path.
Decode/speculative verification retains B12x; it is not chunk-direct. The
full-expert scratch reservation also remains because those other paths need it.
This is not independent execution of whichever chunk becomes ready first: a
wave's required chunks are acquired before its kernels launch.

The source buffers must cover the active and following waves. At the deployed
18 chunks per expert and 12 experts per wave, the focused comparison used 512
buffers/queue entries, four I/O workers and the unchanged 100 GiB reservation
per rank. Staging consumes approximately 0.27 GiB within that reservation.
The first 256-buffer version reduced wave size and read-ahead and regressed;
the final version restores 12-expert waves and two-expert demand look-ahead.

## Serving result

These are focused diagnostics, not formal deployment-campaign throughput rows.
The same tool-enabled prompts and warming sequence were used on both ranks,
with prefix reuse disabled, fixed speculative depth three, 1M context and the
same memory budget and 512-buffer configuration. No competing Spark workload
or fabric transfer ran during requests; compilation ran on the separate build
host. Both ranks stayed at zero swap.

| Prefill workload | Reconstruction, seconds | Chunk-direct, seconds |
|---|---:|---:|
| 2,209 tokens, cold | 25.037 | 25.006 |
| 2,209 tokens, steady warm | 1.417 / 1.418 | 1.421 / 1.424 |
| 4,559 tokens, first after smaller prompt | 7.042 | 7.363 |
| 4,559 tokens, repeated miss-heavy | 10.476 / 11.004 | 10.583 / 10.286 |
| Original 24,317-token tool prompt, after that sequence | 25.126 | 24.695 |

All eight final responses, normalized tool arguments, token usage and speculative
acceptance matched the control. Cold reads matched at 95.317 GiB per rank; warm
requests read zero expert bytes. Miss-heavy read volumes vary slightly with
cache allocation. The roughly 0.43-second long-prompt difference is too small
in this comparison to justify a production switch. Decode remains unchanged
in implementation; no decode speedup is claimed.

The control is implementation revision `8f8f0cdc`, with SHA-256
`b54e66dea53677a8954c2e7124b3ab6fd70830a26a2d21820f8eb8ded3d038b7`.
The measured candidate is `341e61e1` plus this implementation, with SHA-256
`b8541ae7b8719ba09b04898b5cba81aec084ba52811be35bc70ba57ca1fb2694`.
The version stamp predates the final commit. The final release rebuild changes
that stamp. At the end of that comparison production returned to 64 staging
buffers and queue entries. The memory-reclamation follow-up below supersedes
that configuration, retaining the endpoint, keys and token limits.

Validation: all 27 host cache tests passed. The B12x paging fixture explicitly
exercised chunk-direct prefill, checked zero reconstruction-copy bytes for those
misses, and matched canonical outputs bitwise. The staging lifetime fixture
covers a later transfer sharing a buffer with a pending kernel reader. The
generic NVFP4/MXFP4 paged-wave fixture also passed.

## Precision tradeoff screen and remaining work

A separate short screen enabled the existing native FP4 activation path with
dynamic scaling (`DGPP_MOE_W4A4=1`, `DGPP_MOE_W4A4_DYNAMIC=1`). Steady warm
2,209-token prefill rose to 3.444 / 3.449 seconds, from about 1.42 seconds with
BF16 activations. It also changed speculative acceptance. It remains disabled;
no broader quality qualification was performed because it failed the speed
screen. This does not establish that an optimized native FP4 implementation
cannot win, only that this existing path is not a useful deployment choice.

Chunk-direct decode, direct partial-residency reads and independent chunk
readiness remain unimplemented. Smaller scratch reservations are evaluated
below using existing configuration controls. No matched EXL3 2.9-bit serving or quality comparison was
run, so these results do not decide the user's model-format choice.


## Reclaiming RAM without enlarging the expert cache

The follow-up reduces real allocations rather than merely bypassing copies.
Both arms use implementation revision `a7320e7b`, binary SHA-256
`91afe785695b63768441e3184f33e1d698f631ed58209d8bb5b322cf62cc2c14`
on both ranks. No rebuild, competing workload or fabric transfer ran during
measurements. These are focused configuration diagnostics, not campaign rows.

| Per-rank setting | Prior production | Retained lower-memory configuration |
|---|---:|---:|
| Chunk-direct | Off | On |
| Reconstruction slots | 24 | 12 |
| Staging buffers / queue depth | 64 / 64 | 256 / 256 |
| Read workers | 4 | 4 |
| Total expert reservation, bytes | 107374182400 | 107242240000 |
| Accounted expert allocation, bytes | 103764337408 | 103632395008 |
| Resident chunk capacity, bytes | 103250891520 | 103250891520 |

Each reconstruction slot occupies 19,908,096 bytes; each staging buffer
557,056 bytes. Removing twelve slots and adding 192 buffers therefore releases
131,942,400 bytes (125.83 MiB) per rank. Lowering the total reservation by that
exact amount prevents the planner from giving the saving to resident experts.
The resident chunk capacity is identical. The configuration uses
`DGPP_EXPERT_CHUNK_DIRECT=1`, 1M context and the existing 262144 output default;
activation quantization remains disabled.

The sequence is four recomputed 2,209-token tool requests followed by the
original 24,317-token request. Prefix-cache reuse is disabled. All five outputs,
normalized tool arguments, token usage and speculative acceptance match, and
both ranks remain at zero swap. This warming sequence differs from the earlier
comparison, so its long-prompt time must not be compared directly to that table.

| Prefill diagnostic | Prior production, seconds | Lower-memory, seconds |
|---|---:|---:|
| 2,209 tokens, cold | 24.305 | 25.490 |
| 2,209 tokens, steady warm | 1.418 / 1.415 | 1.414 / 1.413 |
| 24,317-token tool prompt after that sequence | 21.829 | 20.780 |

Engine decode rates at concurrency one are essentially unchanged on the warm
small requests: 53.09 / 52.95 tokens/s before, 52.77 / 53.02 after. The long
request records 15.31 versus 15.58 decode tokens/s. These decode rates exclude
prefill; no broad concurrency or decode improvement is established.

After the long request, Linux `MemAvailable` was 3,595 / 3,893 MiB on ranks
0 / 1 before, and 3,962 / 4,202 MiB after. Those are whole-node snapshots;
the observed 367 / 310 MiB increases include runtime and system variation and
must not all be attributed to the exact 125.83 MiB allocation reduction.
Filesystem `Cached` stayed essentially constant: approximately 3,303 / 3,123
MiB in both arms. Thus RAM was available to the OS, but this workload did not
turn it into substantially more filesystem cache. Expert reads use `O_DIRECT`;
these results do not establish a filesystem-cache-driven speedup.

The lower-memory configuration is retained on both Sparks: it frees memory,
preserves resident expert capacity and warm speed, and improved the long
request in this comparison. The cold-prefill penalty of approximately 1.18
seconds is explicit. No claim is made that all reconstruction scratch has
been removed; decode and fallback execution still require twelve slots.

## Cache history follow-up (2026-11-15)

The retained two-Spark deployment now also sets
`engine.expert_streaming.decay_routes = 65536`, replacing the default 1024.
All lower-memory settings above remain unchanged. This is a deployment-specific
choice; the global planner default and shipped example templates are unchanged.

The planner already shares capacity across layers and scores reuse against load
cost. Its decay clock advances once per routed token/expert assignment across
all layers. A large prefill can therefore age earlier layers' observations
rapidly. Longer history is a low-cost way to test retention without adding a
second layer-allocation policy. It also changes the normalized benefit in
periodic adaptation; this is a comparison of that configuration, not an isolated
proof that demand eviction alone caused the gain.

A focused diagnostic compared freshly restarted deployments at 1024 and 65536,
using the same binary from `a7320e7b`, tree `c147652c`, SHA-256
`91afe785695b63768441e3184f33e1d698f631ed58209d8bb5b322cf62cc2c14`
on both ranks. The five sequential C1 requests used the full tool-enabled
system prompt (approximately 24.3K input tokens), a 64-token output limit,
fixed speculation depth three and disabled prefix caching. The tasks requested
directory listings of the repository root, `src`, `tests`, then root twice.
The root task additionally requested a brief confirmation and two directory
names. Both arms replayed identical request bodies in the same order.

Cold prefill was effectively unchanged: 40.017 versus 39.974 seconds. Across
the four subsequent requests, summed prefill fell from 87.160 to 62.908 seconds
(27.8%). Their individual prefill times were 24.505/15.507/21.894/25.254 seconds
before and 16.086/14.397/16.100/16.325 seconds after. Expert reads, summed across
those requests and averaged across ranks, fell from 184.54 to 92.81 GiB.
These warmed requests still miss the expert cache; they are not fully resident
execution measurements.

Decode did not improve: summed engine decode time for the four warmed requests
rose from 11.651 to 12.149 seconds (4.3%), with reported per-request engine rates
of 18.16/16.08/16.44/16.21 versus 17.46/14.51/15.99/16.38 tokens/s. The labelled
service-level result, including prefill and request overhead, improved from
98.979 to 75.259 seconds across those requests (24.0%). All five normalized
outputs, usage and speculative acceptance counts matched. Prefix cached-token
counts were zero, and both processes remained at zero swap.

A subsequent short-prompt anchor on the retained candidate reached 100% expert
hits and zero expert reads on both ranks. After layout preparation settled,
the 2,209-token prompt took 1.427 seconds to prefill and decoded at 52.91
tokens/s. This is consistent with the earlier small warm diagnostic above,
but its preceding cache-warming sequence differs, so it is not a fresh paired
comparison against that earlier result.

These are focused diagnostics, not new benchmark campaign rows or evidence
across concurrency levels and diverse applications. The cold anchor resets
the process/expert cache; it does not claim to flush every OS or SSD cache.
The longer history remains deployed because it substantially improves this
partially warm tool workload without consuming more memory or changing outputs.
Long-output decode, abrupt topic shifts and a wider request mix remain open.
Independent readiness for chunk-direct prefill is also still unimplemented;
the existing ready-expert executor currently serves the reconstructed B12x path.

## Ready-prefix prefill experiment

`DGPP_EXPERT_READY_PREFILL=1` enables an optional scheduling change, requiring
chunk-direct NVFP4 prefill to be eligible. After acquiring an expert, the CPU
checks whether every missing chunk of the next expert has a completed read
ticket. If not, it submits the acquired prefix immediately instead of waiting
for the rest of the wave. Existing demand lookahead continues to feed reads.
Native-only groups keep their larger grouping. This changes launch grouping,
not expert selection, weights, activation precision or accumulation order.

The readiness query neither consumes tickets nor changes residency; acquisition
still propagates failed reads. Existing staging-reader fences and execution
leases protect launched work. A readiness result is only a hint: a subsequent
cache admission can evict weights before acquisition. The planner must still
perform normal acquisition checks.

This is deliberately an ordered prefix scheduler. It does not jump past an
unready expert to run arbitrary later experts, execute individual chunks as
they arrive, or remove the first expert's acquisition wait. Smaller groups can
also reduce GPU efficiency, so readiness alone is not evidence of a speedup.

The host cache suite includes a completed-ticket readiness test. The
checkpoint-sized B12x paging fixture increases its test-only staging pool to
256 buffers when this flag is set, allowing multi-expert chunk-direct waves;
the usual 64-buffer fixture would cap those waves at one expert. The generic
paged-wave fixture still covers fallback execution and bitwise accumulation,
but its 32-wide chunks do not exercise chunk-direct eligibility.

The ready-prefix experiment remains **disabled in production**. A focused
five-request comparison used the cache-history sequence above with decay 65536
and identical binary SHA-256
`ecb2bc043318e04da6c075e0dcad5d82f562ecdb33d06bbe9709b111249bad84`
on both ranks in both arms. Its implementation is `9de2eef1` (the executable
was built from those changes before commit, stamped with parent `dde78285`).
Only the ready-prefix environment flag differed between arms.

Cold prefill was 41.109 seconds disabled and 40.660 enabled. Summed prefill
for the four warmed long requests was 63.246 versus 64.403 seconds; summed
engine decode time was 11.383 versus 11.831 seconds. Their per-request engine
decode rates were 18.28/16.40/16.83/16.90 versus 17.48/15.64/16.46/16.23 tokens/s.
Service-level elapsed time summed to 74.797 versus 76.422 seconds. Expert
reads averaged across ranks totaled 92.02 versus 93.22 GiB. This single paired
sequence establishes no useful improvement; the small differences do not
isolate scheduling cost from cache-population and timing variation.

All five normalized outputs, usage and speculative acceptance matched, prefix
cached-token counts stayed zero and both ranks stayed at zero swap. The enabled
candidate's subsequent fully warm 2,209-token anchor settled at 1.429 seconds
prefill and 52.62 decode tokens/s with zero expert reads. This is a warm anchor,
not a fresh paired small-prompt comparison. These results are diagnostics, not
campaign rows. Production retains longer cache history and chunk-direct, with
ready-prefix scheduling off. Arbitrary ready-expert execution and a scheduler
that balances ready-group size against GPU work remain unimplemented.

Validation passed: all 28 host cache tests, the generic paged-wave GPU fixture,
and the checkpoint-sized B12x paging fixture with chunk-direct, ready-prefix,
GPU scatter and resident-first enabled. The latter checks bitwise prefill
agreement with canonical execution and confirms discontiguous chunk reads.

## Longer history and proportional-horizon control

A further configuration-only diagnostic at the same `9de2eef1` implementation
and binary hash `ecb2bc043318e04da6c075e0dcad5d82f562ecdb33d06bbe9709b111249bad84`
tested decay 1048576. Ready-prefix remained disabled and memory stayed fixed.
The control is the preceding ready-prefix-off sequence (decay 65536, horizon
8192); the same five requests were replayed after fresh process restarts.

With horizon held at 8192, cold prefill was 40.717 seconds. Summed prefill for
four warmed requests fell from the control's 63.246 to 54.238 seconds (14.2%),
and average per-rank expert traffic fell from 92.02 to 51.82 GiB. Summed decode
time rose from 11.383 to 12.831 seconds (12.7%); the per-request engine rates
were 17.58/13.92/14.92/14.75 tokens/s. Service-level elapsed time fell from
74.797 to 67.250 seconds (10.1%). This is a prefill/decode tradeoff, not a
uniform improvement.

The external review suggested preserving horizon/decay = 8. Holding decay at
1048576 and raising horizon to 8388608 instead produced 40.831-second cold
prefill, 60.169 seconds summed warmed prefill, 15.027 seconds summed decode,
and 75.395 seconds service-level elapsed time. It did not beat the retained
65536/8192 control. Both larger-decay arms matched all five normalized outputs,
usage and speculative acceptance. Both ranks stayed at zero swap and each
request recomputed its prompt with zero prefix-cached tokens.

The fixed-horizon result cannot isolate ageing from normalized periodic
adaptation benefit. The prefix-enabled continuation comparison below tests whether, once prompt
reuse removes most prefill work, the decode penalty outweighs that saving. The proportional
horizon setting is not selected merely because it preserves the default ratio.


The subsequent direct-DGPP tool-continuation comparison **rejects decay
1048576 for the deployment**. Each arm restarted, made the root-listing tool
request with prefix reuse enabled, then supplied the same fixed representative
tool result. Follow-ups requested a prose investigation plan, a Python cache
helper review, and a repeat of the prose request. Tools remained enabled with
auto choice, and each follow-up had a 256-token output cap. These are controlled
representative continuations, not captured Combo-client traffic or a quality
benchmark: all three textual outputs reached the cap.

All normalized outputs, usage, acceptance and finish reasons matched. The first
two follow-ups reused 24,192 tokens; the repeated prose reused 24,448. Against
65536/8192, the 1048576/8192 candidate increased their summed service-level time
from 47.878 to 63.866 seconds (33.4%), summed prefill from 6.689 to 8.311 seconds,
and summed engine decode from 41.057 to 55.411 seconds. Per-request engine decode
rates fell from 15.10/19.05/23.64 to 11.59/12.90/18.71 tokens/s. Average per-rank
expert reads across the three continuations rose from 55.60 to 101.15 GiB.
Both ranks remained at zero swap. The initial cold tool request took 43.051
versus 42.471 seconds in prefill and about 46.2 seconds service-level in each arm.

This reverses the result from recomputed long prompts. The data support testing
phase-aware retention rather than simply lengthening one global history; they
do not yet prove a particular replacement policy. Production is restored to
**decay 65536, horizon 8192, four read workers**, with ready-prefix disabled.
The next cheap configuration test is reader concurrency, before adding I/O or
scheduling machinery. Exact normal-client tool loops, uncapped task completion,
phase-aware histories and a future-aware traffic bound remain untested.

## Reader-count follow-up

The deployment now retains **eight expert read workers**, up from four, with
256 staging buffers/queue entries, twelve execution slots and the same expert
memory reservation. Decay remains 65536 and horizon remains 8192; ready-prefix
is disabled. No model, precision, context, Engram or prefix-cache setting changed.

The direct-DGPP prefix-enabled continuation sequence above was replayed on fresh
processes with eight and sixteen workers. The four-worker control is the preceding
`loop65536` sequence, with the same binary
`ecb2bc043318e04da6c075e0dcad5d82f562ecdb33d06bbe9709b111249bad84`
on both ranks. These are focused configuration diagnostics, not benchmark
campaign rows or a measurement of physical NVMe command depth.

Cold initial-request prefill was 43.051 seconds with four workers, 39.377 with
eight and 38.384 with sixteen. Across the three prefix-reusing continuations,
summed prefill was 6.689 / 5.233 / 5.373 seconds for four/eight/sixteen workers;
summed engine decode was 41.057 / 35.401 / 37.521 seconds. Per-request engine
decode rates were 15.10/19.05/23.64 tokens/s with four, 18.17/22.14/25.90 with
eight, and 17.08/20.77/24.73 with sixteen. Service-level elapsed time totaled
47.878 / 40.766 / 43.033 seconds. Eight workers reduced that continuation
sequence's elapsed time by 14.9% relative to four; sixteen was better on the
cold prefill but did not improve the continuation sequence further.

Average per-rank expert reads across continuations were nearly unchanged:
55.60 / 55.72 / 55.34 GiB. All normalized outputs, token usage, prefix hits and
speculative acceptance matched. Both ranks remained at zero swap. Answers were
still capped at 256 tokens, so the comparison establishes bounded serving
performance, not completed-task quality. No actual overlapping-read trace or
CPU attribution was collected; these results select a useful worker setting,
not a precise explanation of the sixteen-worker slowdown. Cache allocation
was held fixed; additional thread overhead was not subtracted from the budget.

After retaining eight workers, the short recomputation anchor settled at
1.437/1.421 seconds prefill and 52.81/52.77 decode tokens/s, with zero expert
reads on both ranks. This preserves the previous fully warm performance. Its
initial process-cold 2,209-token prefill took 21.463 seconds; that is an anchor,
not a new paired short-cold comparison. Serving remains running with eight
workers. GPU binding-publication timing, phase-aware cache admission, physical
read coalescing and exact normal-client traffic remain open.

## GPU binding-publication attribution

Implementation `201dde62` adds opt-in `DGPP_EXPERT_PROFILE_PUBLISH=1` diagnostics.
When a prepared revision changes, it records CPU table-build time separately
from copy submission plus the existing service-stream wait. Per-pass cumulative
logs also record calls, bytes, serviced misses and enclosing walk wall time.
No new synchronization is added; with the flag off, no profiling clocks or
per-pass profiling logs run. The executable was built from those changes before
commit, stamped with parent `941cd893`, SHA-256
`2b99816f986a274878b720fdbb93ab2d8a8b050fdf86cf3e5a196134f97a26d7`
on both ranks.

One instrumented run used the eight-reader prefix-enabled tool sequence and
then the small recomputation anchor. These are attribution data, not throughput
benchmark results. Both ranks remained at zero swap; all four normalized tool
sequence outputs, usage and acceptance matched the preceding eight-reader run.
Peer logs were collected from Spark-2's active stage directory, not the stale
peer-log copy in Spark-1's deployment directory. Request boundaries were taken
from retirement records and matched between ranks by target replay count.

For the three continuations, publication was 3.44/3.59%, 3.59/3.85% and
3.15/3.26% of enclosing walk time on ranks 0/1 respectively. Rank 0 spent
1.105 seconds building tables and 0.087 seconds submitting/waiting for copies;
rank 1 spent 1.159 and 0.091 seconds. There were 4007/4027 publications across
283 target replays, averaging approximately 294–313 microseconds per publication.
The initial cold request's decode walk was similar: 3.57/3.67% publication.
The final fully warm anchor had thirteen target replays, zero layer misses,
zero expert reads and **zero publications** on both ranks.

The enclosing clock includes publication, graph execution, miss servicing,
hit accounting and final adaptation; it excludes graph capture, drafting and
prefill. Publication is nested inside it and must not be added to it. Rank
wall times overlap and must not be summed into a speedup prediction. Copy/wait
time can include earlier service work; it is not a GPU memcpy-only timer.

Full table rebuilding is measurable and mostly CPU work, but it is not the
dominant cost of these misses. Incremental dirty publication remains a possible
small optimization, not an explanation for the much larger resident/miss-heavy
performance gap. It is deferred behind reducing decode misses and separating
prefill/decode cache history. Publication-before-resume remains mandatory, and
any dirty tracking must cover evictions, demotions and relocations. Profiling
has been disabled in the retained eight-reader deployment.

## Per-expert prefill-demand history experiment

Implementation `2c1bdcfd` adds opt-in
`DGPP_EXPERT_PREFILL_DEMAND_HISTORY=1`. For paged MMA invocations, it observes
each routed expert once instead of once per routed token. DSV4.1 sends prefill
reader rows to MMA and decode/verification rows to GEMV, including split mixed
batches. Decode token observations and GPU-native hit accounting remain unchanged.
This is not a separate phase-aware history: deduplication changes access counts,
global clock advancement and periodic adaptation cadence together. Counts are
per layer invocation, including prefill sub-batches, not per entire request.

The experiment is **disabled in production**. A fresh-process off/on comparison
used eight readers, decay 65536, horizon 8192 and the same binary on both ranks,
SHA-256 `89acf5c823d5f1f84f4275c0153e7277cc02ef1e63ebac7357e2f4c40ee62afe`.
The executable contains `2c1bdcfd` changes built before commit and stamped with
parent `be0ba509`. Each arm ran the prefix-enabled four-request tool sequence,
one 24,317-token recomputation with prefix reuse disabled, then three repeated
2,209-token recomputations. Budget, precision, speculative depth and all other
settings were held fixed. These are focused diagnostics, not campaign rows.

Cold tool-request prefill was essentially unchanged at 38.617 versus 38.855
seconds. The three continuations increased from 44.120 to 47.811 seconds summed
service-level elapsed time (8.4%). Their summed prefill rose from 5.630 to 6.817
seconds, and engine decode from 38.364 to 40.876 seconds. Average per-rank expert
reads increased from 55.32 to 69.88 GiB (26.3%).

The long recomputation took 20.181 versus 19.223 seconds in prefill, but engine
decode fell from 17.99 to 14.94 tokens/s. The short-prompt adaptation sequence
was worse: the control reached zero expert reads on both ranks at the third
request, with 1.458-second prefill and 52.31 decode tokens/s. The candidate still
read 7.20/7.43 GiB on ranks 0/1 at its third request, with 2.389-second prefill
and 36.00 decode tokens/s. It was not forced through extra repetitions to hide
that adaptation penalty; candidate fully resident steady state is unmeasured.

All eight normalized outputs, usage and acceptance records matched between
arms, and both ranks remained at zero swap. The generic GPU fixture passed
bitwise arithmetic checks and checks that prefill history deduplicates while
GEMV observations retain token counts. The checkpoint-sized packed B12x fixture
also passed with the flag enabled. The retained deployment uses token-frequency
history, eight readers, decay 65536 and horizon 8192.

Counting each touched expert equally discards prefill popularity differences;
this control does not establish that all phase-aware policies are ineffective.
Preserving relative frequency while bounding prefill influence remains open.
A separate constant-load-cost control can test the queue-age weighting without
simultaneously removing popularity information. Neither alternative is yet
implemented or measured.

## Fixed initial load-cost control

The current deployment sets `engine.expert_streaming.learn_load_cost = false`.
Implementation `9c91bbd1` adds this explicit, serialized setting. The general
default remains `true`. Disabling learning retains each expert's initial cost
estimate; it does not force arbitrary unequal experts to equal costs. This
DSV4.1 NVFP4 target population starts with equal estimates. Input validation,
popularity observations, ageing, demand reads and memory accounting are unchanged.
The existing cache still measures read waits for telemetry.

The motivation is that the learned estimate includes elapsed queue time and
extrapolates individual chunk waits to an expert. It can weight cache retention
by scheduling delay rather than future avoided work. Fixed costs are a control
for that feedback, not a claim that load cost never matters.

A same-binary learned/fixed comparison used eight workers, decay 65536, horizon
8192, ordinary token-frequency history and the preceding eight-request sequence:
four prefix-enabled tool requests, one long recomputation, then three small
recomputations. Both ranks used SHA-256
`36df49d50e413b4ea0d18d30004f5bef5d3fabf3324c4c0f8623ae71feef0411`.
The executable contains `9c91bbd1` changes built before commit and stamped with
parent `626a9918`. A reversed fixed/learned repeat replayed just the four-request
tool loop after fresh process starts to check the modest timing difference.

Across the three continuations, learned versus fixed summed service-level time
was 42.786 versus 40.816 seconds in the first pair and 43.463 versus 43.080 in
the reversed repeat: fixed was 4.6% and 0.9% faster. Summed prefill was 5.389
versus 5.369 seconds in the first pair, and 5.536 versus 6.111 in the repeat.
Summed engine decode was 37.245 versus 35.322 seconds initially and 37.803
versus 36.847 in the repeat. The repeated per-request engine decode rates were
16.87/20.34/25.12 tokens/s learned and 17.34/21.18/25.25 fixed. These are two
focused comparisons, not a precise general-throughput estimate or campaign rows.

Average per-rank expert traffic across continuations was 55.56/55.53 GiB in the
two learned runs and exactly 47.01 GiB in each fixed run, a roughly 15% reduction.
Both fixed runs also had identical read volumes between ranks in this sequence;
that does not guarantee rank-identical residency under other resource pressures.
Initial cold prefill stayed between 38.60 and 38.72 seconds across all four boots.

The first long recomputation's elapsed time was slightly worse with fixed costs
(23.29 versus 22.92 seconds), and the first small-prompt warmup was also slower
(5.93 versus 5.26 seconds). By the third small request both had zero expert reads:
1.452/1.447 seconds prefill and 52.45/52.73 decode tokens/s for learned/fixed.
These tradeoffs remain explicit; fixed costs were selected for the repeated
prefix-enabled continuations rather than for every individual request.

All normalized outputs, usage and speculative acceptance matched in both orders,
and both ranks stayed at zero swap. The 17 residency, three paging-config and
28 cache host tests passed, including fixed-cost ranking, retained validation,
and configuration round-trip coverage. The 256-token continuation cap still
precludes a completed-task quality claim. The retained deployment uses fixed
initial costs with eight workers; phase-aware weighting, coalesced physical
reads, replay bounds and broader normal-client workloads remain open.

## Speculative-depth refresh after I/O and cache changes

The retained configuration remains **fixed DSpark depth three**, with confidence
scheduling disabled. A focused depth-three/depth-two comparison used the same
binary (`36df49d50e413b4ea0d18d30004f5bef5d3fabf3324c4c0f8623ae71feef0411`),
eight readers, fixed initial load costs, decay 65536 and horizon 8192. Each arm
restarted, ran the prefix-enabled tool sequence and then three small recomputed
requests. No weight, precision or memory-budget change was made. Depth N means
N draft predictions and N+1 target verification rows.

Cold initial prefill was essentially unchanged: 38.543 versus 38.605 seconds.
Across the three continuations, summed prefill was 5.415 versus 5.372 seconds,
engine decode was 35.569 versus 34.744 seconds, and service-level elapsed time
was 41.116 versus 40.246 seconds. Average per-rank expert reads were 47.01 versus
43.67 GiB. Per-request engine decode rates were 18.63/21.53/25.41 tokens/s at
three and 19.39/21.34/26.43 at two. Raw accepted predictions per draft pass
were 1.68/1.80/1.68 versus 1.38/1.36/1.38; these counters are not substituted
for committed output tokens when computing throughput.

**The generated continuations did not match**, so the apparent 2.1% elapsed-time
gain is not a clean same-output performance improvement. The initial tool call
matched, but later prose/code text changed, including a one-token difference in
reported reasoning count on prose. Small tool requests kept the same normalized
tool arguments but changed their introductory wording. This screen does not
establish whether numerical batching differences or another depth-dependent
execution detail caused the text differences, and is not a quality evaluation.

At the third small recomputation both ranks had zero expert reads in both arms.
Depth three decoded at 52.70 tokens/s, while depth two reached 44.79 (15% lower).
Their prefill times were 1.652 and 1.598 seconds. This is the first zero-read
anchor after the workload switch, not a separate long steady-state campaign.
Both ranks stayed at zero swap. Given the warm decode loss and changed outputs,
depth two is not selected. Depths one/five and speculation off were not repeated
on this latest configuration. No general optimal-depth claim is made.

## Bulk binding lookup

Implementation `ee09ba71` retains a smaller optimization to the measured
publication cost without adding dirty tracking. `ExpertCache::prepared_addresses`
validates ownership once for a batch, validates every index before writing any
output, and returns the same borrowed prepared pointers as individual lookups.
The DSV4.1 publisher uses one batch per layer: 40 ownership mutex acquisitions
instead of 15,360 per rebuild. Backend pointer publication, full table clearing,
all pointer-field writes, full upload and publication-before-resume ordering
remain intact. No bindings are cached across mutations by this API.

Bulk lookup is the default. `DGPP_EXPERT_BULK_BINDINGS=0` retains the individual
lookup control for diagnostics. The 28 host cache tests passed, including bulk
versus scalar comparisons through randomized packing, eviction, relocation and
fractional demotion; wrong-thread access; size/index validation; no partial
output on an invalid index; and explicit null entries for unprepared experts.

A paired scalar/bulk run used publication profiling in both arms and the same
binary on both ranks, SHA-256
`9272694e941958be6d7dc88bdfed3a6b61c46852521a6b1c0c56fabfdbefb486`.
It contains `ee09ba71` changes built before commit and stamped with parent
`1cb6dc8e`. Each arm restarted, ran the four-request prefix-enabled tool loop,
then three small recomputations. Eight workers, fixed initial load costs,
decay 65536, horizon 8192 and fixed speculative depth three were unchanged.
Peer log analysis selects the latest replay-counter epoch because the remote
stage log can contain earlier process runs.

Across the three continuations, rank 0 CPU table-building time fell from
1178.754 to 506.408 ms; rank 1 fell from 1269.529 to 527.186 ms, a 57–59%
reduction. Calls were identical between arms: 4299 on rank 0 and 4440 on rank 1.
Mean CPU build time fell from 274/286 to 118/119 microseconds per publication.
Copy submission/wait remained approximately 94–101 ms per rank. Total publication
fell from 3.90/4.17% to 1.91/1.98% of the enclosing target walk on ranks 0/1.
These overlapping rank clocks are not summed into a latency prediction.

The instrumented service-level continuation total fell from 41.167 to 39.844
seconds. Summed prefill was 5.422 versus 5.313 seconds and engine decode was
35.616 versus 34.395 seconds. Cold initial prefill was 38.490 versus 38.647
seconds. At the final zero-read small anchor, prefill was 1.616 versus 1.604
seconds and decode was 52.55 versus 51.98 tokens/s. That anchor still made one
publication in each arm during preparation; it is not the later zero-publication
steady-state anchor from the earlier attribution run.

All seven normalized outputs, usage and acceptance records matched. Expert read
bytes were identical (47.01 GiB per rank across continuations), and both ranks
stayed at zero swap. These are instrumented diagnostics, not an uninstrumented
campaign throughput claim. The direct CPU reduction, preserved pointer semantics
and unchanged output justify retaining the bulk lookup. Profiling and the
comparison override are removed from the live deployment, leaving default bulk
publication enabled. Incremental dirty tables remain unimplemented.

## Adjacent-read feasibility screen

A read-only storage probe at `cb90da1a` screened groups of four adjacent
544 KiB records before changing the serving queue. Eight Python worker threads
read the same 128 deterministically selected complete experts from each rank's
local image using O_DIRECT. Each arm read 1,283,457,024 bytes. Widths were run
in order 1,4,4,1,1,4. This is a short synthetic storage screen, not a serving
benchmark, cold-prefill measurement or evidence of application speedup.

Separate anonymous mappings for each record failed with EINVAL for the grouped
preadv on Spark-1, although individual reads succeeded. Slices of one contiguous
mapping succeeded. The host has 64 KiB pages and a 4 KiB ext4 filesystem block;
the precise kernel constraint was not isolated. Do not assume that separately
4 KiB-aligned buffers are sufficient for a vector of these half-page-ending
records, or that simply increasing base alignment fixes it.

Contiguous slices reduced calls from 2,304 to 640. Median elapsed times were
192.38 versus 185.69 ms on rank 0 and 195.37 versus 187.61 ms on rank 1,
about 3.5–4% lower. The initial scalar rank-1 sample was 248.46 ms; it is
included in the three-sample median, not presented as the expected gain.
Python dispatch, repeated offsets, device caching and the tiny duration limit
these results. These calls bypassed the serving queue and did not measure
staging contention or time to first ready expert.

No production code or deployment settings changed. A future implementation
would need contiguous staging ownership (or a proven compatible vector
layout), per-record leases, and nonblocking acquisition of extra batch slots.
It must not hold a partial batch waiting for more free staging. Given the
modest screen result and possible readiness delay, adjacent coalescing remains
a lower-priority candidate than reducing avoidable demand traffic. No warm or
cold serving improvement is claimed for this candidate.

## Layer-local history candidate

The opt-in `engine.expert_streaming.layer_local_history` setting replaces the
frequency estimator's global ageing clock with one clock per `(draft, layer)`.
Each selected expert still contributes one observation, including repeats.
Local elapsed selections are multiplied by the number of groups before applying
`decay_routes`, preserving approximately the global half-life over balanced
model traversals. The global observation count, cooldown, adaptation cadence,
load-cost setting, memory allocation and admission rules remain unchanged.
One group's routes no longer age another group's popularity merely because it
runs later in a prefill traversal. Groups that stop executing retain history;
this is not phase-aware admission or separate prefill/decode histories.

The default remains false pending serving evidence. Tests cover layer-order and
observation-batching invariance of final ranking, preserved popularity,
workload-switch adaptation, invalid-observation atomicity, exact single-group
agreement with the existing estimator, and configuration serialization.

### Serving screen: not selected

Implementation `818b4297` was compared in the same binary on both ranks,
SHA-256 `5ed042a6881e26ea327cb65845b5e2847bf62521eaf269ed62fceef380ab1ff7`
(built before commit, stamped with parent `74929a75`). The global-clock control
ran first, followed by the local-clock candidate. Each restarted before the
same four-request prefix-enabled tool loop, then ran three small recomputations.
Other settings retained eight workers, fixed load costs, decay 65536, horizon
8192, fixed depth three and the same cache budget. This is a focused diagnostic,
not a campaign row or an uncapped completed-task quality comparison.

Cold prefill was effectively unchanged: 38.735 versus 38.700 seconds. Across
three continuations, expert reads fell from 47.009 to 44.516 GiB per rank (5.3%),
but service wall time increased from 39.738 to 40.045 seconds. Summed prefill
was 5.318 versus 5.159 seconds; engine decode was 34.286 versus 34.768 seconds.
All four normalized outputs, usage and speculative acceptance records matched.

The workload switch exposed the larger regression. Control small-request
prefills were 7.892, 2.833 and 1.601 seconds; candidate times were 6.035, 4.151
and 3.460 seconds. The first candidate request benefited, but subsequent
adaptation was worse. On the third repeat, control had zero expert reads and
52.87 decode tokens/s; candidate still read 13.326 GiB per rank and decoded at
34.02 tokens/s. Three additional candidate-only repeats still did not establish
a fully resident anchor: reads declined to 9.460 GiB per rank on the sixth
repeat, with 2.734 seconds prefill and 37.76 decode tokens/s. Those extra repeats
are a convergence check, not matched additional control observations.

All seven matched outputs and usage records agreed, and both ranks stayed at
zero swap throughout. The control warm-helper initially lacked its copied
prompt file; after supplying it, execution continued on the same process and
cache state, without replaying the completed loop. Layer-local history remains
disabled and the global-clock deployment is restored. This experiment does not
reject phase-aware admission or other layer-aware estimators; it rejects this
clock substitution as the serving choice for the tested workload.

## Weighted prefill history candidate

`engine.expert_streaming.prefill_route_weight` defaults to 1.0 and accepts
values from 1/1024 through 1. The candidate uses 1/16. Each prefill selection
contributes that weight to both its expert's popularity and the frequency
estimator's elapsed clock. Decode/verification contributes one. This retains
relative token popularity within prefill while reducing how far a long scan
ages earlier decode evidence. It is a shared weighted history, not independent
prefill/decode histories or a reserved decode cache. The raw observation count,
adaptation interval, cooldown, admission rules and memory budget are unchanged.

The DSV4.1 execution path marks MMA observations as prefill; GEMV and native
graph-hit observations retain unit decode weight. The earlier deduplicated
prefill experiment and layer-local clocks remain disabled for this comparison.
Tests cover preservation of recent decode evidence through a short weighted
scan, eventual learning of a repeated new prefill distribution, unchanged raw
observation count, invalid-input atomicity, configuration bounds/serialization,
and the default phase-independent behavior.

### Weighted-history serving screen: not selected

Implementation `ceb19b0f` was tested with the same binary on both ranks,
SHA-256 `1de43683a2673e464473c49cc9c01d243fd1dd75d5b4f5b61c877d209c222109`
(built before commit, stamped with parent `7e303b4d`). Unit-weight control ran
first, then prefill weight 1/16, each restarted before the same four-request
prefix-enabled tool loop and three small recomputations. Other retained
settings were unchanged. This is a focused diagnostic, not a campaign row.
All 21 planner, 28 cache and three paging configuration tests passed on Spark-1.

Cold prefill was 38.979 versus 38.507 seconds, with initial-request expert reads
109.950 versus 110.772 GiB per rank. Across the three continuations, service wall
time increased from 40.342 to 41.882 seconds (3.8%) and expert reads rose from
47.009 to 56.572 GiB per rank (20.3%). Summed prefill was 5.403 versus 6.330
seconds; engine decode was 34.820 versus 35.426 seconds. The final repeated
continuation improved from 10.514 to 9.181 seconds wall, but the first two
continuations lost more than that saved.

Small-request recomputation prefills were 7.829, 2.862 and 1.604 seconds for
control, versus 5.966, 4.054 and 3.498 seconds for the candidate. Control reached
zero reads on the third repeat; candidate still read 14.353 GiB per rank. Decode
on that repeat was 51.87 versus 52.35 tokens/s, so the candidate could warm the
decode working set while remaining worse at recomputing the prompt. A fully
resident prefill anchor was not established for the candidate. No further
convergence sweep was justified by the continuation and repeat-prefill losses.

All seven normalized outputs and usage records matched, as did tool-loop
speculative acceptance. Both ranks stayed at zero swap. The default unit
weight is restored in serving; 1/16 is not selected. The result argues against
further tuning this scalar without new evidence. It does not establish that
separate phase histories or scan-resistant admission cannot help, but those
remain unimplemented rather than demonstrated optimization opportunities.

## Prefill demand read-ahead window

`DGPP_EXPERT_PREFILL_LOOKAHEAD` controls how many known upcoming prefill experts
are offered to `lookahead_demand`. It defaults to the existing two, is clamped
to 1–16, and does not change decode's two-expert window. With demand lookahead
disabled, the existing one-expert speculative path remains unchanged. The
candidate is six, matching the retained configuration's maximum miss-wave
width. It starts reads earlier before the resident group and after each miss
acquisition, without reordering expert execution or changing the staging cap.
`lookahead_demand` continues to count queued, pending-copy and direct-held
buffers against its existing half-pool limit.

This is not arbitrary ready-expert grouping. That requires supporting
noncontiguous activation segments or rebuilding each wave's row mapping;
the current compact activation range assumes contiguous execution order.
The six-expert GPU paging fixture passed for both whole and fractional cache
policies under constrained staging, with bitwise agreement with the resident
chain. Full serving comparison is required to determine whether earlier
submission helps rather than merely increasing outstanding demand.

### Read-ahead serving screen: retain two

Implementation `627f9316` was compared using binary SHA-256
`0eccf37889336b8f822507ca3a0d3f8f36d8f949f5f74deabf9dc825c71e7254`
on both ranks (built before commit, stamped with parent `46185841`). The
existing two-expert window ran first, then six, each restarting before the
four-request prefix-enabled tool loop and three small recomputations. Cache
capacity, eight workers, history settings and fixed depth three were unchanged.
The GPU paging fixture passed before serving. This is a focused diagnostic,
not a campaign throughput row.

Cold prefill was 38.601 versus 38.579 seconds. Three continuations took 40.265
versus 40.798 seconds wall, with summed prefill 5.274 versus 5.385 seconds and
engine decode 34.856 versus 35.286 seconds. Expert reads were identical at
47.009 GiB per rank. Small-repeat prefills were 7.851, 2.859 and 1.604 seconds
for two, versus 7.806, 2.827 and 1.597 seconds for six. Both reached zero expert
reads on the third repeat; decode then measured 52.12 versus 51.56 tokens/s.
There is no meaningful demonstrated benefit from the larger window in this
screen, and the small timing differences do not establish a causal regression.

All seven normalized outputs, usage and expert read volumes matched; tool-loop
acceptance matched and both ranks stayed at zero swap. The temporary six-window
override is removed and default two-expert lookahead is restored. Arbitrary
ready-expert execution remains unimplemented; this test changes submission
lead only and neither proves nor rules out a benefit from readiness scheduling.

## Ready-group prefill candidate

`DGPP_EXPERT_READY_GROUPS=1` enables an experimental ready-group scheduler for
resident-first, chunk-direct NVFP4 prefill. After the resident group, each wave
offers a bounded window of known demand to the existing read queue and stably
selects ready experts. If none are ready it waits for the first queued expert;
it does not busy-spin or reserve extra staging. Group size remains bounded by
the existing wave capacity and the queue retains its half-pool reservation cap.
This candidate changes both submission lead and completion-based selection.

Gathered rows and final route-slot mapping stay fixed. Each expert retains its
original absolute row interval; only the wave's segment descriptors are
permuted. A segmented BF16 SwiGLU launcher handles noncontiguous row intervals
with the same two BF16 roundings as the ordinary activation kernel. It leaves
other rows untouched and keeps activation work proportional to selected rows.
Pinned wave descriptors are uploaded once per slice, then left unchanged until
the forward finishes. Shared-expert placement and final ordered accumulation
remain intact. The earlier ready-prefix control does not truncate these waves.

The candidate is opt-in and is not selected for deployment without serving
evidence. The isolated activation test covers nonmonotonic uneven segments,
zero-length segments, untouched gaps and bitwise arithmetic parity.

### Ready-group serving screen: not selected

Implementation `e5d19f1e` was built before commit, stamped with parent
`f3d217f0`, and tested on both ranks with binary SHA-256
`e1ae2b2e5ab48fe311ee41f69bd670e46d9962c38b07873e838683a06779a625`.
The segmented-activation parity test and checkpoint-shaped b12x paging test
passed with ready groups, chunk-direct prefill and resident-first execution
enabled. The latter covers whole/fractional policies, misses, hits, pressure,
prepared prefill and graph paths. These tests do not constitute full-model
quality evaluation.

The first ordered run produced a different continuation stream from previous
binaries: 39.113 seconds total continuation wall time and 44.376 GiB of reads
per rank, with changed acceptance. The initial tool response matched. The
ready-group run instead matched the earlier output stream. Therefore those
two timings cannot support a same-output scheduler speedup/regression claim.
A single repeated ordered run on this same binary returned to the earlier
stream. The initial outlier and its unresolved cause are retained; neither
correctness failure nor harmless numerical variation has been established.

The ready-group candidate and repeated ordered control matched all seven
normalized outputs, usage and read volumes, plus tool-loop acceptance. Cold
prefill was 38.333 seconds ready versus 38.622 ordered. Three continuations
were 41.165 versus 40.972 seconds wall, with summed prefill 5.395 versus 5.465
seconds and engine decode 35.637 versus 35.351 seconds. Both read 47.009 GiB
per rank over the continuations. These are focused diagnostics, not campaign
rows or evidence of a meaningful speedup.

Small-repeat prefills were 7.583, 2.970 and 1.614 seconds ready, versus 7.810,
2.821 and 1.631 ordered. Both reached zero expert reads on the third repeat;
decode then measured 51.60 versus 52.88 tokens/s. All runs stayed at zero swap.
The default ordered path remains deployed, with the temporary ready-groups
override removed. The last ordered repeat left serving ready. Arbitrary ready
grouping is now implemented and fixture-tested, but this bounded candidate
has no demonstrated serving advantage. No broader readiness-tuning sweep was
performed.

## Prefill quantum: 4096 versus 2048

A focused configuration screen compared the deployed 4096-token initial,
idle ceiling and idle-floor settings against 2048 for all three, on the same
binary `e1ae2b2e5ab48fe311ee41f69bd670e46d9962c38b07873e838683a06779a625`
(implementation `e5d19f1e`, stamped with parent `f3d217f0`). Both ranks
were restarted before the same four-request prefix-enabled tool sequence,
followed by three 2209-token recomputations. The 4096 setting ran first; this
is a single focused comparison, not a published campaign or a concurrency test.

Cold initial prefill increased from 38.581 to 39.903 seconds at 2048. The three
continuations increased from 38.209 to 39.893 seconds wall, with summed engine
prefill 5.057 versus 5.334 seconds and decode 33.033 versus 34.405 seconds.
Continuation expert reads fell from 47.009 to 45.898 GiB per rank. On the third
small recomputation, both ranks had zero expert reads; prefill was 1.594 versus
1.750 seconds and decode was 52.47 versus 52.05 tokens/s. All seven normalized
outputs and usage records matched, as did tool-loop speculative acceptance;
both ranks remained at zero swap.

The 2048 setting is not selected for this workload. The 4096 values were
restored and serving restarted. This screen does not establish the best setting
for concurrent requests or larger chunks; it shows the smaller idle quantum
hurt the tested cold, partially warm and fully warm cases.

## Speculative depth five versus three on the current deployment

A focused same-binary screen compared fixed speculative depths three and five
on binary `e1ae2b2e5ab48fe311ee41f69bd670e46d9962c38b07873e838683a06779a625`
(implementation `e5d19f1e`, stamped with parent `f3d217f0`). Depth three ran
first; both arms restarted before the same prefix-enabled four-request tool
sequence and three 2209-token recomputations. Depth means draft predictions
per verification; the target verifies one additional pending row. This screen
is not a campaign or an uncapped quality evaluation.

Initial cold prefill was 38.730 versus 38.607 seconds, with identical initial
normalized tool response and usage. The three continuations took 40.914 versus
43.398 seconds wall, with summed prefill 5.365 versus 5.695 seconds and engine
decode 35.404 versus 37.573 seconds. Expert reads were 47.009 versus 51.062 GiB
per rank. Depth five accepted more predictions per pass: for the prose, code,
prose continuations respectively, 172/84, 178/79, 172/84, versus 161/96,
164/91, 161/96 at depth three. Generated continuation text differed, including
usage differences on both prose requests, so these elapsed numbers are not a
same-output comparison and do not establish a numerical correctness issue.

The small recomputations had prefill 7.841, 2.829, 1.602 seconds at depth three
and 7.862, 3.345, 1.748 at depth five. On their third repeat, depth three read
zero expert bytes per rank and reached 50.37 decode tokens/s; depth five still
read 0.243 GiB per rank and reached 52.33 tokens/s. Thus the depth-five anchor
is not a zero-read comparison, and the small rate difference is not a proven
fully resident improvement. Tool-call arguments and introductory text should
not be assumed identical: the three small responses differed in their full
normalized messages, although their usage records matched. Both ranks remained
at zero swap. Fixed depth three is restored. This single screen does not rule
out depth five for other workloads or a separately designed depth scheduler.

## One additional GiB of expert cache per rank

A focused same-binary screen added 1,073,741,824 bytes per rank to the configured
expert budget, from 107,242,240,000 to 108,315,981,824 bytes. The existing
configuration ran first; each arm restarted before the four-request
prefix-enabled tool sequence and three 2209-token recomputations. Binary
`e1ae2b2e5ab48fe311ee41f69bd670e46d9962c38b07873e838683a06779a625`
contains implementation `e5d19f1e` and was stamped with parent `f3d217f0`.
This is a single focused comparison, not a campaign or long-context test.

Initial cold prefill was 38.629 versus 38.595 seconds. Three continuations took
40.649 versus 39.925 seconds wall; summed engine prefill was 5.505 versus 5.276
seconds and decode was 35.023 versus 34.531 seconds. Expert reads fell from
47.009 to 44.600 GiB per rank. This 1.8% wall difference is small for one
ordering; it does not establish a general throughput improvement. All seven
normalized outputs and usage records matched, as did tool-loop speculative
acceptance. Server processes remained at zero swap at captured boundaries.

The third small recomputation reached zero expert reads in both arms. Prefill
was 1.630 versus 1.604 seconds and decode 51.82 versus 52.53 tokens/s. At the
end of that sequence, MemAvailable was approximately 4.06/4.30 GiB with the
original budget and 2.97/3.34 GiB with the larger budget on ranks 0/1. These
are system snapshots, not guarantees of memory available for longer contexts,
concurrency or other processes. The larger budget was not retained: the small
single-run improvement has not established a worthwhile trade for reducing
headroom at the configured million-token context. The original budget and
serving configuration were restored. A larger cache may still help another
working set; no capacity sweep or long-context qualification was performed.

## Zero versus 25% promotion margin

A focused configuration screen compared the default 0.25 promotion margin to
zero on the same binary `e1ae2b2e5ab48fe311ee41f69bd670e46d9962c38b07873e838683a06779a625`
(implementation `e5d19f1e`, stamped with parent `f3d217f0`). The margin affects
both demand admission replacement and the periodic planner's promotion cost;
this is not an isolated test of either. The default ran first, each arm
restarted before the same four-request prefix-enabled tool loop and three
2209-token recomputations. This is a single focused diagnostic, not a campaign
or evidence about concurrent or long-context workloads.

Cold prefill was 38.650 versus 38.656 seconds. Three continuations took 40.619
versus 41.231 seconds wall, with summed prefill 5.281 versus 5.528 seconds and
engine decode 35.209 versus 35.554 seconds. Expert reads were 47.009 GiB per
rank in both arms. The timing difference in this single ordering is not proof
of a causal slowdown. On the third small recomputation both ranks had zero
expert reads; prefill was 1.603 versus 1.641 seconds and decode 50.60 versus
52.64 tokens/s. All seven normalized outputs, usage and per-request read
volumes matched, as did tool-loop speculative acceptance; both server
processes remained at zero swap at the recorded boundaries.

Zero margin showed no meaningful benefit in this workload, so the default 0.25
was restored. Equal aggregate reads do not prove that the internal residency
or eviction decisions were identical. Other margins, workload switches and
phase-specific admission remain untested by this screen.

## Buffered versus direct expert reads

A focused screen compared `io_mode=direct` and `io_mode=buffered` using the
same binary `e1ae2b2e5ab48fe311ee41f69bd670e46d9962c38b07873e838683a06779a625`
(implementation `e5d19f1e`, stamped with parent `f3d217f0`). Direct ran first;
each arm restarted before the same four-request prefix-enabled tool loop and
three 2209-token recomputations. The OS page cache was not cleared between
arms. This is a single serving diagnostic, not an isolated storage benchmark,
fully cold filesystem test or campaign result. Buffered reads add a kernel
page-cache path and a copy into the existing staging buffers; no cache or RAM
budget was changed.

Fresh-process initial prefill took 38.626 seconds direct versus 45.217 buffered.
Across three continuations, wall time increased from 40.030 to 44.904 seconds;
summed engine prefill increased from 5.245 to 6.459 seconds and decode from
34.628 to 38.293 seconds. Logical expert reads were identical at 47.009 GiB per
rank. On the third small recomputation, both ranks had zero expert reads;
prefill was 1.602 versus 1.607 seconds and decode 52.19 versus 52.79 tokens/s.
All seven normalized outputs and usage records matched, as did tool-loop
acceptance; both server processes had zero swap at recorded boundaries.

Rank-zero process `/proc/PID/io` read-byte deltas for the three continuations
were 19.376/19.691/8.078 GiB direct versus 19.857/20.193/8.301 buffered. They
include all process reads and filesystem effects, not just expert payload, and
are not exact device-bandwidth or cache-hit measurements. After the sequence,
whole-system `Cached` was about 2.93/3.08 GiB on ranks 0/1 for direct versus
4.46/4.47 GiB for buffered. This shows more filesystem cache at the snapshot,
not that the extra cache held useful expert records. The tested buffered path
was slower despite that increase. Direct I/O was restored; different cache
sizes, explicit page-cache policies and other workloads remain untested.

## Single-donor whole-expert admission screen

Implementation `73a11363` tested a single-pass donor selection when every whole
expert has the same allocation size and the cache lacks room for a demanded
expert. In that case one eviction suffices, so the path selects the same
minimum density and expert-ID tie break without constructing the general heap.
Unequal expert sizes, fractional residency and nonfinite candidate scores
fell back to the existing path. A randomized 1500-step comparison against the
general heap covered varying costs, cooldown, ties and workload shifts; the
22 planner and 28 cache tests passed on Spark-1. The candidate was selected
only with `DGPP_EXPERT_SINGLE_DONOR=1`.

A focused serving screen used identical binary SHA-256
`bb5a51289cc38a323c35754259d65523cf734c9f6ef8de3bdbc5bc0e6e2e2a0e`
(built before `73a11363`, stamped with parent `12290b9a`) on both ranks. The
heap control ran first; both arms restarted before the four-request
prefix-enabled tool sequence and three small recomputations. This is a single
serving diagnostic, not a campaign row.

Cold prefill was 38.708 versus 38.660 seconds. Continuation wall time was
40.706 versus 40.595 seconds, with summed prefill 5.363 versus 5.371 seconds
and engine decode 35.228 versus 35.090 seconds. Both read 47.009 GiB per rank.
On the third small recomputation both ranks read no expert bytes; prefill was
1.615 versus 1.610 seconds and decode 52.54 versus 52.45 tokens/s. All seven
normalized outputs, usage and read volumes matched, as did tool-loop
acceptance; processes had zero swap at recorded boundaries. These timings do
not demonstrate a meaningful serving benefit, nor do they independently
measure planner CPU time. The candidate is removed and the existing planner
is restored; the running binary contains the now-disabled experimental path.

## FP4 prefill tile crossover screen

Implementation `22d5edc0` tested selecting the 16-row FP4 grouped MMA tile
for expert segments of up to 32 rows instead of the default 16. Both gate/up
and down projections used the same crossover. The checkpoint-shaped paging
fixture and the resident-chain parity fixture passed on Spark-1 with the
candidate enabled. This did not measure how often serving selected that tile.

The same binary on both ranks, SHA-256
`93ba290787172314bbbb81b60c863d3cc5d353926787e26ebaad529aadb5de49`
(built before the candidate commit and stamped with parent `fc059765`), ran
the default followed by the candidate. Each fresh process served the same
prefix-enabled four-request tool sequence and three 2209-token recomputations.
This is a focused single-order diagnostic, not a campaign result.

Cold initial prefill was 38.650 versus 38.619 seconds. The three continuations
took 40.667 versus 40.552 seconds wall; summed prefill was 5.378 versus 5.370
seconds and engine decode 35.176 versus 35.063 seconds. Each rank read 47.009
GiB of expert data across continuations. In the third small recomputation both
ranks read zero expert bytes; prefill was 1.597 versus 1.600 seconds and engine
decode 51.38 versus 51.74 tokens/s. All seven normalized messages and usage
records matched, as did tool-loop speculative counters. Both server processes
had zero swap at the captured boundaries.

The comparison shows no meaningful gain on this workload, so the crossover
switch was removed and the default restored. The deployed binary still contains
the disabled experimental switch until the next rebuild. Other shapes and
concurrent workloads were not evaluated.

## Whole-expert demand traffic bound

The opt-in `DGPP_EXPERT_DEMAND_TRACE=/path/prefix` diagnostic records the
initial residency and every grouped cache observation, including native graph
hits and repeated token selections. Files are suffixed with rank, process and
cache identity. Tracing is disabled by default; when enabled it flushes after
each event, so timings from this mode are diagnostic only. Trace files expose
expert routes and should be handled as workload data, not committed results.

`python3 scripts/expert_demand_bound.py TRACE` computes an offline lower bound
for equal-sized whole experts. It deduplicates each layer/batch's demands and
grants the oracle future knowledge and unlimited free staging within each
batch; at each boundary it retains the experts needed soonest. The starting
residency comes from the trace. The script rejects partial starting residency
and unequal expert geometry. `--start-ns`/`--end-ns` select a Unix-nanosecond
window; a noninitial window requires `--free-initial`, which grants an ideal
starting cache for free. This makes the window result more optimistic still.
The bound is not an implementable cache policy, a prediction of latency, or
proof that a particular number of reads can be avoided. It ignores staging
limits and loading order, and fixes the observed routes and outputs.

An instrumented fresh-process run used binary SHA-256
`1ceeec1ee162f2a3df7cf286a4998e293e354d99453dfbf4cd2b93666365fc08`
(source based on `e63e5926`, with the diagnostic trace added), eight workers,
direct I/O, and the existing cache and depth-three settings. Each rank had
15,360 equally sized experts and room for 10,359 full experts; initial residency
was empty. The four-request prefix-enabled tool sequence and three short
recomputations generated 14,040 observed batches per rank. The two ranks had
matching route-bound totals and actual expert-read bytes in these windows.

The initial request read 109.950 GiB per rank, versus 108.932 GiB for the
relaxed oracle. The following three continuations read 47.009 GiB per rank.
For those continuations alone, an oracle given a perfect starting cache for
free has a 10.076 GiB lower bound. Over the entire initial request plus
continuations, starting from the actual empty cache, the lower bound is
129.831 GiB per rank versus 156.959 GiB read. The 27.128 GiB difference is an
upper bound on the traffic reduction available under these relaxed assumptions,
not a demonstrated saving. The short recomputation sequence read 33.385,
6.275 and then zero GiB per rank, demonstrating adaptation to that workload.
The zero-read third request is a warm anchor, not evidence about long-context
or concurrent requests. Processes remained at zero swap at captured boundaries.

This result makes cache allocation worth further work, especially across
continuations, but does not establish that the rejected history variants or a
future-aware allocation are practical improvements. The trace override was
removed and the normal service configuration restored after the capture.

## Decode-aware admission candidate

The demand trace suggested giving recent decode demand additional weight when
admitting experts during decode, while leaving prefill admission and periodic
planning on the original popularity history. A quick fixed-route replay with
free per-batch staging predicted 47.009 versus 42.611 GiB of continuation reads
per rank at boost 16 and retained the zero-read repeated-prefill anchor. That
replay is a policy screen with relaxed staging, not a throughput measurement.

Implementation `f016793f` tested a separate, exponentially decayed decode-only
history. Its initial implementation calculated an additional exponential per
candidate in each admission scan. A first same-binary, default-first tool-loop
comparison read 47.009 versus 42.611 GiB per rank, but continuation wall time
was 37.925 versus 38.982 seconds. Initial cold prefill was 38.439 versus
38.709 seconds. That binary was SHA-256
`68d6328584045a6fd76007e8132f5e82ad322e679e1c3baee58b2cec887c4d2f`,
built before the implementation commit and stamped with parent `6dd6c531`.
The traffic result alone did not establish a speedup.

Implementation `276997a1` used a shared decay scale and occasional rebasing to
remove the extra exponential from the admission scan. It was tested in one
binary on both ranks, SHA-256
`c6cbe984c4a36cd9430aba16722589d936418c0b5d3c19862e888fe96101448e`,
built before that commit and stamped with parent `f016793f`. Each arm restarted
before the same prefix-enabled four-request tool loop. A candidate-first pair
also ran three short recomputations; a default-first repeat ran the tool loop
only. These are focused diagnostics, not published campaign rows.

| Setting | Candidate-first pair: default / boost | Default-first repeat: default / boost |
| --- | --- | --- |
| Initial cold prefill | 38.626 / 38.627 s | 38.643 / 42.461 s |
| Three continuation wall times summed | 41.385 / 37.755 s | 40.366 / 40.911 s |
| Three continuation engine-prefill times summed | 5.417 / 5.330 s | 5.355 / 6.086 s |
| Three continuation engine-decode times summed | 35.831 / 32.271 s | 34.895 / 34.673 s |
| Expert reads across continuations, each rank | 47.009 / 42.611 GiB | 47.009 / 42.611 GiB |

In the candidate-first pair, the third small recomputation read zero expert
bytes on both ranks. Prefill was 1.645 versus 1.607 seconds and engine decode
52.77 versus 52.32 tokens/s. All normalized outputs and usage matched, as did
tool-loop speculative acceptance. Both ranks' server processes had zero swap
at the captured boundaries. The repeat's four outputs, usage and acceptance
also matched, with zero swap. The cold timing difference in that repeat
occurred with identical expert read bytes; its cause was not established.

The candidate reliably reduced logical expert traffic in this sequence, but the
two optimized serving pairs did not establish a consistent wall-time win. It
was therefore removed and the original policy restored, rather than enabling
another unproven setting. The deployed binary contains the disabled candidate
until the next rebuild. Different workload sequences, concurrency and a
cheaper or more adaptive admission policy remain open.

## Combined graph service and binding wait

Implementation `a54e2648` tested one service-stream wait in place of two
consecutive waits on a serviced target-graph miss. It builds and enqueues the
binding-table publication while miss execution is still queued, then waits
once before releasing the graph gate. An unchanged table revision still forces
that wait, because the serviced output must complete even when no pointers
changed. The checkpoint-shaped b12x paging fixture passed; it covers the
underlying graph and expert arithmetic, not this model-specific wait itself.

The same binary on both ranks, SHA-256
`0323d7a0658a82cdf6f57d4151c11073bd31e26ad95454fd85b1949451b651d3`
(built before the candidate commit, stamped with parent `99aaa7c8`), ran the
existing wait followed by the combined wait. Each arm restarted before the
four-request prefix-enabled tool loop and three 2209-token recomputations.
This is a single-order diagnostic, not a deployment campaign.

Cold initial prefill was 38.627 versus 38.529 seconds. Three continuations
took 37.898 versus 40.477 seconds wall; summed prefill was 5.134 versus 5.489
seconds and engine decode 32.607 versus 34.865 seconds. Both ranks read
47.009 GiB of expert data across the continuations. On the third short
recomputation, both ranks read zero expert bytes; prefill was 1.602 versus
1.594 seconds and engine decode 53.01 versus 51.52 tokens/s. All seven
normalized messages and usage records matched, as did tool-loop speculative
acceptance; both server processes had zero swap at captured boundaries.

The single comparison did not demonstrate a serving improvement and cannot
attribute the timing difference to the wait alone. The candidate and override
were removed, and the prior service configuration restored. The current binary
contains the disabled experiment until its next rebuild.

## 8192-token prefill workspace versus expert-cache capacity

The production configuration uses a 4096-token prefill quantum. An attempt
to raise initial, idle and idle-floor budgets to 8192 while retaining the
existing 107,242,240,000-byte per-rank expert budget was rejected by the
startup memory preflight: rank 0's plan required 116.60 GiB plus the configured
4.00 GiB headroom against approximately 118.30 GiB available. No serving
request ran at that configuration.

A second attempt traded 3.5 GiB of the expert budget per rank for the larger
workspace, keeping the million-token context and other serving settings. It
was compared with the preceding 4096-token control on the same binary,
SHA-256 `0323d7a0658a82cdf6f57d4151c11073bd31e26ad95454fd85b1949451b651d3`
(source based on `99aaa7c8` plus the disabled combined-wait experiment). The
control ran first; each fresh process served the same four-request prefix-enabled
tool sequence and three 2209-token recomputations. This is a focused diagnostic,
not a campaign result or an isolated test of prefill quantum at equal capacity.

Cold initial prefill was 38.627 versus 38.666 seconds, with expert reads of
109.950 versus 110.753 GiB per rank. Across the three continuations, wall time
was 37.898 versus 43.669 seconds; summed prefill was 5.134 versus 6.330 seconds
and engine decode 32.607 versus 37.200 seconds. Expert reads rose from 47.009
to 56.040 GiB per rank.

The third short recomputation was a zero-read warm anchor under the original
configuration: 1.602-second prefill and 53.01 decode tokens/s. The traded
configuration still read 17.220 GiB per rank on that repeat, with 3.742-second
prefill and 26.30 decode tokens/s; its eventual all-resident steady state was
not established. Both processes remained at zero swap at captured boundaries,
and all seven normalized messages and usage records matched, as did tool-loop
acceptance. At the final snapshot, system MemAvailable was approximately
4.11/4.34 GiB for the control and 4.35/4.35 GiB for the traded configuration.

This memory trade was not retained: the tested normal continuations and short
recomputations were slower, while cold prefill did not improve. The result does
not isolate the effect of 8192 rows at unchanged expert capacity or establish
performance for concurrent or longer-context requests. The 4096-token budgets
and original expert capacity were restored.

## Recent-decode admission replay screen

A fixed-route offline screen tested a cheaper alternative to the rejected
separate decode-frequency history. It added a constant score bonus to experts
selected within the most recent 4,096, 16,384 or 65,536 decode selections,
only when admitting during decode. The all-phase exponential popularity score
and the 65,536-selection half-life were otherwise unchanged. Bonuses of 1, 4
and 16 were screened, using the same rank-0 demand trace and request boundaries
as the whole-expert bound. The simulator assumes free staging within each batch
and selects a cache at batch boundaries; it is a policy screen, not an exact
replay of admission, execution timing or a serving benchmark.

The no-bonus control predicted 47.009 GiB of reads during the three tool
continuations. The 4,096-selection window produced the same result at all
three bonuses; the longer windows predicted 47.037 GiB. The control and all
candidates predicted zero reads on the third small recomputation. Initial
request reads were 109.950 GiB in each case. These results give no reason to
implement this particular recent-use bonus, and do not rule out other admission
policies or workloads. No serving settings changed.

## Incremental binding publication screen

An opt-in prototype compared each expert's prepared address against the last
published address and rebuilt only changed entries, uploading each changed
layer's table. Its first publication initialized every layer; address removal
cleared both projection bindings. The existing service-stream synchronization
and publication-before-graph-resume dependency were retained. It still scanned
all expert addresses; this was a changed-layer upload experiment, not a cache
mutation-driven dirty list. The full-table control and prototype ran in one
binary, SHA-256 `2ed00508b0944417da4e3c7fa670dc6f902738694b4e9da85c1b5f1a517ea74a`,
built from `8c01cf2c` plus the prototype.

A focused single-order comparison restarted for each arm, ran the same
prefix-enabled four-request tool sequence and three 2209-token recomputations.
Cold initial prefill was 38.495 versus 38.630 seconds for full versus
incremental publication. The three continuation wall times summed to 40.010
versus 40.615 seconds; their summed engine decode times were 34.476 versus
35.141 seconds. Each rank read 47.009 GiB across continuations in both arms.
On the third short recomputation, both ranks read zero expert bytes; prefill
was 1.598 versus 1.606 seconds and decode was 52.99 versus 51.40 tokens/s.
All normalized outputs, usage and speculative acceptance matched, and both
ranks had zero swap at the captured boundaries.

This is a focused diagnostic, not a campaign result. The single comparison
shows no improvement and cannot establish the cause of the timing difference.
The prototype was removed and the prior serving binary and settings restored.
It does not rule out a more efficient mutation-driven dirty list, but the
previous publication profile and this result do not justify retaining this
particular implementation.

## Reconstruction wave capacity: twelve versus twenty-four slots

The earlier RAM trade changed several settings together. This configuration-only
screen isolates execution-slot capacity more closely: the control uses twelve
slots and the candidate uses twenty-four. The candidate's per-rank expert
budget increased by 238,897,152 bytes, exactly twelve 19,908,096-byte slots,
to keep the resident expert capacity unchanged. Staging remains at 256 buffers
and eight read workers; chunk-direct, fixed depth three and the million-token
context remain enabled. Both arms use the same deployed binary, SHA-256
`0323d7a0658a82cdf6f57d4151c11073bd31e26ad95454fd85b1949451b651d3`
(source based on `99aaa7c8` plus a disabled combined-wait experiment).

Each fresh process served the prefix-enabled four-request tool sequence and
three 2209-token recomputations. Cold initial prefill was 38.540 versus 39.834
seconds for twelve versus twenty-four slots. The three continuation wall times
summed to 38.314 versus 41.373 seconds; their engine decode intervals summed to
33.107 versus 35.815 seconds. Per-rank expert reads were identical: 109.950 GiB
for the initial request and 47.009 GiB across the continuations. Both arms had
186,462 resident chunks at the captured post-request boundaries.

The three repeated-prompt prefills were 7.843, 2.870 and 1.602 seconds with
twelve slots, versus 13.593, 3.827 and 1.603 seconds with twenty-four. The
third repeat had zero expert reads on both ranks, and decoded at 52.69 versus
52.79 tokens/s. All normalized responses, usage and speculative acceptance
matched; both ranks had zero swap at captured boundaries.

This is a single-order focused diagnostic, not a campaign row. The result does
not identify the cause of the slower miss-heavy requests or establish behavior
at other concurrency levels. It provides no reason to spend the additional
memory on twenty-four slots for this workload, so the twelve-slot configuration
was restored.
