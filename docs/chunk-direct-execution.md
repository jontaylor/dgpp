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

## Independent decode-miss completion on the current configuration

The earlier independent-completion screen used four readers, sixteen staging
buffers and twenty-four execution slots. The deployment now uses eight readers,
256 buffers and twelve slots. A same-binary follow-up compared grouped misses
with `DGPP_MOE_ASYNC_MISSES=1` under those current settings. The binary on both
ranks was SHA-256
`0323d7a0658a82cdf6f57d4151c11073bd31e26ad95454fd85b1949451b651d3`
(source based on `99aaa7c8` plus a disabled combined-wait experiment). The
option changes small-batch miss servicing; it retains bounded staging and
ordered final accumulation. Expert capacity, direct I/O, million-token context,
chunk-direct prefill and depth-three speculation were unchanged.

Each arm restarted and ran the same prefix-enabled four-request tool sequence
and three 2209-token recomputations. A control-first pair was followed by a
candidate-first pair. These are focused diagnostics, not benchmark campaign
rows or evidence across concurrency levels.

| Measurement | Control-first: grouped / independent | Candidate-first: grouped / independent |
| --- | --- | --- |
| Initial cold prefill | 38.562 / 38.652 s | 38.720 / 38.998 s |
| Three continuation wall times, summed | 40.400 / 38.043 s | 41.559 / 38.169 s |
| Three continuation engine decode times, summed | 34.899 / 33.080 s | 35.996 / 33.302 s |
| Third short recomputation prefill | 1.601 / 1.604 s | 1.642 / 1.605 s |
| Third short recomputation decode | 50.76 / 52.84 tok/s | 52.72 / 52.26 tok/s |

Each rank read 109.950 GiB for the initial request and 47.009 GiB across
continuations in every arm. The three short recomputations read 33.385, 6.275
and zero GiB per rank. Normalized messages, usage and speculative acceptance
matched across all arms; both ranks had zero swap at captured boundaries. The
initial cold decode times were 2.614 / 2.377 seconds in the first pair and
2.637 / 3.038 seconds in the reverse-order pair, so a cold decode improvement
is not established.

Independent completion improved the measured partially warm continuations in
both orders while preserving the zero-read warm anchor. It is enabled for this
Spark deployment via `DGPP_MOE_ASYNC_MISSES=1`. The tests do not establish a
general throughput gain, a concurrent-serving result or the exact contribution
of I/O overlap versus launch scheduling.

## Reader count with independent decode-miss completion

The previous reader-count screen used grouped miss execution. A configuration-only
follow-up compared eight and twelve readers with
`DGPP_MOE_ASYNC_MISSES=1` enabled in both arms, leaving the 256 staging buffers,
twelve execution slots, resident budget, million-token context and depth-three
speculation unchanged. Both used binary SHA-256
`0323d7a0658a82cdf6f57d4151c11073bd31e26ad95454fd85b1949451b651d3`
(source based on `99aaa7c8` plus a disabled combined-wait experiment). This
single-order focused diagnostic is not a benchmark campaign row.

Each fresh process served the prefix-enabled four-request tool sequence and
three 2209-token recomputations. Cold initial prefill was 39.137 versus 38.875
seconds with eight versus twelve readers. The three continuation wall times
summed to 37.929 versus 37.213 seconds; engine decode intervals summed to 32.943
versus 32.242 seconds. Each rank read 109.950 GiB initially and 47.009 GiB over
the continuations in both arms. The third short recomputation had zero expert
reads per rank, 1.600 versus 1.603 seconds of prefill and 52.73 versus 52.86
decode tokens/s. Normalized messages, usage and speculative acceptance matched;
both ranks had zero swap at the captured boundaries.

The 0.716-second continuation difference is small relative to timing variation
across fresh processes in earlier screens; this one ordered pair does not
establish a reliable gain or explain the effect of extra readers. Eight readers
were restored. Independent decode-miss completion remains enabled.

## Noncontiguous ready-expert grouping during decode misses

A temporary opt-in prototype collected all currently ready reconstructed
experts into one callback, even when their execution slots were nonadjacent.
It did not wait for other experts: it moved the ready leases into a contiguous
callback view, kept them alive through submission, then restored their original
slot mapping for admission and retirement. Ordered final route-slot reduction
remained unchanged. The control retained the existing contiguous-ready runs;
both used independent miss completion, eight readers, 256 staging buffers
and twelve execution slots.

The 28-test host cache suite and the B12x paging arithmetic fixture passed with
the prototype enabled. These fixtures do not establish how frequently the
nonadjacent case occurs in serving. The same server binary on both ranks,
SHA-256 `1ed75cb119b3466efd8b5b5431b0a284d17990279618e33d291ee0545b001ffd`
(source based on `70fc97ef` plus the prototype), ran the control followed by
the candidate. Each fresh process served the four-request prefix-enabled tool
sequence and three 2209-token recomputations. This is a focused single-order
diagnostic, not a benchmark campaign row.

Cold prefill was 38.581 versus 38.490 seconds; the three continuation wall
times summed to 38.106 versus 38.069 seconds, and engine decode times summed
to 33.053 versus 33.171 seconds. Each rank read 109.950 GiB initially and
47.009 GiB across continuations in both arms. The third short recomputation
read zero expert bytes per rank; prefill was 1.594 versus 1.601 seconds and
decode was 52.81 versus 52.73 tokens/s. Normalized messages, usage and
speculative acceptance matched; both ranks had zero swap at captured boundaries.

No meaningful serving improvement was observed, and launch grouping frequency
was not measured. The prototype was removed and the prior binary and serving
configuration restored. Independent completion remains enabled.

## Separately aligned staging and adjacent direct reads

The earlier adjacent-read screen found that a four-record `preadv` succeeded
with slices of a contiguous mapping but failed with separately mapped buffers.
A follow-up tested whether separate mappings would work when every destination
base was aligned to the Spark's 64 KiB host page size. On each rank, four
independent anonymous mappings were overallocated, sliced to 64 KiB-aligned
addresses and passed to one direct `preadv` for four adjacent 557,056-byte
records in the local expert image. Both ranks returned `EINVAL`; no throughput
measurement or serving request was involved.

Alignment of each separate base alone is therefore insufficient for this
specific grouped-read layout on the tested systems. This does not isolate the
kernel's exact constraint or rule out all other vector layouts. A coalescing
implementation should use a proven contiguous staging layout or establish an
alternative, while preserving individual chunk lifetime and CUDA registration
semantics. No serving code or configuration changed.

## Coalesced adjacent expert reads: serving screen

A temporary opt-in prototype allocated one contiguous host staging slab,
registered it once with CUDA and kept separate logical leases and completion
fences for every record. A worker opportunistically combined up to four queued,
physically adjacent records into one read when neighboring staging slots were
immediately free; it did not wait for a batch to form. It preserved per-record
checksum validation and trace events, demand priority, bounded staging and
individual ticket completion. The control used separate staging allocations
and individual reads. Both used independent decode-miss completion, eight
readers, 256 staging buffers, twelve execution slots, the same expert budget,
one-million-token context and depth-three speculation.

The expert-store, read-queue and cache suites passed with coalescing enabled,
including a test that forced four queued records to share one syscall and
verified that their leases remained independent. The B12x paging arithmetic and
paged-prefill GPU fixtures also passed. A same-binary serving screen, SHA-256
`cc7928c34e120a00034a5414dc15ff7401dbeb58904817e93db1eb5ab346e186`
(source based on `27fc3aa7` plus the prototype), ran control first and candidate
second. Each fresh process served the same prefix-enabled four-request tool
sequence and three 2209-token recomputations. This is a focused single-order
diagnostic, not a benchmark campaign row.

Initial cold prefill was 38.537 versus 39.247 seconds. The three continuation
wall times totaled 37.874 versus 38.118 seconds; engine decode intervals totaled
32.915 versus 33.158 seconds. Rank 0's expert read syscalls fell from 211,932
to 63,989 for the initial request and from 90,612 to 28,257 across the three
continuations. Both ranks read the same expert bytes: 109.950 GiB initially and
47.009 GiB across continuations. The third short recomputation read zero expert
bytes on both ranks, with prefill 1.592 versus 1.601 seconds and decode 52.80
versus 52.85 tokens/s. Normalized messages, usage and speculative acceptance
matched; both ranks had zero swap at captured boundaries.

The large reduction in syscall count did not translate to a serving gain in
this comparison. The prototype was removed and the prior serving binary and
configuration restored. This does not establish the cause of the result or
rule out other batching widths, storage devices, workloads or scheduling
approaches; no further width sweep was run.

## Planner frequency memoization: serving screen

A temporary prototype memoized each expert's decayed frequency within an
observation epoch, avoiding repeated exponential calculations during admission
without changing the residency policy. A deterministic planner comparison and
the host cache suite passed. A same-binary, control-then-candidate screen used
the four-request prefix-enabled tool sequence and three 2209-token
recomputations, with eight readers and independent miss completion. This was a
focused diagnostic, not a benchmark campaign row.

Cold prefill was 38.583 versus 38.539 seconds; the three continuation wall
times totaled 38.257 versus 38.012 seconds. Both ranks read 109.950 GiB on
the initial request and 47.009 GiB across continuations in both arms. The
final recomputation read no expert bytes, with prefill 1.600 versus 1.604
seconds and decode 52.82 versus 52.79 tokens/s. Response content, usage and
speculative acceptance matched, and captured swap was zero on both ranks.
The small continuation difference is within variation seen in previous screens;
the prototype was removed and the prior serving binary restored.

## Recent-layer decode prefetch replay

A fixed-route replay of the previously captured demand stream screened a
simple next-visit predictor: at each decode layer visit, offer currently
nonresident experts used in that layer's previous N decode visits. The replay
uses the earlier approximate whole-expert admission model, free staging at
batch boundaries, and observed routes; it does not simulate prefetch changing
residency, queue contention or latency. The same result appeared on both rank
traces. Only the three tool continuations were counted; the cold request and
short recomputations established the surrounding cache history.

Of 3,443 simulated continuation decode misses per rank, none had appeared in
the preceding four decode visits to that layer, and one had appeared in the
preceding eight or sixteen visits. The eight-visit predictor would offer 99
nonresident experts, only one of which was needed on the immediately following
visit; sixteen visits offered 563 with the same one useful expert. These are
upper-bound opportunities for this particular immediate-next-visit predictor,
not measurements of serving speed or a rejection of other prediction methods.
The result does not justify implementing this predictor on the tested stream.

## Lossless compression feasibility screen

A read-only sample from rank 0's local NVFP4 expert image tested whether
record-level lossless compression might reduce miss traffic. Four 8 MiB image
slices compressed to approximately 85.3–85.6% of their original size with
Zstandard level 1. A separate 64-record check excluded on-disk padding and
confirmed about 14% reduction after rounding each compressed record to a 4 KiB
direct-I/O extent. Single-thread decompression in that check reached roughly
954 MiB/s; this is a local microtest, not a serving or physical bandwidth
measurement.

A 512-record prototype sampled experts across the image, checked each
compression/decompression round trip, and read with eight direct-I/O workers.
Zstandard reduced 272 MiB to 233.8 MiB, but its read-and-decompress timings
were slower than raw reads in this small test. The raw records came from
scattered image offsets while compressed records were in a compact temporary
file, so those timings do not isolate codec overhead or predict serving.

LZ4 reduced the same 272 MiB to 253.0 MiB after 4 KiB alignment. A second
screen wrote both raw and compressed records to compact temporary files, flushed
them, warmed each path once and alternated order across four direct-I/O pairs.
Raw reads took 38.80–39.07 ms; compressed reads plus decompression took
36.61–38.22 ms. Those files were unlinked temporary files on rank 0's local
SSD, and every sampled record passed a byte-for-byte round trip. This small
screen does not include serving, GPU or Engram contention, cache admission,
both ranks, or a larger working set. It does not establish a serving speedup.

No image format or serving configuration changed. A complete separately
compressed image would require substantially more local disk than the
approximately 17/22 GiB free on ranks 0/1 at the time. A bounded overlay or
other incremental conversion would be needed to test serving without replacing
the existing images. The limited isolated LZ4 result is a candidate for such a
test, not evidence that the conversion is worthwhile.

## Direct admission of packed decode misses

`DGPP_EXPERT_DIRECT_NATIVE_ADMISSION=1` enables an optimization for whole-expert
independent decode misses. When an admitted expert was wholly absent and an
aligned free cache extent is available, the cache publishes it directly in the
prepared native layout. If the execution callback already packed the expert,
the CUDA backend copies that packed payload into the cache; otherwise it packs
the live canonical execution slot. Fragmented space, partial admissions and
fractional residency retain the existing chunk path. Cache publication and
GPU reader/write fences remain ordered. No expert budget or model arithmetic
changes. The setting is opt-in and enabled in the tested deployment.

A checkpoint-shaped B12x GPU fixture with independent miss completion verifies
that direct admission is exercised, copies a prepared execution payload, and
preserves output across cold misses, resident graph reuse and cache pressure.
The 28 host cache tests also passed. This is not a broad quality evaluation.

Two opposite-order same-binary comparisons used SHA-256
`242a2cab726077dcf78e7c200b12a956dfca38e4ff48f64ef5f42743c846f64b`
on both ranks, eight readers, fixed depth three and the existing memory budget.
Each fresh process served the prefix-enabled four-request tool sequence and
three 2209-token recomputations. These are focused diagnostics, not benchmark
campaign rows. The control-first pair's three continuation wall times totaled
39.308 versus 36.853 seconds; the candidate-first pair totaled 38.262 versus
37.139 seconds for control versus candidate. Their summed engine decode times
were 33.937 versus 31.958 seconds and 33.281 versus 32.249 seconds. Cold
prefill was 39.834 versus 38.541 seconds in the first pair and 38.514 versus
38.612 seconds in the second; the decode-only change does not explain the
first pair's cold difference, so it is not attributed to the optimization.

Both ranks read 109.950 GiB initially and 47.009 GiB across continuations in
every arm. The final recomputation read zero expert bytes, with control versus
candidate prefill 1.621 versus 1.601 seconds and decode 52.87 versus 52.92
tokens/s in the first pair; the reverse pair was 1.602 versus 1.599 seconds
and 52.52 versus 52.52 tokens/s. Normalized messages, usage and speculative
acceptance matched, and captured swap was zero on both ranks. These limited
comparisons support a modest improvement for this tool sequence, not a general
throughput claim. The earlier direct-admission prototype repacked from canonical
scratch and did not show a consistent gain across two orderings.

## Previous-layer decode prediction replay

A separate causal fixed-route screen trained expert co-occurrence counts from
earlier consecutive decode-layer visits, then used the immediately preceding
layer's selected experts to rank currently nonresident experts in the next
layer. Counts were updated only after evaluating each prediction; prefill
events and nonconsecutive layers broke the input sequence. This tests one
simple online predictor, not a learned router or a serving implementation.

On the saved three tool continuations, both rank traces gave the same result.
The approximate whole-expert replay counted 3,443 decode misses per rank,
3,356 of them at consecutive-layer transitions. Offering the top one
prediction per eligible visit made 8,713 offers, of which 676 were actually
demanded in the next layer. Offering the top three made 24,538 offers for 900
hits; six made 45,453 offers for 965 hits. A confidence cutoff based on the
training co-occurrence fraction offered 28 predictions and hit 20 at 5%; a
10% cutoff offered none. These figures assume every offer can be made for
free, without affecting cache contents, staging, storage bandwidth or timing.
They neither establish useful prefetch lead time nor translate directly into
latency. The low coverage at the conservative cutoff and low precision without
it do not justify deploying this predictor on the tested sequence. Other
predictors and workloads remain untested.

## Independent-miss copy versus read wait

An opt-in prototype changed independent decode-miss scheduling to wait for a
demanded read when reads were outstanding and staging was available, instead
of first waiting for the oldest GPU staging copy. It retained the copy wait
when no read was pending or staging was held by copies and prefetches. The
28-test host cache suite passed with the candidate enabled, including its
bounded ready-batch fixture; this suite does not prove all scheduling cases.

A same-binary control-then-candidate screen on both ranks used binary SHA-256
`5b505fb2299f3d6fe1d35212a87601fab083a049c83f6002055f782095cef84c`, the
prefix-enabled four-request tool sequence, then three 2209-token recomputations.
The candidate did not improve serving: initial cold prefill was 38.618 versus
38.712 seconds and the three continuation wall times totaled 37.489 versus
37.915 seconds. Each rank read 109.950 GiB initially and 47.009 GiB across
continuations in both arms. The final recomputation read zero expert bytes;
prefill was 1.598 versus 1.599 seconds and decode was 52.85 versus 52.49
tokens/s. Response content, usage and speculative acceptance matched, and
captured process swap was zero on both ranks. This single ordered comparison
is a focused diagnostic, not a benchmark campaign or proof of a regression.
The prototype was removed and the prior serving configuration restored.

## B12x host-upload buffer ring

An opt-in prototype replaced the single pinned host packet used by B12x
`prepare()` with an eight-buffer ring. It waited for an upload only before
reusing that buffer; device packet updates remained ordered on the execution
stream. The control used the first buffer only. Both modes reserved the same
ring, so this comparison does not measure the additional pinned-memory cost
against the prior binary. The checkpoint-shaped B12x paging fixture passed
with independent completion and the candidate enabled.

The same binary on both ranks, SHA-256
`62ee0971000e98f736f3a1e36aff5b02c76517c6c69973c354ea39fccf8aee40`, ran
control then candidate on the four-request prefix-enabled tool sequence and
three 2209-token recomputations. This is a focused diagnostic, not a benchmark
campaign row. Cold prefill was 38.477 versus 38.600 seconds and the three
continuation wall times totaled 37.854 versus 38.220 seconds. Both ranks read
109.950 GiB initially and 47.009 GiB across continuations in each arm. On the
final zero-read recomputation, prefill was 1.595 versus 1.612 seconds and
decode was 52.58 versus 52.71 tokens/s. Response content, usage and speculative
acceptance matched; captured swap was zero on both ranks.

The screen found no useful improvement; it does not establish that upload
synchronization is never material in other workloads. The prototype was
removed and the prior service binary and settings restored.

## Direct packing of staged decode chunks: not selected

A prototype writes each wholly missing NVFP4 expert chunk directly from its
staging buffer into the B12x packed layout, reconstructing canonical weights
only for a fallback that needs them. The checkpoint-shaped GPU fixture passed
with both direct native admission enabled and disabled, including pressure and
graph replay. The host cache tests also passed.

A focused two-rank serving screen did not complete: the control's cold
24,317-token prefill took 38.500 seconds, and its three tool continuations
took 37.269 seconds combined. On the candidate's first request, prefill
completed but the captured decode graph stalled and the service reported CUDA
error 719 after approximately 60 seconds. The kernel log recorded Xid 43.
The direct cause has not been established. There is no candidate throughput
result, and the isolated fixture is insufficient evidence that this path is
safe for the full serving graph. A shorter 2,209-token cold request did
complete with the candidate. On the long request, an opt-in event marker
showed the transfer stream reached the queued pack kernel while the service
stream was idle. Reducing the kernel block size from 256 to 64 threads did not
remove the stall. The marker does not establish whether the kernel started;
the scheduling hypothesis remains unproven. The prototype was removed and
the previous binary and configuration restored. Revisit only with a concrete
way to resolve the stall and compare cold, partially warm, and fully warm
serving paths.

## Fused packing of reconstructed experts

An opt-in prototype combined the four NVFP4 code and scale packing kernels
and the global-scale copy into one kernel, reading the existing canonical GPU
reconstruction. It left staging and cache admission unchanged. The
checkpoint-shaped packing round trip and B12x paging fixture passed, including
graph replay and cache pressure.

Two opposite-order focused comparisons used the same binary on both ranks,
SHA-256 `f48203131ad723263cc88fb9b3d5fb36f46ed149a4029bdf167d57f078eae73c`,
built from `82d0ef24` plus the prototype. Each fresh process ran the four-request
prefix-enabled tool sequence and three 2,209-token recomputations. These are
diagnostic comparisons, not benchmark campaign rows.

| Measurement | Candidate-first: control / fused | Control-first: control / fused |
| --- | --- | --- |
| Initial cold prefill | 42.611 / 39.143 s | 38.578 / 39.193 s |
| Three continuation wall times, summed | 38.631 / 36.843 s | 37.323 / 37.272 s |
| Three continuation engine decode times, summed | 33.153 / 31.974 s | 32.426 / 32.334 s |
| Third short recomputation prefill | 1.598 / 1.655 s | 1.601 / 1.664 s |
| Third short recomputation decode | 52.72 / 52.56 tok/s | 52.48 / 52.49 tok/s |

Each rank read 109.950 GiB initially and 47.009 GiB across continuations in
every arm. The third recomputation had zero expert reads on both ranks.
Normalized messages, tool arguments, usage and speculative acceptance
matched within each pair, and both ranks had zero swap at the captured
boundaries. The reverse-order continuation result did not reproduce the first
pair's apparent improvement. The prototype was removed and the previous
service binary and settings restored; the experiment does not establish a
meaningful serving gain or a general conclusion about kernel fusion.

## Decode probation and layer quotas: fixed-route screens

Two offline cache screens used rank 0's existing 14,040-event demand trace
from the four-request tool sequence and three short recomputations. These are
approximate grouped-demand simulations with fixed observed routes and free
per-event staging, not serving measurements. They use the recorded equal-sized
whole-expert capacity and the existing global-frequency ranking; they do not
reproduce every admission, staging or timing detail. The replay control read
109.950 GiB initially and 47.009 GiB across continuations; the three short
recomputations read 33.385, 6.275 and zero GiB.

One candidate delayed admission of a missing decode expert until it had been
selected twice within a recent window. Windows of 80, 320, 1,280 and 5,120
grouped events produced 65.145, 60.298, 55.862 and 53.686 GiB respectively
across continuations. The two shorter windows also retained reads on the final
short recomputation. This particular two-touch rule is not a promising
implementation candidate for the observed sequence.

A second candidate gave newly missed decode experts a bounded probationary
area and promoted them to the main population on a later hit. At capacities of
64, 256, 1,024 and 2,048 experts, continuation reads were 56.693, 47.327,
47.607 and 47.766 GiB. The 256-expert candidate's three short recomputations
read 34.244, 9.366 and zero GiB; larger probation areas had not reached a
zero-read third repeat. This does not rule out other scan-resistant policies.

An equal per-layer quota also lost on this trace: it read 191.129 GiB initially
and 59.187 GiB in continuations. Minimum per-layer allocations from 160 to
250 experts did not improve continuation traffic; higher floors also delayed
adaptation on the short recomputations. These simple allocations do not test a
learned, uneven allocation across layers. No serving code or settings changed.

## Small router-weight screen

An opt-in diagnostic captured rank 0's target prefill expert IDs and router
weights for the prefix-enabled four-request tool sequence followed by two
2,209-token recomputations. It synchronized the route stream to copy the
weights, so its timings are not performance results. The 380 captured calls
covered 598,920 token-layer rows and 3,593,520 routed assignments. They include
large initial-prefill chunks and shorter warmish continuation/recomputation
chunks; no fully resident third recomputation or decode-graph weights were
captured. Router weights and routes are workload data, not benchmark artifacts
to commit.

For each row, the screen divided each selected expert's weight by the sum of
the six routed weights. The smallest selected expert averaged 11.81% of that
routed sum. Hypothetically dropping every assignment below 5% would remove
0.112% of assignments and 0.026% of routed weight per row on average. Grouping
by each captured prefill call, 0.441% of the distinct expert demands would then
have no remaining rows. A 10% cutoff would remove 5.591% of assignments and
2.897% of the routed weight per row on average, with 4.679% of distinct
prefill-call expert demands disappearing. These percentages are aggregate
opportunity bounds for this trace: they do not distinguish resident from
missing experts, measure execution time, or account for the shared expert.

As an additional proxy only, grouping adjacent prefill rows in fours would
eliminate 0.113% and 5.592% of distinct four-row expert demands at the 5% and
10% thresholds respectively. Those groups are not the actual speculative
verification routes, and the lost router weight is not a model-quality metric.
No expert was actually dropped and no quality change was measured. The small
threshold gives too little apparent headroom on this prefill sample to justify
implementing the approximation; a larger cutoff or decode-specific policy
would require separate route, quality and serving evidence. The diagnostic
instrumentation was removed and the previous server restored.

## Lossless NVFP4 scale compaction feasibility

A read-only screen sampled 512 evenly spaced 64-wide target-expert records
from each rank's local NVFP4 image: 240 MiB of code bytes and 30 MiB of scale
bytes per rank. Every sampled scale byte was one of 64, 72, 80, 88, 96, 104,
112 or 120. These E4M3 encodings have zero low three bits, so each sampled
scale can be represented exactly as a four-bit value and restored by shifting
left three bits. This is a sample, not validation of all records. An actual
implementation must check the values and preserve a fallback for unsupported
scales, formats and checkpoints.

For the deployed 5,120-by-64 chunk geometry, halving the three scale arrays
would reduce the 553,728-byte canonical chunk allocation to approximately
523,008 bytes before any new metadata or alignment. That is 5.55% less per
chunk, or an ideal capacity increase of about 5.87% at a fixed byte budget.
The code bytes themselves had marginal nibble entropy of about 3.78 bits on
each rank; this does not by itself justify an assumed lossless three-bit code.

An approximate fixed-route replay tested the value of extra capacity, without
implementing compaction. Starting from the recorded 10,359 whole-expert slots,
an ideal additional 150, 300 and 600 experts reduced simulated continuation
reads from 47.009 GiB per rank to 43.918, 41.378 and 35.850 GiB respectively.
Initial reads were 109.950, 109.782, 109.577 and 109.268 GiB. The third short
recomputation was zero-read in all four simulations. This is a grouped-demand
traffic screen with fixed routes and free staging, not evidence that a
compacted serving cache can realize those capacities or timings.

An isolated GB10 CUDA kernel packed and expanded synthetic supported scale
bytes exactly. Expanding the scale arrays of one, six and twelve complete
experts took approximately 4.97, 18.43 and 65.36 microseconds respectively
per launch in this screen. This excludes admission, allocation, graph
coordination, weight execution and SSD I/O, and is not a serving result.
The prepared B12x and chunk-direct paths currently expect byte-wide scales;
changing their cache representation without losing direct graph hits is the
main unresolved design problem. A hybrid or indirection scheme also needs to
charge its metadata, fallback and scratch bytes to the same memory budget.
No production implementation or setting changed, and the normal server was
restored after the kernel screen.

## Mixed native and compact-scale cache: route replay screen

A follow-up fixed-route simulation explored the cost of retaining a directly
executable native tier alongside a compact-scale tier. It uses the same saved
14,040 grouped demand events and initial empty cache as the earlier capacity
screen. The two ranks produced identical aggregate results. Each event admits
all demanded experts using free staging, updates the decayed popularity score,
then ranks residents. The highest-scored experts fill a fixed native tier; the
remainder fit in the same total byte budget at the ideal compact size. This is
an approximation, not the actual planner or a serving implementation. It
assumes all scale bytes are compressible and ignores metadata, alignment,
scratch space, allocation fragmentation, expansion and relocation cost.

| Native tier, fraction of original expert capacity | Native / compact expert slots | Initial reads GiB/rank | Continuation reads GiB/rank | Continuation compact hits | Compact hits on final zero-read repeat |
| --- | --- | ---: | ---: | ---: | ---: |
| 100% | 10,359 / 0 | 109.950 | 47.009 | 0 | 0 |
| 90% | 9,323 / 1,097 | 109.903 | 45.515 | 3,026 | 3,055 |
| 75% | 7,769 / 2,742 | 109.782 | 43.872 | 10,657 | 5,312 |
| 50% | 5,179 / 5,484 | 109.567 | 41.276 | 32,516 | 9,267 |
| 0% | 0 / 10,967 | 109.259 | 35.729 | 214,360 | 20,575 |

Hits count unique demanded experts per grouped event, not tokens or kernel
launches. They indicate possible scale expansions; they are not measured
expansions or latency. On the final short repeat, the naive 90% ranking also
made 3,055 compact-to-native promotions and 3,055 demotions despite zero SSD
reads. A simple multiplier favoring existing native experts reduced, but did
not eliminate, this movement: with a 4x multiplier, the same case had 2,833
compact hits and 2,335 promotions on that repeat. This heuristic also affects
ranking and is not a validated admission policy.

This screen strengthens the case for testing a mixed representation but warns
against moving experts between tiers whenever their popularity rank changes.
It does not establish that the read savings exceed scale expansion, pointer
publication or allocator costs. A practical prototype should preserve prepared
weight pointers, support independent byte-wide scale buffers and bounded
promotion/demotion, and account for every byte within the existing budget.
Its fast path must still support graph replay. Check actual scale values on
admission and fall back for unsupported data. Validate real partially warm
tool continuations first, with fresh-process cold and fully warm controls.
No serving code or deployment setting changed in this screen.

### Scale expansion and unsupported-value detection screen

A separate unintegrated CUDA prototype packed four byte-wide scales into a
16-bit word, set a device flag when any scale had a nonzero low three bits or
the high bit set, and expanded compact values back to byte-wide scales. On
Spark-1, it round-tripped synthetic supported values byte-for-byte for buffers
equal to one, six and twelve complete experts' scale arrays. It also correctly
classified every possible single byte value from 0 through 255, including
unsupported low bits and high bits. Unsupported values must take an uncompressed
fallback; the prototype's compact output for them is not valid.

In repeated synthetic expansion timing, one expert took about 4.1–4.7 us, six
about 10.3–10.4 us and twelve about 66.5–70.9 us. These are isolated kernels
over contiguous arrays, not individually addressed cache entries or captured
serving graphs. They do not include validation-flag handoff, allocations,
pointer publication, kernel scheduling, memory contention or end-to-end
execution, so they do not establish a speedup or a production cost estimate.

The prepared execution bindings already carry separate weight and scale
pointers, so a design can in principle leave prepared weights in place and
bind expanded scales separately. The current publisher and cache interface,
however, derive all pointers from one contiguous prepared expert, while the
allocator budgets fixed-size slots. A useful implementation therefore still
needs mixed-size accounting, a way to detect unsupported values without a
per-admission GPU-to-CPU synchronization, safe scale-buffer lifetimes across
graph execution, and eviction or relocation invalidation. This prototype is
not connected to serving and does not implement the required fallback. No
deployment setting was changed.

### Prepared graph with separately expanded scales

The optional `DGPP_TEST_MIXED_SCALES=1` branch of the checkpoint-shaped B12x
paging fixture now exercises the prepared graph with separate scale buffers.
It limits the synthetic source scales to exactly representable values, packs
and expands the prepared scales, and publishes the expanded scale pointers
alongside prepared weight and global pointers. The same captured
six-row graph produced bitwise-identical output and no host miss service, for
both whole-expert and fractional policy fixtures. The ordinary fixture also
passed with the option absent, preserving its arbitrary synthetic scale data.
Both were run on Spark-1 with `DGPP_MOE_B12X=1`.

This establishes that this graph's B12x binding format can consume separate
scale buffers without changing its computed output. The fixture allocates both
the original and expanded scales; it neither implements mixed-size cache
storage nor demonstrates a RAM or serving improvement. Production prefill,
real checkpoint values, eviction safety and
warm/cold end-to-end performance remain to be addressed before deployment.

### Classifying staged scales without a GPU round trip

The read queue now supports an internal, default-off `scan_nvfp4_scales`
option. After a successful read, the worker checks the three scale spans and
publishes `kExact` only if every byte has its high bit and low three bits
clear. Unsupported NVFP4 data is reported as `kUnsupported`; disabled scans
and other formats report `kUnknown`. The original staged record is unchanged,
and the classification shares its immutable lease and completion lifetime.
No serving configuration currently enables or consumes the classification.

The Spark-1 queue suite passed with exact scales, unsupported low and high bits,
disabled scans and MXFP4. A separate read-only screen of 512 evenly spaced
records on each local rank image found no unsupported records. CPU time inside
the scale scan, excluding reads and queue overhead, was approximately 3.04 and
2.96 microseconds per chunk on ranks 0 and 1 respectively (about 55 and 53
microseconds when multiplied by 18 chunks). This sample does not establish
complete checkpoint coverage, and this timing is not a serving measurement.
The cache must still aggregate all chunks for an expert and preserve a full
uncompressed fallback whenever the status is unknown or unsupported.

The cache now retains those per-record results when scanning is enabled and
exposes an expert-level result: every chunk must be known exact, and any
unsupported chunk rejects the expert. A successful subsequent read replaces
the corresponding record's result; the image is treated as immutable just as
for existing expert caching. No per-record classification array is allocated
when the scan is off. The Spark-1 host cache suite passed, including ordinary
and independent-completion reads, partial initial residency, an unsupported
chunk, and eviction. The status is not yet used to allocate compact storage.

### Compact prepared expert representation

The FP4 packing library now provides a compact prepared-expert layout and
stream-ordered conversion functions. Weights and global scales are preserved
exactly; supported scale bytes are packed two per byte. Expansion produces
independent byte-wide scale pointers while the returned weight and global
pointers still refer directly to the compact expert. The caller must establish
that all source scales are supported before compaction; a false verification
flag is rejected, and unsupported or unknown inputs need the uncompressed
path. The conversion itself does not inspect its inputs. Source, destination
and expansion buffers must be disjoint and remain alive through their GPU
consumers.

For the checkpoint's 5,120-by-1,152 TP-local expert, the prepared extent is
9,953,312 bytes and the compact extent is 9,400,352 bytes. A currently
executing compact expert also needs 1,105,920 bytes of expanded scale scratch;
that scratch, metadata, alignment and any fallback must be charged to the
same budget before claiming cache savings. The Spark-1 CUDA packing test passed
byte-exact round trips and graph replay at small and checkpoint geometry. The
optional B12x paging fixture now binds compact-resident weights and expanded
scales to the captured six-row graph; its outputs matched the normal prepared
path bit-for-bit. These are execution fixtures, not serving timings. The real
cache still uses its original fixed-size slots and does not yet call these
functions, so there is no memory saving or serving change in this revision.

### Compact overflow serving screen: not selected

A temporary whole-expert prototype reserved 1,000 compact prepared slots
inside the existing expert budget, reducing the ordinary native cache by the
same bytes. It scanned scale bytes on read and admitted only fully verified
evicted native experts to the compact tier. A compact hit expanded scales into
the already reserved execution scratch and unpacked the expert without a file
read. The ordinary uncompressed path handled unknown or unsupported values.
Compact entries used an LRU replacement order; both ordinary acquisition and
independent decode-miss completion could use them. The Spark-1 B12x fixture
confirmed a byte-exact compact reconstruction and a hit without a file read.

One control-first, same-binary two-rank serving screen compared this candidate
with the compact tier disabled. Both arms used binary SHA-256
`f70547da87a7e993b7d2e34b1f625e870b9a2b014379e72a0fe09a14986340d7`, built
from `31a2c65f` plus the temporary changes. The four-request prefix-enabled
tool sequence and three short recomputations are focused diagnostics, not
benchmark campaign rows.

| Measurement | Control | Compact overflow |
| --- | ---: | ---: |
| Initial 24,317-token prefill | 38.638 s | 39.834 s |
| Three tool-continuation wall times, summed | 37.040 s | 39.877 s |
| Three continuation engine decode intervals, summed | 32.129 s | 33.418 s |
| Continuation expert reads, per rank | 47.009 GiB | 48.459 GiB |
| Third short recomputation prefill | 1.595 s | 4.831 s |
| Third short recomputation decode | 52.82 tok/s | 42.21 tok/s |
| Third short recomputation expert reads, per rank | 0 GiB | 12.740 GiB |

The candidate registered 2,557 compact hits per rank across the continuations,
but also performed many compact admissions and replacements. The changed
allocation and LRU behavior produced more continuation reads and did not
converge to the control's zero-read short repeat. Normalized messages, tool
arguments, usage and speculative acceptance matched; both ranks remained at
zero swap at captured boundaries. This comparison does not identify the exact
cause or rule out other tier policies. It does show that this implementation
is not a useful speedup for the observed warmish workload. The temporary
cache integration was removed and the previous production binary and settings
were restored. The separately tested representation and read classification
remain available for a substantially different design.

### Compact overflow: readiness and promotion follow-up

Two fixes were tested on the temporary prototype. Lookahead and prefetch now
recognised compact-resident experts rather than reading them unnecessarily;
ordinary acquisition also allowed popular compact hits to be promoted back to
native residency. The checkpoint-shaped B12x fixture passed reconstruction,
read suppression, promotion and independent-completion eviction checks.

A candidate-first same-binary serving screen used SHA-256
`36678d34f7b67d4dccad5b83f4dbb597210b876f8f8e90e74bbe6defb6d842e1`
with 1,000 compact slots versus zero. These are focused diagnostics, not
benchmark campaign rows. The same prefix-enabled four-request tool sequence
was followed by three short recomputations.

| Measurement | Control | Compact with promotion |
| --- | ---: | ---: |
| Initial 24,317-token prefill | 38.554 s | 38.536 s |
| Three continuation wall times, summed | 37.172 s | 39.209 s |
| Three continuation engine decode intervals, summed | 32.268 s | 33.658 s |
| Continuation expert reads, per rank | 47.009 GiB | 47.747 GiB |
| Third short recomputation prefill | 1.605 s | 3.977 s |
| Third short recomputation decode | 52.70 tok/s | 38.69 tok/s |
| Third short recomputation expert reads, per rank | 0 GiB | 12.822 GiB |

Normalized choices, tool arguments, usage and speculative acceptance matched;
captured swap was zero on both ranks. The promotion change did not recover the
warm working set or improve the tool sequence. The integration was removed and
the original serving configuration restored. This result does not rule out a
different compact-cache design; it does not demonstrate a serving benefit from
this one.

### Decode-aware admission on the current miss path

The separate decode-history admission policy was revisited after independent
decode-miss completion and direct native admission were enabled. A positive
`decode_admission_boost` gives recent decode selections additional weight
only when scoring demand admission during decode; prefill admission and
periodic planning retain the ordinary history. The default is zero. The
23-test residency suite and three configuration tests passed on Spark-1.

The two-rank focused screens used the same binary, SHA-256
`a95179efb496d1f515467c523cca5e2a929c19b9c3809bf5afe4e4d333f18593`,
with boost 0 or 16, equal expert budgets, eight readers, independent misses,
direct native admission and depth-three speculation. Each fresh process served
the prefix-enabled four-request tool sequence and three short recomputations.
These are diagnostics, not benchmark campaign rows.

| Measurement | Candidate-first: control / boost | Control-first: control / boost |
| --- | --- | --- |
| Initial prefill | 38.400 / 38.518 s | 38.480 / 45.888 s |
| Three continuation wall times, summed | 37.667 / 35.637 s | 37.187 / 35.751 s |
| Three continuation decode intervals, summed | 32.633 / 30.691 s | 32.301 / 30.820 s |
| Continuation expert reads, per rank | 47.009 / 42.611 GiB | 47.009 / 42.611 GiB |
| Third short recomputation prefill | 1.603 / 1.605 s | 1.605 / 1.612 s |
| Third short recomputation decode | 52.63 / 52.60 tok/s | 52.87 / 52.37 tok/s |

The last recomputation read zero expert bytes on both ranks in both arms.
Normalized outputs, usage and speculative acceptance matched; captured swap
was zero. The second boosted cold prefill was slower even though each rank
read the same 109.950 GiB of expert data. Its cause was not established. A
third fresh boosted process took 38.764 s for initial prefill and 35.558 s for
the continuations, with the same read volumes and acceptance. This does not
explain the outlier or establish results for other workloads or concurrency.
The repeatable improvement on these partially warm continuations and the
preserved warm anchor support selecting boost 16 for this Spark deployment.

### Stronger decode admission weight

A candidate-first focused screen compared the selected boost of 16 with 64
using the same binary and deployment as the preceding experiment. Both fresh
processes served the same prefix-enabled tool sequence and three short
recomputations; these are diagnostics, not benchmark campaign rows. Initial
prefill was 38.591 versus 38.640 seconds. The three continuation wall times
totalled 35.589 versus 35.551 seconds, and their engine decode intervals
totalled 30.686 versus 30.635 seconds. Both ranks read 42.611 GiB across the
continuations in either arm. On the final short recomputation, prefill was
1.608 versus 1.605 seconds and decode was 52.60 versus 52.74 tokens/s, with
zero expert reads on both ranks. Normalized choices, usage and speculative
acceptance matched, and captured swap was zero.

The difference is too small to establish a serving benefit; the stronger
weight also did not reduce reads on this sequence. Boost 16 remains selected.
This single screen does not establish that the two weights behave identically
on other workloads.

### Parallel small Engram gathers

`DGPP_ENGRAM_PARALLEL_SMALL=1` gives each direct Engram row cache a persistent
16-worker pool. Gathers of at most 256 rows still use the serial path when
fewer than two requested rows are absent; otherwise the existing hash-set
shards are processed concurrently. Large gathers retain their prior execution
path. The cache lock serializes callers and each shard retains its own buffers;
output offsets preserve request order. This adds 32 worker threads per rank for
the two-table deployment. The default remains off.

The Spark-1 direct-row cache suite passed with the option enabled and disabled,
including concurrent small gathers, warm reuse and failed reads. Two focused
same-binary, two-rank screens used SHA-256
`985c5e118dacd69eb4181074a0dfe0022d182fdaa437a1d605b83be9946249e0`,
decode admission boost 16 and the existing expert budget. Fresh processes ran
the prefix-enabled four-request tool sequence and three short recomputations;
these are diagnostics, not benchmark campaign rows.

| Measurement | Candidate-first: serial / parallel | Control-first: serial / parallel |
| --- | --- | --- |
| Initial prefill | 38.672 / 38.671 s | 38.578 / 38.753 s |
| Three continuation wall times, summed | 35.495 / 33.660 s | 35.319 / 33.725 s |
| Three continuation decode intervals, summed | 30.588 / 28.791 s | 30.378 / 28.839 s |
| Continuation expert reads, per rank | 42.611 / 42.611 GiB | 42.611 / 42.611 GiB |
| Third short recomputation prefill | 1.605 / 1.609 s | 1.611 / 1.613 s |
| Third short recomputation decode | 52.90 / 52.84 tok/s | 52.75 / 52.45 tok/s |

Both ranks had zero expert reads on the final repeat and zero swap at captured
boundaries. Normalized outputs, usage and speculative acceptance matched.
The results support selecting the option for this deployment's tested tool
sequence; they do not quantify long-running memory cost, other workloads or
concurrent-serving throughput. The exact contribution of overlapping storage
reads versus other host scheduling effects was not isolated.

### Selective Engram worker wakeups

A temporary prototype gave the persistent Engram workers individual wakeups
and notified only shards with requests. A same-binary candidate-first screen
used SHA-256 `7a6dbb793017c44d8d66e95735f431cd193e13490d58fd0040e7391f005a36f7`
and the same prefix-enabled tool sequence and three short recomputations as
the preceding experiment. The direct-row tests passed in both modes,
including changing active worker sets. This is a focused diagnostic, not a
benchmark campaign row.

Waking all workers versus selective wakeups produced initial
prefill of 38.562 versus 38.614 seconds. The three continuation wall times
totalled 33.973 versus 33.993 seconds; engine decode totalled 29.084 versus
28.981 seconds. Each rank read 42.611 GiB of expert data in either arm. The
final short recomputation read no expert data, with prefill 1.616 versus
1.614 seconds and decode 52.81 versus 52.78 tokens/s. Normalized responses,
usage and acceptance matched, and captured swap was zero on both ranks.

No useful improvement was established, so the prototype was removed and the
previous parallel Engram implementation retained. Other worker counts or
workloads were not tested by this screen.

### Reusing Engram workers for large gathers

A temporary option reused the existing 16-worker pool for gathers larger than
256 rows, instead of creating per-gather threads. It retained the existing
sharding, sorting and ordered outputs. The Spark-1 direct-row cache tests passed
in both modes, including concurrent gathers and a failed large read.

One candidate-first focused screen used the same binary on both ranks,
SHA-256 `acb659f0a7f6dc36558aa9928d758e4bcb27196b0d1dad7d880e74bfd7df5626`,
and the prefix-enabled tool sequence followed by three short recomputations.
This is a diagnostic, not a benchmark campaign row. Initial prefill was
38.502 seconds with existing large-gather threads versus 38.899 seconds with
the persistent pool. Three continuation wall times totalled 34.017 versus
33.972 seconds; their engine-prefill intervals totalled 4.836 versus 4.819
seconds and decode totalled 29.042 versus 29.028 seconds. Each rank read
42.611 GiB of expert data across continuations. On the third short repeat,
both ranks read zero expert bytes; prefill was 1.613 versus 1.606 seconds and
decode was 52.70 versus 52.57 tokens/s. Normalized responses, usage and
speculative acceptance matched, with zero swap at captured boundaries.

The comparison found no useful serving improvement. The option was removed
and the previously selected binary restored; other large-gather shapes and
concurrency levels remain untested.

### Doubling the Engram row cache

A temporary option doubled each direct Engram table's row-cache slots, adding
about 72.4 MiB per rank for this checkpoint while leaving expert capacity
unchanged. The Spark-1 direct-row cache tests passed with both capacities.
One candidate-first, same-binary screen used SHA-256
`9d3be8059ff0a26464d4a36aa22c1cd48bffed9352c2c59c401be7e37f3b0e63`
and the same prefix-enabled tool sequence and three short recomputations. This
is a focused diagnostic, not a benchmark campaign row.

Initial prefill was 38.658 seconds with the existing cache versus 38.545 with
the larger cache. Three continuation wall times totalled 33.723 versus 33.887
seconds; engine decode totalled 28.766 versus 28.949 seconds. Expert reads
were 42.611 GiB per rank in each arm. Rank 0 process reads minus expert reads
were approximately 0.135 versus 0.133 GiB across continuations; this residual
includes other process I/O and is not an exact Engram read counter. On the
third short recomputation, both ranks read zero expert bytes; prefill was
1.606 versus 1.611 seconds and decode was 52.82 versus 52.77 tokens/s.
Normalized responses, usage and speculative acceptance matched, and captured
swap was zero on both ranks.

This sequence did not benefit from the extra row-cache memory. The prototype
was removed and the prior serving binary restored. Different prompt reuse and
longer workloads may have different cache working sets.

### Reader count on longer tool continuations

The eight-versus-twelve-reader comparison was revisited with decode admission
boost 16 and parallel small Engram gathers enabled. Both ranks used the same
production binary, SHA-256
`985c5e118dacd69eb4181074a0dfe0022d182fdaa437a1d605b83be9946249e0`,
the same expert budget, 256 staging buffers and twelve execution slots. Each
fresh process ran the four-request prefix-enabled tool sequence with a
1,024-token continuation cap, then three short recomputations. This single
candidate-first comparison is a focused diagnostic, not a campaign row.

Initial prefill was 38.678 seconds with eight readers versus 38.312 with
twelve. The three continuation wall times totalled 74.887 versus 74.287
seconds; engine decode totalled 69.264 versus 68.623 seconds. Each rank read
83.308 GiB of expert data across the continuations in either arm. The two
prose responses completed at 413 tokens each; the code response reached the
1,024-token cap. The third short recomputation read zero expert bytes on both
ranks, with prefill 1.622 versus 1.616 seconds and decode 52.63 versus 52.59
tokens/s. Normalized outputs, usage and speculative acceptance matched, and
captured swap was zero on both ranks.

The 0.600-second difference over the continuation sequence is too small for
this single pair to establish a reliable improvement. Eight readers remain
selected. The capped code response and one sequential request stream do not
establish behavior for uncapped work or concurrent serving.

### Half-sized Engram worker pools

A temporary candidate reduced each small-gather Engram pool from 16 to eight
workers, with each worker serving two existing cache shards. Each rank would
therefore have 16 rather than 32 persistent Engram workers for the two-table
deployment. The Spark-1 row-cache suite passed in both modes. A candidate-first
focused screen used the same binary, SHA-256
`03ad829c2f486d64e7064244b74c9bc958d2887e31bf2b057c96ea7cba54278b`,
for the prefix-enabled tool sequence and three short recomputations. This is a
diagnostic, not a benchmark campaign row.

Initial prefill was 38.540 seconds with 16 workers per table versus 38.879
with eight. Three continuation wall times totalled 33.778 versus 33.952
seconds, and engine decode totalled 28.885 versus 29.048 seconds. Each rank
read 42.611 GiB of expert data across continuations in either arm. The third
short recomputation read zero expert bytes per rank, with prefill 1.612 versus
1.607 seconds and decode 52.73 versus 52.68 tokens/s. Normalized responses,
usage and speculative acceptance matched, and captured swap was zero.

The smaller pool did not improve this single-request screen, so the prototype
was removed and the 16-worker-per-table configuration restored. This does not
establish the effect of reduced thread count under concurrent serving.

### Fractional residency on the current serving path

A candidate-first focused comparison revisited adaptive fractional residency
using the current production binary, SHA-256
`985c5e118dacd69eb4181074a0dfe0022d182fdaa437a1d605b83be9946249e0`,
with decode admission boost 16, parallel small Engram gathers, eight readers
and the same overall expert budget. Fractional residency used a zero minimum
and an initial fraction of 0.05, rounded to one chunk per expert; whole-expert
caching started empty. Both used 12 configured execution slots, although the
fractional ready-miss path reserves one of them for its own operations. Direct
native admission of fully absent misses applies only to the whole-expert path.
This is a comparison of the deployed implementations, not an isolated cache
policy comparison or a benchmark campaign row.

Each fresh process ran the same prefix-enabled four-request tool sequence and
three short recomputations. Whole versus fractional initial prefill was 42.683
versus 37.584 seconds, but fractional had already read 7.969 GiB of expert
chunks per rank before the first request, versus zero for whole-expert. First
request expert reads were 109.950 versus 104.629 GiB per rank; the first
request timing therefore excludes different startup work and does not
establish a cold-start improvement.

The three continuation wall times totalled 35.141 versus 36.409 seconds;
engine decode totalled 29.766 versus 30.943 seconds. Each rank read 42.611
versus 42.605 GiB of expert data across those continuations. The third short
recomputation read zero expert bytes on both ranks, with prefill 1.613 versus
1.608 seconds and decode 52.80 versus 52.61 tokens/s. Normalized choices, usage
and speculative acceptance matched, and captured swap was zero on both ranks.

Fractional residency reached the fully warm state but did not improve this
warmish sequence. The whole-expert configuration remains selected. The single
ordering and distinct startup/execution paths limit causal attribution; this
does not rule out other initial allocations or fractional implementations.

### Fractional direct native admission screen

An opt-in prototype extended ready-miss direct native admission to fractional
residency when an expert was wholly absent, admission selected all its chunks,
and an aligned whole-expert extent was free. Fractional demotions continued to
use the separately reserved scratch slot. A checkpoint-shaped GPU fixture
exercised exact reconstructed expert bytes through admissions and evictions;
a constructed churn sequence with previously prepared native experts produced
77 direct admissions. Without native preparation, the same churn produced no
direct admissions because no suitable contiguous extent was found. The host
cache suite also passed. The prototype was removed after the serving screen;
these tests are not a general proof of every possible partial-demotion sequence.

The focused control-first comparison used a single binary on both ranks,
SHA-256 `d6ea57586f94a6a9159e7675055c42279d7ba86389f7e4e939680b79f986e5da`
(source `18c4a912` plus the prototype),
with fractional policy, zero minimum, initial fraction 0.05, the existing expert
budget, eight readers, depth-three speculation and the selected Engram settings.
Only the prototype switch differed. Both fresh processes served the same
prefix-enabled four-request tool sequence followed by three short uncached
recomputations. This is a diagnostic, not a benchmark campaign row.

| Measurement | Control | Fractional direct admission |
| --- | ---: | ---: |
| Initial prefill | 37.233 s | 37.742 s |
| Three continuation wall times, summed | 34.541 s | 35.756 s |
| Three continuation decode intervals, summed | 29.592 s | 30.322 s |
| Continuation expert reads per rank | 42.605 GiB | 42.605 GiB |
| Third short recomputation prefill, zero reads | 1.608 s | 1.610 s |
| Third short recomputation decode | 52.92 tok/s | 52.99 tok/s |

Initial-request reads were 104.629 GiB per rank in both arms, after the same
fractional startup population. Normalized outputs, usage and speculation
acceptance matched; captured swap was zero on both ranks. The candidate
increased native-pack counts during the continuations, indicating its path was
exercised, but did not improve elapsed time in this one ordering. This does not
rule out an improvement for a different allocation or workload. Whole-expert
caching remains the selected deployment.

### Carrying decode popularity into prefill admissions

A temporary prototype applied the existing separately aged decode popularity
history to **prefill** demand-admission scores as well as decode admission
scores. Its prefill boost was 16, while the deployed decode boost stayed at 16;
periodic planning, observation and budgets were unchanged. Decode history ages
only on decode selections, so this can protect an old decode working set during
an extended prefill. A planner test verified that a short prefill could protect
an earlier decode resident, while a sufficiently sustained new demand could
still replace it. The standard planner suite passed with the candidate disabled.
The prototype was removed after the serving screen.

The control-first focused comparison used the same binary on both ranks,
SHA-256 `82f3505c30331f73e36734e38a03563d11463b2782d448776aa882886c4cdd52`
(source `1e5125ea` plus the prototype), and changed only the prefill boost.
Both arms used whole-expert caching, eight readers, the selected budget and
fixed depth three. Each fresh process ran the four-request prefix-enabled tool
sequence and three 2,209-token uncached recomputations. This is a diagnostic,
not a benchmark campaign row.

| Measurement | Control | Prefill decode boost 16 |
| --- | ---: | ---: |
| Initial prefill | 38.747 s | 38.567 s |
| Three continuation wall times, summed | 33.908 s | 34.044 s |
| Three continuation decode intervals, summed | 29.014 s | 28.965 s |
| Continuation expert reads per rank | 42.612 GiB | 43.526 GiB |
| Recomputed short-prefill times | 7.794 / 2.864 / 1.608 s | 6.336 / 4.987 / 4.797 s |
| Third short recomputation reads per rank | 0 GiB | 22.123 GiB |
| Third short recomputation decode | 52.77 tok/s | 52.44 tok/s |

Initial-request reads were 109.950 GiB per rank in both arms. Normalized
outputs, usage and speculative acceptance matched; captured process swap was
zero on both ranks. This particular boost did not improve the normal tool loop
and held back prefill adaptation after the workload switch. Its third repeat
was not fully resident, so the decode rates are not a matched zero-read anchor.
The result does not rule out a differently aged or bounded cross-phase policy.

### FP4 prefill activation-cache hint screen

The resident FP4 prefill kernel prefetches activation lines into L2 with an
`evict_last` hint before its existing asynchronous loads. A temporary switch
disabled only that hint; the activation loads, weight copies and arithmetic
were unchanged. Grouped FP4/MXFP4 segment tests and the checkpoint-shaped B12x
paging fixture passed with the hint disabled. The candidate was removed after
the serving screen.

The same binary on both ranks, SHA-256
`53771a342f6e49f12e4f0e12cdcf5cea9b17822c74c678a3c1181180ffea3f8c`
(source `6ff25c13` plus the switch), ran control first and hint disabled
second. Both used the existing whole-expert budget, eight readers, depth three
and Engram settings; each fresh process ran the prefix-enabled four-request
tool sequence and three 2,209-token uncached recomputations. This is a focused
diagnostic, not a benchmark campaign row.

| Measurement | Hint enabled | Hint disabled |
| --- | ---: | ---: |
| Initial prefill | 38.499 s | 38.468 s |
| Three continuation wall times, summed | 33.817 s | 33.718 s |
| Three continuation decode intervals, summed | 28.886 s | 28.854 s |
| Continuation expert reads per rank | 42.612 GiB | 42.612 GiB |
| Third short recomputation prefill, zero reads | 1.601 s | 1.610 s |
| Third short recomputation decode | 52.77 tok/s | 52.74 tok/s |

Initial-request reads were 109.950 GiB per rank in both arms. Normalized
outputs, usage and speculative acceptance matched; captured process swap was
zero on both ranks. This single ordering did not show a meaningful gain from
removing the hint; it does not establish performance for other shapes or
hardware. The prior kernel and selected serving configuration were restored.

### Confidence scheduling at depth three on the current serving path

The current deployment's fixed depth three was compared with its existing
confidence scheduler enabled at the same maximum depth and default scheduling
parameters. The scheduler can stop drafting early and verify fewer rows. Both
arms used the deployed binary, SHA-256
`985c5e118dacd69eb4181074a0dfe0022d182fdaa437a1d605b83be9946249e0`,
whole-expert policy, the same cache budget, eight readers, the selected Engram
settings and one-million-token context. These are focused diagnostics, not
benchmark campaign rows or a general quality evaluation.

In a control-first screen, each fresh process ran the four-request
prefix-enabled tool sequence with 256-token continuation caps, followed by
three 2,209-token uncached recomputations. Initial prefill was 38.578 seconds
fixed versus 38.655 scheduled, and each rank read 109.950 GiB. The three
continuations totaled 33.738 versus 32.641 seconds wall, and 42.612 versus
39.987 GiB expert reads per rank. Their output text changed and all three hit
the cap. The final recomputation read zero expert bytes on both ranks in both
arms; prefill was 1.606 versus 1.593 seconds and decode was 52.83 versus
52.74 tokens/s. The short tool call and recomputation messages matched.

A second, candidate-first screen restarted each arm for the same tool sequence
with 1,024-token continuation caps. The following are service-level times and
engine decode rates, not matched-token execution timings:

| Continuation | Fixed: wall / decode rate / output tokens | Scheduled: wall / decode rate / output tokens |
| --- | --- | --- |
| Prose | 18.816 s / 24.34 tok/s / 413 | 16.032 s / 25.96 tok/s / 369 |
| Code | 39.869 s / 27.87 tok/s / 1,024 | 37.599 s / 29.32 tok/s / 1,024 |
| Repeated prose | 16.808 s / 25.51 tok/s / 413 | 13.704 s / 27.86 tok/s / 369 |

Both prose outputs finished; both code outputs reached the cap. Each rank read
83.308 versus 65.714 GiB over the three continuations. The initial prefill was
38.705 versus 38.582 seconds and both ranks read 109.950 GiB. The initial tool
response matched; generated continuations and speculative acceptance differed.
Both processes had zero captured swap on both ranks.

Scheduling is a promising performance option on this sample, but the shorter
prose output contributes to its lower wall time and its changed responses and
routes prevent attribution to kernel speed alone. The limited completed prose
sample and capped code response do not establish general task quality or a
robust deployment win. Fixed depth three remains selected; different scheduling
parameters and broader completed-task workloads remain untested.

### Scheduled depth: completed code-review follow-up

The preceding 1,024-token code responses both reached their cap. A follow-up
raised the limit to 4,096 and presented the same Python cache-helper review
twice after the initial prefix-enabled tool request. Control ran first and
scheduled depth three second, with a fresh process for each. Both used the
same deployed binary and serving settings as the previous comparison. These
are focused observations of one task, not campaign rows or a quality suite.

| Measurement | Fixed depth three | Confidence scheduled |
| --- | --- | --- |
| Initial prefill, 24,317 tokens | 38.616 s | 38.829 s |
| First code response wall / decode | 67.021 s / 27.99 tok/s | 55.945 s / 29.57 tok/s |
| First response output tokens | 1,788 | 1,562 |
| Repeated code response wall / decode | 60.184 s / 30.14 tok/s | 55.180 s / 31.27 tok/s |
| Repeated response output tokens | 1,788 | 1,703 |
| Code-response expert reads, per rank | 107.139 GiB | 94.616 GiB |

Both code responses in each arm ended normally, rather than reaching the
limit; the initial tool call matched. Output content differed across modes, and
the two scheduled responses differed in length. The scheduled first answer
incorrectly said a cached `None` would be reloaded; the fixed first answer
incorrectly said eviction occurs on a hit when the cache is merely full, even
though the supplied code checks `len(cache) > 3`. Thus neither answer provides
a general quality pass or establishes that the observed change is a trivial
quality loss. Initial requests read 109.950 GiB per rank in both arms; both
ranks had zero swap at the captured boundaries. The repeated task was warmish,
not fully resident; the earlier short-recomputation screen supplied the
zero-read anchor.

Scheduled depth improved observed decode rate by about 4–6% on this sample,
but differing token counts and expert routes also affect wall time. Fixed depth
three remains selected until the performance and quality tradeoff is better
established on completed real tasks.

### Longer decode-only admission history

`engine.expert_streaming.decode_history_multiplier` scales the separate
history used by decode demand admission, without changing prefill admission,
periodic planning or the global history. The default is 1. A value of 4 makes
its half-life four times `decay_routes` in decode selections. It has no effect
when `decode_admission_boost` is zero. A planner fixture covers retaining an
older decode resident and replacing it when new demand persists; configuration
round-trip and invalid-value tests cover the setting on both ranks.

Two focused paired screens compared multipliers 1 and 4 on the same binary,
SHA-256 `3e1b77244ff6befe8a5f5613bb35b7af63ff625e2e3ed2cb7e163f3661c57ff5`.
It contained source `dcbe0784` plus the planner prototype and the prior
activation-cache-hint switch at its unchanged default; the final configurable
implementation is a separate rebuild. Both ranks used whole-expert caching,
decode boost 16, eight readers, the existing budget, fixed speculation depth
three and the selected Engram settings. Each arm restarted before a
prefix-enabled four-request tool loop with 1,024-token continuation caps,
then three short uncached recomputations. Control ran first in the first pair
and candidate first in the second. These are diagnostics, not campaign rows.

| Measurement | Control / multiplier 4, control-first | Control / multiplier 4, candidate-first |
| --- | --- | --- |
| Initial prefill | 38.846 / 38.568 s | 38.657 / 39.098 s |
| Three continuation wall times, summed | 75.140 / 73.633 s | 75.290 / 73.311 s |
| Three continuation decode intervals, summed | 69.473 / 68.151 s | 69.553 / 67.903 s |
| Continuation expert reads, per rank | 83.308 / 78.919 GiB | 83.308 / 78.919 GiB |
| Third short recomputation prefill | 1.615 / 1.638 s | 1.619 / 1.617 s |
| Third short recomputation decode | 52.59 / 52.61 tok/s | 52.27 / 52.44 tok/s |

Each initial request read 109.950 GiB per rank. The third short recomputation
read zero expert bytes on both ranks in all arms. Normalized outputs, usage and
speculative acceptance matched within each pair, and captured process swap
was zero on both ranks. The first code continuation reached its output cap;
this is not an uncapped completed-task or multi-request comparison. Longer
history decreased reads on the repeated prose continuation but increased them
on the preceding code continuation. The consistent roughly 2% loop wall-time
improvement and unchanged warm anchor support selecting multiplier 4 in this
specific Spark deployment; other workload changes can prefer a shorter history.
The rebuilt deployment binary has SHA-256
`e163306747831667813d601306ad38a68827a75cea118706751c75cb3307b8d6`.
After deployment, a 2,209-token tool-enabled smoke request produced a
`list_directory` call successfully, with zero captured swap on both ranks. That
smoke is a functional check, not another performance comparison.
