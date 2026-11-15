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
