# Expert-streaming performance review response

Review baseline: `8acd7305`. Target-layer graph baseline: `ba60af73`.
This response concerns the existing BF16-activation MXFP4/NVFP4 paging paths.
It does not change sampling, quantization, cache budgets, routing, or ordered
expert accumulation.

## Changes

- Canonical expert waves activate only their executed gathered rows. The shared
  expert's activation runs once, with its projections, in the final wave.
  Existing clamp and BF16 rounding kernels remain unchanged.
- Large prefills stage bounded groups of canonical experts using the existing
  execution-slot reservation. One view-table upload and one grouped chain serve
  each wave. Empty expert IDs between selected IDs retain empty segments.
  Whole-model prefill chunks are unchanged. This initial correction still
  unpacked prepared NVFP4 weights; the direct-prepared follow-up below removes
  that conversion for W4A16 prefill. Small batches retain direct b12x.
- Telemetry distinguishes direct native bindings from native hits that unpack
  into canonical scratch. Canonical output-byte counts are logical bytes, not
  physical DRAM traffic. Enqueue time is CPU time, not GPU unpack duration.
  Backend completion-wait time includes pending copies/expert execution and
  overlaps existing reconstruction/layout wall timers.
- Speculation logs effective eligibility on the first decode and whenever its
  reasons change. Reasons include sampling temperature, requested logprobs,
  bias, penalties and context capacity. Tool grammar alone is not disqualifying.
  Optional per-pass timing logs now explicitly include proposed token counts.

## Focused validation

A work-count assertion failed on the old whole-batch activation sweeps, then
passed with segment-local activation. The paged-vs-resident fixture covers both
FP4 formats, FP8/FP4 shared experts, both cache policies, changing routes, empty
expert segments, single/multi-expert scratch waves, and token counts
1, 7, 64, 256, 1024 and 4096. Large cases use the prefill MMA recipe.
Output comparisons are bitwise. Activation work no longer scales with the
product of wave count and full batch rows.

The prepared b12x fixture covers native hits, canonical unpack accounting,
actual cold reads, grouped misses and GPU-demand graph replay. The host cache
suite covers direct-hit accounting and existing residency invariants.

## Interpretation and remaining boundaries

The legacy `native_hits` counter remains compatible as an acquisition count;
it must not be interpreted as copy-free execution. Use the direct-hit and
canonical-unpack counters alongside native preparation/relocation counters.

Canonical groups still wait for their required reconstructions before execution.
This change introduces no readiness scheduler, new transfer stream or prefetch
policy. Reassess exposed stalls after removing redundant prefill work.

The EXL3 comparison uses different precision, execution kernels and workloads.
The reference's Engram tables are disk-backed even though transformer weights
are resident. Neither its rates nor logical activation-byte counts establish an
SSD throughput limit for this deployment. Exact drive-bandwidth characterization
is outside this pass, consistent with the user's instruction that exact
bandwidth numbers are unnecessary.

## Serving validation

The two-Spark NVFP4 deployment ran source `bff4edaf`, binary SHA-256
`0195a9cd19aa37e3dc54d779df19b82a4807ce9262fdc3487b0cb0ff6e341407`.
It retained whole-expert caching, the 100 GiB per-rank reservation including
draft weights, 24 canonical execution slots, four I/O workers, 16 staging
buffers, direct I/O, fixed speculation depth three, and target-layer graphs.
The serving endpoint and authentication configuration were unchanged.
Both live Engram mappings and expert-image directories resolved to the local
NVMe ext4 filesystem on each rank.

The existing 24,128-token greedy tool-enabled diagnostic was replayed once from
a freshly started server and twice warm. This is a focused regression comparison,
not a new standard serving-campaign row set or a replay of a historical user
conversation. Against the recorded fresh-server `ba60af73` diagnostic:

- Cold engine prefill fell from 93.704 s to 66.329 s (29.2% less time);
  prompt processing rose from 257.5 to 363.8 tokens/s.
- Expert-image reads stayed approximately 118.8 GiB per rank across the complete
  first request. These are logical expert-file bytes, not physical NVMe counters
  and not decode-only traffic.
- First-request wall time fell from 103.657 s to 76.541 s. Cold decode was
  14.5 tokens/s versus 14.9 previously; this patch targets prefill.
- The final warm repeat stayed at 45.7 decode tokens/s, with zero expert reads,
  reconstruction, native preparation and canonical unpacking on both ranks.
- All three outputs matched the earlier transcript, including reasoning and tool
  arguments, excluding generated call IDs. Each generated 149 tokens in
  45 verification passes, accepting 2.31 predictions per pass.
- Both ranks reported zero process swap. The allocated target expert subsystem
  remained 103,764,177,664 bytes within its 103,764,459,520-byte budget.

The timing improvement measures the combined activation and wave-grouping
changes; it does not isolate their individual contributions. It still does not
establish that the remaining latency is entirely storage-bound.

## Normal agent-path check

A Codex subagent using the `spark-primary` model override exercised the normal
OpenCodex → Combo → DGPP path. Its transcript records exactly one
`ls -la` execution in `/home/jon/dgpp`, exit status zero, followed by a
successful final answer. The task took 75.372 s:

- Initial request: 24,138 prompt tokens, no cached prefix, 54.799 s engine
  prefill, 90 generated tokens, 62.4 s server request duration.
- Tool-result continuation: 25,162 prompt tokens, 24,064 cached tokens,
  9.377 s engine prefill, 30 generated tokens, 11.5 s server request duration.
- Combined server durations were approximately 73.9 s, leaving approximately
  1.5 s for tool execution, proxy/client handling and other time outside the
  recorded server requests. This is not an isolated proxy microbenchmark.
- Both requests logged speculation eligible, temperature zero, depth three
  and active tool grammar. A separate tool-enabled temperature-0.7 control
  logged `reason=temperature` and generated four tokens with zero draft passes,
  confirming fallback without changing the requested sampling behavior.
- Across the complete two-request sequence, ranks read 78.919/78.881 GiB
  of expert-image bytes and recorded 49,709/49,722 native-to-canonical unpacks.
  Logical canonical output volume was about 494.8/494.9 GB. These unpack counts
  are now distinct from 23,570/23,571 direct native executions. All these
  counters include both prefill and decode; none is a decode-only bandwidth.
- Both ranks remained at zero process swap.

This actual agent sequence is different from the earlier direct diagnostic;
their throughputs are not an apples-to-apples proxy comparison. The server's
own durations do establish that most of this agent's wall time was inside DGPP.
The new telemetry demonstrates remaining conversion work and substantial
expert reads despite approximately 92% full-resident acquisitions. The changes
do not make the deployment fully resident or eliminate its capacity penalty.

## Review disposition

Findings 1–3 now have concrete implementation fixes: segment-local activation,
bounded mixed-layout prefill, direct prepared bindings, one-time preparation of
complete resident experts, removal of unused scratch clears, and explicit
conversion/clear/wait accounting. Finding 5's eligibility logging and tool-aware
speculation checks remain validated. Findings 6–8 are addressed by separating
warm results and quantization differences, checking local storage placement,
and matching the translated request bodies and acceptance below.

Finding 4 now has a retirement-fence fix and an independently completed miss
path, tested against grouped execution. CUDA scratch reuse is stream ordered.
The optional small-batch path reconstructs multiple experts within the existing
reservation, consumes ready file tickets without waiting on the oldest ticket,
and submits ready expert groups before other reads finish. Deferred admission
uses reconstructed scratch. The serving comparison did not establish a decode
benefit, so grouped acquisition remains the default. Large prefill retains
bounded mixed-layout groups. This is measured
follow-through on the scheduling finding, not a claim that all miss stalls or
host completion fences have been eliminated.

Finding 9's bounded numerical, serving, proxy and GPU-attribution work is
recorded below. Exact drive-bandwidth characterization remains outside scope
at the user's request. No claim is made that all remaining latency is storage
transfer time.

## Direct prepared prefill implementation

The W4A16 prefill tile can now consume the prepared NVFP4 code and scale
layout directly. It gathers only the current weight tile into shared memory,
then retains the canonical BF16 decode, MMA order and output rounding. Packing,
unpacking and prefill share the same layout-index functions.

The host reconstructs misses before pinning native bindings, then executes both
layouts in the same bounded wave. It revisits prepared selections evicted by
those admissions before taking native leases. Source descriptors may remain
metadata-only. W4A4 and GEMV keep their canonical fallback; small b12x batches
and graph replay are unchanged. `DGPP_MOE_PREPARED_PREFILL=0` retains the previous
large-prefill binding for comparisons.

The checkpoint-geometry fixture (H=5120, I=1152, top-6) passes bitwise comparisons
against canonical MMA at 65, 129 and 256 rows, including metadata-only source
descriptors, prepared hits, residency pressure and subsequent verification.
Prepared hits add no canonical unpack operations and leave no live native
leases. The 24 host cache tests and both-format paged-wave regression pass.
An initial implementation split waves at layout transitions. Although it removed
unpacking, the matched tool continuation regressed from 7.690 s to 8.757 s
prefill. That scheduling choice was removed. Mixed-layout waves measured
7.425 s on the same continuation (963 computed tokens, 24,064 cached), with
zero canonical unpacks versus 7,499 per rank previously. This is one focused
comparison, not an established percentage speedup. Initial uncached prefill was
64.304 s versus 65.941 s; neither initial prefill performed canonical unpacking.
The initial and continuation generated identical outputs to baseline, excluding
call IDs, with 50 and 29 tokens respectively. Both ranks remained at zero swap.

The expanded fixture also passes with four execution slots, exercising bounded
prepared and reconstructed groups. The matched proxy comparison and GPU attribution follow below.

## Matched translation/proxy comparison

The runtime implementation corresponding to `9e05096b` used binary SHA-256
`72e9c2e1df06e5dc52af5259c361720d07776bd5cea318f626d4119558d4760f`.
This focused diagnostic used the same authored long system prompt and directory
listing tool exchange on both paths; it is not a captured historical user chat.
Both sequences began with a freshly started server at the existing production
configuration.

Equivalent supplied messages were initially insufficient to match rendering:
OpenCodex appended a 956-character tool-contract instruction during Responses
translation, adding 197 initial prompt tokens and 198 continuation tokens. Only
the two test request bodies after translation were captured, excluding HTTP
headers, then replayed unchanged against DGPP's Chat endpoint.

- Normal Responses → OpenCodex/Combo → DGPP sequence: 69.810 s initial request,
  10.019 s continuation, 79.828 s combined.
- Exact translated bodies sent directly to DGPP: 69.262 s initial request,
  10.124 s continuation, 79.386 s combined.
- Both paths rendered 24,317 initial prompt tokens with no cached tokens, then
  25,225 continuation tokens with 24,192 cached and 1,033 computed tokens.
  Visible text and tool arguments matched, with 50 and 29 output tokens.
- Speculation also matched: 14 initial verification passes / 38 accepted
  predictions, then 9 passes / 21 accepted predictions. Tools remained enabled.
- Engine prefill plus decode-step timing accounts for approximately 79.702 s
  of the proxy path's 79.828 s. The residual includes HTTP/client/proxy handling
  and is not an isolated measurement of any one component. Engine variation
  prevents attributing the 0.442 s between sequences solely to proxy overhead.
- Both ranks remained at zero swap, with zero canonical unpacks. Initial expert
  reads matched exactly at about 110.118 GiB per rank. Continuation reads were
  approximately 12.95–12.99 GiB per rank, including prefill and decode.

The proxy is not the dominant latency source in this matched tool sequence.
Its added instruction does change tokenization and routing, so comparisons
must use the rendered request rather than assume identical supplied messages
produce identical model inputs.

## GPU cost attribution after direct prepared binding

A short two-rank Nsight capture used the identical translated tool continuation.
The profiler could not reliably fit beside the million-token reservation and
4 GiB startup headroom, so this diagnostic used a 262,144-token KV reservation;
expert-cache capacity, execution slots, weights, speculation and the 25K-token
request were unchanged. Normal serving retains the million-token reservation.
This capture is cost attribution, not an uninstrumented throughput comparison.

The prefill-to-first-target-graph interval was about 7.45 s on each rank:

- No GPU activity occupied 3.56 / 3.91 s on ranks 0 / 1.
- Routed prefill kernels occupied about 1.04 s per rank.
- Expert layout/scatter kernels occupied about 1.03 s per rank, mostly chunk
  scatter rather than canonical unpacking (the latter remained zero).
- Copies/clears occupied about 0.81 s of exclusive timeline per rank. Across
  the full turn, rank 0 executed 9,409 memsets, clearing 182.9 GB in 0.690 s.
- Dense projections accounted for about 0.157 s per rank in this interval.
  They are not its dominant remaining cost.

The following roughly 2.14 s target-graph interval contained 0.72–0.73 s with
only the expert-demand gate active and no useful overlapping GPU work. The
longer total gate duration overlaps reconstruction and expert execution and
must not be counted as entirely idle. Both ranks issued roughly 188K CUDA
launches and 31K stream synchronizations across the full turn. CUDA API sums,
kernel sums and GPU-empty intervals overlap and must not be added together.
The first-target-graph boundary includes a small amount of initial draft work;
it is not an exact engine-timer boundary.

These observations support removing repeated resident reconstruction and
unnecessary scratch initialization/retirement fences. They do not establish
that all remaining latency is SSD transfer time, nor that extra I/O threads
alone would remove the GPU-empty intervals.

## Resident preparation, scratch reuse and ready-miss comparison

Runtime source: `1f2f7032`; binary SHA-256
`fdc35e553441c0140b5dd15b243d02fd107b955530f4c1396544ae54069b8a80`.
All uninstrumented runs below use the production million-token KV reservation,
100 GiB per-rank expert subsystem budget, 24 execution slots, four I/O workers,
16 staging buffers, fixed depth three and target-layer graphs.

Complete resident experts are prepared before a wave owns native or scratch
leases, so later prefills can execute them directly rather than repeatedly
scatter their chunks. Preparation uses existing reserved scratch and does not
increase cache capacity. `DGPP_MOE_PREFILL_PREPARE_RESIDENT=0` disables this
follow-up independently of direct prepared consumption.

CUDA reconstruction no longer clears its two scratch reservations before
assembly/packing: every consumed payload, scale and global is overwritten;
allocation padding is not read by consumers. Clear-byte telemetry reports zero
for this backend. Canonical scratch retirement now follows the same stream-order
contract as native retirement. Staging-buffer completion fences remain, and
callers must still quiesce readers before switching execution streams.

The host cache suite (24 tests), both-format paged-wave fixture and b12x paging
fixture pass. The latter now also checks repeated large prefill without a
preceding decode: after preparation, a feasible hot set needs no file reads,
reconstruction or repacking, stays within budget and remains bitwise canonical.

The same translated initial request and fixed tool-result continuation were
replayed from fresh servers. Against direct-prepared mixed waves (`9e05096b`):

- Initial prefill: 65.119 → 62.496 s; initial request wall: 69.262 → 65.427 s.
- Continuation prefill: 7.644 → 6.402 s; continuation wall: 10.124 → 8.295 s.
- Decode intervals: 4.113 → 2.892 s initially and 2.433 → 1.837 s on continuation.
- Combined wall time: 79.386 → 73.722 s. These are focused paired observations,
  not a general throughput claim or an isolated contribution for each change.
- Outputs, tool arguments, prompt/cache counts and acceptance matched: 50/29
  output tokens, 14/9 passes and 38/21 accepted predictions. Both ranks stayed
  at zero swap. Canonical-unpack and reconstruction-clear counters stayed zero.
- Expert-image reads remained approximately 110.12 GiB per rank initially and
  12.97–12.98 GiB on continuation. Removing implementation work did not remove
  the deployment's missing expert bytes.

`DGPP_MOE_MISS_READY_LIMIT=1` launches each reconstructed small-batch miss
immediately, with next-expert prefetch and stream-ordered scratch reuse. It
changes no memory/I/O reservation. With that diagnostic cap, initial decode was
2.860 s versus 2.892 s grouped; continuation decode was 2.010 s versus 1.837 s.
The complete sequence was 74.409 s versus 73.722 s. Output and acceptance still
matched. There is no demonstrated advantage here, so production retains grouped
misses and the diagnostic override is removed. This comparison does not test a
fully asynchronous multi-expert reconstruction pipeline or prove it cannot help.

That immediate-launch experiment alone did not separate demand submission,
reconstruction completion and execution readiness. The following batch path
implements that separation and measures its tradeoff against grouped launches.


## Independent miss completion

Runtime source: `2e5996ff`; binary SHA-256
`818489e7ebe61870501847455b06cf25f3e71193943f4c9f6c4e739766cdbd3b`.
`DGPP_MOE_ASYNC_MISSES=1` selects the optional small-batch NVFP4 path.

The cache snapshots resident source chunks into reserved execution slots before
admission can evict them. A bounded read window consumes any completed ticket;
a condition variable wakes the engine when any outstanding read completes.
Completed experts execute on the same stream as assembly while later reads
remain outstanding. Currently ready contiguous groups share a launch, but
staggered completion can still require one launch per expert. Scratch leases
remain owned through the wave. Staging stays alive until its completion fence.
No extra file workers, payload allocations or transfer stream are introduced.

Admission follows execution submission and copies chunks from canonical
scratch, without rereading storage. Fractional caching reserves one execution
slot for unpacking a partially evicted native donor; a one-slot configuration
uses the existing sequential path. This prevents donor conversion from
clobbering a reconstructed expert. These extra admission copies and early
small launches are costs, so independent readiness is not automatically faster.
Reconstruction wall accounting for this optional batch includes its callbacks
and completion fences; it is not an isolated GPU reconstruction duration.

All 26 host-cache and seven read-queue tests pass, including exact bytes for
both weight formats/policies, staging bounds, early expert execution before
all reads complete, skipping a blocked first ticket, and callback failure
poisoning. The CUDA b12x paging fixture passes with independent completion,
including bitwise output, native/miss transitions, pressure and GPU-demand graph
service. Its fractional scratch expectation reserves the donor slot explicitly.

The same translated tool-request pair was replayed from fresh servers with
independent completion and then the grouped control, using this same binary.
Both used the production million-token context, 100 GiB expert budget per rank,
24 execution slots, four read workers, 16 staging buffers, fixed depth three and
target graphs. This is a focused diagnostic, not a standard serving campaign.

| Path | Request | Prefill (s) | Decode (s) | Decode (tok/s) | Wall (s) |
|---|---|---:|---:|---:|---:|
| Independent completion | Initial | 62.722 | 3.040 | 16.12 | 65.810 |
| Grouped control | Initial | 63.262 | 3.002 | 16.32 | 66.299 |
| Independent completion | Continuation | 6.476 | 1.962 | 14.27 | 8.477 |
| Grouped control | Continuation | 6.680 | 1.833 | 15.27 | 8.552 |

The optimization targets small-batch misses, not these large prefills. Combined
decode time was 5.002 s versus 4.835 s grouped; neither request demonstrated a
decode gain. Combined wall time was 74.288 s versus 74.850 s, with the difference
coming from prefill. That is insufficient evidence to replace grouped execution,
and the optional path remains disabled in production. No further width sweep
or broad campaign was run.

Outputs and tool arguments matched exactly after removing generated call IDs.
Both arms generated 50/29 tokens, used 14/9 verification passes and accepted
38/21 predictions. Prompt/cache counts matched, and both ranks had zero swap
throughout. The normal endpoint, keys, million-token context and grouped-miss
configuration are active after the comparison. The tested optional path is
retained for review, not presented as a performance improvement. Eliminating
remaining miss fences or optimizing large-prefill readiness would be further
implementation work, not a result established by this comparison.

## Updated long-prefill attribution and read identity

After review of `60d10b92`, one new two-rank capture replayed the exact translated
initial request and tool-result continuation. Runtime source: `a3412b27`; binary
SHA-256 `d1ab72063de4f3e50cb510ad993be1ba0319ebc26c1dd21d05342ba23fd626f1`.
This includes all previous execution fixes and grouped misses. The only new
runtime facility is optional read tracing. Nsight again used 262,144 KV tokens
for profiler headroom; the request bodies, expert budgets, execution slots,
worker/staging counts, sampling and speculation remained unchanged. These are
instrumented attribution results, not production throughput measurements.

The store can write successful-read completion records when
`DGPP_EXPERT_READ_TRACE` is a filename prefix. Output is
`<prefix>.r<rank>.p<pid>.bin`; it contains timestamps, expert/chunk identity,
record bytes and file offsets, never weights or prompt text. Tracing is off by
default. `scripts/expert_read_trace.py` aggregates these records per expert and
accepts `--start-ns` / `--end-ns` Unix-nanosecond windows. It distinguishes:

- First-fetch bytes: chunks not previously read anywhere in that trace.
- Reread bytes: chunks already fetched earlier in the trace.
- Repeats within the selected window: chunks fetched more than once inside it.

Classification is by chunk before aggregation, so partially resident experts
are not misclassified as wholly new or wholly repeated. Counts are successful
logical record bytes, including alignment; they are not physical SSD traffic.
Read-duration sums overlap between workers and must not be treated as wall time.

The six store tests pass, including trace identity, byte counts and duplicate
reads. Analyzer validation distinguishes earlier-window rereads from repeats
inside the current window. Full-model outputs, tool arguments, prompt/cache
counts and speculative acceptance match the uninstrumented grouped control:
50/29 output tokens, 14/9 target passes and 38/21 accepted predictions. Both
ranks remained at zero swap. Read totals agree with existing rank telemetry.

### Read volume

The phase boundary is the first target-graph demand kernel. Thus the prefill
window includes a small amount of initial draft/graph setup; it is not an exact
engine timer boundary. Both ranks produced identical prefill read counts:

| Phase | Expert reads (GiB/rank) | First fetch (GiB/rank) | Previously fetched (GiB/rank) | Repeated within phase (GiB/rank) |
|---|---:|---:|---:|---:|
| Initial prefill through first target graph | 106.317 | 106.317 | 0 | 0 |
| Continuation prefill through first target graph | 10.646 | 3.166 | 7.480 | 0 |

The initial window fetched chunks belonging to 11,385 target experts. The
continuation fetched 1,140 target experts; 801 had previously fetched chunks.
Neither prefill window reread a chunk internally. Across complete requests,
including decode, reads were 110.118 and 12.971 GiB per rank.

This rules out repeated chunk reloads *within these prefill phases* as the
explanation for their latency. It does not establish optimal eviction between
requests: 7.480 GiB of continuation prefill had been fetched previously.
Replacement changes alone cannot remove the initial first-touch reads while
preserving the same required payload and initially cold cache.

### Updated GPU/file-read overlap

| Prefill window | Rank | Window (s) | No GPU activity (s) | No GPU activity while either rank reads (s) | No GPU activity and neither rank reads (s) |
|---|---:|---:|---:|---:|---:|
| Initial | 0 | 63.874 | 40.643 | 30.343 | 10.301 |
| Initial | 1 | 63.872 | 41.116 | 30.886 | 10.230 |
| Continuation | 0 | 7.090 | 3.917 | 3.491 | 0.426 |
| Continuation | 1 | 7.095 | 4.622 | 4.077 | 0.545 |

An active read means a successful store read's start-to-completion interval,
not SSD utilization. Accounting for both ranks matters: checking only local
reads would attribute approximately 14 seconds of initial GPU inactivity to
non-I/O work, versus approximately 10.3 seconds with the peer included.
These gaps can include host planning, synchronization, communication progress
and instrumentation overhead; they are not automatically recoverable time.
Overlap does not establish causation or prove ready resident work existed
throughout each gap. Untraced Engram page faults are not expert-image reads.

Initial routed-prefill kernel sums were approximately 7.79 / 7.72 s on ranks
0 / 1; expert layout/scatter sums were 3.24 / 3.22 s. Continuation routed-prefill
sums were 1.39 / 1.38 s. Kernel sums can overlap other activity and must not be
added to the wall timeline categories. Engine prefill timers were 63.810 and
7.053 s; instrumented decode timers were 3.000 and 1.890 s.

Process-level counters expose another contributor which the expert trace does
not include: the initial request incurred 596,072 / 598,290 major page faults
and 148.748 / 148.867 GiB of process reads on ranks 0 / 1. That is roughly
38.6–38.7 GiB beyond expert-image reads. These counters include other mapped
files and read amplification; they do not individually identify Engram traffic,
but the mmap-backed Engram gather is a concrete source to investigate. Thus the
10.3-second gap must not be labeled pure host overhead or non-storage time.

Inspection found a specific Engram prefetch defect: `gather()` aligned
`MADV_WILLNEED` addresses to 4 KiB, whereas both deployment hosts use 64 KiB
pages. A direct syscall check on Spark confirmed that a 4 KiB-offset address
returns `EINVAL`, while the host-page-aligned address succeeds. The constructor
already uses `MADV_RANDOM`; simply adding that advice again is not a remedy.

Large-prefill resident-first scheduling remains a supported hypothesis: submit
demand reads while executing prepared resident work, and avoid limiting resident
groups by reconstruction-slot capacity. Its lease/admission ordering still needs
implementation and a matched comparison. The concrete Engram alignment defect
takes priority over speculative overlap changes. The capture does not establish
that reordering will win or that the remaining time is an irreducible SSD limit. No additional tuning of the optional decode readiness
path was performed. Production was restored once after capture, retaining its
million-token context, endpoint, credentials and grouped misses; diagnostic
tracing/profiling is not enabled there.

### Engram page-size correction and hint policy

The Engram gather now derives alignment from `_SC_PAGESIZE`, and each hint
covers the actual payload/scale row extent. Invalid ranges are reported instead
of silently accepting `EINVAL`. Both FP8 and MXFP4 Engram GPU fixtures pass
bitwise on the 64 KiB-page Spark, with hints enabled and disabled.

The corrected-hint diagnostic (`fcffa57f`, production 1M context, profiling and
read tracing off) did not demonstrate a speedup: initial prefill was 65.054 s,
continuation prefill 6.936 s, and total pair wall time 77.044 s. The earlier
grouped control recorded 63.262 / 6.680 s prefill and 74.850 s pair wall.
Major faults dropped from 581,299 / 583,562 to 6,411 / 47,055 initially, but
process reads remained 148.691 / 151.569 GiB. The hints moved work ahead of the
demand faults without removing read volume; fewer major faults did not imply
faster serving. These observations do not isolate all OS-cache/run variation.

Explicit hints are now optional with `DGPP_ENGRAM_PREFETCH=1`; the alternative
uses the existing parallel gather threads to fault demanded pages directly.
This removes the per-row hint syscalls rather than preserving misaligned hints
as an accidental optimization. Neither path changes gathered bytes or routing.

The no-hint diagnostic used runtime source `5a516a9c`, binary SHA-256
`7bf02101d8c628cb8ba3e702da7b816b4db0e5654b3e2067eee777076db95a32`.
With production 1M context and no profiling/tracing:

| Request | Prefill (s) | Decode (s) | Decode (tok/s) | Wall (s) |
|---|---:|---:|---:|---:|
| Initial | 63.153 | 2.885 | 16.99 | 66.077 |
| Continuation | 6.467 | 1.863 | 15.03 | 8.382 |

The pair took 74.459 s, versus 77.044 s with corrected explicit hints and
74.850 s for the earlier grouped control. This supports leaving hints disabled;
it does not establish a material prefill gain over the original control.
All three arms matched outputs, tool arguments, usage and acceptance, with zero
swap on both ranks. The no-hint binary is deployed with the existing endpoint,
keys, 1M context and grouped expert misses. Aligned hints remain opt-in for
workloads where a separate comparison establishes benefit.

This pass adds current attribution and fixes a real page-size assumption; it
does not claim to solve the remaining prefill latency. Future work should include
Engram's mapped-file read amplification alongside resident-first prefill
scheduling. A chunk identity trace alone cannot declare either subsystem the
irreducible hardware bottleneck.

### Compact Engram reads and large-prefill scheduling

`DGPP_ENGRAM_DIRECT=1` replaces demand-faulted Engram gathers with aligned
4 KiB direct reads and a bounded, four-way compact row cache. Payload and
scale rows are cached together; large gathers sort requests within worker
shards while preserving destination order. The two MXFP4 tables reserve about
72.4 MiB per rank, plus budgeted per-call indices. This reservation is charged
to staging memory; the 100 GiB expert budget is unchanged. The checkpoint's
Engram tables are MXFP4 even when its routed experts are NVFP4.

The mode is explicit because it requires a filesystem supporting direct I/O.
The mmap gather remains available. Host tests cover block boundaries, short
final blocks, warm hits, concurrent gathers, invalid rows and failed reads;
FP8 and MXFP4 GPU gather fixtures match the reference bitwise on Spark.

A focused replay of the same initial tool request and cached continuation,
with production 1M context, fixed speculative depth three, unchanged sampling
and no tracing, produced:

| Configuration | Initial prefill (s) | Continuation prefill (s) | Combined request wall (s) |
|---|---:|---:|---:|
| mmap, no explicit hints | 63.153 | 6.467 | 74.459 |
| Direct compact cache | 54.149 | 6.438 | 65.696 |
| Direct cache + resident-first experts | 54.949 | 6.123 | 66.070 |
| Direct cache + paged prefill overlap | 54.777 | 6.610 | 66.483 |

Direct caching reduced initial prefill by 14.3% and pair wall time by 11.8%
in this comparison. Initial process reads fell from about 149 GiB to about
113 GiB per rank, while expert reads stayed at 110.118 GiB. Both ranks had
zero major faults and zero swap. Outputs, tool arguments, token usage and
speculative acceptance matched: 38 accepted predictions across 14 initial
passes, and 21 across nine continuation passes. These are focused request
pairs, not a broad throughput or concurrency result. Process-read counters
include all process I/O, not exclusively Engram reads.

`DGPP_MOE_RESIDENT_FIRST=1` groups prepared resident experts ahead of misses
without capping the resident group by reconstruction-slot capacity. It preserves
route-slot accumulation order and prefetches the first miss before resident
execution. Prepared/mixed prefill and cache-pressure fixtures pass bitwise.
The continuation improved, but the complete pair did not; this option remains
disabled in production. Larger resident groups alone do not guarantee faster
execution.

Both TP ranks process the same tokens and layers, splitting every expert's
intermediate width (1,152 dimensions per rank), rather than assigning whole
experts or layers to different ranks. Shared experts and attention projections
are also partitioned, and reductions join partial outputs. Engram hash heads
are partitioned. A slower read or kernel on either rank can therefore delay
both; adding rank times would double-count overlapping work.

`DGPP_EXPERT_PREFILL_OVERLAP=1` allows the existing two-row-block attention,
Engram and boundary-reduction schedule to run with paging. The paged expert
chain retains its ordinary ordered accumulation; both row blocks become ready
together, rather than claiming independent expert completion. NVFP4 and MXFP4
model fixtures, for both cache policies, match ordinary paged prefill and three
subsequent decode logits bitwise. The two-rank request pair also matched outputs,
usage and acceptance, with zero major faults and swap. It did not improve total
time and remains disabled. This does not rule out a more granular expert
completion schedule; the candidate deliberately retains the existing expert
chain and its synchronization.

Runtime sources were `5a516a9c` for no hints, `ba5f70db` for direct caching,
`0f661f5f` for resident-first, and `0f661f5f` plus the paged overlap changes for
the overlap candidate. Final deployment enables only `DGPP_ENGRAM_DIRECT=1`
from this pass, retaining the endpoint, keys, 1M context, expert cache budget,
read workers and fixed speculative depth. Engram amplification was a removable
cost; the remaining expert I/O, reconstruction/layout preparation and rank
coordination are still optimization opportunities, not established hardware
limits.


### Subsequent warm-path optimization

The later [warm prefill investigation](warm-prefill-optimization.md) revisits
resident-first grouping under zero-read conditions and adds contiguous copies
of prepared NVFP4 weights. Warm 2,209-token prefill falls from about 2.08 to
1.42 seconds with matching outputs and acceptance. Resident-first is now enabled
in the Spark deployment based on that warm result; the earlier cold-pair result
above remains valid. The attempted compact tile grid did not improve serving
and was removed. Warm, cold and prefix-cache performance remain separate claims.

### Demand scheduling and residual review scope

The [bounded demand read-ahead follow-up](warm-prefill-optimization.md#bounded-demand-read-ahead-follow-up)
addresses pressure-blocked reads for already-selected experts, attention metadata
staging and redundant clears. A larger staging pool within the same cache budget
reduces cold prefill by about 20% and repeated miss-heavy prefill by about 34–35%;
steady warm performance is effectively unchanged. The linked section records
validation and explicitly lists remaining stream ownership, independent readiness,
chunk-direct execution and activation-precision work. Those architectural findings
are still open, rather than resolved by the demand queue change.

### Cache copy/compute ownership follow-up

The [copy/compute overlap implementation](warm-prefill-optimization.md#copycompute-overlap-and-cache-completion-fences)
now resolves the previously open cache transfer-stream and acquisition/admission
drain findings with per-buffer staging completion, scratch/native reader fences
and rotating prefill scratch. Cold prefill improves about 4.8% in the focused
comparison; steady warm performance is unchanged. Independent ready-expert
launches, chunk-direct kernels and activation-precision qualification remain open.

### Follow-up review of `c147652`: current disposition

The external review "DGPP serving performance: next experiments" was checked
against `bc330320`. Its source observations are hypotheses about performance,
not measurements from the live Sparks or instructions to implement every idea.

- **Global estimator clock: verified concern, partially addressed.** The clock
  still advances for every selected token/expert across layers. The retained
  65536 half-life reduced warmed prefill time by 27.8% in the focused tool
  sequence; see [cache history results](chunk-direct-execution.md#cache-history-follow-up-2026-11-15).
  The comparison held horizon at 8192, so periodic normalization changed too.
  Layer-aware ageing, phase-aware histories and fetch-event statistics are not
  implemented. Larger decay helped recomputed prefill but hurt prefix-enabled
  continuations; the proportional-horizon control also failed to improve serving.
  Decay 65536 with horizon 8192 remains selected. A subsequent opt-in
  once-per-expert prefill history control also regressed continuations and
  short-prompt adaptation; it is disabled. This does not resolve phase-aware
  weighting or layer-aware ageing. A separate fixed initial load-cost control
  reduced continuation expert traffic by about 15% and elapsed time by 0.9–4.6%
  in two orderings; it is now selected for this deployment. The general default
  still learns costs, and short-prompt warmup was slightly slower with fixed costs.
- **Blocking-reader concurrency: verified and tested.** Each worker
  calls the blocking store reader before taking another job. Queue capacity
  256 is not 256 concurrent expert read calls, and the queue's active count
  includes staging acquisition. Eight workers reduced the prefix-enabled
  continuation sequence's elapsed time by 14.9% versus four; sixteen improved
  cold prefill slightly further but was slower on the continuations than eight.
  Eight is now deployed with the same staging pool and cache capacity. No
  asynchronous I/O backend has been implemented or separately justified.
- **Adjacent-range coalescing: verified gap, deferred.** Current coalescing
  deduplicates one record; it does not combine neighboring records. Preserve
  alignment, per-chunk views and GPU reader lifetimes if physical coalescing
  proves useful. No image rebuild or chunk-width change has been made.
- **Full GPU binding publication: verified and timed.** A
  prepared revision change still rebuilds every target layer's table, copies
  all entries and synchronizes the service stream. An unchanged revision exits
  immediately. Opt-in timing measured publication at about 3–4% of target
  walk time per rank in the tool continuations, mostly CPU rebuilding; a fully
  warm anchor published nothing. A subsequent bulk-lookup change reduced CPU
  rebuilding by 57–59% without dirty tracking and is enabled by default. It
  reduces ownership checks from 15,360 to 40 per publication while rebuilding
  every entry. Dirty publication remains deferred; any future dirty set must
  include removals and relocations. See the attribution and bulk-lookup results
  in the chunk-direct doc.
- **Ready prefill groups: partially investigated, no serving win.** The newer
  opt-in ready-prefix experiment was correct but did not improve the matched
  serving sequence, and is disabled. It preserves order and does not run
  arbitrary later-ready experts. Full readiness-based grouping remains open;
  the reader-count comparison subsequently selected eight workers before
  attempting another scheduler rewrite.
- **Normal tool continuations and demand replay: remaining coverage gaps.**
  The original comparisons disabled prefix reuse and used short forced tool
  calls. A new direct-DGPP continuation comparison enables prefix reuse and
  includes capped prose/code answers; it rejected the larger-history candidate.
  Exact Combo-client traffic and uncapped task completion remain untested.
  Successful-read traces cannot provide a complete alternative
  policy replay: hits, grouped demand events and starting residency are also
  required. No future-aware traffic bound has been measured.

Existing resident-first execution, prepared layouts, Engram direct caching and
transfer-stream ownership should not be reimplemented. W4A4 remains disabled
because its earlier serving screen was slower and changed acceptance. The
context and precision settings remain unchanged. These conclusions do not
claim that cache policy or storage alone explains all remaining latency.

## Deployment reproducibility and binding publication (2026-11-16)

The NVFP4 two-Spark template now records the engine settings and execution
switches of the `f87df92b` deployment. Launcher-owned `environment` defaults
apply to every rank; site and exported overrides remain available. The
[benchmark record](benchmarks.md#two-spark-nvfp4-streaming-measurement-2026-11-16)
publishes the completed serving/prefill campaign, including its binary hash,
zero-swap and operation-stream checks. Quality, long-context and speculative
versus plain equivalence were not evaluated by that performance campaign.

Binding publication now compares native base addresses, rebuilding only changed
expert entries and transferring only changed layers. Adjacent changed layers
share a transfer. First publication initializes the whole device table, and
null addresses clear both projections on eviction. All addresses are still
scanned when the cache revision changes; this is not a per-expert mutation
journal. The existing service-stream fence remains because pointer equality
alone does not establish completion of earlier payload mutations. A standalone
host regression covers initialization, unchanged bindings, relocation,
eviction and multiple changed experts. The focused comparison below validates this implementation separately; it is
not included in the `f87df92b` throughput rows.

Ready-chunk batched scatter is now implemented and evaluated below. A batch
contains only already-completed reads; it does not wait for a complete expert
or wave. Per-buffer fences retain staging until its batched kernel finishes.

Existing cold-prefill device-counter captures are documented in
[the prefetch opportunity analysis](chunk-direct-execution.md#prefetch-opportunity-live-io-gaps-and-absent-expert-forecasts).
The 56–58% device-accounted busy fractions are observations from those requests,
not a measurement of peak drive bandwidth. Cache capacity alone does not
predict bytes read, and a high GPU busy fraction does not establish that
warm speculative verification has reached a performance floor. The MXFP4
format comparison remains unrun on matched deployments.

### Incremental binding publication: focused serving comparison

A paired diagnostic compares the deployed `f87df92b` binary against the
`21a30943` implementation on the same two Sparks. Both fresh processes use the
production engine settings, fixed depth three and publication profiling. Each
runs one 24,317-token tool request, prose/code/repeated-prose continuations, then
three uncached 2,209-token tool probes. These are focused diagnostic requests,
not a replacement for the standard campaign rows in `benchmarks.md`.

| Cumulative publication work | Baseline rank 0 / 1 | Incremental rank 0 / 1 |
| --- | ---: | ---: |
| Host binding build | 730.6 / 772.8 ms | 475.9 / 480.2 ms |
| Transfer and wait | 122.1 / 130.1 ms | 70.1 / 69.7 ms |
| Binding upload bytes | 4.370 / 4.626 GB | 0.192 / 0.192 GB |

The counters include 737 target passes and 6,606 serviced layer misses per
rank. Publication-call counts can vary with host service timing; the table
reports measured cumulative work rather than assuming identical call counts.
Build and transfer/wait clocks are nested in the full walk clock and must not
be added to it. The full target walk was 65.43 / 65.70 seconds in the baseline
and 65.96 / 65.63 seconds in the candidate, so this does not establish a serving
throughput gain despite the lower publication work.

Cold tool prefill was 39.135 versus 39.107 seconds. The three continuation wall
times summed to 72.491 versus 72.474 seconds. The final zero-expert-read probe
was 1.633 versus 1.620 seconds of prefill and 52.92 versus 51.62 decode tokens/s.
The short decode spans do not establish a regression or improvement. Each
request read identical expert bytes on both ranks, used identical token counts
and had identical speculative acceptance. Normalized response choices matched
in all seven requests; normalization removes generated tool-call IDs only.
Captured process swap remained zero. Production was restored after the pair.

### Ready-chunk scatter batching: focused serving comparison

`DGPP_EXPERT_BATCH_SCATTER=1` combines up to 32 completed reconstruction chunks
into one 256-thread-block scatter launch. The kernel uses a second grid axis
for independent chunk plans. Ready experts still execute independently of
outstanding reads. Source leases remain owned until each staging event covers
its last kernel read. Repeated NVFP4 scalar globals retain the final copy in
submission order, avoiding overlapping stores; payload and scale regions remain
disjoint. Single-record completions and unsupported backends use the existing
path. This adds no expert, execution or staging payload allocation.

One binary runs a paired off/on comparison with publication profiling disabled,
fixed depth three and otherwise unchanged production settings. The workload is
the same four-request tool sequence and three short recomputations described
above. The binary SHA-256 is `89105da65cd51d8cf896bf2d136a00e6a37953a3e95dfcf7f2f42a8dfc9b7388`.

| Request | Off prefill / decode | On prefill / decode |
| --- | ---: | ---: |
| tool.0 | 39.209 s / 22.97 tok/s | 39.203 s / 23.20 tok/s |
| tool.1 | 1.838 s / 24.77 tok/s | 1.818 s / 24.99 tok/s |
| tool.2 | 3.005 s / 28.08 tok/s | 2.940 s / 28.36 tok/s |
| tool.3 | 0.452 s / 29.52 tok/s | 0.417 s / 30.04 tok/s |
| -anchor.0 | 8.323 s / 27.99 tok/s | 8.241 s / 27.85 tok/s |
| -anchor.1 | 2.936 s / 52.88 tok/s | 2.946 s / 52.65 tok/s |
| -anchor.2 | 1.613 s / 52.32 tok/s | 1.626 s / 52.31 tok/s |

The three continuation wall times total 72.471 seconds off and 71.600 seconds
on, a 1.20% reduction in this pair. Cold prefill is effectively unchanged;
the final zero-expert-read probe remains approximately 52.3 decode tokens/s.
The template selects batching because all three miss-heavy continuations improve
in this pair, but the gain is small and has not been established across repeated
campaigns or concurrency levels. These diagnostic timings do not replace the
older full benchmark rows.

Every request matches normalized response choices (generated tool-call IDs
removed only), usage, speculative acceptance and expert read bytes on both
ranks. Both ranks use the recorded binary and captured process swap is zero.
The 31-test host cache suite and five-test CUDA packing/scatter suite pass on
Spark-1. CUDA coverage includes mixed mapped/device sources, pitched copies,
padding/guards, graph replay and repeated-global ordering. The launcher,
benchmark client and campaign Python suites also pass.

### Preparing a matched MXFP4 deployment

The MXFP4 streaming template now uses the NVFP4 template's engine settings,
cache budget, fixed depth-three speculation and execution switches. Native
MXFP4 checkpoints can explicitly declare packed MXFP4 Engram tables using
`engram_embed_quant: "mxfp4"`, `engram_embed_block_size: 32`,
`engram_embed_scale_dtype: "ue8m0"` and the exact `engram_embed_layers` list.
The loader continues to validate the actual tensors. This allows the comparison
to retain the same packed Engram data rather than introducing the upstream
checkpoint's larger FP8 tables as a second variable. An unmodified upstream
checkpoint still selects its original FP8 tables.

Both formats now keep DSpark draft experts resident when expert paging is
selected. Their allocation is subtracted from the configured expert budget
before sizing the target cache; the expert image contains target layers only.
Model tests cover budget refusal and bitwise resident/paged draft, verify,
rollback, tool and JSON constraint behavior for both formats.

This preparation does not give MXFP4 the NVFP4-specific b12x prepared-expert,
GPU paging and chunk-direct execution paths. A serving comparison therefore
measures the available implementations, not an isolated prediction from the
4.25-versus-4.5-bit weight ratio. Cache capacity, miss/read counters, prefill and
decode must be reported separately. No MXFP4 serving speedup has yet been
established on the current implementation.

### Two-Spark MXFP4 serving result (2026-11-17)

The focused seven-request comparison completed. MXFP4 remains serving on the
existing port, keys and client-facing model alias, as requested. The underlying
checkpoint is native MXFP4 for routed experts with the same packed Engram data
as the NVFP4 baseline. The configured expert budget, resident draft experts,
fixed depth-three speculation, request slots, KV pool, prefix arena and I/O
settings are held constant. All sources used during serving are local SSD files.

The NVFP4 baseline uses `b752003d` and binary SHA-256
`89105da65cd51d8cf896bf2d136a00e6a37953a3e95dfcf7f2f42a8dfc9b7388`.
MXFP4 uses implementation source `0e4b5e949ccaa83622dab2de0c762c81a013f638`
and binary SHA-256
`073efab0fcc4400c7c9e510760c079864ee7bb7e3cf4b2c23fdc2ee3943e148d`
on both ranks. Changes since the baseline enable native MXFP4 with packed Engram
and resident drafts, and fix initial expert-image preparation; this is a
comparison of the available implementations, not a single-binary isolated
quantization experiment. These focused results do not replace the standard
campaign rows in `benchmarks.md`.

The target cache has the same 103,632,517,120-byte budget after charging the
resident drafts. NVFP4 provides 186,465 chunks of 553,728 bytes; MXFP4 provides
197,965 chunks of 522,240 bytes. At 18 chunks per expert, that is capacity for
10,359 versus 10,998 whole experts across the 40 target layers: **6.17% more**.
The difference includes smaller staging/execution reservations and chunk
alignment, as well as the 4.25-versus-4.5-bit representation.

All decode rates below are C1 engine decode throughput, excluding prefill and
queueing. Each cell is one focused request, not a repeated campaign mean.

| Request | NVFP4 prefill s | MXFP4 prefill s | NVFP4 decode tok/s | MXFP4 decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Initial 24,317-token tool prompt, zero prefix-cache tokens | 39.041 | 43.982 | 22.73 | 17.05 |
| Prose tool continuation | 1.830 | 3.396 | 24.96 | 17.91 |
| Code tool continuation | 2.998 | 3.020 | 28.17 | 22.45 |
| Repeated prose continuation | 0.437 | 0.258 | 29.52 | 30.39 |
| 2,209-token uncached probe, first pass | 8.319 | 7.534 | 27.79 | 27.60 |
| Same uncached probe, second pass | 2.929 | 4.295 | 52.73 | 31.28 |
| Same uncached probe, third pass | 1.617 | 3.447 | 52.65 | 31.20 |

More capacity did not consistently reduce reads in this implementation. The
three tool continuations read 78.919 GiB per rank on NVFP4, versus
141.803 / 135.738 GiB on MXFP4. The final probe read zero expert bytes on NVFP4
and 13.271 / 12.050 GiB on MXFP4. The latter is therefore not a resident-only
kernel comparison. Native/generic execution and admission behavior differ,
and generated continuations diverge; the byte totals alone do not identify
which mechanism accounts for the extra reads.

The initial tool-call choices match after removing generated call IDs: both
select `list_directory` with `/home/jon/dgpp`. The other six response choices
differ. All seven input token counts match, but continuation lengths and
acceptance differ, so this run establishes neither quality equivalence nor a
pure weight-format speed effect. Both ranks stayed at zero process swap and
ran the recorded binary; their operation streams match at the end. The first
request has zero cached prefix tokens. No build, transfer or other serving
world ran during the measured requests.

MXFP4's extra capacity is real, but the pair shows no overall serving win for
its current path. Porting the NVFP4-specific b12x/GPU-paging/chunk-direct path
remains unaddressed. Broader concurrency, quality and long-context evaluations
remain unrun. MXFP4 was left serving, rather than switching back to NVFP4.

### Interrupted image preparation fixes

The failed first attempt never reached timing requests. Its ten-minute launcher
deadline killed image construction, and service restarts accumulated abandoned
named temporary images until the disks filled. A service EnvironmentFile also
overrode the initial checkpoint-cache drop-in; the active EnvironmentFile now
selects the MXFP4 cache directly.

The launcher now accepts an explicit `--startup-timeout`; the successful first
build used 3,600 seconds with automatic restart disabled. Both models completed
construction after the old deadline. On filesystems supporting `O_TMPFILE`,
expert images remain anonymous until publication, so killed builders release
incomplete images automatically. Complete-image disk allocation is reserved
before packing where the filesystem supports it; allocation failures such as
ENOSPC fail before the long write. Filesystems without anonymous-file support
retain the named temporary-file fallback.

Seven expert-store tests pass on Spark-1, including publication/readback and a
SIGKILL regression that leaves no partial image. The 28-test launcher suite
passes with one optional dependency skip. The successful build used one process
per rank throughout. Normal `Restart=on-failure` supervision was restored after
the comparison without restarting the running MXFP4 world.
