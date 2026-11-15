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

Finding 4 has measured follow-through and a retirement-fence fix, but is not
fully closed. CUDA scratch reuse is now stream ordered rather than host-fenced.
An immediate-ready-expert variant was compared with grouped misses and did not
improve this tool sequence. Acquisition still reconstructs each expert
synchronously; truly asynchronous reconstruction across multiple experts is
not implemented. The review remains open on that architectural issue.

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

The remaining design work is to separate demand submission, reconstruction
completion and execution readiness within the existing reservations. It needs
explicit ownership of in-flight source chunks and staging leases, cache
publication only after completion, and ready-group execution that avoids both
whole-group waits and a persistent-kernel launch per expert. The current trace
motivates that work; the immediate-launch experiment does not close it.
