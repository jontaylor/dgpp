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
  Whole-model prefill chunks are unchanged. Prepared NVFP4 weights still need
  unpacking for this large-prefill recipe; small batches retain direct b12x.
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

The review remains open. Segment-local activation, bounded canonical grouping,
conversion counters and effective speculation eligibility are implemented and
validated above. They do not close the following gaps:

- Prepared-weight large prefill still needs a measured direct execution path.
- The remaining prefill cost needs GPU attribution; enqueue and synchronization
  wall counters cannot isolate unpack, expert, dense and communication time.
- The proxy check and direct check above used different requests. A matched
  tool-request sequence is still required to compare the paths.
- Miss groups still wait for synchronous acquisitions. Independent readiness
  requires follow-through after profiling the corrected prefill path.

No drive-limit claim is made. Exact drive-bandwidth characterization remains
outside scope at the user's request.

## Direct prepared prefill implementation

The W4A16 prefill tile can now consume the prepared NVFP4 code and scale
layout directly. It gathers only the current weight tile into shared memory,
then retains the canonical BF16 decode, MMA order and output rounding. Packing,
unpacking and prefill share the same layout-index functions.

The host executes contiguous prepared runs directly and reconstructs other
runs within the existing execution-slot budget. It retires native leases before
miss admission can relocate cache storage. Source descriptors may remain
metadata-only. W4A4 and GEMV keep their canonical fallback; small b12x batches
and graph replay are unchanged. `DGPP_MOE_PREPARED_PREFILL=0` retains the previous
large-prefill binding for comparisons.

The checkpoint-geometry fixture (H=5120, I=1152, top-6) passes bitwise comparisons
against canonical MMA at 65, 129 and 256 rows, including metadata-only source
descriptors, prepared hits, residency pressure and subsequent verification.
Prepared hits add no canonical unpack operations and leave no live native
leases. The 24 host cache tests and both-format paged-wave regression pass.
Serving latency validation remains pending.
