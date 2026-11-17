# MXFP4 warm prefill and verification profile

A two-rank GB10 profile on 17 November 2026 identifies execution costs on the
native MXFP4 deployment. Routed-expert kernels remain the largest target in
both phases. Warm prefill also spends substantial time in attention, dense
projections and communication. GPU idle gaps are a smaller part of these
captures, so removing host waits alone cannot explain a large speedup.
A running kernel does not establish full SM utilization or peak bandwidth.

## Capture and scope

The source was `68dbe581`, with implementation `80453335` in server SHA-256
`c870b9f50ed31a6f2e569e2333122e045c34e25225b319cb88bb4b890046513d` on both
ranks. The deployment used native MXFP4, fixed-depth-three DSpark (four target
rows per verification), 256 read buffers, eight workers, twelve execution
slots and the existing expert budget. Normal service configuration, port,
client alias and keys were preserved.

Nsight Systems captured CUDA API calls and individual graph nodes on both
ranks, with CPU sampling and context-switch tracing disabled. Existing MTP
phase and binding-publication timers were enabled only in the profiled
processes. No model build, transfer or other serving world ran during the
requests. Both ranks had zero process swap and matching final operation
streams. Normal production serving was restored before trace export.

A 2,209-token tool prompt was repeated three times before capture. A separate
24,317-token tool prompt was repeated twice before its capture. Prefix reuse
was disabled throughout. These are focused C1 diagnostics, not deployment
campaign rows or uninstrumented throughput claims.

| Profiled request | Prefill seconds | Engine decode tokens/s | Expert reads per rank, whole request |
| --- | ---: | ---: | ---: |
| Resident 2,209-token prompt | 1.387 | 54.06 | 0 GiB |
| Warmed 24,317-token prompt | 12.388 | 47.48 | 5.159 GiB |

The longer request remains a mixed execution/I/O case: approximately 98.97%
full-expert hits do not mean zero exposed miss cost. The preceding unprofiled
resident request took 1.382 seconds to prefill and decoded at 55.70 tokens/s;
tracing changed decode timing modestly.

## Largest costs

The prefill window runs from the first GPU operation to the first target
graph's demand gate. It includes initial draft setup and a short part of the
first verification; its span is about seven milliseconds longer than reported
prefill. Times below are rank-zero kernel durations; rank one showed similar
major execution costs. Concurrent kernel times are not additive.

| Prefill kernel family | Resident prompt, seconds | Warmed 24k prompt, seconds |
| --- | ---: | ---: |
| Routed FP4 grouped projections | 0.562 | 3.857 |
| Attention, including its partial-result merge | 0.205 | 2.244 |
| Dense GEMM/GEMV projections | 0.195 | 1.594 |
| Communication kernels, including rank waits | 0.127 | 1.362 |
| Hyperconnection operations | 0.081 | 0.822 |
| Shared experts | 0.066 | 0.569 |

The resident prefill also used 0.065 seconds of small-row b12x execution.
The longer prefill's routed gate/up projections took 2.406 seconds and down
projections 1.404 seconds in the main FP4 tile shape. Its attention window
kernel took 0.402 seconds, compressed-source attention 1.154 seconds and the
partial-result merge 0.660 seconds. These identify concrete kernel targets;
the trace does not establish that any of them has reached a bandwidth floor.

The resident request executed thirteen four-row target graph replays. Their
combined span was 0.754 seconds on rank zero and 0.748 seconds on rank one.
Rank-zero routed b12x execution took 0.414 seconds, dense projections 0.204,
attention 0.049 and communication 0.040. Shared-expert kernels took 0.197
seconds but had no shared-only interval within these target graph windows:
their work was hidden by other GPU work. Their duration is not an additional
0.197 seconds that another overlap optimization could remove.

Existing engine timers measured an average resident speculative cycle of
69.49 milliseconds: target verification 60.68, drafting 7.67, picking 0.94
and state handling 0.20. Verification includes the head and output handling
outside the captured backbone. The warmed 24k request averaged 73.49
milliseconds per cycle, including 64.87 verification and 7.62 drafting.

## Stalls and their limits

Across all GPU streams, resident prefill had 22–23 milliseconds with no
kernel, memcpy or memset active, about 1.6% of its window. The longer prefill
had 711–715 milliseconds, about 5.7%. These aggregate many short gaps; the
largest observed gap was under two milliseconds. Routing decisions,
metadata preparation and miss scheduling are relevant boundaries, but this
capture does not separate CPU computation, descheduling and storage waits
inside every gap.

Resident decode had 38–50 milliseconds of GPU-empty time over a roughly
0.899-second window. Only about eight milliseconds lay inside the thirteen
target graph windows. Repeated gaps around 0.8 milliseconds on rank zero
(and up to 1.3 milliseconds on rank one) occurred after hidden-state storage
and before the next host-driven collective. Source inspection connects this
boundary to materializing outputs, CPU token selection and starting the next
stage; it does not attribute every microsecond to one CPU function.

The resident target demand gates took about 4.7 milliseconds in total. In
the warmed longer request, gates took 61–62 milliseconds, of which 40–41
milliseconds had no other GPU work active. Engram readiness kernels took
88–100 milliseconds in that request's prefill window. Those waits matter,
but they are much smaller than its expert and attention execution costs.

Rank zero issued approximately 817,000 `cudaEventQuery` calls during the
resident capture and 891,000 during the longer capture, with about 0.31 and
0.35 seconds of summed API duration. The model polls graph completion while
checking demand flags. This overlaps GPU execution and is primarily a CPU
usage finding; those API durations are not extra wall time to subtract.
Likewise, summed `cudaStreamSynchronize` durations include GPU execution
already counted above and do not measure avoidable stall time.

## Work to advance or avoid

1. **Advance prefill shared-expert execution.** In
   `GlmMoeLayer::enqueue_host`, shared prefill work is issued in the routed
   grouped chain, after routing and host cache decisions. It depends on the
   FFN input, not expert IDs. A side stream can start that work before route
   readback and descriptor construction, using independent scratch and an
   event before ordered accumulation. The longer capture has about 0.569
   seconds of shared work and 0.711 seconds of GPU-empty gaps; these are
   opportunity sizes, not additive or promised savings. Resident verification
   already hides its shared work and should retain that behavior.
2. **Reduce prefill attention partial traffic.**
   `Csa2Layer::kPrefillSplit` is four, while prefill processes 128 rows per
   attention tile. Main and window attention write FP32 partial vectors;
   `csa2_attn_finish` rereads them, merges softmax statistics and applies the
   final transformation. A full 128-row, 32-local-head tile with four main
   splits and one window split reserves 40 MiB of partial vectors, excluding
   statistics. This is logical buffer volume, not measured DRAM traffic.
   First compare one/two/four main splits at the actual row/context shape;
   then consider a combined-source kernel that avoids these intermediates.
   The merge's 0.660 seconds is measured; changing splits also changes producer
   execution and reduction order, so both speed and numerical effects need
   checking.
3. **Keep selection and routine outputs on the GPU.**
   `SessionModel::finish_run` copies decode logits to pinned host memory and
   then copies them into `Outputs::logits`. It also materializes final hidden
   states, although the eager DSpark picker does not consume those host
   vectors. `local_row_maxes` and `mtp_row_max` scan host logits afterward.
   A serving output mode could return GPU maxima/candidates and retain full
   vectors only for logprobs, diagnostics or a required grammar fallback.
   Proposal prefixes are known before target execution, so per-row grammar
   state/mask preparation can move ahead of verification completion. Preserve
   exact constrained selection; simply dropping the mask is not an option.
   The measured pick phase is around one millisecond per cycle, so this is a
   bounded cleanup rather than the dominant throughput opportunity.
4. **Optimize the measured expert and dense kernels.** The resident target
   spends roughly 55% of its graph span in b12x and 27% in dense projections
   (overlap prevents adding these as an exclusive partition). Prefill also
   runs gate, up, activation and down as separate stages. A native MXFP4
   fused gate/up/activation candidate could reduce intermediate traffic, but
   its gain is unmeasured. Weight compression is another distinct option
   requiring quality evaluation. Earlier NVFP4 occupancy and W4A4 screens
   do not establish either success or failure for new MXFP4 implementations.
5. **Treat replicated rank work selectively.** Routers, indexers,
   compressors, hyperconnection operations and the low-rank query/window-KV
   projections are replicated across ranks; routed experts, shared experts
   and the larger attention/output projections are sharded. Router kernels
   alone used about 0.294 seconds per rank during the longer prefill, and
   index scoring/selection roughly 0.21 seconds. Computing those once would
   require transmitting their outputs and preserving ordering/graphs. Equal
   work on both ranks is not evidence of accidental duplicate expert
   execution, nor proof that another collective would be cheaper.

Existing reuse should be preserved: attention selections already carry
shape-checked reuse across layers, native expert layouts were not rebuilt in
the resident capture, shared verification work overlaps routed work, and
Engram already has chunk-ahead staging. The warmed larger request still shows
small residual Engram waits, but moving known gathers earlier is a secondary
opportunity here. Replacing graph polling with a wakeup mechanism could reduce
CPU consumption; a throughput claim needs its own comparison.

The next experiments should target prefill attention split/merge traffic and
earlier shared prefill execution, with resident b12x/dense execution as the
main verification kernel targets. No execution change or speedup from these
proposals is claimed by this profiling pass.

The first three follow-up changes and their focused serving screens are
described in [MXFP4 execution optimizations](mxfp4-execution-optimizations.md).
