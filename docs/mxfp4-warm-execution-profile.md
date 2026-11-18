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

## Where a paged decode round waits on the two Sparks

On the 18 November MXFP4 deployment the engine keeps the DSpark draft experts
resident. It feeds the pending token and three proposals to the target in one
pass. A matching history continuation can supply two further proposals. All
608 target passes in the long-tool diagnostic below had four rows and six
routed assignments per row per layer.

| Stage | Data movement and dependency |
| --- | --- |
| Prepare the round | DSpark consumes accepted hidden state and token IDs. The engine assembles the proposed IDs and positions. The GPU gathers embeddings and computes Engram row IDs. A side-stream CPU callback gathers the rows from each rank's local Engram block cache or backing file into mapped staging. Only the two Engram consumers need their tables ready. |
| Attention | Each rank reads its resident projections and its attention cache, writes new positions and computes its attention output shard. The selected FP8 path packs the attention result for the output projection. The ranks reduce the partial output before the residual update and FFN input normalization. |
| Shared expert and routing | Both need the same normalized input. The graph forks the shared FP8 expert before top-six routing and writes its contribution into separate FP32 slots. A single GPU block tests the routed IDs against the layer's published native bindings. On a hit it groups just the live assignments and pads each group to eight. The b12x kernel consumes the layer's bindings directly. |
| Expert miss | The GPU publishes exact IDs in a mapped mailbox. The CPU primes local SSD reads, issues the resident subset and schedules ready misses using the service stream. The rank-local MXFP4 shard occupies 9 MiB on disk. Native binding changes are tracked by layer so the host rebuilds only affected tables. The gate must see completed expert outputs and valid bindings before the main stream can consume them. |
| Combine and communicate | Routed weights stay packed through the b12x input. The kernel takes BF16 activations and writes unweighted FP32 output slots. After the shared lane joins, the GPU accumulates the slots once in routing order, then the ranks reduce their FFN contributions. The next layer depends on this residual update. Both Sparks work on the same expert IDs using their own weight shards. |
| Verify and commit | The vocabulary-sharded head scores the rows. The engine obtains each rank's local candidates, applies any grammar constraint, exchanges winners and accepts the matching prefix. It restores the accepted position, retains the accepted hidden state and starts the next draft. These host decisions occur once per speculative round. |

Previously the graph and miss service both ran the shared expert on a serviced
layer. Each had access to the same input and weights. Giving the graph sole
ownership also allows its shared work to overlap routing and SSD service.
Using the published bindings directly removes a copy of 18 KiB at each layer,
or 720 KiB per rank per 40-layer pass. These are logical pointer-copy bytes.

A 24,292-token uncached tool call and six tool-enabled prose/code follow-ups
exercised 24,320 target layers per rank. On rank zero, 22,845 layers read zero
expert shards, 1,225 read one, and 250 read two or more. Mean GPU layer spans
include routing, computation, communication and any exposed service delay.

| Rank-zero mean layer time | Original | Shared expert once | Also fuse route packet | Also skip unchanged layer scans |
| --- | ---: | ---: | ---: | ---: |
| Zero expert reads | 1.404 ms | 1.384 ms | 1.370 ms | 1.369 ms |
| One expert read | 4.205 ms | 4.124 ms | 4.080 ms | 3.961 ms |
| Two or more reads | 6.853 ms | 6.797 ms | 6.801 ms | 6.723 ms |

The per-pass GPU timestamps locate 13.84 GiB of expert demand per rank in
the target verification of the first prose and code follow-ups. Including
their uncached prompt tails brings the two requests to 22.46 GiB per rank.
A zero-read layer can still require host service to prepare a resident
expert's native layout.

### Uninstrumented serving comparison

The last change submits binding transfers immediately behind miss kernels,
then waits once for both before releasing the graph. Each pass also reuses its
host hit-accounting buffer and clears only the request flag in each mailbox.
The production control was built at `1165a662` (SHA-256
`d697f037cbb5bab6be8da49d70a1f27197957a874b4289015dc414fca201e98f`).
The selected binary was built from the source committed as `cc20721e`
(SHA-256 `f606b6210dbf5d86646ee2cc5ce2dd6ff2a7017f1847f0b98d2f967a31c9dd6f`).

Each binary ran the same seven C1 requests on two independent process starts.
The ranges below are the two observed results. Every start matched the
generated tokens, tool arguments, finish reasons, accepted predictions,
round counts, expert read bytes on each rank and rank operation streams.
The code continuation reached its 320-token cap on every arm. Each serving
rank had zero process swap. This focused decode comparison used the existing
11,524-expert pool and fixed DSpark depth three with lookup tail two.

| C1 engine decode | Control tok/s | Selected tok/s | Control ms/round | Selected ms/round |
| --- | ---: | ---: | ---: | ---: |
| First prose and code follow-ups, with expert reads | 30.91–30.96 | 31.91–32.07 | 88.80–88.93 | 85.71–86.15 |
| Last three follow-ups, zero expert reads | 42.53–42.57 | 43.58–43.61 | 64.37–64.43 | 62.84–62.89 |
| All seven responses, including the tool call | 37.67–37.72 | 38.73–38.78 | 73.44–73.53 | 71.42–71.51 |

Throughput divides tokens produced during decode by total engine decode
time across the indicated requests. Cold prefill of the tool prompt took
36.97–36.98 s on the control and 37.03–37.07 s on the selected binary.

### GPU activity and weight traffic

A separate two-rank Nsight Systems capture sampled the `gb20b` hardware
metrics at 2 kHz. A Rust continuation ran three times with identical output.
Its first request read 5.06 GiB of expert records per rank and the two repeats
read zero. Sampling excluded the first and last 300 ms of each engine decode
window. On the selected binary the graphics/compute engine was active
98–99% of the time across those windows. On the first request the SM activity
counter fell below 10% in 11.8% of samples on Spark 1 and 12.7% on Spark 2.
On the read-free repeats the corresponding ranges were 0.65–0.68% and
0.93–1.03%. A waiting demand kernel can keep the engine active while most
SMs have no work. The control exhibited the same pattern. Sampling affected
latency, so the throughput comparison above uses independent unprofiled runs.

This counter set provides SM, issue, tensor and copy-engine activity. It
omits physical DRAM throughput. The warm 320-token code response gives a
separate estimate of the routed weights consumed. A matched one-token request
with the same cached prefix and 26 uncached prompt tokens isolated prefill
expert counts. Both requests were read-free. The remaining 78,860 distinct
(layer, expert) visits counted across 117 target passes represent 6.15 GB of
packed expert weights per pass per rank, or **98.1 GB/s per Spark** over the
7.334 seconds of engine decode. This counts the 9,123,840-byte packed codes
and scales once per selected expert per layer invocation. Cache reuse,
physical rereads, attention, dense projections, the draft and other traffic
all affect the memory controller's actual byte count.

### Prefetching resident weights during decode collectives

Each rank can start reading the next known weights into its own L2 while
the current layer waits for its tensor-parallel reduction. The attention
reduction provides a window for the FFN norm, router and shared expert. The
FFN reduction provides a window for the next layer's attention or Engram
projection. Window size is a byte cap per boundary per rank. Routed experts
continue to use the existing demand path and local SSD cache policy.

The paged target graph captures the side stream's first window and joins
the stream once at the end of the layer loop. Replay therefore issues the
boundary reads without a per-layer CPU launch or host barrier. The serving
setting also enables boundary prefetch in the draft. The diagnostic used the
light issue rate with attention-internal prefetch off.

The C1 comparison used the same two-Spark binary on every start, built at
`fafa9116` (SHA-256
`e9b2eaff13e4e0e7e5ab0eb830ee0ba784ce6d797cc62694e69f45b268bc904c`).
Each start began with the 24,292-token tool prompt at zero cached tokens,
then the same six prose and code continuations. The two first continuations
exercise SSD demand. The final three read zero expert bytes on both ranks.
Engine decode throughput divides the total tokens generated after prefill
by the total decode time for the indicated group.

| Boundary window per rank | Warm decode, zero expert reads (tok/s) | Decode with expert reads (tok/s) | All seven responses, decode only (tok/s) |
| --- | ---: | ---: | ---: |
| Off, two starts | 43.49–43.60 | 31.79–32.10 | 38.68–38.79 |
| 4 MiB | 43.86 | 32.23 | 39.04 |
| 8 MiB | 44.42 | 32.45 | 39.47 |
| 12 MiB | 44.41 | 32.44 | 39.41 |
| 16 MiB | 44.63 | 31.90 | 39.23 |
| 20 MiB | 44.88 | 32.61 | 39.75 |
| 24 MiB | 44.82 | 32.38 | 39.63 |

20 MiB led the width comparison on both decode groups. The 0.06 tok/s warm
difference between 20 and 24 MiB is smaller than the spread of the two off
starts. Cold tool prefill took 36.89–36.90 s with prefetch off, 36.96 s
at 20 MiB, and 36.79–40.23 s across the other widths. These boundary reads
run only during decode.

The canonical deployment repeated the 20 MiB workload at 44.83 tok/s warm
and 32.78 tok/s with reads. Its cold prefill took 36.93 s. Separate timed
starts isolated the change within the speculative round. Across the same 316
read-free rounds, mean target verification fell from 55.91 to 54.14 ms and
drafting took 6.67 ms in both configurations. The two ranks generated matching
operation streams, accepted the same proposals, and agreed with the control
on outputs and expert bytes. Both ranks had zero process swap. These timings
measure latency and throughput; physical DRAM bandwidth was unavailable.
