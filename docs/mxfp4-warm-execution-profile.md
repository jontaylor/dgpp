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

### Grouping adjacent expert records on local SSD

Each rank reads its own MXFP4 shard from local NVMe. When consecutive
expert records have reserved staging buffers, the reader can fill several
buffers with one `preadv` call. Direct I/O on these GB10 hosts required
64 KiB alignment for the component buffers. Each NVMe block queue also
reported a 256 KiB hardware request limit. The file-call count measures
submissions by DGPP. The kernel and device can split those submissions.

At revision `d7d7f258`, both ranks used binary SHA-256
`e13f6814ba1b37623717d266ed6aac9795d9fbd50944bf56dd03bbf387bc28a3`.
Every width used 20 MiB boundary prefetch and resident DSpark graph replay.
On the cold 24,292-token tool call each rank read 101.40 GiB. One record
per submission took 207,666 file calls per rank. Four took 55,093–55,719.
The generated responses, accepted drafts, rank operation streams and
per-rank expert bytes matched at every width. Both ranks had zero process
swap. These are single-client diagnostics.

| Maximum records per submission | Fresh process starts | Cold prefill (s) | Decode with expert reads (tok/s) | Warm zero-read decode (tok/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 36.90–36.97 | 32.65–32.82 | 45.04–45.06 |
| 2 | 1 | 36.50 | 32.75 | 45.11 |
| 4 | 2 | 37.45–37.60 | 32.89–33.03 | 45.12–45.15 |
| 8 | 1 | 37.37 | 32.90 | 45.06 |
| 16 | 1 | 38.51 | 32.58 | 45.15 |

Four records improved decode with expert reads on both starts while
extending cold prefill. The phase-specific controls select the maximum
prefill group with `DGPP_EXPERT_PREFILL_ADJACENT_READS` and decode group
with `DGPP_EXPERT_ADJACENT_READS`. Omitting the prefill override uses the
decode setting for both phases.

### Where warm verification spends time between GPU launches

A two-rank Nsight Systems capture followed the same 96-token code completion
through 31 speculative rounds. Both ranks served every routed expert from the
resident cache. The capture used the paged target graph, resident draft replay,
20 MiB boundary prefetch and four-row verification. The target accepted 65
proposals in each capture.

From the first route packet in layer 0 to the end of the routed kernel in
layer 39, the baseline averaged 53.46 ms per pass on each rank. Taking the
union of the GPU kernel intervals leaves 0.225 ms on rank 0 and 0.220 ms on
rank 1 with no kernel running. Between the end of that span and the next
pass's first packet, the 30 complete intervals averaged 10.52 ms. The
corresponding kernel-free times were 1.14 and 1.39 ms. That interval includes
the target vocabulary projection, distributed token selection, drafting and
the beginning of the next target pass. A kernel can be waiting for memory or
its peer while it appears active. These intervals measure GPU scheduling
and overlap, with profiler overhead included.

The packed b12x reader also ran with lower L2 retention on its weight codes.
Across the 40 routed kernels per pass, rank 0 went from 28.93 to 28.52 ms
and rank 1 from 28.59 to 28.97 ms. The first-packet-to-last-routed interval
was 53.82 ms with the hint. The two ranks execute concurrently, so adding
their kernel times would misstate the latency seen by a request.

The baseline serving binary was built from `2df097cf`, SHA-256
`ed15cfc94f1a7b2ff8ec444ba075ddbdebac20bbabe2036104bb8d0b59583bbe`.
The comparison linked the low-retention b12x export from `37de8533`, SHA-256
`fb50bffe2dc673c30f13670265749050a78d29303db3bb0c31a63047731a0e3c`.
These are single-request diagnostic traces. Uninstrumented serving runs
determine the deployment choice.

### Compact staging for the four-row dense projections

The grid-scaled FP8 projections load only the live BF16 activation rows
from global memory. The standard kernel reserves two shared-memory panes
of 16 rows each and initializes the unused entries once per block. With
`DGPP_MMA_COMPACT_ACTIVATIONS=1`, launches with one through eight rows
reserve only the live rows in each pane. The tensor-core instruction still
computes a 16-row tile. Its discarded outputs reuse valid activation
fragments. Weight traffic, split-K boundaries and the stored rows' MMA
accumulation order stay the same.

In the four-row verification case this frees 12,672 bytes of shared memory
per block. The two-Spark control/candidate/candidate/control comparison
passed the row-invariance and projection tests and preserved serving output
and acceptance. Compact staging had mixed projection timings and did not
improve warm serving throughput across the four starts. The deployment
continues to use the standard staging.

### Packed expert reads inside a warm verification pass

The TP2 replay used four target rows with six routes per row. Across 34
recorded layer calls, the mean was 17.59 distinct expert shards per rank.
Each shard contains 9,123,840 packed weight and scale bytes. The replay
rotated eight copies of the prepared weights and checked routed FP32 outputs
against the original reader. Synthetic activations and real routing IDs
were identical across variants.

With five copy stages and low L2 retention for the weight codes, mean
expert time was 679.18 µs. That represents **236 GB/s of distinct expert
payload per Spark**. The numerator counts each selected shard once. It
excludes activations, scratch traffic and physical rereads. The advertised
273 GB/s would take about 588 µs to read the same payload from DRAM once.
The difference is a comparison of byte volumes and elapsed time. The
hardware trace does not measure how many of those bytes hit L2.

Moving the routed activation address lookup outside the K loop and issuing
the packed weight and scale copies first reduced mean replay time to
678.77 µs. It won 19 of the 34 route groups. In two fresh C1 starts per
configuration, the three read-free serving continuations ran at 45.53–45.58
tok/s for the original schedule and 45.48–45.52 for the changed schedule.
The first two continuations, which needed expert reads, ran at 33.12–33.18
and 33.22–33.28 tok/s respectively. All seven responses and accepted
proposals matched. These are engine decode rates in a targeted comparison.

The source for the export options is `92d3684a`. The compared server hashes
were `2b15ef35ff7a9f7743bd75529584f3f6d984a19e1a79a9a1e4f3470b53f0f074`
and `ba0f1f1de65d7315797fa8bdf259df03d9a9a6537996648c7164a28c46d111bf`.
To reproduce the candidate export, add `--reuse-route-offset` and
`--stage-weights-first` to `--format mxfp4-nibble --pipeline-stages 5
--stream-weight-codes`. The down projection's copy depth can be set
independently with `--down-pipeline-stages`.

### Which work already overlaps during decode

The two-rank, 31-round resident trace described above also identifies which
kernels ran simultaneously. The following values are elapsed milliseconds
per target pass between the layer 0 route packet and the last routed kernel
in layer 39. Each row measures the union of its intervals on one GPU.

| Interval | Spark 1 | Spark 2 |
| --- | ---: | ---: |
| Whole span | 53.46 | 53.46 |
| Routed expert active | 28.93 | 28.59 |
| Routed expert running alone | 24.48 | 24.01 |
| Dense projection running alone | 6.65 | 6.67 |
| TP collective active | 2.19 | 2.66 |
| TP collective running alongside L2 prefetch | 2.09 | 2.56 |
| Shared expert running alone | 0.014 | 0.010 |
| No kernel running | 0.225 | 0.220 |

The TP collective already overlaps weight prefetch for about 95–96% of its
duration. Eliminating every kernel-free interval in this span could recover
at most 0.23 ms per pass. The larger opportunities are inside active kernels
and in the number of target passes needed to generate the answer. Between
successive spans, the ranks spent 5.22–5.33 ms running projection kernels
alone. Only about 0.05 ms of that 10.52 ms interval had a CUDA copy active
with no kernel running. Device loads in kernels, including weight reads on
the unified memory bus, are invisible to the CUDA copy count.

Inside the target span, the two attention output projections were the
largest sites running alone after the routed experts. The split-K
`wo_b` launch (80 × 2 blocks of 256 threads) accounted for 4.03–4.11 ms
per pass. The preceding `wo_a` projection accounted for 2.95–2.97 ms.
The corresponding intervals exclude every other kernel and CUDA copy on
that rank. This identifies about 7 ms of exposed attention output work
in the captured four-row shape.

The vocabulary head accounted for most of the projection time between those
spans. Each of the 30 complete intervals contained two full reads of the
rank-local head, one to verify the target and one to propose the next block.

| Mean per interval (ms) | Spark 1 | Spark 2 |
| --- | ---: | ---: |
| Target vocabulary projection | 2.05 | 1.98 |
| Draft vocabulary projection | 2.05 | 2.00 |
| Time across both projections with no overlapping kernel or CUDA copy | 3.99 | 3.87 |

Each projection reads a 64,640 × 5,120 shard. Its lossless 12-bit weight
payload is 496 MB per rank, excluding row metadata and escapes. The two
observed projection rates correspond to roughly 242–251 GB/s of packed
payload per Spark. For comparison, one pass through those bytes at the
advertised 273 GB/s takes 1.82 ms. The draft and target consume different
hidden states. A history match that supplies a complete proposal can skip
the neural draft projection while preserving target verification.

### Reusing the aligned prefix in a tool reply

When DSpark commits several tokens in one verification pass, the committed
prefix can cross a cache block boundary inside that pass. The eager engine
can save the target and draft state at that interior row. A fixture resumes
with a changed suffix and compares prefill, draft and verification logits
bit for bit with a snapshot taken at the same position during ordinary execution.

The two-Spark comparison used the same directory request, actual directory
entries and follow-up at concurrency one. On two fresh starts each, the
previous binary (`ec9a7886`, SHA-256
`c1e700be2cd9eaa898ceb835167044c9b8b4cf0802df449223a4f890e9e19b88`)
attached 24,192 reply-prefix tokens and processed 178. With the interior
snapshot (`37ca0482`, SHA-256
`92ebbaa02c9960797c67e7db8b8ffeea72c1f8c42006194d3f7cb5e4f2f51f84`),
those figures were 24,320 and 50. Across the two warm directory exchanges
on each start, combined call and reply wall time fell from 1.95–1.99 s to
1.60–1.73 s. The first exchange after the workload change took 2.39–2.42 s
before and 2.49–2.51 s after. The request still needed nine verification
rounds to produce the tool call. These are direct service times for the
tool-enabled conversation, including prefill on both legs.

### Four-token history tail on a completed edit

The tool-enabled edit contains 82,756 prompt tokens and requires reproducing
a small Python fixture with exactly one value changed. Each arm ran the edit
three times. The last two reused 82,688 prompt tokens, read zero expert bytes
and returned the correct 376-token answer. A separate directory request
checked the tool name, its arguments and the reply after the tool returned.

The two configurations below used the same `37ca0482` binary and fixed neural
depth three. The wider configuration also lets a sufficiently long history
match supply the complete proposal block. The target verifies the proposed
tokens in both cases.

| C1 measurement | Two extra tokens | Four extra tokens with whole-block lookup |
| --- | ---: | ---: |
| Warm edit, engine decode (tok/s) | 74.38 | 87.87 |
| Warm edit, complete request (s) | 5.41 | 4.65 |
| Target passes per completed edit | 65 | 50 |
| Target passes per directory tool call | 9 | 7 |
| Warm directory call plus reply (s) | 1.69 | 1.56 |
| Ordinary warm prose and code, zero expert reads (tok/s) | 46.38 | 46.45 |
| First prose and code continuations, with expert reads (tok/s) | 33.63 | 33.53 |
| Cold 24,292-token tool prefill (s) | 36.53 | 36.85 |

These are one process start per configuration with both ranks at zero swap.
The directory time averages two completed exchanges. The seven standard
responses matched, including both prose and code continuations. The wider
tail helped where the prompt contained text the answer needed to reproduce.

A fresh start on revision `f0a3a1f7`, SHA-256
`aecb709392c95b404d4f33bdf43578e3154f664f55b1c22cea74907fd9156297`,
repeated the wider result at 87.85 tok/s for the edit and 1.56 s for the
directory exchange. The two-token and four-token warm edits each processed
68 uncached prompt tokens. Rank-local expert executions over the whole
request fell from 71,863 to 67,774. Wider verification reuses some experts
across rows and still needs more distinct experts in an individual pass.

This revision can also shorten the optional history tail when requests join
a batch. On a 16-row fixture, a single request committed eight tokens,
three requests each committed five on the next tick, and the original
request went back to eight when it ran alone. On both Sparks, 16 simultaneous
tool calls returned the requested name and exact path, each in 180 output
tokens. The scheduler recorded 16 requests in one decode step and the ranks'
operation streams matched. This checks service capacity on a constrained
request. Concurrency throughput requires the deployment campaign.

The native packed-scale expert object on these builds uses five copy
stages, low-retention weight-code loads and the native FP4 nibble decoder.
After generating the standard NVFP4 and MXFP4 b12x objects, generate the
packed-scale specialization in the same AOT directory on SM121:

```sh
python scripts/export_b12x_moe.py --format mxfp4-nibble --pipeline-stages 5 \
  --stream-weight-codes --nibble-decoder native-fp4 --output "$aot_dir"
```

Configure the server build with `-DDGPP_B12X_AOT_DIR="$aot_dir"`. The
two-rank MXFP4 streaming template contains the runtime settings. The
expert policy is whole-expert caching on local SSD with 109,121,288,192
bytes per rank budgeted for paging, including its staging and execution
space and the resident draft experts. The target graph issues 20 MiB
merged-range prefetch windows alongside TP collectives.

### Moving work across the draft and target boundary

The two-Spark screen used fixed depth three, the four-token lookup tail and
the same native expert object throughout. Each experimental setting used a
fresh process. Prose, code, tool arguments, acceptance counts and per-rank
read volumes matched the wider-tail reference for all seven requests. The
LRU implementation passed executable checks in each arm. Times below use
engine decode for prose and code and combined service wall time for the
two warm directory exchanges.

| Change from wider-tail reference | Warm decode (tok/s) | Decode with reads (tok/s) | Directory exchange (s) |
| --- | ---: | ---: | ---: |
| Reference before individual changes | 46.55 | 33.51 | 1.560 |
| Project draft input after the last required target layer | 46.27 | 33.51 | 1.567 |
| Project draft input during token selection | 46.49 | 32.86 | 1.577 |
| Route before issuing shared-expert reads | 46.23 | 33.56 | 1.587 |
| Return tile winners for neural draft logits | 46.52 | 33.56 | 1.580 |
| Draft tile winners plus sparse grammar fallback | 46.48 | 33.39 | 1.589 |
| Reference after individual changes | 46.51 | 33.65 | 1.582 |

Both ranks captured and used the side-stream projected input. Earlier issue
can compete with work still running on the same memory controller. The
timings show the net effect of each schedule. They cannot isolate a change
in physical bus utilization.

Two separate fixed-width checks kept the lookup tail at two and changed
the neural depth. Depth two took 109 prose and 144 code verification passes.
Depth four took 74 and 113. Depth three took 82 and 117. Six of the seven
standard responses changed at depths two and four, so their respective
42.29 and 43.76 warm tok/s also reflect different generated text. The
completed LRU and directory tasks passed at both widths.

The final comparison also tested limiting the neural vocabulary head to
the required proposal rows, omitting unused confidence, and returning
only the picked chain token (`DGPP_DSV41_DRAFT_OUTPUTS=compact`). On two
starts, warm decode was 46.43–46.54 tok/s and decode with reads was
33.43–33.55 tok/s. With the ordinary draft outputs these ranges were
46.51–46.55 and 33.51–33.65. The completed warm edit took 4.65 versus
4.64 s, respectively. The selected template uses ordinary draft outputs.

Against the preceding production binary (`fafa9116`, SHA-256
`e9b2eaff13e4e0e7e5ab0eb830ee0ba784ce6d797cc62694e69f45b268bc904c`),
the selected two-Spark configuration produced the following focused
single-client results. Each range spans two fresh starts. The older
configuration also completed the checked LRU task.

| Measurement | Preceding deployment | Selected deployment |
| --- | ---: | ---: |
| Warm prose and code, zero expert reads (engine tok/s) | 44.81–44.83 | 46.51–46.55 |
| Prose and code with expert reads (engine tok/s) | 32.55–32.71 | 33.51–33.65 |
| Cold 24,292-token tool prefill (s) | 37.39 | 36.51–36.61 |
| Completed warm LRU implementation (engine tok/s) | 57.49–57.58 | 59.91–60.27 |

The separate service and kernel measurements support this recipe for
single-client tool traffic. Physical DRAM bandwidth, throughput under a
sustained concurrent workload and performance on long generations from
new user conversations remain open.

### Completed turns through the coding client

On the selected binary, Codex used the configured `spark-primary` route
through OpenCodex. A request to run `pwd` executed the shell command once
and returned the correct directory on both attempts. Client wall time fell
from 20.97 to 3.97 s as cached input rose from 17,408 to 35,072 tokens
across the model calls in the turn. Both attempts used 16 decode steps.

A separate turn fixed an out-of-order update in a Python function, edited
the file and ran the existing tests. Both implementations preserved the
tests and passed an independent check with one-pass inputs, timestamp ties
and falsy values. The initial run took 38.96 s. Repeating from the original
source in the same directory took 22.11 s. The repeat computed 1,061 prompt
tokens and spent 4.13 s in prefill and 15.08 s in 201 decode steps. The
client reported 802 and 742 output tokens across the two turns. Differences
in cache reuse and generated work both affect these wall times.

A third task required creating an interval-coverage implementation from
scratch. The agent ran the visible tests, and its code passed 200 independent
randomized cases plus a case spanning a quadrillion positions. The turn
took 64.85 s, including 18.26 s in prefill and 43.65 s in decode. The
service accepted 1,363 draft tokens in 549 passes, or 2.48 per pass.
On a second attempt with the source removed, the agent again wrote a
passing implementation and ran its own 3,000-case check. The client reached
its 85 s time limit before it received a final answer. That attempt gives
no completed-turn latency. Warm cache reuse alone cannot predict how many
tool calls or output tokens the agent will choose.

### Resident verification on the deployed MXFP8 draft head

On 19 November both Sparks ran the selected MXFP4 binary from `f0a3a1f7`,
SHA-256 `aecb709392c95b404d4f33bdf43578e3154f664f55b1c22cea74907fd9156297`.
The profiled Rust continuation reproduced the earlier 96 output tokens, 31
target passes and 65 accepted predictions. It processed 26 uncached prompt
tokens, read zero expert bytes on both ranks and produced identical rank
operation streams. The ordinary service was restored after capture.

| Mean GPU time per target span (ms) | Spark 1 | Spark 2 |
| --- | ---: | ---: |
| First route packet to last routed kernel | 53.07 | 53.02 |
| Routed experts running alone | 24.49 | 23.57 |
| Attention output `wo_a` running alone | 3.02 | 2.94 |
| Attention output `wo_b` running alone | 4.11 | 4.01 |
| Query expansion `wq_b` running alone | 1.23 | 1.57 |
| CSA index scores running alone | 1.05 | 1.05 |
| Neither a kernel nor a CUDA copy running | 0.24 | 0.25 |

The table covers 31 spans. The two output projections had no overlapping
kernel or CUDA copy in these spans. The `wo_b` time counts the main
projection with two K splits. The subsequent reduction has its own kernel.

| Mean GPU time in the 30 complete intervals between spans (ms) | Spark 1 | Spark 2 |
| --- | ---: | ---: |
| Whole interval | 9.91 | 9.96 |
| Target vocabulary projection, lossless 12-bit weights | 2.14 | 1.97 |
| Draft vocabulary projection, MXFP8 | 1.47 | 1.39 |
| Draft Markov bias across three positions | 0.301 | 0.295 |
| BF16 rescoring across three positions | 0.015 | 0.016 |
| Draft projection start through final pick collective end | 2.16 | 2.16 |

The last row includes the draft projection, bias, rescoring, confidence,
pick collectives and gaps between them. Each MXFP8 head reads a logical
341,299,200-byte weight and scale payload per rank. Dividing by projection
time gives 232–245 GB/s of logical payload. Physical memory-controller
throughput was unavailable. The measured rescoring leaves at most 0.016 ms
per neural round to recover. A 10% reduction in `wo_b` projection time at
this shape would save about 0.4 ms per pass before downstream effects.

### Where a 16-minute branch review spent its time

The real Codex review went through OpenCodex to this two-Spark deployment.
It completed in 963.823 s across 34 sequential model calls. All 34 matched
rank-zero retirement records by timestamp and exact input, cached-input
and output counts. The service spent 118.788 s in request prefill and
834.9 s in request decode, or 86.6% of the client wall time in decode.
About 10.1 s fell outside those request clocks, including client and tool
work. The scheduler rounds each logged request's decode time to 0.1 s.

Across the calls, 2,372,480 of 2,460,316 input tokens came from prefix
cache. The service computed 87,836 input tokens and generated 34,290
output tokens. The client counted 26,175 of those as reasoning tokens.
Excluding each call's first token, decode delivered 34,256 tokens in
10,623 passes, or 3.225 tokens/pass and 41.03 tokens/s of request decode.
The input total counts reused conversation prefixes on each call. The
longest individual prompt was 113,732 tokens.

For the next review, use its completed Codex rollout and the UTC
rank-zero serving log covering that turn:

```sh
python scripts/serve_agent_turn.py "$rollout" "$rank_zero_log"
```

The command defaults to the last completed turn in the rollout. Pass
`--turn-id` to select an earlier turn or `-` for a log piped from the
serving host. It reports per-request wall clocks. Under concurrency these
clocks can overlap. Missing or ambiguous matches produce an error.

### Exact TP2 attention output projection

Each target layer projects four verification rows through a rank-local
5,120 × 4,096 `wo_b` matrix. The ordinary path writes two FP32 partial
sums to global scratch and launches a reduction. The exact kernel gives
the two K halves to two warps of the same block. They reduce through
shared memory and write BF16 output directly. Weights, scales, activation
precision and accumulation order match the ordinary path. The fast path
supports one through eight rows and needs no additional persistent memory.

The first version used 512-byte copies and explicit L2 lookahead. At
four rows its median fell from 96.94 to 79.71 µs in a graph replay
rotating five synthetic matrices. During warm serving, each rank executed
the new kernel 1,240 times across 31 verification passes, with zero expert
reads. The mean time from projection start through reduction completion
was 107.41 → 109.28 µs on Spark 1 and 104.98 → 107.50 µs on Spark 2.
The separate, uninstrumented C1 comparisons also showed no decode gain.

Changing the synthetic weight base offset by up to 12,032 bytes preserved
the five-matrix advantage. Increasing the rotation to 40 distinct matrices
(800 MiB of weights per rank, one matrix per target layer) removed it.
On that rotation the 512-byte version took 95.84 µs versus 94.74 µs for
the ordinary path. The replay demonstrates sensitivity to weight reuse.
It does not measure cache hit rates or physical DRAM traffic.

The revised kernel issues both 1,024-byte copies for each row of its K
half before consuming the first window. It uses the low-retention bulk
load and omits the separate L2 prefetch. The integrated four-row replay
took 97.38 µs versus 101.84 µs for the ordinary path. Output bits matched
for every width from one through eight, including comparison with
separate single-row launches. Reproduce the 40-matrix geometry with
`dsv41_verify_dense_bench --wo-b-mxfp8 4 40`.

A targeted two-Spark C1 comparison ran the ordinary, exact, exact and
ordinary modes on the same binary built at `c30b4490` (SHA-256
`b7adc6e0a4b1687f54089b55d2a3d8cf528e802f1bce65450388c09894e93e89`).
Across the two runs of each mode, pooled engine decode was 46.66 versus
46.51 tok/s on zero-read warm continuations and 33.71 versus 33.59 tok/s
on continuations requiring expert reads. Mean cold 24,292-token tool
prefill was 36.55 versus 36.62 seconds. The tool call and reply, prose and
code outputs, speculative acceptance and expert bytes matched. Both ranks
had zero process swap and matching nonempty operation streams in each run.
The ordinary projection remains the deployment choice.

A separate Nsight capture of the two modes used a 32-token coding response
and reduced the expert budget by 3 GiB per rank to leave room for profiling.
Both modes attached 24,448 prefix tokens, read zero expert bytes and
accepted 21 draft tokens in ten passes. Across 400 projections per rank,
mean `wo_b` time including the reduction went from 106.26 to 108.32 µs on
Spark 1 and from 106.84 to 108.33 µs on Spark 2. This locates the loss
inside the projection even after removing its second kernel launch.

#### Following the result through the TP2 fold

The fusion hypothesis was that removing the global partials and reduction
would let both ranks reach the attention fold sooner. At four rows it
eliminates 160 KiB of partial writes and the corresponding reads per layer.
The ordinary kernel already completed its reduction, including the gap
before launch, in about 2.0 µs. Recovering all of that time with an equally
fast main projection would save about 0.08 ms over 40 layers, before peer
waits, in a roughly 62 ms profiled round.

In the same 400 layer visits per rank and mode, the time from the end of
`wo_a` to the first `wo_b` kernel was 0.47–0.49 µs. From the projection's
last write kernel ending to the TP fold starting it was 0.56–0.59 µs.
Every projection ran as a graph node with zero recorded overlap with
another kernel, CUDA copy or memset on its rank. The FFN weight prefetch
still overlapped the fold in both modes. From `wo_a` completion to the
first downstream residual update, the mean went from 156.88 to 153.53 µs
on Spark 1 and from 138.55 to 141.98 µs on Spark 2. Those are local
elapsed intervals with an intervening collective. Adding them would
double-count shared progress. The extra projection time was on the GPU,
with graph replay active and zero expert reads.

#### Replaying the producer

`dsv41_verify_dense_bench --wo-b-handoff` varies one part of the serving
sequence on a single GB10. Four rows pass through the same 40 synthetic
`wo_b` matrices in the same order, with identical BF16 inputs and outputs
checked across every layer. The real MXFP8 `wo_a` kernel either supplies
the input immediately beforehand, writes the same values to an independent
buffer while `wo_b` consumes precomputed values, or is omitted from the
timed graph. A per-layer activation buffer keeps the consumer's addresses
identical across cases. Serving reuses layer scratch. The dependent and
independent cases stream identical producer weights. CUDA event timing
covers each complete sequence.

| Producer before `wo_b` | Fused minus ordinary time per layer on Spark 2 |
|---|---:|
| Omitted, inputs precomputed | −3.65 to +1.50 µs |
| Writes the input consumed by `wo_b` | +6.78 to +10.06 µs |
| Writes an independent buffer | +16.18 to +19.69 µs |

Each range spans the medians of paired differences from four process
starts, with 12 rotating rounds per start. A positive value favors the
ordinary path. Adding the producer reproduced the reversal on one GPU.
Writing directly to the consumer's buffer reduced the fused penalty in
all four runs. In an initial two-start screen without the independent
buffer case, the isolated fused path was 0.21–3.79 µs slower and the
dependent pair was 10.12–12.99 µs slower. The isolated ranking varied.
The penalty with the producer present reproduced across both screens.

One behavior fusion lost is activation sharing. In the captured grid,
eight warps share a staged activation window across 64 output columns in
the ordinary block. The fused block covers eight output columns and its
two warps read separate halves of the BF16 input directly. More blocks
consume the same activation addresses. The trace reports 49,664 and
35,104 shared-memory bytes per block respectively. Both fit at most two
blocks per GB10 SM by that limit, giving ceilings of 16 and four resident
warps. The trace supplies allocation sizes. Actual warp occupancy and
physical DRAM traffic remain unmeasured.

A targeted attempt to hide the direct input load prefetched each
128-byte activation line into L1 before waiting for the corresponding
weight window. With `wo_a` producing the input, it increased the median
paired sequence time by 3.88–4.51 µs per layer against the same fused
kernel without the hint across three process starts. The projection
outputs matched.

**Outcome.** The ordinary kernel serves the measured workload. The
classification of fusion as a design is inconclusive. The experiment
identifies a reproducible integration penalty and rejects the simple L1
hint. The relative contributions of cross-warp reuse, weight streaming
and memory-latency hiding remain open.

### Decode priorities for a long tool-using review

The November 19 review used 834.9 seconds of request decode across 10,623
target passes. Every millisecond removed from every pass would have saved
10.6 seconds. The four longest decode calls accounted for 408.9 seconds
(49%). Grouping all 34 calls by their prompt length gives the following
pass-weighted results. Each band's decode seconds, tokens and passes sum
back to the recorded turn.

| Prompt tokens at call entry | Calls | Passes | Request decode (s) | ms/pass | Decode tokens/pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Below 32,768 | 4 | 291 | 26.4 | 90.7 | 4.04 |
| 32,768–65,535 | 8 | 2,420 | 186.7 | 77.1 | 3.21 |
| 65,536–98,303 | 16 | 4,186 | 331.6 | 79.2 | 3.32 |
| 98,304 and above | 6 | 3,726 | 290.2 | 77.9 | 3.06 |

Pass cost has no monotonic trend across these calls. The final call alone
took 154.5 seconds, with 2.97 tokens/pass over 2,042 passes. The review log
lacks per-pass expert reads, proposal sources and verification widths.
Context, acceptance, SSD service and width can all change together.

#### What holds up the next layer on resident hits

The separate, zero-expert-read Rust capture on the selected binary
(`f0a3a1f7` above) used four verification rows at a roughly 24K context.
For each of 31 passes and 40 layers, follow the GPU route packet through
b12x, the shared expert's down projection, the ordered slot sum, the FFN
TP fold and the first dependent residual update. The routed expert lane
finished last on **all 1,240 visits on each rank**.

| Mean elapsed time per layer (µs, local GPU clock) | Spark 1 | Spark 2 |
| --- | ---: | ---: |
| Routed kernel | 715.9 | 689.6 |
| Shared down completion to routed completion | 612.1 | 589.2 |
| Routed completion to slot sum start | 0.82 | 0.78 |
| FFN TP fold | 26.9 | 55.4 |
| Routed start to dependent residual start | 746.7 | 748.9 |

The faster routed lane often paid the difference in the fold. Across
matched layer visits the difference in routed durations and the difference
in time from routed completion to residual start had correlation −0.946.
The smaller 400-visit ordinary-projection capture described above reversed
which Spark averaged the longer routed duration and showed the same
relationship (−0.934). This compares durations at the same logical layer.
No subtraction of timestamps between machines is required.

Inside the 53 ms target span, intervals with neither kernels nor CUDA
copies active totaled 0.24–0.25 ms per pass. In the 30 complete intervals
from the last routed completion to the next first-layer packet, an interval
union including kernels, copies and memsets found 0.90/1.60 ms idle on
Spark 1/2 out of 9.91/9.96 ms. Only 0.023 ms per rank had a copy or memset
as its sole activity. The rest includes the target head, rollback and picks,
proposal work and the next first layer. A gap on one GPU can coincide with
useful work on its peer. Device loads issued inside kernels are outside
the CUDA copy count.

#### Where expert misses add time

On the two ordinary-projection starts of the `c30b4490` comparison above,
the first prose continuation took **86.6–86.8 ms/engine pass** versus
**59.0–59.1** on its zero-read repeat. Code took **77.6–78.3** versus
**58.5–58.6**. Each comparison held the generated output and pass count
fixed (82 prose, 117 code). Decode-step timers exclude the uncached
prompt tail. The file-byte counters include it.

An earlier four-row MXFP4 capture with per-layer timestamps allows a
closer comparison. Pair the first prose and code with their respective
last warm repeats by request, pass ordinal, layer and rank. The repeats
had zero expert reads. The table gives Spark 1's mean **difference**
in layer duration, grouped by whole shards demanded in the first run.
Spark 2's means were within 0.002 ms of Spark 1's in every group.

| Shards read in first visit | Prose visits | Extra ms/visit | Extra ms/prose pass | Code visits | Extra ms/visit | Extra ms/code pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2,636 | 0.082 | 2.62 | 3,998 | 0.066 | 2.25 |
| 1 | 507 | 2.635 | 16.29 | 611 | 2.665 | 13.92 |
| 2 or more | 137 | 5.454 | 9.11 | 71 | 5.201 | 3.16 |

The differences include computation, cache handling, local reads and TP
waits through the end of each layer. The five layers contributing the most
extra time among read-bearing visits accounted for 26% of that time in
prose and 20% in code. The two top-five sets were disjoint.

#### Engineering order

1. **Earn more tokens from the passes the review already pays for.**
   The history index chooses the longest prior suffix, breaking ties by
   recency. For neural-tail extension the engine then checks whether that
   one continuation agrees with the neural prefix. Another prior occurrence
   can agree even when the first one diverges. Shadow a bounded search for
   an agreeing continuation wherever the existing rule declines to extend.
   Use only history available at proposal time. Score its extra candidates
   against subsequently committed tokens, recording proposal source, width,
   and the cost of both target rows and host search. Preserve the existing
   whole-block copy and grammar paths, neural prefix, row budget, rollback
   and target verification. The break-even is concrete. At 3.40 tokens/pass
   for the same answer, the review could afford 4.27 additional ms/pass
   before losing the gain. With unchanged pass cost it would save 43 s.
   Sparse matches in long reasoning stretches or more expensive wide
   passes can erase the benefit. Shadow matches establish coverage. A timed
   completed tool turn establishes the gain.
2. **Shorten the path from one exact miss to an executable shard.**
   Single-shard misses outnumber multi-shard misses 1,118 to 208 in the
   paired capture. Measure demand publication, first submission, last
   required read, native weights ready, routed completion, gate release
   and dependent residual on each rank. If assembly or admission lies after
   I/O on that path, try writing the packed result directly into an eligible
   retained extent and executing there. The existing packed admission can
   copy the prepared shard from execution scratch into the cache. Preserve
   early read submission, independent miss readiness, existing eviction
   choices and readers of the old extent. Fall back to scratch if reserving
   space would delay demand or displace an active reader. Recovering
   0.25 ms per single miss would save 1.55 ms/prose pass and 1.31 ms/code
   pass at the observed frequencies. If the residual instead follows the
   last SSD completion with little intervening work, prioritize avoidance
   of those demanded reads. Read volume alone cannot establish that choice.
3. **Time routed-kernel changes through the FFN residual on both ranks.**
   The long routed tail offers room for an improvement to reach the join.
   The prior packed W4A8 replay is a concrete candidate. Compare it with
   today's five-stage packed W4A16 reader on identical inputs, routes and
   width, including activation quantization, slot handling, the overlapping
   shared expert and the TP fold. A 3.5% saving on a 700 µs routed lane
   would be about 1 ms across 40 layers if both peers deliver it through
   the residual. The historical single-layer result used different activation
   inputs and a different timing boundary. Check ordinary serving for route
   changes, expert reads, acceptance and correct tool completion before
   selecting a backend. A faster rank-local kernel with an unchanged
   downstream residual calls for investigating rank skew or displaced work.

The review's read-bearing fraction remains unknown. The 24K capture also
leaves long-context attention open. At that shape the index-score and
selection kernels occupied about 1.05 and 0.37 ms per pass respectively
on each rank. A short resident four-row continuation at a review-sized
context can distinguish growth in selection and attention through the
output projection from changes in pass width or paging. All comparisons
in this section are engineering diagnostics.

#### Following a single missing shard through the residual

A two-rank Nsight capture followed 31 target verification walks during a
96-token coding continuation. Each rank had 283 single-shard visits among
1,240 layer visits. Five successful grouped file calls requested the 9 MiB
rank-local shard. The other 872 visits issued zero expert reads and 85
issued reads for two or more shards. The profiling configuration reserved
3 GiB of additional headroom. The following intervals are medians across
the same single-shard visits on each rank. The baseline trace used binary
`26d7c7a71215` and the direct-admission trace used `fcbe9ef89310`, both
built from the experiment changes on base `e5cdf17a`.

| Interval, µs | Spark 1 | Spark 2 |
| --- | ---: | ---: |
| GPU route-packet kernel start to first read entry | 150 | 161 |
| First read entry to last required read return | 2,067 | 2,114 |
| Last read return to native scale pack complete | 240 | 223 |
| Missing-expert GPU execution | 259 | 259 |
| Missing-expert execution complete to route gate release | 402 | 262 |
| Gate release through slot reduction, TP fold and FFN residual | 46 | 180 |

The GPU start and end of the packet kernel bracket route publication and
release. On 278 of 283 single-shard visits on each rank, admission copied
the prepared payload into a retained extent. That 9,123,840-byte copy
took a median 66 µs on each rank. The ready lease already marks the packed
scratch valid. The baseline packs once and then copies those bytes. Whole
cache planning, submission and binding publication can also delay release
after the missing expert finishes. Overlap with the other rank matters at
the residual. Summing these per-rank or per-interval medians would lose it.

Mode 2 of `DGPP_EXPERT_DIRECT_NATIVE_ADMISSION` lets a single fully
missing decode expert use its final cache extent. Once all demanded reads
are queued, the engine asks the existing whole-expert planner for the
admission. Fixed load costs make that decision independent of outstanding
read timings. If an aligned extent is available after eviction, packing
writes it directly and the expert kernel reads it there. Extent fences
protect the previous and new GPU readers. The canonical scratch stays
valid for consumers of that representation. Multi-expert jobs, prefill,
learned load costs, projection reads and fragmented extents use their
ordinary execution and retention paths.

The mode-2 capture had the same miss histogram, 31 passes and 65 accepted
predictions. The admission copy disappeared from all 283 single-shard
intervals per rank. Median time from the end of missing-expert computation
to gate release fell from 402 to 14 µs on Spark 1 and from 262 to 15 µs on
Spark 2. From packet start through the dependent FFN residual, the medians
fell from 3.345 to 2.962 ms and from 3.335 to 2.968 ms. The five-read
window still took 2.07 and 2.12 ms respectively in mode 2.

Four uninstrumented service starts ran the same tool call, prose, code,
resident repeats and completed tool exchange in A/B/B/A order. Both ranks
used binary `fcbe9ef89310` and zero swap. Output, acceptance, prefix
attach, expert bytes and rank operation streams were checked. Engine
decode rates below pool generated tokens and scheduler step time within
each cache condition. These are focused concurrency-one diagnostics.

| Two-Spark MXFP4 decode, tok/s | Baseline A1 | Direct B1 | Direct B2 | Baseline A2 |
| --- | ---: | ---: | ---: | ---: |
| Prose and code with required reads | 33.51 | 34.15 | 34.27 | 33.62 |
| Fully resident repeats | 46.63 | 46.64 | 46.49 | 46.57 |

Direct admission improved the read-bearing trials by 1.9% on average.
Cold 24K prefill took 36.4–37.0 seconds across all four starts. The cache
budget and local SSD traffic were the same across configurations.

#### Alternate lookup occurrences

The optional `DGPP_DSV41_LOOKUP_ALTERNATE=shadow|1` examines up to 32
history occurrences when the usual lookup fails to extend the neural
draft. Candidates must agree with the entire neural prefix, fit the
available target rows and have a continuation in the committed history.
The shadow mode compares offered extra positions against subsequent tokens.

An eight-token committed suffix supplied no extra candidates across two
1,536-token review continuations. Counting the three agreeing DSpark
tokens toward that threshold yielded five extensions in each review
shadow, with eight matching positions out of 20 offered. On a completed
82,756-token-context edit, all four offered positions matched. Applying
them still took 50 passes to produce the same checked edit. The capped
review took 576 passes with the extension and 575 without, and the text
diverged. The coding continuation also differed and took 123 versus
117 passes. The directory call took seven in both arms. Lookup extensions
remain opt-in. These workload-specific probe rates do not establish a
same-answer throughput improvement.

#### Packed W4A8 through the two-rank FFN residual

The layer-10 replay used both ranks' local checkpoint shards, four BF16
activation rows and 33 observed top-six route masks. The W4A8 path included
GPU dispatch, input quantization, gate/up, SwiGLU and requantization, down
and ordered weighted combination. The W4A16 path included GPU route
preparation, the selected five-stage b12x reader and DGPP slot reduction.
Both ran the shared expert on a side stream.

An initial rank-1-only graph with uniform routing weights favored b12x on
all 33 masks. W4A8 took a paired median 5.1% longer. The TP2 replay used
the same PyTorch container on both nodes, nonuniform weights, the serving
shared-expert activation stride and block-32 scales. Each graph used a
synchronizing untimed fold before its measurement, the DGPP write-transport
all-reduce, a 20 MiB light range prefetch over a synthetic dense buffer,
and the four-stream FFN residual. Router weights, input activations and
shared-expert weights were synthetic and matched between implementations.
The measured shared branch used 48 blocks. Routing-score computation and
the next layer's consumption of prefetched weights were outside the timer.

| TP2 replay boundary | Route sets favoring W4A8 | Paired median W4A8 time change |
| --- | ---: | ---: |
| One FFN through residual | 30/33 | −1.12% |
| Four successive FFNs, expert preparation through local combination | 32/33 | −1.29% |
| Four successive FFNs, expert preparation through residual | 31/33 | −1.21% |

The four-FFN replay cycled through four successive recorded route masks,
with independent shared weights and prefetch buffers. Each table entry
takes the slower peer per replay, the median across six repetitions for
each backend and the paired change across route sets. The last row saved
9.6 µs per FFN. The advantage existed before communication in that same
replay. The local and TP2 setups differed in several respects, so these
runs do not identify a single cause for the earlier reversal.

The median relative L2 difference at the synthetic residual was 3.2%.
At 40 similar target layers, 9.6 µs per layer would save about 0.38 ms
per verification pass before any routing or acceptance changes. The
W4A8 reader addresses 128-column tile-major weights and tiled scale codes.
DGPP's cache publishes b12x-swizzled payloads and scale words. Serving
W4A8 within the existing budget needs a common expert layout across
prefill, verification and paging, or a reader that consumes the existing
cache addresses. Whole-model throughput, acceptance and tool quality with
that arithmetic remain open. The serving configuration keeps b12x.
