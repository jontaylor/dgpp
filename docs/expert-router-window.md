# Exact-routing to expert-read timing

The November 16 two-Spark diagnostic finds little scheduling slack between the
completed FFN input producer and router start. The larger interval is between
completed expert selections and the next observed expert `pread`: typically
about 0.4 ms. Most of that remains after the CPU has received the routing IDs.
This identifies a place to try earlier exact-demand submission; it does not
establish that the entire interval is removable or predict serving speedup.

## Capture

Both ranks ran the existing deployed binary, SHA256
`e163306747831667813d601306ad38a68827a75cea118706751c75cb3307b8d6`.
No serving source or production settings were changed. The analysis checkout
started at `4c9a1e18`. Speculation stayed at fixed depth three, giving four-row
target verification; the target routes to six experts per row.

An initial 24,317-token tool prompt and prose continuation warmed the model
before profiling. Captured traffic then comprised a code continuation with
308 computed prompt tokens, repeated prose with 15 computed prompt tokens,
and an uncached 2,209-token short recomputation. Continuation output caps were
256; the short request used 64. These scripted requests produce actual model
routes, not an independently sampled user workload. Decode below includes
verification. This is a scheduling diagnostic, not a benchmark campaign or
uninstrumented throughput comparison.

Nsight Systems collected CUDA graph-node and OS-runtime activity on each rank:
`--trace=cuda,osrt --sample=none --cpuctxsw=none --osrt-threshold=1000
--cuda-graph-trace=node`, with collection started after warm-up.
`DGPP_EXPERT_READ_TRACE` was enabled in the diagnostic wrapper. Eight expert
read workers were identified separately from Engram workers by their almost
one-to-one `pread` and diagnostic trace-write counts. Each rank had 109,314
captured expert-worker `pread` calls. Exactly eight qualifying workers and one
CUDA process/device/context were required.

Read-trace Unix timestamps drifted relative to the profiler timeline. Therefore
**all measured intervals use Nsight timestamps for both GPU kernels and OSRT
`pread` calls**. Unix request windows only label requests; a 5 ms guard excludes
observations near either boundary. Do not subtract raw Unix file-trace times
from GPU or profiler timestamps.

## Results

Numbers are per-rank medians in microseconds. Each cell is rank 0 / rank 1.
Counts are matched read-associated observations per rank. Shapes remain
separate because router cost changes substantially with row count.

| Execution shape | Count | Producer end to router start | Router kernel envelope | Selections to next expert pread | Routing fence end to pread |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four-row graph verification | 1,423 | 0.64 / 0.64 | 28.13 / 28.16 | 382.18 / 423.26 | 313.33 / 347.38 |
| 128-row prefill | 73 | 2.69 / 2.75 | 106.91 / 108.64 | 428.32 / 335.57 | 415.50 / 319.06 |
| 2,176-row prefill | 20 | 1.87 / 2.30 | 1,240.37 / 1,256.10 | 383.80 / 517.21 | 362.87 / 500.82 |
| 15-row prefill tail | 31 | 2.24 / 2.24 | 44.83 / 44.67 | 391.14 / 545.12 | 374.38 / 529.54 |
| 33-row prefill tail | 12 | 2.66 / 2.72 | 57.73 / 57.86 | 1,266.82 / 1,047.26 | 1,250.63 / 1,030.50 |
| 52-row prefill tail | 40 | 2.64 / 2.72 | 59.95 / 61.78 | 815.95 / 823.15 | 801.47 / 810.70 |

For graph verification, the selections-to-read interval includes mailbox
handoff, a repeated service-stream router, routing D2H/fence, and subsequent
host/cache/queue work. Graph selections to service-router start took
28.70 / 27.78 us; that repeated router took 10.02 / 9.66 us. Its selections
to the routing fence took 16.91 / 17.07 us. The last table column starts at
that **service** fence. Medians of components do not add to a median total.
The graph selections-to-read p95 was 695 / 781 us.

The warm code continuation's graph selection-to-read medians were 346 / 359 us;
repeated prose gave 410 / 496 us. The short recomputation gave 574 / 578 us.
Thus the gap is present in warm continuations as well as the uncached request.
Larger prefills also spend about 1.25 ms in the router kernel envelope, but
that is routing/dependency work, not evidence of a free pre-routing window.

## Interpretation and limits

The FFN router normally depends on normalized activations produced after
attention and its stream update. Moving the existing router above those
operations would use the wrong input. Producer completion is a conservative
checkpoint: earlier partial row readiness or a split mHC implementation could
expose input sooner, but this capture does not measure that possibility.
Kernel envelopes can include in-kernel dependency waiting.

The initial capture identified an opportunity to submit a bounded first wave
of already-known misses earlier, ahead of nonessential host preparation, and
reuse the graph's selected IDs instead of rerouting on the service stream.
The latter alone accounts for only about 10 us. Preserve cache decisions,
staging lifetimes, demand priority and shared-expert overlap. Cache checks,
queueing and buffer backpressure still impose real dependencies; a measured
400 us interval is not a demonstrated 400 us saving.

Attribution assumes one serial model walk and no cross-layer speculative reads.
The analyzer associates the next expert-worker `pread` within the invocation;
it does not capture queue acceptance, record identities in OSRT, or physical
SSD dispatch. Host intervals stop at the next router globally, including graph
virtual streams, so resident draft routing cannot inherit a later target read.
Graph and service routers are paired one-to-one within a complete demand gate.

For this captured graph implementation, the fused FFN producer is the mHC dots
node two capture-node IDs before the router, with deferred comb between them.
The analyzer checks the producer's kernel identity and requires its execution
to follow the preceding graph router, preventing fallback to an old replay.
This is a capture-specific mapping, not a general CUDA graph dependency rule.
Routing-fence metrics require matching stream-sync correlation and two
same-stream D2H copies of the expected ID/weight size after selection.

All reported observations had recognized producers and validated routing-copy
fences. Reports retain excluded observations: 6,348 graph gates without a
service router, 229 serviced graph invocations without a new read in the gate,
seven larger host invocations without an associated read, and 626 small host
invocations without one. These exclusions include hits and resident draft
work; they are not silently treated as zero-latency SSD misses.

Nsight's automatic report conversion exhausted Spark-1's spare RAM after all
requests completed, killing the profiled target. The raw capture was retained
and converted with serving stopped; neither requests nor routes were rerun.
For future profiling, reserve memory for report conversion separately from
model execution. Normal serving was restored to the original binary above,
with zero process swap on both ranks. Instrumentation overhead and this
conversion failure rule out using these request wall times as serving results.

## Reuse

Keep generated profiles, read traces and reports untracked. Export one rank's
Nsight report to SQLite, then analyze locally using the Python standard library:

```bash
python3 scripts/expert_router_window.py \
  --sqlite RANK.sqlite --reads READS.bin --io-workers 8 \
  --windows WINDOWS.json --output REPORT.json
python3 tests/python/expert_router_window_test.py
```

Windows use the same format as the [cache replay guide](expert-cache-whatif.md).
Output contains per-shape and per-window statistics, individual observations,
worker evidence, exclusions and the analyzer's source hash. The seven focused
tests cover service pairing, selection/hash completion, cross-stream read
ownership, graph producer mapping and rejection of ambiguous worker counts.


## Early exact demand and captured-route reuse

The host path now primes the first two ascending selected missing experts as
soon as routing IDs are available, before host route copies, counting,
observation, segmentation or shared-expert preparation. The graph miss handler
primes from the acquire-loaded mailbox before accounting preceding layers or
rebinding the layer. It passes the captured device IDs and weights into the
existing service executor, removing the repeated router and its routing
D2H/fence. The main graph remains paused until service output is complete.
Supplied routes must belong to the executor's captured device scratch; host
weight diagnostics are cleared because that path does not copy weights back.

Priming uses existing demand tickets, coalescing, staging and the half-pool
lookahead bound. It changes neither residency nor access observations. Larger
populations than 384 experts keep ordinary submission. Resident graph hits
still execute without a host decision. Set `DGPP_EXPERT_EARLY_DEMAND=0` to
restore the former route/read path for a same-binary diagnostic control.

A focused two-rank NVFP4 diagnostic compared that control with early demand
using the same binary (SHA256
`1faedfd88ec4a40446d83ac90f73fa235c088e1d10e8c68180482c08bd903939`),
built from `3db61a47` plus this implementation. Both arms used identical
warm-up history, tool requests and fixed depth three. Measured requests were
a code continuation, repeated prose and a 2,209-token uncached prompt. All five
responses and usage counts matched after normalizing generated tool-call IDs.
Per-record expert-read counts matched on both ranks, with 318,168 read records
per rank including warm-up. Both ranks had zero process swap.

Median selections-to-first-expert-`pread`, in microseconds:

| Execution shape | Observations per rank | Control rank 0 / 1 | Early demand rank 0 / 1 |
|---|---:|---:|---:|
| Four-row graph decode miss | 501 | 393.86 / 362.77 | 156.53 / 161.81 |
| Small host prefill tails | 70 | 676.50 / 659.67 | 380.38 / 383.99 |
| Larger host prefill | 75 | 281.70 / 376.72 | 240.26 / 289.78 |

These are instrumented handoff diagnostics, not deployment-template throughput
rows. They measure `pread` entry, not physical SSD dispatch. Decode duration
improved modestly in these requests; the handoff reduction is not an equivalent
end-to-end speedup. The analyzer's `--reuse-graph-routes` mode attributes graph
miss reads without requiring a second service router.

## Keeping a reader ready

One representative early-demand decode observation signaled a reader about
18 us after selections, but its condition-variable wait returned about 136 us
after selections, with `pread` starting about 157 us after selections. This
identified worker wake-up as another source of handoff delay.

`DGPP_EXPERT_READY_WORKER=1` keeps worker zero polling while the graph walk is
active, or during the ordinary host routing window. Other workers use the
condition variable. The hint is unwound on normal and exceptional exits;
there is no polling between requests. It reserves no additional weight or
staging capacity.

`DGPP_EXPERT_READY_WORKER=2` instead uses `LDXR` plus timed ARM `WFET` on Linux
AArch64 with `HWCAP2_WFXT`; unsupported machines fall back to polling. A CPU
publisher advances a dedicated notification word after queue submission,
window changes and shutdown. The reader arms its exclusive monitor, checks
for changes before parking, and rechecks queue/shutdown state after every
wake. Each wait is bounded to about 1 ms. This applies the
[GB10 timed-wait approach](https://gist.github.com/jetnet/94c8dcfd5381985efd5030d8a4dca36d)
to CPU-published queue work; it does not rely on GPU stores waking a CPU.

A second same-binary diagnostic kept early demand on and changed only the
ready-reader setting (binary SHA256
`45d306390e8cb0e38e5da31fcee7ab4b283a098aea0c890c5e9834a1c4b2018e`). Its responses, usage and per-expert/chunk read counts matched
across the same five requests. Median selection-to-read latency was:

| Execution shape | Sleeping reader rank 0 / 1, us | Polling reader rank 0 / 1, us |
|---|---:|---:|
| Four-row graph decode miss | 146.85 / 198.51 | 43.34 / 42.91 |
| Small host prefill tails | 316.28 / 398.74 | 42.20 / 39.86 |
| Larger host prefill | 248.58 / 296.53 | 127.15 / 136.93 |

The measured request decode durations improved by about 0.6–1.5%; prefill
changes were mixed and below 1%. Removing a handoff delay helps, but it is
only one component of complete serving time.

Both ready-reader modes keep one thread runnable during the active window:
Linux CPU accounting still reports it as running, and it occupies a scheduling
slot. WFET avoids the tight instruction polling and can reduce power; power
savings have not been measured here. The default remains the ordinary sleeping
reader unless a nonzero ready-worker mode is selected. Early demand must also
be enabled for the engine to arm the reader.

A final same-binary comparison used mode 1 versus mode 2 on both Sparks
(SHA256 `80fff51f4811f122d0ccd8586b8acc85d6d49b247fb30b101fa1085af12c7b6c`).
Both machines advertised timed-wait support. Median graph decode
selection-to-read latency was 43.31 / 44.59 us with polling and
44.48 / 44.93 us with WFET (501 observations per rank). Small prefill-tail
medians were 41.27 / 41.16 us versus 39.07 / 32.29 us; larger-prefill medians
were 137.34 / 130.03 us versus 127.41 / 124.16 us. Responses, usage and
per-expert/chunk read counts matched, again with 318,168 records per rank
and zero process swap. WFET therefore retained the fast handoff in this
capture. It did not establish a serving speed advantage over polling:
measured decode durations differed by about 1–1.5% in either direction,
while prefill varied more. No power measurement was taken.

The deployed setting is early demand enabled with
`DGPP_EXPERT_READY_WORKER=2`, using the existing port, authentication and
cache budget. Focused validation passed 30 cache tests, 10 read-queue tests,
the B12x supplied-route GPU test, and seven analyzer tests. All 10 read-queue
tests also passed on Spark with WFET selected. The first profiler startup
attempt for the final pair was refused by the existing memory-headroom guard
before requests ran; restarting with the same configuration succeeded.

## Remaining scope of the initial implementation

This change does not move routing above its attention/normalization inputs or
predict experts. It also does not add staging reservations, preallocated batch
tickets, or a batched queue-submission API. Existing staging bounds already
permit the bounded first wave; the measurements first pointed to worker wake-up
rather than proving an allocation or descriptor bottleneck. It does not remove
all service-stream completion fences, change expert arithmetic, or replace
the engine's other CPU/GPU waits with WFET.


The subsequent [reserved group scheduling](expert-group-scheduling.md) change
replaces the fixed two-expert head start with complete execution-group
reservations and batched queue submission. The measurements above describe
the earlier implementation, before that extension.
