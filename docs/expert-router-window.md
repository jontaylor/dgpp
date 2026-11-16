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

The next focused implementation opportunity is to submit a bounded first wave
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
worker evidence, exclusions and the analyzer's source hash. The six focused
tests cover service pairing, selection/hash completion, cross-stream read
ownership, graph producer mapping and rejection of ambiguous worker counts.
