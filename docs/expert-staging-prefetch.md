# Staging-only prefetch replay

This experiment isolates speculative loading from cache replacement. Persistent
residency, admissions and evictions follow the captured deployed policy exactly.
Speculation occupies existing staging buffers only; consuming a staged expert
never admits it to the persistent cache. Every replay arm asserts identical
resident contents at every recorded layer observation.

## Capture and replay

Enable `DGPP_EXPERT_DEMAND_TRACE`, `DGPP_EXPERT_READ_TRACE` and the optional
`DGPP_EXPERT_RESIDENCY_TRACE` together. The latter requires demand tracing and
writes `PREFIX.rRANK.pPID.cBINDING.bin`. There is no residency scan or trace
allocation when it is disabled. Collect device samples with
`scripts/expert_io_sample.py` on each rank as described in
[the cache replay guide](expert-cache-whatif.md).

Run locally with NumPy, separately for each rank:

```bash
python3 scripts/expert_staging_whatif.py \
  --demand DEMAND.bin --reads READS.bin --residency RESIDENCY.bin \
  --disk-samples DISK.jsonl --windows WINDOWS.json --output REPORT.json
```

Optional `--decisions DECISIONS.jsonl` records decisions. Outputs include input
and source hashes, configuration, phase/window/layer counters, staging peaks,
and `(observation sequence, expert index, phase)` identities for modeled ready
hits. Before intersecting those identities across ranks, check that their
expert metadata and ordered routes match. Do not add two ranks' hits together:
both tensor-parallel shards must be ready for complete expert readiness.

The residency stream uses little-endian 64-bit words. Its header is eight ASCII
bytes `DGRSIDE1`, version 1, expert count, cache slot capacity, then the initial
resident chunk count for every expert in demand-metadata order. Each observation
contains its zero-based sequence, Unix-nanosecond timestamp, delta count, then
pairs of expert index and new resident chunk count. Sequence and timestamp must
match the demand stream exactly. The reader rejects partial residency, invalid
capacity, incomplete streams, and disagreement between recorded residency and
observed read misses.

## Policy and accounting

Four declared probes use the same causal popularity predictor: prefetch off;
one speculative expert with I/O price 0.1 and staging rent 0.001; and one/four
experts with both prices zero. The predictor observes only past/current routes,
uses layer/phase/shape histories, and ranks nonresident future candidates by
probability times miss benefit minus loading price and staging holding cost.
It does not use persistent-memory rent, victim scores or repeated-use rewards.
These probes isolate prefetch behavior; their coefficients are not a fitted
optimum.

Defaults use a shared 6 GB/s byte pipe only during sampled quiet intervals,
1 ms file-to-readiness delay and 1 ms consumer hold. An expert occupies eighteen
record buffers in this capture. Of 256 buffers, 36 remain reserved for demand;
the four-expert arm peaks at 72 speculative buffers. Completed-but-unused reads
retain staging until consumption, cancellation or expiry. Consumption holds the
source through remaining transfer/readiness and the consumer delay. Cancellation
also preserves an active modeled readiness-copy hold. Partial late reads receive
byte credit but do not count as ready hits. All speculative bytes, including
wrong predictions, are charged.

The recorded host timeline stays fixed. Freed baseline read intervals are not
reclaimed, completion credits cannot bank surplus service, and speculative
service pauses instantly outside sampled quiet intervals. These are modeling
choices, not actual SSD command scheduling. Native-hit observations can occur
after GPU execution, and fixed readiness/consumer delays do not establish real
GPU deadlines. This replay predicts neither latency nor serving throughput.

## November 16 result

A fresh two-Spark capture used the existing eight-request scripted tool-serving
sequence: an initial 24,317-token prompt, prose/code/repeated-prose continuations,
then four short uncached recomputations. These are real model routes from scripted
traffic, not a representative sample of user traffic. Each rank recorded 30,660
layer observations. The prefetch-off replay reproduced deployed misses and read
bytes exactly; all arms preserved deployed residency at every observation.

Baseline per rank was 17,519 missing experts / 163.599 GiB during prefill and
7,223 / 67.451 GiB during decode/verification. The table reports intersections
where both shards were modeled ready, and additional logical read volume summed
across both machines. Extra bytes include useful, partial and unused speculation.

| Probe | Prefill both-ready | Extra prefill GiB | Decode both-ready | Extra decode GiB |
| --- | ---: | ---: | ---: | ---: |
| Off | 0 | 0 | 0 | 0 |
| Priced, one expert | 14 | 0.149 | 0 | 0.135 |
| Unpriced, one expert | 23 | 0.149 | 4 | 1.905 |
| Unpriced, four experts | 50 | 0.654 | 11 | 7.321 |

Four slots cover 0.29% of prefill misses and 0.15% of decode misses with both
shards ready. Decode read volume rises about 5.4%. One-shard readiness and partial
transfers may help some demands but cannot be translated into saved latency here.

This conclusion does not depend solely on the initial cold request. Across the
three warm continuations, the four-slot probe covers 28 of 1,889 prefill misses
(1.48%) and 11 of 6,562 decode misses (0.17%) on both ranks. The final fully warm
short requests need no baseline expert reads, yet this predictor still spends
bytes on unused forecasts.

Keep deployed prefetch disabled. This result rejects the tested predictor and
staging choices as a useful deployment candidate; it does not rule out a better
predictor or a controller using actual readiness deadlines. No prefetch controller
or replacement policy was deployed, and no serving speedup is claimed.

Capture source was `85253903` plus the optional residency tracing introduced with
this replay. Diagnostic binary SHA256:
`438e176cf9a8942f040dcc1a7bdb5926ac85eaa8f1b4bdd0dabbea2799598975`.
All 29 cache tests passed on Spark, and both ranks had zero process swap during
the capture. Production was restored once afterward to binary SHA256
`e163306747831667813d601306ad38a68827a75cea118706751c75cb3307b8d6`, with zero
process swap on both ranks. These are diagnostic replay results, not deployment
benchmark campaign rows.

The nine focused Python tests cover frozen admission, ready/partial/copying
states, redundant forecasts, staging lifetimes and trace pairing:

```bash
python3 tests/python/expert_staging_whatif_test.py
```
