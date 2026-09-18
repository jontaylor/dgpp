# Cost of sparse decode buckets on two Sparks

The pre-experiment serving counters showed 1,248,800 verification rows,
332,436 padded rows (26.62%), and 56,086 replays. This is 36.28% extra
rows relative to useful rows, not a measured time penalty. Mean bucket
capacity was 5.57 requests; mean active requests per step was 4.08.

## Controlled live probe

The service was idle. No restart or production code change was needed.
Eight identical streaming requests were admitted sequentially, then selected
streams were disconnected to leave either low or high occupied slot IDs.
Each prompt had 8481 tokens, temperature zero, max output 4096. Measurements
started after cancellations settled and a further one-second warmup.
Each trial sampled metrics for about 15 seconds. Order was compact, sparse,
sparse, compact for each active count. The probe rejected trials with extra
requests or changed occupancy. All eight trials passed these checks.

| Active requests | Selected slots | Padding | Mean ms/step | Decode output tokens/s |
| --- | --- | --- | --- | --- |
| 2 | 2 | 0% | 54.95 | 93.26 |
| 2 | 8 | 75% | 89.49 | 59.83 |
| 5 | 6 | 16.67% | 96.64 | 137.44 |
| 5 | 8 | 37.5% | 105.56 | 126.89 |

Two-request sparse steps cost 62.87% more time. Five-request sparse steps
cost 9.22% more; moving those five from eight slots to six would recover
about 8.44% of step time, or about 9.22% step throughput at fixed acceptance.
Observed output throughput improved 8.31% in the five-request compact case.
The two repeats differed by under 0.5 ms/step in every configuration.

The step timer includes the scheduler's decode call, not exclusively GPU
kernel time. Prompt and sampling settings matched, but generated prefixes,
context growth, route choices and MTP acceptance were not forced identical.
Mean tokens/step were 5.12 vs 5.35 for two requests and 13.28 vs 13.39 for
five. Results therefore establish the cost of these sparse serving cases,
not a pure isolated per-row cost or an overall workload speedup. Repeated
identical prompts have less expert diversity than many real agent workloads.
No power measurements were taken.

## Code explanation

- Qwen run_rows passes full T to dense projections, MoE and TP boundary folds
  (`src/models/qwen/forward.cpp`).
- QwenMoeLayer::enqueue_decode forwards all rows into the routed expert
  chain and shared tail, without a position/inactivity mask. The router
  selects experts for those rows (`src/models/qwen/moe_layer.cpp`,
  `src/models/glm/moe_layer.cpp`). Padding therefore does expert work too.
- Stateful kernels can skip negative-position rows. For example batched
  causal convolution zeros inactive spans and returns (`src/kernels/kda.cu`).
- Wider MTP graphs include draft work too; the public padding counter only
  describes verification rows. It is not a fraction of elapsed GPU time.

Compacting live slots would address sparse occupancy. Adding a five-slot
bucket would address rounding from five to six, but would not fix requests
stranded in high slots. Masking inactive MoE rows is another candidate and
would require graph-safe routing plus correctness/performance validation.
The existing counters cannot tell how much of the historical 26.62% padding
came from each active-count/bucket combination, so an overall recoverable
throughput percentage cannot be calculated from this experiment.

The probe cancelled only its own 64 streams. Cancellation counters and HA
metrics include this diagnostic traffic; exclude it from workload comparisons.
Afterwards the service was healthy, idle, with zero failed requests. Raw
samples and reproducer scripts are under artifacts/padding-cost-20260918.
