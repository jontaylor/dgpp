# Active requests versus decode bucket capacity

Live Qwen3.8-Flash-Next-NVFP4, TP2 on two DGX Sparks, FP8 dense weights,
BF16 KV, C16/MTP3, 512-token prefill chunks. No service restart or code
change was required for this experiment.

Rows are active requests (1 through 16); columns are bucket capacities
1, 2, 3, 4, 8, 12, 16 as requested. The production six-slot family was
not included in this requested matrix. A cell is aggregate output tokens/s,
computed as 1000 * delta(tokens_generated) / delta(step_ms). Equivalently,
1000 divided by aggregate decode milliseconds per output token. This is
not per-request tokens/s and excludes prefill and time outside the scheduler's
measured decode calls. Raw results also retain wall-clock tokens/s and ms/step.

Each supported cell uses one 12-second measurement after a one-second warmup.
Prompt: 8481 tokens, identical across requests, temperature zero, maximum
output 4096. To force capacity B with A active requests, sequentially admit
B streams, then retain slots 0 through A-2 and B-1, cancelling the others.
One active request always takes the scalar graph under the current policy,
so cells with A=1 and B>1 are unsupported. A>B is impossible. Forty cells
are measured, in deterministic shuffled order (seed 20260918).

Every sample checks selected bucket, active count and empty queue. Trials
reject external admissions, prefill during measurement, engine/request
failures and changing occupancy. Finally blocks close only the probe's own
streams. Counters, including cancellation sensors, include diagnostic traffic;
exclude the sweep from agent-workload comparisons.

This is an exploratory matrix with one short sample per cell, not a precision
benchmark. Identical prompts reduce expert diversity. Generated prefixes,
context growth, acceptance and routing can differ across shapes; temperature
zero does not guarantee identical numerics. Interpret small differences with
caution. The earlier padding-cost study used repeated pairs for selected cells.

Reproducer and raw samples: artifacts/batch-matrix-20260918/probe.py and
results.jsonl. The matrix CSV contains the same rounded decode throughput.

## Results

| Active requests \ Bucket slots | 1 | 2 | 3 | 4 | 8 | 12 | 16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 63.2 | — | — | — | — | — | — |
| 2 | — | 89.0 | 77.6 | 73.6 | 57.8 | 42.9 | 37.4 |
| 3 | — | — | 111.3 | 102.8 | 83.2 | 61.4 | 55.1 |
| 4 | — | — | — | 131.3 | 106.4 | 78.2 | 70.9 |
| 5 | — | — | — | — | 129.0 | 93.8 | 86.5 |
| 6 | — | — | — | — | 149.1 | 110.9 | 102.3 |
| 7 | — | — | — | — | 171.2 | 128.2 | 118.5 |
| 8 | — | — | — | — | 193.2 | 146.0 | 133.4 |
| 9 | — | — | — | — | — | 166.2 | 150.0 |
| 10 | — | — | — | — | — | 180.5 | 164.5 |
| 11 | — | — | — | — | — | 199.3 | 180.8 |
| 12 | — | — | — | — | — | 218.1 | 195.2 |
| 13 | — | — | — | — | — | — | 210.1 |
| 14 | — | — | — | — | — | — | 223.8 |
| 15 | — | — | — | — | — | — | 237.6 |
| 16 | — | — | — | — | — | — | 253.5 |

All forty supported cells passed validation. The service remained running;
all test streams were closed after the sweep. See measurements.csv for
unrounded throughput, wall throughput, step time and tokens per step.
