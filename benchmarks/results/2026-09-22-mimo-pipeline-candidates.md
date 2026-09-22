# MiMo pipeline candidate comparison, 2026-09-22

Three isolated Astra Medium agents implemented candidates without Spark access.
All target-hardware tests and deployments were performed by the parent agent.
The common source baseline is `c91a50b53a6df28f365bd491b8a017edf6810c58`, a
snapshot of the existing dirty MiMo checkout on `3ee00e2641dd9413cc61e1d2b95f1828f3fdc89e`.
It corresponds to the deployed cache-only MTP3 implementation, SHA256
`740f574053fea73e11ad21aae2b3e4adc697a9ee7190357445d8276f92d87dff`.

| Candidate | Feature commit |
| --- | --- |
| Final-row prefill head | `8e3d6f7a5689dd2339eab2cd524a19101e3d23da` |
| Incremental rolling snapshots | `0d3467533998ef2cf99db72b1cff78a47500172e` |
| Fused prefill probability generation | `4164527777f45280724f8f92a385b52fb4076972` |
| Combined implementation | `1313508` |

Configuration is held at TP2, MTP3, graph decode, concurrency 2, context 262144,
four prefix slots, and 256/2048 busy/idle prefill budgets. The attention candidate
uses `DGPP_MIMO_FUSED_PREFILL=1` on both ranks; its default remains the baseline.
It retains materialized FP32 scores and existing workspace size, eliminating
probability materialization without changing softmax arithmetic. It is not a
complete FlashAttention replacement.

## Protocol

Each deployment runs the same 15 uncontended greedy requests: two repetitions
of two serial and two concurrent short 256-output probes, cold 7219- and
30882-token prefills (64 outputs), then a cold 67402-token prompt and two cached
repeats (128 outputs each). Short and medium cases disable prefix caching to
isolate compute; the long case enables it to measure snapshots and restoration.
The cold long prompt is a single sample; medium cases and warm decode repeat.
Streaming usage supplies token counts. Decode interval is first text through
completion, so throughput estimates subtract one output token. These are fixed
probe results, not a repository-diverse quality or broad throughput evaluation.

Raw scripts, request/response records, logs, metrics and comparisons:
`/mnt/benchmarks/dgpp-mimo-pipeline-20260922/` on the workstation (durable copy
of the working `/tmp/dgpp-mimo-pipeline-20260922/` directory); GPU logs are included. Build artifacts are on
`/mnt/benchmarks/dgpp-pipeline-builds-20260922/`. Builds were relocated there after
the workstation root volume ran out of temporary build space; no unrelated
artifacts were deleted. The combined incremental build reused common objects
under a container source-path alias, with its committed combined source mounted
at that alias.

## Correctness checks completed

- Snapshot production-copy/arena probe: exact K/V, poisoned-tail restoration,
  four-slot lifecycle, ring wraps, rejected tails and rollback all pass; memcheck
  reports zero errors. Real-model continuation checks follow below.
- Head probe: TP2 checkpoint slices for both ranks, rows 1/8/17/256/2048.
  Hidden buffers and unused output rows remain unchanged; all-row and decode
  paths match exactly. Final-only GEMV differs from wide Lt accumulation by
  small FP32 amounts (probe exit 2 explicitly records this, not an exact pass).
  Maximum absolute difference was 2.47955e-5, maximum relative L2 3.47402e-6.
  Every local argmax agrees. Memcheck reports zero errors. This synthetic-hidden
  probe is not full-model Hugging Face equivalence.
- Attention: all 11 CUDA groups pass; new fused path matches baseline bitwise,
  including 64K, sinks, SWA wrap, short tails and kernel-only graph replay.
  Real dense layer 0 and MoE layer 5, 4103 rows/chunk2048, have identical final
  outputs and attention dumps. Focused full-size memcheck reports zero errors. The full-size race pass was
  stopped after about six minutes of instrumentation without a result; it is
  not counted as a pass. A bounded race pass (global 512, SWA 263, same
  1/17/128 rows, sink modes, serial/parallel normalizers and graph replay)
  reports zero hazards/errors/warnings. Default normal tests retain 64K.
- Combined serving and scheduler suites: 40 and 52 tests pass, respectively.

## Isolated attention timing

On idle Spark-2, five CUDA-event measurements after warmup (128 query rows):

| Context | Existing wide PV, ms | Fused probabilities, ms |
| --- | ---: | ---: |
| 8192 global | 7.998 | 7.796 |
| 65536 global | 89.363 | 81.777 |
| SWA128 | 0.327 | 0.322 |

Kernel timings are supporting evidence, not end-to-end speedup claims.

## Standalone head result

All 15 generated texts and draft counters matched baseline (808 rounds,
accepts 727/623/535). Mean cold 7.2K TTFT was 7.456s vs 7.509s; 30.9K 59.045s
vs 59.098s; 67.4K 246.760s vs 246.743s. Warm 67.4K generation was 4.969s vs 4.950s.
These measurements establish no meaningful end-to-end gain or regression.

The independent real-weight TP2-slice microbenchmark explains the limited
impact: the 2048-row vocabulary projection drops from 17.114ms to 2.680ms,
while 256 rows drop from 3.936ms to 2.704ms (five alternating measured pairs,
two warmups per shape). Saving roughly 14ms per 2048-token chunk is small
beside seconds of backbone processing. There is a local operation improvement,
not a demonstrated whole-model throughput win.

## Standalone snapshot result

All 15 generated texts and draft counters match baseline. Cached 67.4K
128-token generation averaged 4.696s versus 4.950s: **5.41% higher decode
rate**. Average measured snapshot time over the phase fell from 8.242ms
to 1.470ms (cumulative event totals differenced across the phase). Cold
30.9K prefill averaged 59.274s vs 59.098s; cold 67.4K was 247.277s vs
246.743s, providing no evidence of a prefill change.

Fresh real-model regression passes two cold concurrent requests, reverse
restoration, more than four distinct prefixes, recomputation after slot churn,
disconnect after actual generated text, drain, and subsequent reuse. Every
compared response matches. The initial stricter requirement that both reverse
replays must hit failed: the server explicitly logged eviction of one prompt
before its replay. Four snapshot slots do not guarantee both conversations'
retention. The revised diagnostic records misses, requires an immediate retry
hit, and retains exact-output gates; it does not change eviction policy or
numerical thresholds. The fresh rerun asserts both initial prompts are cold.

## Standalone attention result

All 15 generated texts and draft counters match baseline: 808 rounds, accepts
727/623/535. Cold 30.9K TTFT averages 56.195s vs 59.098s (**4.91% lower
latency**, 5.17% higher effective prefill rate). Cold 67.4K TTFT is 230.725s
vs 246.743s (**6.49% lower latency**, 6.94% higher effective prefill rate).
The two 30.9K samples are 56.101s and 56.289s. Cached 67.4K generation
averages 4.962s vs 4.950s: no decode improvement. All three independent
candidate shutdowns verified identical operation streams between the two ranks.

## Combined result

| Variant | 30.9K cold TTFT, s | 67.4K cold TTFT, s | Cached 67.4K generation, s |
| --- | ---: | ---: | ---: |
| Baseline | 59.098 | 246.743 | 4.950 |
| Head only | 59.045 | 246.760 | 4.969 |
| Snapshots only | 59.275 | 247.277 | 4.696 |
| Attention only | 56.195 | 230.725 | 4.962 |
| All three | **56.126** | **230.515** | **4.701** |

Combined 30.9K latency falls **5.03%** (5.30% higher effective prefill rate).
67.4K latency falls **6.58%** (7.04% higher effective prefill rate).
Cached 67.4K decode rate improves **5.30%**, from approximately 25.66 to
27.02 tokens/s over the measured 127-token generation interval. Snapshot time
averages **1.472ms** versus baseline 8.242ms. Short serial/concurrent decode
is effectively unchanged. 7.2K TTFT falls from 7.509s to 7.378s; this smaller
difference is not a strong isolated gain claim.

All 15 combined texts match baseline byte-for-byte. Every standalone candidate
and the combined build records exactly 808 draft rounds and accepts
727/623/535. These are cumulative accepted-position counts; the optimization
does not improve acceptance, and the speedup does not come from generating
less output. All benchmark phases report exactly 15 new requests, zero
failures, no external request overlap, and idle healthy completion.

There are two repetitions per short/medium/warm case and one cold 67.4K
sample per build, without a later baseline re-deployment or confidence
interval. These are measured gains on fixed probes, not a guarantee across
workloads. This run tests up to 67.4K context, not full 256K capacity.

## Combined regression and final deployment

The combined build passes fresh concurrent cold requests, reverse cache
restoration, slot churn beyond four prefixes, exact recomputation, disconnect
after actual generated text, drain, and subsequent request/cache reuse. One
reverse replay was evicted; its immediate retry hit and all outputs matched,
as with the standalone snapshot run. No retention-policy change is claimed.

The original failing harness request was replayed through the actual
`ChatCompletionsClient` and `RecordingTransport`: two turns produced four
valid calls (`list_directory`, `glob`, two `grep_search`) and four successful
real read-only executions. A concurrently running argument probe verifies
`run_shell_command.command`, typed `read_file` offset/limit, and both required
`write_file` fields. Generated shell and write operations were not executed.
This is tool-format regression evidence, not a complete SWE-bench repair run.

The combined build is left serving on `http://192.168.0.171:30001/v1`.
Both ranks have binary SHA256
`0dfee7f6697c0a7fccb67abdf9f6f30237db3afc1a31c15ce221f8ec8355774e`
and `DGPP_MIMO_FUSED_PREFILL=1`. Namespace: `9625d247af6c7000`.
Configuration digest remains `1d44c70e19429d88`: context 262144,
concurrency 2, MTP3, graph decode, four prefix snapshots. Final health checks
confirm idle scheduling, zero failed requests, and byte-identical rank operation
streams. Both rank logs are preserved with the evidence. Configuration values
remain unchanged; attention workspace allocation is not reduced.

Deployment directory on Spark-1:
`/home/jon/dgpp/experimental/mimo-pipeline-combined-20260922/`.
The existing site environment is loaded by filename only. For later restarts,
preserve `DGPP_MIMO_FUSED_PREFILL=1` in the launcher environment; the source
flag defaults off. The durable `deploy.py` helper records this explicitly.
Rollback binary/config remain under
`/home/jon/dgpp/experimental/mimo-cache-only-mtp3-20260922/`.

All three feature changes and their probes were applied to `/home/jon/dgpp`
after checking every affected file against the captured baseline. A whole-tree
hash check confirms all pre-existing snapshot files outside the candidate paths
remain unchanged. Isolated branches/worktrees are retained for review. No PR
or remote push was made.

Machine-readable timing samples: `2026-09-22-mimo-pipeline-summary.json`.
The durable evidence directory includes request/response streams, metric
snapshots, regression outputs, build/GPU/sanitizer logs, deployment logs,
binary hashes, and the exact feature patch. Workstation validation and Spark
validation are distinguished above; the ARM64 runtime was tested on both Sparks.

