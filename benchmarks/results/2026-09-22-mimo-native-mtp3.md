# MiMo MTP with three native blocks

Replaces recursive reuse of block zero with checkpoint blocks 0, 1 and 2 for
successive proposals. Checkpoint revision remains
`3b38d063180c3e4aed9691fdc735f3d10b266ee4`.

Each head has independent dense SWA weights and K/V history. Head d at
absolute draft row p consumes the known token at p+1 and the backbone's
post-final-norm hidden state at p-d. Heads do not consume preceding MTP
outputs. This follows section 2.3.2 of the
[MiMo-V2-Flash technical report](https://github.com/XiaomiMiMo/MiMo-V2-Flash/blob/main/paper.pdf):
"Each head receives the main-model hidden state and token embedding as input."
The V2.6 report refers to this backbone architecture.

Attention positions are translated by -d so each head's populated history
starts at zero; a uniform RoPE translation preserves relative positions.
Rows before the head's origin are padding. Every verified-token catch-up
updates all three caches; the second proposal also fills the final head's
preceding history row. A backbone hidden ring retains the dependencies
across chunks. Pre-draft and pre-chain snapshots include all three K/V
caches and the hidden ring, including prefix attach/rollback.

This currently loads and maintains all three blocks whenever MiMo MTP is
on, even at a smaller configured proposal depth. The old dedicated depth-one
binary remains the depth-one performance control.

Prefix snapshots grow from 3,043,172,352 to 3,097,698,304 bytes. Four snapshots
require 11.539825439453125 GiB. Context and concurrency remain 262144 and two.

Validation and deployment results follow.
Raw evidence: `/tmp/dgpp-mimo-native-20260922/`.

## Validation completed before deployment

- 21 MiMo host tests pass, including all 48 native-block tensors and TP placement.
- 10 CUDA test groups pass under compute-sanitizer, with zero errors. The new
  history fixture covers per-layer offsets, startup padding, reordered request
  slots, repeated overwrites, and ring wrap.
- Real blocks 1 and 2 pass the independent CPU/GPU layer oracle on three rows
  each: relative L2 error ranges 0.019%–0.578%, below the 1% gate. These
  isolated-layer checks include fusion/final norm but do not establish full
  model or chained-drafter reference equivalence.

## Handoff experiment

The first native candidate passed normalized block outputs between blocks.
All four greedy benchmark texts matched the recursive control, but acceptance
was 448/502, 70/502 and 0/502. The zero third-position acceptance makes this
candidate unsuitable. Its binary and logs remain under deployment
`b85e0ac71c647226`; it was stopped after the comparison.

The next candidate passes each block's pre-final-norm residual to the next
block, retaining final normalization for the shared vocabulary head. This
isolates cross-block representation from the cache/position implementation.

The residual-handoff variant also failed to recover acceptance. Source review
then identified the native-head contract in MiMo-V2-Flash section 2.3.2: all
heads consume backbone hidden states, unlike recursive block-zero drafting.
The corrected candidate stores backbone rows once and gathers p-d for head d.
The changed gather passes its CUDA memory check with zero errors. Earlier
handoff candidates are retained only as failed experiments, not reference
implementations.

## Corrected native-head deployment

Live on both Sparks at port 30001: deployment `40063d2bb897e375`, config digest
`1d44c70e19429d88`, binary SHA256
`3b5dd08be3b05af701f8163e084451efa461ed486609958544ef208002e5f5da`.
Both rank logs explicitly confirm loading blocks 0, 1 and 2 at 362,848,512
weight bytes per block per rank. Planned memory is 107.96 GiB/rank, plus 4 GiB
headroom. Rank-zero binary/config are in
`/home/jon/dgpp/experimental/mimo-native-anchor-mtp3-20260922/`.

The API confirms concurrency two, 262144 context, depth three and four
3,097,698,304-byte prefix snapshots. The original recursive release remains
available under `mimo-mtp3-20260922` for rollback.

## Matched greedy measurements

Fresh recursive baseline immediately before maintenance; the same two prompts,
256 output tokens, temperature zero, first serially and then concurrently.
No diagnostic GPU jobs overlapped these timings. Both the first native run
and the warmed repeat produce all four texts byte-for-byte identically to the
recursive control. Rates estimate 255 / time from first nonempty streamed
text to completion, including streaming/snapshot overhead; concurrency figures
are per request. This is a short-prompt comparison, not a 256K-context benchmark.

| Probe | Recursive tok/s | Native warmed tok/s | Ratio |
|---|---:|---:|---:|
| Interval merging, alone | 33.82 | 37.45 | 1.107x |
| Hash tables, alone | 31.69 | 32.82 | 1.036x |
| Interval merging, concurrent | 19.40 | 21.62 | 1.114x |
| Hash tables, concurrent | 18.86 | 20.27 | 1.075x |

Each native run records 320 verification rounds and position accepts
`[276, 226, 208]`: **86.25%, 70.625%, 65.0%** accepted / all rounds. The warmed
repeat has identical counts. This recovers the later-position acceptance;
it does not establish that depth three beats the historical dedicated
single-block/depth-one binary for every concurrency or workload.

A separate 7,200-token prompt crosses multiple 2,048-token prefill chunks and
wraps sliding windows. Its cold and cached responses match exactly; the
repeat reports 7,197 cached tokens. Three additional tool-argument checks
pass for shell, typed file reading and a two-required-field write call;
generated shell/write operations were not executed.

## Final product-path checks

The actual six-turn `ChatCompletionsClient -> RecordingTransport ->
UrllibModelTransport` harness completed **12/12 successful read-only tool
executions**, with required fields populated and the first request matching
the user's original recorded payload. The last turn spent substantial time
in reasoning before making its calls; this does not establish SWE-bench
repair quality or resolve the model's tendency to prolonged reasoning.

All 19 validation requests completed: no failures, cancellations, shedding,
or engine failure; 15 structured tool calls total including the three argument
probes. Both rank logs have no ERROR/FATAL entries. Health is OK and the
scheduler is idle. The installed HA adapter maps the three accepted-position
counters to the existing sensor identities without changes.

Final mixed-workload counters: 3,344 rounds, 10,032 draft tokens, 6,651 accepted
drafts; per-position accepts `[2741, 2162, 1748]`, giving **81.97%, 64.65%,
52.27%**. These include the benchmarks, long-prefix probe and tool harness,
and differ from the fixed short-prompt benchmark's 86.25%/70.625%/65.0%.

Follow-up: [cache-only history updates](2026-09-22-mimo-cache-only-mtp3.md)
remove unused MTP attention/MLP work while preserving matched outputs and
acceptance counts; includes two-Spark deployment and performance results.
