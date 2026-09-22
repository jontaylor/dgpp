# MiMo cache-only MTP history updates

The native three-head implementation previously ran full attention and dense
MLP forwards for every history update, even when the hidden outputs were
unused. Native heads consume backbone states independently, so those outputs
are not needed by the next head.

`MimoDecoderLayer::enqueue` now accepts an opt-in cache-only mode. It executes
the same input RMSNorm, fused QKV projection and K/V append (including RoPE,
value scaling, padding and status handling), then returns before attention,
output projection, MLP and tensor-parallel reductions. Its residual input is
unchanged. Normal callers retain the complete forward path by default.

MiMo uses this mode for all draft heads during prefill and for unselected
heads during verified-token catch-up and proposal generation. Only the
selected head computes a hidden output and final normalization. A normal
three-proposal cycle now has three full head forwards plus three cache-only
updates, replacing six full head forwards and removing six MTP reductions.
Target verification, draft selection, cache formats and memory budgets are
unchanged.

## Validation evidence

The ARM64 serving binary and real-weight layer probe cross-build successfully.
Python compilation and `git diff --check` also pass.
`tools/mimo_cache_only_check.py` compares all three real checkpoint heads in
mapped decode and chunked prefill. It checks exact K/V bytes, identical final
hidden predictions after ring wrap, and unchanged residual inputs during
cache-only execution. Live validation compares fixed serial/concurrent greedy
prompts, a multi-chunk cold/cached prompt and tool argument handling.

All six real-weight comparisons passed on Spark-1: heads 0/1/2, mapped decode
with 145 rows (ring 128), and chunked prefill with 273 rows (ring 256).
Every final K/V buffer and final hidden row matched bit-for-bit, and all
cache-only residual checks passed. A focused Compute Sanitizer memcheck of
head 2 with 145 mapped decode rows reported **0 errors**.

Raw evidence: `/tmp/dgpp-mimo-cache-only-20260922/` on the workstation and
Spark-1. The source checkout is dirty on base
`3ee00e2641dd9413cc61e1d2b95f1828f3fdc89e`; earlier MiMo changes are included.

The fused QKV projection still computes Q. This change removes the unused
work after K/V append; it does not introduce a separate K/V-only projection.
Full-model equivalence against Hugging Face remains outside this validation.


## Live performance comparison

Two baseline and two candidate repetitions, each with two serial then two
concurrent 256-token greedy streamed requests. Each phase added exactly four
requests, with no overlapping harness traffic. Rates below are means of two
repetitions, estimated as 255 tokens divided by first-text-to-completion time;
concurrent rates are per request. These are short fixed probes, not a broad
workload or long-context throughput claim.

| Probe | Baseline tok/s | Cache-only tok/s | Change |
| --- | ---: | ---: | ---: |
| Interval merging, serial | 37.24 | 39.15 | +5.12% |
| Hash tables, serial | 32.61 | 34.32 | +5.24% |
| Interval merging, concurrent | 21.50 | 22.54 | +4.84% |
| Hash tables, concurrent | 20.16 | 20.70 | +2.68% |

Every corresponding generated text matched byte-for-byte across all four
runs. Every phase had exactly 320 draft rounds and position acceptance counts
`[276, 226, 208]`: 86.25%, 70.625%, 65%. Baseline repeat variation was below
0.5%; candidate variation reached about 2.9% for the second concurrent probe,
so its small gain is less precisely established.

The 7,215-token multi-chunk prefix test passed: cold cached tokens 0, warm
cached tokens 7,212, with identical cold/warm messages and identical messages
to the baseline. Single cold request latency was 9.434s before / 9.097s after;
warm 1.894s / 1.809s. These single latency samples are correctness evidence,
not a reliable prefill performance estimate.

## Deployment

Both Sparks run candidate namespace `c7f668cc11ce55a3`, config digest
`1d44c70e19429d88`, with matching binary SHA256:
`740f574053fea73e11ad21aae2b3e4adc697a9ee7190357445d8276f92d87dff`.
Configuration remains MTP3, three native heads, graph decode, 262,144-token
capacity, concurrency 2, and four prefix snapshots. Both ranks captured scalar
and two-slot graphs successfully; graph variants contain 106 collectives.
API: `http://192.168.0.171:30001/v1`.

Candidate on Spark-1:
`/home/jon/dgpp/experimental/mimo-cache-only-mtp3-20260922/`.
Preserved rollback binary/config:
`/home/jon/dgpp/experimental/mimo-native-anchor-mtp3-20260922/`, namespace
`40063d2bb897e375`, binary SHA256
`3b5dd08be3b05af701f8163e084451efa461ed486609958544ef208002e5f5da`.

The recorded first harness payload was replayed unchanged through
`ChatCompletionsClient` / `RecordingTransport`. The bounded two-turn run
produced four valid calls, all successfully executed with read-only repository
tools. Separate shell, typed file-read and two-required-field file-write
argument checks passed (generated shell/write operations were not executed).

Final health was `ok`; all 15 validation requests completed with zero failed,
shed or cancelled requests, no engine failure, and zero active/queued requests.
Both rank logs contained no ERROR, FATAL or mismatch lines. Prefix metrics
confirmed four slots at 3,097,698,304 bytes each. Maintenance lock released.
