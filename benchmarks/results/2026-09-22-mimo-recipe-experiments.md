# MiMo recipe experiments

Baseline source: 7ca896f (previous three optimizations combined).
Recipe source: 13621bb3cc6fd30a94d53609320599d1f1134686.
Parent exclusively performs Spark hardware work; Astra Medium agents implement
in isolated worktrees. Full local evidence is retained under
`/mnt/benchmarks/dgpp-mimo-recipe-20260922`: per-deployment raw requests, SSE,
metrics, both-rank hashes/configurations/logs, frozen binaries, GPU/sanitizer
logs, failed attempts, and old/same-source summaries. Current original dirty
source is preserved; `original-before-integration` archives its 86 dirty files
and records the exact pre-application checks.

## Candidates

1. Native DFlash seven-token speculation. Compare identical greedy target
outputs, draft position attempt/accept counts, C1/C2 decode, native-MTP baseline,
rollback, graph/eager and cached/recomputed requests. No performance claim from
the recipe is treated as a DGPP result.
2. FP8 base-model KV and snapshots. Verify actual allocations and byte
traffic, no hidden BF16 shadow, unit-scale quantization/clipping, ring wrap,
restore and rollback. Numeric oracle plus long-context exact-answer retrieval
and tool/function tasks assess quality separately from successful generation.
3. FP8 resident dense projections. Validate checkpoint scale/QKV chunk mapping,
real-weight error and argmax, graph node safety, memory, layer timing and
end-to-end prefill/decode. Keep BF16 control.
4. Bounded-workspace fused attention. Measure workspace allocation and speed,
check sinks, DiffKV dimensions, causal/window boundaries, ring wrap and
numerical effects of normalization/reassociation. Coordinate FP8 cache support.
5. Optional per-response tool-call cap. Default unchanged; complete calls only,
stream/nonstream parity and correct finish reason. Test synthetic service cases
and actual harness before treating as usable. It is not a throughput win by itself.

## Hardware protocol

Maintenance lock held; counters confirmed no external requests since the previous
handoff. Hardware work proceeds under the user's experiment authorization.
Keep baseline rollback available. Matched input/settings per comparison; record
exact source, binary hashes on both ranks, environments, configuration, cold vs
cached prompts, output lengths, request concurrency and timing distributions.
Use fixed independent prompts for code, JSON, prose, arithmetic and tool calls.
Use 8K/32K/67K cold prefills, cached long decode and repeated short C1/C2 probes.
Quality probes include planted random keys at multiple context depths (up to
128K initially), plus numerical GPU/reference probes. Quantized candidates need
quality evidence; do not demand byte-identical text when arithmetic changes,
but report divergence and reference error instead of hiding it.

Each performance phase bounded to roughly 30 minutes; no overlapping profiling
or unrelated workloads. Run individual candidates first, then compatible
combinations; preserve neutral/regressed results. Keep source and deployment
opt-ins explicit. Never report kernel-only gain as whole-service gain.

## Completed hardware evidence (2026-09-23)

Baseline source `7ca896f`: all 19 timing requests, four category acceptance
probes and exact three-key retrieval at 30,921, 67,437 and 125,491 tokens passed.

Original FP8 dense: all 19 timing requests, four acceptance probes and 31K
retrieval passed. C1 decode time changed -11.6% code, -8.7% JSON, -4.0% prose,
-8.9% math; C2 code/JSON -11.4%/-13.8%. Cold TTFT regressed 7.7% at 7K,
4.8% at 31K and 2.3% at 67K. Saved approximately 1.79 GiB per rank.
Some generated text differs; this panel does not establish quality equivalence.

Original FP8 cache: all 19 timing requests and four acceptance probes passed.
Live backbone KV fell from 6.39 to 3.19 GiB; four snapshots fell from 11.54 to
5.89 GiB, saving 8.842 GiB per rank. Short decode was about 1–3% slower;
67K cold TTFT was 414.284 versus baseline 230.398 seconds. Exact three-key
retrieval passed at 31K and 67K. The 125K probe was explicitly skipped because
its predicted 1394.7-second request exceeded the bounded experiment budget.
The actual orchestration harness passed all three replay cases, including the
original failing request, with complete arguments. Generated tools were not executed.

Attention alternatives are capacity tradeoffs, not demonstrated speed wins.
At 64K the previous fused kernel took about 82 ms, bounded about 389 ms,
online about 182 ms, and split-key about 143 ms. Compact materialized tiles
16/32/64/128 used 0.75/1.5/3/6 GiB at 256K; BF16 kernel times were
169.65/120.77/94.44/83.89 ms. Keep tile 128 for performance. Both-rank parity,
64 mapped rows, graph replay and sanitizer gates passed; memory/race tools
reported no errors/hazards. Do not equate these kernel timings with service rates.

The dense prefill bridge uses 64 MiB
transient BF16 scratch only for eager prefill at least 512 rows. Real-weight
2048-row QKV improved from direct FP8 4.578 ms to bridge 2.045 ms; BF16 was
1.617 ms. Exact FP8 loading, compile-time default off, reduced same-source 64K
FP8 attention from 167.022 to 94.076 ms. All 256 E4M3 codes passed GPU equality
on both ranks under both builds. The same-source control and dense-bridge service panels are now complete (19
requests plus four acceptance probes each). The control matched all 19 old
baseline outputs; median decode-time ratio was 1.00094. Its 67K TTFT was
232.248 seconds versus old baseline 230.398; cached decode remained 4.56 s.
Against that control, dense bridge reduced C1 decode time by 11.69% code,
9.21% JSON, 4.25% prose and 9.81% math; C2 code/JSON by 13.55%/14.01%.
Cold TTFT changed -0.26% at 7K, +0.29% at 31K, +0.30% at 67K; cached
67K decode fell 6.07%. Total planned allocation is 106.22 GiB, approximately
1.73 GiB below control per rank, including bridge scratch. Some output text
differs. Fast-FP8-cache service measurement also completed all 19 requests and four
acceptance probes. All 19 outputs matched the original FP8-cache run exactly.
At 67K, TTFT fell from 414.284 to 258.160 seconds (-37.69%), but remains
11.16% above same-source BF16 cache. At 31K it averaged 64.405 seconds
versus BF16 56.672. The memory saving remains 8.842 GiB per rank; this is
an optional capacity tradeoff, not a prefill speed win over BF16.

DFlash real-weight CPU/GPU oracle passed (maximum absolute error 0.25,
RMS 0.0263843, minimum cosine 0.99991715). The target-only TP2 service control
passed all 11 requests, including C2 graph replay, four-slot prefix reuse,
cancellation and recovery. Actual DFlash startup exposed two integration bugs:
a stale TP reducer and the device picker's six-slot bound. Both are fixed;
the eight-slot picker regression passed eager and graph execution on both Sparks.
Repaired DFlash TP2 startup and all 11 lifecycle checks passed. All 19 timing
requests completed with no failures: C1 decode code +9.9%, JSON -10.4%, prose
+96.5%, math +19.5%; C2 code +2.4%, JSON -25.8%; 67K TTFT +0.8%, cached
67K decode +24.4%. Four serial acceptance probes yielded 4.204/6.486/0.969/
4.313 accepted draft tokens per round for code/JSON/prose/math. Exact 31K
retrieval and all three actual-harness replay cases passed.

Initial DFlash serial, prefix and recovery outputs matched target-only, while
C2 differed. Audit found the target BF16 GEMV cutoff at eight rows: DFlash C2
verifies 16 rows and selects cuBLASLt. Opt-in target graph GEMV (`41fd8a7`)
restored exact agreement across all 20 comparisons in the 11-request panel,
including C2. This establishes agreement on this panel, not general numerical
identity. Corrected C2 timing over two pairs was code 12.009 s / JSON 9.506 s,
versus baseline approximately 10.9 / 10.0 s. DFlash therefore remains an
optional workload-dependent experiment, not a general default speed win.
Both ranks shut down cleanly with identical operation streams
`40ca2e25f0ccf678a363188472976009` after the full DFlash panel.

## Combined service measurement

Source `41fd8a7`, the same frozen fast-load binary as the current native,
dense-bridge and cache-fast controls. DFlash7 plus target graph GEMV, FP8 dense
bridge and FP8 KV completed all 19 timing requests and four acceptance probes
without failures. Relative to same-source native MTP3/BF16:

| Workload | Decode time change |
| --- | ---: |
| C1 code | -2.15% |
| C1 JSON | -16.40% |
| C1 prose | +77.31% |
| C1 arithmetic | +6.10% |
| C2 code | -5.24% |
| C2 JSON | -13.72% |
| Cached 67K | +21.44% |

Cold 31K/67K TTFT changed +13.95%/+11.55%; 67K took 259.075 seconds.
Planned memory is 98.30 GiB per rank, approximately 9.65 GiB below control,
with exactly four prefix snapshots. This combination is not the general-purpose
performance default. Exact three-key retrieval passed at 31K and 67K;
125K also passed all three exact values (125,491 tokens, 967.852 s TTFT,
6.088 s decode). All three actual-harness cases passed, including the original
empty-argument failure. Default eight-call generation and caps one/two passed
streaming/non-streaming parity with complete arguments. All 11 lifecycle
requests passed, including C2 graph replay, prefix seed/warm/recompute,
cancellation and recovery (35.67 s; 4,078 warm cached tokens; one cancellation).
Generated tools were not executed. These are fixed checks, not broad agent
quality or full 256K simultaneous-load validation.

Small fixed panels are not broad quality evidence. Short first-request TTFT
also includes initialization effects; the main prefill comparison uses the
later fixed 7K/31K/67K requests. Raw outputs and counters preserve differences.

## Real-activation FP8 audit

Separate from timing, source `2b966f9` ran native MTP3 with dense bridge, FP8
KV and audit counters. A 7K exact retrieval and all three actual-harness cases
passed, followed by clean shutdown with identical two-rank operation streams.
Across 192 layer/type/rank records: 1,213,601,280 executed conversions, zero
clipped and zero non-finite values. Aggregate relative L2 conversion error was
2.6467%. The largest layer/type error was layer-zero V: 10.9443% on rank zero
and 9.5913% on rank one. Small-magnitude V values make unit-scale quantization
relatively coarse; the aggregate alone would hide that. These counts include
warm capture and replayed/rejected writes, not unique prompt tokens. No broad
quality-equivalence claim follows from retrieval passing. BF16 KV remains the
selected general-purpose configuration.

The same build verified the signed device-free graph-instantiation log on real
hardware, replacing the previous unsigned-underflow allocation report.

## Snapshot budget

At TP2/context262144/forward_rows2048 (ring4096), native MTP3 snapshot:
BF16 base3032678400 + nativeMTP/hidden65019904 = 3097698304 bytes/slot.
FP8 base1516339200 + unchanged65019904 = 1581359104 bytes/slot.
Four FP8 slots require6325436416 bytes =5.891021728515625 GiB.
Both live ranks confirmed exactly four slots. DFlash configurations use their
own four-slot budgets because draft-state size differs.

## Final selection and validation

The selected service uses native MTP3 with `DGPP_MIMO_FUSED_PREFILL=1`,
`DGPP_MIMO_FP8_DENSE=1` and `DGPP_MIMO_FP8_DENSE_PREFILL_BF16=1`. KV remains
BF16. Context is 262,144 tokens per request, concurrency two, four prefix
snapshots (11.539825439453125 GiB budget), API port 30001. DFlash, FP8 KV and
compact/online attention remain opt-in experiments; they are not enabled in
the final general-purpose configuration.

Both ranks run the final source `2b966f9` binary, compiled with the exact-FP8-load
option on (unused while BF16 KV is selected), SHA256
`badebd501e7a8252621bd31631493f06001236298ba376bb869d471d32ebdd50`.
The difference from measured source `41fd8a7` is the signed memory-log fix.
Later commits change only validation tooling and documentation. Frozen binaries
and per-binary manifests preserve this distinction.

Final deployment passed 31K exact retrieval (56.816 s TTFT), all three actual
harness cases, five default/capped streaming/non-streaming tool cases and all
11 native-MTP lifecycle requests (29.54 s; 4,079 warm cached tokens; one
cancellation). A separate mixed-load check started a cold 7K retrieval after
another request began decoding: all three retrieved values were exact, the
other request completed 1,024 tokens, and 20 two-slot graph replays occurred.
That smoke test establishes this concurrent journey's correctness/liveness;
it is not a matched mixed-load performance comparison.

The original dirty checkout is integrated by a baseline-checked patch after
archiving its existing changes. No pull request or publication is part of this
experiment. Raw evidence, including rejected/failed candidates and explicitly
skipped earlier 125K FP8 probes, remains in the experiment directory.
