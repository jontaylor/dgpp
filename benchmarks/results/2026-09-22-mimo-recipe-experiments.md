# MiMo recipe experiments

Baseline source: 7ca896f (previous three optimizations combined).
Recipe source: 13621bb3cc6fd30a94d53609320599d1f1134686.
Parent exclusively performs Spark hardware work; Astra Medium agents implement
in isolated worktrees. Current original dirty source is preserved.

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

Two follow-ups await service measurement. The dense prefill bridge uses 64 MiB
transient BF16 scratch only for eager prefill at least 512 rows. Real-weight
2048-row QKV improved from direct FP8 4.578 ms to bridge 2.045 ms; BF16 was
1.617 ms. Exact FP8 loading, compile-time default off, reduced same-source 64K
FP8 attention from 167.022 to 94.076 ms. All 256 E4M3 codes passed GPU equality
on both ranks under both builds. Neither result is yet an end-to-end speed claim.

DFlash real-weight CPU/GPU oracle passed (maximum absolute error 0.25,
RMS 0.0263843, minimum cosine 0.99991715). The target-only TP2 service control
passed all 11 requests, including C2 graph replay, four-slot prefix reuse,
cancellation and recovery. Actual DFlash startup exposed two integration bugs:
a stale TP reducer and the device picker's six-slot bound. Both are fixed;
the eight-slot picker regression passed eager and graph execution on both Sparks.
Repaired DFlash TP2 startup and service validation remain in progress.

## Snapshot budget

At TP2/context262144/forward_rows2048 (ring4096), native MTP3 snapshot:
BF16 base3032678400 + nativeMTP/hidden65019904 = 3097698304 bytes/slot.
FP8 base1516339200 + unchanged65019904 = 1581359104 bytes/slot.
Four FP8 slots require6325436416 bytes =5.891021728515625 GiB.
Both live ranks confirmed exactly four slots. DFlash configurations use their
own four-slot budgets because draft-state size differs.

## Remaining work

DFlash service correctness and timing; dense bridge and exact-FP8-load service
measurements; current-source native-MTP control; separate real-activation FP8
audit; combined timing/quality/harness/tool-cap/lifecycle validation; final
selection and healthy restoration. Preserve rejected and failed candidates.
All traffic is explicitly paused by the user. Parent holds both ranks exclusively.
No new source has been copied into the original dirty checkout. No PR is requested.
