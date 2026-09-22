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

Status: all five implementations are merged and cross-built in the isolated
integration branch. Combined host suites pass (47 service, 52 scheduler).
Fresh 19-request baseline performance suite passed with no external overlap.
Baseline retrieval passes at 30,921 and 67,437 tokens; 125,491-token retrieval
is currently running. New candidate GPU correctness/performance/quality gates
remain outstanding. No new candidate performance win is claimed.

## FP8 snapshot budget calculation (hardware verification pending)

At TP2/context262144/forward_rows2048 (ring4096), native MTP3 snapshot:
BF16 base3032678400 + nativeMTP/hidden65019904 = 3097698304 bytes/slot.
FP8 base1516339200 + unchanged65019904 = 1581359104 bytes/slot.
Four FP8 slots require6325436416 bytes =5.891021728515625GiB.
Use this reduced budget for matched four-slot tests; leaving the old11.5398GiB
budget would silently increase retention instead of showing memory savings.
