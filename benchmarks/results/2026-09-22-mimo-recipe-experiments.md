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

Await renewed exclusive Spark traffic confirmation before workload/restarts.
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

Status: implementations in progress, new hardware window awaiting confirmation.
