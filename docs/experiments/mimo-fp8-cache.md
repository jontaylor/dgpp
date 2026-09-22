# Unit-scale MiMo FP8 cache candidate

Base: 7ca896f. Enable with `DGPP_MIMO_FP8_KV=1` on **both TP ranks** before startup; unset/0 retains BF16. Other values fail. The setting is latched once per process so allocation, planning, layers and graph replay cannot disagree.

Base-model global and SWA K/V allocations and all base prefix snapshot payloads use one-byte E4M3. Q, activations, score scratch and native MTP remain unchanged. There are no persistent BF16 cache mirrors; scalar/tensor-core paths expand only individual loads/shared tiles. Model key/value plane offsets are bytes. BF16 snapshot format is two bytes per value, FP8 one; snapshots are in-process and must never be transferred across modes. Memory-plan labels and startup logs identify the format. Native MTP cache/hidden snapshots remain BF16, so total process/snapshot bytes decrease by less than 50%.

K is quantized after BF16 RoPE. V first applies the existing .707 factor and BF16 rounding, then nearest-even E4M3 with finite saturation at +/-448. Quantization/dequantization scales are exactly 1, as the recipe's DiffKV patch requires; weight quantization scales are unrelated and remain honored. Checkpoints with explicit K/V scale tensor suffixes are rejected in this mode instead of silently ignored. This candidate does not implement calibrated/dynamic KV scales. Extreme values saturate; model quality is an experimental question.

## Hardware gates (parent-owned; no Spark access by this agent)

Cross-build from this checkout using `dgpp-spark-cross:cuda13`, targets `dgpp_serve_app mimo_fp8_test mimo_snapshot_test mimo_attn_test`.

On an idle Spark, run the native ARM binaries:

```
./mimo_fp8_test
./mimo_snapshot_test
./mimo_attn_test
compute-sanitizer --tool memcheck --error-exitcode 1 ./mimo_fp8_test
compute-sanitizer --tool memcheck --error-exitcode 1 ./mimo_snapshot_test
```

The FP8 probe checks independent CPU exhaustive-codebook nearest-even conversion (including ties, subnormal, signed zero, saturation), byte-exact append with a cache allocation actually half-sized, global and non-power-of-two SWA wrap, scalar/decode/fused-prefill against BF16 caches populated with the independently quantized values, graph append and prefill, and incremental snapshot byte counts/restore. It reports changed-value count, clipping count, RMS and maximum conversion error on synthetic inputs; these are NOT model-activation clipping statistics. The full snapshot lifecycle test executes both element widths: four-slot rolling reuse, rejected-tail invalidation, hop snapshots, reset/cancel, destination reassignment, and exact restored/live-tail checks.

Real checkpoint gate: compare unchanged BF16 control against FP8-only candidate with the same TP2/MTP3/C2/context/prefix slots and prompts. Record both ranks' planned/cache/resident memory, prefill/decode latency and snapshot copy bytes. Exercise 1K/8K/32K/128K/256K contexts, prefix hits and cold misses, multi-request slot reuse, stop/cancel and speculative rejection. Teacher-force identical continuations to compare logits/top-k/KL/loss where probes permit; run held-out long-context retrieval and reasoning checks. Record answer accuracy and score changes separately from token agreement and performance. Sample per-layer post-RoPE K / post-.707 V ranges to count values outside +/-448 and quantify E4M3 conversion errors. Synthetic clipping diagnostics and successful generation do not establish quality. Do not present long-context readiness before those gates pass.

Integration with online attention must change its cache argument types to `const void*` and replace K/V element reads with `mimo_cache_load(cache,index,shape.fp8_cache)` from `kernels/mimo_cache.cuh`; BF16 Q reads stay unchanged. Native draft cache shape must remain `fp8_cache=false` unless its allocation and snapshots are changed too.

## Fixed four-slot prefix budget

`python3 tools/mimo_cache_budget.py --config tests/data/mimo/config.json` reports per-rank budgets. For TP2/C2/context262144/forward_rows2048/native MTP3, ring capacity is 4096. Base BF16 snapshot is 3,032,678,400 bytes; FP8 is 1,516,339,200 bytes. Unchanged MTP and last-hidden state adds 65,019,904 bytes per snapshot. FP8 total is 1,581,359,104 bytes per slot, and exactly four slots require **6,325,436,416 bytes = 5.891021728515625 GiB**. Baseline four slots use 12,390,793,216 bytes = 11.539825439453125 GiB. Lower `prefix_cache_gib` with FP8 instead of retaining the old budget and silently admitting more slots. Startup logs report actual live-cache and session-snapshot bytes; verify they match the computed values before benchmarking.

Local validation: native host MiMo config/reference/snapshot suite passed 10/10 with FP8 disabled and 10/10 enabled. These are host contract checks, not GPU numerical or model-quality evidence.

Workstation cross-build completed successfully for `dgpp_serve_app`, `mimo_fp8_test`, and `mimo_snapshot_test` using CUDA13/SM121a ARM64 image. GPU probes are compiled only; no target-hardware execution or model-quality claim is made here. Default BF16 attention regression binary is also included in the final incremental build.

Combined-path regression: `mimo_fp8_test` additionally compares FP8 and explicitly CPU-dequantized BF16 caches using the same bounded or online algorithm. It covers shared prefill rows 1/17/33, mapped mixed-length online decode and bounded-mode scalar decode, global and wrapped non-power-of-two SWA, sinks on/off, invalid/padded rows, and three graph replays with changed K/V/Q/sinks/positions/mappings. All graph nodes must be kernels and outputs must be bitwise equal, finite, and zero for invalid/padded decode rows. This checks format integration without treating online-versus-baseline rounding changes as equivalent.
