# MiMo fused-probability prefill candidate

Opt in with `DGPP_MIMO_FUSED_PREFILL=1` in each serving process; unset or
set to `0` to retain the baseline. The cluster launcher forwards exported
`DGPP_*` variables to both ranks. The choice is cached on first prefill,
so switching requires a fresh process. Decode and mapped speculative rows
retain their existing paths. This candidate is disabled by default.

QK remains materialized in FP32. The existing max and denominator reductions
produce two FP32 values per query/head in the old probability region. Wide
PV tiles compute BF16 probabilities directly in shared memory. The score,
difference and probability rounding points and ordered WMMA PV chain remain
unchanged; no online softmax rescaling or reduction reassociation is added.
Scratch allocation remains six bytes per score for easy baseline switching.
Only eight bytes per query/head in its probability region are used by the
candidate. Tiny capacities (<4) and narrow-PV calls retain the baseline.

This removes the full probability write/read and a separate score read in
probability generation, while the PV tile reads FP32 scores instead of BF16
probabilities. It does not eliminate the score matrix. Extra exp/div work in
PV may outweigh memory savings; no performance improvement is claimed.

GPU acceptance (not run on this workstation):

```bash
./build-spark-cross/mimo_attn_test
compute-sanitizer --tool memcheck --error-exitcode 1 ./build-spark-cross/mimo_attn_test
compute-sanitizer --tool racecheck --error-exitcode 1 ./build-spark-cross/mimo_attn_test
DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT=8192 ./build-spark-cross/mimo_attn_test
DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT=65536 ./build-spark-cross/mimo_attn_test
```

Benchmark mode 3 is baseline wide PV; mode 4 is fused-probability wide PV.
The exact A/B test covers serial/parallel normalization, global 64K,
SWA128 with a non-power-of-two physical ring wrap, dominant/negligible sinks,
1/17/128 query rows, a tiny-capacity fallback, and kernel-only graph replay
with changed V values. Existing scalar-reference and mapped/padding tests
remain unchanged. This exact output gate supplements the existing numerical
gates; it does not replace real dense/MoE layer or whole-service acceptance.

Local build uses the Spark CUDA13/aarch64 cross toolchain with two jobs.
`scripts/spark-cross` encountered the host's missing `docker0` bridge;
the same image/mount/user/toolchain was invoked with `docker run --network none`.
The root filesystem then exhausted compiler temporary space. Build output and
compiler scratch were moved to `/mnt/benchmarks/dgpp-pipeline-builds-20260922/attention`
and `attention-tmp`, preserving existing objects and using bind mounts.
This is a workstation cross-build only, not GPU validation.

Real-weight A/B (replace `/path/to/snapshot`; repeat with layer 5 for MoE):

```bash
DGPP_MIMO_FUSED_PREFILL=0 DGPP_MIMO_LAYER_ATTN_DUMP=1 ./build-spark-cross/mimo_layer_check /path/to/snapshot 0 /tmp/fused-control.bf16 2048 4103
DGPP_MIMO_FUSED_PREFILL=1 DGPP_MIMO_LAYER_ATTN_DUMP=1 ./build-spark-cross/mimo_layer_check /path/to/snapshot 0 /tmp/fused-candidate.bf16 2048 4103
cmp /tmp/fused-control.bf16 /tmp/fused-candidate.bf16
cmp /tmp/fused-control.bf16.attention /tmp/fused-candidate.bf16.attention
```
