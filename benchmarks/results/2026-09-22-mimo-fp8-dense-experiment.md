# MiMo resident FP8 dense candidate

Opt in with `DGPP_MIMO_FP8_DENSE=1` on **both** ranks. Unset or `0` preserves
BF16 execution. Other values fail at planning/loading. This is an experimental
candidate, not a measured performance improvement.

Checkpoint-FP8 QKV and dense MLP gate/up/down (including draft layers) retain
E4M3 codes and FP32 scales on device. Output projections, EH projections,
routers, norms, embedding/head, and MXFP4 experts retain their existing formats.
There is no runtime full-matrix BF16 expansion or second BF16 resident copy.
Existing scale-aware GEMV handles <= DGPP_DENSE_GEMV_ROWS (default 4) rows;
existing streaming BF16 tensor-core MMA handles wider projections, dequantizing
weight chunks in registers. No DeepGEMM or activation FP8 quantization is used.

All compressed matrices use an exact 64x64 scale grid. QKV rows are permuted
from source TP4 [Q,K,V] chunks to serving-rank [Q,K,V]. Source scale row numbering
restarts at each TP4 chunk, including the 3392-row partial block. Duplicating
source scales into 64-row blocks preserves these restarts and all TP1/2/4 head
boundaries. Column slices preserve source scale offsets, too. This is a scale
layout change only, with no weight requantization. The existing BF16 rounding
of `e4m3(code) * scale` is preserved. Reduction orders differ between scalar,
streaming MMA and cuBLASLt, so tolerance equality, not transcript identity,
is expected near ties.

## Validation

Local host build with GCC C++20, `-Wall -Wextra -Werror`: 11 MiMo weight tests
pass, also under UBSan. Two new tests exhaustively compare compressed reconstruction to BF16
loading over TP1/2/4 QKV layouts and 64-aligned ordinary slices that cross both
128-wide scale axes. Existing tests include independent real checkpoint samples.

Cross-compile targets: `dgpp_serve_app`, `mimo_layer_check`,
`mimo_fp8_dense_check`. CUDA 13 / sm121a ARM64 build is workstation evidence;
GPU correctness and performance require the following hardware gate.

On each rank's Spark (rank 0 then rank 1, world 2), without a serving process
competing for GPU memory:

```sh
./mimo_fp8_dense_check "$CHECKPOINT" 0 2  # rank-0 host
./mimo_fp8_dense_check "$CHECKPOINT" 1 2  # rank-1 host
```

The probe loads real global/sliding QKV plus dense-layer gate/up/down tensors,
checks every reconstructed BF16 weight, compares GPU projections with BF16
baseline for rows 1,4,8,32,128,512, rejects nonfinite results, checks relative RMS
<= 0.01 and max absolute error <= 0.025 * max reference magnitude + 0.001,
requires kernel-only compressed projection graphs, and checks bitwise graph
replay. It emits per-site errors and same-stream CUDA graph timings for both
forms. Timings use 10 repetitions and are a screening measurement, not a service
benchmark. The numerical thresholds are provisional and must be reported with
actual errors rather than described as independently validated tolerances.

After the probe passes on both ranks, compare unchanged BF16 / FP8 candidates
with matched full-service prompts, decode lengths, MTP settings and concurrency.
Measure prefill and decode separately; wider-row streaming MMA can regress
prefill versus optimized BF16 cuBLASLt. Include an unchanged-vs-unchanged control.
Keep this flag opt-in unless the whole service gate and latency comparison pass.
