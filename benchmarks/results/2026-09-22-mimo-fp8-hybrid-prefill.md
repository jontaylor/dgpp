# MiMo FP8 resident weights with transient BF16 prefill

Second isolated candidate atop `7c0cf29`, prompted by parent-run TP2 projection
measurements: direct FP8 QKV saved decode time but lost substantially at 512
prefill rows. Parent-run hardware probes of `51b223f` passed exact GPU dequantization/output
checks on both TP2 ranks. At 128 rows the bridge took approximately 620–810 us
versus direct FP8 237–336 us; at 512 rows it took 738–954 us versus direct FP8
1047–1195 us. These projection measurements motivate the revised 512-row
minimum; whole-service benefit remains to be measured.

Enable **both** flags on both ranks:

```sh
DGPP_MIMO_FP8_DENSE=1
DGPP_MIMO_FP8_DENSE_PREFILL_BF16=1
```

The second flag defaults off and requires the first. It selects the bridge only
for explicit eager prefill (`end_key > 0`, no request-id array), at least 512
rows, and `capture=false`. Decode, smaller chunks and all graph capture retain
the original direct FP8 kernels. Existing BF16 projections retain their path.

Each bridged projection first dequantizes its resident FP8 codes/scales to a
single scratch matrix, then invokes the existing BF16 GEMM interface. It repeats
that dequantization for each projection/chunk. The 64x64 aligned dequant kernel
uses native paired E4M3 conversion and packed BF16 writes; the original 128x128
API retains its scalar path. No activation quantization is introduced.

There is no retained BF16 weight shadow. Serving shares the existing layer
scratch across all backbone and draft layers, so the additional buffer is
bounded by the largest single local projection, not the sum of their sizes.
For the current TP2 config, its bound is 64 MiB (4096 x 8192 x 2 bytes).
`MimoDecoderLayer::workspace_bytes` includes the extra buffer when chunk capacity
is at least 512; the existing model memory plan and shared allocation consume that
value. The bridge uses the model's existing GEMM workspace policy (currently
zero bytes), introducing no unaccounted cuBLAS workspace.

## Checks and hardware commands

The host weight suite passes all 12 tests, also under UBSan. It includes route exclusions for capture, decode, disabled
mode, missing prefill extent and row counts on 511/512/513 and other representative sizes. The extended
GPU probe checks exact full-matrix GPU dequantization against the existing real
checkpoint BF16 loader, exact hybrid output against the same BF16 GEMM call,
and times **dequant plus GEMM on every iteration**, including overwrite of the
scratch, alongside a matched eager zero-workspace BF16 baseline. It still checks the direct FP8 numerical and kernel-only graph paths.
Baseline/probe GEMMs now use zero workspace to match serving; do not directly
compare its timings with the earlier probe's 32 MiB-workspace measurements.

On each Spark, while the parent owns hardware testing:

```sh
./mimo_fp8_dense_check "$CHECKPOINT" 0 2  # rank 0
./mimo_fp8_dense_check "$CHECKPOINT" 1 2  # rank 1
```

The probe runs both candidates, independent of environment flags, and emits
`hybrid ... dequant_plus_gemm_us=... scratch_bytes=...` at rows 128, 256, 512 and 2048. The probe deliberately measures the
bridge below its service threshold to expose the crossover; `service_bridge`
reports whether the current serving policy would select it.
These are ten-repeat screening timings. Follow with matched service baseline,
direct-FP8, and hybrid requests; measure prefill and decode separately, include
an unchanged-vs-unchanged control, and retain opt-in status unless numerical,
whole-service and latency gates pass.
