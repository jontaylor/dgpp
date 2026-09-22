# MiMo FP8 resident weights with transient BF16 prefill

Second isolated candidate atop `7c0cf29`, prompted by parent-run TP2 projection
measurements: direct FP8 QKV saved decode time but lost substantially at 512
prefill rows. This candidate's performance remains unmeasured until its own
hardware probe and service comparison complete.

Enable **both** flags on both ranks:

```sh
DGPP_MIMO_FP8_DENSE=1
DGPP_MIMO_FP8_DENSE_PREFILL_BF16=1
```

The second flag defaults off and requires the first. It selects the bridge only
for explicit eager prefill (`end_key > 0`, no request-id array), more than 64
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
exceeds 64; the existing model memory plan and shared allocation consume that
value. The bridge uses the model's existing GEMM workspace policy (currently
zero bytes), introducing no unaccounted cuBLAS workspace.

## Checks and hardware commands

The host weight suite passes all 12 tests, also under UBSan. It includes route exclusions for capture, decode, disabled
mode, missing prefill extent and row counts on either side of 64. The extended
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
`hybrid ... dequant_plus_gemm_us=... scratch_bytes=...` at rows 128 and 512.
These are ten-repeat screening timings. Follow with matched service baseline,
direct-FP8, and hybrid requests; measure prefill and decode separately, include
an unchanged-vs-unchanged control, and retain opt-in status unless numerical,
whole-service and latency gates pass.
