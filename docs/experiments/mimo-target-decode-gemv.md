# MiMo target graph GEMV diagnostic

`DGPP_MIMO_TARGET_DECODE_GEMV=1` temporarily raises the target BF16 GEMM
GEMV row ceiling to `max_decode_rows_` while constructing a captured decode
run. Default behavior is unchanged. The scoped bound is restored on every
return and exception, so prefill, eager decode, and native draft calls keep
their existing lowering. DFlash owns a separate GEMM instance. FP8 projection
selection (`dense_gemv_rows`) is independent and unchanged.

This isolates an observed boundary: the target default BF16 bound is eight
rows; C1 DFlash verifies eight rows and C2 verifies sixteen. BF16 operations
at sixteen rows otherwise fall through to cuBLASLt. C1 is expected unchanged
because its eight rows already use GEMV. This is a diagnostic hypothesis,
not evidence that rounding explains the observed C2 differences.

Run the existing exact C2 code/JSON requests with only this flag changed.
Compare content and reasoning against both target-only and baseline DFlash.
If C2 converges to target-only, repeat to check stability; if it does not,
inspect shared-prefix target logits at eight and sixteen rows before blaming
the drafter. Neither outcome establishes broad generation parity.

The saved `tp2-correctness-01` SSE streams first differ at reasoning text
chunk 12 for code, after `The user wants a Python function that merges
overlapping intervals,`: control emits ` along`, DFlash emits ` plus`.
For JSON, chunk 28 follows `The user wants a JSON array of 12 objects with
id, square, and parity for integers 1 through 12.\n\n`: control emits `Let`,
DFlash emits `square`. These are SSE chunk ordinals, not independently
verified tokenizer IDs.

Local validation: `git diff --check` passed; the changed `model.cpp`
translation unit cross-compiled with `dgpp-spark-cross:cuda13`. No hardware
run was performed for this change.
