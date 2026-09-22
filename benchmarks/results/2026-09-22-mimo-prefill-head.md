# MiMo final-row prefill vocabulary head

Candidate based on `c91a50b` (the cache-only native-MTP3 snapshot). Ordinary
single-sequence prefill now projects only the final post-normalization hidden
row. Its logits stay at `[T-1, vocab]`, as required by `finish_run`. All hidden
rows and MTP history storage remain unchanged. Diagnostic `all_rows` and
decode/speculative-verification calls still project every row. Other model
families and the shared session core are unchanged.

Intermediate chunks still produce their final logits: the current session
contract returns those outputs, and this change does not introduce a separate
no-output contract. Full logits workspace allocation is retained for diagnostics
and decode. This is a compute reduction, not a memory reduction.

Changing the head GEMM from T rows to one crosses from cuBLASLt to the existing
GEMV implementation for larger chunks. Its reduction order may change FP32
logits. Backbone computation and normalization do not change. GPU numerical
and service acceptance remain required; no numerical threshold is relaxed.

## Local validation

The requested `DGPP_BUILD_JOBS=2 scripts/spark-cross build --target
 dgpp_serve_app mimo_head_check` encountered a missing Docker bridge (`docker0`).
The equivalent container command with `--network none` bypasses the network
issue. A first build then exhausted the workstation root filesystem while
writing CUDA compiler intermediates. The final retry keeps build objects in
`/mnt/benchmarks/dgpp-pipeline-builds-20260922/prefill-head`, linked from the
worktree build directory, and bind-mounts a sibling `prefill-head-tmp` directory
at container `/tmp`. No Spark, serving API, or GPU access is used by this task.

The ARM64 `dgpp-serve` and `mimo_head_check` targets cross-build successfully
with two jobs, and `git diff --check` passes. GPU execution and numerical
acceptance are pending parent-operated validation.

## Parent-operated GPU checks

The `mimo_head_check` executable uses the exact production head helper with
real checkpoint weights. It does not load the backbone or use the fabric.
Run each TP2 rank independently on a GPU, using the same checkpoint:

```bash
for rank in 0 1; do
  for rows in 1 2 4 8 9 31 128 256 2048; do
    ./mimo_head_check "$CHECKPOINT" "$rows" "$rank" 2 "/tmp/head-r${rank}-t${rows}"
    # Exit 2 denotes a numerical difference; retain outputs and evaluate it.
  done
done
```

Use a shell that continues after exit 2. Exit 0 means bitwise equality,
exit 1 is a contract/error failure, and exit 2 means the final-row shape changes
some FP32 logits. Every run checks all hidden rows bitwise, NaN sentinels in
unused logits, and bitwise equality of the full-row diagnostic/decode head
paths. It reports differing-value count, maximum absolute error, relative L2,
and each slice's argmax. Both final-row vectors are saved as
`OUTPUT_PREFIX.baseline.f32` and `.candidate.f32`. Concatenate corresponding
rank slices before comparing the global argmax/top-k. Rank-local argmax alone
cannot establish the model's chosen token.

The default activations are a deterministic sinusoidal fixture. For real
activations, pass a seventh argument containing exactly `[ROWS,4096]`
post-final-norm BF16 values captured from the backbone. Projection checks every
one of these hidden values remains unchanged. The checkpoint determines the
actual width and vocab slice (TP2: H=4096, V=76288); it is not approximated by a
smaller fake matrix. World 1 can also be tested with rank 0, at higher memory
cost. Run a partial-chunk case under Compute Sanitizer memcheck.

For model acceptance, compare baseline/candidate fixed greedy prompts with
MTP3 on and off, cold and cached multi-chunk prompts, concurrency two, and
speculative verification. Preserve existing logits tolerances and near-tie
rules; the standalone exact gate is deliberately stricter and never silently
turns a difference into a pass. Record outputs, draft acceptance counts and
errors before interpreting timing. Repeat matched cold-prefill workloads
(e.g. 480, 2030, 8030 tokens) with the same budgets/cache state, retaining
baseline repeat variance. No speedup or full-model numerical equivalence is
claimed by the cross-build or isolated head probe.
