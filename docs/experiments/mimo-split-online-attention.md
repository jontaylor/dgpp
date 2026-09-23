# MiMo split-key online attention candidate

Opt in with `DGPP_MIMO_SPLIT_ONLINE_ATTN=1`. This takes precedence over the
previous bounded/online switches; unset preserves all previous algorithms.
`DGPP_MIMO_ONLINE_SPLIT_KEYS` accepts 512, 1024 (default), or 2048. Set these
before process startup, including memory planning. Invalid split-key settings fail.

Global attention runs independent key partitions, followed by a stable merge.
Each partition retains its FP32 maximum, denominator, and 128-dimensional
unnormalized FP32 value numerator. The merge rescales all nonempty partials to
the global maximum, adds the optional zero-value sink once, and normalizes.
Empty/causally masked partitions publish zero mass; merge never reads their
uninitialized value storage. Invalid and padded query rows produce zero output.
SWA uses the previously implemented unsplit online kernel.

The split kernel skips shared-memory accumulator rescaling when no query's
maximum changes. This leaves the accumulator in registers at exactly the same
scale; initial empty accumulators are zero and need no rescaling either. This
optimization is a separate compile-time specialization: prior bounded/online
kernels retain their original rescaling behavior.

This retains the previous online candidate's BF16 QK score/scale and BF16
**unnormalized** PV weights. Partitioning changes local maxima and reduction
order and therefore can change outputs further. It does not reproduce eager
BF16 subtraction/normalized-probability rounding. Existing numerical gates have
not been loosened. Target-hardware quality and performance remain separate gates.

## Workspace and capture

Production prefill and mapped decode process at most 32 query rows at once.
Wide DFlash verification (up to64 rows) reuses the same partial allocation across
32-row mapped slices, preserving absolute cache-plane indices. Workspace is exactly
`rows * local_q_heads * splits * 130 * sizeof(float)` bytes, reusable across
layers and tiles on the model stream. `splits=min(256,ceil(capacity/target_keys))`;
actual partition span is rounded to a multiple of 16 and covers the entire
capacity. Capping the count bounds scratch as context grows. At 256K, 32 local
Q heads, and default 1024 target keys, this is 130 MiB (versus 6 GiB baseline
scores/probabilities). At the maximum supported 64 local Q heads it is 260 MiB.
No persistent full-context score/probability buffers or dequantized cache copies
are introduced. The memory ledger reports `MiMo split online partials`, and the
model and real-weight layer probe allocate this exact size. Kernel APIs perform
no allocation, device/host transfer, or synchronization beyond stream ordering;
global split attention captures two kernel nodes.

## Parent GPU checks

Build `mimo_attn_test`, `mimo_fp8_test`, `mimo_layer_check`, `dgpp_serve_app`.
Run each split-key configuration in a fresh process because configuration is
cached at startup:

```sh
DGPP_TEST_FILTER=split_online ./mimo_attn_test
DGPP_TEST_FILTER=bounded_random ./mimo_attn_test
DGPP_TEST_FILTER=bounded_online ./mimo_fp8_test
DGPP_TEST_FILTER=split_online compute-sanitizer --tool memcheck ./mimo_attn_test
DGPP_TEST_FILTER=split_online compute-sanitizer --tool racecheck ./mimo_attn_test
DGPP_MIMO_ATTN_SANITIZER_SMALL=1 DGPP_TEST_FILTER=bounded_random compute-sanitizer --tool racecheck ./mimo_attn_test
DGPP_MIMO_ONLINE_SPLIT_KEYS=512 DGPP_TEST_FILTER=split_online ./mimo_attn_test
DGPP_MIMO_ONLINE_SPLIT_KEYS=2048 DGPP_TEST_FILTER=split_online ./mimo_attn_test
DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT=8192 ./mimo_attn_test
DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT=65536 ./mimo_attn_test
DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT=262144 DGPP_MIMO_ATTN_BENCH_FIRST_MODE=7 ./mimo_attn_test
DGPP_MIMO_ATTN_BENCH_FP8=1 DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT=65536 ./mimo_attn_test
DGPP_MIMO_DECODE_BENCH=1 ./mimo_attn_test
DGPP_MIMO_ATTN_BENCH_FP8=1 DGPP_MIMO_DECODE_BENCH=1 ./mimo_attn_test
```

Prefill mode7 and decode mode4 are the new split path. Earlier mode IDs remain
unchanged. Benchmark random cache data is identical across all algorithms and
optionally converted to E4M3 with `DGPP_MIMO_ATTN_BENCH_FP8=1`; decode benchmarks
now use the same random-cache fixture instead of the old unwritten sentinel
cache. Timing comparisons must rerun all modes on this same fixture.

The analytic test uses zero Q/K and distinct per-key/per-plane V values, checking
partial coverage, odd tails, sinks, mapping, invalid positions, padding, graph
replay, poisoned partial buffers, and a trailing workspace guard. Random tests
report actual errors against both serial and production parallel softmax
oracles. The FP8 test compares split FP8 execution bitwise to the **same split
algorithm** reading an independently expanded BF16 cache, separating storage
integration from algorithmic approximation. Real-weight layer and full-service
checks use the same tools as the previous online candidate with the new flag.
