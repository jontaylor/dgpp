# Compact materialized attention

`DGPP_MIMO_ATTN_TILE_ROWS=16|32|64|128` explicitly opts into a workspace-tiling
candidate. Unset retains existing dispatch and128-row default. Explicit128 is
an arithmetic/control configuration. This option takes precedence over the
experimental bounded/online/split flags, which should be unset for isolated
comparisons. `DGPP_MIMO_FUSED_PREFILL` retains its existing meaning.

Global prefill calls the original materialized attention kernels with the
selected query-row tile. Mapped global decode calls the original fast tensor
decode kernels in slices no wider than that tile. Q/output/positions/map pointers
advance together; cache-plane addresses remain absolute. Thus64-row DFlash
verification fits a16-row attention workspace. No attention score, softmax, or
value-accumulation arithmetic is intentionally changed. SWA retains the original
128-row prefill scheduling and scalar mapped decode; it does not use online
softmax and does not inherit smaller global tiles.

Workspace is the maximum of:

* `min(rows,global_tile) * global_capacity * local_heads * 6` bytes;
* `min(rows,128) * expanded_SWA_ring_capacity * local_heads * 6` bytes.

The SWA floor matters at small contexts or large forward chunks. For example,
512 forward rows,128 global context,1024 ring capacity and32 local heads require
24MiB, even though the compact global matrix alone would be much smaller.
At256K context,32 local heads and ordinary forward chunks, tiles16/32/64/128
require768MiB /1.5GiB /3GiB /6GiB respectively. Ordinary8-row decode fits inside
all those allocations;64-row verification is sliced instead of enlarging scratch.
Model memory planning, allocation and the real-weight layer probe use the same
workspace helper. This is capacity reduction, not a measured speed claim.

## Parent validation

```sh
DGPP_TEST_FILTER=compact_materialized ./mimo_attn_test
DGPP_TEST_FILTER=bounded_online ./mimo_fp8_test
DGPP_TEST_FILTER=compact_materialized compute-sanitizer --tool memcheck ./mimo_attn_test
DGPP_TEST_FILTER=compact_materialized compute-sanitizer --tool racecheck ./mimo_attn_test
DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT=8192 DGPP_MIMO_ATTN_BENCH_FIRST_MODE=8 ./mimo_attn_test
DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT=65536 DGPP_MIMO_ATTN_BENCH_FIRST_MODE=8 ./mimo_attn_test
DGPP_MIMO_ATTN_BENCH_FP8=1 DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT=65536 DGPP_MIMO_ATTN_BENCH_FIRST_MODE=8 ./mimo_attn_test
DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT=262144 DGPP_MIMO_ATTN_BENCH_FIRST_MODE=8 DGPP_MIMO_ATTN_BENCH_LAST_MODE=8 ./mimo_attn_test
```

Benchmark modes8/9/10/11 select global tiles16/32/64/128, with the same random
cache/query workload and original partial-probability-fusion flag enabled.
When starting at mode8, each mode allocates only its required materialized
workspace; prior online partial scratch is omitted. `LAST_MODE=8` permits a
256K tile16 run without subsequently allocating the6GiB tile128 control.
SWA timings should remain comparable across all four modes.

The dedicated GPU gate requires **bitwise equality** to the original algorithms,
with129-row prefill tails and64 mapped decode rows, both fused-probability
settings, global/SWA attention, padding/invalid rows, CUDA graph replay, poisoned
scratch and a trailing allocation guard. FP8 tests also compare the same compact
algorithm against independently expanded BF16 cache contents. CPU checks cover
the long-context allocation formula, small-context SWA floor and8/64-row decode
bounds. Real-weight and serving checks remain required before adoption.
