# MiMo bounded attention experiments

Set before process startup; default behavior is unchanged:

* `DGPP_MIMO_BOUNDED_ATTN=1`: three fused tiled passes for prefill, scalar no-scratch decode. Preserves BF16 dot/scale/subtraction/probability stages. FP32 reduction order differs from parallel baseline; QK/PV tensor-core arithmetic matches the prefill family, not scalar decode. Decode retains scalar arithmetic and recomputes QK.
* `DGPP_MIMO_ONLINE_ATTN=1`: one-pass online prefill and mapped decode. Implies bounded allocation. Retains BF16 QK dot/scale, but computes unrounded max differences and rounds **unnormalized** weights to BF16 for PV before final normalization. Online accumulator rescaling and tensor-core QK/PV also reassociate arithmetic. This is an approximate numerical candidate, requiring real-weight quality validation.

Both eliminate the entire context-sized FP32-score/BF16-probability allocation and its memory-plan charge. With 128 attention tile rows, 32 local heads and 262144 context this removes 6 GiB per rank. Per CTA shared storage is fixed at about 26 KiB; there are no per-context partial arrays, persistent dequantized caches, allocation calls, or host synchronization in either kernel. The tiled kernel supports QK=192, V=128, GQA, zero-value sinks, causal tails and non-power-of-two ring wrap. Prefill's existing public contract requires consecutive nonnegative positions; mapped decode supports inactive/invalid positions with zero output.

Three-pass mode trades score/prob storage for recomputation. Online mode additionally stores/reloads the local FP32 accumulator when its maximum changes. Neither is assumed faster before measured target-hardware evidence. Online decode currently uses one 16-row tensor tile per actual row, so it wastes tensor lanes; this is a first bounded candidate, not a claimed optimized decode design.

## Parent hardware verification

Build `mimo_attn_test` and `dgpp_serve_app` using the Spark cross preset. Run on an idle target GPU:

```sh
DGPP_TEST_FILTER=parallel_chunk ./mimo_attn_test
DGPP_TEST_FILTER=online_mapped ./mimo_attn_test
DGPP_TEST_FILTER=bounded_random ./mimo_attn_test
./mimo_attn_test
DGPP_MIMO_ATTN_SANITIZER_SMALL=1 DGPP_TEST_FILTER=bounded_random compute-sanitizer --tool racecheck ./mimo_attn_test
DGPP_TEST_FILTER=online_mapped compute-sanitizer --tool memcheck ./mimo_attn_test
DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT=8192 ./mimo_attn_test
DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT=65536 ./mimo_attn_test
DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT=262144 DGPP_MIMO_ATTN_BENCH_FIRST_MODE=5 ./mimo_attn_test
DGPP_MIMO_DECODE_BENCH=1 ./mimo_attn_test
```

Prefill benchmark modes: 3=wide materialized baseline, 4=old partial probability fusion, 5=bounded three-pass, 6=online. Default starts at mode3. First-mode5 omits the context-sized score allocation entirely (one-byte placeholder in test helper only). Decode modes: 0=saved-score scalar, 1=existing split tensor PV, 2=no-scratch scalar, 3=online. Tests report actual relative L2, max absolute difference and changed BF16 elements on causal chunks, tails, wrap and reset; online synthetic gate is 2.5% relative L2 and is not a quality claim. Real model logit/top-token and long-context retrieval checks remain required; none were run on the build host.

## Existing real-weight layer oracle

`mimo_layer_check` honors both options and omits score allocation for bounded/online
runs. Use `mimo_layer_check SNAPSHOT LAYER OUTPUT_BF16 CHUNK TOKENS` with
identical snapshot, layer, chunk and token counts across baseline and candidate runs. Set `DGPP_MIMO_LAYER_ATTN_DUMP=1` to retain pre-output-projection
attention alongside final layer output; this distinguishes attention drift from
routing amplification. Compare final rows with `tools/mimo_layer_compare.py
BASELINE_BF16 CANDIDATE_BF16`. Existing tool tolerance is a layer-output gate,
not an end-to-end quality guarantee.
