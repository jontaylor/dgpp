# Native MXFP4 prefill serving comparison

This carries the [weight-prefetch findings](mxfp4-kernel-attribution.md)
into a serving-capable expert backend. The comparisons are focused C1
screens on two GB10 Sparks, not new deployment campaign rows.

## Implemented paths

`DGPP_MOE_MX_NATIVE_PREFILL=1` selects native E4M3-activation/MXFP4-weight
prefill for supported routed experts. Mode 2 selects gate/up only; mode 3
selects down only. Small-row verification retains its existing backend.
Native expert execution supports canonical, prepared K16/N64 and chunk-direct
weights. Prepared weights are copied as contiguous slabs; canonical and
chunk-direct weights use explicit L2 lookahead. No full-expert conversion
or host synchronization is added to the expert chain.

Gate/up inputs are quantized once per layer and reused across paging waves.
Down inputs are quantized by a GPU work list for only the segments executing
in that wave. Both use disjoint regions of the existing gather allocation,
with a capacity guard; there is no new persistent GPU allocation or reduction
in the expert-cache budget. Shared-expert overlap is preserved. Unsupported
shapes retain the prior path.

The native expert tile has 128 output columns and eight warps.
`DGPP_MOE_MX_NATIVE_TILE_ROWS=64|128` and
`DGPP_MOE_MX_NATIVE_STAGES=2|3` select the tested alternatives. Larger
shared-memory configurations explicitly opt in to the required allocation.
The final implementation skips empty 16-row accumulator slabs, removes a
redundant pipeline barrier, and uses a 16-row/two-stage tile for waves with
at most 16 rows. `DGPP_MOE_MX_NATIVE_TAIL_SKIP=0` retains the original
padded computation for diagnostic comparisons.

`DGPP_DSV41_NATIVE_FP8_PREFETCH=1` adds weight-line lookahead to the optional
native dense CSA2 path. Its existing 128-thread/three-stage and
256-thread/two-stage variants remain selectable. This does not alter the
FP8 weights or native arithmetic; activation quantization is the existing
optional precision change.

## Validation and kernel screen

The new native fixture compares all three projection views against the
canonical native kernel, with independent gate/up/down weights and
nonuniform scales. It covers prepared up offsets, row and column chunks,
gathered token rows, padded output strides, BF16/FP32 output, segmented
quantization and graph replay. All four tile/stage combinations pass. Dense
prefetch passes the existing native grid-scale reference and graph checks.
These establish implementation consistency; they are not broad checkpoint
quality evaluation.

On the saved 4,096-token, top-six serving routes, prepared-weight kernel
times (excluding activation quantization) are:

| Expert implementation | Gate or up, ms | Down, ms |
|---|---:|---:|
| Existing BF16 activation | 9.175 | 10.882 |
| Native 64 rows, two stages | 8.868 | 10.541 |
| Native 64 rows, three stages | 8.720 | 10.440 |
| Native 128 rows, two stages | 8.369 | 9.539 |
| Native 128 rows, three stages | 12.986 | 14.410 |

The 128-row/three-stage shared-memory footprint admits only one block per
SM, explaining why another pipeline stage is not automatically helpful.
The existing 128-row BF16 tile already had canonical prefetch and remains
slower in the same replay. The dense native prefetch microbenchmark still
has shape-dependent results: narrow long-prefill projections lose to the
existing path, while wide output projections can win. Full serving, rather
than those isolated times, determines the retained settings.


The refined 128-row/two-stage kernel measured 7.969/9.367 ms for gate-or-up/down
on the original route replay. On the least-skewed and most-skewed saved layers,
it measured 7.923/9.309 and 8.537/10.160 ms respectively, versus
9.070/10.621 and 9.967/11.980 ms for the BF16 backend. These comparisons
exclude activation quantization. The seven-row fixture also exercises the
small-wave dispatch used by the final serving binary.


## Serving selection, 2026-11-17

The final same-binary comparison uses the 24,317-token translated tool request,
depth-three speculation, concurrency one and unchanged 108047546368-byte
streaming budgets per rank. Warm prefill is the median of repetitions 3–5,
with prefix reuse disabled (zero cached prompt tokens). Prose/code decode
rates below are engine decode steps only from the repeated 256-token
continuation probes; they exclude prefill and queueing.

| Configuration | Warm prefill, s | Prose decode, tok/s | Code decode, tok/s |
|---|---:|---:|---:|
| Existing BF16 expert backend | 11.408 | 39.93 | 42.46 |
| Native experts, 128 rows / two stages | 11.190 | 39.26 | 45.47 |
| Selective native dense only | 11.490 | 38.76 | 43.78 |
| Native experts + selective native dense | 11.190 | 39.81 | 39.93 |
| Native experts, 64 rows / three stages | 11.477 | 39.15 | 45.40 |
| **Native experts, 128 rows / two stages + attention merge** | **11.105** | **39.10** | **42.94** |

Select the last configuration: native expert mode 1, tile rows 128, stages 2,
tail skipping enabled, CSA2 merge4 enabled, native dense disabled. All other
serving settings, cache budget, context capacity and endpoint configuration
remain unchanged. This is the lowest observed warm-prefill latency, about
2.7% below baseline (approximately 2,190 versus 2,132 prompt tokens/s).
The margin over native experts without merge is only 0.8%; this short screen
does not establish a statistically robust lead across workloads.

Both ranks had zero expert-file reads and zero process swap in these warm
samples, and their operation streams matched in every arm. This does not
mean all model I/O vanished: Engram reads remain. All arms used the same
binary, built from `9f5e80ac` plus this native-serving change, SHA-256
`80cf8e7c645acfa39c3aa9e36c873ba771019a7f4d34172862db005653746beb`.
No competing build or benchmark ran on the serving fabric during measurement.
These are focused diagnostics, not official campaign rows or cold-prefill
claims. Production is restored once after the comparison.

The decode backend is unchanged. Different prefill arithmetic changes the
continuations and speculative acceptance, so decode differences do not
establish faster verification. Baseline prose/code accepted 161/164 draft
tokens across 96/91 passes; the selected configuration accepted 157/165 across
98/90 passes. The native-only arm's higher code rate therefore does not prove
a superior decode kernel. Continuation probes hit their 256-token cap and
are throughput probes, not completed-answer quality evaluations.

Native activation quantization is a precision tradeoff. Kernel reference,
layout and graph tests plus basic tool behavior have been checked; broad
checkpoint quality remains unmeasured. The projection gains do not translate
into a comparable whole-model gain: attention, communication, Engram access,
other projections and scheduling remain outside this optimization.

The selected binary is deployed with the normal port, keys and service unit.
Both ranks report its recorded hash, zero process swap and matching operation
streams. A completed tool round trip produced the correct directory call and,
after receiving the real listing, a concise 24-token final answer with
`finish_reason=stop`. The initial smoke helper accidentally retained a forced
`list_directory` choice on the final-answer request, causing repetitive output
with and without attention merge. Correcting the follow-up to `tool_choice=auto`
resolved it in both configurations; that contradictory probe is not quality
regression evidence. The deployment parser now accepts and propagates the
selected switches; its 40 Python tests pass, including example-config validation
and explicit propagation to both ranks.
