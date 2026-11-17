# MXFP4 kernel and cache follow-through

This follows the [initial kernel screen](mxfp4-kernel-options.md) on two GB10
Sparks. These are focused diagnostics, not deployment campaign rows. They
use the same 24,317-token tool request and tool-enabled prose/code
continuations, depth-three speculation, and MXFP4 weights. Prefill probes
disable prefix reuse; continuation probes include both first use and repeat.
Decode rates count engine decode steps, excluding prefill and queueing.

The subsequent [kernel attribution](mxfp4-kernel-attribution.md) identifies
missing GPU weight prefetch and reverses the native prototype performance
conclusion below. The serving configuration described here is unchanged.

## Why the extra reads did not converge away

Routing still selects six experts per target token. Expert reads measure
misses, not selections. Eight identical requests per configuration showed
stable route hashes within each configuration and stable misses after the
first request. The issue was cyclic eviction just beyond cache capacity:

| Quantity, per rank | Existing dense arithmetic | Native dense FP8 |
|---|---:|---:|
| Distinct experts in request working set | 11,006 | 11,047 |
| Whole experts fitting the original pool | 10,986 | 10,986 |
| Steady distinct misses / evictions per request | 587 | 1,202 |
| Expert reads per request, GiB | 5.159 | 10.564 |

Only 41 more distinct experts roughly doubled misses. Neither configuration
had unused resident experts at request end. Loading one needed expert
continued to evict another needed later; more identical warm-ups did not
resolve it. Activation quantization changed routes, not the top-k count.

An additional **768 MiB per rank** increased the pool from 197,749 to
199,291 chunks of 522,240 bytes, accommodating 11,071 whole experts at
18 chunks each. Both repeated working sets then reached zero expert reads.
The configured streaming budget changes from 107,242,240,000 to
108,047,546,368 bytes; this includes streaming allocations beyond the chunk
pool. The MXFP4 deployment example carries that increase. It does not imply
all traffic fits: even the tested native working set leaves only 24 whole
expert slots spare.

Warm prefill fell from approximately 12.2 to 11.4 seconds on this request.
The earlier native-dense regression largely disappeared at the larger
capacity. It was not valid to attribute that regression to native matrix
execution alone.

## Native expert execution

The benchmark-only W4A8 prototype now has asynchronous packed-weight
staging, register-side FP4 expansion, activation matrix loads, eight-warp
blocks, staged scale loads, and two/three-stage pipeline choices. Gate/up
quantizes each token once and reuses it across expert assignments; down
uses distinct expert activation rows. Comparisons replay actual top-six
serving routes, including a hottest expert with 1,372 rows in a 4,096-token
chunk, rather than assuming uniform 64-row groups.

On those identical routes, the initial native projection times of about
16–17 ms fell to about 10–12 ms. The existing BF16-activation path remained
about 9–10 ms. Activation quantization alone cost approximately 0.26 ms for
gate/up inputs and 0.36 ms for down inputs: eliminating its separate launch
cannot alone recover the remaining native gap. The experiment does not
perform CPU conversion or transfer expert weights through the CPU per tile.

Profiling the earlier native variant showed low occupancy and substantial
L2 activity, without local-memory spills. This motivated wider blocks,
staged scales and changes to tile geometry. Occupancy alone is not a
throughput result. The native prototype still reads canonical views; it is
not a prepared/chunk-direct serving backend. Its random-input relative L2
difference of about 2.44% is a quantization sanity check, not a model-quality
measurement.

## Compact scheduling and dense execution

`DGPP_MOE_FP4_COMPACT_TILES=1` extends the existing GPU tile-list producer to
FP4 prefill. It reuses allocated scratch, requires no CPU routing readback,
and supports graph capture. It builds one list for routed gate/up/down;
shared-expert execution remains separate. The capacity bound excludes
shared rows. Resident and paged wave tests check bitwise output equivalence.

The first compact version removed empty blocks but traversed output columns
before token tiles, weakening weight reuse. It was essentially tied in
serving. The follow-up preserves token tiles as the fastest index within
each expert, matching the original kernel's weight-reuse order.

`DGPP_DSV41_NATIVE_FP8_WIDE=1` selects eight-warps/two-stage native dense
execution when native CSA2 FP8 is enabled. Its grid-scale reference,
workspace, output-type and graph checks pass. At the enlarged cache budget,
the first serving comparison was:

| Configuration | Median warm prefill, s | Repeated prose decode, tok/s | Repeated code decode, tok/s |
|---|---:|---:|---:|
| Existing arithmetic / rectangular grid | 11.406 | 39.96 | 42.41 |
| First compact grid | 11.431 | 40.01 | 42.45 |
| First compact grid + wider native dense FP8 | 11.408 | 41.59 | 42.85 |

Prefill medians use the last three of five full-prefill requests. The
continuations each generate 256 tokens. Repeated continuations have zero
expert reads on both ranks. Compact scheduling preserves acceptance:
96 passes/161 accepted predictions for prose, 91/164 for code. Native dense
changes those to 92/164 and 90/165, respectively; its apparent decode gain
is not an isolated verification-speed improvement. No dense native gain is
claimed from this comparison.

Both ranks had zero process swap and matching operation streams during
these screens. Approximately 3.3 GiB system memory remained available per
rank with the larger pool; this is not a claim of zero system-wide swap or
of validated maximum concurrency/context capacity. Measurements ran without
another GPU world. Results, source diffs and binary identities are retained
as untracked diagnostics.

For route-matched kernel experiments, `mxfp4_kernel_bench ROUTE_IDS` accepts
whitespace-separated expert IDs, six per input token, in token order. It
uses synthetic activations/weights with those real routing assignments;
this is not a checkpoint-output or model-quality replay. Without a file it
runs uniform synthetic routing. The benchmark reports conversion separately
and includes conversion in each native total; compact totals include the
GPU list producer. Native variants must match the original native
accumulators exactly, and compact BF16 must match rectangular BF16 exactly.

The revised, weight-reuse-preserving compact grid passed the same bitwise
paging tests. In the canonical route replay its down projection improved
from 10.45 to 9.99 ms; gate/up remained close to baseline. In a fresh
same-binary serving comparison, median prefill was **11.427 seconds off
versus 11.437 seconds on**. Repeated prose decode was 40.00 versus 39.90
tok/s, and code was 42.46 versus 42.18, with identical acceptance and zero
expert reads. This is no useful serving improvement, so compact scheduling
remains opt-in and disabled. Native 64×64 and 32×128 tiles also failed to
beat the optimized 64×128 path; higher potential occupancy alone did not
improve throughput.

The first compact/wide-dense comparison used server SHA-256
`82e03b5a8b85bf2b5e11810927f9405085134fb127b8bfea373ac118fd561378`.
The revised compact comparison used
`41ceb702ddd12ba3a0177c80375801ff9f5a2c9c77b1c8d0b27b9c4aef2a022e`.
Both were built from `5bad4ab1` plus the recorded implementation diff.

The optimized native kernel profile reached **49.4% achieved occupancy**
(50% theoretical, 71 registers/thread), versus about 16.5% in the earlier
packed version. It had no local-memory spills, 61.1% L2 throughput and
30.6% compute throughput. These counters motivated a final coalesced
weight-scale layout experiment rather than more occupancy tuning.

With nonuniform scales and exact native-accumulator checks, transposing
weight scales improved the native gate/up total from about 10.37 to
10.07 ms, still slower than 9.35 ms for current arithmetic. Down remained
about 12.22 ms with transposed scales versus 10.28 ms for current arithmetic.
The scale transpose is preparation outside timing and uses no additional
scale bytes in the representation; the benchmark holds both layouts for
comparison. It is not integrated into serving. Its modest gate/up gain and
down regression do not establish a faster model backend.

## Retained configuration and limits

Retain MXFP4, depth-three speculation, four attention splits, early shared
expert prefill, 12 execution slots, and the larger streaming budget.
Compact FP4 scheduling, wider native dense FP8, 128-row expert tiles and
four-way attention merge remain disabled. Port, credentials and client
model alias are unchanged. The serving binary is the revised compact
comparison binary above, running with those alternatives off.

Not implemented here: native W4A8 execution of prepared/chunk-direct expert
views or fused native expert activation production. The optimized canonical
prototype still loses to current arithmetic on identical routes, and
conversion accounts for only about 0.3 ms of its total; those larger changes
have not demonstrated a serving benefit. This is a limitation of the
current implementation, not proof that native arithmetic cannot win.
Maximum-context/concurrency memory qualification and broad model-quality
evaluation were not performed in these focused C1 diagnostics. No quality
tradeoff is enabled in the retained configuration.
