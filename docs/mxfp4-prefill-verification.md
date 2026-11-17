# Two-Spark MXFP4 prefill and verification

This comparison uses DeepSeek V4.1 Flash 1017 MXFP4 on two GB10s, tensor
parallelism two, six routed experts per token and three DSpark predictions.
The 1 GiB prefix cache, one-million-token KV capacity and approximately
100.6 GiB expert pool per rank stayed at their serving settings.

## Implementation

* `DGPP_MX_FUSED_ACTIVATION` fuses the routed SwiGLU into the native MX
  down-input quantizer. Gate and up stay in BF16 through the same clamp and
  intermediate rounding steps. Each live K32 group writes MXFP8 codes and
  its E8M0 scale once. The shared expert retains its own activation. Mode 1
  uses 96 CTAs and mode 2 uses 384. Mode 3 computes 64 gate and 64 up
  columns together from the resident packed W13 table. Its epilogue applies
  the same BF16 rounding and emits the down-input codes and scales directly.
  A wave containing canonical or chunk-direct weights uses mode 2. The
  launcher determines eligibility from the leases it already holds.
* `DGPP_CSA2_LISTED_PREFILL=1` scores the candidate blocks from layer 20
  directly against the paged keys at reindex layers 24, 28, 32 and 36.
  It includes the newest incomplete block. Above 32,768 visible entries
  it bypasses the full gather and uses the existing FP8 tensor-core head
  reduction on eligible entries. The selector reads their original logical
  positions. Dense diagnostics can still request every score. Bounded
  prefill runs these decoder layers on the tail segment.
* `DGPP_AUDIT_PREFIX_REUSE=1` compares the rendered token IDs of a request
  with the preceding retired request's prompt and generated IDs. The scheduler
  logs lengths, the first differing position, the actual attach position and
  snapshot alignment. Token IDs and text stay out of the log. Use consecutive
  requests at concurrency one for turn-to-turn attribution.
* `DGPP_MX_VERIFY_SINGLETONS=1` partitions four-row W4A16 verification on the
  GPU. A dedicated kernel reads each singleton expert directly from its prepared
  K16/N64 weight slabs and combines the gate/up projections with SwiGLU.
  Reused experts go through the existing b12x kernel. Both write the normal
  routed output slots for reduction. The ordinary b12x backend handles other
  verification widths.

## Measurement

The C1 diagnostics used a 24,317-token translated tool prompt and a 288,240-token
prompt built by repeating its system content. Each full prompt bypassed prefix
attachment. The 24K warm samples have zero expert-file reads on both ranks.
The 288K requests follow those same warmups and still incur expert reads.
Prose and code probes use the engine's decode-step rate. These measurements
are separate from the deployment campaign in [benchmarks.md](benchmarks.md).

The recorded MXFP4 routing trace contains 4,480 four-row target layer calls.
Of 24 assignments per call, an average of 17.57 experts were distinct and
13.01 served exactly one row. The verification kernel benchmark replays
those IDs with synthetic weights and activations. The tool-serving comparison
uses the checkpoint.

At 288,240 entries and 32 query rows per launch, median GPU time for gathering
and scoring all visible entries was 2.928 ms. Scoring the listed keys directly
took 0.203 ms. With L2 eviction before each sample those times were 2.940 and
0.240 ms. The selectors and the rest of serving are outside these GPU times.

On a saved 4,096-token prefill layer with 24,576 real routed rows and 1,152
intermediate columns per rank, separate SwiGLU and quantization took 1.440 ms.
The fused 96-CTA launch took 1.070 ms and the 384-CTA launch took 0.713 ms.
A 576-CTA diagnostic took 0.686 ms. The kernel test compares every code and
scale, including the bytes in gaps between segments.

| Flags on the first serving binary | Mean warm 24K prefill, s | 288K prefill, s |
| --- | ---: | ---: |
| Previous kernels | 10.985 | 160.418 |
| 96-CTA fusion and listed scoring | 11.036 | 160.491 |
| 96-CTA fusion | 11.016 | 160.457 |
| Listed scoring | 11.099 | 160.746 |

Each 24K mean uses the five zero-expert-read requests after two warmups. Listed
scoring is below its dispatch threshold at 24K. The two comparisons of the
96-CTA fusion change sign. Each longer request read 7.242 GiB of expert
records per rank. All four settings returned the same content, tool arguments
and acceptance on these prompts. The 288K prompt-only times span 0.33 s,
which limits what this comparison establishes about the tail optimization.

The same binary served all four arms (base revision `efe7e58d` plus the prefill
changes, SHA-256 `e795846d1ebc112719572e8600a75959866a6291e78555465334d7237ee2d634`).

A second comparison kept activation fusion off and changed only listed scoring.
A 96,296-token seed request populated the prefix cache. Six controlled tool-result
suffixes with different leading labels each attached 96,256 tokens and computed
249. Listed scoring saved 12.6, 14.6, 7.7, 21.1, 16.1 and 28.5 ms respectively
in engine prefill. Mean latency went from 1,089.9 to 1,073.1 ms (1.54%).
The last four averaged 868.4 and 850.1 ms (2.11%). Expert reads declined
through the sequence from 7.08 to 0.27 GiB per rank and matched exactly for
each pair. This is one pair of server starts. It supports enabling the
listed scorer on long cached continuations without extrapolating a comparable
gain on a complete 288K prefill.

The second binary (`d1553285`, SHA-256
`592ab4506f33233dcd746cf7505556e8a3647d557f72b00825bd62e2c7257e67`)
also compared the 384-CTA fusion with the original two launches. Four warm
24K requests per boot averaged 11.115 s with listed scoring alone and 10.995 s
with fusion and listed scoring, a 1.08% reduction. The separate control boot
with both switches off averaged 11.037 s; listed scoring does not dispatch
on this prompt. The effect size is boot-sensitive. The larger grid also cut
isolated fused activation time on the same route replay.

Across 34 captured four-row routing patterns spanning 10 through 24 distinct
experts, the baseline b12x chain averaged 717 µs and splitting singleton
experts averaged 760 µs. Three patterns favored the split. Outputs were finite,
both singleton and reused groups agreed within the benchmark's printed
precision (normalized RMS error below 0.0000005), and the shared output slots
remained untouched. The replay includes both ends of the observed reuse range.

The first serving test exposed a dependence on which experts were resident.
Compacting the b12x packet from the live slots alone removed masked groups
from its split-K schedule. The corrected partition derives group membership
from all 24 routed IDs and carries masked groups at their scheduled positions.
It also loads eight adjacent scale bytes at once and prefetches upcoming
weight slabs. A regression check runs the same routes as one resident group
and as two disjoint waves and requires bitwise equality of the assembled
routed outputs and the shared-slot sentinel.

On a wider distribution of synthetic weights and inputs, the corrected
split passed that check on all 34 real routing patterns. It averaged
740 µs versus 714 µs for b12x and lost on all 34. Maximum normalized RMS
difference from b12x was 0.000040 on singleton outputs and 0.000105 on
reused outputs. The changed reduction order between backends can therefore
change greedy outputs and speculative acceptance even though the mask
itself preserves the final split backend's arithmetic.

## Proxy prefix reuse

An actual ephemeral Codex client selected `spark-primary` through its configured
OpenCodex Responses endpoint. The model invoked the shell, ran `pwd` and returned
the correct working directory. The next request shared all 17,480 previous
prompt tokens, attached at 17,408, and computed 249 of 17,657 tokens. Codex's
usage independently reported 17,408 cached input tokens. A separate Responses
tool loop used the model's actual call ID and arguments and returned a real
directory listing. That continuation shared 24,312 previous prompt tokens,
attached 24,192, and computed 329 of 24,521.

The snapshot alignment is 128 tokens. Both attaches reached the largest
aligned point inside the measured common sequence. In both examples the first
difference occurred at the first previously generated token. The audit retained
only counts and cannot identify the exact normalization of assistant output
responsible for that difference. No prefix-cache policy change was warranted by
these two exchanges.

## Deployment decision

The final binary is revision `87321ccd`, SHA-256
`2c381e9d047ad81f03f3ff579a528730677c98c4971b512587fa9500fd47a3e6`.
Both ranks ran it with zero process swap and matching operation streams.
Each decode figure is the mean of three consecutive C1 probes with zero
expert-file reads on both ranks and the same prefix attach point.

| Verification backend | Prose decode, tok/s | Prose accepted / passes | Code decode, tok/s | Code accepted / passes |
| --- | ---: | ---: | ---: | ---: |
| b12x | 38.90 | 157 / 98 | 42.84 | 165 / 90 |
| Singleton split plus b12x for reused groups | 40.76 | 164 / 91 | 43.48 | 169 / 87 |

The split generated different continuations. Decode time divided by draft
passes increased from 66.89 to 68.75 ms for prose and from 66.13 to 67.41 ms
for code. Both probes stopped at 256 output tokens, so these rates alone
cannot establish time to a correct answer.

Four further directory-tool tasks required a correct call and the first name
in the actual listing. Each task ran to completion twice with warm experts and
zero expert reads. Both backends returned all four expected answers, with the
same tool arguments, generation lengths and accepted predictions. Mean engine
decode time for call plus final answer was 900.3 ms with b12x and 917.4 ms
with the split. All four tasks favored b12x on decode. Mean client wall time
for the two requests was 2.098 and 2.143 s, respectively. These short tool
prompts fell below the prefix admission floor.

The service uses 384-CTA routed activation fusion, listed CSA2 prefill and the
ordinary b12x four-row verification backend. The split remains available for
controlled comparisons. The proxy audit is disabled for ordinary traffic.
Full deployment campaign rows and broad model-quality scores are outside this
focused comparison.

## Paired gate/up prefill on two Sparks

The resident specialization in revision `e28aa505` consumes the existing
packed W13 payload and scale table directly. A thread block computes 64
columns of each projection using the same input rows. Its epilogue reuses
the projection pipeline's shared memory, pads each row by eight BF16
elements and writes one set of MXFP8 down-input codes and scales. Mixed
waves use the existing projections and 384-CTA quantizer. Both paths preserve
the gate, up, SiLU and product rounding, the shared-expert stream and the
expert pool budget. The existing intermediate allocations still serve
fallbacks. This change reduces intermediate traffic without freeing those
allocations for other experts.

In a C1 diagnostic, each arm processed the same 24,317-token tool request
eight times. The final six requests per start had zero cached prompt tokens,
zero expert-file reads on both ranks, matching operation streams and zero
process swap. All returned the same 50 tokens and directory arguments, with
38 accepted predictions across 14 speculative passes. The prior production
binary averaged 10.913 s of warm prefill. The first implementation supported
all weight layouts inside the paired kernel and took 11.219 s against
11.139 s for its own control. The final specialization gave the following
same-binary comparison (SHA-256
`a3cf4d608a2c7764be5037b882ac6256909551d48af43c6fc790ca3b9802b32c`).

| Routed prefill path | Mean prefill, s | Range, s | Prompt tokens/s | C1 engine decode, tok/s |
| --- | ---: | ---: | ---: | ---: |
| Separate gate/up, mode 2 | 10.967 | 10.917–11.003 | 2,217 | 49.14 |
| Packed gate/up, mode 3 | 10.425 | 10.393–10.456 | 2,332 | 49.66 |

With the real 4,096-token routing patterns and synthetic packed weights, the
gate/up-to-quantized-activation chain took 17.00 to 12.58 ms on layer 10.
The least-skewed and most-skewed saved patterns went from 16.69 to 12.51 ms
and 17.85 to 13.63 ms respectively. Codes and scales matched byte-for-byte on
all three. The CUDA regression also exercises noncontiguous routed rows,
a second M tile, small waves, and graph replay. These focused comparisons
measure the paired prefill path. Decode continues to use grouped b12x.

After promotion the normal service repeated the six warm prefills in 10.454 s
on average. A 96,296-token prompt following the same 24K sequence took
46.649 s in mode 2 and 44.464 s in mode 3, one request per mode. Both ranks
read 8.675 GiB of expert records in each arm, cached zero prompt tokens and
completed with identical tool arguments, 45 output tokens and 33 accepted
predictions in 11 passes. A separate two-turn request on the restored
endpoint listed the source directory and returned its first entry correctly.
The two-Spark MXFP4 streaming template selects mode 3.

## Four-row verification on 48 SMs

At speculative depth three the target routes four rows to six experts each.
The packet groups the 24 assignments by expert. An expert with one live
assignment gets one eight-row tile with seven masked lanes. Across the
4,480 recorded layers, live assignments filled 17.1% of those tile lanes.
Each rank reads its local tensor-parallel shard of each selected expert once
in the ideal weight-traffic model. Padding the row dimension adds arithmetic
without multiplying the expert's weight size.

The route replay used 32 evenly spaced records and both overlap extremes from
that trace. The 34 calls averaged 17.59 distinct experts. For a rank-local
MXFP4 expert, three 5,120 × 1,152 matrices of four-bit weights plus one
scale byte per 32 weights occupy 9,400,320 bytes. This gives a mean of
165.34 MB of selected weights and scales per layer per rank.

| Isolated change | 32-block b12x control, µs | Candidate, µs | Faster routes out of 34 |
| --- | ---: | ---: | ---: |
| b12x, 40 persistent blocks | 711.94 | 712.29 | 17 |
| b12x, 48 persistent blocks | 716.48 | 718.75 | 12 |
| Two launches, grouped live rows in BF16 MMA | 718.24 | 737.47 | 9 |
| Two launches, FP32 FMAs on live rows only | 718.50 | 758.52 | 0 |
| Live-row FP32, unrolled accumulators and packed exponent scaling | 715.75 | 746.50 | 0 |
| Grouped BF16 MMA, software-pipelined register loads | 717.46 | 734.97 | 7 |

The two-launch kernels give each block 64 output channels for one expert.
The gate/up kernel applies SwiGLU and the down kernel writes the routed
output slots. Both retain CUDA graph replay, support masked residency waves
and can spread work across 48 SMs without a device-wide software barrier.
The MMA version uses the tensor instruction's eight columns and stores the
live ones. The scalar version skips arithmetic on all padded rows. Changing
the block size, adjacent channel tiles and weight prefetch distance also
left mean replay time above the control.

To compare against the cost of reading the same weights, run
`mxfp4_backend_bench --weight-scan <four-row-route-file>`. Each line supplies
24 expert IDs in token-major order. The scan deduplicates the IDs and reads
each selected expert's prepared packed weights and scales with coalesced
16-byte loads. A warp XORs the words and writes a single result to keep
the reads observable. The timer includes this reduction. It measures GPU
memory access on resident synthetic data. SSD record verification is a
separate option. The first paired replay averaged **708.56 µs for the scan
and 722.99 µs for the full routed computation**. The scan supplies an
empirical reference. Cache reuse and the access patterns of each kernel
affect the result.

The serving diagnostic used two Sparks at concurrency one with mode-3
prefill and depth three. The test executable was built from `b004bb00` plus
the experimental grid and tile changes (SHA-256
`e3455b658677d424ad4e3a5637e522d0953fadefa36984fef08ccc3de56f114e`).
The 48-block fused kernel stalled on the first 24,317-token tool request.
Both rank logs continued to report the same graph generation (419) until
the test was stopped. Isolated replay on an otherwise empty GPU had passed
for all 34 routes. The b12x software barrier waits for every persistent
block to arrive, so reserving all 48 SMs creates a forward-progress risk
when other kernels share the GPU.

A bounded co-residency probe reserved 96 KiB of one SM's shared memory for
12 ms in a separate stream, then launched b12x on each of three routes.
The 32- and 40-block kernels finished in 0.76–0.85 ms while the reservation
was live. The 48-block kernel finished in 12.76–12.84 ms. This reproduces the
dependence on the last block being able to run. The identity of the kernel
holding the contested resource in the serving stall was not captured.

### Checklist review of the compact kernels

| Checklist items | Evidence and remaining limits |
| --- | --- |
| Execution shape, routes, realistic inputs | The earlier TP2 warm four-row profile identified routed verification as a major cost. The replay used both overlap extremes, real top-six IDs and synthetic BF16 token inputs. Down consumed distinct activations from each routed gate/up. Prefill chunk sizing does not apply to this four-row dispatch. |
| Packed weights, supported views, vector loads and scale reuse | Both projections read the production K16/N64 layout directly. A warp loads 16 weight bytes per lane and eight packed scale bytes per four-lane group. Each K32 scale supplies both K16 halves. Serving supplies prepared bindings for the current residency wave. Canonical and chunk conversion remain in the existing caller. |
| Lookahead, staging and pipeline depth | Tested two- and six-iteration L2 lookahead, explicit next-iteration loads into registers and two- and three-stage asynchronous shared-memory rings for weights and scales. The rings averaged 739.10 and 734.82 µs against their respective 722.12 and 720.35 µs controls. |
| SM resources, ready warps and local-memory traffic | CUDA's occupancy API reported four resident 256-thread blocks/SM for both original grouped projections, three for the two-stage ring and two for the three-stage ring. In the sampled gate/up call, achieved occupancy fell from 64.9% to 47.4% with the two-stage ring. Eligible warps per scheduler were 0.16 and 0.17. Nsight identified thread-local traffic. Storing four route IDs in separately selectable registers removed the 16-byte local allocation and the measured local load/store sectors on both projections. The register-pipelined replay then averaged 722.42 µs against 718.21 µs for b12x. |
| Tile ordering, sparsity and barriers | A block owns one expert's output channels and reuses each weight fragment across its live rows. Tested 128-, 256- and 512-thread blocks and adjacent channel tiles. Empty and masked groups return as a block. The MMA instruction still has eight columns. The scalar alternative explicitly skips padded arithmetic. Projection launches delimit the gate/up → down dependency. |
| Routing preparation, scratch and caller overlap | Both kernels reuse the dispatcher's packet and existing activation/output scratch. They add no device allocation, CPU route decision, copy or stream drain to a hit. GPU packet construction and shared-expert scheduling stay in the caller. No new wait gates were added. Graph capture and fresh-process serving were exercised. Per-stream overlap on the new kernels was inspected in source; its duration was not separately measured. BF16 activations need no input quantizer. |
| Arithmetic, output slots and first use | Synthetic inputs have independent gate, up and down weights with nonuniform scales. Each measured route passed finite-output, shared-slot sentinel and bitwise full-versus-two-wave checks. Reduced error on these inputs does not establish checkpoint-wide scale coverage. The final serving candidate used the original full-range scale decoder. Changing the order of reductions can change continuations and acceptance. |
| Dispatch, missed experts and user-visible timing | Both ranks' executable hashes and switches were checked from the live processes. Warm probes required zero expert-file reads on both ranks and a 24,448-token prefix attach. Both ranks had zero process swap and matching operation streams on completed arms. Four directory tasks checked the tool arguments and final answer against the local filesystem. Nsight measurements were separate from uninstrumented serving. Broader model quality, concurrency and every possible tail shape remain unmeasured. |
| Read-only comparison kernel | This benchmark reads the contiguous span of each deduplicated expert, including scales. It launches only in the standalone benchmark and consumes no serving scratch. The final benchmark build repeated the 34-route scan in 712.68 µs against 729.30 µs for full routed computation. |

Each warm continuation below had three measured requests. Milliseconds per
pass is engine decode time divided by the number of draft passes, including
draft, verification and other decode work.

| Serving kernel | Prose tok/s | Prose ms/pass | Accepted / passes | Code tok/s | Code ms/pass | Accepted / passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32-block b12x | 39.16 | 66.45 | 157 / 98 | 42.95 | 65.97 | 165 / 90 |
| 40-block b12x | 39.68 | 66.26 | 158 / 97 | 40.28 | 65.94 | 160 / 96 |
| First grouped MMA | 36.69 | 68.28 | 154 / 101 | 42.98 | 67.42 | 168 / 88 |
| Grouped MMA with register-resident slots and pipelined loads | 37.09 | 67.54 | 154 / 101 | 43.40 | 66.76 | 168 / 88 |

The two grouped versions produced matching continuations and pass counts
with one another. The correction reduced cycle time on both workloads.
The code probe accepted three more draft tokens than the 32-block control,
which accounts for the higher output rate despite longer cycles. The corrected
executable used the same source base and full-range decoder (SHA-256
`ffb9495083cd615860290152ecfe00ee7a3dcac4e7376ace0b5a3afc9f8fd78b`).
The later diagnostic launches used a 3 GiB startup headroom threshold after
a 4 GiB threshold declined startup. The 114.11 GiB memory plan was identical.

The four directory tasks produced matching arguments, answers, token counts
and accepted predictions in all four completed configurations. Each task
had two warmed repetitions with zero expert-file reads on either rank.
Mean engine decode time summed across the call and answer was 898.59 ms
for 32 blocks, 893.05 ms for 40 blocks, 921.34 ms for the first grouped
kernel and 910.86 ms for the corrected grouped kernel. All four tasks
favored 40 over 32 on this start, by 0.62% in aggregate. The longer code
continuation favored 32 blocks by 2.67 tok/s. Each configuration had one
deployment start per workload, so the 0.62% difference still includes
between-start variability.

A two-projection Nsight sample of the corrected register-pipelined replay
showed zero thread-local load and store sectors. The down projection had
948 shared-memory store bank conflicts across its 1,920 launched blocks.
Padding the reduction scratch by one float per warp row averaged 732.18 µs
against 722.03 µs for its paired b12x control. The stable mixed-workload
selection remains 32-block b12x. The two-Spark service resumed with the
original binary, mode-3 prefill and its 4 GiB startup headroom threshold.
