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

## Four-row dense projections and per-expert handoff

The `dsv41_verify_dense_bench` diagnostic replays six TP2 FP8 projection
shapes on four independent BF16 inputs. It rotates synthetic packed weights
through a working set at least four times the device L2 capacity, alternates
the order of candidates, and compares the four-row result with the same
candidate run one row at a time. Pairing adjacent CSA2 output groups reduced
the median time for all four groups from 90.54 to 83.48 µs on the first
run and from 93.80 to 84.95 µs on the second. Each group retained its
accumulation order. Four warps for the unsplit Engram projection took
315.05 versus 329.74 µs and 341.80 versus 345.21 µs. Both changes matched
the reference outputs bit for bit on these inputs. Splitting the following
CSA2 output projection over K changed the reduction order. Its isolated
result varied between runs.

The serving experiment used two GB10s, TP2, depth three, and the same
24,448-token prefix for three warm 256-token continuations each on prose
and code. Both ranks recorded zero expert reads on these repetitions.
Decode throughput counts engine decode steps and excludes prompt time.
The directory measurement sums decode time across the tool call and its
answer, averaged over eight zero-read exchanges on four tasks. All four
arms produced the correct arguments and answers on those tasks.

| Diagnostic arm | Prose decode tok/s | Prose ms/pass | Code decode tok/s | Code ms/pass | Directory decode ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior 32-block b12x | 39.08 | 66.58 | 42.88 | 66.08 | 899.89 |
| Paired output groups and four-warp Engram | 39.16 | 66.45 | 43.39 | 65.30 | 891.44 |
| Also split the output projection over K | 41.39 | 66.24 | 42.72 | 65.60 | 888.33 |
| Per-expert handoff with corrected reset | 36.64 | 71.75 | 38.17 | 71.07 | 970.72 |

Splitting the output projection increased prose acceptance from 157 to 162
predictions per 256-token request and reduced code acceptance from 165 to
164. The two dense variants matched on tokens, accepted predictions and
pass counts across the eight directory exchanges. The 3.11 ms difference
between them on that workload includes between-start variation.

On the 34-route kernel replay, allowing down work to start when each expert
became ready reduced the mean from 828.59 to 820.39 µs. The existing b12x
kernel took 710.62 µs. The first serving implementation stalled. After the
readiness state was reset on every replay, the completed full-model
comparison still favored b12x. The production control used binary
`a3cf4d608a2c`. The three candidates used source base `9771204e` and binaries `c0d433c2e968`,
`d7210c932efa` and `07bd53510dd0`, respectively. These measurements
compare individual implementation choices at concurrency one.

FlashDecoding++ describes the trade between output-tile width and available
blocks for shallow GEMMs on A100. MonoMoE schedules the two projections with
independent readiness flags on H200. Here the depth-three verifier supplies four
rows on each GB10 and runs alongside the shared expert and TP2 communication.
[FlashDecoding++, pp. 7–9](https://arxiv.org/pdf/2311.01282#page=7) ·
[MonoMoE, pp. 8–9](https://arxiv.org/pdf/2609.04244#page=8)

### Selecting intermediate channels during verification

Each four-row forward has 24 expert assignments. For each token and TP rank,
the reader scores the 6 × 1,152 local intermediate channels and keeps a fixed
quota across its six assigned experts. The gate score uses the activated gate,
routing weight and down-column norm. The product score also uses the up value.
The sketch estimates gate and up from compressed weights and recovers a
low-rank approximation to the contribution from omitted channels. Only the
four-row target path changes. The grouped implementation applies the same rule
on a cache hit and miss. An initial implementation switched back to exact
computation on a miss and returned different greedy continuations across
repeats. It was excluded from the comparison.

The archive reader transposes down weights to make each selected channel
contiguous and charges every touched 32-byte sector. Its sketch adds about 7%
of a BF16 checkpoint and its recovery map another 0.5%. In this TP2 deployment,
reserving memory for per-resident metadata reduces capacity from 11,071 to
8,513 complete expert shards per rank. An exact-computation control with that
smaller pool read 12.08 and 5.81 GiB per rank on two code continuations after
prefix attachment. With the full pool the exact control read zero bytes.
The lighter gate and product readers retain the full pool. Their norm table
occupies 67.5 MiB per rank. The diagnostic also reserves scratch for the 24
active expert shards on a missed verification pass.
[Kim and Hwang, pp. 3, 8–10](https://arxiv.org/pdf/2610.37386#page=8)

The isolated replay sampled twelve verification passes each from layers 6,
20 and 36. It used rank-0 checkpoint weights, captured activations and routing
weights for the readers. The b12x benchmark used the same route IDs and
geometry with synthetic weights and activations. On layer 20, gate selection
at 12.5% took 501 µs with channel-contiguous weights and 619 µs on the
K16/N64 serving layout. A profiled serving-layout replay spent 249 µs on the
gate, 151 µs on selected up weights, and 175 µs on down. The corresponding
b12x mean was 709 µs per layer call.

On the same layer, unioning the selected channels for rows routed to the
same expert covered 14.5% of its channels on average. From the packed addresses,
that selection touched 57.6% of the up and down weight sectors for those
experts. These are address-coverage estimates. The gate projection still reads
all its weights. Choosing sector-aligned groups of 32 channels by summed squared
score at the same quota raised the mean relative error on the rank-local routed
output from 45.1% to 72.9%. Group selection was an offline calculation. The
256-token code continuation can use the entire allowance on reasoning, so it
measures generation speed and acceptance separately from a completed code edit.
[Park et al., p. 9](https://arxiv.org/pdf/2605.08575#page=9) ·
[Lan, pp. 3–6](https://arxiv.org/pdf/2610.34440#page=5)

| Rank-local routed operation, µs | Layer 6 | Layer 20 | Layer 36 |
| --- | ---: | ---: | ---: |
| Exact b12x | 643 | 709 | 756 |
| Gate, 12.5%, serving layout | 577 | 619 | 618 |
| Gate, 25%, serving layout | 639 | 698 | 689 |
| Gate × up, 25%, serving layout | 679 | 739 | 770 |
| Sketch and recovery, 12.5%, serving layout | 1,172 | 1,243 | 1,261 |

The reader entries are medians of timed CUDA graph replays over the twelve
inputs. The b12x entries are means of the timings for the twelve routes.
In a separate layer-20 profile, the sketch screen itself took 580 µs per
pass. Computing the selected gate/up projection took another 321 µs, and
down took 190 µs. This isolates a compute cost even at full residency.
On the first serving start for each reader, two prefix-attached continuations
gave the following pooled engine decode rates. Rates include misses after
prefix attachment. The automatic tool tasks required the correct tool,
arguments and answer. They used deterministic tool-result fixtures. The
directory exchanges separately used a live filesystem listing.

| Four-row reader | Prose tok/s | Code tok/s | Prose accepted/pass | Code accepted/pass | Automatic tools |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate 12.5% | 36.62 | 43.23 | 1.45 | 1.90 | 4/4 |
| Gate 25% | 36.47 | 44.74 | 1.58 | 2.19 | 4/4 |
| Gate × up 25% | 33.33 | 38.51 | 1.41 | 1.78 | 4/4 |
| Sketch and recovery 12.5% | 18.38 | 16.98 | 1.75 | 1.97 | 3/4 |

The sketch run read 12.95 and 6.76 GiB per rank on those two code requests.
Its failed automatic task chose the right search tool but exhausted its
256-token allowance before returning the requested count. Both gate settings
and product ranking reached zero reads on the second code repeat. These
arms used diagnostic executable `2b45204764e4` and reader `2767aaba53ff`,
built from source base `9771204e` with experimental hooks.

Against the kernel checklist, the grouped reader consumes the existing packed
weights, reuses a fragment across all rows routed to that expert, pipelines
vector loads, and retains the shared-expert stream and graph replay. Scores
and selection stay on the GPU. The exact path still handles prefill and the
other target widths. The miss path snapshots the selected experts into bounded
scratch before scoring the whole top-six set. The checkpoint comparison,
sector estimates and component timings cover four-row TP2 verification.
Compiled occupancy and spill counters for each reader variant, other
concurrencies and general model quality were not measured.

#### Warm repeats and completed tool workloads

The two gate readers and a fresh build of the exact dense-split configuration
each ran in a second deployment. The table pools the last three consecutive
read-free continuations per workload on both ranks, after a 24,448-token
prefix attach. Directory times average eight read-free two-turn exchanges.
Repair times sum engine decode time over four three-turn tasks. All three
configurations read the requested source, selected the correct repair ID and
returned the tool receipt on all four tasks.

| Four-row verification | Prose tok/s | Prose ms/pass | Code tok/s | Code ms/pass | Directory decode ms | Four repairs decode s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact, dense split | 41.56 | 65.97 | 42.66 | 65.69 | 889 | 23.74 |
| Gate 12.5% | 39.05 | 62.79 | 46.00 | 62.99 | 857 | 22.60 |
| Gate 25% | 38.98 | 66.08 | 48.31 | 65.99 | 896 | 28.57 |

The 25% reader accepted 2.19 code predictions/pass against 1.80 for exact,
with similar cycle times. At 12.5%, the shorter cycle helped both the code
continuation and directory exchanges. Prose accepted 1.45 predictions/pass
at 12.5%, 1.58 at 25%, and 1.74 with exact verification. On the separate
automatic tool-choice cases all three chose the correct tool and arguments
four times. The exact model returned `7 matches` when asked to return only
`7`, giving it three of four strict final answers. Both gate readers returned
all four in the requested format.

The completed-code task used a 24k tool-enabled conversation and required a
three-entry LRU function. A separate Python process checked hit promotion,
eviction, falsy values, number of loader calls and preservation on a loader
exception. With a 1,536-token limit the exact model produced a passing function
in 653 tokens (19.76 s engine decode, 36.24 s wall). Both approximate readers
exhausted the limit. A 4,096-token retry let the 25% reader finish in 1,776
tokens and 58.12 s of decode. The retry reused a prefix. Engine decode time
and generated tokens are the comparable figures across the two budgets.
At 12.5% the longer retry consumed all 4,096 tokens and 130.01 s of decode
without emitting the function.

For the two-Spark MXFP4 deployment, the selected configuration uses exact
b12x routed experts, paired CSA2 output groups, the TP2 output split and
four-warp Engram verification. The release build from `411476b8` has
SHA-256 `91170e2e77c33dba62b581689ca38365afb1a44450e049252e81d5486b3ed70e`.
The second deployment used this executable on both ranks with matching
operation streams and zero process swap. The approximate alternatives won
individual throughput and repair probes. Exact execution returned a working
implementation sooner on the longer coding job and led on the prose probe.
These are concurrency-one diagnostics on a small task set.

## Testing the two-Spark forum recipe on TP2

These diagnostics use the 1017 MXFP4 checkpoint, top-six target routing and
TP2 rank dimensions of 5,120 hidden by 1,152 intermediate channels. The
four-row route samples come from actual verification. Input activations are
synthetic in the single-layer kernel tests. The serving tests exercise the
complete two-rank model. [Forum recipe](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438/6)

### Lossless scales and the FP8-activation expert kernel

For all 384 routed expert shards on layer 10 of each rank, one base exponent
per matrix row plus four-bit differences reconstructed all scale bytes.
The largest observed delta was three. Each shard occupied 9,131,264 bytes
versus 9,400,320 bytes in DGPP's layout. Keeping 11,071 shards in that
representation would free 2.77 GiB per Spark. A tightly packed arena of
the current size would fit 11,397 shards (+326). Aligning each shard to a
64 KiB page instead would fit 11,343 (+272). The current allocator still
uses the original slots.

A replay using the residency planner on the larger capacity reduced decode
misses from 5,195 to 4,443 and prefill misses from 15,715 to 15,093 on one
saved demand trace. The other trace gave identical miss counts at both
capacities. These are simulated admission decisions with fixed demand.

The authors' patched TileLang compiler generated the SM121 FP8 × FP4
instructions. The captured chain included GPU dispatch, input quantization,
gate/up, SwiGLU and quantization, down and combination. Packing and expanding
scales inside this chain preserved every output bit against its byte-scale
control on all tested routes.

| One layer, resident experts | DGPP b12x W4A16 | TileLang W4A8 byte scales | TileLang W4A8 packed scales |
|---|---:|---:|---:|
| Four verification rows, median of 33 common routes | 705 µs | 697 µs | 665 µs |
| One recorded 4,096-token prefill, 328 distinct experts | — | 23.12 ms | 22.89 ms |

The packed chain beat byte scales on 33/33 verification routes, by a median
4.60% paired by route. Its median paired advantage over b12x was 3.51%.
Both backends used rank 1's layer-10 weights and matching route IDs. The
b12x measurement begins after its route preparation and ends at routed
output slots. TileLang performs dispatch and combination inside its timer.
The activation inputs differed between backends. The bitwise check tests
the scale encoding within TileLang. A full-model W4A8 verification comparison
would also need to measure the effects of activation rounding on routing,
draft acceptance and completed responses. DGPP's resident prefill and
verification readers would both need to consume the smaller layout before
the projected capacity gain could be realized.
[Scale packing](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438/30)

### Starting gate/up while down weights are in flight

Both local SSDs read the same expert IDs in alternating control and overlap
trials with eight workers, 512 KiB direct reads and mapped buffers. Each pair
used the same full expert bytes and produced identical output. The experiment
repacked one real layer by projection on each node and issued gate/up first
in both arms. The intervention was when GPU computation could begin.

| Miss wave | Spark 1 wait for all → early gate/up | Spark 2 wait for all → early gate/up |
|---|---:|---:|
| One expert, one row | 2.282 → 2.206 ms | 2.378 → 2.166 ms |
| Six experts, 32 rows per expert | 11.272 → 10.537 ms | 11.312 → 10.756 ms |
| Six experts, 64 rows per expert | 10.963 → 10.276 ms | 11.317 → 10.852 ms |

Each entry is the median of 60 timed waves from submission through down
completion on that node. All twelve host/shape combinations improved at the
median. This measures exposed work within a miss wave. Full-service savings
depend on which rank is on the critical path, existing overlap and the number
of exposed misses. The deployed store interleaves the three projections per
chunk. It would need a suitable read layout and split readiness in the
production scheduler to exploit this result.
[Two-rank prototype](https://github.com/christopherowen/spark-ds41f/pull/981)

### Prompt-copy proposals in eager DSpark serving

We compared three lookup rules at DSpark depth three. The first replaced
proposals at the existing width. The second could also append up to two
tokens on a strong match. The selected rule preserves all three DSpark
predictions and appends up to two tokens only when the history matches
the entire prediction. This recovers the short directory call that took an
extra pass when lookup could overwrite a correct DSpark token.
[CopyDrafter workload](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438/138)

Median concurrency-one **engine decode tokens/s** across the final three
consecutive repeats with zero expert-file reads on both ranks:

| Completed task | Base DSpark | Same-width replacement | Replace and extend | Base DSpark on final build | Extend on full agreement |
|---|---:|---:|---:|---:|---:|
| Return an exact edited file | 59.46 | 58.63 | 73.05 | 59.43 | 73.05 |
| Write the complete edit after a read tool result | 58.50 | 57.61 | 67.77 | 58.50 | 68.76 |
| Call the directory tool | 60.08 | 54.84 | 54.26 | 60.13 | 59.56 |
| Answer the prose question | 37.23 | 37.23 | 37.14 | 37.19 | 37.24 |

All four requests used automatic tool choice and were scored through
completion, including finish reason, file contents and tool name/arguments.
The final build returned the same decoded messages with lookup on and
off, excluding generated tool-call IDs. On the file and write tasks,
verification passes fell from 58 to 40 and from 71 to 53. The selected
rule used 11 passes on the directory call and 26 on prose in both arms.
The first three columns used prototype binary SHA-256
`2bb837026200b337c4eb3de6b377af124364fe58eae4fe9a0cc081f9bf8b4852`
based on `b972cd84`. The last two used the same `dcda4609` source and binary
`ee5fabad912d0948d580472cc2fb100e320e86bb7491bd7e73f57c96978c27b2`.
Each configuration had zero process swap on both ranks and matching rank
operation streams. Earlier probes with an inherited forced tool choice
were discarded from this comparison.

On a separate 24,292-token agent prompt, the first prefill with zero cached
tokens took 21.38 s with lookup off and 21.39 s with the selected rule.
The real directory listing was fed back to the model. Both configurations
completed the call and follow-up, named correct directories and produced
the same final text. On warm repeats the median combined engine decode was
1.442 versus 1.437 s. Service-level wall time summed over both HTTP requests
was 2.100 versus 2.093 s. This exchange reused 24,192 prefix tokens on the
call and 24,448 on the reply. A further simultaneous edit and directory call
ran in two active slots and both passed their output checks.

The MXFP4 two-Spark template selects `lookup_draft: true`, `lookup_tail: 2`
and fixed `mtp_depth: 3`, with exact b12x verification. The serving evidence
covers greedy requests, four short tasks and one long-context tool exchange.
Broader throughput by concurrency and full-model projection-level SSD
readiness remain unmeasured.

## Lossless MXFP4 scale packing on the two Sparks

The entire routed-expert SSD image on each rank was checked, covering
276,480 chunk records per rank. E8M0 scale bytes ranged from 117 to 127 on
rank 0 and 117 to 126 on rank 1. All have the same upper nibble. Storing
the lower nibble cuts routed-expert storage from 4.25 to 4.125 bits per
weight. Each local full shard occupies 9,123,840 bytes, down from
9,400,320. The measured cache allocation changed from 199,291 chunks of
522,240 bytes to 205,329 chunks of 506,880 bytes, including a separate
522,240-byte conversion buffer in the same budget. In groups of 18 this
gives 11,407 complete shards per rank versus 11,071. The ranks hold
complementary halves of the same experts.

A captured b12x W4A16 verification replay used the same rank-1 layer-10
weights, four input rows, top-six routes and synthetic BF16 activations
for both layouts. Across 34 recorded route groups all routed FP32 output
slots were bit-identical. Median times were 692 µs with byte scales and
660 µs with packed scales. Packed won 31/34 routes with a median paired
latency reduction of 4.25%. The single-layer paging test also compared
complete outputs under whole-expert and fractional policies, including
cold reconstruction and prepared hits. It covered both W4A16 prefill
and the deployment's native MX prefill with fused gate/up.

The serving screen used the same two-rank checkpoint, local SSD images,
expert budget, 1M KV capacity and fixed-depth-three DSpark with a lookup
tail of two. The byte-scale control and initial packed reader both used
source `d8e3a350`, executable SHA-256
`a75d3140f369e2b1d827aaee61682d5b400d2ac09f293757a6c00601e77197d7`.
The selected prefill reader shares its permuted-channel address and nibble
selection across adjacent K32 groups. It used source `237461c8`, executable
SHA-256 `cc76ae9259995db5e2bf70573f721b223a7a2b6c0c687e6647c62ba575c1e7e8`.

| Two-Spark C1 diagnostic | Byte scales | Packed scales | Packed with paired prefill reads |
|---|---:|---:|---:|
| Cold 24,292-token prefill, seconds | 34.81 | 35.24 | 35.35 |
| Resident 24,292-token prefill, seconds | 10.377 | 10.482 | 10.370 |
| Resident 310-token prefill, seconds | 0.467 | 0.481 | 0.461 |
| Return to the long prompt after the short workload, prefill seconds | 14.24 | 12.89 | 12.79 |
| Expert-file reads per rank on that returning request, GiB | 23.04 | 14.92 | 14.92 |
| Directory call on the long prompt, engine decode tok/s | 55.33 | 55.87 | 55.84 |
| Answer after the directory result, engine decode tok/s | 57.20 | 57.57 | 57.70 |
| Complete a short Python function, engine decode tok/s | 59.16 | 59.51 | 59.73 |

The long prompt had zero prefix hits on all four full presentations in each
boot. Each initial request read 95.61 GiB of expert records per rank. The
following three presentations read zero on both ranks. Short resident prefill
uses the median of six complete prompt evaluations. The returning long
request used zero cached prompt tokens and generated the same 44-token tool
call in all three boots. The 35.3% read reduction on that request came from
retaining more experts across the intervening workload. Each SSD read still
has the original record size.

Decode rates use generated tokens after the first token divided by engine
decode-step time. Directory call, reply and code medians use three, five
and five expert-read-free requests, respectively. The follow-ups reused
24,320 prefix tokens. Greedy tool calls, arguments, final answers, code
and finish reasons matched across layouts, excluding generated tool-call IDs.
Both ranks had zero sampled process swap and matching operation streams
within every boot. Each configuration had one cold start. Differences below
one percent in this short C1 comparison have no established error bound.

## FP8 attention output into the TP2 verification projection

`DGPP_DSV41_MX_WOA_DECODE=1` lets the target attention finish emit E4M3
codes with one E8M0 scale per 32 channels. Normalization, the BF16 rounding
before inverse rotation, and the BF16 rounding of the rotated pairs precede
quantization. The scale uses DeepSeek's minimum absolute maximum of 0.0001.
The producer can also write the original BF16 attention output for diagnostics.
The grouped `wo_a` projection reads the codes directly from the existing GEMM
workspace along with the checkpoint's FP8 weights and expanded grid scales.
Its matrix multiply accumulates in FP32 and emits BF16 for `wo_b`.
[DeepSeek reference attention](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/inference/model.py#L781-L791)

The dispatch covers one through eight target decode rows, four local groups,
4,096 input channels and 1,024 output channels per group. Two warps stream
opposite halves of K for the same 16 output channels. The block double-buffers
512-byte weight-row windows and prefetches the next window into L2. The 256
blocks cover all four groups in one launch. No additional expert-cache or
persistent weight allocation is required. Prefill and the draft block use
their existing projections.

`dsv41_verify_dense_bench --woa-handoff` times attention finish plus all four
groups in both CUDA graphs, rotating the weight working set beyond L2.
It checks the emitted codes and scales against a CPU implementation, including
inverse rotation and padding, and checks the projection against an FP64 sum
of the decoded operands. The final four-row replay measured 93.14 µs for
the paired BF16-input path and 81.93 µs for native FP8. FP8 won all 12
alternating pairs. The median paired reduction was 9.06%. The normalized
projection error against the quantized reference was 0.001725, including
rounding the output to BF16.

A same-binary, concurrency-one serving diagnostic toggled only this path on
the packed-scale MXFP4 TP2 deployment. Both arms enabled the per-pass DSpark
timer. The directory and short-code follow-ups attached 24,320 tokens of
prefix and used seven zero-expert-read repetitions per arm. The 256-token
probes attached 24,448 and used four repetitions. Both ranks reported zero
process swap and had matching operation streams within each arm.

| Warm workload | Mean verify ms/pass, BF16 → FP8 | Accepted predictions/pass, BF16 → FP8 | Median engine decode tok/s, BF16 → FP8 |
|---|---:|---:|---:|
| Directory result | 58.25 → 58.05 | 2.83 → 2.83 | 57.93 → 57.97 |
| Complete Python clamp | 59.77 → 59.10 | 3.00 → 3.00 | 59.73 → 59.96 |
| Prose, 256-token cap | 58.25 → 58.00 | 1.74 → 1.75 | 42.09 → 42.23 |
| Code review, 256-token cap | 58.76 → 58.02 | 1.78 → 1.93 | 42.21 → 45.11 |

The two short tasks produced identical greedy answers, and the clamp passed
tests for interior, exterior and boundary values. The longer continuations
changed. Code review used 87 speculative passes with FP8 and 92 with BF16.
Both four- and six-row passes occur when the lookup draft extends a proposal.
The measured sub-percent throughput differences on the short tasks have
between-start uncertainty. The measured binary used source base `c183bb16` with the FP8 handoff later committed in `7194602d`,
binary SHA-256 `f20207e86fcd34286d9dabe6374f11433429ece62c3cd82363223c9a561c324e`.

For a completed bounded-LRU implementation, the original projection generated
502 tokens and took 10.80 s of decode. FP8 generated 658 tokens and took
14.14 s. Both implementations passed checks for hit promotion, eviction,
falsy values and a loader exception. These requests attached 24,448 prefix
tokens and incurred zero expert-file reads on either rank. Mean target
verification fell from 58.90 to 57.91 ms/pass. Producing the longer answer
took 218 passes compared with the original 164.

Raising the code-review output cap to 4,096 let both arms finish and pass
executable checks for cache hits, loader calls, falsy values and the
capacity bound. Across three repeat requests per arm with 24,448 prefix
tokens attached, median engine decode was 42.22 s for BF16 (1,691 tokens,
40.03 tok/s) and 53.73 s for FP8 (2,150 tokens, 39.99 tok/s). Reasoning
accounted for 422 of the 459 extra tokens. Median expert-file reads were
11.53 and 14.02 GiB per rank, respectively. On these two completed coding
tasks the BF16-input projection returned passing work sooner because its
greedy completions were shorter.

### Optimized FP8 verification on two Sparks

The TP2 MXFP4 streaming template selects native FP8 for the target attention
output projection. Attention finish keeps each lane's four values in registers
through rotation, BF16 rounding and K32 quantization. Each projection warp
stages 16 weight rows using asynchronous 512-byte copies, alternates between
two shared-memory buffers, and uses `ldmatrix.x4` to feed the tensor cores.
It reads four adjacent scales together. At the last K window it directs its
L2 prefetch to the first window of the output tile 96 blocks ahead. These
changes preserve the original FP8 path's quantization and accumulation order.

In 20 interleaved graph timing samples per shape on GB10, median attention finish
plus projection time changed as follows. Each replay rotated seven copies of
the weights. The percentages are medians of paired reductions.

| Target rows | Main + window splits | Original FP8, µs | Optimized FP8, µs | Reduction |
|---:|---:|---:|---:|---:|
| 4 | 0 + 1 | 74.43 | 69.62 | 6.48% |
| 4 | 4 + 1 | 75.38 | 70.41 | 6.60% |
| 6 | 4 + 1 | 77.09 | 71.66 | 7.07% |

The release build reproduced the optimized timings at 69.52, 70.25 and
72.41 µs, respectively. For four rows with four main splits and one window
split, its BF16-input control took 88.28 µs. Codes and scales matched the
CPU quantizer, and the normalized RMS projection error against decoded
operands summed in FP64 was 0.00163 for this shape.

In a full-model concurrency-one diagnostic, each FP8 build ran the same
directory tool round trip followed by the bounded-LRU coding task. We timed
the last four consecutive LRU completions per build with 24,448 cached prefix
tokens and zero expert-file reads on both ranks. Every completion generated the
same 658 tokens, including 389 reasoning tokens, passed the executable checks,
and accepted 439 predictions over 218 speculative passes. The rank operation
streams matched within each run and both ranks reported zero process swap.

| Warm coding diagnostic | Original FP8 | Optimized FP8 |
|---|---:|---:|
| Mean target verification, all passes | 57.972 ms | 57.768 ms |
| Mean four-row verification, 852 passes per build | 57.624 ms | 57.412 ms |
| Mean six-row verification, 20 passes per build | 72.819 ms | 72.951 ms |
| Median engine decode | 46.44 tok/s | 46.57 tok/s |
| Engine decode range over four completions | 46.34–46.51 tok/s | 46.54–46.65 tok/s |
| Median engine decode time | 14.148 s | 14.107 s |
| Median request wall time | 14.499 s | 14.464 s |

The 0.29% engine throughput gain measures this completed answer on one
server start per build. Four-row passes accounted for 213 of 218 passes per
answer. The six-row serving sample was small and its mean rose 0.13 ms.
The optimized binary used revision `781f2da9a57f`, SHA-256
`1bd1025c27bd742b7d824fa783dc90140b3d7e4dcef29b1a8cd238eb42e1145b`.
The per-pass timer was enabled for both diagnostic runs and disabled for
normal serving.

### Activation lookahead, diagnostic writes and weight pipeline depth

On the TP2 FP8 decode path, ordinary serving emits the packed attention
output alone. Layer diagnostics can request both outputs. Omitting the
BF16 copy saves 128 KiB of stores per layer per rank at four rows and
192 KiB at six. The normalization and rotation still round at the same
points. At four main splits and one window split, 32 alternating
producer-only timing pairs measured median attention finish times of
3.326 → 3.212 µs at four rows and 3.711 → 3.615 µs at six.

The projection now skips activation code and scale loads for lanes beyond
the supplied token row count. A padding position inside that count receives
zeros from attention finish. All lanes still participate in the matrix multiply.
With eight rows, the kernel issues the current 512-channel activation
window's loads before waiting for the weight copy. The GB10 disassembly
shows predicated global loads at four rows and 36 activation load
instructions before the first weight wait at eight.

The following comparisons use the same four main splits and one window
split. Times are medians in µs for attention finish plus projection. Each
pair shares its run, weight working set and producer setting. The activation
comparisons use the known K of 4,096 in both arms.

| Trial | Four rows | Six rows | Eight rows |
|---|---:|---:|---:|
| Load the current activation window before the weight wait | 72.14 → 72.04 | 73.11 → 73.05 | 72.27 → 72.15 |
| Specialize the four- and six-row loads to skip unused lanes | 72.14 → 72.02 | 73.11 → 73.03 | All lanes have live rows |
| Three 512-byte weight stages, one resident block per SM | 70.59 → 76.70 | 71.03 → 77.35 | 71.64 → 77.84 |
| Three compact 512-byte stages, two resident blocks per SM | 72.66 → 77.40 | 72.02 → 77.48 | 71.55 → 77.26 |

In the three-stage tests, the third buffer let each warp issue two future
weight windows. Recovering two resident blocks still left the compact
kernel slower. Three stages of 256 bytes fit three blocks per SM and took
74.94 µs at four rows, versus 70.49 µs for the two-stage 512-byte control.
The register lookahead trials also covered loading the next full window
and the next 128 channels. Copying activations into shared memory alongside
weights took 76.40 µs at eight rows versus 71.55 µs for its control.
The selected kernel uses two weight stages at every width.

In the final comparison against the deployed FP8 kernel, each shape ran
60 interleaved timing pairs with seven rotating weight copies. At four,
six and eight rows the median paired reductions were 0.41%, 0.32% and
0.47%. The FP8 codes, scales and BF16 projection outputs matched exactly
at every supported width from one through eight, including a padded
position where the shape allowed it. The production binary used revision
`1165a6626080`, SHA-256
`d697f037cbb5bab6be8da49d70a1f27197957a874b4289015dc414fca201e98f`.

The first candidate server start generated a different passing LRU answer.
Its greedy token stream diverged at the third token and it accepted 479
predictions across 215 passes, producing 695 tokens at a median 49.90
engine decode tok/s. A second start of the very same binary reproduced
the incumbent's 658-token answer. The cause of the between-start variation
is unresolved. In a separate diagnostic, both kernels executed on the
same real inputs on each rank throughout the tool and coding requests.
The FP8 codes, scales, BF16 projections and BF16-only attention finish
outputs agreed byte for byte. Feeding either path's results onward
produced the 658-token answer in that diagnostic.

For the matching-answer comparison, each release produced the same text and
reasoning and accepted 439 predictions over 218 passes. Each column uses 12
warm concurrency-one requests from one server start, 24,448 cached prompt
tokens, zero expert-file reads and zero process swap on both ranks. Both
ranks recorded the same operation stream within each run. The DSpark
per-pass timer was enabled on both builds.

| Warm 658-token coding task | Previous FP8 | Selected FP8 |
|---|---:|---:|
| Median engine decode tok/s | 46.646 | 46.590 |
| Engine decode tok/s range | 46.535–46.763 | 46.476–46.686 |
| Median engine decode seconds | 14.085 | 14.102 |
| Mean target verification ms/pass | 57.682 | 57.766 |
| Median request wall seconds | 14.442 | 14.454 |

The selected kernel reduced the measured handoff time and memory traffic.
Matched-answer serving throughput was 0.12% lower in this comparison;
overlapping run ranges and variation across starts leave the effect on
full-model decode inconclusive. Normal serving uses the selected binary
with per-pass diagnostic timing disabled.
