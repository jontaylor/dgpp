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
