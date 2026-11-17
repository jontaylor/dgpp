# MXFP4 attention, shared-expert overlap and compact verification

These changes follow the [warm execution profile](mxfp4-warm-execution-profile.md).
They address three specific costs without changing the expert-cache budget:
attention partial results, exposed shared-expert prefill work, and host output
materialization during speculative verification.

## Attention split screen

`DGPP_CSA2_PREFILL_SPLITS=1|2|4` controls compressed-source prefill attention.
The default remains four. Scratch is reserved for the maximum split count,
so this switch isolates execution rather than reallocating cache capacity.
Selection and attention oracle tests cover one and two splits, subsequent
decode, and tensor-parallel geometries.

The focused two-Spark screen on 17 November 2026 retained four splits.
The split-screen binary SHA-256 was
`cc765f6b4146dc83374c4b117c70a2910549f2a9dd5d21031fcab8735c2a58f6`.
Two splits slightly shortened the resident 2,209-token prefill but slowed the
warmed 24,317-token prefill. One split did not improve the longer request.
Fewer partial vectors do not guarantee faster attention: less parallelism
and more work within each block can outweigh the smaller merge.
The changed accumulation order also changed the short prompt's generated
wording and speculative acceptance. Decode rates from those different
transcripts are not a clean isolated kernel comparison.

## Shared-expert prefill overlap

`DGPP_EXPERT_SHARED_PREFILL_EARLY=1` starts the FP8 shared expert on its
existing side stream before routing and the host cache decision. The three
projections and SwiGLU use the existing shared tail of the activation scratch,
disjoint from routed waves. Accumulation waits for the shared completion event.
The routed waves omit shared execution and activation, preventing duplicate
work and concurrent writes. This path applies to native MXFP4 paged MMA
prefill with compatible FP8 shared weights. Small-row b12x execution and the
existing decode overlap remain on their established paths.

The screen compared the same binary with this switch off and on (SHA-256
`ca09ac8aa85151b541101ce81dc716020b7d185a21a52ea9b0867c4d0e549a33`). Resident
short prefill fell from about 1.39 to 1.37 seconds; final warmed 24k prefill
fell from 12.43 to 12.10 seconds, about 2.6%. Decode was essentially unchanged.
Compared final requests had identical messages, speculative acceptance and
expert read volume: zero reads for the resident short request and 5.159 GiB
per rank for the longer request. The native MXFP4 paging fixture, including
pressure and repeated prefill, passed its existing bitwise canonical-output
checks with early shared execution enabled. CTest exposes this arm as
`glm_moe_test_mxfp4_early_shared`.

## Compact greedy verification

`DGPP_DSV41_GPU_CANDIDATES=1` makes the single-request eager DSpark verifier
reduce each rank's target logits to exact local candidates on the GPU.
It copies those candidates to the CPU and keeps the existing distributed
row-pick collective. It omits full logits and hidden-state host outputs for
this call; the device hidden-state storage and rollback snapshots remain.

Tool grammars retain their lazy masking rule: an allowed unmasked local
winner is already the masked winner. If the grammar forbids it, only that
row's target logits are downloaded and the existing host mask is applied.
This happens before rollback or drafting can reuse the logits buffer.
Temperature sampling, penalties, logprob reporting, bias, multi-request
verification and diagnostic model calls retain full outputs. Draft-head
output materialization is also unchanged.

Persistent candidate scratch is charged to the memory plan. At eight decode
rows its logical payload is 268 device bytes and 96 pinned bytes per rank;
allocator and page overhead are additional. It does not
release the existing host mirrors, which remain necessary for other paths.
The compact fixture checks exact candidate IDs and logits, exact downloaded
fallback rows, omitted host outputs, session position and rollback. The
existing device-picker fixture covers canonical tie ordering and sharded
vocabulary offsets.

The same-binary on/off screen (SHA-256
`f2fd9a2406e2f46efb01af8bc91fecacbb0afbe5d68c03f18b826201939fac18`)
showed no material end-to-end gain. Resident prose decode was 40.18 tokens/s
in both arms; resident code moved from 41.00 to 41.21 tokens/s. Short tool
responses were slightly slower with compact selection. Every compared tool,
prose and code response, acceptance count and expert-read byte count matched.
The first prose/code continuations still incurred reads; their repeated
resident requests incurred none. The code probe had a 512-token output cap,
so this is an execution comparison, not a completed-code quality evaluation.

The shipped MXFP4 example and the Spark service enable early shared prefill,
retain four attention splits and leave compact candidates disabled. The
compact path remains available for future comparisons, rather than treating
reduced host traffic as a demonstrated throughput improvement.

## Measurement scope

These are focused optimization screens, not rows for the deployment
benchmark table. Each arm restarted the same two-rank deployment, used the
same warm-up order and tool prompts, and disabled prefix reuse for the
2,209- and 24,317-token probes. Both ranks were checked for zero process swap
and matching operation streams. Binary hashes, source revision and the
working diff hash were recorded. No remote build or other serving world ran
during timing. Cold-start requests were retained but did not determine the
warm execution choice. The later prose and code continuations kept tools
enabled and used identical supplied tool results across arms.

A complete deployment campaign, concurrency scaling and a wider quality
evaluation are outside this focused comparison. These changes do not address
the larger routed-expert or dense-projection kernel costs identified by the
profile.


The subsequent [kernel alternatives screen](mxfp4-kernel-options.md) tested
128-row experts, matched b12x formats, native W4A8 experts, native dense FP8
and four-way attention merging. None displaced this serving configuration.
