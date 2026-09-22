# MiMo-V2.6-Flash-RL initial checkpoint contract

Base: `3ee00e2`, branch `codex/mimo2.6-flash-rl`. This record covers host
configuration and tensor binding only, with no throughput or numerical claims.

The Hugging Face revision inspected was
`3b38d063180c3e4aed9691fdc735f3d10b266ee4`. Configuration, reference Python,
the full tensor index and a range-read header from
`model_pp0_ep0_shard0.safetensors` were inspected on the workstation. The
header contains all main-layer attention metadata and a subset of experts.
No reference Python was executed and no weight payloads were downloaded.

Findings incorporated into the implementation:

- 72,574 main-backbone tensors, 48 conventional MTP tensors, 459 multimodal
  tensors, and 65 shards in the main index.
- Global QKV uses FP8 `[13568,4096]` payloads with F32 `[108,32]` scales.
  This is not a standard contiguous 128-row scale grid; the exact packing
  remains a loader gate. All nine global layers agree.
- Sliding QKV uses `[14848,4096]` payloads and `[116,32]` scales.
- Expert payloads and their scale tensors are both stored as U8. Treating
  those scale bytes as NVFP4 e4m3 scales would be an unjustified assumption.
- Three conventional MTP layers exist under `model.mtp.layers`; the separate
  five-layer DFlash model is an additional speculation path.

Validation:

- Four native host test groups passed with `-Wall -Wextra -Werror`, then
  passed after formatting with AddressSanitizer and UndefinedBehaviorSanitizer.
  Coverage includes 13 incompatible-config mutations, independent observed
  tensor-header fixtures, binding failures and KV/TP geometry.
- The native CLI accepted the complete published index: 72,574 backbone
  names, 48 unvalidated MTP entries, 459 unvalidated multimodal entries.
- CLI probes rejected a missing embedding-head entry, an unknown tensor,
  incorrect source TP metadata and a parent-relative shard path.
- The `mimo_checkpoint_check` CMake target cross-built for ARM64/Spark.
  Docker's default bridge was unavailable; the build succeeded using the
  existing cross image with `--network none`, without host network changes.
- Changed C++ files passed clang-format checking; `git diff --check` passed.

The full repository test suite, full downloaded-checkpoint header check,
GPU execution, TP2 numerics and serving remain untested. No Spark process,
download, live configuration or deployment was modified. See the
[port plan](../../docs/mimo26_flash_rl_plan.md) for the next gates.

## Host weight loading continuation

Spark-1 was confirmed at `192.168.0.171`; its pinned snapshot is complete.
All 65 downloaded shards were inspected, and all 72,574 backbone tensor
headers matched the binding, including dtypes and shapes. Auxiliary counts
remain 48 MTP and 459 multimodal tensors; those are not semantically validated.

The QKV scale-layout question is resolved: payloads are four `[Q,K,V]` chunks,
each with its own scale grid. The pinned vLLM and SGLang sources linked in
the port plan agree. Global chunks have 3392 payload rows and 27 scale rows;
sliding chunks have 3712 and 29. The host loader gathers canonical rank-local
Q/K/V after source-scale dequantization, returning BF16 without requantization.

New host code provides QKV unpacking, ordinary block-FP8 slices, packed
MXFP4 expert slices and a checkpoint reader with owned outputs. TP slices
are tested across source chunk, scale block and expert partition boundaries.
Synthetic separate-shard tests exercise mmap lifetime and both row/column
expert sharding. The initial TP2 placement estimates 87,088,227,776 weight
bytes per rank, excluding all runtime and allocator overhead.

Validation of this continuation:

- Ten host test groups passed under ASan/UBSan with warnings treated as errors.
- Every QKV row is mapped exactly once at TP1/2/4, with head order checked
  independently. Invalid ranges, types and truncated buffers are rejected.
- Actual checkpoint goldens: 44 QKV windows (704 BF16 values), four expert
  slices (256 values); all pass exact bit comparison at each TP geometry.
- The expert comparison exposed signed-zero loss in the Python golden
  generator's integer e2m1 zero. Corrected to floating zero; no engine
  change was needed for that discrepancy.
- Both checker targets cross-built for ARM64 and ran on Spark-1's CPU.
  The header checker passed on the full downloaded snapshot. The weight
  checker exercised the real file-backed reader and reported 44 QKV rows
  and four expert slices passing separately at TP1, TP2 and TP4.
- These checks used small temporary executables/fixtures under `/tmp` and
  read checkpoint files. They did not allocate GPU memory or change serving.

These are host-storage checks, not tensor-parallel GPU or numerical-forward
validation. GPU upload, attention kernels, assembled forward, distributed
execution and HTTP serving remain pending.

## Initial attention implementation

Added a CPU attention oracle and CUDA append/attention kernels for QK=192,
V=128 GQA. Append applies 64-dimensional half-split RoPE, scales values by
0.707, and writes request-private linear or 128-token ring caches. Negative
positions are padding. Each call appends one token per request; initial
prefill must iterate in order. CUDA position errors are reported in device
status flags without mutating the affected cache row.

The attention kernel is deliberately an eager-numerics baseline with three
score passes and bounded shared storage. It includes BF16 score/probability
rounding and the zero-value attention sink. It has no internal allocation,
copy or stream synchronization. It is not an optimized prefill kernel or
an assembled model layer.

Independent goldens were generated using CPU PyTorch 2.14.0 in an isolated
temporary environment. The generator uses published tensor operations and
does not import checkpoint code. Sixteen cases cover global/SWA attention
around positions 127/128 and 255/256. Six additional RoPE-only cases cover
8191, 32767 and 1048575 at both rotary bases. CPU comparisons pass within
one BF16 ulp and 0.2% relative L2. The high-position tests do not allocate
or validate a full 1M-token attention cache.

Validation of this continuation:

- All fourteen MiMo host test groups passed with ASan/UBSan,
  `-Wall -Wextra -Werror -ffp-contract=off`.
- Analytical tests verify sink normalization, exact window eviction,
  paused-row isolation, slot reset and rejection before host cache mutation.
- `mimo_attn_test` and its CUDA kernels cross-built for ARM64/SM121a.
  An initial missing CUDA math-constants include was corrected before the
  successful build.
- Three CUDA execution groups are ready: global/SWA parity with actual
  TP4 head counts; graph replay over 65 rows with padding; and invalid
  position status/cache preservation. These tests were **not run**.
- Read-only process inspection showed `dgpp-serve` active on Spark-1,
  using about 104,923 MiB at the time of inspection. No GPU test, restart,
  deployment or live configuration change was performed.
- Changed C++/CUDA files pass clang-format checking and `git diff --check`.

The next hardware gate is to run the focused CUDA target and memory/race
checks serially on idle test hardware. GPU parity, actual graph capture and
performance are unverified. GPU weight upload, assembled layers, full-model
logits, TP collectives and serving remain separate unfinished work.

## Authorized two-Spark attention checks

On 2026-09-21 the user granted both ranks for maintenance. With the shared
maintenance lock held, the recorded Qwen deployment was drained and stopped.
Both GPUs had no compute processes before testing. Shutdown operation-stream
MD5s matched (`d5f3b897c4d5d90a090c86bb05d3fef0`).

The ARM64/SM121a `mimo_attn_test` executable had SHA256
`5e0b311950f13bbd8488bb4f2e4f3da104ab91d123690d218b89014eda7fc0c1`.
All three groups passed independently on Spark-1 (`192.168.0.171`) and
Spark-2 (`192.168.0.172`): global/ring reference parity, graph replay with
65 rows and padding, and invalid-position status/cache preservation.
Compute Sanitizer memcheck, racecheck and synccheck each reran all three
groups on each GPU: zero errors, and zero race hazards or warnings.
No tolerance changes or kernel fixes were needed during this hardware gate.

These checks exercise single-GPU kernels with TP4 head geometry, not TP4
execution, two-rank collectives, full-model logits or serving. Attention
performance is unmeasured. GPU weight upload, layer assembly and runtime
integration remain unfinished.

Production was restored using the installed launcher and original deployment
namespace `483e9e6962452f5e`, becoming ready at 22:30:41 UTC. Both running
executables are release `0.1.0+g116d3c375d87`, SHA256
`126c060ca0eff544dba2379f9a38b91c43663557e698b5803a936d4c2224e57b`.
Both resolved rank configurations exactly match the pre-maintenance JSON.
Health and model discovery passed; startup reported 148 prefix-cache slots
and metrics reported 45,760 pool blocks. Three identical thinking-disabled
requests returned `ready`; cached prompt tokens were respectively 0, 24, 24.
Both rank logs show successful request retirement.

Raw test logs and restore evidence are in
`/tmp/dgpp-mimo-gpu-20260921` on the development host. Stopped production logs
and operation streams were preserved in the owned remote temporary test
directories before restart. These temporary paths are operational evidence,
not permanent repository fixtures.


## Ordinary layer/global loading continuation

Added rank-local BF16 reads for every remaining main-text weight role:
attention output projections, norms, sliding sinks, dense gate/up/down,
router weights, embeddings and vocabulary head. The correction bias has a
separate FP32 reader. Dense FP8 dequantization retains source scale blocks;
BF16 copies preserve bits. QKV and packed expert layouts remain separate.
Source shapes/dtypes and rank-local window bounds are validated before copy.

Validation:

- All sixteen host groups pass ASan/UBSan with warnings treated as errors.
- Sparse synthetic shards exercise separate weight/scale mappings, output
  and dense column slicing, norm replication, sink slicing and FP32 bias.
- Manifest-wide ordinary placements cover all TP1/2/4 partitions/replicas.
  Unaligned BF16 copies preserve signed zero, infinities and NaN payload bits;
  malformed payloads and invalid windows/roles are rejected.
- `mimo_weight_check` cross-builds and passes on Spark-1's CPU against the
  actual pinned checkpoint: prior 44 QKV/four expert windows at each world,
  plus 110 ordinary-weight samples giving 110/147/221 rank comparisons at
  TP1/2/4. Replicated samples are checked on every rank placement.
- Goldens come from `mimo_layer_weight_sample.py`, a bounded standard-library
  byte reader that imports no model code and uses no GPU.

This continuation did not stop serving or allocate GPU memory. The reader
now covers all main-text weight roles, but GPU residency, assembled layer
outputs, routed execution, collectives and serving remain unfinished.

## Resident model and initial two-rank service

The later authorized serving phase supersedes the earlier restored-Qwen state:
MiMo now replaces Qwen on port 30001, with the maintenance lock held throughout
installation and acceptance. Implementation remains uncommitted on base
`3ee00e2641dd`, version `0.1.0+g3ee00e2641dd.dirty`.

Added aligned GPU residency, full decoder layers and an eager `SessionModel`
adapter for all 48 layers, with two TP boundary reductions per layer. BF16
QKV/dense weights are unpacked on the host; experts remain packed MXFP4.
FP4 dispatch includes widths 2048 and 4096. Rank-local memory planning reports
81.58 GiB (81.50 device, 0.08 pinned) plus 4 GiB headroom at 32K context.
The checkpoint is complete on both Sparks at the pinned revision above.

Validation before deployment:

- Sixteen MiMo host groups pass ASan/UBSan with warnings treated as errors.
- Real-weight single-GPU dense layer 0, three sequential tokens: maximum
  relative L2 5.889e-6, maximum absolute error 0.00012207; two rows exact.
- Real-weight MoE layer 1: relative L2 0.0009510, 0.0009820, 0.0017127;
  maximum absolute error 0.0078125. Independent CPU PyTorch 2.14.0 oracle
  reads/dequantizes checkpoint tensors without importing model Python.
- Both real-layer Compute Sanitizer memcheck runs report zero errors.
- All ten FP4/MXFP4 GPU regression groups pass on both Sparks.
- Dedicated MiMo NFC/single-digit tokenizer matches all 94 HF tokenizers
  0.23.1 goldens. Existing Qwen tokenizer also passes all 94 cases.
- Sixteen parser groups and all 39 host HTTP tests pass, including generated
  thinking-prefix separation, SSE, folded reasoning and reasoning token counts.

The first full-model build answered arithmetic, Python generation and a
multi-turn recall question correctly; repeated 522-token recall prompts
returned ORCHID across sliding-ring wraps. Its thinking prefix leaked into
content. The final build recognizes an opening generated `<think>` token,
separates reasoning in normal/SSE output and counts its tokens. It also
exposes logical KV admission units so an oversized token budget is rejected
before execution.

### Final deployment and acceptance

Final service became ready at 2026-09-21 23:21:55 UTC:

- Endpoint: `http://192.168.0.171:30001/v1`
- Model: `XiaomiMiMo/MiMo-V2.6-Flash-RL`
- Both ranks loaded 48/48 layers; both running executable SHA256 values are
  `94319e40c3fb5d9df7a5c1ab85d9065ea047e65832071c024ec0a1d5adde6d68`.
- Resolved config SHA256 matches on both ranks:
  `d7e81057a762afddfb51aba5f67677e17639f0666fa19ce8d8f5cd7ec7838840`.
- Deployment namespace: `6e7fcaa12fac5362`.
- Durable rank-0 binary/config:
  `/home/jon/dgpp/experimental/mimo-20260921/{dgpp-serve,deployment.json}`.
- Rank-0 logs: `/home/jon/dgpp/log/deployments/6e7fcaa12fac5362/`;
  rank-1 staging/logs: `/tmp/bus4/deployments/6e7fcaa12fac5362/`.

Final live acceptance passed health/model discovery, arithmetic with thinking
off, and thinking-enabled 7 times 8. The latter returns content
`7 times 8 is **56**.`, reasoning `7 × 8 = 56`, and 10 reasoning tokens;
no literal thinking tags leak into content. SSE returns matching fields,
usage and `[DONE]`. Prompt 15 + max_tokens 32768 returns HTTP 400 with
`context_length_exceeded`. Two simultaneous arithmetic requests both return
the correct answers through the single slot. A forced synthetic weather
function produces a structured `lookup_weather` call with city Paris and
`finish_reason: tool_calls`; no external tool was executed. A final
480-token prompt spanning several ring wraps returns ORCHID. Both ranks
retire the same requests successfully and return to zero live/queued work.

Shared deployment template:
`deploy/cluster_mimo_v2.6_flash_rl_w2.example.json`. On Spark-1, the installed
launcher used for this build is:

```bash
DGPP_ENV_FILE=/home/jon/dgpp/releases/dgpp-0.1.0+gf1b05f544293.dirty/.env \
python3 /home/jon/dgpp/releases/dgpp-0.1.0+g116d3c375d87/scripts/dgpp-cluster up \
  --config /home/jon/dgpp/experimental/mimo-20260921/deployment.json \
  --bin /home/jon/dgpp/experimental/mimo-20260921/dgpp-serve
```

Initial limits: text only, one active request, 32768-token combined budget,
sequential prefill, no decode graphs, MTP/DFlash or prefix cache. Prompt
processing is roughly 20 tokens/s in these short checks; this is not a
controlled performance benchmark. Full-model logit/NLL equivalence remains
unverified. DGPP's weighted MoE partials accumulate in FP32 before the final
BF16 fold, differing from eager per-expert rounding. Coherent HTTP responses
and sampled layer parity do not establish full-model numerical equivalence.

Raw final responses are retained under `/tmp/dgpp-mimo-gpu-20260921/final-*`
on the workstation. These temporary artifacts supplement this record.

## Chunked and parallel prefill continuation

The initial ~20 token/s sequential prefill is superseded by a 128-row eager
prefill path. QKV/output/dense projections use batched GEMMs; routed experts
use the existing grouped MXFP4 tensor-core path. Each layer folds a whole
chunk at its two TP boundaries. Decode remains a single-row walk.

Attention now supports consecutive rows sharing one sequence cache. The
sliding ring retains `128 + chunk_capacity - 1` positions, rather than only
128. This retains the earliest query's entire window while appending the
full chunk, allowing all causal queries to run concurrently. Global queries
share the linear cache and mask by absolute position. The shared-cache API
rejects insufficient ring capacity. Single-token and independent-request
kernel calls retain their existing layout and default behavior.

Validation:

- Four attention GPU groups pass on both Sparks. The new group compares
  chunk queries to sequential CPU attention for global/sliding caches,
  physical-ring wraps, causal bounds, a short final chunk and slot reuse.
- All four groups pass Compute Sanitizer memcheck and racecheck on Spark-1:
  zero errors, zero race hazards/warnings.
- Real-weight dense and MoE layer probes compare all 259 output rows from
  chunk size 128 against chunk size 1. Maximum per-row relative L2 is
  0.000212233 (dense) and 0.001077815 (MoE), maximum absolute errors
  0.00390625 and 0.0078125. Parallel attention outputs are bit-identical to
  the intermediate implementation using batched GEMMs/MoE with serial
  attention. This isolates attention concurrency from GEMM rounding changes.
- The intermediate grouped-MoE layer also passed memcheck at 128-row chunks.
- The layer probe accepts optional CHUNK/TOKENS arguments, and the comparison
  tool accepts raw BF16 reference rows for reproducible chunking regressions.

An intermediate build with batched projections/MoE but serial attention
reduced the same 480-token ORCHID prompt from 23.110 seconds of prefill to
2.836/2.843 seconds (two requests, no cache). It still took 19.986 seconds at
2030 tokens and 197.361 seconds at 8030 tokens. All returned ORCHID. This
exposed serial attention as a remaining long-prompt bottleneck and motivated
the parallel-query implementation above. These intermediate numbers are not
the final deployment's performance.

The parallel-query build reduced time to first content to 1.488/1.489 s at
480 tokens, 8.636 s at 2030, and 71.661 s at 8030. All responses remained
ORCHID, with zero cached tokens. A subsequent optimization retains QK scores
in caller-owned FP32 scratch, avoiding their recomputation during the
softmax denominator and value passes. The shared buffer is reused across
layers on the same stream and adds 512 MiB per rank at 128 rows / 32K context.
The score-reuse attention suite passes all four reference groups and
Compute Sanitizer memcheck on Spark-1 with zero errors.

### Final prefill deployment

Final build ready at 23:49:35 UTC on the unchanged port 30001. Both running
executables have SHA256
`f7076a87c929da8e8b0a554dec466e521053544b5f0c412aad4d4d87e0c95269`;
both resolved configurations hash to
`58dd7e1123b1dee3f4c578e92adf58b96062ad9a880c74796a5f3c6b400f9f44`.
Both ranks load 48/48 layers. Planned memory is 84.16 GiB per rank plus
4 GiB headroom. Durable rank-0 deployment directory is
`/home/jon/dgpp/experimental/mimo-prefill-scores-20260921/`, containing
`dgpp-serve` and `deployment.json`. Use the same installed launcher/environment
as above with these paths. Namespace is `a88199c9b6be510c`; rank-0 logs are
under `/home/jon/dgpp/log/deployments/`, rank-1 under `/tmp/bus4/deployments/`.
The initial and intermediate binaries/configurations remain available for
rollback. Context stays 32768 and active concurrency stays one.

Final timings (temperature 0, thinking off, stream=true, max_tokens=32,
zero cached tokens, the same ORCHID recall prompt with 45/200/800 repetitions
of `The quiet river flows past the old stone bridge. `):

| Prompt tokens | Time to first content | Total request time |
|---|---:|---:|
| 480 | 1.145 s | 1.296 s |
| 480, repeat | 1.143 s | 1.296 s |
| 2030 | 5.497 s | 5.675 s |
| 8030 | 34.980 s | 35.261 s |

All return ORCHID and terminate normally. For the matched 480-token prompt,
server prefill is 1.125/1.113 seconds versus the original 23.110 seconds,
about 20.6x faster. This is a bounded prompt comparison, not a general
throughput claim. The initial single-token build was not measured at 2030
or 8030 tokens, so no baseline speedup is inferred at those sizes.

Final API acceptance passes reasoning separation and token counts, SSE with
usage and DONE, a structured synthetic tool call, simultaneous arithmetic
requests through the one-slot queue, and oversized-budget rejection. Both
ranks retire all ten requests and return to zero live/queued work. Health
reports MiMo. Raw timings, probes and acceptance responses are under
`/tmp/dgpp-mimo-gpu-20260921/{scores-prefill-probe*,prefill-final-*}`.

This is substantial prefill progress, not a fully optimized implementation:
8030 tokens still take 35 seconds to first content, concurrency remains one,
and attention still uses scalar dot products and three softmax/value passes.
Tiled/tensor-core attention, larger chunk tuning, concurrent sessions, decode
graphs and prefix caching remain future work. Full-model numerical equivalence
is still unverified; GPU/layer parity and functional HTTP checks have the
bounded scope described above.
