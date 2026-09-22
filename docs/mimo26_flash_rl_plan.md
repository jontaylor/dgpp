# MiMo-V2.6-Flash-RL port

The text backbone has resident TP serving across two Sparks, with asymmetric
global/sliding attention, dense/MXFP4 routed FFNs and the shared HTTP engine.
The deployment template enables two independent sessions, 256K context per
request, decode graphs and exactly four prefix snapshots on port 30001. The
prefix budget with native MTP is 11.539825439453125 GiB (four 3,097,698,304-byte snapshots). Prefill
yields after 256 tokens while another request decodes and uses up to 2048 tokens
when idle. Earlier 64K/eight-session validation covered concurrent conversation
reuse, ring-wrapping restores and a 65,030-token prompt. See the
[256K deployment record](../benchmarks/results/2026-09-22-mimo-256k.md) for the
current configuration and its validation scope.

Attention uses tensor-core prefill, bounded score tiles, and a split tensor-core
global decode path for histories of at least 512 keys. Shorter histories preserve
the original scalar arithmetic. MoE and layer execution workspace are shared across sequential
layers; graph weight tables and RoPE constants retain separate stable storage.
Tool-call constraints use MiMo's native compact XML delimiters, preserving
string-value newlines. Declared required tool arguments are enforced even
without `strict: true`; see the [required-argument repair](../benchmarks/results/2026-09-22-mimo-required-tools.md)
and the [tool-format repair](../benchmarks/results/2026-09-22-mimo-tool-format.md).
Native MTP supports one to three draft tokens using distinct checkpoint blocks
0, 1 and 2 and the shared vocabulary head. Each head has independent K/V
history and consumes a backbone hidden row shifted by its prediction depth. See the
[native-block record](../benchmarks/results/2026-09-22-mimo-native-mtp3.md).
The [recursive baseline](../benchmarks/results/2026-09-22-mimo-mtp3.md) and
[depth-one record](../benchmarks/results/2026-09-22-mimo-mtp.md) retain earlier
measurements. DFlash and multimodal execution remain unavailable. Full-model HF logits/NLL
equivalence remains unverified; layer and kernel numerical gates are narrower
claims.

The deployment template is
[`deploy/cluster_mimo_v2.6_flash_rl_w2.example.json`](../deploy/cluster_mimo_v2.6_flash_rl_w2.example.json).
It requires the checkpoint on both ranks. The published tokenizer uses NFC,
GLM-style letter classes and single-digit pretokens; the dedicated scanner
pattern passes 94 cases byte-for-byte against HF tokenizers 0.23.1.

## Checkpoint contract

The source is [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/tree/3b38d063180c3e4aed9691fdc735f3d10b266ee4),
revision `3b38d063180c3e4aed9691fdc735f3d10b266ee4`. The parser deliberately accepts
this release's text geometry; it does not claim general MiMo-family support.

| Component | Contract |
|---|---|
| Backbone | 48 layers, hidden 4096, vocabulary 152576 |
| Global attention | Layers 0, 5, 11, 17, 23, 29, 35, 41, 47; 64 Q heads, 4 KV heads |
| Sliding attention | Other 39 layers; 64 Q heads, 8 KV heads, window 128 |
| Head dimensions | Q/K 192, V 128; fused QKV, no projection bias |
| RoPE | First 64 dimensions, from `int(192 * 0.334)`; theta 1e7 global and 1e4 sliding |
| Attention sinks | Learned bias for sliding layers only; softmax includes a zero-value sink |
| Values | Reference multiplies projected V by 0.707 before cache insertion |
| FFN | Layer 0 dense, intermediate 16384; 47 MoE layers, intermediate 2048 |
| Routing | Sigmoid, corrected top-8 selection among 256 experts, normalized uncorrected scores; no shared experts |
| Main text tensors | 72,574 entries |

The reference router computes the linear projection in FP32 despite BF16 stored
router weights. Preserve this distinction when selecting existing MoE kernels.
The tensor manifest distinguishes BF16 norms, embeddings, output head and attention
output projections; FP8 dense FFN/QKV with F32 scales; and U8 packed MXFP4 experts
with U8 scale bytes. Header dtypes alone do not establish scale decoding or nibble
order.

**QKV storage is TP4-interleaved.** Each checkpoint chunk holds its own
`[Q,K,V]` rows, and its 128-row scale grid restarts at the chunk boundary.
Global chunks contain 3392 rows and 27 scale rows, explaining the full
`[13568,4096]` payload and `[108,32]` scales. Sliding chunks contain 3712 rows
and 29 scale rows, producing `[14848,4096]` and `[116,32]`.

This mapping is corroborated by
[vLLM's loader](https://github.com/vllm-project/vllm/blob/382970ee6ca490aeaaaf4e32c53695b581ff61ba/vllm/model_executor/models/mimo_v2.py)
and [SGLang's loader](https://github.com/sgl-project/sglang/blob/acac4dd9d9ecf415628fd0ab99f101903c86f653/python/sglang/srt/models/mimo_v2.py).
The host reader gathers the rank's heads in canonical `[all Q, all K, all V]`
order and applies each source row's original scale before converting to BF16.
It does not requantize to FP8. TP1/2/4 geometry and scale-boundary samples from
the actual checkpoint pass exact BF16 checks. This establishes sampled unpacking
parity, not full-model numerical equivalence.

`src/models/mimo/weights.*` provides bounded FP8-to-BF16 matrix slices and
MXFP4 row/column slices. Expert bytes remain packed, with low-nibble-first e2m1
values and e8m0 scales per 32 elements. A separate host decoder supports tests.
`MimoCheckpointWeights` maps source shards on demand, validates requested tensor
shapes and returns owned QKV, expert or ordinary text-weight slices. It caches one shard; a pair may
keep a second mapping alive while reading scales stored in a different shard.

There are two distinct speculation sources: three conventional MTP blocks under
`model.mtp.layers.{0,1,2}` (48 tensors in the main index), and a separate five-layer
`dflash/` checkpoint. The former explains `num_nextn_predict_layers: 3`; it is
not stale metadata. The main text binding counts these separately; enabling MTP
loads and validates all three blocks through a 48-tensor contract.
The checker separately counts 459 visual/audio/speech tensors without validating
their semantics. Full multimodal serving is a later phase.

## Host inspection

Build `mimo_checkpoint_check` with the usual CMake configuration, or compile the
host-only slice on a workstation without CUDA:

```bash
g++ -std=c++20 -O2 -Isrc src/loaders/minijson.cpp \
  src/models/mimo/config.cpp src/models/mimo/binding.cpp src/models/mimo/weights.cpp \
  apps/mimo_checkpoint_check.cpp -o /tmp/mimo_checkpoint_check
/tmp/mimo_checkpoint_check /path/to/snapshot --index-only
/tmp/mimo_checkpoint_check /path/to/snapshot
```

The first mode checks configuration, index metadata and tensor-name coverage,
and works during the download. It does not check tensor shapes or payloads.
The second maps one shard at a time, checks index/header agreement, text tensor
dtypes/shapes and byte extents, and does not read weight values or allocate GPU
memory. Neither mode proves numerical correctness or download checksums.

`kv_bytes(context, world)` estimates per-rank BF16 K/V storage with bounded SWA
rings. At 32K context and TP2 this is 390,266,880 bytes per sequence. At 1M it
is about 11.26 GiB per rank per sequence. These figures exclude allocation
rounding, prefix copies, speculative state and workspaces. They are geometry
estimates; the integrated 32K serving plan is described below. The main index declares
172,923,364,096 bytes, including auxiliary tensors; runtime residency cannot
be inferred by simply dividing this by two.

The initial resident weight layout uses BF16 QKV, output projections
and the dense FFN, packed MXFP4 experts, replicated embeddings/norms/routers,
and a vocabulary-sharded output head. Its TP2 weight estimate is
87,088,227,776 bytes (81.11 GiB) per rank. The estimate is checked against the
per-tensor placement but is not measured GPU residency. The integrated 32K plan with 128-row prefill reports 84.16 GiB per rank
plus 4 GiB headroom,
including K/V, scratch and session buffers; both ranks successfully load it.

Build `mimo_weight_check` to check the host reader against the retained payload
samples on a downloaded checkpoint:

```bash
mimo_weight_check /path/to/snapshot tests/data/mimo/weight_samples.json \
  tests/data/mimo/layer_weight_samples.json
```

This CPU-only check exercises all three supported TP geometries on one host;
it does not run GPU kernels or multi-rank collectives. The goldens can be
regenerated with `python3 tools/mimo_reference_sample.py /path/to/snapshot`.
The sampler reads bounded file ranges and imports no model code.

## Ordinary layer and global weights

`MimoCheckpointWeights::bf16` supplies norms, attention output projections,
sliding sinks, dense gate/up/down projections, routers, embeddings and the
vocabulary head. `placement` exposes each rank's source-coordinate rectangle;
optional reader windows are relative to that rectangle. Vectors are returned
as `[N,1]`. Omitted lengths consume the local remainder, so callers should use
explicit bounded windows when inspecting large embeddings or projections.

Output/down projections partition columns; dense gate/up, sinks and the
vocabulary head partition rows. Embeddings, norms and routers replicate.
BF16 copies preserve bits, including signed zero. Dense FP8 converts with its
original two-dimensional scale grid. QKV and packed experts retain their
specialized readers and cannot accidentally use the ordinary layout.
`router_bias` returns the replicated FP32 correction vector. Router weights
remain stored BF16; execution uses FP32 dot products.

The ordinary reader validates the complete source shape and dtype before
copying a window, retains a weight shard while acquiring a separate scale
shard, and returns owned storage. It does not allocate device memory.
110 independent samples from the pinned checkpoint cover all these roles,
including TP4 partition endpoints and dense FP8 scale boundaries. The ARM64
checker passes 110/147/221 rank comparisons at TP1/TP2/TP4 respectively.
These are sampled host storage checks, not distributed forward-pass parity.

## Implementation status and remaining gates

1. Release config, TP geometry, text manifest and real checkpoint validation
   are implemented; native MTP heads have a separate optional execution path.
   Multimodal tensors are not executed.
2. Host loading and aligned GPU residency cover all main-text weight roles.
   Embeddings/head upload in bounded row chunks; experts stay packed MXFP4.
3. Initial attention passes CPU/PyTorch and focused GPU parity, graph replay
   and sanitizer checks. Chunk attention executes queries in parallel with
   retained ring history.
4. Complete real-weight layer 0 (dense) and layer 1 (MoE) pass a three-token
   PyTorch comparison on Spark-1. The MoE uses DGPP's FP32 partial weighted
   accumulation, which differs from eager per-expert rounding. Full-model
   logits/NLL equivalence remains unverified.
5. `MimoModel` assembles all 48 layers with two boundary folds per layer,
   expanded SWA rings, request-private global K/V and up to eight sessions.
   Scalar and compact batched graphs execute one token per active request.
   Resumable prefill interleaves with decode; snapshots copy only the
   selected request's committed global history and pack its live 128-token
   sliding windows. Multi-row speculative verification and prefill
   spanning multiple requests remain rejected.
6. Architecture registration, tokenizer, published chat template and HTTP
   serving are connected. End-to-end acceptance evidence is recorded in the
   [initial engineering record](../benchmarks/results/2026-09-21-mimo-contract.md) and
   [optimization record](../benchmarks/results/2026-09-22-mimo-optimization.md).
7. Native MTP supports depths one through three using distinct blocks and
   independent histories. DFlash and multimodal execution remain separate work.

## Resident decoder blocks

`MimoDeviceLoader` owns one aligned GPU allocation per layer and one for
shared globals. It checks the allocation plan against the actual allocation
cursor. Source buffers are temporary, at most one projection or expert slice;
large global matrices are copied in 1024-row chunks. Weight views must
outlive their decoder layers.

`MimoDecoderLayer` composes two-rounding RMSNorm, BF16 projections, MiMo
attention, SwiGLU and the existing sigmoid/MXFP4 MoE path. MXFP4 dispatch now
includes MiMo's 4096/2048 widths as well as TP2/TP4 down widths. Routers are
replicated; output/down projections produce partial hidden rows that fold
before the residual add. All attention status flags are checked before the
session commits the token on the eager path; graph positions are validated by
the shared session core. The memory plan includes weights, all request planes,
shared MoE/layer execution workspace, session buffers and prefix snapshots.
Attention temporary storage is limited to 128 query rows, independent of the
projection/MoE chunk size: FP32 scores plus BF16 probabilities use 1.5 GiB per
rank at 64K context. Decode reuses dead scores for its smaller split-PV partials.
Global physical planes are rounded to 128 tokens for alignment, while the public
per-request context bound stays exact. Snapshot planning includes that global
padding but reserves only 128 entries per sliding layer. Copy traffic follows
committed history, not maximum context. The session core passes snapshot position
to MiMo restore so packed windows return to the correct physical ring slots;
other model families retain their existing restore interface.

`mimo_layer_check SNAPSHOT LAYER OUTPUT_BF16` runs three sequential real-weight
rows on an idle GPU. `tools/mimo_layer_reference.py` produces independent CPU
PyTorch goldens without importing checkpoint Python, and
`tools/mimo_layer_compare.py REFERENCE_JSON GPU_BF16` checks finite outputs,
per-row relative L2 <=1% and maximum error <=5% of the reference maximum.
Observed errors are substantially below those gates; see the dated record.

## Initial attention path

`models/mimo/attention.*` defines the cache geometry and CPU oracle;
`kernels/mimo_attn.*` implements append and attention on CUDA. Q/K heads are
192 wide and values are 128 wide. Fused projection inputs are already BF16.
Append rotates the first 64 dimensions with half-split pairs, scales V by
0.707 and stores K/V in request-private cache slots. Global layers use a
linear cache; sliding layers attend to 128 tokens and retain the next power of two at or above 128 + chunk_capacity - 1
slots per request (4096 slots with 2048-token chunks). The extra slots prevent later chunk appends from evicting
keys needed by the earliest query.

An invocation appends one token per request, with negative positions marking
padding. The caller must supply consecutive positions and reset a reused
request slot to position zero. Prefill batches projections and grouped MoE
across up to 128 rows, appends
the chunk into the expanded ring, then runs all queries in parallel. Causal
position bounds exclude later tokens.
The shared-cache kernel rejects a chunk if the ring cannot retain its
earliest query history. It requires consecutive positions from one sequence.
Device status flags report out-of-range positions;
those rows return zero output without cache writes, and the caller must
surface the error. The decoder checks these flags before committing each session token.

The initial attention kernel mirrors eager-reference rounding: QK products
accumulate in FP32, scores and scaled scores round to BF16, max-subtracted
scores round to BF16, softmax runs in FP32, probabilities round to BF16,
and the FP32 value accumulation rounds to BF16 output. A sink adds a logit
to the denominator with no value contribution. Three score passes preserve
this contract with bounded shared memory and no context-sized score buffer.
This is a correctness baseline, not an optimized serving kernel.

CPU PyTorch 2.14.0 goldens cover 16 global/sliding attention cases at positions
0, 1, 126, 127, 128, 129, 255 and 256, plus six RoPE-only cases at 8191,
32767 and 1048575. Comparisons permit one BF16 ulp with relative L2 at most
0.2%; passing the last rotary position does not validate 1M-token attention
or its memory/performance. Additional analytical tests cover sink mass,
oldest-token eviction, request isolation, reset and invalid positions.

Build and run `mimo_attn_test` only on idle test hardware. Its three GPU
groups cover actual TP4 head geometries, window wraps, paused requests,
slot reset, graph replay with 65 rows and invalid-position status. All three groups pass independently on both Sparks, as do Compute Sanitizer
memcheck, racecheck and synccheck (2026-09-21). These are single-GPU kernel
checks using TP4 head geometry, not multi-rank tensor-parallel validation
or assembled-layer parity. Performance remains unmeasured.
