# MiMo implementation comparison — 2026-09-23

Compared upstream `HawkBearPig/dgpp` master `c6ca191`, including MiMo support commit `7d8bb61aeb8f9b3278567fda5d95501b4d966cfb`, with our implementation frozen at `03a7f5f`. This is a source review plus a comparison of recorded experiments. Upstream has not been deployed or benchmarked on our Sparks during this comparison.

## Conclusion

Upstream has the stronger general serving foundation and a promising long-context prefill kernel. We already have several optimizations upstream explicitly leaves for later: all three native MTP heads and compact sliding-window rings. We also skip unnecessary MTP history computation and project only the last prefill row through the vocabulary head. Replacing our implementation wholesale would lose those features.

The first performance candidate to port is upstream's query-tiled prefill kernel. Its fused decode path is a second candidate, with more modest published end-to-end gains. Preserve our native multi-head MTP, cache-only history path and ring-cache layout during any port. Longer term, converge onto the upstream family rather than maintaining conflicting implementations indefinitely.

## Implementation differences

| Area | Our implementation | Upstream | Implication |
|---|---|---|---|
| Model/checkpoint | Same MiMo text backbone, native MXFP4 experts and checkpoint FP8 dense weights; BF16 matrices remain BF16 | Same formats, plus lossless BF12 companions for eligible BF16 decode matrices | BF12 is compression, not 12-bit requantization; a possible bandwidth/storage improvement for those matrices |
| Native MTP | Loads heads 0, 1 and 2; depth-three drafting uses the corresponding head, conditioned on backbone history | Loads head 0 only; templates and published sweep use depth one. Generic deeper drafting chains that same head | Setting upstream depth to three does not reproduce our native three-head implementation |
| MTP history | Returns after QKV append when only KV history is needed | `mtp_run_rows` runs the complete draft layer before its `head_rows == 0` return | Our cache-only optimization is useful work to retain/port upstream |
| Vocabulary projection during prefill | Last row only for normal serving; all rows for decode verification/diagnostics | All prefill rows, deliberately retaining its full-forward bitwise gate | We avoid an expensive vocabulary projection over unused rows; changing upstream requires updating numerical gates |
| Sliding-window KV | Compact rings for 39 SWA layers; full history only for nine global layers | Full paged history for SWA and global layers; SWA mask applied during reads | Our live cache is substantially more efficient at long context; upstream's own roadmap calls out ring conversion |
| Live global KV | Contiguous, reserved per request | Shared paged pool with block tables/refcounts and copy-on-write | Upstream has more flexible sharing/admission; ours has simpler contiguous attention access |
| Prefix snapshots | Original private snapshots; optional immutable 256-token block sharing, with SWA/MTP state private | Prefixes retain pool pages and small session metadata | Upstream avoids our global restore copies; our new snapshot sharing is not a paged live cache |
| Prefill attention | Materialized scores, fused softmax/PV path; new opt-in SWA tile reuse and repaired cuBLASLt QK | Query-tiled online softmax with tensor-core QK and PV; 64 query vectors reuse each staged KV tile | Upstream removes the large score materialization and repeated per-query KV staging, especially attractive at long context |
| Decode kernels | Separate append/attention stages, our compact/mapped and split paths | Fused QKV finish, split attention and last-block combine; fused residual-add/RMSNorm; hardware MXFP4 conversion fast path | Concrete port candidates, but published aggregate decode gain is modest |
| Optional FP8 KV | Unit-scale E4M3, native MTP cache remains BF16; full-attention prefill expansion bridge | Per-head-row E4M3 with FP32 absmax scales, integrated dequantization into attention tiles | Formats are not interchangeable; changing storage requires fresh numerical/quality validation |
| Tool calls/tokenization | MiMo template/dialect, schema grammar, streaming/non-streaming handling; exercised through the user's actual harness | MiMo template/dialect, Qwen2 tokenizer handling, parser/grammar tests, model-opened thinking | Both address the dialect; upstream has not been tested against our exact harness request here |
| Validation | Two-Spark GPU/sanitizer gates, fixed output/acceptance comparisons, real harness tools, cache/branch/cancellation tests and fixed long retrieval | Recorded two- and four-node runs; full HumanEval164, GSM8K300 and extraction100; greedy/plain/MTP and solo/batched checks | Upstream has broader published quality and topology evidence; our checks cover our production client and native MTP3 path |

Upstream explicitly lists SWA rings and native MTP layers 1/2 as remaining work in [its plan](https://github.com/HawkBearPig/dgpp/blob/c6ca191/docs/mimo_v26_flash_plan.md).

## Deployment capacity is not equivalent

Our current requested configuration is **two concurrent requests, each with 262144-token capacity**, and four prefix snapshot slots. Native MTP depth is three.

Upstream's two-node template has **four request slots sharing one 262144-token FP8 pool**, with MTP depth one. Its two-node BF16 benchmark uses a shared131072-token pool. Four slots do not each get the full advertised pool capacity, and prefix pages also occupy pool space. These configurations cannot be compared by the labels C2/C4 or256K alone.

## Performance evidence

The following are observations from different workloads, not a matched A/B speedup claim:

| Measurement | Our recorded runs | Upstream published two-node runs |
|---|---|---|
| Short C1 decode | About44.3 code,48.2 JSON,28.0 prose and48.3 math tok/s on our four prompts, native MTP3 | BF16:42.3–49.2 engine tok/s across five classes; FP8:39.9–48.2, MTP1 |
| Approximately31–32K cold prefill | 30876 tokens:56.60s control,56.73s attention-only candidate | Approximately32K:57.21s BF16 and58.51s FP8 median engine prefill |
| Long prefill | 67396 tokens:232.56s control,222.15s attention-only candidate | 130288 tokens:338.64s engine prefill,338.76s client TTFT;238855 tokens:838.78s engine prefill |
| Shared snapshots | At67K, owned snapshot storage11.54->1.05GiB per rank; cached TTFT398->424ms, unchanged decode in the isolated run | Page-sharing architecture, but no matched test of our four-snapshot workload here |

Upstream's75.8–85.8tok/s headline is **four nodes**, not two. Its four-node121266-token prefill was142.86s. Do not compare that directly with our two-Spark deployment.

Decode prompts, thinking mode, MTP depth and timing scope differ: upstream's short sweep disables thinking and reports engine decode counters, while our panel uses the model's default thinking behavior and client time after the first streamed token. Even overlapping ranges do not establish equal performance. The long-context numbers and kernel design make upstream prefill a high-priority candidate, but its gain on our prompts has not been measured.

Upstream's fused decode/norm round reports roughly1.5% C1 MTP improvement in its four-node fabric A/B, with smaller C2/C4 changes. Do not expect its attractive kernel microbenchmarks to become a large end-to-end decode gain automatically.

## Quality and numerical boundaries

Upstream reports154/164 HumanEval,295/300 GSM8K and100/100 extraction for two-node BF16; the two-node FP8 row reports153/164,294/300 and100/100. These are the upstream author's recorded results, not tests rerun here. We have no corresponding broad score for our current implementation, so cannot claim better model quality from our narrower spot checks.

The attention arithmetic also differs. Upstream explicitly uses flash-style online normalization, BF16 probability terms and an unrounded denominator. Its tensor-core prefill changes accumulation order; the plan records relative-L2 checks and up to two BF16 ULPs in its cases. Our materialized path preserves different intermediate BF16 rounding points. Its kernel cannot be dropped in with a claim of universal bitwise identity. Validate against an independent reference, logit/token comparisons and the same quality corpus before selecting it.

Our recent attention-only and shared-only panels each matched all19 control outputs and all four MTP acceptance vectors. The final attention+shared BF16 panel matched18/19 control outputs and all four MTP acceptance vectors; one concurrent code response differed. A10-request diagnostic then found that solo code and every JSON response matched, while code wording differed in both staggered concurrent orders, consistently across repetitions. This was not repeated on the unmodified control, so the cause is not attributed to the optimization. All final tool, cache, cancellation, mixed-prefill and31K/67K retrieval assertions passed; universal greedy bitwise equivalence is not claimed. See `2026-09-23-mimo-attention-shared-snapshots.md`.

## Provenance

Upstream's published optimized campaign identifies binary `4638e3c595a2765d6d50c4626a9fef3aa2aa82936c6c50e3d0eec871fcde8e00`, built from `a895db96bb17.dirty` before the MiMo changes were committed. Its manifest describes the dirty tree. This comparison reads the subsequently committed implementation at `c6ca191`; the benchmark labels alone are not an exact rebuild recipe for a clean `a895db9` checkout.

The upstream checkpoint manifest uses revision `5711b268169967567844e1e560e8a3966da959b1`; ours uses `3b38d063180c3e4aed9691fdc735f3d10b266ee4`. Both Hugging Face file manifests were fetched during this review. All text weight, text config and tokenizer blob IDs match; the only differing blobs are README, the technical-report PDF and `dflash/config.json`. Neither compared native-MTP configuration uses DFlash.

Evidence and fetched manifests: `/mnt/benchmarks/dgpp-mimo-attention-20260923/`.

## Primary source pointers

- [Upstream model card](https://github.com/HawkBearPig/dgpp/blob/c6ca191/docs/model_cards/MiMo-V2.6-Flash.md)
- [Upstream measured results and method](https://github.com/HawkBearPig/dgpp/blob/c6ca191/benchmarks/results/2026-09-22-mimo-v26-flash-opt/overview.md)
- [Query-tiled prefill kernel](https://github.com/HawkBearPig/dgpp/blob/c6ca191/src/kernels/mimo_attn_prefill.cu)
- [Live KV layout and scaled FP8 format](https://github.com/HawkBearPig/dgpp/blob/c6ca191/src/models/mimo/kv_pool.hpp)
- [Upstream session/MTP implementation](https://github.com/HawkBearPig/dgpp/blob/c6ca191/src/models/mimo/forward.cpp)
- [Lossless BF12 format](https://github.com/HawkBearPig/dgpp/blob/c6ca191/src/kernels/bf12_gemv.hpp)
