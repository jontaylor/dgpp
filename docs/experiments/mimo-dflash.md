# Experimental MiMo DFlash

Enable on **both ranks** with `DGPP_MIMO_DFLASH=1`, `--mtp`, `--mtp-depth 7`,
`--decode-graph`. Without the environment flag, MiMo native MTP1–3 is unchanged.
The serving API already requires graph mode for speculation; the model's eager
session draft interface and the standalone drafter probe remain available.
No deployment is performed by this branch.

This loads the actual release's five-layer BF16 Qwen3 drafter from
`CHECKPOINT/dflash/dflash_draft_model.safetensors`. It captures target outputs
from layers 0/11/23/35/47 before the backbone's final norm, concatenates them,
and applies the learned FC and hidden norm. Each draft layer independently
projects context K/V. Query blocks contain one pending token and seven learned
mask embeddings from `mask_embedding.pt`, **not** the target vocabulary's mask
row. The release's torch ZIP is read without executing pickle; metadata must
match the released contiguous BF16[4096] storage and mask token 151675.

The kernel uses per-head Q/K RMSNorm, 64-dimensional NeoX RoPE, 1024-token SWA,
noncausal block attention, zero-value learned sinks and value scale 0.612.
These follow checkpoint configuration plus the recipe's patched vLLM path.
The checkpoint's standalone `dflash.py` omits sinks/value scaling and is not an
exact oracle for those equations. Drafter output rows 1–7 predict the seven
masked positions; row 0 is discarded. A parallel block is computed once and
exposed through the existing draft-chain picking interface. It is not recursive
native MTP.

Q/K/V, sink heads and dense MLP are tensor-parallel; output and down projections
reduce with the serving reducer. FC and norms are replicated. The drafter
borrows target embedding and rank-local vocabulary head; native MTP weights
are not loaded. Base and draft cache lifecycles remain distinct. Only accepted
target features enter draft rings; noisy block K/V never enters those rings.
Draft snapshots restore context rings, target rollback retains its existing
position-masked expanded rings, and prefix snapshots also retain the final
20480-wide target feature. Padding never writes draft context. Decode supports
up to eight slots / 64 target verification rows.

The speculative wire bounds are widened together: `kSpecRows=8`,
`kSpecMaxDrafts=7`, `kSampleVerdictRows=8`, cluster JSON depth 1–7. Existing
`/v1/metrics` fields `num_draft_tokens_per_pos_total` and
`num_accepted_tokens_per_pos_total` include all seven draft positions.

## Validation

Local cross-build (no GPU execution):

```sh
cmake --preset spark-cross
cmake --build --preset spark-cross --parallel 2 --target dgpp_serve_app mimo_dflash_check glm_pick_test
```

On idle Spark hardware, the standalone probe loads only the target global
weights and drafter, not the full backbone (roughly 6 GiB plus scratch at
world 1). It checks a real-weight two-request block, exact eager/graph replay,
and context snapshot/restore after poisoning the cache:

```sh
./mimo_dflash_check "$CHECKPOINT" "$OUT"
python3 tools/mimo_dflash_reference.py "$CHECKPOINT" "$OUT"
DGPP_TEST_FILTER=sample_pick_full_block ./glm_pick_test
DGPP_TEST_FILTER=spec_positions_rows64 ./glm_pick_test
```

The CPU oracle needs only torch and independently maps the BF16 safetensors;
it reports max absolute error, RMS and cosine agreement for all 16 hidden
rows. Its provisional acceptance gate is min cosine >0.999 and RMS <0.08;
inspect reported errors and actual greedy logits/acceptance in addition to that
numerical gate. It is a synthetic-context equation probe, not evidence of
end-to-end target-feature alignment or quality.

Required serving gates: compare greedy token IDs to the same target with
speculation disabled on arithmetic, prose, code, counting, tool schemas and
prefix-resume journeys; test C1/C2/C4, mixed lengths, rejection at all seven
positions, all-accept, EOS, context boundary, cancellation, warm prefix hits,
reset/reuse, graph replay and repeated requests. Compare target baseline and
DFlash with identical context, batch, target precision and prompt tokens.
Track draft/accept counts per position and all attempts. Validate TP2 separately
from the world-1 equation probe. Run compute-sanitizer on the standalone probe
and the full-width pick test before performance claims.

## Limits

This is a correctness-first experiment. The drafter attention uses a simple
online-softmax CTA per query/head; no optimized FlashAttention speed is claimed.
Full draft-ring snapshot copies remain and the FC projection is replicated.
Changing global sample verdict bounds changes the wire layout; ranks must use
the same build. Only the pinned MiMo-V2.6-Flash-RL shapes and mask archive schema
are supported. Host tests/cross compilation do not establish GPU correctness,
acceptance, throughput or safe full-context capacity.
