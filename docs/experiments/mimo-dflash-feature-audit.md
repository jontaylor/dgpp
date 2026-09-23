# DFlash target features and active reducer audit

Audited against combined runtime base `e2d724c`, recipe vLLM revision
`487ecf187`, and checkpoint revision `3b38d063180c3e4aed9691fdc735f3d10b266ee4`.
No Spark access was used for this audit.

## Feature taps

No indexing error was found relative to the recipe's actual vLLM path:

1. Checkpoint `dflash/config.json` lines 18–25 selects `[0,11,23,35,47]`.
2. Pinned [gpu_model_runner.py lines 5650–5654](https://github.com/vllm-project/vllm/blob/487ecf187/vllm/v1/worker/gpu_model_runner.py#L5650)
   explicitly adds one to DFlash IDs, selecting hook indices `[1,12,24,36,48]`.
3. Recipe `patches/files/mimo_v2.py` lines 592–608 reserves hook zero for the
   initial embedding, invokes each decoder block, calls the hook with `idx+1`,
   and only afterward applies the backbone final norm.
4. Pinned [interfaces.py lines 1477–1486](https://github.com/vllm-project/vllm/blob/487ecf187/vllm/model_executor/models/interfaces.py#L1477)
   captures `hidden_states + residual`. The recipe block returns the MLP
   output plus separate attention residual (MiMo source lines 443–457), so the
   hook obtains the complete block output, not the isolated MLP output.
5. DGPP `src/models/mimo/model.cpp` lines 262–281 copies the full residual after
   decoder blocks 0/11/23/35/47 and before final normalization. The concatenation
   uses the same ascending order and five complete 4096-wide vectors.

The layer-47 feature is therefore **pre-final-norm** in the recipe and DGPP.
The native MTP path separately consumes the normalized final hidden state.

The checkpoint's `dflash/dflash.py` helper (lines 200–209) adds one when indexing
an externally supplied hidden-state list, consistent with embedding at index
zero. However, the actual checkpoint's
[modeling_mimo_v2.py lines 1663–1680](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/3b38d063180c3e4aed9691fdc735f3d10b266ee4/modeling_mimo_v2.py#L1663)
returns only `last_hidden_state` and the cache, without constructing the
`hidden_states` tuple expected by that standalone helper. Thus this establishes
recipe runtime equivalence; it does not independently prove the unavailable
training feature exporter, particularly its treatment of the final norm.
Do not infer the final feature's normalization from a generic Transformers
hidden-state tuple convention.

## TP reduction and normalization ordering

DGPP `src/models/mimo/layers.cpp` lines 218–243 applies the attention output
all-reduce, adds its residual, normalizes the complete residual for the MLP,
applies the MLP all-reduce, and adds its residual. The feature copy follows on
the same CUDA stream. `fold()` uses the active model reducer and either drains
before a host-driven reducer or relies on its stream-ordered contract.
GraphRecordReducer records the collective on that stream. Thus feature vectors
are complete TP-reduced block outputs, not rank-local partials.

Drafter context processing uses replicated FC, then hidden norm, then local
sharded K/V projections (`src/models/mimo/dflash.cpp`, `context()`). This matches
recipe `qwen3_dflash.py` lines 796 and 534–543. Context processing itself needs
no all-reduce and retains no reducer. Arithmetic reduction/rounding differences
between implementations remain numerical questions; source ordering alone
cannot prove tensor parity on actual backbone outputs.

## Separate TP2 proposal bug found and fixed

The original drafter retained the reducer passed at construction. During graph
capture, `SessionModel::set_boundary` replaces the model's active reducer with
GraphRecordReducer (`graph_engine.hpp` lines 1617–1640), but that did not update
the drafter's stored pointer. A default BusStreamReducer also returns null from
`stage()`, which is legitimate for eager execution; the drafter incorrectly
rejected that case unconditionally. The world-1 drafter probe did not exercise
either failure.

The fix removes the stored pointer entirely. Every proposal receives the
model's current reducer explicitly. Eager execution falls back to the drafter's
local projection output when no staging buffer exists; capture still requires
a staged buffer. The stateless fold helper preserves project -> required host
synchronization -> reduce -> residual-add ordering. Prompt context projection
has no collective to reroute; its backbone features already use the model's
active reducer. Replay executes recorded nodes, and later eager proposals use
the restored reducer without retaining a temporary capture pointer.

Four host tests execute the production fold helper with mocks: eager -> capture
-> destroyed recorder -> eager selection, stream-ordered eager fallback,
capture rejection before projection without staging/current reducer, and the
world-1 path. These tests do not replace an actual TP2 graph startup, replay,
proposal, rollback and service correctness run.
