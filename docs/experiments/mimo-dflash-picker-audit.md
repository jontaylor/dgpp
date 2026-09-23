# DFlash7 picker slot regression

Runtime `9f65b0c` advanced through the plain scalar graph capture and then
failed `device pick: slot` while capturing the speculative variant. At seven
drafts, both `record_scalar_mtp` and `record_batch_mtp` use picker slot 0 for
verification, slot 1 for the first proposal, and slots `1+c` for `c=1..6`.
The highest slot is 7, requiring eight slots. DevicePicker still declared six
slots, so the bounds check correctly rejected slot 6. This was a missed
independent constant in the earlier DFlash width audit.

The fix retains the bounds check and derives DevicePicker slots, engine verify
rows, kernel maximum drafts and sample outcome rows from one dependency-light
`common/spec_limits.hpp`. Graph-engine static assertions additionally require
those public bounds to agree. There is no relaxed or unchecked access.

All DevicePicker allocations and indexing were audited: pinned verdicts,
device verdicts, local candidate mirrors, and sampling outcomes size by
`kSlots` and index by the same slot-major stride. They therefore grow together
from six to eight slots. The graph engine's proposal arrays derive from
`kSampleProposalSlots=7`; `GlmSpecDrafts` derives from `kSpecMaxDrafts=7`;
session token buffers and per-request feeds derive from `kSpecRows=8`. The
maximum decode row count remains 64, supporting eight request groups at full
depth. Request slots, speculative picker slots and bus graph variant IDs are
separate dimensions; this fix changes only the formerly undersized pick-phase
slots. No other operative literal-five/six speculative bound was found in
engine/kernel/serve code; remaining search hits are historical comments or
unrelated encoding dimensions.

## Regression

`device_picker_dflash7_all_slots_eager_and_graph_loopback` in
`tests/cuda/glm_tp_test.cpp` constructs actual world-two DevicePickers, each
with sampling mirrors enabled. It runs all eight picker slots eagerly, records
all eight in one CUDA graph, and replays twice. Slot zero verifies two groups
of eight rows; slots one through seven pick two independent requests. The test
checks retained verdicts, locals, device/outcome slot separation, and strict
rejection of slots -1 and 8. This needs CUDA and the existing loopback bus test
hardware; it does not load the MiMo checkpoint.

Parent validation command after building `glm_tp_test`:

```sh
DGPP_TEST_FILTER=device_picker_dflash7_all_slots_eager_and_graph_loopback ./glm_tp_test
```

Follow with the actual TP2 DFlash7 graph startup and service protocol. Merely
passing compilation/static assertions does not establish graph runtime
correctness. The preparation agent did not access either Spark.
