# Expert-streaming performance review response

Review baseline: `8acd7305`. Target-layer graph baseline: `ba60af73`.
This response concerns the existing BF16-activation MXFP4/NVFP4 paging paths.
It does not change sampling, quantization, cache budgets, routing, or ordered
expert accumulation.

## Changes

- Canonical expert waves activate only their executed gathered rows. The shared
  expert's activation runs once, with its projections, in the final wave.
  Existing clamp and BF16 rounding kernels remain unchanged.
- Large prefills stage bounded groups of canonical experts using the existing
  execution-slot reservation. One view-table upload and one grouped chain serve
  each wave. Empty expert IDs between selected IDs retain empty segments.
  Whole-model prefill chunks are unchanged. Prepared NVFP4 weights still need
  unpacking for this large-prefill recipe; small batches retain direct b12x.
- Telemetry distinguishes direct native bindings from native hits that unpack
  into canonical scratch. Canonical output-byte counts are logical bytes, not
  physical DRAM traffic. Enqueue time is CPU time, not GPU unpack duration.
  Backend completion-wait time includes pending copies/expert execution and
  overlaps existing reconstruction/layout wall timers.
- Speculation logs effective eligibility on the first decode and whenever its
  reasons change. Reasons include sampling temperature, requested logprobs,
  bias, penalties and context capacity. Tool grammar alone is not disqualifying.
  Optional per-pass timing logs now explicitly include proposed token counts.

## Focused validation

A work-count assertion failed on the old whole-batch activation sweeps, then
passed with segment-local activation. The paged-vs-resident fixture covers both
FP4 formats, FP8/FP4 shared experts, both cache policies, changing routes, empty
expert segments, single/multi-expert scratch waves, and token counts
1, 7, 64, 256, 1024 and 4096. Large cases use the prefill MMA recipe.
Output comparisons are bitwise. Activation work no longer scales with the
product of wave count and full batch rows.

The prepared b12x fixture covers native hits, canonical unpack accounting,
actual cold reads, grouped misses and GPU-demand graph replay. The host cache
suite covers direct-hit accounting and existing residency invariants.

## Interpretation and remaining boundaries

The legacy `native_hits` counter remains compatible as an acquisition count;
it must not be interpreted as copy-free execution. Use the direct-hit and
canonical-unpack counters alongside native preparation/relocation counters.

Canonical groups still wait for their required reconstructions before execution.
This change introduces no readiness scheduler, new transfer stream or prefetch
policy. Reassess exposed stalls after removing redundant prefill work.

The EXL3 comparison uses different precision, execution kernels and workloads.
The reference's Engram tables are disk-backed even though transformer weights
are resident. Neither its rates nor logical activation-byte counts establish an
SSD throughput limit for this deployment. Exact drive-bandwidth characterization
is outside this pass, consistent with the user's instruction that exact
bandwidth numbers are unnecessary.

Serving validation and proxy/direct request accounting are pending.
