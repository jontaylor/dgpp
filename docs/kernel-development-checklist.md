# Checklist for new GPU kernels

Use this when adding or replacing a DGPP kernel, before concluding that an
arithmetic format or execution strategy is slower. A working prototype is not
an optimized comparison if it omits advantages already present in the baseline.

Check each applicable item, or record why it does not apply. This is an
implementation checklist, not a requirement to run every possible experiment.
Start with the actual bottleneck and use short, targeted comparisons. Follow
concrete leads when a prototype loses; do not declare the approach exhausted
while known baseline optimizations are missing.

## Establish the execution shape

- [ ] **Read the current winning kernel and its caller.** List its prefetch,
  layouts, staging, reuse, tail handling, dispatch and overlap before writing
  a replacement. Compare the execution paths, not just their MMA instructions.
- [ ] **Target the measured bottleneck.** Profile the actual prefill or
  verification shape. A one-row decode profile does not explain batched
  verification; separate expert work, dense work, communication and CPU gaps.
- [ ] **Match the checkpoint and rank geometry.** Use actual local H/I sizes,
  quantization group sizes, scales, top-k, tensor-parallel partition and output
  dtype/stride. This checkpoint routes top-six target experts and top-three
  draft experts. Speculative depth N means N+1 target rows.
- [ ] **Use the executed prefill chunk size.** A 24k prompt served in 4,096-token
  chunks is not a 24k-row GEMM. Include short final chunks and small paging waves.
- [ ] **Replay real route distributions.** Include uneven expert populations,
  hot experts, reuse and sparsely populated tails. Uniform groups alone hid
  important differences. Check representative low- and high-skew layers.
- [ ] **Keep gate/up and down inputs realistic.** Gate/up share token inputs;
  down consumes distinct expert-specific activations. Do not invent reuse in
  the benchmark that the model cannot exploit.

## Move weights and scales efficiently

- [ ] **Keep weights packed in global memory.** Expand only the fragments
  consumed by MMA where possible; avoid materializing whole BF16 experts.
  Native FP4 arithmetic does not reduce weight traffic when the baseline
  already reads packed FP4 and expands register fragments.
- [ ] **Consume production prepared layouts directly.** Copy contiguous prepared
  slabs into shared memory and extract their fragments there. Do not gather or
  unpack them back to canonical order inside the hot path.
- [ ] **Support the views serving actually uses.** Handle prepared, canonical,
  row-chunk and column-chunk sources as applicable, including gate/up offsets,
  scale permutations, gathered input rows and padded output strides. A fast
  canonical-only kernel may not replace the deployed resident path.
- [ ] **Coalesce weight, activation and scale accesses.** Preserve vectorized
  slab copies and matrix-load instructions where appropriate. Check alignment
  and shared-memory access layout, not only arithmetic instruction count.
- [ ] **Stage and reuse scales.** Avoid reloading/recomputing scales per fragment
  when they can be shared. Respect the actual scale layout; a transpose can
  help one projection and hurt another.
- [ ] **Preserve explicit GPU weight-line prefetch.** For known upcoming weight
  addresses, consider L2 lookahead beyond the shared-memory pipeline. Async
  copies alone did not hide our weight-fetch latency. Prepared and canonical
  layouts need different access/prefetch implementations.
- [ ] **Tune lookahead to the consumer.** One upcoming line beat two in our
  native canonical screen. More prefetch is not automatically better. This
  is GPU cache prefetch, distinct from speculative SSD expert loading.
- [ ] **Match the interval between uses of a weight matrix.** For TP2
  `wo_b`, rotate through all 40 target-layer matrices before reusing the
  first. Five matrices exceeded L2 capacity but still favored a different
  tile and prefetch policy. Check the winning projection in a serving trace.
- [ ] **Pipeline copies with compute.** Use asynchronous staging and a working
  shared-memory ring; issue the next useful work before the consumer needs it.
  Check dependencies and buffer lifetime rather than assuming async means overlap.

## Spend GPU resources on useful work

- [ ] **Use the target device's limits.** GB10 has 48 SMs, 1,536 threads/SM and
  102,400 bytes shared memory/SM, not server-Blackwell resource limits. Query
  the device and inspect compiled registers/shared memory for the actual variant.
- [ ] **Calculate resident blocks for each tile/stage choice.** Threads,
  registers and shared memory all constrain residency. A 1,024-thread block
  cannot reach full thread occupancy on GB10; 256 threads only permits full
  occupancy if the other resources allow it. Do not treat a block size as a win.
- [ ] **Test grid-wide waits with the other serving streams active.** A software
  barrier holds resident blocks until every participant arrives. On GB10 the
  48-block b12x replay completed in isolation and stalled during TP2 graph
  execution. Check progress on both ranks with real traffic.
- [ ] **Time a read of the same selected weights.** For sparse verification
  tiles, replay the same expert IDs with a contiguous weight scan alongside
  the full projections. This measures a practical reference cost for reading
  the weights even when few of the tile's row lanes are useful.
- [ ] **Balance pipeline depth against occupancy.** Our native 128-row/three-stage
  variant allowed only one block/SM and lost badly to two stages. Check dynamic
  shared-memory opt-in where required; another stage can make latency worse.
- [ ] **Check register pressure and spills.** Inspect compiler/profiler evidence;
  do not trade fewer launches for excessive live state or local-memory traffic.
- [ ] **Reuse weight fragments across token rows.** Load/decode once and apply
  to multiple rows when possible, rather than independently repeating GEMV work
  for every token/expert assignment.
- [ ] **Preserve weight reuse in grid ordering.** Our first compact scheduler
  changed traversal order and weakened locality. Keep token tiles as the fast
  index within an expert where that benefits the consumer.
- [ ] **Skip empty work inside partially occupied tiles.** Avoid MMA on empty
  m16 slabs, not just out-of-range stores. The replay had 24,576 assignments
  but 39,296 padded rows; skipping empty slabs cut that to 27,424 live slab rows.
- [ ] **Dispatch tiny waves separately.** Preserve a small-row fast path; our
  final native implementation uses a 16-row tile for waves with at most 16 rows.
  Large prefill tiles must not silently replace efficient verification paths.
- [ ] **Remove redundant barriers without removing dependencies.** Reuse the
  next iteration's barrier when it already protects ring reuse. Verify all
  participating threads agree on barrier execution, including tails.
- [ ] **Charge compact scheduling for its producer.** Removing empty blocks can
  help, but list construction, ordering and indirection can erase the gain.
  Build lists on the GPU and reuse them across projections where possible.
- [ ] **Judge occupancy by time and ready work.** More eligible warps, fewer
  scoreboard stalls and better throughput can matter more than theoretical
  occupancy. Our prefetch win kept register occupancy unchanged and executed
  more instructions. Exact physical bandwidth numbers are not a prerequisite.

## Preserve efficiency through the caller

- [ ] **Quantize shared inputs once.** Reuse gate/up quantization across experts
  and paging waves within the layer. Invalidate cached readiness when the
  source or reused scratch changes.
- [ ] **Quantize only active down rows.** Use the existing GPU segment/tile list
  rather than converting the entire allocation for every wave.
- [ ] **Include conversion and preparation costs at their real frequency.**
  Separate one-time weight preparation from per-layer/per-wave work. Count
  quantization, gathering, packing, reduction and list production in totals.
- [ ] **Fuse useful work, not just launches.** Consider quantization, activation,
  reduction and layout production together, but check added registers, barriers
  and lost overlap. Conversion cost of about 0.3 ms could not explain a multi-ms
  native-kernel gap; fusion alone was not the missing optimization.
- [ ] **Avoid new CPU decisions and stream drains.** Keep routing/layout metadata
  on the GPU where possible. Do not add host readback, synchronization or
  allocation to resident expert hits or every paging wave.
- [ ] **Exercise first-use initialization before adding GPU waits.** Lazy CUDA
  loading and runtime setup can introduce hidden synchronization. Preloading
  function attributes was insufficient for the deferred bulk path: a real tiny
  startup reduction was needed before queueing a completion waiter. Test a
  fresh full-model process, dependent copies and failure release, not only a
  transport fixture that has already warmed the same operation.
- [ ] **Preserve CUDA graph replay and shared-expert overlap.** Prepare layouts
  and stable workspaces before capture. Keep the early shared-expert branch
  independent; do not serialize it behind routed expert preparation.
- [ ] **Preserve read/copy/compute overlap.** Avoid a new wait-for-all barrier
  before executing ready experts; retain buffer ownership until consumers finish.
- [ ] **Account for scratch in the cache budget.** Reuse disjoint existing scratch
  with size/alignment/lifetime guards where safe. Extra persistent buffers can
  displace hot experts or filesystem cache and reverse a kernel-level win.
- [ ] **Distinguish selections from misses.** Equal top-k does not mean equal
  I/O. Changed arithmetic can alter routes; a working set just beyond capacity
  can keep evicting itself indefinitely. More warm-up will not fix that.
- [ ] **Confirm dispatch on both ranks.** Check the actual kernel selected,
  supported shapes and fallbacks. Ensure new switches reach both ranks through
  the launcher and are accepted by the deployment-config parser.

## Compare fairly, then select for serving

- [ ] **Check arithmetic and layouts with meaningful inputs.** Use independent
  gate/up/down data and nonuniform scales; cover tails, offsets, strides,
  chunk views and graph replay. Scheduling-only changes should preserve the
  intended arithmetic; new quantization needs separate numerical assessment.
- [ ] **Separate kernel consistency from model quality.** Agreement with a
  quantized reference proves implementation consistency, not unchanged model
  quality. For precision changes, inspect actual outputs and speculative
  acceptance as well as numerical error.
- [ ] **Run a warm comparison that exposes execution cost.** Record expert reads
  on both ranks and distinguish expert residency from prompt-prefix reuse.
  Disabling prefix reuse is necessary to measure full warm prefill. Then check
  cold/miss behavior separately; neither alone represents all use.
- [ ] **Use matched timing and small causal comparisons.** Same routes, shapes,
  settings and binary where possible; toggle the suspected optimization rather
  than changing several causes at once. Keep profiler captures separate from
  uninstrumented timings. Do not sum overlapping stall/phase counters.
- [ ] **Measure the complete serving path.** Kernel wins do not automatically
  survive attention, Engram, communication and scheduling. Our projection gains
  translated to only about 2.7% lower warm-prefill latency in the final screen.
- [ ] **Separate decode speed from acceptance changes.** Record accepted drafts
  and pass counts. Changed continuations can raise tok/s without making the
  verification kernel any faster. Report prefill and engine-only decode separately.
- [ ] **Complete a real tool round trip.** Short capped continuations are
  throughput probes, not quality checks. Give the final answer enough tokens
  and use the intended tool policy: do not retain a forced tool call while
  asking the model to answer without tools.
- [ ] **Record scope and provenance without starting a broad campaign by default.**
  Keep focused diagnostics labelled as such. Official rows follow
  [Workload and timing](benchmarks.md#workload-and-timing): source revision and
  binary hash, zero process swap on every rank, matching operation streams,
  an idle fabric and zero cached tokens for cold probes. Keep generated artifacts
  untracked; restore the chosen production configuration once at the end.
- [ ] **Explain losses before rejecting the design.** Identify missing baseline
  features, cache-capacity changes and unsupported-path fallbacks. Fix concrete
  omissions and remeasure. A negative result applies to the implementation and
  workload tested, not every possible implementation of that arithmetic.

## Evidence behind the checklist

- [Kernel attribution](mxfp4-kernel-attribution.md): disabling existing weight
  lookahead cost about 20–21%; adding it reversed the native prototype result.
  Gathering prepared weights cost roughly twice direct slab consumption.
- [Cache and kernel follow-through](mxfp4-kernel-followthrough.md): 41 additional
  distinct experts roughly doubled misses near capacity; a 768 MiB/rank pool
  increase removed repeated-request expert reads. Compact scheduling and higher
  occupancy did not by themselves establish serving gains.
- [Native serving integration](mxfp4-native-serving.md): direct prepared/chunk
  views, scratch reuse, quantization reuse, tail skipping and small-wave dispatch;
  final tile/stage comparisons and the limited end-to-end gain.
- [Initial kernel alternatives](mxfp4-kernel-options.md): hardware/shape corrections,
  shape-dependent dense gains and earlier negative results later revisited.
  Read the follow-ups before treating those initial conclusions as current.
