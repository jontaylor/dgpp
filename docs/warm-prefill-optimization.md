# Warm prefill optimization

This work separates warm expert execution from prefix-cache reuse and cold
storage costs. The diagnostic replays a tool-enabled request with
`prefix_cache=false`: every request computes all 2,209 prompt tokens. After
warming, both ranks report 100% expert hits, zero expert reads, zero
reconstruction/layout preparation, and zero swap. The deployment retains
TP2 NVFP4, 1M context, the 100 GiB expert reservation per rank, four expert
read workers, direct Engram caching and fixed speculative depth three.

| Configuration | Warm prefill (ms), two successive requests |
|---|---:|
| Previous production | 2,080 / 2,086 |
| Resident-first grouping | 1,951 / 1,952 |
| Resident-first + contiguous prepared-weight copies | 1,418 / 1,412 |

These are focused serving diagnostics, not a broad throughput/concurrency
campaign. Responses, tool arguments, usage and speculative acceptance match:
50 output tokens, 13 target passes and 36 accepted predictions. Decode remains
roughly 0.92–0.93 seconds; the changes target prefill. The first resident-first
request and the copy candidate each read 95.317 GiB per rank and took 32.715
and 31.806 seconds of prefill respectively. Those cold results are reported
separately, rather than used to dismiss the warm improvement.

The prepared NVFP4 layout follows MMA lane ordering. The old prefill kernel
gathered it byte by byte into canonical row order, then decoded it into MMA
fragments again. The new path copies contiguous 512-byte n64/k16 slabs with
`cp.async`, retaining the prepared layout in shared memory and decoding the
fragments directly. It preserves the ascending-k16 accumulation chain, scale
arithmetic, BF16 intermediate and FP32 down outputs. The existing fixture
compares prepared and canonical results bitwise, including mixed misses and
cache pressure. `DGPP_MOE_PACKED_PREFILL_COPY=0` retains the previous reader
for diagnosis; the contiguous copy is the default.

Resident-first grouping is controlled by `DGPP_MOE_RESIDENT_FIRST=1`.
Its previous cold-heavy pair did not demonstrate a gain, but the zero-read
warm case does. This illustrates why cold, warm and prefix-cache performance
must be assessed separately.

A short two-rank warm capture of the copy candidate found about 0.59 seconds
in routed-expert prefill, 0.21 in attention, 0.19 in dense projections and
0.13 in communication. GPU-idle gaps were about 21 ms per rank. Categories
are attribution within a diagnostic trace, not additive benchmarks; the
prefill-to-first-target-graph interval also includes the transition into
generation. Host synchronization API durations include waiting for GPU work
and must not be interpreted as additional idle time.

A larger 4,559-token request continues to read approximately 37 GiB per rank
on repeated passes. It is a warm-cache, miss-heavy workload, not an all-resident
execution measurement. A fourfold improvement has not been demonstrated;
remaining costs span expert kernels, attention, dense projections and
communication, rather than a single CPU gap or an established SSD limit.

The compact-tile follow-up passed bitwise fixtures but recorded 1,424 / 1,421 ms
warm prefill, versus 1,418 / 1,412 ms for contiguous copying alone. Although it
removed empty blocks from highly skewed launch grids, it did not improve serving.
That candidate was removed rather than enabled or retained as another runtime
option. The final deployment keeps resident-first grouping and contiguous copies.

The baseline runtime source is `8eea0cfc`. The retained candidate was built from
that source plus the prepared-copy kernel change in this delivery; the version
stamp predates the final commit. Both ranks had matching binary hashes:

| Runtime | SHA-256 |
|---|---|
| Baseline / resident-first | `d8398763ccb2006e1c0c7d768cf22e6b5c404a6557654330576974eb87e9b47c` |
| Retained prepared-copy implementation | `d6f15940e8e458194135ceef6d478331eb5ff129decec276bd32508f00bafa6e` |
