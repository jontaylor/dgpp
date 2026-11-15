# Chunk-direct expert execution

An opt-in NVFP4 prefill path now consumes missing experts directly from the
registered SSD-read buffers. It removes the full-expert reconstruction copy,
but the initial copy-only comparison did not demonstrate a meaningful serving
improvement. The later memory-reclamation comparison below keeps chunk-direct
enabled with fewer reconstruction slots. BF16 activations remain enabled.

## Implementation and limits

`DGPP_EXPERT_CHUNK_DIRECT=1` enables the path on both ranks. A bounded device
pointer table occupies an existing execution slot; the W4A16 tensor-core tile
follows it for gate/up rows and down-projection columns. The kernel preserves
ascending accumulation and applies the original global scale once per complete
dot product. No weights or activations are requantized.

Read leases stay alive through GPU consumption. Transfer completion and kernel
reader completion have separate events, so a later coalesced transfer cannot
erase an earlier reader's lifetime fence. Completed staging leases are retired
as a prefix rather than repeatedly scanning and erasing the entire live vector.
Admission still copies retained chunks into the persistent cache. Native hits
keep their existing prepared layout and fast execution path.

This implementation handles complete misses in NVFP4 prefill with power-of-two
chunk widths of at least 64, dividing the rank's intermediate dimension.
Partial hits and unsupported shapes use the existing reconstruction path.
Decode/speculative verification retains B12x; it is not chunk-direct. The
full-expert scratch reservation also remains because those other paths need it.
This is not independent execution of whichever chunk becomes ready first: a
wave's required chunks are acquired before its kernels launch.

The source buffers must cover the active and following waves. At the deployed
18 chunks per expert and 12 experts per wave, the focused comparison used 512
buffers/queue entries, four I/O workers and the unchanged 100 GiB reservation
per rank. Staging consumes approximately 0.27 GiB within that reservation.
The first 256-buffer version reduced wave size and read-ahead and regressed;
the final version restores 12-expert waves and two-expert demand look-ahead.

## Serving result

These are focused diagnostics, not formal deployment-campaign throughput rows.
The same tool-enabled prompts and warming sequence were used on both ranks,
with prefix reuse disabled, fixed speculative depth three, 1M context and the
same memory budget and 512-buffer configuration. No competing Spark workload
or fabric transfer ran during requests; compilation ran on the separate build
host. Both ranks stayed at zero swap.

| Prefill workload | Reconstruction, seconds | Chunk-direct, seconds |
|---|---:|---:|
| 2,209 tokens, cold | 25.037 | 25.006 |
| 2,209 tokens, steady warm | 1.417 / 1.418 | 1.421 / 1.424 |
| 4,559 tokens, first after smaller prompt | 7.042 | 7.363 |
| 4,559 tokens, repeated miss-heavy | 10.476 / 11.004 | 10.583 / 10.286 |
| Original 24,317-token tool prompt, after that sequence | 25.126 | 24.695 |

All eight final responses, normalized tool arguments, token usage and speculative
acceptance matched the control. Cold reads matched at 95.317 GiB per rank; warm
requests read zero expert bytes. Miss-heavy read volumes vary slightly with
cache allocation. The roughly 0.43-second long-prompt difference is too small
in this comparison to justify a production switch. Decode remains unchanged
in implementation; no decode speedup is claimed.

The control is implementation revision `8f8f0cdc`, with SHA-256
`b54e66dea53677a8954c2e7124b3ab6fd70830a26a2d21820f8eb8ded3d038b7`.
The measured candidate is `341e61e1` plus this implementation, with SHA-256
`b8541ae7b8719ba09b04898b5cba81aec084ba52811be35bc70ba57ca1fb2694`.
The version stamp predates the final commit. The final release rebuild changes
that stamp. At the end of that comparison production returned to 64 staging
buffers and queue entries. The memory-reclamation follow-up below supersedes
that configuration, retaining the endpoint, keys and token limits.

Validation: all 27 host cache tests passed. The B12x paging fixture explicitly
exercised chunk-direct prefill, checked zero reconstruction-copy bytes for those
misses, and matched canonical outputs bitwise. The staging lifetime fixture
covers a later transfer sharing a buffer with a pending kernel reader. The
generic NVFP4/MXFP4 paged-wave fixture also passed.

## Precision tradeoff screen and remaining work

A separate short screen enabled the existing native FP4 activation path with
dynamic scaling (`DGPP_MOE_W4A4=1`, `DGPP_MOE_W4A4_DYNAMIC=1`). Steady warm
2,209-token prefill rose to 3.444 / 3.449 seconds, from about 1.42 seconds with
BF16 activations. It also changed speculative acceptance. It remains disabled;
no broader quality qualification was performed because it failed the speed
screen. This does not establish that an optimized native FP4 implementation
cannot win, only that this existing path is not a useful deployment choice.

Chunk-direct decode, direct partial-residency reads and independent chunk
readiness remain unimplemented. Smaller scratch reservations are evaluated
below using existing configuration controls. No matched EXL3 2.9-bit serving or quality comparison was
run, so these results do not decide the user's model-format choice.


## Reclaiming RAM without enlarging the expert cache

The follow-up reduces real allocations rather than merely bypassing copies.
Both arms use implementation revision `a7320e7b`, binary SHA-256
`91afe785695b63768441e3184f33e1d698f631ed58209d8bb5b322cf62cc2c14`
on both ranks. No rebuild, competing workload or fabric transfer ran during
measurements. These are focused configuration diagnostics, not campaign rows.

| Per-rank setting | Prior production | Retained lower-memory configuration |
|---|---:|---:|
| Chunk-direct | Off | On |
| Reconstruction slots | 24 | 12 |
| Staging buffers / queue depth | 64 / 64 | 256 / 256 |
| Read workers | 4 | 4 |
| Total expert reservation, bytes | 107374182400 | 107242240000 |
| Accounted expert allocation, bytes | 103764337408 | 103632395008 |
| Resident chunk capacity, bytes | 103250891520 | 103250891520 |

Each reconstruction slot occupies 19,908,096 bytes; each staging buffer
557,056 bytes. Removing twelve slots and adding 192 buffers therefore releases
131,942,400 bytes (125.83 MiB) per rank. Lowering the total reservation by that
exact amount prevents the planner from giving the saving to resident experts.
The resident chunk capacity is identical. The configuration uses
`DGPP_EXPERT_CHUNK_DIRECT=1`, 1M context and the existing 262144 output default;
activation quantization remains disabled.

The sequence is four recomputed 2,209-token tool requests followed by the
original 24,317-token request. Prefix-cache reuse is disabled. All five outputs,
normalized tool arguments, token usage and speculative acceptance match, and
both ranks remain at zero swap. This warming sequence differs from the earlier
comparison, so its long-prompt time must not be compared directly to that table.

| Prefill diagnostic | Prior production, seconds | Lower-memory, seconds |
|---|---:|---:|
| 2,209 tokens, cold | 24.305 | 25.490 |
| 2,209 tokens, steady warm | 1.418 / 1.415 | 1.414 / 1.413 |
| 24,317-token tool prompt after that sequence | 21.829 | 20.780 |

Engine decode rates at concurrency one are essentially unchanged on the warm
small requests: 53.09 / 52.95 tokens/s before, 52.77 / 53.02 after. The long
request records 15.31 versus 15.58 decode tokens/s. These decode rates exclude
prefill; no broad concurrency or decode improvement is established.

After the long request, Linux `MemAvailable` was 3,595 / 3,893 MiB on ranks
0 / 1 before, and 3,962 / 4,202 MiB after. Those are whole-node snapshots;
the observed 367 / 310 MiB increases include runtime and system variation and
must not all be attributed to the exact 125.83 MiB allocation reduction.
Filesystem `Cached` stayed essentially constant: approximately 3,303 / 3,123
MiB in both arms. Thus RAM was available to the OS, but this workload did not
turn it into substantially more filesystem cache. Expert reads use `O_DIRECT`;
these results do not establish a filesystem-cache-driven speedup.

The lower-memory configuration is retained on both Sparks: it frees memory,
preserves resident expert capacity and warm speed, and improved the long
request in this comparison. The cold-prefill penalty of approximately 1.18
seconds is explicit. No claim is made that all reconstruction scratch has
been removed; decode and fallback execution still require twelve slots.
