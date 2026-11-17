# Nonblocking bulk-reduction handoff

The large-prefill CPU capture on both Sparks found the model execution thread
spending much of its CPU time in CUDA's stream-synchronization wait. The caller
was `BusStreamReducer::reduce`: drain the producing stream, submit a bulk
allreduce, then wait for the result before submitting more model work. CPU
utilisation here was not evidence that arithmetic on the CPU was the bottleneck.

`DGPP_BULK_STREAM_ORDERED=1` changes the large-buffer path:

1. Record a producer-ready event and submit the request immediately.
2. The existing bus engine polls readiness without blocking its progress loop.
3. Queue a one-warp GPU kernel that waits on a mapped completion word.
   The submitting thread can enqueue subsequent model work. The waiter uses
   a system-scope acquire load and yields with nanosleep between polls.
4. After the final allgather kernel retires, the engine publishes completion.
   Dependent GPU work can then proceed without a host resubmission gap.

Readiness events and the mapped word are allocated once per reducer, not per
fold. Completion generations increase monotonically; the word is never reset
while a GPU wait may still reference it. A separate event tracks retirement of
the GPU wait for teardown, even if the model stream is destroyed first. The bus
checks actual final-kernel retirement, not just the kernel's earlier done stamp.
Failure paths release the GPU wait too; the host still checks the collective
verdict before exposing the model result.

The switch requires `CUDA_MODULE_LOADING=EAGER` before CUDA initialization;
startup rejects lazy loading. First-use kernel loading can synchronize the
context while holding the driver lock needed by the bus, deadlocking deferred
GPU waits. A raw driver stream-memory wait and a separate signal-stream event
bridge both passed transport fixtures but stalled at the first bulk reduction
in serving, including with `CUDA_DEVICE_MAX_CONNECTIONS=32`. They are not used
by the implementation. The GPU-kernel waiter also stalled without an actual
startup bulk reduction.
A four-byte synchronous bulk reduction during stream binding resolved the
full-model stall. It exercises both reduce-scatter and allgather before the
first deferred request. Function-attribute preloading alone was insufficient.
The exact internal CUDA setup dependency was not identified; neither a higher
stream priority nor a noncooperative single-block diagnostic fixed it. Those
diagnostic changes are not retained.

The switch defaults off; the focused serving comparison did not justify
enabling it in production. Existing decode/graph paths
retain their dispatch. World one remains the identity. The production bulk
reduce-scatter/allgather arithmetic and rank order are unchanged.

## Scope and remaining joins

This removes the prefill-class `reduce()` producer drain and its immediate
host completion wait. It is not an unbounded asynchronous collective queue:
the existing bus permits one outstanding bulk request. The reducer retires it
at the next collective or final `settle()`, after intervening model work has
been submitted. Small stream collectives still settle before bulk submission.

The optional split-row prefill-overlap path (`begin_async_after`) is unchanged;
it is disabled in the current deployment. The new path does not automatically
make two dependent transformer layers execute simultaneously. Other model
synchronizations, route readbacks and Engram work remain. Time sampled inside
CUDA synchronization includes productive GPU execution and cannot all be
counted as recoverable time.

## Validation

The distributed two-Spark bulk fixture covers small buffers, segment boundaries,
small tails and multi-segment buffers against the existing bitwise oracle.
For the new path it deliberately holds the producer behind a callback: submission
must return before that producer is released. It also checks dependent reads
and reuses completion generations with a small stream reduction between them.
The deployment parser propagates the switch to both ranks.

## Focused serving comparison and deployment decision

On 2026-11-17, two GB10 ranks ran the same MXFP4 tool-enabled workload,
whole-expert budget of 108,047,546,368 bytes per rank, 12 execution slots,
eight read workers, one-million-token KV capacity and speculative depth three.
Both arms used the same executable, `CUDA_MODULE_LOADING=EAGER`,
`CUDA_DEVICE_MAX_CONNECTIONS=32`, and a 128 MiB prefix cache. Eager loading
failed the normal memory-headroom guard with the 1 GiB prefix reservation;
only that reservation was reduced, identically on both arms. The expert
pool and KV capacity were unchanged. The asynchronous arm ran first.

These are focused C1 diagnostics, **not campaign benchmark rows**. The
source was `0e4b226a8dad87a98f9b051018da183edffc0c53` plus this implementation;
the measured tracked-diff SHA-256 was
`904dff69b258b711f0377cddec49e8e62b658efb17238fc5f758f097ec1bcfef`.
The executable SHA-256 was
`614fbee76cb2ae273babaf154091feb5b1046b5abc1d6adbf56461b6a1491571`.
Subsequent source changes clarified comments and documentation only.

| Engine measurement | Existing handoff | Deferred handoff |
| --- | ---: | ---: |
| First 24,317-token prefill, zero cached tokens | 35.404 s | 35.486 s |
| Warm experts, zero prefix reuse, median of final three prefills | 11.026 s | 11.006 s |
| Warm prose decode, C1 | 39.09 tok/s | 39.11 tok/s |
| Warm code decode, C1 | 42.79 tok/s | 42.97 tok/s |

Decode figures cover engine decode steps only. Warm full-prompt prefills had
zero expert reads on both ranks. Both ranks had zero swap and matching rank
operation streams. All nine paired responses had identical output content
(ignoring generated tool-call IDs), acceptance counts and expert-read byte
counts. A single ordered pair of runs does not establish a sub-percent gain.

The approximately 0.2% warm-prefill difference does not justify trading prefix
capacity for eager loading. Production retains the original handoff and its
1 GiB prefix reservation. This feature is available for further targeted work,
not enabled as a claimed speedup. The high CPU utilization that motivated it
included waiting for real GPU work; deleting this particular host wait does
not delete that work or the later routing synchronization.

The final transport fixture also checks failure: holding the producer past
the bus watchdog must report failure and release the completion waiter. It
covers dependent GPU kernels and pinned-host copies queued before producer
release. The deployment environment tests cover forwarding the opt-in and
CUDA settings to both ranks.

## Explicit memory-headroom override

`DGPP_MEMORY_HEADROOM_MIB` overrides the startup reserve (default 4096 MiB).
It accepts integer values from 1024 to 16384 MiB and is forwarded to every
rank by the deployment launcher. It does not resize the expert pool, prefix
cache or KV cache, and the memory-plan check still rejects a plan whose total plus the configured reserve exceeds
available memory. This reserve covers memory
outside the model plan; lowering it is an explicit deployment choice, not a
memory saving. The repeated comparison requested after the initial screen
uses 3072 MiB and keeps the normal 1 GiB prefix cache in both arms.

## Repeated comparison with the full prefix cache

A second, fixed-size comparison on 2026-11-17 kept the full 1 GiB prefix
cache and used the explicit 3072 MiB reserve on both arms. It completed
144 requests across twelve service starts: six on/off pairs, each with two
warm-up requests and eight measured 24,317-token prefills per arm. Pair
orders were off/on, on/off, off/on, on/off, off/on, on/off. Three pairs also
included prose and code, each requested twice per arm; the second request
provided the warm decode sample. The run was not stopped on a favourable
intermediate result. All requests were tool-enabled, greedy and C1.

Both arms used eager CUDA loading, 32 CUDA connections, the same expert
pool, KV capacity, depth-three speculation and the same executable. Source:
`f44b7456` (the working tree committed as that revision); binary SHA-256:
`80567e8f4402c93219023bb32c7403f1ed973f46f3793f08b358025d8f8e5ad0`.
These remain focused diagnostics, not the full serving campaign row set.

| Measurement | Existing handoff | Deferred handoff |
| --- | ---: | ---: |
| Mean warm prefill, 48 requests per arm | 11.1224 s | 11.1570 s |
| Mean warm prose decode, three samples per arm | 39.03 tok/s | 39.00 tok/s |
| Mean warm code decode, three samples per arm | 42.82 tok/s | 42.85 tok/s |

The primary analysis averages log prefill duration within each boot, then
computes six paired differences. A Student-t interval with five degrees of
freedom treats the paired boots as replicates; it does not treat the 96
requests as independent observations. The estimated deferred-path latency
increase is **0.313%**, with a paired **95% interval from 0.048% faster to
0.676% slower**. Every pair favoured the existing handoff:

| Pair | Deferred-path latency increase |
| --- | ---: |
| 1 | 0.914% |
| 2 | 0.034% |
| 3 | 0.143% |
| 4 | 0.155% |
| 5 | 0.091% |
| 6 | 0.547% |

This does not establish an exact regression at 95% confidence: parity and a
very small gain remain inside the interval. A 0.2% gain is outside this
interval for this workload and configuration. The initial 128 MiB-prefix
screen and this full-prefix comparison are different configurations; they
must not be pooled. No first-pair exclusion or favourable-subset selection
was used. Decode figures are descriptive secondary measurements, with only
three paired warm samples per class.

All 96 measured prefills had zero prefix reuse and zero expert-read bytes on
both ranks. All 72 paired requests, including warm-ups and decode probes,
had identical response content (ignoring generated tool-call IDs), acceptance
counts and expert-read byte counts. Every arm had matching rank operation
streams and both rank processes reported zero swap. The lowest sampled host
available memory was 1.407 GiB on rank 0 and 1.446 GiB on rank 1. There were
no allocation failures or OOMs: the full prefix cache did fit.

Production was restored with the tested executable, the original CUDA loading
settings, the full 1 GiB prefix cache and the deferred mode disabled. That
choice follows the repeated results favouring the baseline, rather than a
minimum percentage threshold for accepting an improvement. The override and
asynchronous implementation remain available for further work.
