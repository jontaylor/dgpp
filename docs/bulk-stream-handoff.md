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
