# Decode graph bucket telemetry

The metrics endpoint now exposes scheduler.decode_batch: last launched
slot capacity, active requests, rows per request, cumulative launches by
slot capacity, and total/padded verification rows. Recording happens after
successful CUDA graph launch; the scheduler publishes the snapshot through
its existing meters path and the HTTP service reads its locked copy.
No GPU synchronization or device copies were added for these counters.

Last values remain visible while idle or prefilling. Counters cover scalar
and batched verification graph launches, not draft-chain kernels, acceptance
or elapsed GPU work. For sparse occupancy, padding counts the actual selected
family capacity minus participating requests. Histogram key 1 is scalar;
zero-count keys do not advertise supported families. Counters reset on restart.

Validation: ARM64 build passed. The focused two-rank Qwen FP8 C16/MTP3 GPU
suite passed (2 tests, 0 failures). Added assertions check the selected shape,
per-replay row totals, histogram increments and padding after cancellation.
The previous serving epoch shut down before testing, and the new build started
on both Sparks with the existing 512-token prefill and C16/MTP3 settings.
Full GPU/RDMA suites were not rerun; this is observability, not a performance
optimization or throughput benchmark.

Live HTTP validation passed: five concurrent requests completed; JSON parsed,
replay histogram summed to the total, padding stayed within total rows, and
batched occupancy was observed. A subsequent sample reported six slots,
five active requests and four rows/request, directly exposing four padded
rows in the last replay. Independent client traffic was present. The service
remains running; local evidence is in artifacts/batch-metrics-20260918.
