# Reserved expert-group reads

The paged MoE path submits complete execution groups ahead of useful GPU work.
The scheduler preserves exact routes, expert arithmetic, cache admission policy
and the existing final token/route-slot accumulation order. It does not predict
experts or enlarge the memory budget.

## Submission and ownership

`ExpertReadQueue::reserve_demand(records)` is a nonblocking, all-or-none
capacity operation for unique record IDs. It checks queue entries and physical
staging destinations together. Already-running reads and completed tickets
reuse their destinations; queued jobs receive a reserved slot before workers
can claim them. The batch shares existing tickets and promotes speculative
jobs to demand priority. Submission uses one queue lock and one final batch
notification. Allocation failure remains exceptional; ordinary capacity
refusal submits nothing.

Reserved runnable jobs take priority over unreserved demand. A worker waiting
for a free buffer yields its unreserved job if reserved work is waiting. This
prevents all workers from blocking behind buffers owned by queued reserved
jobs. Shutdown/cancellation returns unconsumed slots; after a read starts,
its lease retains the slot through the last CPU/GPU consumer as before.

`ExpertCache::prime_group()` deduplicates experts, excludes resident chunks,
reaps completed copies and reserves the complete missing-record set. It
records no accesses or admissions. Existing queued results, including errors,
remain authoritative. Small staging configurations which cannot reserve a
complete group retain the ordinary bounded streaming fallback.

## Execution schedule

The existing selected-expert order and wave cursor form the schedule. Prefill
partitions the resident subset first and reserves the first streamed group
before constructing/uploading the resident layout or launching its kernels.
Once the current group's source ownership is established, it reserves and
submits the following group before the current group's kernel launches. A
buffer held by an unfinished copy or chunk-direct GPU reader cannot be reused.
Ready-group selection and ordered output accumulation remain in place.

The b12x small-batch miss path also reserves its first reconstruction group
before shared/resident execution. Independent reconstruction retains its
existing scratch-group size and tries to reserve the following group before
launching completed experts. It preserves tickets belonging to that following
group instead of discarding them as unrelated prefetch.

The early routing hook now starts a complete bounded group instead of two
experts. Its limit comes from half the staging pool and half the execution
slots, using the store's common expert geometry. Subsequent execution paths
use their actual group boundaries. `DGPP_EXPERT_GROUP_READS=0` restores the
previous two-expert/rolling-lookahead schedule for diagnostic comparisons.
The setting defaults to enabled. The existing early-demand and ready-worker
controls remain independent.

## Capacity on the two-Spark NVFP4 deployment

Each rank has 256 staging slots of 544 KiB: 136 MiB. One expert shard is 18
records, or 9.56 MiB. A normal streamed prefill wave uses at most six experts:
108 slots, or 57.38 MiB. Two such groups use 216 slots, or 114.75 MiB, leaving
40 slots for other outstanding consumers. This is capacity arithmetic, not a
guarantee that those slots are always free. Reservations use actual free slots;
if unavailable, execution falls back to bounded progress rather than stealing
live destinations or waiting for the whole layer.

The independent b12x reconstruction path has up to 12 scratch experts per
group. Its sources are copied into reconstruction storage, allowing staging
to retire independently of expert execution. A following complete group is
submitted when sufficient destinations have retired. The check also preserves
credits for current reads which are queued without destinations, preventing
future reservations from starving the current group. Otherwise the ordinary
rolling submission continues. This change does not reserve 384 expert shards,
change execution wave sizes, or add more SSD workers.

## Validation

Host coverage includes atomic capacity refusal, ticket/destination coalescing,
reserved work progressing despite buffer-blocked workers, shutdown cleanup,
exact reconstructed payloads and a check that the next group's reads are
accepted before the current group's execution callback. The Spark checks also
exercise the existing B12x supplied-route GPU path and paged expert-wave
bitwise equivalence through 4,096-row prefills.

Performance comparisons here are focused diagnostics, not deployment-template
benchmark rows. Generated request logs and measurements remain untracked.


A focused two-rank comparison used the same binary with group scheduling off
and on, fixed speculation depth three and `DGPP_EXPERT_READY_WORKER=2` in both
arms. Source was `31cc1b7d` plus this implementation; binary SHA256 was
`ced82f672d3d438006b464f01d39f9b0cda0da8a0fcff76ec4d8896f1c0bf958`.
Both arms used the same six tool requests and warm-up order. All response
choices and token usage matched after normalizing generated tool-call IDs.
Operation streams matched across ranks within each arm, and both ranks had
zero process swap before and after the measured requests. GPU checks ran with
serving stopped before the comparison; no builds or other GPU worlds ran on
the Sparks during it.

The 271-token continuation's prefill changed from 1.829 to 1.808 seconds;
the 308-token continuation from 2.396 to 2.254 seconds (about 6% shorter).
The 15-token tail changed from 0.203 to 0.207 seconds. The repeated 2,209-token
full prefill, with warm expert/I/O history and zero cached prompt tokens,
changed from 2.537 to 2.573 seconds (about 1.4% longer). Its preceding first
prefill was 7.181 versus 7.161 seconds. Initial 24,317-token prefill was
38.686 versus 39.284 seconds. Request decode durations ranged from about
1.1% shorter to 2.1% longer. These single matched observations support a
localized gain, not a general throughput or prefill speedup claim. No new
formal deployment benchmark rows are established by this diagnostic.

Validation passed 31 cache tests, 13 queue tests and the two focused GPU tests.
The cache group test additionally covers a pool large enough for two complete
groups and a pool that requires recycling. The deployed setting enables group
reads; port, authentication, cache/staging budgets and eight I/O workers per
rank are unchanged. The fallback is `DGPP_EXPERT_GROUP_READS=0`.
