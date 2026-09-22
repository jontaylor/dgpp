# MiMo incremental rolling prefix snapshots

Base: `c91a50b` (MiMo native MTP3 with cache-only draft prefill). This feature
changes snapshot traffic, not the four-slot allocation, snapshot layout,
scheduler policy, attention arithmetic or MTP state representation.

A registered arena destination remembers its request and copied committed
position. A refresh from that same request copies only new global K/V rows.
Sliding-window snapshots are still fully packed from the expanded ring, and
all MTP snapshots retain their existing full-copy path. Raw session snapshot
buffers do not opt in and keep full-copy behavior.

Explicit arena release, prefill destination reservation, destruction, request
reset/close, restore and reassignment invalidate provenance. A rollback that
crosses a remembered position invalidates it before a later append can overwrite
that history. Rejection after a saved committed prefix preserves the prefix.
Graph settlement only advances the host committed position; the snapshot runs
outside the graph and uses that position, including post-row hop offsets.
No scheduler or graph journal changes are required.

## Local checks

- The host provenance regression passes with `-Wall -Wextra -Werror` and
  ASan/UBSan. It covers four destinations, failed writes, request reassignment,
  release, rollback past and after a saved position, reset and unregistration.
- `mimo_snapshot_test` compiles for ARM64 against CUDA 13 with warnings as errors.
  The exact-copy probe also passes under a temporary local host CUDA-memory
  emulation with ASan/UBSan: 6,945,280 copied bytes. Emulation validates test
  logic and bounds; it is not GPU execution or stream-order evidence.
- Full ARM64/CUDA 13 server and `mimo_snapshot_test` CMake crossbuild passed
  with `DGPP_BUILD_JOBS=2` after the storage relocation described below.
- No Spark access, remote API calls, deployment, model execution or GPU tests
  were performed by this feature task.

## GPU handoff

Build with `DGPP_BUILD_JOBS=2 scripts/spark-cross build --target dgpp_serve_app
mimo_snapshot_test`, or build the same targets natively. The local Docker bridge
was absent; crossbuild used a temporary Docker wrapper adding `--network none`
to `docker run`, without changing the repository build script. Docker overlay
storage then filled during the unrelated large MoE kernel build. Preserved build
objects were relocated to `/mnt/benchmarks/dgpp-pipeline-builds-20260922/snapshots`
with a worktree symlink; that root and a per-feature compiler temp directory
were bind-mounted into the container for the retry.

On idle target hardware, run `mimo_snapshot_test` and then
`compute-sanitizer --tool memcheck --error-exitcode=1 ./mimo_snapshot_test`.
The probe needs no checkpoint or network. Exit 2 means no GPU and is a skip,
not a pass. Its independent host oracle compares every populated K/V element
bit-for-bit against the full-copy cache view. Restore also compares every
unpopulated tail element against poison, so an oversized restore fails.
It exercises four slots, multiple ring wraps, post-row hops, rejected tails,
rollback followed by overwrite beyond the old boundary, close/reset reuse,
restore, destination release, owner reassignment and prefill reservation.
Each rolling refresh asserts its expected byte count.

The probe exercises production copy/restore helpers and PrefixArena hooks with
a synthetic model. It does not validate real-model MTP logits or engine graph
execution. Before adoption, run the existing strict real MiMo MTP3 cached/cold
continuation comparison on both ranks, including rejected drafts, a cancelled
request, more than four distinct prefixes and a cached continuation after reuse.
Compare with the full-copy base under the same tokens, sampling and configuration;
retain the existing numerical gates. Test eager and scalar/batched graph paths.

## Traffic accounting

`MimoModel::snapshot_copied_bytes()` reports enqueued base K/V copy bytes;
`snapshot_saved_bytes()` reports global K/V bytes omitted versus full-copy
snapshots. Both are per-rank cumulative diagnostic getters, not HTTP metrics.
They exclude unchanged MTP snapshot copies. The GPU probe asserts traffic directly.

For the current TP2 model, nine global layers require 11,520 bytes per token per
rank. A same-owner refresh from P to P+d copies 11,520*d global bytes instead of
11,520*(P+d), plus the unchanged bounded sliding-window and MTP state. At P=32,768,
one refresh avoids 360 MiB of global traffic per rank. This is a byte-count
prediction, not measured end-to-end latency or throughput improvement.
