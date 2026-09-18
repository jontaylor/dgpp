# Qwen 512-token prefill experiment

Qwen's fixed prefill chunk size changes from 1024 to 512 tokens. Regular
prefix-cache cuts follow this size, while structural prompt boundaries
remain eligible. The scheduler still saves the deepest eligible prefill
snapshot, not a separate entry at every chunk.

For uniformly distributed prompt ends and chunk-only cuts, the expected
uncached tail falls by about 256 tokens. Structural boundaries and actual
agent histories determine the real savings. Smaller chunks may lower
prefill throughput; no net performance benefit has yet been established.

Validation on two DGX Sparks with Qwen3.8-Flash-Next-NVFP4:

- ARM64 server and Qwen GPU test target rebuilt successfully.
- Focused FP8 C16/MTP3 integration and real-shape projection tests passed
  (2 tests, 0 failures), including cached/yielded continuation.
- Previous service stopped with matching operation-stream digests on both
  ranks. Matching new server binaries were installed on both ranks.
- Full-model C16/MTP3 startup succeeded. KV capacity 2928640 tokens, 8 GiB
  prefix snapshots, queue 64 and port 30001 retained.

The restart begins with a cold cache, so initial service metrics should
not be compared directly against the previous warm-cache epoch. This is
a functional experiment, not a controlled throughput benchmark. Full
GPU/RDMA suites were not rerun for this constant change.
- Two identical HTTP prompts completed with 32 output tokens each. The
  repeat reused 1860 of 1867 prompt tokens (a structural boundary); this
  confirms cache reuse but does not isolate the benefit of 512-token cuts.
