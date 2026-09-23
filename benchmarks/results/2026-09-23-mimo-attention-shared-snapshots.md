# MiMo attention and shared-snapshot validation — 2026-09-23

Selected deployment: BF16 live KV, native MTP3, C2,262144 context per request and four prefix slots. API: `http://192.168.0.171:30001/v1`. Both ranks use source `03a7f5f`, binary SHA256 `68845fee37a20a99b4107bec90321e57ab363277cbc2f71d71205d290115f6e6`.

Enabled flags: `DGPP_MIMO_FUSED_PREFILL=1`, `DGPP_MIMO_FP8_DENSE=1`, `DGPP_MIMO_FP8_DENSE_PREFILL_BF16=1`, `DGPP_MIMO_SWA_QK_KEY_TILE=32`, `DGPP_MIMO_QK_GEMM=1`, `DGPP_MIMO_SHARED_SNAPSHOTS=1`. DFlash and FP8 KV are disabled. All newly added experimental options remain off by default in source.

## Performance

Same fixed prompts, temperature0, no competing requests. Two repetitions for each short class and31K, one67K cold seed, two cached67K repeats. Client decode timing starts at first streamed token; it is not upstream's engine-counter timing. These small panels do not establish a broad workload average.

| Measurement | Control | Selected | Change |
|---|---:|---:|---:|
| 7.2K cold TTFT | 7.443s | 7.024s | -5.62% |
| 30.9K cold TTFT | 56.601s | 56.734s | +0.23% |
| 67.4K cold TTFT | 232.558s | 222.210s | -4.45% |
| 67.4K cached TTFT | 0.398s | 0.397s | -0.34% |

C1 decode tokens/s: code 44.50, JSON 48.33, prose 28.09, math 48.40. Decode is essentially unchanged. First-request initialization and preflight warmup differ, so short first-request TTFT is not claimed as a speedup.

Owned snapshot storage after the67K workload: **1.047GiB per rank**, versus11.540GiB fixed allocation before. It grows with retained unique prefixes; it is not a fixed1GiB cap. `snapshot_storage_bytes` excludes CUDA allocator reserve/rounding. Live caches remain contiguous; this is snapshot sharing, not paged live KV or additional concurrency.

## Individual and combined candidates

- Attention-only:67K TTFT232.558->222.151s;19/19 output matches and all four acceptance vectors unchanged.
- FP8 bridge only:67K234.010s, removing nearly all the previous FP8 prefill penalty. It saves8.84GiB of planned KV/snapshot storage per rank, but same-output67K decode was about5% slower; keep as an opt-in capacity tradeoff.
- Shared-only, after the alignment fix:67K233.921s, decode unchanged,19/19 outputs and all four acceptance vectors matched. Owned snapshots1.047GiB after long workload; cached TTFT about26ms higher in isolation.
- All three combined (attention, FP8 bridge, shared snapshots):67K224.876s, same-output long decode about3% slower than control, owned snapshot storage0.645GiB.19 timing +4 acceptance +8 branch/reuse +3 actual-harness requests and31K exact retrieval completed. Not selected because FP8 reduced decode performance.
- Selected attention plus shared snapshots with BF16: table above.18/19 timing outputs match control; one concurrent code response differs in valid wording after a changed request arrival order. All eight C1 outputs, all cold/warm long outputs and four MTP acceptance vectors match. Do not claim universal bitwise equivalence; the focused order probe is retained separately.

## Correctness and limits

Both Sparks passed FP8 all-code conversion/graph gates, attention tests, compact/mapped graphs, and memcheck/racecheck before service timing. Host sanitizer/ThreadSanitizer checks cover shared ownership and concurrent metrics accounting. The numerical repair around BF16 halfway points passes the recorded attention gates; it is not a proof of exact equality for every activation.

An independent source review found the original compact snapshot's8-byte header misaligned native MTP copies. The incomplete `shared` panel was stopped and is not a performance result. `03a7f5f` uses16-byte aligned header and padded slot strides. The new GPU regression exercises actual SessionModel/PrefixArena MTP copies and restores across all four slots, hop/reuse, hidden rows and an old-layout negative control. Both GPUs and memcheck pass.

The selected service passed real native-MTP lifecycle, actual custom-harness tool arguments, branch/reuse, tool-cap checks, mixed prefill/decode and exact31K/67K retrieval gates. Final verification is recorded below. This is not a full256K C2 stress test or a broad HumanEval/SWE-bench quality evaluation.125K retrieval was not added to this final bounded validation phase.

PV tile32/64, integer FP8 load expansion and full-attention grouped-QK candidates regressed in microbenchmarks and remain disabled. Raw cuBLASLt QK without numerical repair failed its strict numerical gate; the selected path includes repair.

## Provenance and integration

Baseline149bc6f; control and attention binaries frombd48198. ca9ce0c corrected concurrent memory accounting;03a7f5f corrected compact MTP alignment. Inference kernels are unchanged across these revisions. Each deployed binary/flag set was attested on both ranks. Frozen binaries, raw requests/SSE, manifests, failed experiments and all logs are under `/mnt/benchmarks/dgpp-mimo-attention-20260923/`.

The baseline-checked20-file patch was applied to the original working checkout after backups; every resulting file matched the isolated worktree and the Git index remained unchanged. No upstream code was merged and no PR was created. The independent upstream comparison is in `2026-09-23-mimo-upstream-comparison.md`.

## Final verification

Completed:19 uncontended timing requests, four matching MTP acceptance vectors, eight branch/reuse checks with four actual cache hits, three actual-harness tool requests, five tool-cap/stream checks, exact31K and67K retrieval,11 native-MTP lifecycle requests (including cancellation/recovery), and mixed7K retrieval during a1024-token generation with20 two-slot decode replays. All functional assertions passed.

The additional10-request order probe matched the control solo text in6/10 cases: solo code/JSON and all JSON responses matched; code wording differed in both deliberately staggered concurrent orders, consistently across repetitions. The JSON-first/code-second variant reproduced the changed panel response. This establishes batch/arrival-sensitive text, not its cause; an identically staggered unmodified-control run was not performed. It is not evidence of full greedy bitwise equivalence or, by itself, proof of request-state contamination.

Final attestation: both ranks have identical binary hash and MiMo flags, health is OK, zero failed requests, no ERROR/FATAL rank-log entries, active/queued0/0, four prefix slots and MTP depth3. The verification record is `raw/final-verification/`, with `final-health.json`. Original Git index remains unchanged.
