# Spark Home Assistant bridge

These files preserve the installed Spark bridge and its existing vLLM discovery
identities. `spark_dgpp_ha.py` imports `spark_vllm_ha.py`; deploy them beside the
existing `spark1-ha.py` using its Python environment (`prometheus-client`). The
compatibility adapter requires the existing discovery backup at
`~/.local/share/spark1-ha/vllm-discovery-backup.json`. Do not replace that backup
or rename entities when updating the adapter.

DGPP `/v1/metrics` now exposes `scheduler.spec_decode`:

- `depth`: configured maximum speculative depth.
- `num_drafts_total`: completed per-request verification rounds with at least
  one speculative token (first-position attempts).
- `num_draft_tokens_total`: sum of speculative positions submitted for verification.
- `num_accepted_tokens_total`: sum of accepted speculative positions.
- `num_draft_tokens_per_pos_total`: verification attempts, zero-based positions.
- `num_accepted_tokens_per_pos_total`: accepted counts, zero-based positions.

Counters are cumulative since engine startup, exclude padded requests and the
non-speculative token, and are published after completed engine passes. They
measure verification decisions before response stop/length trimming, not just
emitted tokens. Unused lookahead generation and unverified positions are not
counted. For variable-depth scheduling, the draft-token total is the sum of
position attempts, not rounds multiplied by maximum depth. Counters should be
read from rank 0 once, not summed over TP ranks.

The compatibility adapter maps totals and accepted positions to the original
`spec decode ...` sensor names, preserving discovery topics, unique IDs and
state keys. Missing/unsupported positions remain unavailable. Dashboard ratios
use accepted tokens / rounds, drafted tokens / rounds, and position accepts /
all rounds; guard division by zero. Position ratios are not conditional
acceptance probabilities. Existing backend fallback and reset behavior remain.

Run adapter tests with `python test_mtp_metrics.py` in this directory using the
bridge's Python environment. The HTTP integration test
`serve_specDecodeMetrics_preserveRoundDenominator` covers non-MTP and unequal
position-attempt counts. Test deployment should retain the previous server
release and adapter for rollback.

## Validation and Spark deployment (2026-09-18)

- ARM64 server build passed; all 38 `serve_test` HTTP/service tests passed.
- Four adapter tests passed (variable depth, missing metrics, zero/reset,
  invalid values).
- Both ranks run release `0.1.0+g42b105d-mtpmetrics`, binary SHA256
  `ab1b88a00de32dedb0de63d285e584651d69c99185c106c4e6d7e549b9585fae`.
- A live 64-output-token request produced 19 rounds, 57 verified draft
  tokens, 45 accepts and position accepts `[18,14,13]`. This is a smoke
  test, not a workload acceptance benchmark.
- The installed compatibility adapter returned those counts using the original
  discovery configurations; the HA bridge was restarted and is active.
- The stable cluster config has a `.before-mtpmetrics` rollback copy;
  `~/spark_dgpp_ha.py.before-mtpmetrics-20260918` preserves the previous adapter.
- Local build/test logs are under `artifacts/mtp-metrics-20260918/`.
