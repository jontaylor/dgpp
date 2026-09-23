# Parent-run DFlash service correctness panel

`tools/mimo_dflash_service_check.py` performs no deployment changes. Run it
only after the parent has independently verified both ranks' binary/config/
environment and cleared external traffic. The preparation agent did not make
network requests or access either Spark.

Use the same checkpoint, target precision, context, maximum concurrency,
prefix budget and other engine settings in both modes. Target-only control
must explicitly set `engine.mtp=false`; DFlash must set `mtp=true`,
`mtp_depth=7`, and the parent must verify `DGPP_MIMO_DFLASH=1` on both ranks.
The supplied deployment configuration is saved as engine fields plus a SHA256;
it does not prove what is running. Metrics independently check zero draft
rounds for control and depth seven with positive draft rounds for DFlash.

## Execute

Capture control first, then switch to the already-authorized DFlash candidate
and capture it. Use the same pair ID and model; output directories must not
exist. Labels should identify actual binary head, target precision and mode.

```sh
python3 tools/mimo_dflash_service_check.py self-test
python3 tools/mimo_dflash_service_check.py capture \
  --url http://192.168.0.171:30001/v1 --mode control \
  --deployment "$CONTROL_CONFIG" --pair-id tp2-correctness-01 \
  --runtime-label "$CONTROL_HEAD target-only verified-on-both-ranks" \
  --out "$RESULTS/control"

python3 tools/mimo_dflash_service_check.py capture \
  --url http://192.168.0.171:30001/v1 --mode dflash \
  --deployment "$DFLASH_CONFIG" --pair-id tp2-correctness-01 \
  --runtime-label "$DFLASH_HEAD DFlash7 verified-on-both-ranks" \
  --out "$RESULTS/dflash"

python3 tools/mimo_dflash_service_check.py compare \
  --control "$RESULTS/control" --dflash "$RESULTS/dflash" \
  --out "$RESULTS/comparison.json"
```

Each capture has **11 requests**, comprising ten completed requests capped at
**864 total output tokens** plus one 256-token request disconnected after three
text-bearing SSE events:

- Four sequential C1 prompts: math, code, JSON, prose, up to 96 tokens each.
- Simultaneous C2 code and JSON, using byte-identical payloads to their C1 peers.
- A 400-record prefix fixture: cached seed, warm cached repeat, uncached full
  recomputation, up to 64 tokens each. A real warm cache hit is required.
- Early streaming disconnect, scheduler drain, and a repeat of the original
  code request to check recovery and slot reuse.

Normal expected budget is roughly 2–4 minutes per capture, bounded by a
**300-second stage deadline** and **45-second individual request deadline**.
Draining is limited to 15 seconds and stays within the stage budget. DNS/
connection setup uses the same socket timeout; use the numeric LAN endpoint to
avoid resolver delays. Stop on a request failure; completed and partial evidence
is retained. The parent must handle deployment/startup time separately. The two
captures total 22 requests, at most 1728 completed output tokens, plus the two
interrupted streams. Explicit timeout overrides are available if the control
is slower, but preserve the same protocol and record why.

## Evidence and interpretation

Every request has its exact JSON payload, timestamped raw SSE data lines,
parsed events, joined content/reasoning, finish reason, usage and timing.
Metrics snapshots bracket phases, including cancellation; drain polling saves
all snapshots. `summary.json` checks request accounting (no external traffic),
server errors, one observed cancellation, a real cache hit, speculation mode,
and C2 client overlap. C2 replay counts by slot count are also reported: inspect
these before claiming the requests exercised a particular batched graph shape.

The offline comparison checks the matched control/DFlash panel and separately
compares C1/C2, seed/warm/recompute, and pre-cancellation/recovery within each
mode. It reports the **first differing character** with context independently
for content and reasoning, plus finish reasons and prompt/completion counts.
It compares decoded text, not token IDs; SSE chunks are not token boundaries.
A successful exit means observed output agreement on this panel with complete
operational evidence. It is not a general correctness or speed claim.

Row-shape GEMM rounding can alter near-tied target choices. A divergence is
recorded and causes a nonzero exit; it is not automatically classified as a
DFlash correctness bug. Investigate the first differing target logits under
matched row shapes. Empty/incomplete, missing, timed-out or failed attempts
remain visible and cannot silently become successful comparisons. A cache miss
or unobserved cancellation yields `inconclusive`, not a passing check.
