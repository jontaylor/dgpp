# Expert-cache what-if replay

`scripts/expert_cache_whatif.py` walks recorded serving routes while maintaining
independent hypothetical expert caches. It does not execute the model, issue
SSD reads, change production residency, or predict serving throughput. NumPy is
required. Generated traces, decision logs and reports stay untracked.

## Inputs and invocation

The existing `DGPP_EXPERT_DEMAND_TRACE` and `DGPP_EXPERT_READ_TRACE` environment
variables capture routing observations and successful file reads. Both are
opt-in. Demand filenames include rank, PID and cache binding; select one
target-only cache and the corresponding read trace from the same process.
Collect block-device counters on that same node while traffic runs:

```bash
python3 scripts/expert_io_sample.py --device nvme0n1 --output DISK.jsonl --seconds 600
```

Copy the completed capture to the analysis host, then run:

```bash
python3 scripts/expert_cache_whatif.py \
  --demand DEMAND.bin --reads READS.bin --disk-samples DISK.jsonl \
  --windows WINDOWS.json --output REPORT.json --decisions DECISIONS.jsonl
```

`--windows` is optional. It contains ordered, nonoverlapping objects with
`name`, inclusive `start_ns`, and exclusive `end_ns` in Unix nanoseconds. The
entire trace prefix is always replayed, so evaluation windows inherit causal
history and residency rather than silently restarting warm or empty. Run ranks
separately: they share routing but have different device activity and clocks.

This is an offline replay of actual model routing, including speculative
verification rows. It is not a continuously running live shadow controller.
The first version accepts equal-sized, whole experts and rejects mixed draft
caches, partial initial residency, ambiguous read attribution, duplicate chunks,
clock regression and incomplete records. Disk record bytes are not RAM charges:
resident capacity uses the header's cache slots and per-expert chunk counts.
The header seeds residency at the first observation; it is not assumed empty.

## Policies and causal state

Three independent arms share the same initial residency, capacity and demand:

1. Demand-only LRU, with deterministic expert-ID order for batch ties.
2. Deadline-weighted retention, with no speculative loading.
3. The same retention rule with bounded speculative loading.

The recorded implementation is an additional observed reference, not a fourth
simulated policy. Its demanded-expert hit/read outcomes can be reconstructed;
its complete evolving residency and eviction choices were not captured. Cache
overlap is therefore reported between simulation arms, not against an invented
production cache snapshot.

Each visit is scored against entry residency before demand admission. Repeated
routing selections require only one weight image per expert for that visit.
After observing current demand, the predictor updates past inclusion frequencies
by layer, phase and route-list-size bin. A broader phase history supplies the
fallback until a shape has enough observations. Histories decay by layer visits
within their profile; an inactive phase does not age merely because another
phase runs. No future route or precomputed whole-trace popularity is a feature.

For equal-sized experts the initial retention score is:

```
reward = p(needed) * miss_cost * exp(-time_to_next_layer / cycle_time)
         * (1 + exp(-1) + exp(-2))
score  = reward - memory_rent * time_to_next_layer
```

This rewards three discounted future visits and charges holding memory until
the first benefit. A replacement must beat the victim by `--switch-margin-ms`.
Demand bytes already fetched incur no speculative loading charge. Demand
admission considers candidates in score order and protects the current batch's
resident experts, avoiding arbitrary expert-ID preference for limited space.

Speculative candidates additionally pay an I/O price inversely proportional to
the estimated spare fraction, plus a staging holding charge. The spare estimate
uses only intervals already traversed. Estimated layer times use past adjacent
observations; tiny host-accounting bursts and long gaps are excluded by explicit
sample limits. Forecasts expire one estimated cycle after their predicted next
opportunity. Phase changes cancel outstanding old-phase forecasts.

Defaults are an initial hypothesis, not fitted parameters or a validated optimum:
6 GB/s aggregate bandwidth, 1 ms file-completion-to-readiness delay, one pending
speculative expert, 256 staging buffers with 36 reserved for demand, 64-visit
history half-life, a 0.05 ms replacement margin, decode/prefill miss-benefit
proxies of 3.2/1.7 ms, and initial layer-time proxies of 1.6/20 ms. Other cost
coefficients are recorded in each report. Equal-sized experts make the memory
rent a per-expert holding cost. These proxy units do not establish saved latency.

## Shared I/O and staging model

The recorded clock stays fixed. Device-quiet intervals require zero in-progress
I/O at both samples and unchanged completed-read/write counters. All speculative
loads share one aggregate byte pipe within those intervals; they cannot each
claim the same gap. The pipe can pause immediately when an interval ends. This
is an idealized fluid model, not a model of nonpreemptible SSD commands, file
workers, queue submissions or GPU transfers.

Every simulated demand read initially creates service debt. Baseline transfer
credit repays it only when the corresponding recorded file read completes;
future read labels cannot grant early capacity. Unpaid debt consumes later
quiet capacity before speculation. Saved baseline reads do not reclaim their
busy intervals. This treatment is causal within the fixed trace, but it cannot
reconstruct changed stalls, overlap windows or subsequent observation times.

Pending and completed-but-unadmitted forecasts hold staging until demand,
admission, expiry or cancellation. Full-expert forecasts reserve their entire
chunk count; they never borrow the demand reserve. Readiness, admissions and
expiry are processed in time order even between observations. Speculative
cache admission may evict a lower-value resident; that pollution is included in
subsequent hit/miss outcomes. Extra persistent memory is not granted.

## Outputs and limits

The JSON report records input hashes, the replay-script hash, all assumptions,
phase/window/per-layer outcomes, resident and staging peaks, and sampled cache
overlap between retention with and without prefetch. Optional JSONL decisions
include visits, forecast issues, shared-pipe service, admission rejections with
scores, evictions, readiness and expiry. Log timestamps are relative to the
report's `trace_epoch_unix_ns`, avoiding floating-point epoch precision loss.

Report demand bytes **and total bytes including speculation**. Resource bytes
in each window are attributed by service time; forecast outcomes are attributed
to the issuing window and phase. A prefetch may therefore consume bytes in a
later window. Sum phases when comparing a window's total resource use. Late
forecasts can provide partial bytes without being ready hits. Unconsumed tail
forecasts are censored, not declared useless. A prefetched expert is credited
as used only on its first demand before eviction; repeated hits are ordinary
cache reuse, not additional useful speculative loads.

Recorded host observations of native hits can occur after GPU execution, so a
modeled ready hit is not proof of meeting a real routing deadline. Prefill
route-list sizes may be deduplicated by capture settings and are not guaranteed
token counts. Prefill also sometimes revisits encoder layers before advancing
through the decoder; the initial cyclic deadline model does not capture that
hidden scheduling state. The report counts deviations from the assumed loop.

No output establishes physical SSD bytes saved, exact memory consumption,
prefill seconds saved, decode tokens/s, or production speedup. Model routes and
quality are unchanged because the experiment never changes execution.

## November 16 first replay

The initial experiment replays both ranks of the recorded eight-request
tool-serving sequence: initial long prompt, prose and code continuations,
repeated prose, then four short recomputations. These are actual model-generated
routes from a scripted workload, not an organic user-traffic distribution or
an independent held-out workload. The two ranks are not independent samples.
Capture binary SHA256 was
`e163306747831667813d601306ad38a68827a75cea118706751c75cb3307b8d6`, with zero
process swap on both ranks at the recorded request boundaries. No new serving
campaign or deployment change is needed to replay it.

The simulation starts from the captured zero-resident header and replays all
30,660 layer observations with a 10,359-whole-expert capacity. Separate decode
and prefill results are described below. These are policy diagnostics, not
benchmark campaign throughput rows.

For rank zero, logical file-read volume across the complete captured sequence
was as follows. Simulation totals include all speculative bytes, including
unused and partial transfers; the observed reference is the actual read trace.

| Policy | Prefill GiB | Decode / verification GiB |
| --- | ---: | ---: |
| Observed deployment | 163.599 | 67.451 |
| Demand-only LRU | 163.263 | 77.901 |
| Deadline retention | 166.802 | 210.468 |
| Deadline retention plus prefetch | 167.082 | 211.215 |

The deadline-only arm had 22,538 decode misses versus 7,223 observed and 8,342
under LRU. On the final repeated short request the observed policy and LRU had
zero decode misses; deadline retention still had 292 and its prefetch arm 288.
The prefetch arm issued 191 prefill-origin and 173 decode-origin forecasts;
110 and 28 respectively were consumed as modeled ready hits. Late forecasts
and unused bytes are charged separately. All forecast outcomes reconciled
including censored tails. Resident capacity never exceeded 10,359 experts;
the default single outstanding forecast used at most eighteen staging buffers.

Rank one's retention-only volumes were 166.793 / 210.459 GiB and its prefetch
arm 166.905 / 210.851 GiB. Small differences reflect its host-clock and device
samples; LRU and observed volumes were identical across ranks. Decode followed
the assumed layer cycle throughout; prefill had eleven noncyclic transitions.

The default deadline score is not selected for deployment. Its demand-only arm
repeatedly rejects recurring experts immediately after use, when their next
benefit lies a full cycle away. An incumbent in another layer has a shorter
deadline and can receive a higher score. Rejected experts get no later demand
admission opportunity as their deadline approaches. Repeated requests can thus
continue missing even when LRU and the observed policy have warmed completely.
This is a negative result for this particular score/admission combination, not
a rejection of adaptive retention or predictive loading.

Run the resource/causality checks with:

```bash
python3 tests/python/expert_cache_whatif_test.py
```
