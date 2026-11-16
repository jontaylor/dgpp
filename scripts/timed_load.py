#!/usr/bin/env python3
"""Run serve_load with reconciled scheduler counters around every phase.

Requires an otherwise idle server. Accept an idle snapshot only when its
prompt/token deltas equal the completed requests' usage, with a five-second
deadline. Polling is outside the client and engine timing intervals.
Use --server-timings to record and validate per-phase server histogram
brackets for TTFT and token-delivery-pass p50/p95/p99. Speculation can
deliver several tokens per pass.
Prometheus reads run outside client/engine timing intervals.
Arguments and JSON fields are the same as serve_load, plus each phase's
engine counters and decode tokens/s. Warmup and isolation are reconciled too.

Two rates come out of every phase, and the tables keep them apart:
  decode tok/s  (engine)  tokens generated / the engine's decode-step time —
                prefill and queueing excluded; the headline throughput, the
                figure docs/benchmarks.md reports by concurrency.
  wall tok/s    (client)  tokens generated / the request group's wall time —
                prefill and queueing included; the service-level view.
The per-class table printed first is the decode rate; serve_load's own
table (the wall rate) follows it.
"""
import argparse
import math
import re
import time
import urllib.request

import serve_load
from serve_prefill_probe import get


COUNTERS = ("step_ms", "prefill_ms", "decode_steps", "decode_rows", "tokens_generated",
            "prompts_prefilled", "prompt_tokens", "prompt_tokens_computed")


def reconciled_metrics(host, port, before=None, requests=(), *, timeout=5):
    """Read immediately; retry only while the known work has not been published."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("metrics timeout must be finite and positive")
    expected = {"prompts_prefilled": len(requests),
                "tokens_generated": sum(r["tokens"] for r in requests)}
    deadline = time.monotonic() + timeout
    observed = None
    metrics = None

    def failure(reason):
        detail = (f"expected delta {expected}, observed delta {observed}"
                  if before is not None else "expected an idle scheduler")
        gauges = ({k: metrics[k] for k in ("active", "queued")}
                  if metrics is not None else None)
        return RuntimeError(f"scheduler counters {reason}; {detail}; active/queued={gauges}")

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise failure("did not reconcile before the deadline")
        try:
            metrics = get(host, port, "/v1/metrics", timeout=remaining)["scheduler"]
        except TimeoutError as error:
            raise failure("timed out reading metrics") from error
        if before is not None:
            observed = {k: metrics[k] - before[k] for k in expected}
            if any(value < 0 for value in observed.values()):
                raise failure("regressed (server restart or reset)")
            if any(observed[k] > expected[k] for k in expected):
                raise failure("include unexpected requests or tokens")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise failure("did not reconcile before the deadline")
        if (not metrics["active"] and not metrics["queued"] and
                (before is None or observed == expected)):
            return metrics
        # This only limits polling frequency. Readiness requires exact counters,
        # even if multiple consecutive reads return the same stale snapshot.
        time.sleep(min(0.01, remaining))


def prometheus_snapshot(host, port):
    with urllib.request.urlopen(f"http://{host}:{port}/metrics/prometheus", timeout=30) as response:
        return response.read().decode()


def histogram_quantiles(before, after, name, expected):
    """Nearest-rank server histogram brackets without interpolation.

    Use observed counts when expected is None (multi-token delivery passes).
    """
    def buckets(text):
        series = {}
        for line in text.splitlines():
            if not line.startswith(name + "_bucket{"):
                continue
            match = re.fullmatch(re.escape(name) + r'_bucket\{(.*)\}\s+(\d+)', line)
            if not match:
                raise RuntimeError(f"malformed histogram: {name}")
            labels, value = match.groups()
            bound_label = re.search(r'(?:^|,)le="([^"]+)"(?:,|$)', labels)
            if bound_label is None:
                raise RuntimeError(f"missing histogram bound: {name}")
            bound = float(bound_label.group(1))
            if math.isnan(bound) or bound < 0:
                raise RuntimeError(f"invalid histogram bound: {name}")
            identity = re.sub(r'(?:^|,)le="[^"]+"', '', labels).strip(',')
            row = series.setdefault(identity, {})
            if bound in row:
                raise RuntimeError(f"duplicate histogram bucket: {name}")
            row[bound] = int(value)
        bounds = None
        for row in series.values():
            if math.inf not in row or (bounds is not None and set(row) != bounds):
                raise RuntimeError(f"missing or inconsistent histogram bounds: {name}")
            bounds = set(row)
            previous = 0
            for bound, value in sorted(row.items()):
                if value < previous:
                    raise RuntimeError(f"nonmonotonic histogram: {name}")
                previous = value
        if not series:
            raise RuntimeError(f"missing server histogram: {name}")
        return series
    a, b = buckets(before), buckets(after)
    if a.keys() != b.keys():
        raise RuntimeError(f"histogram series changed: {name}")
    counts = {}
    for identity, row in a.items():
        if row.keys() != b[identity].keys():
            raise RuntimeError(f"histogram bounds changed: {name}")
        delta = {bound: b[identity][bound] - row[bound] for bound in row}
        values = [value for _, value in sorted(delta.items())]
        if any(value < 0 for value in values) or values != sorted(values):
            raise RuntimeError(f"histogram delta/count mismatch: {name}; series {identity}")
        for bound, count in delta.items():
            counts[bound] = counts.get(bound, 0) + count
    values = [value for _, value in sorted(counts.items())]
    if any(value < 0 for value in values) or values != sorted(values) or (expected is not None and counts[math.inf] != expected):
        raise RuntimeError(f"histogram delta/count mismatch: {name}; expected {expected}, got {counts}")
    expected = counts[math.inf]
    result = {"count": expected, "definition": "nearest-rank server histogram brackets; no interpolation"}
    for pct in (50, 95, 99):
        result[f"p{pct}"] = None
        if not expected:
            continue
        rank, lower = (expected * pct + 99) // 100, 0.0
        for bound, count in sorted(counts.items()):
            if count >= rank:
                result[f"p{pct}"] = {"lower_exclusive_seconds": lower,
                                    "upper_inclusive_seconds": bound if math.isfinite(bound) else None,
                                    "right_censored": not math.isfinite(bound)}
                break
            lower = bound
    return result


class TimedLoad:
    def __init__(self, server_timings=False):
        self.server_timings = server_timings
        self.previous = None
        self.decode_rates = []  # (class, concurrency, engine tok/s) per measured phase

    def settle(self, host, port, requests):
        self.previous = reconciled_metrics(host, port, self.previous, requests)
        return self.previous

    def phase(self, host, port, model, c, max_tokens, think, prompt_base,
              prompts=None, temperature=0, label="", reports=None):
        before = self.settle(host, port, [])
        hist_before = prometheus_snapshot(host, port) if self.server_timings else None
        measured = []
        rate = serve_load.phase(host, port, model, c, max_tokens, think, prompt_base,
                                prompts, temperature, label, measured)
        result = measured[-1]
        after = self.settle(host, port, result["requests"])
        delta = {name: after[name] - before[name] for name in COUNTERS}
        tokens = delta["tokens_generated"] - delta["prompts_prefilled"]
        if tokens <= 0 or not math.isfinite(delta["step_ms"]) or delta["step_ms"] <= 0:
            raise RuntimeError("no measurable decode work")
        result["engine"] = {**delta, "decode_tokens": tokens,
                            "tokens_per_s": tokens * 1000 / delta["step_ms"]}
        # Preserve the already-read head-rank snapshots so a warm rate can be
        # separated from expert I/O and speculative acceptance after the run.
        # Peer-rank counters require a separate endpoint; these are rank zero.
        for name in ("expert_streaming", "spec_decode"):
            if name in before and name in after:
                result["engine"][name] = {"rank": 0, "before": before[name], "after": after[name]}
        if self.server_timings:
            hist_after = prometheus_snapshot(host, port)
            result["server_ttft_quantile_brackets"] = histogram_quantiles(
                hist_before, hist_after, "dgpp_time_to_first_token_seconds", len(result["requests"]))
            result["server_itl_quantile_brackets"] = histogram_quantiles(
                hist_before, hist_after, "dgpp_inter_token_latency_seconds", None)
            # The server observes one gap per request's token-delivery pass,
            # not per output token: MTP may deliver several tokens together.
            gaps = result["server_itl_quantile_brackets"]
            minimum = sum(r["tokens"] > 1 for r in result["requests"])
            if not minimum <= gaps["count"] <= tokens:
                raise RuntimeError("token-delivery histogram count outside completed-work bounds")
            gaps["observation_unit"] = "request token-delivery pass"
            result["prometheus_before"] = hist_before
            result["prometheus_after"] = hist_after
        if reports is not None:
            reports.append(result)
        self.decode_rates.append((label.strip() or "mixed", c, result["engine"]["tokens_per_s"]))
        print(f"   engine decode: {result['engine']['tokens_per_s']:.3f} tok/s (decode steps only; the wall rate above includes prefill)", flush=True)
        return rate


def decode_table(rates):
    """Median decode-only rate per (class, concurrency) over the repetitions, as a markdown table."""
    import statistics
    cells = {}
    for cls, c, rate in rates:
        cells.setdefault((cls, c), []).append(rate)
    classes = list(dict.fromkeys(cls for cls, _, _ in rates))
    cs = sorted({c for _, c, _ in rates})
    lines = ["== per-class engine decode tokens/s (decode steps only; prefill and queueing excluded)",
             "| class | " + " | ".join(f"c={c}" for c in cs) + " |", "|---|" + "---|" * len(cs)]
    for cls in classes:
        lines.append(f"| {cls} | " + " | ".join(f"{statistics.median(cells[(cls, c)]):.1f}" if (cls, c) in cells else "—" for c in cs) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--server-timings", action="store_true",
                        help="validate per-phase TTFT and token-delivery-pass histograms (supports speculation)")
    options, remaining = parser.parse_known_args()
    load = TimedLoad(server_timings=options.server_timings)
    serve_load.main(phase_runner=load.phase, settle=load.settle, argv=remaining)
    if load.decode_rates:
        print(decode_table(load.decode_rates), flush=True)


if __name__ == "__main__":
    main()
