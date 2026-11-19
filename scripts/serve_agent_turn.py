#!/usr/bin/env python3
"""Account for one completed Codex turn using DGPP rank-zero retirement records."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


# The scheduler logs UTC on the Sparks. Its request decode clock includes time
# waiting between steps when other requests are active.
RETIRE = re.compile(
    r"^(?P<at>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) .*?sched: request '(?P<id>[^']+)' "
    r"retired \([^)]+\): (?P<output>\d+) tok in [\d.]+ s — prefill (?P<input>\d+) tok "
    r"\((?P<cached>\d+) cached\) in (?P<prefill_ms>\d+) ms; decode (?P<decode>\d+) tok "
    r"/ (?P<passes>\d+) pass(?:es)? in (?P<decode_s>[\d.]+) s"
)


def read_turn(path, turn_id):
    calls = defaultdict(list)
    completed = {}
    with path.open() as records:
        for line in records:
            event = json.loads(line)
            data = event.get("payload", {})
            if event["type"] == "token_usage_record":
                calls[data["turn_id"]].append((datetime.fromisoformat(event["timestamp"]), data["usage"]))
            elif event["type"] == "event_msg" and data.get("type") == "task_complete":
                completed[data["turn_id"]] = data
    if not completed:
        raise ValueError("no completed turn in rollout")
    selected = turn_id or next(reversed(completed))
    if selected not in completed or not calls[selected]:
        raise ValueError(f"missing completion or per-call usage for turn {selected}")
    return completed[selected], calls[selected]


def summarize(completion, calls, rank_log):
    by_usage = defaultdict(list)
    for line in rank_log:
        match = RETIRE.match(line)
        if match:
            key = tuple(int(match[k]) for k in ("input", "cached", "output"))
            at = datetime.fromisoformat(match["at"]).replace(tzinfo=timezone.utc)
            by_usage[key].append((at, match))

    seen = set()
    matches = []
    for at, usage in calls:
        key = tuple(usage[k] for k in ("input_tokens", "cached_input_tokens", "output_tokens"))
        # Translation at the proxy need not preserve the server's request ID.
        # Require both the three exact counts and a nearby retirement time.
        candidates = [row for end, row in by_usage[key] if abs((at - end).total_seconds()) <= 5]
        if len(candidates) != 1 or candidates[0]["id"] in seen:
            raise ValueError(f"could not uniquely match model call at {at.isoformat()} to a rank-zero retirement")
        row = candidates[0]
        matches.append(row)
        seen.add(row["id"])

    def total(field):
        return sum(int(row[field]) for row in matches)

    decode_s = sum(float(row["decode_s"]) for row in matches)
    passes = total("passes")
    reasoning = [usage.get("reasoning_output_tokens") for _, usage in calls]
    return {
        "turn_id": completion["turn_id"],
        "wall_s": completion["duration_ms"] / 1000,
        "model_calls": len(matches),
        "input_tokens": total("input"),
        "cached_input_tokens": total("cached"),
        "computed_input_tokens": total("input") - total("cached"),
        "output_tokens": total("output"),
        "client_reasoning_output_tokens": sum(reasoning) if None not in reasoning else None,
        "decode_tokens_after_first": total("decode"),
        "request_prefill_s": total("prefill_ms") / 1000,
        "request_decode_s": round(decode_s, 3),
        "decode_passes": passes,
        "decode_tokens_per_pass": round(total("decode") / passes, 3) if passes else None,
        "request_decode_tok_s": round(total("decode") / decode_s, 2) if decode_s else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path, help="Codex rollout JSONL containing the review")
    parser.add_argument("rank_log", type=argparse.FileType("r"), help="rank-zero UTC service log, or - for stdin")
    parser.add_argument("--turn-id", help="defaults to the last completed turn in the rollout")
    args = parser.parse_args()
    try:
        completion, calls = read_turn(args.rollout, args.turn_id)
        result = summarize(completion, calls, args.rank_log)
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(f"turn accounting: {exc}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
