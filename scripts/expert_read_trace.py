#!/usr/bin/env python3
"""Summarize DGPP_EXPERT_READ_TRACE completions (one rank/process per input).

Records are eight native uint64s written on little-endian serving hosts:
start/end Unix ns, layer, expert, chunk, draft, record bytes, file offset.
First-fetch/reread classification is by chunk, then aggregated by expert.
This counts logical successful reads, not physical SSD traffic or exposed wait.
"""
import argparse
import collections
import json
from pathlib import Path
import struct

RECORD = struct.Struct('<8Q')


def summarize(path, start_ns=0, end_ns=2**64-1):
    data = Path(path).read_bytes()
    if len(data) % RECORD.size:
        raise ValueError(f'{path}: incomplete trace record')
    events = sorted(RECORD.iter_unpack(data), key=lambda r: r[1])
    seen, window_seen = set(), set()
    experts = collections.defaultdict(collections.Counter)
    for start, end, layer, expert, chunk, draft, size, offset in events:
        if end > end_ns:
            break
        key = (layer, expert, chunk, draft, offset)
        first = key not in seen
        seen.add(key)
        if end < start_ns:
            continue
        stats = experts[(draft, layer, expert)]
        stats['reads'] += 1
        stats['bytes'] += size
        stats['first_fetch_bytes' if first else 'reread_bytes'] += size
        stats['first_in_window_bytes' if key not in window_seen else 'repeat_in_window_bytes'] += size
        stats['read_duration_ns_overlapping'] += end-start
        window_seen.add(key)
    totals = sum(experts.values(), collections.Counter())
    return {'trace': str(path), 'start_ns': start_ns, 'end_ns': end_ns,
            'totals': dict(totals), 'experts': [
                {'draft': bool(d), 'layer': l, 'expert': e, **stats}
                for (d, l, e), stats in sorted(experts.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', nargs='+', type=Path)
    parser.add_argument('--start-ns', type=int, default=0)
    parser.add_argument('--end-ns', type=int, default=2**64-1)
    args = parser.parse_args()
    if args.end_ns < args.start_ns:
        parser.error('end must not precede start')
    print(json.dumps([summarize(p, args.start_ns, args.end_ns) for p in args.traces], indent=2))


if __name__ == '__main__':
    main()
