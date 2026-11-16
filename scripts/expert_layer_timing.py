#!/usr/bin/env python3
"""Summarize opt-in DGPP_EXPERT_LAYER_TRACE records from one rank/process.

The trace contains eight native little-endian u64 fields per layer: target
pass, row count, layer, GPU start/end nanoseconds, demanded file bytes,
completed file records, pass-end Unix nanoseconds. GPU clocks are local to
each rank. Do not sum ranks or subtract their clocks. A span includes main
stream work, dependencies, communication and exposed demand-gate waits.
Adjacent spans share a timestamp marker; one extra kernel per boundary adds
diagnostic overhead. Graph setup, draft generation and head/sampling are out
of scope. Only the paged target decode/verification graph is instrumented.

Whole-expert load bins require --expert-bytes matching the deployed store.
Partial loads fail rather than being rounded. Optional windows JSON contains
{name, start_ns, end_ns}, tested against the pass-end Unix timestamp.
"""
import argparse
import json
from pathlib import Path
import statistics
import struct


def summarize(records, expert_bytes):
    groups = {'0': [], '1': [], '2+': []}
    loads = {k: [] for k in groups}
    for _, _, _, start, end, file_bytes, _, _ in records:
        if end < start:
            raise ValueError('GPU timestamps run backwards')
        n, remainder = divmod(file_bytes, expert_bytes)
        if remainder:
            raise ValueError('partial expert read: whole-expert bins are invalid')
        key = str(n) if n < 2 else '2+'
        groups[key].append((end-start)/1e6)
        loads[key].append(n)
    total_ms = sum(map(sum, groups.values()))
    output = {}
    for key, values in groups.items():
        if not values:
            output[key] = {'count': 0}
            continue
        ordered = sorted(values)
        output[key] = {
            'count': len(values), 'mean_ms': statistics.mean(values),
            'median_ms': statistics.median(values),
            'p90_ms': ordered[int((len(values)-1)*0.9)],
            'mean_expert_loads': statistics.mean(loads[key]),
            'total_ms': sum(values),
            'time_fraction': sum(values)/total_ms if total_ms else 0}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--expert-bytes', type=int, required=True)
    p.add_argument('--windows', type=Path)
    args = p.parse_args()
    if args.expert_bytes <= 0:
        p.error('--expert-bytes must be positive')
    records = list(struct.iter_unpack('<8Q', args.trace.read_bytes()))
    windows = json.loads(args.windows.read_text()) if args.windows else [
        {'name': 'all', 'start_ns': 0, 'end_ns': 2**64}]
    result = {}
    for w in windows:
        rows = [r for r in records if w['start_ns'] <= r[7] < w['end_ns']]
        result[w['name']] = {
            'all_widths': summarize(rows, args.expert_bytes),
            'by_target_rows': {str(n): summarize([r for r in rows if r[1] == n], args.expert_bytes)
                               for n in sorted({r[1] for r in rows})}}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
