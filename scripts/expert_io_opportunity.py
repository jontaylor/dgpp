#!/usr/bin/env python3
"""Measure gaps between DGPP expert reads inside supplied request windows.

Windows JSON is a list of {name, start_ns, end_ns}. The read trace is the
existing eight-u64 DGPP_EXPERT_READ_TRACE format. Optional device samples are
JSON lines [Unix_ns, *Linux block-device stat fields]. Gaps are opportunities,
not achievable prefetch savings: prediction, queueing, GPU work and admission
are not simulated. Keep traces and generated reports outside version control.
"""
import argparse
import json
from pathlib import Path
import struct


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError('read completion precedes start')
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def summarize(reads, start, end):
    if start >= end:
        raise ValueError('invalid request window')
    selected = [r for r in reads if r[0] < end and r[1] > start]
    intervals = merge((max(start, r[0]), min(end, r[1])) for r in selected)
    gaps = [b[0] - a[1] for a, b in zip(intervals, intervals[1:])]
    return {'window_seconds': (end - start) / 1e9,
            'completed_read_bytes': sum(r[6] for r in reads if start <= r[1] < end),
            'expert_read_union_seconds': sum(b - a for a, b in intervals) / 1e9,
            'internal_gap_seconds': sum(gaps) / 1e9,
            'internal_gap_count': len(gaps),
            'internal_gaps_at_least_ms': {
                str(t): {'count': sum(g >= t * 1e6 for g in gaps),
                         'seconds': sum(g for g in gaps if g >= t * 1e6) / 1e9}
                for t in (1.7, 3.4, 5, 10)}}


def device_summary(samples, start, end):
    rows = [r for r in samples if start <= r[0] < end]
    if len(rows) < 2:
        raise ValueError('insufficient device samples in window')
    if any(len(r) < 12 for r in rows):
        raise ValueError('device sample needs timestamp and 11 Linux stat fields')
    elapsed = (rows[-1][0] - rows[0][0]) / 1e9
    if elapsed <= 0 or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError('device timestamps must increase')
    sectors = rows[-1][3] - rows[0][3]
    busy_ms = rows[-1][10] - rows[0][10]
    if sectors < 0 or busy_ms < 0:
        raise ValueError('device counters reset')
    quiet = merge((a[0], b[0]) for a, b in zip(rows, rows[1:])
                  if a[9] == b[9] == 0 and a[1] == b[1] and a[5] == b[5])
    return {'sample_seconds': elapsed, 'read_bytes': sectors * 512,
            'accounted_busy_fraction': busy_ms / 1000 / elapsed,
            'sampled_quiet_seconds': sum(b - a for a, b in quiet) / 1e9,
            'sampled_quiet_at_least_5ms_seconds': sum(b - a for a, b in quiet if b - a >= 5e6) / 1e9}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--disk-samples', type=Path)
    args = parser.parse_args()
    data = args.trace.read_bytes()
    if len(data) % 64:
        raise ValueError('truncated read trace')
    reads = list(struct.iter_unpack('<8Q', data))
    if any(r[1] < r[0] for r in reads):
        raise ValueError('read completion precedes start')
    samples = [json.loads(line) for line in args.disk_samples.read_text().splitlines()] if args.disk_samples else None
    output = []
    for w in json.loads(args.windows.read_text()):
        start, end = int(w['start_ns']), int(w['end_ns'])
        item = {'name': w['name'], **summarize(reads, start, end)}
        if samples is not None:
            item['device'] = device_summary(samples, start, end)
        output.append(item)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
