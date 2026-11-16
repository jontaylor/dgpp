#!/usr/bin/env python3
"""Score diagnostic absent-expert forecasts against the next target-layer visit.

Requires numpy. Inputs must describe the same rank/process. Demand format is
DGEMAND1. Shadow records are ten little-endian uint64 fields: timestamp,
observation sequence (1-based), source layer, target layer, lookahead layers,
history mode (0=all selections, 1=decode selections), forecast rank (1..3),
expert ID within layer, historical count, and prefill flag. No reads are issued.

Gap feasibility grants a separate ideal 1.7ms read to each correct forecast
inside a future gap without expert reads. It ignores other device I/O, shared
gap capacity, staging limits and interference. Deadlines use actual read start,
which follows routing. Results are optimistic opportunities, not speedups or
an executable prefetch schedule. Host-accounted layer visits are not GPU layer
timestamps. Evaluation includes only decode forecasts, though history may
include preceding prefill. Generated routing artifacts remain untracked.
"""
import bisect
import collections
import json
from pathlib import Path
import numpy as np
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--shadow', type=Path, required=True)
parser.add_argument('--demand', type=Path, required=True)
parser.add_argument('--reads', type=Path, required=True)
parser.add_argument('--start-ns', type=int, default=0)
parser.add_argument('--end-ns', type=int, default=2 ** 64 - 1)
parser.add_argument('--expert-load-ms', type=float, default=1.7, help='assumed whole-expert prefetch time; not a measured latency')
parser.add_argument('--rank', type=int, default=0, help='output label only; inputs must belong to one rank/process')
args = parser.parse_args()
if not np.isfinite(args.expert_load_ms) or args.expert_load_ms <= 0:
    parser.error('expert-load-ms must be positive and finite')
if args.start_ns >= args.end_ns:
    parser.error('start must precede end')
for path, width in ((args.demand, 8), (args.reads, 64), (args.shadow, 80)):
    if path.stat().st_size % width:
        raise ValueError(f'truncated trace: {path}')
results = []
for rank in (args.rank,):
    w = np.fromfile(args.demand, dtype='<u8')
    if len(w) < 4 or int(w[0]) != 3550048448651020100 or int(w[1]) != 1:
        raise ValueError('invalid demand header')
    n = int(w[3])
    meta = w[4:4 + n * 6].reshape(n, 6)
    if np.any(meta[:, 0] != 0):
        raise ValueError('shadow scorer requires target-only demand trace')
    pos = 4 + n * 6
    events = []
    bylayer = collections.defaultdict(list)
    lookup = {(int(e[1]), int(e[2])): i for i, e in enumerate(meta)}
    while pos < len(w):
        phase, count, t = map(int, w[pos:pos + 3])
        pos += 3
        if phase not in (0, 1) or count > len(w) - pos:
            raise ValueError('invalid demand event')
        ids = set(map(int, w[pos:pos + count]))
        pos += count
        if not ids or max(ids) >= n:
            raise ValueError('invalid expert IDs')
        if len({int(meta[e, 1]) for e in ids}) != 1:
            raise ValueError('mixed-layer demand event')
        l = int(meta[next(iter(ids)), 1])
        idx = len(events)
        events.append((t, phase, l, ids))
        bylayer[l].append(idx)
    reads = np.fromfile(args.reads, dtype='<u8').reshape(-1, 8)
    readby = collections.defaultdict(list)
    for r in reads:
        readby[int(r[2]), int(r[3])].append((int(r[0]), int(r[1]), int(r[6])))
    for rows in readby.values():
        rows.sort()
    intervals = []
    for a, b in sorted(((int(r[0]), int(r[1])) for r in reads)):
        if intervals and a <= intervals[-1][1]:
            intervals[-1][1] = max(b, intervals[-1][1])
        else:
            intervals.append([a, b])
    gaps = [(u[1], v[0]) for u, v in zip(intervals, intervals[1:])]
    gapends = [b for a, b in gaps]
    shadow = np.fromfile(args.shadow, dtype='<u8').reshape(-1, 10)
    agg = {}
    for row in shadow:
        t, seq, src, target, distance, mode, predrank, expert, score, phase = map(int, row)
        if seq < 1 or seq > len(events) or events[seq - 1][2] != src or ((target, expert) not in lookup):
            raise ValueError('shadow/demand identity mismatch')
        if mode not in (0, 1) or predrank not in (1, 2, 3) or target - src != distance:
            raise ValueError('invalid forecast')
        if phase or not args.start_ns <= t < args.end_ns:
            continue
        ix = bisect.bisect_right(bylayer[target], seq - 1)
        if ix >= len(bylayer[target]):
            continue
        eventidx = bylayer[target][ix]
        ev = events[eventidx]
        if ev[1] != phase:
            continue
        nextidx = bylayer[target][ix + 1] if ix + 1 < len(bylayer[target]) else len(events)
        until = events[nextidx][0] if nextidx < len(events) else 2 ** 64 - 1
        demanded = lookup[target, expert] in ev[3]
        candidate_reads = [r for r in readby[target, expert] if ev[0] <= r[0] < until] if demanded else []
        lead = (candidate_reads[0][0] - t) / 1000000.0 if candidate_reads else None
        fitsgap = False
        if candidate_reads:
            deadline = candidate_reads[0][0]
            for ga, gb in gaps[bisect.bisect_right(gapends, t):]:
                if ga >= deadline:
                    break
                if min(gb, deadline) - max(ga, t) >= args.expert_load_ms * 1e6:
                    fitsgap = True
                    break
        for k in [1, 3]:
            if predrank > k:
                continue
            key = (mode, distance, k)
            a = agg.setdefault(key, {'offers': 0, 'selected': 0, 'read_hits': 0, 'lead_ms': [], 'gap_feasible_hits': set(), 'unique_hits': set()})
            a['offers'] += 1
            a['selected'] += demanded
            if candidate_reads:
                a['read_hits'] += 1
                a['lead_ms'].append(lead)
                a['unique_hits'].add((eventidx, expert))
                if fitsgap:
                    a['gap_feasible_hits'].add((eventidx, expert))
    for (mode, distance, k), a in sorted(agg.items()):
        leads = a.pop('lead_ms')
        feasible = a.pop('gap_feasible_hits')
        unique = a.pop('unique_hits')
        out = {'rank': rank, 'history': 'decode' if mode else 'all', 'lookahead_layers': distance, 'experts_offered_per_visit': k, **a, 'precision': a['read_hits'] / a['offers'] if a['offers'] else 0, 'unique_read_hits': len(unique), 'gap_feasible_unique_hits': len(feasible), 'assumed_expert_load_ms': args.expert_load_ms, 'gap_feasible_read_GiB': sum(int(meta[lookup[events[i][2], e], 5]) for i, e in feasible) / 2 ** 30, 'lead_ms_p50_p90': np.percentile(leads, [50, 90]).tolist() if leads else []}
        results.append(out)
print(json.dumps(results, indent=2))
