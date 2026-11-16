#!/usr/bin/env python3
"""Optimistic whole-expert traffic bound for DGPP's diagnostic demand trace.

All experts must have identical sizes. An entire batch's missing experts may
use unlimited free staging until its end, and the oracle knows future events.
This relaxation is deliberately more powerful than the actual bounded cache.
It is not a replay of the production planner or an achievable speed forecast.
"""
import argparse
from collections import deque
import heapq
import json
import struct
from pathlib import Path

WORD = struct.Struct('<Q')
MAGIC = 0x31444e414d454744


def load(path):
    data = Path(path).read_bytes()
    if len(data) % WORD.size:
        raise ValueError('truncated trace word')
    words = [x[0] for x in WORD.iter_unpack(data)]
    pos = 0

    def take(n):
        nonlocal pos
        if n > len(words) - pos:
            raise ValueError('truncated trace record')
        values = words[pos:pos + n]
        pos += n
        return values

    magic, version, slots, count = take(4)
    if magic != MAGIC or version != 1:
        raise ValueError('unsupported demand trace')
    experts = [take(6) for _ in range(count)]
    if not experts or any(e[3] == 0 or e[4] > e[3] for e in experts):
        raise ValueError('invalid expert geometry')
    if any(e[4] not in (0, e[3]) for e in experts):
        raise ValueError('bound requires whole-expert initial residency')
    if len({(e[3], e[5]) for e in experts}) != 1:
        raise ValueError('bound requires equal-size experts and read records')
    events = []
    while pos < len(words):
        prefill, n, timestamp = take(3)
        ids = take(n)
        if prefill > 1 or any(e >= count for e in ids):
            raise ValueError('invalid demand event')
        events.append((prefill, timestamp, set(ids)))
    return slots, experts, events


def bound(slots, experts, events, free_initial=False):
    chunks, read_bytes = experts[0][3], experts[0][5]
    capacity = slots // chunks
    future = [deque() for _ in experts]
    for i, (_, _, ids) in enumerate(events):
        for e in ids:
            future[e].append(i)
    inf = len(events) + 1
    resident = {i for i, e in enumerate(experts) if e[4]}
    if len(resident) > capacity:
        raise ValueError('initial residency exceeds capacity')
    if free_initial:
        # An arbitrary perfect starting cache is free. This produces a lower
        # bound for an independently selected window even when its actual
        # starting residency was not recorded in the trace header.
        resident = set(sorted(range(len(experts)),
                              key=lambda e: future[e][0] if future[e] else inf)[:capacity])
    # heap entries carry a version so updates leave stale entries harmless.
    heap, versions = [], [0] * len(experts)

    def refresh(e):
        versions[e] += 1
        nxt = future[e][0] if future[e] else inf
        heapq.heappush(heap, (-nxt, e, versions[e]))

    for e in resident:
        refresh(e)
    rows = []
    for i, (prefill, timestamp, ids) in enumerate(events):
        missing = ids - resident
        for e in ids:
            assert future[e].popleft() == i
            resident.add(e)
            refresh(e)
        while len(resident) > capacity:
            _, e, version = heapq.heappop(heap)
            if e in resident and version == versions[e]:
                resident.remove(e)
        rows.append({'time_ns': timestamp, 'prefill': bool(prefill),
                     'unique_experts': len(ids), 'misses': len(missing),
                     'lower_bound_bytes': len(missing) * read_bytes})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace')
    parser.add_argument('--events', action='store_true', help='emit per-event JSON lines')
    parser.add_argument('--start-ns', type=int, default=0, help='inclusive Unix timestamp')
    parser.add_argument('--end-ns', type=int, default=2**64-1, help='exclusive Unix timestamp')
    parser.add_argument('--free-initial', action='store_true',
                        help='grant an ideal starting cache for a selected window')
    args = parser.parse_args()
    slots, experts, events = load(args.trace)
    if args.start_ns and not args.free_initial:
        parser.error('a window after trace start requires --free-initial')
    events = [e for e in events if args.start_ns <= e[1] < args.end_ns]
    rows = bound(slots, experts, events, args.free_initial)
    if args.events:
        for row in rows:
            print(json.dumps(row))
    else:
        print(json.dumps({'events': len(rows), 'experts': len(experts),
                          'capacity_experts': slots // experts[0][3],
                          'lower_bound_bytes': sum(x['lower_bound_bytes'] for x in rows),
                          'prefill_lower_bound_bytes': sum(x['lower_bound_bytes'] for x in rows if x['prefill']),
                          'decode_lower_bound_bytes': sum(x['lower_bound_bytes'] for x in rows if not x['prefill'])}, indent=2))


if __name__ == '__main__':
    main()
