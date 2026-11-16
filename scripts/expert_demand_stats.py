#!/usr/bin/env python3
"""Summarize per-layer expert popularity from an opt-in DGEMAND1 trace.

Requires numpy and matplotlib. Trace files contain workload routing information;
write generated reports outside version control. Counts describe selections,
not misses, bytes read, prefetch timeliness, or current resident contents.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--start-ns', type=int, default=0, help='inclusive event timestamp')
    p.add_argument('--end-ns', type=int, default=2**64 - 1, help='exclusive event timestamp')
    p.add_argument('--output', type=Path, required=True, help='output path prefix')
    args = p.parse_args()
    if args.start_ns >= args.end_ns:
        p.error('start must precede end')
    words = np.fromfile(args.trace, dtype='<u8')
    if words.size < 4 or args.trace.stat().st_size % 8:
        raise ValueError('truncated trace')
    magic, version, _, count = map(int, words[:4])
    if magic != 0x31444e414d454744 or version != 1 or words.size < 4 + count * 6:
        raise ValueError('invalid trace header')
    experts = words[4:4 + count * 6].reshape(count, 6)
    if np.any(experts[:, 0] != 0):
        raise ValueError('mixed draft/target traces require separate processing')
    layers = sorted(map(int, np.unique(experts[:, 1])))
    ids = sorted(map(int, np.unique(experts[:, 2])))
    if layers != list(range(len(layers))) or ids != list(range(len(ids))):
        raise ValueError('chart requires contiguous layer and expert IDs starting at zero')
    lindex = {v: i for i, v in enumerate(layers)}
    eindex = {v: i for i, v in enumerate(ids)}
    coords = [(lindex[int(e[1])], eindex[int(e[2])]) for e in experts]
    if len(set(coords)) != count:
        raise ValueError('duplicate layer/expert identity')
    shape = (2, len(layers), len(ids))
    selections = np.zeros(shape, dtype=np.int64)
    visits = np.zeros(shape, dtype=np.int64)
    layer_visits = np.zeros((2, len(layers)), dtype=np.int64)
    pos = 4 + count * 6
    event_counts = [0, 0]
    while pos < words.size:
        if words.size - pos < 3:
            raise ValueError('truncated event header')
        phase, n, timestamp = map(int, words[pos:pos + 3]); pos += 3
        if phase not in (0, 1) or n > words.size - pos:
            raise ValueError('invalid event')
        selected = words[pos:pos + n]; pos += n
        if not args.start_ns <= timestamp < args.end_ns:
            continue
        if not n or np.any(selected >= count):
            raise ValueError('empty event or invalid expert index')
        selected = selected.astype(np.int64)
        batch_layers = experts[selected, 1]
        if np.any(batch_layers != batch_layers[0]):
            raise ValueError('event spans multiple layers')
        layer = lindex[int(batch_layers[0])]
        expert_cols = np.array([eindex[int(x)] for x in experts[selected, 2]])
        np.add.at(selections[phase, layer], expert_cols, 1)
        visits[phase, layer, np.unique(expert_cols)] += 1
        layer_visits[phase, layer] += 1
        event_counts[phase] += 1
    if not sum(event_counts):
        raise ValueError('no events in selected interval')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['phase', 'layer', 'expert', 'selections', 'selection_share', 'demand_visits', 'layer_visits', 'demand_visit_fraction'])
        for phase, name in ((0, 'decode'), (1, 'prefill')):
            for li, layer in enumerate(layers):
                total, nvisits = int(selections[phase, li].sum()), int(layer_visits[phase, li])
                for ei, expert in enumerate(ids):
                    count_sel, count_visits = int(selections[phase, li, ei]), int(visits[phase, li, ei])
                    writer.writerow([name, layer, expert, count_sel, count_sel / total if total else 0,
                                     count_visits, nvisits, count_visits / nvisits if nvisits else 0])
    summary = {'phase_events': dict(zip(('decode', 'prefill'), event_counts)), 'layers': len(layers),
               'experts_per_layer': len(ids), 'start_ns': args.start_ns, 'end_ns': args.end_ns, 'phases': {}}
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for phase, name in ((0, 'Decode / verification'), (1, 'Prefill')):
        totals = selections[phase].sum(axis=1, keepdims=True)
        shares = np.divide(selections[phase], totals, out=np.zeros_like(selections[phase], dtype=float), where=totals != 0)
        sorted_shares = -np.sort(-shares, axis=1)
        top10, top20 = sorted_shares[:, :10].sum(axis=1), sorted_shares[:, :20].sum(axis=1)
        nonempty = totals[:, 0] > 0
        heat = axs[phase, 0].imshow(shares * 100, aspect='auto', interpolation='nearest', origin='lower', cmap='magma')
        axs[phase, 0].set(title=f'{name}: selection share by expert', xlabel='Expert ID', ylabel='Layer')
        fig.colorbar(heat, ax=axs[phase, 0], label='% of layer selections')
        axs[phase, 1].plot(layers, top10 * 100, label='10 most selected')
        axs[phase, 1].plot(layers, top20 * 100, label='20 most selected')
        axs[phase, 1].axhline(100 * 10 / len(ids), color='C0', linestyle=':', alpha=.6)
        axs[phase, 1].axhline(100 * 20 / len(ids), color='C1', linestyle=':', alpha=.6)
        axs[phase, 1].set(title=f'{name}: concentration within each layer', xlabel='Layer', ylabel='% of layer selections', ylim=(0, 100))
        axs[phase, 1].legend(loc='upper right')
        if nonempty.any():
            summary['phases'][name] = {'selections': int(selections[phase].sum()), 'layer_visits': int(layer_visits[phase].sum()),
                'top10_share_median': float(np.median(top10[nonempty])), 'top10_share_range': [float(top10[nonempty].min()), float(top10[nonempty].max())],
                'top20_share_median': float(np.median(top20[nonempty])), 'top20_share_range': [float(top20[nonempty].min()), float(top20[nonempty].max())]}
    fig.suptitle('Routed expert demand: selected trace window\nDotted lines show uniform-selection baseline; decode includes speculative verification rows')
    fig.savefig(args.output.with_suffix('.png'), dpi=160)
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
