#!/usr/bin/env python3
"""Fit bounded cache-score weights on an initial prefix; evaluate frozen weights.

Requires NumPy and fork support. All candidate decisions remain causal; only
prefix outcomes select weights. No latency/global-optimum claims. Later
requests are excluded from selection, not asserted to be pristine held-out
data. Existing what-if traces can already have been inspected. Keep reports
untracked. The recorded model and server are never executed or modified.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import time

import expert_cache_whatif as sim


DATA = None


def metrics(report):
    result = {}
    for phase in ('prefill', 'decode'):
        s = report['phases'].get(phase, {})
        result[phase] = {
            'demands': s.get('unique_expert_demands', 0),
            'misses': s.get('misses', 0),
            'demand_bytes': s.get('demand_read_bytes', 0),
            'total_bytes': s.get('total_read_bytes', 0),
            'observed_misses': s.get('both_miss', 0)+s.get('sim_hit_observed_miss', 0),
            'observed_bytes': s.get('observed_read_bytes', 0)}
    return result


def ratios(m, baseline):
    return {p: m[p]['misses']/max(1, baseline[p]['misses']) for p in ('prefill', 'decode')}


def rank_key(m, baseline):
    r = ratios(m, baseline)
    return (max(r.values()), sum(r.values()), sum(s['total_bytes'] for s in m.values()))


def dominates(a, b):
    # Preserve phase distinctions instead of hiding a regression in a sum.
    aa = [a[p][k] for p in ('prefill', 'decode') for k in ('misses', 'total_bytes')]
    bb = [b[p][k] for p in ('prefill', 'decode') for k in ('misses', 'total_bytes')]
    return all(x <= y for x, y in zip(aa, bb)) and any(x < y for x, y in zip(aa, bb))


def run(task):
    slots, meta, events, actual, quiet, credits, windows, base, train_end, split_ns = DATA
    args = copy.copy(base)
    for key, value in task['weights'].items():
        setattr(args, key, value)
    policy = sim.Policy(task['policy'], meta, slots, args, quiet, credits=credits, windows=windows)
    stop = train_end if task['training_only'] else len(events)
    initial_stats = None
    for i in range(stop):
        ev = events[i]
        window = policy.window_at(ev.ns)
        if i == train_end:
            policy.advance(split_ns)
            initial_stats = copy.deepcopy(policy.stats)
        policy.step(ev, actual[i], window)
        if int(policy.resident.sum()) > policy.capacity:
            raise AssertionError('resident capacity exceeded')
        if len(policy.jobs)*policy.chunks > args.staging_buffers-args.demand_reserve_buffers:
            raise AssertionError('staging capacity exceeded')
    if task['training_only']:
        policy.advance(split_ns)
    report = policy.report()
    result = dict(id=task['id'], policy=task['policy'], weights=task['weights'],
                  metrics=metrics(report), report=report)
    if initial_stats is not None:
        # Phase resource bytes and outcome counters, rather than summing the
        # per-request windows: this includes activity between requests too.
        evaluation = dict(phases=copy.deepcopy(report['phases']))
        for phase, counters in evaluation['phases'].items():
            old = initial_stats.get(1 if phase == 'prefill' else 0, {})
            for key in list(counters):
                counters[key] -= old.get(key, 0)
            # Derived totals were not yet materialized at the split.
            counters['total_read_bytes'] = counters.get('demand_read_bytes', 0)+counters.get('speculative_read_bytes', 0)
            counters['net_read_byte_delta_vs_observed'] = counters['total_read_bytes']-counters.get('observed_read_bytes', 0)
        result['evaluation'] = metrics(evaluation)
        result['evaluation_report'] = evaluation
    return result


def batch(tasks, workers, stage, output):
    results = []
    context = multiprocessing.get_context('fork')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        pending = {pool.submit(run, t): t['id'] for t in tasks}
        for future in as_completed(pending):
            row = future.result()
            results.append(row)
            compact = dict(id=row['id'], weights=row['weights'], metrics=row['metrics'])
            with (output/f'{stage}.progress.jsonl').open('a') as out:
                out.write(json.dumps(compact)+'\n')
            print(json.dumps(dict(stage=stage, done=len(results), total=len(tasks), **compact)), flush=True)
    return sorted(results, key=lambda r: r['id'])


def choose(rows, baseline, io_headroom=None):
    feasible = []
    for row in rows:
        m = row['metrics']
        if any(m[p]['misses'] > baseline[p]['misses'] for p in baseline):
            continue
        if io_headroom is not None and any(
                m[p]['total_bytes'] > baseline[p]['total_bytes']*(1+io_headroom) for p in baseline):
            continue
        feasible.append(row)
    # A null result explicitly selects the control, not the least-bad regression.
    return min(feasible, key=lambda r: (*rank_key(r['metrics'], baseline),
                                       r['policy'] == 'deadline-prefetch', r['id'])) if feasible else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('demand', 'reads', 'disk-samples', 'windows', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--train-requests', type=int, default=2)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--io-headroom', type=float, default=.1)
    args = p.parse_args()
    if args.train_requests < 1 or args.workers < 1 or not math.isfinite(args.io_headroom) or args.io_headroom < 0:
        p.error('invalid training count, worker count or I/O headroom')
    slots, meta, events = sim.read_demand(args.demand)
    actual, credits = sim.match_reads(args.reads, meta, events)
    quiet = sim.quiet_windows(args.disk_samples)
    windows = json.loads(args.windows.read_text())
    if not 1 <= args.train_requests < len(windows):
        p.error('training requests must leave at least one evaluation request')
    if any(w['start_ns'] >= w['end_ns'] for w in windows) or any(
            a['end_ns'] > b['start_ns'] for a, b in zip(windows, windows[1:])):
        p.error('windows must be ordered, valid and nonoverlapping')
    split = windows[args.train_requests-1]['end_ns']
    train_end = next((i for i, ev in enumerate(events) if ev.ns >= split), len(events))
    if train_end == len(events) or not train_end:
        p.error('split leaves no training or evaluation events')
    epoch = events[0].ns
    for ev in events:
        ev.ns -= epoch
    quiet = [(max(0, a-epoch), min(b-epoch, events[-1].ns)) for a, b in quiet
             if b > epoch and a-epoch < events[-1].ns]
    credits = [(ns-epoch, size) for ns, size in credits]
    windows = [dict(w, start_ns=w['start_ns']-epoch, end_ns=w['end_ns']-epoch) for w in windows]
    base = sim.parser().parse_args(['--demand', str(args.demand), '--reads', str(args.reads),
                                  '--disk-samples', str(args.disk_samples), '--output', 'unused'])
    global DATA
    DATA = slots, meta, events, actual, quiet, credits, windows, base, train_end, split-epoch
    args.output.mkdir(parents=True, exist_ok=True)
    # A reusable output directory could accidentally mix different searches.
    if (args.output/'plan.json').exists():
        p.error('output already contains a fitting plan; use a fresh directory')
    grid = dict(deadline_weight=[0., .25, 1.], memory_rent=[0., .0001, .001],
                switch_margin_ms=[0., .05])
    combinations = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    plan = dict(training_requests=[w['name'] for w in windows[:args.train_requests]],
                evaluation_requests=[w['name'] for w in windows[args.train_requests:]],
                training_events=train_end, evaluation_events=len(events)-train_end,
                retention_grid=grid,
                prefetch_grid=dict(io_price=[0., .1], staging_rent=[0., .001], max_prefetch_experts=[1, 4]),
                selection='No phase may increase training misses vs control; minimize worst phase miss ratio, then sum ratios, then total bytes.',
                prefetch_io_headroom=args.io_headroom,
                trace_epoch_unix_ns=epoch,
                input_sha256={key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
                              for key in ('demand', 'reads', 'disk_samples', 'windows')},
                source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (Path(__file__), Path(sim.__file__))},
                limits=['Finite tested family, not a global optimum or latency model.',
                        'Chronological selection split, not an independently held-out workload.',
                        'No future routes or quiet windows enter candidate scoring.',
                        'Later outcomes never enter weight selection.',
                        'Weights fit rank zero only; rank peers are not independent traffic.'])
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    started = time.monotonic()
    tasks = [dict(id='lru-control', policy='lru', weights={}, training_only=True)]
    tasks += [dict(id=f'retention-{i:02d}', policy='deadline', weights=w, training_only=True)
              for i, w in enumerate(combinations)]
    fitted = batch(tasks, args.workers, 'retention', args.output)
    control = next(r for r in fitted if r['id'] == 'lru-control')
    candidates = [r for r in fitted if r['id'] != 'lru-control']
    selected = choose(candidates, control['metrics']) or control
    frontier = [r['id'] for r in candidates if not any(
        dominates(other['metrics'], r['metrics']) for other in [control]+candidates if other is not r)]
    frozen = dict(selected_retention=selected['id'], weights=selected['weights'],
                  policy=selected['policy'], training_metrics=selected['metrics'],
                  retention_pareto_frontier=frontier)
    best_retention = min(candidates, key=lambda r: (*rank_key(r['metrics'], control['metrics']), r['id']))
    frozen['best_tested_family_retention'] = {k: best_retention[k] for k in ('id', 'weights', 'metrics')}
    # Record retention-only results before the predeclared joint search.
    (args.output/'retention-selection.json').write_text(json.dumps(frozen, indent=2)+'\n')
    pref_grid = plan['prefetch_grid']
    tasks = []
    for i, weights in enumerate(combinations):
        for j, values in enumerate(itertools.product(*pref_grid.values())):
            tasks.append(dict(id=f'prefetch-{i:02d}-{j:02d}', policy='deadline-prefetch',
                              weights=dict(weights, **dict(zip(pref_grid, values))), training_only=True))
    prefetch = batch(tasks, args.workers, 'joint', args.output)
    # Search the entire finite joint grid, not just prefetch on one retention
    # winner. Neither phase may regress in misses; I/O may grow by the stated
    # headroom. Retention-only and LRU remain eligible zero-speculation choices.
    joint = choose(fitted+prefetch, control['metrics'], args.io_headroom) or control
    frozen['selected_joint'] = joint['id']
    frozen['joint_policy'] = joint['policy']
    frozen['joint_weights'] = joint['weights']
    frozen['joint_training_metrics'] = joint['metrics']
    best_joint = min(candidates+prefetch, key=lambda r: (*rank_key(r['metrics'], control['metrics']),
                                                       r['policy'] == 'deadline-prefetch', r['id']))
    frozen['best_tested_family_joint'] = {k: best_joint[k] for k in ('id', 'weights', 'metrics')}
    (args.output/'frozen-selection.json').write_text(json.dumps(frozen, indent=2)+'\n')
    tasks = [dict(id='lru-control', policy='lru', weights={}, training_only=False),
             dict(id='original-deadline', policy='deadline', weights={}, training_only=False),
             dict(id='fitted-retention', policy=selected['policy'], weights=selected['weights'], training_only=False)]
    if joint['id'] != selected['id']:
        tasks.append(dict(id='fitted-joint', policy=joint['policy'],
                          weights=joint['weights'], training_only=False))
        if joint['policy'] == 'deadline-prefetch':
            tasks.append(dict(id='joint-without-prefetch', policy='deadline',
                              weights=joint['weights'], training_only=False))
    # Even a training-dominated family has a best tested point. Evaluate those
    # points descriptively, selected by the same prefix-only ranking, rather
    # than presenting an LRU fallback as fitted coefficients for this family.
    for label, row in [('best-family-retention', best_retention), ('best-family-joint', best_joint)]:
        tasks.append(dict(id=label, policy=row['policy'], weights=row['weights'], training_only=False))
    evaluated = batch(tasks, args.workers, 'evaluation', args.output)
    result = dict(plan=plan, frozen_selection=frozen,
                  training_retention=[dict(id=r['id'], weights=r['weights'], metrics=r['metrics']) for r in fitted],
                  training_prefetch=[dict(id=r['id'], weights=r['weights'], metrics=r['metrics']) for r in prefetch],
                  evaluated=evaluated, elapsed_seconds=time.monotonic()-started)
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(frozen_selection=frozen, elapsed_seconds=result['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
