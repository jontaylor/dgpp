#!/usr/bin/env python3
"""Causal cache-policy replay on recorded DGPP routing, without model execution.

Requires NumPy. This is an open-loop policy diagnostic, NOT a latency simulator.
Routes and host observation times remain fixed. Recorded device quiet intervals
are an exogenous, retrospective service envelope; a predictor never sees future
routes, reads or quiet windows. The fluid I/O model can pause at demand activity.
It does not model nonpreemptible SSD commands, GPU copies or changed execution
times. An explicit readiness delay approximates copies. Do not infer tok/s.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


@dataclass
class Event:
    seq: int
    ns: int
    phase: int
    layer: int
    selections: int
    ids: np.ndarray


def read_demand(path):
    raw = path.read_bytes()
    if len(raw) % 8:
        raise ValueError('truncated demand trace')
    w = np.frombuffer(raw, dtype='<u8')
    if len(w) < 4 or tuple(map(int, w[:2])) != (0x31444e414d454744, 1):
        raise ValueError('invalid demand header')
    slots, n = map(int, w[2:4])
    if not n or len(w) < 4 + n*6:
        raise ValueError('truncated expert metadata')
    meta = w[4:4+n*6].reshape(n, 6).astype(np.int64)
    if np.any(meta[:, 0]) or np.any(meta[:, 3] <= 0):
        raise ValueError('requires target-only experts with positive chunk counts')
    if np.any((meta[:, 4] != 0) & (meta[:, 4] != meta[:, 3])):
        raise ValueError('whole-expert replay does not support partial initial residency')
    if len(set(map(tuple, meta[:, :3]))) != n or int(meta[:, 4].sum()) > slots:
        raise ValueError('duplicate identity or initial residency exceeds slot budget')
    if len(set(meta[:, 3])) != 1 or len(set(meta[:, 5])) != 1:
        raise ValueError('first replay requires equal-sized experts')
    layers = sorted(set(map(int, meta[:, 1])))
    if layers != list(range(len(layers))):
        raise ValueError('requires contiguous target layers starting at zero')
    events, pos = [], 4+n*6
    while pos < len(w):
        if len(w)-pos < 3:
            raise ValueError('truncated event header')
        phase, count, ns = map(int, w[pos:pos+3]); pos += 3
        if phase not in (0, 1) or not count or count > len(w)-pos:
            raise ValueError('invalid demand event')
        ids = np.unique(w[pos:pos+count]).astype(np.int64); pos += count
        if ids[-1] >= n or len(set(meta[ids, 1])) != 1:
            raise ValueError('invalid IDs or mixed-layer event')
        if events and ns < events[-1].ns:
            raise ValueError('host clock regressed; preserve sequence, do not sort it away')
        events.append(Event(len(events), ns, phase, int(meta[ids[0], 1]), count, ids))
    if not events:
        raise ValueError('empty demand trace')
    return slots, meta, events


def match_reads(path, meta, events):
    """Evaluator only: assign reads to the last preceding visit of their layer."""
    raw = path.read_bytes()
    if len(raw) % 64:
        raise ValueError('truncated read trace')
    by_layer = defaultdict(list)
    for ev in events:
        by_layer[ev.layer].append(ev.seq)
    times = {l: np.array([events[i].ns for i in seq], dtype=np.uint64)
             for l, seq in by_layer.items()}
    lookup = {(int(e[1]), int(e[2])): i for i, e in enumerate(meta)}
    actual = [Counter() for _ in events]
    coverage = defaultdict(set)
    credits = []
    for start, end, layer, expert, chunk, draft, size, _ in struct.iter_unpack('<8Q', raw):
        if draft or (layer, expert) not in lookup or end < start:
            raise ValueError('read/cache identity mismatch or invalid interval')
        e = lookup[layer, expert]
        if chunk >= meta[e, 3] or size*meta[e, 3] != meta[e, 5]:
            raise ValueError('record geometry mismatch')
        at = int(np.searchsorted(times[layer], start, side='right'))-1
        if at < 0:
            raise ValueError('read precedes first demand observation')
        seq = by_layer[layer][at]
        if e not in events[seq].ids:
            raise ValueError('read not selected at attributed visit; unsupported background reads')
        if chunk in coverage[seq, e]:
            raise ValueError('duplicate chunk in attributed expert load')
        coverage[seq, e].add(chunk)
        actual[seq][e] += size
        credits.append((end, size))
    for row in actual:
        if any(size != meta[e, 5] for e, size in row.items()):
            raise ValueError('partial/repeated read: whole-expert attribution is ambiguous')
    return actual, sorted(credits)


def quiet_windows(path):
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    result = []
    for a, b in zip(samples, samples[1:]):
        if len(a) < 12 or len(b) < 12 or b[0] <= a[0]:
            raise ValueError('invalid block-device samples')
        # timestamp, then Linux block stat: reads completed=1, writes=5,
        # I/Os in progress=9. No future quiet window is exposed to scoring.
        if a[9] == b[9] == 0 and a[1] == b[1] and a[5] == b[5]:
            if result and result[-1][1] == a[0]:
                result[-1] = (result[-1][0], b[0])
            else:
                result.append((a[0], b[0]))
    return result


class History:
    """Past-only inclusion frequency by phase and approximate route-count bin."""
    def __init__(self, meta, args):
        self.meta, self.args = meta, args
        self.n = len(meta)
        self.layers = int(meta[:, 1].max())+1
        self.members = [np.flatnonzero(meta[:, 1] == l) for l in range(self.layers)]
        self.data = {}
        self.times = np.array([[args.decode_layer_ms]*self.layers,
                               [args.prefill_layer_ms]*self.layers], dtype=float)
        self.previous = None
        self.scores = np.zeros(self.n)
        self.tau = np.zeros(self.n)
        self.probability = np.zeros(self.n)
        self.cycle = 0

    @staticmethod
    def width(ev):
        # A history key, not a claim about exact token count: capture flags can
        # deduplicate prefill selections. Use the observable route-list length.
        return min(16, (max(1, ev.selections)-1).bit_length())

    def profile(self, key):
        if key not in self.data:
            self.data[key] = (np.zeros(self.n), np.zeros(self.layers))
        return self.data[key]

    def observe(self, ev):
        if self.previous:
            old = self.previous
            dt = (ev.ns-old.ns)/1e6
            if (old.phase == ev.phase and ev.layer == (old.layer+1) % self.layers
                    and self.args.min_layer_sample_ms <= dt <= self.args.max_layer_sample_ms):
                self.times[ev.phase, old.layer] = .9*self.times[ev.phase, old.layer]+.1*dt
        self.previous = ev
        decay = math.exp(-math.log(2)/self.args.history_half_life)
        for key in ((ev.phase, -1), (ev.phase, self.width(ev))):
            counts, visits = self.profile(key)
            counts[self.members[ev.layer]] *= decay
            visits[ev.layer] = visits[ev.layer]*decay+1
            counts[ev.ids] += 1
        broad, broad_visits = self.profile((ev.phase, -1))
        narrow, narrow_visits = self.profile((ev.phase, self.width(ev)))
        use_narrow = narrow_visits[self.meta[:, 1]] >= self.args.shape_min_visits
        counts = np.where(use_narrow, narrow, broad)
        visits = np.where(use_narrow, narrow_visits[self.meta[:, 1]],
                          broad_visits[self.meta[:, 1]])
        self.probability = counts/(visits+self.args.prior_visits)
        times = self.times[ev.phase]
        self.cycle = float(times.sum())
        # Observation is at the routed-work boundary. Approximate the source
        # layer's remaining duration by its learned whole-layer duration.
        tau = np.zeros(self.layers)
        elapsed = 0.
        for distance in range(1, self.layers+1):
            elapsed += times[(ev.layer+distance-1) % self.layers]
            tau[(ev.layer+distance) % self.layers] = elapsed
        self.tau = tau[self.meta[:, 1]]
        benefit = self.probability * (self.args.prefill_miss_ms if ev.phase else self.args.decode_miss_ms)
        # Three discounted future visits: rewards recurring experts beyond
        # the next use; time charges memory until the first benefit.
        weight = self.args.deadline_weight
        reward = benefit*np.exp(-weight*self.tau/self.cycle)*sum(math.exp(-k) for k in range(3))
        self.scores = reward-self.args.memory_rent*self.tau


@dataclass
class Forecast:
    expert: int
    remaining: float
    issued: int
    expires: int
    phase: int
    ready_at: float = math.inf
    read_bytes: float = 0.
    window: str | None = None
    attempted_revision: int = -2


class Policy:
    def __init__(self, name, meta, slots, args, quiet, log=None, credits=(), windows=()):
        self.name, self.meta, self.args = name, meta, args
        self.resident = meta[:, 4] == meta[:, 3]
        self.capacity = slots//int(meta[0, 3])
        self.size = int(meta[0, 5])
        self.chunks = int(meta[0, 3])
        self.history = History(meta, args)
        self.quiet, self.gap_at = quiet, 0
        self.credits, self.credit_at = credits, 0
        self.evaluation_windows = windows
        self.jobs = {}
        self.prefetched = {}  # resident experts whose speculative load is unused
        self.last_used = np.full(len(meta), -1, dtype=np.int64)
        self.protected = np.zeros(len(meta), dtype=bool)
        self.stats = defaultdict(Counter)
        self.windows = defaultdict(Counter)
        self.layer_stats = defaultdict(Counter)
        self.peak_resident = int(self.resident.sum())
        self.peak_staging = 0
        self.last_ns = None
        self.spare_fraction = .1
        self.debt = 0.
        self.log = log
        self.current_seq = -1
        self.clock_ns = 0

    def window_at(self, ns):
        return next((w['name'] for w in self.evaluation_windows
                     if w['start_ns'] <= ns < w['end_ns']), None)

    def emit(self, action, **kwargs):
        if self.log:
            self.log.write(json.dumps(dict(policy=self.name, seq=self.current_seq,
                                           action=action, **kwargs))+'\n')

    def add(self, phase, key, value=1, window=None):
        self.stats[phase][key] += value
        if window is not None:
            self.windows[(window, phase)][key] += value

    def discard(self, e, reason):
        job = self.jobs.pop(e)
        self.add(job.phase, reason, window=job.window)
        self.add(job.phase, 'prefetch_discarded_bytes', job.read_bytes, job.window)
        self.emit(reason, expert=e, bytes=job.read_bytes)

    def admit(self, e, speculative=False):
        if self.resident[e]:
            return True
        if int(self.resident.sum()) >= self.capacity:
            eligible = self.resident & ~self.protected
            if not eligible.any():
                self.emit('admission_rejected', expert=e, reason='all residents protected', speculative=speculative)
                return False
            if self.name == 'lru':
                victim = int(np.argmin(np.where(eligible, self.last_used, np.iinfo(np.int64).max)))
            else:
                victim = int(np.argmin(np.where(eligible, self.history.scores, math.inf)))
                if self.history.scores[e] <= self.history.scores[victim]+self.args.switch_margin_ms:
                    self.add(self.history.previous.phase, 'prefetch_admission_rejected' if speculative else
                             'demand_admission_rejected', window=self.window_at(self.clock_ns))
                    self.emit('admission_rejected', expert=e, victim=victim, speculative=speculative,
                              candidate_score=float(self.history.scores[e]),
                              victim_score=float(self.history.scores[victim]),
                              reason='replacement margin')
                    return False
            self.resident[victim] = False
            phase = self.history.previous.phase
            self.add(phase, 'evictions', window=self.window_at(self.clock_ns))
            if victim in self.prefetched:
                origin, window = self.prefetched.pop(victim)
                self.add(origin, 'prefetch_evicted_unused', window=window)
            self.emit('evict', expert=victim, replacement=e,
                      victim_score=float(self.history.scores[victim]),
                      replacement_score=float(self.history.scores[e]))
        self.resident[e] = True
        self.peak_resident = max(self.peak_resident, int(self.resident.sum()))
        if speculative:
            self.prefetched[e] = (self.jobs[e].phase, self.jobs[e].window)
        return True

    def due(self, now):
        self.clock_ns = now
        # Baseline transfer credit becomes available only when its recorded
        # read completes. Future read labels never affect earlier service.
        while self.credit_at < len(self.credits) and self.credits[self.credit_at][0] <= now:
            self.debt = max(0., self.debt-self.credits[self.credit_at][1])
            self.credit_at += 1
        revision = self.history.previous.seq if self.history.previous else -1
        for e, job in list(self.jobs.items()):
            if job.ready_at <= now and job.attempted_revision != revision:
                job.attempted_revision = revision
                if self.admit(e, speculative=True):
                    del self.jobs[e]
                    self.add(job.phase, 'prefetch_admitted', window=job.window)
                    self.emit('prefetch_admit', expert=e, ready_ns=int(job.ready_at))
                    continue
            if job.expires <= now:
                self.discard(e, 'prefetch_expired')

    def service_bytes(self, job, amount, start, end):
        self.add(job.phase, 'speculative_read_bytes', amount)
        self.emit('prefetch_service', expert=job.expert, start_ns=start, end_ns=end, bytes=amount)
        for w in self.evaluation_windows:
            overlap = max(0, min(end, w['end_ns'])-max(start, w['start_ns']))
            if overlap:
                self.windows[w['name'], job.phase]['speculative_read_bytes'] += amount*overlap/(end-start)

    def advance(self, now):
        if self.last_ns is None:
            self.last_ns = now
            self.due(now)
            return
        previous = self.last_ns
        quiet_ns = 0
        at = previous
        self.due(at)
        while at < now:
            while self.gap_at < len(self.quiet) and self.quiet[self.gap_at][1] <= at:
                self.gap_at += 1
            gap = self.quiet[self.gap_at] if self.gap_at < len(self.quiet) else None
            idle = gap is not None and gap[0] <= at < gap[1]
            boundary = min(now, gap[1] if idle else gap[0] if gap else now)
            if self.credit_at < len(self.credits):
                boundary = min(boundary, self.credits[self.credit_at][0])
            for job in self.jobs.values():
                boundary = min(boundary, job.expires)
                if job.ready_at > at:
                    boundary = min(boundary, job.ready_at)
            loading = [job for job in self.jobs.values() if job.remaining > 0]
            job = min(loading, key=lambda j: (j.expires, j.expert)) if loading else None
            if idle and job:
                boundary = min(boundary, at+max(1, math.ceil((self.debt+job.remaining)/self.args.bandwidth_gbs)))
            if boundary <= at:
                raise AssertionError('clock failed to advance')
            if idle:
                quiet_ns += boundary-at
                debt_service = min(self.debt, (boundary-at)*self.args.bandwidth_gbs)
                self.debt -= debt_service
                service_start = at+debt_service/self.args.bandwidth_gbs
                if job and service_start < boundary:
                    served = min(job.remaining, (boundary-service_start)*self.args.bandwidth_gbs)
                    job.remaining -= served; job.read_bytes += served
                    self.service_bytes(job, served, service_start, boundary)
                    if job.remaining <= 1e-6:
                        job.remaining = 0
                        job.ready_at = boundary+math.ceil(self.args.ready_delay_ms*1e6)
            at = boundary
            self.due(at)
        if now > previous:
            self.spare_fraction = .9*self.spare_fraction+.1*quiet_ns/(now-previous)
        self.last_ns = now

    def step(self, ev, actual, window=None):
        self.current_seq = ev.seq
        self.advance(ev.ns)
        # Only completed prior jobs are considered before seeing current routes.
        cached = self.resident[ev.ids].copy()
        ready = np.array([e in self.jobs and self.jobs[e].ready_at <= ev.ns for e in ev.ids])
        hits = cached | ready
        layer_stats = self.layer_stats[ev.phase, ev.layer]
        layer_stats['demands'] += len(ev.ids)
        layer_stats['hits'] += int(hits.sum())
        layer_stats['observed_misses'] += len(actual)
        layer_stats['observed_read_bytes'] += sum(actual.values())
        old = self.protected
        old[:] = False; old[ev.ids] = True
        self.history.observe(ev)  # current demand known; future routes unavailable
        self.last_used[ev.ids] = ev.seq
        self.add(ev.phase, 'layer_visits', window=window)
        self.add(ev.phase, 'unique_expert_demands', len(ev.ids), window)
        self.add(ev.phase, 'observed_read_bytes', sum(actual.values()), window)
        demand_bytes = 0.
        for e, hit in zip(ev.ids, hits):
            e = int(e)
            actual_hit = e not in actual
            agreement = ('both_hit' if hit and actual_hit else 'sim_hit_observed_miss' if hit else
                         'sim_miss_observed_hit' if actual_hit else 'both_miss')
            self.add(ev.phase, agreement, window=window)
            layer_stats[agreement] += 1
            if hit:
                self.add(ev.phase, 'hits', window=window)
                if e in self.prefetched:
                    origin, origin_window = self.prefetched.pop(e)
                    self.add(origin, 'prefetch_used', window=origin_window)
            else:
                self.add(ev.phase, 'misses', window=window)
            if e in self.jobs:
                job = self.jobs.pop(e)
                if hit:
                    self.add(job.phase, 'prefetch_used', window=job.window)
                else:
                    self.add(job.phase, 'prefetch_late', window=job.window)
                needed = job.remaining
                if not hit:
                    demand_bytes += needed
            elif not hit:
                demand_bytes += self.size
        # Admission happens after demand accounting. Prioritize the strongest
        # candidates instead of letting ascending expert IDs fill all space.
        candidates = sorted(map(int, ev.ids), key=lambda e:
                            (-self.history.scores[e], e) if self.name != 'lru' else (0, e))
        for e in candidates:
            self.admit(e)
        self.add(ev.phase, 'demand_read_bytes', demand_bytes, window)
        layer_stats['demand_read_bytes'] += demand_bytes
        # All demanded bytes start as debt. Baseline read completions service
        # debt only at their recorded times; no future labels grant credit.
        self.debt += demand_bytes
        self.emit('visit', phase=ev.phase, layer=ev.layer, ns=ev.ns,
                  demands=len(ev.ids), hits=int(hits.sum()), demand_bytes=demand_bytes,
                  observed_bytes=sum(actual.values()), resident=int(self.resident.sum()),
                  staging_experts=len(self.jobs), debt_bytes=self.debt)
        if self.name == 'deadline-prefetch':
            self.forecast(ev)

    def forecast(self, ev):
        for e, job in list(self.jobs.items()):
            if job.phase != ev.phase:
                self.discard(e, 'prefetch_phase_cancelled')
        max_jobs = min(self.args.max_prefetch_experts,
                       (self.args.staging_buffers-self.args.demand_reserve_buffers)//self.chunks)
        while len(self.jobs) < max_jobs:
            available = ~self.resident & ~self.protected & (self.history.probability > 0)
            for e in self.jobs:
                available[e] = False
            if not available.any():
                break
            io_ms = self.size/(self.args.bandwidth_gbs*1e6)
            predicted_load_ms = io_ms/max(.05, self.spare_fraction)+self.args.ready_delay_ms
            # Holding staging and occupying a share of scarce spare I/O are
            # charged separately from persistent-cache holding cost.
            cost = self.args.io_price*io_ms/max(.05, self.spare_fraction)
            value = self.history.scores-cost-self.args.staging_rent*self.history.tau
            if int(self.resident.sum()) >= self.capacity:
                victims = self.resident & ~self.protected
                if not victims.any():
                    break
                value -= max(0., float(self.history.scores[victims].min()))
            available &= self.history.tau > predicted_load_ms
            e = int(np.argmax(np.where(available, value, -math.inf)))
            if not available[e] or value[e] <= self.args.switch_margin_ms:
                break
            expires = ev.ns+int((self.history.tau[e]+self.history.cycle)*1e6)
            self.jobs[e] = Forecast(e, self.size, ev.ns, expires, ev.phase,
                                    window=self.window_at(ev.ns))
            self.peak_staging = max(self.peak_staging, len(self.jobs)*self.chunks)
            self.add(ev.phase, 'prefetch_issued', window=self.window_at(ev.ns))
            self.emit('prefetch_issue', expert=e, target_layer=int(self.meta[e, 1]),
                      probability=float(self.history.probability[e]),
                      tau_ms=float(self.history.tau[e]), net_value=float(value[e]),
                      spare_fraction=self.spare_fraction, expires_ns=expires)

    def report(self):
        for job in self.jobs.values():
            self.add(job.phase, 'prefetch_tail_pending', window=job.window)
        for phase, window in self.prefetched.values():
            self.add(phase, 'prefetch_tail_resident_unused', window=window)
        for c in list(self.stats.values())+list(self.windows.values()):
            c['total_read_bytes'] = c['demand_read_bytes']+c['speculative_read_bytes']
            c['net_read_byte_delta_vs_observed'] = c['total_read_bytes']-c['observed_read_bytes']
        return dict(phases={('prefill' if p else 'decode'): dict(c) for p, c in self.stats.items()},
                    windows={f'{name}:{"prefill" if p else "decode"}': dict(c)
                             for (name, p), c in self.windows.items()},
                    layers={f'{"prefill" if p else "decode"}:{l}': dict(c)
                            for (p, l), c in sorted(self.layer_stats.items())},
                    peak_resident_experts=self.peak_resident,
                    resident_capacity_experts=self.capacity,
                    peak_speculative_staging_buffers=self.peak_staging,
                    ending_unserviced_demand_debt_bytes=self.debt)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demand', type=Path, required=True)
    p.add_argument('--reads', type=Path, required=True)
    p.add_argument('--disk-samples', type=Path, required=True)
    p.add_argument('--windows', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--decisions', type=Path)
    p.add_argument('--bandwidth-gbs', type=float, default=6.)
    p.add_argument('--staging-buffers', type=int, default=256)
    p.add_argument('--demand-reserve-buffers', type=int, default=36)
    p.add_argument('--max-prefetch-experts', type=int, default=1)
    p.add_argument('--ready-delay-ms', type=float, default=1.)
    p.add_argument('--decode-layer-ms', type=float, default=1.6)
    p.add_argument('--prefill-layer-ms', type=float, default=20.)
    p.add_argument('--decode-miss-ms', type=float, default=3.2)
    p.add_argument('--prefill-miss-ms', type=float, default=1.7)
    p.add_argument('--history-half-life', type=float, default=64.)
    p.add_argument('--deadline-weight', type=float, default=1.)
    p.add_argument('--prior-visits', type=float, default=4.)
    p.add_argument('--shape-min-visits', type=int, default=2)
    p.add_argument('--memory-rent', type=float, default=.001)
    p.add_argument('--staging-rent', type=float, default=.001)
    p.add_argument('--io-price', type=float, default=.1)
    p.add_argument('--switch-margin-ms', type=float, default=.05)
    p.add_argument('--min-layer-sample-ms', type=float, default=.2)
    p.add_argument('--max-layer-sample-ms', type=float, default=1000.)
    return p


def main():
    p = parser(); args = p.parse_args()
    positive = ('bandwidth_gbs', 'decode_layer_ms', 'prefill_layer_ms', 'history_half_life', 'prior_visits')
    nonnegative = ('ready_delay_ms', 'decode_miss_ms', 'prefill_miss_ms', 'memory_rent', 'deadline_weight',
                   'staging_rent', 'io_price', 'switch_margin_ms', 'min_layer_sample_ms')
    if any(not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0 for k in positive):
        p.error('bandwidth, layer times, half-life and prior visits must be finite and positive')
    if any(not math.isfinite(getattr(args, k)) or getattr(args, k) < 0 for k in nonnegative):
        p.error('costs and readiness delay must be finite and nonnegative')
    if (args.staging_buffers <= 0 or not 0 <= args.demand_reserve_buffers <= args.staging_buffers
            or args.max_prefetch_experts < 0 or args.shape_min_visits < 1
            or not math.isfinite(args.max_layer_sample_ms)
            or args.max_layer_sample_ms < args.min_layer_sample_ms):
        p.error('invalid buffer, forecast or layer-sample limits')
    slots, meta, events = read_demand(args.demand)
    actual, credits = match_reads(args.reads, meta, events)
    quiet = quiet_windows(args.disk_samples)
    windows = json.loads(args.windows.read_text()) if args.windows else []
    if any(w['start_ns'] >= w['end_ns'] for w in windows):
        raise ValueError('invalid evaluation window')
    if any(a['end_ns'] > b['start_ns'] for a, b in zip(windows, windows[1:])):
        raise ValueError('evaluation windows must be ordered and nonoverlapping')
    # Work in relative nanoseconds: fractional fluid-service calculations on
    # Unix epoch floats otherwise lose hundreds of ns and can fail to advance.
    epoch = events[0].ns
    for ev in events:
        ev.ns -= epoch
    credits = [(ns-epoch, size) for ns, size in credits]
    quiet = [(max(a-epoch, 0), min(b-epoch, events[-1].ns)) for a, b in quiet
             if b > epoch and a-epoch < events[-1].ns]
    windows = [dict(w, start_ns=w['start_ns']-epoch, end_ns=w['end_ns']-epoch) for w in windows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.decisions.open('w') if args.decisions else None
    try:
        policies = [Policy(name, meta, slots, args, quiet, log, credits, windows)
                    for name in ('lru', 'deadline', 'deadline-prefetch')]
        overlap = []
        for ev, observed in zip(events, actual):
            window = next((w['name'] for w in windows if w['start_ns'] <= ev.ns < w['end_ns']), None)
            for policy in policies:
                policy.step(ev, observed, window)
                if int(policy.resident.sum()) > policy.capacity:
                    raise AssertionError('resident capacity exceeded')
                if len(policy.jobs)*policy.chunks > args.staging_buffers-args.demand_reserve_buffers:
                    raise AssertionError('speculative staging exceeded')
            if ev.seq % 1000 == 0:
                a, b = policies[1].resident, policies[2].resident
                overlap.append(dict(seq=ev.seq, ns=ev.ns, phase=ev.phase,
                                    resident_intersection=int((a & b).sum()),
                                    resident_union=int((a | b).sum())))
        result = dict(
            format='dgpp-cache-whatif-v1', mode='fixed-recorded-host-clock',
            trace_epoch_unix_ns=epoch,
            replay_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=[
                'No predicted latency, speedup, physical SSD savings or quality change.',
                'Host demand observations, especially native hits, are not routing deadlines.',
                'Fluid speculative service only in sampled recorded device-quiet intervals.',
                'Quiet intervals can conceal sub-sample activity; transfers can pause instantly.',
                'A fixed delay approximates copy/pack readiness; engine execution is not simulated.',
                'Demand is charged immediately; baseline transfer credit arrives only on recorded completion.',
                'Baseline busy windows are never reclaimed; unpaid demand consumes later spare capacity.',
                'Demand service and admission are atomic at each visit; batch execution order is absent.',
                'Actual cache contents/evictions are unrecorded; agreement covers demanded experts only.',
                'Route-count bins are observable list sizes, not guaranteed exact batch row counts.',
                'All trace prefixes train causally; evaluation windows do not reset state.',
                'Decision-log times are nanoseconds relative to trace_epoch_unix_ns.',
                'Unconsumed tail forecasts are censored, not counted as proven waste.'],
            assumptions={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            input_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in [('demand', args.demand), ('reads', args.reads),
                                             ('disk_samples', args.disk_samples)]},
            events=len(events), experts=len(meta), cache_slots=slots,
            noncyclic_same_phase_transitions={('prefill' if phase else 'decode'):
                sum(a.phase == b.phase == phase and b.layer != (a.layer+1) % (int(meta[:, 1].max())+1)
                    for a, b in zip(events, events[1:])) for phase in (0, 1)},
            expert_chunks=int(meta[0, 3]), expert_file_bytes=int(meta[0, 5]),
            initial_resident_experts=int((meta[:, 4] == meta[:, 3]).sum()),
            quiet_seconds=sum(b-a for a, b in quiet)/1e9,
            policies={policy.name: policy.report() for policy in policies},
            retention_prefetch_cache_overlap=overlap)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({name: report['phases'] for name, report in result['policies'].items()}, indent=2))
    finally:
        if log:
            log.close()


if __name__ == '__main__':
    main()
