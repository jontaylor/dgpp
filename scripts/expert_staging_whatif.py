#!/usr/bin/env python3
"""Replay staging-only prefetch while freezing the recorded deployed residency.

No model/SSD operations. Requires NumPy, DGEMAND1, DGRSIDE1, read and device
traces from the same rank/process/cache. Readiness uses fixed host observation
times and the shared fluid I/O model, not actual routing deadlines or latency.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import expert_cache_whatif as sim


def read_residency(path, slots, meta, events):
    raw = path.read_bytes()
    if len(raw) % 8 or len(raw) < 32 or raw[:8] != b'DGRSIDE1':
        raise ValueError('invalid residency header')
    w = np.frombuffer(raw, dtype='<u8')
    version, count, capacity = map(int, w[1:4])
    if version != 1 or count != len(meta) or capacity != slots or len(w) < 4+count:
        raise ValueError('residency/demand geometry mismatch')
    current = w[4:4+count].astype(np.int64).copy()
    if not np.array_equal(current, meta[:, 4]):
        raise ValueError('residency/demand initial state mismatch')
    pos, deltas = 4+count, []
    while pos < len(w):
        if len(w)-pos < 3:
            raise ValueError('truncated residency event')
        seq, ns, n = map(int, w[pos:pos+3]); pos += 3
        if seq != len(deltas) or seq >= len(events) or ns != events[seq].ns or n > (len(w)-pos)//2:
            raise ValueError('residency/demand event mismatch')
        changes = w[pos:pos+2*n].reshape(n, 2).astype(np.int64); pos += 2*n
        if n:
            ids, chunks = changes[:, 0], changes[:, 1]
            if ids.min() < 0 or ids.max() >= count or len(set(ids)) != n:
                raise ValueError('invalid or duplicate residency delta')
            if np.any((chunks != 0) & (chunks != meta[ids, 3])):
                raise ValueError('staging-only replay requires whole-expert residency')
            current[ids] = chunks
        if current.sum() > slots:
            raise ValueError('recorded residency exceeds capacity')
        deltas.append(changes)
    if len(deltas) != len(events):
        raise ValueError('incomplete residency observations')
    return deltas


class StagingPolicy(sim.Policy):
    def __init__(self, name, meta, slots, args, quiet, log=None, credits=(), windows=(), enabled=True):
        super().__init__(name, meta, slots, args, quiet, log, credits, windows)
        self.enabled = enabled
        self.held = []
        self.ready_demands = []

    def admit(self, expert, speculative=False):
        # Persistent admission/eviction is exclusively the recorded policy.
        return False

    def due(self, now):
        super().due(now)
        self.held = [end for end in self.held if end > now]

    def consume(self, job, now):
        # A source cannot be recycled at routing time while compute/copies
        # still read it. This explicit assumed hold is not a measured lifetime.
        ready = max(now, job.ready_at if math.isfinite(job.ready_at) else
                    now+job.remaining/self.args.bandwidth_gbs+self.args.ready_delay_ms*1e6)
        self.held.append(ready+self.args.consumer_hold_ms*1e6)

    def discard(self, e, reason):
        job = self.jobs[e]
        # Cancellation cannot recycle a completed read's source while the
        # modeled readiness copy is still using it. Fluid reads themselves
        # remain instantly pausable/cancellable, an explicit model limitation.
        if job.remaining == 0 and job.ready_at > self.clock_ns and math.isfinite(job.ready_at):
            self.held.append(job.ready_at)
        super().discard(e, reason)

    def report(self):
        result = super().report()
        result['ending_copy_or_consumer_holds'] = len(self.held)
        result['ready_demands'] = self.ready_demands
        result['configuration'] = {key: getattr(self.args, key) for key in (
            'max_prefetch_experts', 'io_price', 'staging_rent', 'staging_buffers',
            'demand_reserve_buffers', 'bandwidth_gbs', 'ready_delay_ms',
            'consumer_hold_ms', 'history_half_life', 'prior_visits',
            'shape_min_visits', 'decode_layer_ms', 'prefill_layer_ms',
            'decode_miss_ms', 'prefill_miss_ms')}
        return result

    def step_recorded(self, ev, actual, changes):
        self.current_seq = ev.seq
        self.advance(ev.ns)
        if len(changes):
            self.resident[changes[:, 0]] = changes[:, 1] == self.meta[changes[:, 0], 3]
        self.peak_resident = max(self.peak_resident, int(self.resident.sum()))
        missing = set(map(int, ev.ids[~self.resident[ev.ids]]))
        if missing != set(actual):
            raise ValueError(f'visit {ev.seq}: recorded residency and file-read misses disagree')
        # Only the newly observed resident state is visible; never consult a
        # future delta to skip a forecast that will later prove redundant.
        for e in list(self.jobs):
            if self.resident[e]:
                self.discard(e, 'prefetch_became_resident')
        self.history.observe(ev)
        self.last_used[ev.ids] = ev.seq
        self.protected[:] = False; self.protected[ev.ids] = True
        window = self.window_at(ev.ns)
        self.add(ev.phase, 'layer_visits', window=window)
        self.add(ev.phase, 'unique_expert_demands', len(ev.ids), window)
        self.add(ev.phase, 'observed_read_bytes', sum(actual.values()), window)
        self.add(ev.phase, 'persistent_misses', len(missing), window)
        demand_bytes = 0.
        layer = self.layer_stats[ev.phase, ev.layer]
        layer['demands'] += len(ev.ids); layer['persistent_misses'] += len(missing)
        for e in map(int, ev.ids):
            if e not in missing:
                self.add(ev.phase, 'persistent_hits', window=window)
                continue
            job = self.jobs.pop(e, None)
            if job and job.ready_at <= ev.ns:
                self.ready_demands.append((ev.seq, e, ev.phase))
                self.add(ev.phase, 'ready_staging_hits', window=window)
                self.add(job.phase, 'prefetch_used', window=job.window)
                self.add(ev.phase, 'demand_bytes_avoided_ready', self.size, window)
                layer['ready_staging_hits'] += 1
            else:
                self.add(ev.phase, 'unready_misses', window=window)
                layer['unready_misses'] += 1
                demand_bytes += job.remaining if job else self.size
                if job:
                    self.add(job.phase, 'prefetch_late', window=job.window)
                    self.add(ev.phase, 'demand_bytes_avoided_partial', job.read_bytes, window)
            if job:
                self.consume(job, ev.ns)
        self.add(ev.phase, 'demand_read_bytes', demand_bytes, window)
        self.debt += demand_bytes
        self.emit('visit', phase=ev.phase, layer=ev.layer, ns=ev.ns,
                  persistent_misses=len(missing), demand_bytes=demand_bytes,
                  resident=int(self.resident.sum()), staged=len(self.jobs), held=len(self.held))
        if self.enabled:
            self.forecast(ev)
        occupied = (len(self.jobs)+len(self.held))*self.chunks
        self.peak_staging = max(self.peak_staging, occupied)
        if occupied > self.args.staging_buffers-self.args.demand_reserve_buffers:
            raise AssertionError('staging reserve consumed')

    def forecast(self, ev):
        for e, job in list(self.jobs.items()):
            if job.phase != ev.phase:
                self.discard(e, 'prefetch_phase_cancelled')
        limit = min(self.args.max_prefetch_experts,
                    (self.args.staging_buffers-self.args.demand_reserve_buffers)//self.chunks)
        while len(self.jobs)+len(self.held) < limit:
            candidates = ~self.resident & ~self.protected & (self.history.probability > 0)
            for e in self.jobs:
                candidates[e] = False
            io_ms = self.size/(self.args.bandwidth_gbs*1e6)
            predicted_ms = io_ms/max(.05, self.spare_fraction)+self.args.ready_delay_ms
            benefit = self.history.probability*(self.args.prefill_miss_ms if ev.phase else self.args.decode_miss_ms)
            value = (benefit-self.args.io_price*io_ms/max(.05, self.spare_fraction)
                     -self.args.staging_rent*self.history.tau)
            candidates &= self.history.tau > predicted_ms
            if not candidates.any():
                break
            e = int(np.argmax(np.where(candidates, value, -math.inf)))
            if value[e] <= 0:
                break
            expires = ev.ns+int((self.history.tau[e]+self.history.cycle)*1e6)
            self.jobs[e] = sim.Forecast(e, self.size, ev.ns, expires, ev.phase, window=self.window_at(ev.ns))
            self.add(ev.phase, 'prefetch_issued', window=self.window_at(ev.ns))
            self.emit('prefetch_issue', expert=e, phase=ev.phase, ns=ev.ns,
                      probability=float(self.history.probability[e]),
                      tau_ms=float(self.history.tau[e]), value=float(value[e]))


def main():
    p = sim.parser()
    p.description = __doc__
    p.add_argument('--residency', type=Path, required=True)
    p.add_argument('--consumer-hold-ms', type=float, default=1.)
    args = p.parse_args()
    if not math.isfinite(args.consumer_hold_ms) or args.consumer_hold_ms < 0:
        p.error('consumer hold must be finite and nonnegative')
    for field in ('bandwidth_gbs', 'history_half_life', 'prior_visits', 'decode_layer_ms', 'prefill_layer_ms'):
        if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            p.error(f'{field} must be finite and positive')
    for field in ('ready_delay_ms', 'decode_miss_ms', 'prefill_miss_ms', 'deadline_weight',
                  'memory_rent', 'min_layer_sample_ms'):
        if not math.isfinite(getattr(args, field)) or getattr(args, field) < 0:
            p.error(f'{field} must be finite and nonnegative')
    if (args.staging_buffers < 1 or not 0 <= args.demand_reserve_buffers <= args.staging_buffers
            or args.shape_min_visits < 1 or not math.isfinite(args.max_layer_sample_ms)
            or args.max_layer_sample_ms < args.min_layer_sample_ms):
        p.error('invalid buffer, history or sampling limits')
    slots, meta, events = sim.read_demand(args.demand)
    actual, credits = sim.match_reads(args.reads, meta, events)
    deltas = read_residency(args.residency, slots, meta, events)
    quiet = sim.quiet_windows(args.disk_samples)
    windows = json.loads(args.windows.read_text()) if args.windows else []
    if any(w['start_ns'] >= w['end_ns'] for w in windows) or any(
            a['end_ns'] > b['start_ns'] for a, b in zip(windows, windows[1:])):
        p.error('windows must be ordered, nonoverlapping and nonempty')
    epoch = events[0].ns
    for ev in events:
        ev.ns -= epoch
    credits = [(ns-epoch, size) for ns, size in credits]
    quiet = [(max(0, a-epoch), min(b-epoch, events[-1].ns)) for a, b in quiet
             if b > epoch and a-epoch < events[-1].ns]
    windows = [dict(w, start_ns=w['start_ns']-epoch, end_ns=w['end_ns']-epoch) for w in windows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.decisions.open('w') if args.decisions else None
    try:
        arms = []
        # Explicit, predeclared probes rather than choosing a winner after
        # seeing evaluation traffic: priced one-slot and zero-price 1/4 slots.
        for name, enabled, buffers, io_price, staging_rent in [
                ('deployed-no-prefetch', False, 0, 0., 0.),
                ('staging-priced-1', True, 1, .1, .001),
                ('staging-unpriced-1', True, 1, 0., 0.),
                ('staging-unpriced-4', True, 4, 0., 0.)]:
            config = copy.copy(args)
            config.max_prefetch_experts = buffers
            config.io_price = io_price; config.staging_rent = staging_rent
            arms.append(StagingPolicy(name, meta, slots, config, quiet, log, credits, windows, enabled))
        for ev, observed, changes in zip(events, actual, deltas):
            for arm in arms:
                arm.step_recorded(ev, observed, changes)
            if any(not np.array_equal(arm.resident, arms[0].resident) for arm in arms[1:]):
                raise AssertionError('prefetch changed deployed retention')
        reports = {arm.name: arm.report() for arm in arms}
        result = dict(format='dgpp-staging-whatif-v1', trace_epoch_unix_ns=epoch,
                      events=len(events), policies=reports,
                      consumer_hold_ms=args.consumer_hold_ms,
                      invariant='Recorded persistent residency is identical in every arm at every visit.',
                      assumptions='Fixed host timeline; sampled quiet fluid pipe; fixed copy/readiness/consumer delays. No speedup prediction.',
                      code_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (Path(__file__), Path(sim.__file__))},
                      input_sha256={name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                                    for name in ('demand', 'residency', 'reads', 'disk_samples')})
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({name: value['phases'] for name, value in reports.items()}, indent=2))
    finally:
        if log:
            log.close()


if __name__ == '__main__':
    main()
