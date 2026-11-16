#!/usr/bin/env python3
"""Selection checks: phase regressions, I/O budgets and no forced winner."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import expert_cache_fit as fit
import expert_cache_whatif as sim
import numpy as np


def metric(prefill, decode, total=100):
    return {'prefill': dict(misses=prefill, total_bytes=total),
            'decode': dict(misses=decode, total_bytes=total)}


def candidate(name, m, policy='deadline'):
    return dict(id=name, metrics=m, policy=policy)


class SelectionTests(unittest.TestCase):
    def test_improvement_cannot_hide_other_phase_regression(self):
        base = metric(100, 100)
        rows = [candidate('bad', metric(101, 1)), candidate('good', metric(99, 99))]
        self.assertEqual(fit.choose(rows, base)['id'], 'good')

    def test_io_budget_and_zero_miss_baseline(self):
        base = metric(0, 100)
        rows = [candidate('new-miss', metric(1, 1)),
                candidate('too-much-io', metric(0, 1, 111)),
                candidate('valid', metric(0, 90, 110))]
        self.assertEqual(fit.choose(rows, base, .1)['id'], 'valid')

    def test_no_candidate_is_better_than_accepting_regression(self):
        self.assertIsNone(fit.choose([candidate('bad', metric(101, 99))], metric(100, 100)))

    def test_exact_tie_prefers_no_speculation(self):
        rows = [candidate('a-prefetch', metric(90, 90), 'deadline-prefetch'),
                candidate('z-retention', metric(90, 90))]
        self.assertEqual(fit.choose(rows, metric(100, 100))['id'], 'z-retention')

    def test_pareto_front_keeps_phase_tradeoffs(self):
        a, b, c = metric(80, 100), metric(100, 80), metric(110, 110)
        self.assertFalse(fit.dominates(a, b))
        self.assertFalse(fit.dominates(b, a))
        self.assertTrue(fit.dominates(a, c))

    def test_zero_deadline_weight_removes_only_position_bias(self):
        a = sim.parser().parse_args(['--demand', 'd', '--reads', 'r', '--disk-samples', 's', '--output', 'o'])
        a.deadline_weight = 0; a.memory_rent = 0
        meta = np.array([[0, 0, 0, 1, 0, 100], [0, 1, 0, 1, 0, 100]])
        h = sim.History(meta, a)
        h.observe(sim.Event(0, 0, 0, 0, 1, np.array([0])))
        h.observe(sim.Event(1, 1000000, 0, 1, 1, np.array([1])))
        self.assertEqual(h.probability[0], h.probability[1])
        self.assertAlmostEqual(h.scores[0], h.scores[1])
        expected = h.probability[0]*a.decode_miss_ms*sum(np.exp(-k) for k in range(3))
        self.assertAlmostEqual(h.scores[0], expected)

    def test_future_suffix_cannot_change_training_metrics(self):
        a = sim.parser().parse_args(['--demand', 'd', '--reads', 'r', '--disk-samples', 's', '--output', 'o'])
        m = np.array([[0, i % 2, i//2, 1, 0, 100] for i in range(4)])
        prefix = [sim.Event(i, i*1000000, 0, i % 2, 1, np.array([i])) for i in range(2)]
        task = dict(id='test', policy='deadline', weights={}, training_only=True)
        reports = []
        for expert in (0, 2):
            events = prefix+[sim.Event(2, 2000000, 1, 0, 1, np.array([expert]))]
            actual = [{0: 100}, {1: 100}, {expert: 100}]
            fit.DATA = 2, m, events, actual, [], [], [], a, 2, 1500000
            reports.append(fit.run(task)['metrics'])
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(reports[0]['prefill']['demands'], 0)

    def test_evaluation_phase_deltas_reconcile_with_full_replay(self):
        a = sim.parser().parse_args(['--demand', 'd', '--reads', 'r', '--disk-samples', 's', '--output', 'o'])
        m = np.array([[0, i % 2, i//2, 1, 0, 100] for i in range(4)])
        events = [sim.Event(i, i*1000000, int(i >= 2), i % 2, 1, np.array([i])) for i in range(4)]
        actual = [{i: 100} for i in range(4)]
        fit.DATA = 2, m, events, actual, [], [], [], a, 2, 1500000
        task = dict(id='test', policy='lru', weights={}, training_only=True)
        training = fit.run(task)
        task['training_only'] = False
        full = fit.run(task)
        for phase in ('prefill', 'decode'):
            for key in ('demands', 'misses', 'total_bytes', 'observed_bytes'):
                self.assertEqual(full['metrics'][phase][key],
                                 training['metrics'][phase][key]+full['evaluation'][phase][key])
        phase = full['evaluation_report']['phases']['prefill']
        self.assertEqual(phase['net_read_byte_delta_vs_observed'],
                         phase['total_read_bytes']-phase['observed_read_bytes'])


if __name__ == '__main__':
    unittest.main()
