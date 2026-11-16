#!/usr/bin/env python3
"""Resource and causality checks for the recorded-traffic cache replay."""
from pathlib import Path
import sys
import tempfile
import struct
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import expert_cache_whatif as sim


def args():
    a = sim.parser().parse_args(['--demand', 'd', '--reads', 'r', '--disk-samples', 's', '--output', 'o'])
    a.ready_delay_ms = 0
    a.bandwidth_gbs = 1
    return a


def meta(count=4, resident=()):
    return np.array([[0, i % 2, i//2, 1, int(i in resident), 100] for i in range(count)], dtype=np.int64)


class ReplayTests(unittest.TestCase):
    def test_shared_pipe_does_not_double_book_gap(self):
        p = sim.Policy('deadline-prefetch', meta(), 4, args(), [(0, 150)])
        p.jobs = {i: sim.Forecast(i, 100, 0, 10000, 0) for i in (0, 1)}
        p.advance(0); p.advance(150)
        self.assertEqual(p.stats[0]['speculative_read_bytes'], 150)
        self.assertTrue(p.resident[0])
        self.assertEqual(p.jobs[1].remaining, 50)

    def test_extra_demand_gets_spare_capacity_first(self):
        p = sim.Policy('deadline-prefetch', meta(), 4, args(), [(0, 150)])
        p.debt = 75
        p.jobs[0] = sim.Forecast(0, 100, 0, 10000, 0)
        p.advance(0); p.advance(150)
        self.assertEqual(p.debt, 0)
        self.assertEqual(p.jobs[0].remaining, 25)

    def test_copy_readiness_holds_staging_after_file_completion(self):
        a = args(); a.ready_delay_ms = .0001  # 100 ns
        p = sim.Policy('deadline-prefetch', meta(), 4, a, [(0, 1000)])
        p.jobs[0] = sim.Forecast(0, 100, 0, 10000, 0)
        p.advance(0); p.advance(150)
        self.assertFalse(p.resident[0])
        self.assertEqual(p.jobs[0].remaining, 0)
        p.advance(200)
        self.assertTrue(p.resident[0])
        self.assertNotIn(0, p.jobs)

    def test_expiry_charges_partial_bytes_without_admission(self):
        p = sim.Policy('deadline-prefetch', meta(), 4, args(), [(0, 1000)])
        p.jobs[0] = sim.Forecast(0, 100, 0, 50, 0)
        p.advance(0); p.advance(1000)
        self.assertFalse(p.resident[0])
        self.assertEqual(p.stats[0]['speculative_read_bytes'], 50)
        self.assertEqual(p.stats[0]['prefetch_discarded_bytes'], 50)

    def test_completion_before_expiry_does_not_depend_on_observation_cadence(self):
        def make():
            p = sim.Policy('deadline-prefetch', meta(), 4, args(), [(0, 1000)])
            p.jobs[0] = sim.Forecast(0, 100, 0, 150, 0)
            p.advance(0)
            return p
        p, q = make(), make()
        p.advance(200)
        q.advance(100); q.advance(200)
        np.testing.assert_array_equal(p.resident, q.resident)
        self.assertTrue(p.resident[0])

    def test_future_read_labels_cannot_grant_early_service_credit(self):
        def make(credits):
            return sim.Policy('deadline-prefetch', meta(), 4, args(), [(0, 1000)], credits=credits)
        p, q = make([]), make([(500, 100)])
        ev = sim.Event(0, 0, 0, 0, 1, np.array([0]))
        p.step(ev, {})
        q.step(ev, {0: 100})
        for arm in (p, q):
            arm.jobs = {1: sim.Forecast(1, 100, 0, 10000, 0)}
            arm.advance(150)
        self.assertEqual(p.jobs[1].remaining, 50)
        self.assertEqual(q.jobs[1].remaining, 50)
        self.assertEqual(p.stats[0]['speculative_read_bytes'], q.stats[0]['speculative_read_bytes'])

    def test_batch_retains_highest_value_candidate_first(self):
        p = sim.Policy('deadline', meta(), 1, args(), [])
        ev = sim.Event(0, 0, 0, 0, 2, np.array([0, 2]))
        def observe(_):
            p.history.scores[:] = [.18, 0, 1.05, 0]
            p.history.previous = ev
        p.history.observe = observe
        p.step(ev, {0: 100, 2: 100})
        self.assertTrue(p.resident[2])
        self.assertFalse(p.resident[0])

    def test_window_costs_include_shared_speculative_service(self):
        windows = [dict(name='a', start_ns=0, end_ns=75), dict(name='b', start_ns=75, end_ns=150)]
        p = sim.Policy('deadline-prefetch', meta(), 4, args(), [(0, 150)], windows=windows)
        p.jobs[0] = sim.Forecast(0, 100, 0, 10000, 0, window='a')
        p.advance(0); p.advance(150)
        self.assertEqual(p.windows['a', 0]['speculative_read_bytes'], 75)
        self.assertEqual(p.windows['b', 0]['speculative_read_bytes'], 25)

    def test_read_parser_rejects_duplicate_chunk_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'r.bin'
            m = np.array([[0, 0, 0, 2, 0, 100]])
            record = (100, 101, 0, 0, 0, 0, 50, 0)
            path.write_bytes(struct.pack('<16Q', *(record+record)))
            events = [sim.Event(0, 0, 0, 0, 1, np.array([0]))]
            with self.assertRaisesRegex(ValueError, 'duplicate chunk'):
                sim.match_reads(path, m, events)

    def test_residency_margin_prevents_pointless_replacement(self):
        p = sim.Policy('deadline', meta(resident=(0,)), 1, args(), [])
        p.history.previous = sim.Event(0, 0, 0, 0, 1, np.array([0]))
        p.history.scores[:] = [1, 1.04, 0, 0]
        self.assertFalse(p.admit(1))
        p.history.scores[1] = 1.06
        self.assertTrue(p.admit(1))
        self.assertEqual(int(p.resident.sum()), 1)

    def test_batch_larger_than_cache_never_evicts_live_selected_experts(self):
        p = sim.Policy('lru', meta(resident=(0,)), 1, args(), [])
        ev = sim.Event(0, 0, 0, 0, 2, np.array([0, 2]))
        p.step(ev, {2: 100})
        self.assertEqual(int(p.resident.sum()), 1)
        self.assertTrue(p.resident[0])
        self.assertEqual(p.stats[0]['hits'], 1)

    def test_observed_read_labels_cannot_change_retention_decisions(self):
        p = sim.Policy('deadline', meta(), 2, args(), [])
        q = sim.Policy('deadline', meta(), 2, args(), [])
        events = [sim.Event(i, i*1000000, 0, i % 2, 1, np.array([i % 4])) for i in range(12)]
        for ev in events:
            p.step(ev, {})
            q.step(ev, {int(ev.ids[0]): 100})
            np.testing.assert_array_equal(p.resident, q.resident)
            np.testing.assert_array_equal(p.history.scores, q.history.scores)

    def test_forecast_occupancy_and_no_unseen_expert_fallback(self):
        a = args(); a.max_prefetch_experts = 10; a.staging_buffers = 3; a.demand_reserve_buffers = 2
        p = sim.Policy('deadline-prefetch', meta(), 4, a, [])
        ev = sim.Event(0, 0, 0, 0, 1, np.array([0]))
        p.history.observe(ev)
        p.protected[ev.ids] = True
        p.forecast(ev)
        self.assertFalse(p.jobs)  # only seen expert is current demand, not speculation
        p.history.probability[:] = [0, 1, 1, 1]
        p.history.scores[:] = 100
        p.forecast(ev)
        self.assertEqual(len(p.jobs), 1)
        self.assertLessEqual(p.peak_staging, a.staging_buffers-a.demand_reserve_buffers)

    def test_header_seeds_actual_initial_residency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'d.bin'
            m = meta(resident=(1,))
            words = [0x31444e414d454744, 1, 2, 4, *m.reshape(-1), 0, 1, 100, 1]
            path.write_bytes(struct.pack('<'+'Q'*len(words), *words))
            slots, parsed, events = sim.read_demand(path)
            p = sim.Policy('deadline', parsed, slots, args(), [])
            self.assertEqual(np.flatnonzero(p.resident).tolist(), [1])
            self.assertEqual(events[0].layer, 1)


if __name__ == '__main__':
    unittest.main()
