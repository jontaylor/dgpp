#!/usr/bin/env python3
"""Frozen residency, staging truth table and source-lifetime invariants."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import expert_cache_whatif as sim
import expert_staging_whatif as staging


def policy(enabled=False):
    a = sim.parser().parse_args(['--demand', 'd', '--reads', 'r', '--disk-samples', 's', '--output', 'o'])
    a.consumer_hold_ms = 0; a.ready_delay_ms = 0; a.bandwidth_gbs = 1
    a.staging_buffers = 8; a.demand_reserve_buffers = 2; a.max_prefetch_experts = 1
    m = np.array([[0, i % 2, i//2, 1, 0, 100] for i in range(4)])
    return staging.StagingPolicy('test', m, 2, a, [], enabled=enabled)


EMPTY = np.empty((0, 2), dtype=np.int64)


def event(seq=0, ns=0):
    return sim.Event(seq, ns, 0, 0, 1, np.array([0]))


class StagingTests(unittest.TestCase):
    def test_disabled_reproduces_baseline_and_only_deltas_admit(self):
        p = policy()
        p.step_recorded(event(), {0: 100}, EMPTY)
        self.assertEqual(p.stats[0]['demand_read_bytes'], 100)
        self.assertFalse(p.resident.any())
        p.step_recorded(event(1, 1), {}, np.array([[0, 1]]))
        self.assertTrue(p.resident[0])
        self.assertEqual(p.stats[0]['persistent_hits'], 1)

    def test_ready_staging_hit_never_becomes_persistent_residency(self):
        p = policy()
        p.jobs[0] = sim.Forecast(0, 0, 0, 1000, 0, ready_at=0, read_bytes=100)
        p.step_recorded(event(), {0: 100}, EMPTY)
        self.assertEqual(p.stats[0]['ready_staging_hits'], 1)
        self.assertEqual(p.stats[0]['demand_read_bytes'], 0)
        self.assertFalse(p.resident.any())
        p.step_recorded(event(1, 1), {0: 100}, EMPTY)
        self.assertEqual(p.stats[0]['unready_misses'], 1)
        self.assertEqual(p.stats[0]['demand_read_bytes'], 100)

    def test_partial_late_forecast_charges_only_remaining_bytes(self):
        p = policy()
        p.jobs[0] = sim.Forecast(0, 40, 0, 1000, 0, read_bytes=60)
        p.step_recorded(event(), {0: 100}, EMPTY)
        self.assertEqual(p.stats[0]['demand_read_bytes'], 40)
        self.assertEqual(p.stats[0]['prefetch_late'], 1)
        self.assertEqual(p.stats[0]['demand_bytes_avoided_partial'], 60)
        self.assertFalse(p.resident.any())

    def test_file_complete_is_not_ready_before_copy_finishes(self):
        p = policy()
        p.jobs[0] = sim.Forecast(0, 0, 0, 1000, 0, ready_at=100, read_bytes=100)
        p.step_recorded(event(), {0: 100}, EMPTY)
        self.assertEqual(p.stats[0]['unready_misses'], 1)
        self.assertEqual(p.stats[0]['demand_read_bytes'], 0)
        self.assertEqual(p.held, [100])

    def test_resident_hit_makes_forecast_redundant_not_useful(self):
        p = policy()
        p.jobs[0] = sim.Forecast(0, 0, 0, 1000, 0, ready_at=0, read_bytes=100)
        p.step_recorded(event(), {}, np.array([[0, 1]]))
        self.assertEqual(p.stats[0]['prefetch_became_resident'], 1)
        self.assertEqual(p.stats[0]['prefetch_discarded_bytes'], 100)
        self.assertEqual(p.stats[0]['ready_staging_hits'], 0)

    def test_cancellation_does_not_recycle_active_copy_source(self):
        p = policy()
        p.jobs[0] = sim.Forecast(0, 0, 0, 1000, 0, ready_at=100, read_bytes=100)
        p.step_recorded(event(), {}, np.array([[0, 1]]))
        self.assertEqual(p.held, [100])
        p.advance(99)
        self.assertEqual(len(p.held), 1)
        p.advance(100)
        self.assertFalse(p.held)

    def test_consumer_hold_blocks_another_forecast(self):
        p = policy(True)
        p.held = [1000]
        p.history.observe(event())
        p.history.probability[:] = 1
        p.history.tau[:] = 100
        p.forecast(event())
        self.assertFalse(p.jobs)

    def test_baseline_mismatch_is_rejected(self):
        p = policy()
        with self.assertRaisesRegex(ValueError, 'disagree'):
            p.step_recorded(event(), {}, EMPTY)

    def test_sequence_and_timestamp_pair_residency_to_demand(self):
        p = policy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'r.bin'
            words = [int.from_bytes(b'DGRSIDE1', 'little'), 1, 4, 2, 0, 0, 0, 0,
                     0, 0, 0, 1, 10, 1, 0, 1]
            path.write_bytes(struct.pack('<'+'Q'*len(words), *words))
            result = staging.read_residency(path, 2, p.meta, [event(), event(1, 10)])
            self.assertEqual(result[1].tolist(), [[0, 1]])
            with self.assertRaisesRegex(ValueError, 'event mismatch'):
                staging.read_residency(path, 2, p.meta, [event(), event(1, 11)])


if __name__ == '__main__':
    unittest.main()
