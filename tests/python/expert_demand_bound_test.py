#!/usr/bin/env python3
"""Check the batched-demand lower bound against exact small-state search."""
import itertools
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from expert_demand_bound import MAGIC, bound, load


class DemandBoundTest(unittest.TestCase):
    def test_trace_records_initial_state_phase_and_repeated_routes(self):
        words = [MAGIC, 1, 1, 2,
                 0, 0, 0, 1, 1, 64, 0, 0, 1, 1, 0, 64,
                 1, 3, 1234, 0, 1, 1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.bin'
            path.write_bytes(struct.pack('<' + 'Q' * len(words), *words))
            slots, experts, events = load(path)
            self.assertEqual(slots, 1)
            self.assertEqual([e[4] for e in experts], [1, 0])
            self.assertEqual(events, [(1, 1234, {0, 1})])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                load(path)

    def test_free_initial_cache_is_optimistic_for_a_window(self):
        experts = [[0, 0, 0, 1, 1, 64], [0, 0, 1, 1, 0, 64]]
        events = [(False, 1, {1}), (False, 2, {1})]
        self.assertEqual(sum(r['lower_bound_bytes'] for r in bound(1, experts, events)), 64)
        self.assertEqual(sum(r['lower_bound_bytes'] for r in bound(1, experts, events, True)), 0)

    def test_against_exhaustive_batched_admission(self):
        rng = random.Random(123)
        for _ in range(500):
            n, capacity = 5, 2
            events = [(0, i, {e for e in range(n) if rng.randrange(3) == 0})
                      for i in range(5)]
            initial = set(rng.sample(range(n), capacity))
            experts = [[0, 0, e, 1, int(e in initial), 1] for e in range(n)]
            states = {frozenset(initial): 0}
            for _, _, ids in events:
                new = {}
                for resident, cost in states.items():
                    pool = resident | ids
                    for k in range(min(capacity, len(pool)) + 1):
                        for candidate in itertools.combinations(pool, k):
                            candidate = frozenset(candidate)
                            value = cost + len(ids - resident)
                            new[candidate] = min(new.get(candidate, value), value)
                states = new
            self.assertEqual(sum(row['misses'] for row in bound(capacity, experts, events)),
                             min(states.values()))


if __name__ == '__main__':
    unittest.main()
