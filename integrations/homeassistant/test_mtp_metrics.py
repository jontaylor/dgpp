"""Run with the HA bridge's Python environment (prometheus-client required)."""
import json
import unittest
from spark_dgpp_ha import CompatibleInferenceSensors, DgppSensors


class SpecMetricsTest(unittest.TestCase):
    def setUp(self):
        self.sensor = CompatibleInferenceSensors.__new__(CompatibleInferenceSensors)
        self.sensor.dgpp = DgppSensors('spark_1', 'spark-1', 'spark_1', 5)
        names = ['num drafts total', 'num draft tokens total', 'num accepted tokens total']
        names += [f'num accepted tokens per pos total (position={p})' for p in range(4)]
        self.sensor.legacy = {f'ha/sensor/key{i}/config': {'name': 'spec decode ' + n}
                              for i, n in enumerate(names)}
        self.data = {'scheduler': {'pool_blocks_total': 1}, 'service': {}, 'prefix_cache': {}}

    def values(self, spec=None):
        if spec is not None:
            self.data['scheduler']['spec_decode'] = spec
        return self.sensor.compatible_values(json.dumps(self.data), 1)

    def test_variable_depth_and_existing_keys(self):
        values = self.values({'num_drafts_total': 10, 'num_draft_tokens_total': 24,
                              'num_accepted_tokens_total': 13,
                              'num_accepted_tokens_per_pos_total': [7, 4, 2]})
        self.assertEqual([values[f'key{i}'] for i in range(7)], [10, 24, 13, 7, 4, 2, None])
        self.assertEqual(values['key5'] / values['key0'], 0.2)

    def test_missing_older_server(self):
        values = self.values()
        self.assertTrue(all(values[f'key{i}'] is None for i in range(7)))

    def test_zero_and_reset(self):
        self.test_variable_depth_and_existing_keys()
        values = self.values({'num_drafts_total': 0, 'num_draft_tokens_total': 0,
                              'num_accepted_tokens_total': 0,
                              'num_accepted_tokens_per_pos_total': [0, 0, 0]})
        self.assertEqual([values[f'key{i}'] for i in range(6)], [0] * 6)

    def test_invalid_counts(self):
        values = self.values({'num_drafts_total': True, 'num_draft_tokens_total': -1,
                              'num_accepted_tokens_total': '13',
                              'num_accepted_tokens_per_pos_total': [None, -1, False]})
        self.assertTrue(all(values[f'key{i}'] is None for i in range(7)))


if __name__ == '__main__':
    unittest.main()
