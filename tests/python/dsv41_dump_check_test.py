"""Offline checks for the TP numerical gate; no CUDA or checkpoint required."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('dump_check', Path(__file__).resolve().parents[2] / 'scripts/dsv41_dump_check.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class DumpCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ref, self.candidate = [Path(self.tmp.name) / name for name in ('reference', 'candidate')]
        self.cfg = dict(num_hidden_layers=1, hc_mult=4, hidden_size=2, num_experts_per_tok=1,
                        index_source_layer_ids=[0], index_topk=2, vocab_size=8, n_routed_experts=2)
        self.state = (b'DSV41ST3' + struct.pack('<iii', 1, 2, 8) + struct.pack('<16H', *([0x3f80] * 16)) +
                      struct.pack('<8f', *([0.25] * 8)) + struct.pack('<2q', 1, 3) +
                      struct.pack('<i2i2i4i', 1, 0, 1, 1, 2, -1, -1, 0, 1))
        for directory in (self.ref, self.candidate):
            directory.mkdir()
            for rank in range(2):
                (directory / f'rank{rank}.states').write_bytes(self.state)
                (directory / f'rank{rank}.logits').write_bytes(b'DSV41LG1' + struct.pack('<iii8f', 2, rank * 4, 4, *range(8)))

    def run_check(self):
        return check.compare(self.ref, self.candidate, self.cfg, 2, 2)

    def test_complete_exact_shards_pass(self):
        self.assertEqual(self.run_check()['status'], 'byte_exact')

    def test_changed_logit_is_refused(self):
        path = self.candidate / 'rank1.logits'
        data = bytearray(path.read_bytes());struct.pack_into('<f', data, 20, 0.125);path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'logits differ'):
            self.run_check()

    def test_matching_nonfinite_logits_are_refused(self):
        for directory in (self.ref, self.candidate):
            path = directory / 'rank0.logits'
            data = bytearray(path.read_bytes());struct.pack_into('<f', data, 20, float('nan'));path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'nonfinite logits'):
            self.run_check()

    def test_matching_nonfinite_states_are_refused(self):
        for directory in (self.ref, self.candidate):
            path = directory / 'rank0.states'
            data = bytearray(path.read_bytes());struct.pack_into('<H', data, 20, 0x7f80);path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'nonfinite BF16'):
            self.run_check()

    def test_overlapping_vocab_shards_are_refused(self):
        path = self.candidate / 'rank1.logits'
        data = bytearray(path.read_bytes());struct.pack_into('<i', data, 12, 0);path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'vocabulary slice'):
            self.run_check()

    def test_truncated_state_is_refused(self):
        (self.candidate / 'rank1.states').write_bytes(self.state[:-1])
        with self.assertRaisesRegex(ValueError, 'incomplete or excess state'):
            self.run_check()

    def test_reference_rank_disagreement_is_refused(self):
        for directory in (self.ref, self.candidate):
            data = bytearray(self.state);struct.pack_into('<H', data, 20, 0x4000)
            (directory / 'rank1.states').write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'replicated captures'):
            self.run_check()


if __name__ == '__main__':
    unittest.main()
