#!/usr/bin/env python3
"""Compare mimo_layer_check's raw BF16 rows to the CPU layer reference.

Tolerance gates residual-output drift, not full-model equivalence. Routing
and full-model logits require their own checks before deployment.
"""
import json
import math
from pathlib import Path
import struct
import sys


def floats(bits):
    return [struct.unpack('<f', struct.pack('<I', int(b) << 16))[0] for b in bits]


def compare(reference, actual):
    if Path(reference).suffix == '.bf16':
        data = Path(reference).read_bytes()
        if len(data) % 8192 or not data:
            raise ValueError('expected complete 4096-wide BF16 reference rows')
        doc = dict(output_bf16=struct.unpack('<' + 'H' * (len(data)//2), data),
                   layer='differential', torch_version=None)
    else:
        doc = json.loads(Path(reference).read_text())
    raw = Path(actual).read_bytes()
    ref = floats(doc['output_bf16'])
    if len(raw) != len(ref) * 2 or not ref or len(ref) % 4096:
        raise ValueError('expected matching complete 4096-wide BF16 rows')
    got = floats(struct.unpack('<' + 'H' * len(ref), raw))
    if not all(math.isfinite(x) for x in ref + got):
        raise ValueError('nonfinite layer output')
    reports = []
    for p in range(len(ref)//4096):
        a, b = got[p*4096:(p+1)*4096], ref[p*4096:(p+1)*4096]
        rel = math.sqrt(sum((x-y)**2 for x,y in zip(a,b)) / max(sum(y*y for y in b), 1e-30))
        max_abs = max(abs(x-y) for x,y in zip(a,b))
        max_scaled = max_abs / max(max(abs(y) for y in b), 1e-30)
        reports.append(dict(position=p, relative_l2=rel, max_abs=max_abs, max_scaled=max_scaled))
        if rel > .01 or max_scaled > .05:
            raise ValueError(f'layer {doc["layer"]} position {p} exceeds tolerance: {reports[-1]}')
    return dict(layer=doc['layer'], torch_version=doc['torch_version'], rows=reports)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('usage: mimo_layer_compare.py REFERENCE_JSON_OR_BF16 GPU_BF16')
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=2))
