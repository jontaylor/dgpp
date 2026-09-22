#!/usr/bin/env python3
"""Bounded independent samples of ordinary MiMo layer/global weights.

Standard-library only; uses byte reads, never executes checkpoint model code.
Samples straddle FP8 scale boundaries and cover each TP4 partition endpoint.
"""
import json
from pathlib import Path
import struct
import sys

from mimo_reference_sample import Reader, bf16, fp8


def sample(root):
    reader = Reader(root)
    tensors = [
        ('model.embed_tokens.weight', 152576, 4096, 'BF16', 'replicated'),
        ('lm_head.weight', 152576, 4096, 'BF16', 'row'),
        ('model.norm.weight', 4096, 1, 'BF16', 'replicated'),
    ]
    for layer in (0, 1):
        p = f'model.layers.{layer}.'
        tensors += [(p + n + '.weight', 4096, 1, 'BF16', 'replicated')
                    for n in ('input_layernorm', 'post_attention_layernorm')]
        tensors.append((p + 'self_attn.o_proj.weight', 4096, 8192, 'BF16', 'column'))
    tensors += [
        ('model.layers.1.self_attn.attention_sink_bias', 64, 1, 'BF16', 'row'),
        ('model.layers.1.mlp.gate.weight', 256, 4096, 'BF16', 'replicated'),
        ('model.layers.1.mlp.gate.e_score_correction_bias', 256, 1, 'F32', 'replicated'),
    ]
    for proj in ('gate', 'up', 'down'):
        down = proj == 'down'
        tensors.append((f'model.layers.0.mlp.{proj}_proj.weight',
                        4096 if down else 16384, 16384 if down else 4096,
                        'F8_E4M3', 'column' if down else 'row'))
    result = {'revision': Path(root).name, 'samples': []}
    for name, rows, cols, dtype, axis in tensors:
        coordinates = {(0, 0), (rows - 1, cols - 1)}
        for part in range(4):
            if axis == 'row':
                coordinates |= {(part * (rows // 4), min(120, cols - 1)),
                                ((part + 1) * (rows // 4) - 1, min(120, cols - 1))}
            elif axis == 'column':
                coordinates |= {(127, part * (cols // 4) + 120),
                                (128, (part + 1) * (cols // 4) - 16)}
            else:
                coordinates.add((part * (rows // 4), min(120, cols - 1)))
        for row, col in sorted(coordinates):
            count = min(16, cols - col)
            # Windows must stay inside each possible TP partition.
            if axis == 'column':
                count = min(count, cols // 4 - col % (cols // 4))
            size = {'BF16': 2, 'F32': 4, 'F8_E4M3': 1}[dtype]
            data = reader.read(name, (row * cols + col) * size, count * size)
            if dtype == 'BF16':
                values = list(struct.unpack('<' + 'H' * count, data))
            elif dtype == 'F32':
                values = list(struct.unpack('<' + 'f' * count, data))
            else:
                values = []
                for j, code in enumerate(data):
                    scale_index = (row // 128) * ((cols + 127) // 128) + (col + j) // 128
                    scale = struct.unpack('<f', reader.read(name[:-7] + '.weight_scale_inv',
                                                           scale_index * 4, 4))[0]
                    values.append(bf16(fp8(code) * scale))
            result['samples'].append(dict(name=name, dtype=dtype, axis=axis,
                                          row=row, col=col, values=values))
    return result


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: mimo_layer_weight_sample.py SNAPSHOT_DIR')
    print(json.dumps(sample(sys.argv[1]), indent=2, allow_nan=False))
