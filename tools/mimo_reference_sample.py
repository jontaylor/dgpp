#!/usr/bin/env python3
"""Read bounded MiMo checkpoint samples and emit independent BF16 goldens.

Uses only the standard library. Reads config/index/header metadata and small
payload ranges; does not import checkpoint code, allocate a GPU, or write to
the checkpoint. Run with SNAPSHOT_DIR and redirect stdout to the fixture.
The QKV layout is corroborated by vLLM _shard_fp8_qkv_proj at
382970ee6ca490aeaaaf4e32c53695b581ff61ba.
"""

import json
import math
from pathlib import Path
import struct
import sys


def bf16(value):
    # Force an IEEE FP32 product, then round-to-nearest-even to BF16.
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    if bits & 0x7fffffff > 0x7f800000:
        return (bits >> 16 & 0x8000) | 0x7fc0
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def fp8(code):
    sign = -1 if code & 128 else 1
    exp, mantissa = (code >> 3) & 15, code & 7
    if exp == 15 and mantissa == 7:
        return float('nan')
    return sign * (math.ldexp(mantissa, -9) if exp == 0 else
                   math.ldexp(1 + mantissa / 8, exp - 7))


class Reader:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / 'model.safetensors.index.json').read_text())
        if self.index['metadata']['tp_size'] != 4:
            raise ValueError('expected checkpoint TP4')
        self.headers = {}

    def read(self, name, byte_offset, count):
        shard = self.index['weight_map'][name]
        path = self.root / shard
        if shard not in self.headers:
            with path.open('rb') as f:
                length = struct.unpack('<Q', f.read(8))[0]
                if length > 16_000_000:
                    raise ValueError('oversized header')
                self.headers[shard] = (8 + length, json.loads(f.read(length)))
        start, header = self.headers[shard]
        desc = header[name]
        begin, end = desc['data_offsets']
        if byte_offset < 0 or byte_offset + count > end - begin:
            raise ValueError('sample outside tensor')
        with path.open('rb') as f:
            f.seek(start + begin + byte_offset)
            data = f.read(count)
        if len(data) != count:
            raise ValueError('incomplete sample')
        return data


def sample(root):
    reader = Reader(root)
    result = {'revision': Path(root).name, 'qkv': [], 'experts': []}
    # Independent head-wise traversal, in canonical Q/K/V order; stored
    # rows and scale indices are derived from each head's checkpoint chunk.
    for layer, kv_heads in [(0, 4), (1, 8)]:
        base = f'model.layers.{layer}.self_attn.qkv_proj'
        chunk_width = 16 * 192 + (kv_heads // 4) * (192 + 128)
        scale_stride = math.ceil(chunk_width / 128)
        part_offset = 0
        canonical_offset = 0
        for kind, heads, dim in [('q', 64, 192), ('k', kv_heads, 192), ('v', kv_heads, 128)]:
            per_chunk = heads // 4
            for head in sorted({0, per_chunk - 1, per_chunk, heads - 1}):
                for element in [0, dim - 1]:
                    chunk, local_head = divmod(head, per_chunk)
                    within = part_offset + local_head * dim + element
                    row = chunk * chunk_width + within
                    scale_row = chunk * scale_stride + within // 128
                    col_begin, cols = 120, 16  # straddles a scale-column boundary
                    payload = reader.read(base + '.weight', row * 4096 + col_begin, cols)
                    scales = struct.unpack('<2f', reader.read(base + '.weight_scale_inv', scale_row * 32 * 4, 8))
                    result['qkv'].append({
                        'layer': layer, 'kind': kind, 'head': head, 'element': element,
                        'canonical_row': canonical_offset + head * dim + element,
                        'source_row': row, 'scale_row': scale_row,
                        'col_begin': col_begin, 'payload': list(payload), 'scales': list(scales),
                        'bf16': [bf16(fp8(v) * scales[(col_begin + j) // 128]) for j, v in enumerate(payload)],
                    })
            part_offset += per_chunk * dim
            canonical_offset += heads * dim
    levels = [0.0, .5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    for projection, total_rows, total_cols in [('gate_proj', 2048, 4096), ('down_proj', 4096, 2048)]:
        base = f'model.layers.1.mlp.experts.0.{projection}'
        for row in [0, total_rows - 1]:
            col_begin, cols = 32, 64
            payload = reader.read(base + '.weight', row * (total_cols // 2) + col_begin // 2, cols // 2)
            scales = reader.read(base + '.weight_scale', row * (total_cols // 32) + col_begin // 32, cols // 32)
            values = []
            for j in range(cols):
                code = (payload[j // 2] >> (4 * (j % 2))) & 15
                scale = scales[j // 32]
                value = (-1 if code & 8 else 1) * levels[code & 7]
                values.append(bf16(value * (math.ldexp(1, scale - 127) if scale != 255 else float('nan'))))
            result['experts'].append({'projection': projection, 'rows': total_rows, 'cols': total_cols,
                                     'row': row, 'col_begin': col_begin, 'payload': list(payload),
                                     'scales': list(scales), 'bf16': values})
    return result


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: mimo_reference_sample.py SNAPSHOT_DIR')
    print(json.dumps(sample(sys.argv[1]), indent=2, allow_nan=False))
