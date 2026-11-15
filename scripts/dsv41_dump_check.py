#!/usr/bin/env python3
"""Compare complete per-rank DSV41ST3/DSV41LG1 dumps against a TP control.

Directories contain rank<N>.states and rank<N>.logits. The reference must come
from a separate whole-layer streaming run at the same world size and row shape.
This checks numerical artifacts, not process identity, swap or run isolation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_states(path, cfg, tokens):
    layers, width = cfg['num_hidden_layers'], cfg['hc_mult'] * cfg['hidden_size']
    topk, sources, selections = cfg['num_experts_per_tok'], len(cfg['index_source_layer_ids']), cfg['index_topk']
    state_bytes = layers * tokens * width * 2
    pre_bytes = layers * tokens * 4 * 4
    route_bytes = layers * tokens * topk * 4
    size = 20 + state_bytes + pre_bytes + tokens * 8 + 4 + route_bytes + 8 + sources * tokens * selections * 4
    require(path.stat().st_size == size, f'{path}: incomplete or excess state bytes')
    data = path.read_bytes()
    require(data[:8] == b'DSV41ST3', f'{path}: wrong state magic')
    require(struct.unpack_from('<iii', data, 8) == (layers, tokens, width), f'{path}: wrong state shape')
    offset = 20
    state = memoryview(data)[offset:offset + state_bytes]
    require(all((x[0] & 0x7f80) != 0x7f80 for x in struct.iter_unpack('<H', state)), f'{path}: nonfinite BF16 state')
    offset += state_bytes
    pre = memoryview(data)[offset:offset + pre_bytes]
    require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', pre)), f'{path}: nonfinite collapse coefficients')
    offset += pre_bytes
    ids = struct.unpack_from('<' + 'q' * tokens, data, offset)
    require(all(0 <= x < cfg['vocab_size'] for x in ids), f'{path}: invalid token IDs')
    offset += tokens * 8
    require(struct.unpack_from('<i', data, offset)[0] == topk, f'{path}: wrong route width')
    offset += 4
    routes = memoryview(data)[offset:offset + route_bytes]
    require(all(0 <= x[0] < cfg['n_routed_experts'] for x in struct.iter_unpack('<i', routes)), f'{path}: invalid expert IDs')
    offset += route_bytes
    require(struct.unpack_from('<ii', data, offset) == (sources, selections), f'{path}: wrong index capture shape')
    return data, ids


def read_logits(path, cfg, tokens, rank, world):
    begin, end = cfg['vocab_size'] * rank // world, cfg['vocab_size'] * (rank + 1) // world
    require(path.stat().st_size == 20 + tokens * (end - begin) * 4, f'{path}: incomplete or excess logits')
    data = path.read_bytes()
    require(data[:8] == b'DSV41LG1', f'{path}: wrong logits magic')
    require(struct.unpack_from('<iii', data, 8) == (tokens, begin, end - begin), f'{path}: wrong vocabulary slice or rows')
    require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', memoryview(data)[20:])), f'{path}: nonfinite logits')
    return data


def compare(reference, candidate, cfg, tokens, world):
    require(world in (1, 2, 4) and tokens > 0, 'expected world 1/2/4 and positive token count')
    ranks, shared_states, shared_ids = [], None, None
    for rank in range(world):
        ref_state, ids = read_states(Path(reference) / f'rank{rank}.states', cfg, tokens)
        cand_state, cand_ids = read_states(Path(candidate) / f'rank{rank}.states', cfg, tokens)
        ref_logits = read_logits(Path(reference) / f'rank{rank}.logits', cfg, tokens, rank, world)
        cand_logits = read_logits(Path(candidate) / f'rank{rank}.logits', cfg, tokens, rank, world)
        require(ref_state == cand_state, f'rank {rank}: layer states, routes or selections differ from streaming control')
        require(ref_logits == cand_logits, f'rank {rank}: complete logits differ from streaming control')
        if shared_states is not None:
            require(ref_state == shared_states and ids == shared_ids, f'rank {rank}: replicated captures differ across ranks')
        shared_states, shared_ids = ref_state, ids
        ranks.append({'rank': rank, 'states_sha256': hashlib.sha256(cand_state).hexdigest(),
                      'logits_sha256': hashlib.sha256(cand_logits).hexdigest()})
    return {'status': 'byte_exact', 'world': world, 'tokens': tokens, 'vocab_size': cfg['vocab_size'], 'ranks': ranks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--tokens', required=True, type=int)
    parser.add_argument('--world', required=True, type=int)
    args = parser.parse_args()
    root = json.loads(Path(args.config).read_text())
    cfg = root.get('text_config', root)
    print(json.dumps(compare(args.reference, args.candidate, cfg, args.tokens, args.world), indent=2))


if __name__ == '__main__':
    main()
