#!/usr/bin/env python3
"""Compare real MTP heads after full versus cache-only history updates.

Runs on an idle CUDA node: SNAPSHOT LAYER_PROBE OUTPUT_DIR.
Checks all K/V bytes and the final predicted hidden row after ring wrap, for
both mapped one-row decode and chunked prefill. The probe also checks that
cache-only execution leaves its input residual unchanged.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


def check(snapshot, probe, output):
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for layer in (48, 49, 50):
        for mode, chunk, tokens in (("decode", 1, 145), ("prefill", 8, 273)):
            paths = []
            for cache_only in (False, True):
                path = output / f"head{layer-48}-{mode}-{'cache' if cache_only else 'full'}.bf16"
                env = os.environ.copy()
                env['DGPP_MIMO_LAYER_KV_DUMP'] = '1'
                for key in ('DGPP_MIMO_LAYER_DECODE', 'DGPP_MIMO_LAYER_CACHE_ONLY_PREFIX'):
                    env.pop(key, None)
                if mode == 'decode':
                    env['DGPP_MIMO_LAYER_DECODE'] = '1'
                if cache_only:
                    env['DGPP_MIMO_LAYER_CACHE_ONLY_PREFIX'] = '1'
                result = subprocess.run([str(probe), str(snapshot), str(layer), str(path),
                                         str(chunk), str(tokens)], env=env, check=True,
                                        text=True, capture_output=True)
                path.with_suffix('.log').write_text(result.stdout + result.stderr)
                paths.append(path)
            full, cached = (p.read_bytes() for p in paths)
            assert len(full) == len(cached) == tokens * 4096 * 2, 'wrong output shape'
            assert full[-8192:] == cached[-8192:], f'head {layer-48} {mode}: final prediction differs'
            full_kv, cached_kv = (Path(str(p) + '.kv').read_bytes() for p in paths)
            assert full_kv == cached_kv, f'head {layer-48} {mode}: K/V differs'
            reports.append(dict(head=layer-48, mode=mode, rows=tokens,
                                kv_bytes=len(full_kv), kv_identical=True, final_hidden_identical=True))
            print(f'PASS head {layer-48} {mode}: exact K/V and final hidden equality', flush=True)
    (output / 'comparison.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    check(Path(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
