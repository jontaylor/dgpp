#!/usr/bin/env python3
"""Report per-rank live cache and fixed-slot prefix budgets for MiMo cache modes."""
import argparse
import json
from pathlib import Path


def budget(config, context, forward_rows, world, requests, slots, mtp_blocks):
    if min(context, forward_rows, world, requests, slots) < 1 or not 0 <= mtp_blocks <= 3:
        raise ValueError("positive dimensions and 0..3 MTP blocks required")
    pattern = config["hybrid_layer_pattern"]
    if any(value not in (0, 1) for value in pattern):
        raise ValueError("unknown attention pattern")
    global_layers = pattern.count(0)
    swa_layers = pattern.count(1)
    global_heads = config["num_key_value_heads"]
    swa_heads = config["swa_num_key_value_heads"]
    if global_heads % world or swa_heads % world:
        raise ValueError("TP world must divide both KV head counts")
    capacity = (context + 127) // 128 * 128
    rows = max(8, forward_rows)
    ring = 1 << (128 + rows - 2).bit_length()
    hidden = config["hidden_size"]
    snapshot_values = (global_layers * capacity * global_heads // world
                       + swa_layers * 128 * swa_heads // world) * 320
    live_values = (global_layers * capacity * global_heads // world
                   + swa_layers * ring * swa_heads // world) * 320
    draft_bytes = (mtp_blocks * ring * swa_heads // world * 320 * 2
                   + ring * hidden * 2 + hidden * 2) if mtp_blocks else 0
    result = {"scope": "per TP rank", "ring_capacity": ring, "prefix_slots": slots,
              "native_mtp_and_hidden_bytes_per_snapshot": draft_bytes}
    for name, width in (("bf16", 2), ("fp8_e4m3_unit", 1)):
        snapshot = snapshot_values * width + draft_bytes
        result[name] = {"live_base_cache_bytes": requests * live_values * width,
                        "base_snapshot_bytes": snapshot_values * width,
                        "snapshot_bytes": snapshot,
                        "prefix_arena_bytes": slots * snapshot,
                        "prefix_cache_gib": slots * snapshot / 2**30}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--forward-rows", type=int, default=2048)
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--mtp-blocks", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(budget(json.loads(args.config.read_text()), args.context, args.forward_rows,
                            args.world, args.requests, args.slots, args.mtp_blocks), indent=2))
