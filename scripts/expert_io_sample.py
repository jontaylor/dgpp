#!/usr/bin/env python3
"""Record Linux block-device counters for expert-cache what-if replay.

Reads sysfs only. Stop with Ctrl-C or SIGTERM; output is JSON lines containing
[Unix nanoseconds, *Linux block stat fields]. No model requests are issued.
"""
import argparse
import json
import math
from pathlib import Path
import signal
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='nvme0n1', help='name in /sys/block')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--interval-ms', type=float, default=2.)
    p.add_argument('--seconds', type=float, default=600., help='zero means until stopped')
    args = p.parse_args()
    if (not math.isfinite(args.interval_ms) or args.interval_ms <= 0
            or not math.isfinite(args.seconds) or args.seconds < 0):
        p.error('interval must be positive and duration nonnegative, both finite')
    if Path(args.device).name != args.device:
        p.error('device must be a block-device name')
    source = Path('/sys/block')/args.device/'stat'
    source.read_text()  # fail before creating output if the device is absent
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.output.open('x') as out:
        while running and (not args.seconds or time.monotonic()-started < args.seconds):
            tick = time.monotonic()
            stats = list(map(int, source.read_text().split()))
            out.write(json.dumps([time.time_ns(), *stats])+'\n')
            time.sleep(max(0., args.interval_ms/1000-(time.monotonic()-tick)))


if __name__ == '__main__':
    main()
