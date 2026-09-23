"""Parent runs only while both services are down; all gates precede microbenchmarks."""
import json
import pathlib
import subprocess
import time
root = pathlib.Path(__file__).resolve().parent
assert not json.loads((root / 'deployment-state.json').read_text())['running']
remote = '/tmp/dgpp-mimo-recipe-split'
def run(label, command, timeout=600):
    print('START', label, flush=True)
    with (root / (label + '.log')).open('w') as f:
        subprocess.run(command, cwd=root, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=True)
    print('PASS', label, flush=True)
for rank in [0, 1]:
    host = 'jon@192.168.0.' + str(171 + rank)
    assert subprocess.run(['ssh', host, 'pgrep -x dgpp-serve'], stdout=subprocess.DEVNULL).returncode == 1
    run(f'split-mkdir-r{rank}', ['ssh', host, 'mkdir -p ' + remote])
    run(f'split-stage-r{rank}', ['scp', *[str(root / 'split-online/build-spark-cross' / n) for n in ['mimo_attn_test', 'mimo_fp8_test']], host + ':' + remote + '/'])
    run(f'split-attention-r{rank}', ['ssh', host, remote + '/mimo_attn_test'])
    run(f'split-fp8-r{rank}', ['ssh', host, remote + '/mimo_fp8_test'])
for keys in [512, 2048]:
    run(f'split-keys-{keys}', ['ssh', 'jon@192.168.0.171', f'env DGPP_MIMO_ONLINE_SPLIT_KEYS={keys} DGPP_TEST_FILTER=split_online {remote}/mimo_attn_test'])
for kind in ['memcheck', 'racecheck']:
    run('split-' + kind, ['ssh', 'jon@192.168.0.171', f'env DGPP_TEST_FILTER=split_online /usr/local/cuda/bin/compute-sanitizer --error-exitcode 99 --tool {kind} {remote}/mimo_attn_test'])
for context in [8192, 65536]:
    run(f'split-micro-{context}', ['ssh', 'jon@192.168.0.172', f'env DGPP_MIMO_ATTN_BENCH=1 DGPP_MIMO_ATTN_BENCH_CONTEXT={context} {remote}/mimo_attn_test'])
run('split-decode-micro', ['ssh', 'jon@192.168.0.172', f'env DGPP_MIMO_DECODE_BENCH=1 {remote}/mimo_attn_test'])
print('DONE split functional/sanitizer/microbench gates', flush=True)
