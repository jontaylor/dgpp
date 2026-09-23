"""Parent-owned sequential phase controller; refuses overlap and stops on failure."""
import json
import pathlib
import subprocess
import time

root = pathlib.Path(__file__).resolve().parent
def run(label, args, timeout=1800):
    print('START', label, flush=True)
    with (root / (label + '.log')).open('w') as f:
        subprocess.run(args, cwd=root, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    print('PASS', label, flush=True)

deadline = time.monotonic() + 1800
while not (root / 'raw/fp8-dense/metrics.json').exists():
    if time.monotonic() > deadline:
        raise TimeoutError('dense benchmark did not finish successfully')
    time.sleep(2)
run('fp8-dense-acceptance', ['python3', 'acceptance.py', 'fp8-dense'])
run('fp8-dense-quality', ['python3', 'quality.py', 'fp8-dense', '2800'])
run('fp8-dense-down', ['python3', 'deploy.py', 'down'])
for rank in [0, 1]:
    host = 'jon@192.168.0.' + str(171 + rank)
    assert subprocess.run(['ssh', host, 'pgrep -x dgpp-serve'], stdout=subprocess.DEVNULL).returncode == 1
    remote = '/tmp/dgpp-mimo-recipe-20260922'
    run(f'followup-stage-r{rank}', ['scp', str(root / 'combined/build-spark-cross/mimo_fp8_test'), str(root / 'fp8-hybrid/build-spark-cross/mimo_fp8_dense_check'), host + ':' + remote + '/'])
    run(f'fp8-audit-gpu-r{rank}', ['ssh', host, remote + '/mimo_fp8_test'])
    checkpoint = '/home/jon/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL/snapshots/3b38d063180c3e4aed9691fdc735f3d10b266ee4'
    run(f'hybrid-dense-gpu-r{rank}', ['ssh', host, remote + '/mimo_fp8_dense_check ' + checkpoint + ' ' + str(rank) + ' 2'])
print('DONE dense service and follow-up GPU checks; both servers stopped', flush=True)
