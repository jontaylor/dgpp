"""Read-only parent capture of the current two-rank deployment."""
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
from benchmark import metrics

root = pathlib.Path(__file__).resolve().parent
state = json.loads((root / 'deployment-state.json').read_text())
assert state['running']
tag = sys.argv[1]
dest = root / 'raw' / tag
dest.mkdir(parents=True, exist_ok=True)
assert not (dest / 'verified-deployment.json').exists(), 'preserve evidence'
records = []
for rank in range(2):
    host = 'jon@192.168.0.' + str(171 + rank)
    # Only opt-in variables are exposed; the site environment contains secrets.
    script = '''import pathlib,json,subprocess,hashlib
pid=subprocess.check_output(['pgrep','-x','dgpp-serve'],text=True).strip()
assert pid.isdigit(),pid
p=pathlib.Path('/proc')/pid
argv=(p/'cmdline').read_bytes().decode().split('\\0')
config=pathlib.Path(argv[argv.index('--config')+1])
env=(p/'environ').read_bytes().decode().split('\\0')
log=config.parent/('serve_r'+str(RANK)+'.log')
print(json.dumps(dict(pid=int(pid),sha256=hashlib.sha256((p/'exe').read_bytes()).hexdigest(),argv=argv,config=json.loads(config.read_text()),env=[x for x in env if x.startswith('DGPP_MIMO_')],log=log.read_text())))
'''.replace('RANK', str(rank))
    result = subprocess.run(['ssh', host, 'python3 -c ' + shlex.quote(script)], capture_output=True, text=True, check=True)
    record = json.loads(result.stdout)
    (dest / f'rank{rank}-startup.log').write_text(record.pop('log'))
    records.append(record)
assert records[0]['sha256'] == records[1]['sha256']
assert sorted(records[0]['env']) == sorted(records[1]['env'])
(dest / 'verified-deployment.json').write_text(json.dumps(dict(state=state, ranks=records, metrics=metrics()), indent=2))
print('Verified identical binaries and MiMo flags on both ranks:', records[0]['sha256'])
