import pathlib,json,subprocess,sys
r=pathlib.Path(__file__).resolve().parent;out=r/'raw'/sys.argv[1]
d=json.loads((out/'verified-deployment.json').read_text())
for rank,entry in enumerate(d['ranks']):
 argv=entry['argv']; path=pathlib.PurePosixPath(argv[argv.index('--config')+1]).parent/f'serve_r{rank}.log'
 dest=out/f'rank{rank}-final.log';assert not dest.exists()
 subprocess.run(['scp',f'jon@192.168.0.{171+rank}:{path}',str(dest)],check=True)
print('Archived both rank logs',sys.argv[1])
