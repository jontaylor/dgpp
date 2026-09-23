"""Same-source comparisons, preserving separate legacy-baseline summaries."""
import json,pathlib,statistics
r=pathlib.Path(__file__).resolve().parent
base={x['name']:x for x in json.loads((r/'raw/native-current/results.json').read_text())}
summary={}
for tag in ['dense-bridge','cache-fast','all-three']:
 p=r/'raw'/tag/'results.json'
 if not p.exists():continue
 rows=json.loads(p.read_text());groups={}
 for row in rows:groups.setdefault(row['name'].rsplit('-r',1)[0],[]).append((row,base[row['name']]))
 out={}
 for group,pairs in groups.items():
  values={k:statistics.mean(x[k] for x,y in pairs) for k in ['ttft','decode_seconds']}
  refs={k:statistics.mean(y[k] for x,y in pairs) for k in values}
  out[group]=dict(samples=len(pairs),candidate=values,control=refs,changes_percent={k:100*(values[k]/refs[k]-1) for k in values},equal_text=sum((x['content'],x['reasoning'])==(y['content'],y['reasoning']) for x,y in pairs),completion_tokens=[(x['usage']['completion_tokens'],y['usage']['completion_tokens']) for x,y in pairs])
 summary[tag]=dict(requests=len(rows),complete=len(rows)==19 and (p.parent/'metrics.json').exists(),groups=out)
 print(tag,'requests',len(rows),'complete',summary[tag]['complete'])
 for k,v in out.items():print(k,'TTFT %.2fs (%+.2f%%), decode %.2fs (%+.2f%%)'%(v['candidate']['ttft'],v['changes_percent']['ttft'],v['candidate']['decode_seconds'],v['changes_percent']['decode_seconds']))
(r/'same-source-summary.json').write_text(json.dumps(dict(control='native-current',source='41fd8a7',scope='Matched fixed prompts and output limits; text may differ; two repetitions except long seed. Small changes may be noise.',candidates=summary),indent=2))
