"""Correctness/liveness under concurrent prefill+decode; not a performance comparison."""
import concurrent.futures,json,time,urllib.request,sys
from benchmark import ROOT,URL,metrics,call
from quality import prompt
out=ROOT/'raw'/sys.argv[1];dest=out/'mixed-prefill.json';assert not dest.exists()
before=metrics();assert before['scheduler']['active']==before['scheduler']['queued']==0
request=dict(model='XiaomiMiMo/MiMo-V2.6-Flash-RL',messages=[dict(role='user',content='Write a detailed Python tutorial with five complete examples of parsing structured log files, including malformed records, filtering, aggregation, time zones, and streaming. Explain each example carefully.')],temperature=0,max_tokens=1024,stream=True,stream_options={'include_usage':True},prefix_cache=False)
p,expected=prompt(700);events=[];future=None;started=time.monotonic();first=None
with concurrent.futures.ThreadPoolExecutor(1) as pool:
 with urllib.request.urlopen(urllib.request.Request(URL+'chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'}),timeout=180) as response:
  for line in response:
   if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]':continue
   e=json.loads(line[6:]);events.append(e)
   if first is None and any(c.get('delta',{}).get('content') or c.get('delta',{}).get('reasoning_content') for c in e.get('choices',[])):
    first=time.monotonic();future=pool.submit(call,'mixed-retrieval-700',p,256,False)
 assert future is not None
 target_done=time.monotonic();retrieval=future.result()
usage=next((e['usage'] for e in reversed(events) if e.get('usage')),None)
parsed=json.loads(retrieval['content'].strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
after=metrics()
replays_before=before['scheduler']['decode_batch']['replays_by_slots']
replays_after=after['scheduler']['decode_batch']['replays_by_slots']
# Metrics versions may expose integer slots as object keys or positional arrays.
def two(values):return values.get('2',0) if isinstance(values,dict) else values[2]
record=dict(scope=__doc__,first_request=request,first_events=events,first_usage=usage,first_seconds=target_done-started,first_ttft=first-started,retrieval=retrieval,expected=expected,exact_answer=parsed==expected,two_slot_replays=two(replays_after)-two(replays_before),before=before,after=after)
dest.write_text(json.dumps(record,indent=2))
assert usage and usage['completion_tokens']>0
assert parsed==expected
assert after['service']['requests_total']-before['service']['requests_total']==2
assert after['service']['requests_failed']==before['service']['requests_failed'] and not after['service']['engine_failed']
assert after['scheduler']['active']==after['scheduler']['queued']==0
assert record['two_slot_replays']>0,'No simultaneous decode observed; inspect before calling this C2 coverage'
print('PASS cold7K exact retrieval during ongoing generation;',usage['completion_tokens'],'first-request tokens;',record['two_slot_replays'],'two-slot replays; no performance comparison',flush=True)
