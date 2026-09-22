import json,pathlib,sys,urllib.request
from benchmark import ROOT,URL,metrics
out=ROOT/'raw'/sys.argv[1];out.mkdir(parents=True,exist_ok=True);dest=out/'toolcap.json';assert not dest.exists()
tool={'type':'function','function':{'name':'inspect_file','description':'Inspect one named file.','parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}}}
base=dict(model='XiaomiMiMo/MiMo-V2.6-Flash-RL',messages=[{'role':'user','content':'Emit exactly eight separate inspect_file calls, in order, for file1.py, file2.py, file3.py, file4.py, file5.py, file6.py, file7.py, file8.py. Do not combine paths. Call the tools now.'}],tools=[tool],tool_choice='required',temperature=0,max_tokens=2048)
before=metrics();records=[]
for cap,stream in [(None,False),(1,False),(1,True),(2,False),(2,True)]:
 req=dict(base,stream=stream)
 if cap is not None:req['max_tool_calls']=cap
 if stream:req['stream_options']={'include_usage':True}
 events=[];calls={};finish=None
 with urllib.request.urlopen(urllib.request.Request(URL+'chat/completions',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'}),timeout=300) as f:
  if stream:
   for line in f:
    if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]':continue
    e=json.loads(line[6:]);events.append(e)
    for c in e.get('choices',[]):
     if c.get('finish_reason'):finish=c['finish_reason']
     for t in c.get('delta',{}).get('tool_calls',[]):
      entry=calls.setdefault(t['index'],dict(name='',arguments=''))
      for k,v in t.get('function',{}).items():
       if k in entry:entry[k]+=v
  else:
   e=json.load(f);events=[e];c=e['choices'][0];finish=c['finish_reason'];calls={i:t['function'] for i,t in enumerate(c['message'].get('tool_calls',[]))}
 parsed=[]
 for i,c in sorted(calls.items()):
  args=json.loads(c['arguments']);assert c['name']=='inspect_file' and isinstance(args.get('path'),str) and args['path'],c
  parsed.append(dict(name=c['name'],arguments=args))
 record=dict(cap=cap,stream=stream,request=req,events=events,calls=parsed,finish_reason=finish);records.append(record);dest.write_text(json.dumps(records,indent=2))
 assert finish=='tool_calls',record
 assert len(parsed)>=3 if cap is None else len(parsed)==cap,record
 if cap is not None:assert parsed==records[0]['calls'][:cap],record
 print('PASS cap',cap,'stream',stream,'complete calls',len(parsed),flush=True)
after=metrics();assert after['service']['requests_total']-before['service']['requests_total']==len(records),'external traffic'
assert not after['service']['engine_failed'] and after['service']['requests_failed']==before['service']['requests_failed']
(out/'toolcap-metrics.json').write_text(json.dumps(dict(before=before,after=after),indent=2));print('PASS default, cap1/2, stream parity, complete arguments; generated calls not executed',flush=True)
