#!/usr/bin/env python3
"""Measure exact-route handoff in one rank's Nsight Systems + expert read trace.

Diagnostic only: kernel completion is a conservative input-ready boundary;
pread entry is not physical SSD dispatch. No speedup estimate.
Uses Nsight's host/device timeline throughout; identifies expert read threads
from diagnostic trace writes, excluding Engram pread workers.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import json
import hashlib
from pathlib import Path
import sqlite3
import statistics
import struct


def summarize(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    def quantile(q):
        at = (len(values)-1)*q
        lo = int(at)
        return values[lo]+(values[min(lo+1, len(values)-1)]-values[lo])*(at-lo)
    return dict(count=len(values), mean_us=statistics.mean(values)/1000,
                p50_us=quantile(.5)/1000, p95_us=quantile(.95)/1000,
                min_us=values[0]/1000, max_us=values[-1]/1000)


def analyze(database, read_paths, windows, expected_workers=None):
    c = sqlite3.connect(f'file:{database.resolve()}?mode=ro', uri=True)
    epoch, = c.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()
    identity = []
    columns={x[1] for x in c.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
    if {'globalPid','deviceId','contextId'} <= columns:
        identity=c.execute('SELECT DISTINCT globalPid,deviceId,contextId FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
        if len(identity)!=1:
            raise ValueError('requires one process/device/context per capture')
    raw = c.execute('''SELECT k.start,k.end,k.streamId,s.value,k.gridX,k.gridY,k.graphNodeId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        ORDER BY k.start''').fetchall()
    streams = defaultdict(list)
    for start,end,stream,name,gx,gy,node in raw:
        streams[stream].append(dict(start=start,end=end,stream=stream,name=name,gx=gx,gy=gy,node=node))
    target_records = 0
    for path in read_paths:
        b = path.read_bytes()
        if len(b) % 64:
            raise ValueError('truncated expert read trace')
        target_records += sum(not x[5] for x in struct.iter_unpack('<8Q', b))
    # Expert read workers write the opt-in 64-byte trace after each pread.
    # Engram readers do not. Identify that worker set by near-one-to-one
    # pread/write counts, and require the configured worker count below.
    calls = c.execute("""SELECT a.start,a.end,a.globalTid,s.value FROM OSRT_API a
        JOIN StringIds s ON a.nameId=s.id WHERE s.value='write' OR s.value LIKE 'pread%'
        ORDER BY a.start""").fetchall()
    counts = defaultdict(lambda: [0,0])
    for start,end,tid,name in calls:
        counts[tid][0 if name.startswith('pread') else 1] += 1
    workers = {tid for tid,(n,w) in counts.items() if n and .9 <= w/n <= 1.1}
    if not workers or (expected_workers is not None and len(workers)!=expected_workers):
        raise ValueError(f'ambiguous expert worker set: {len(workers)} candidates, expected {expected_workers}')
    reads = [dict(start=start,end=end,thread=tid) for start,end,tid,name in calls
             if tid in workers and name.startswith('pread')]
    read_starts = [r['start'] for r in reads]
    sync_ends=defaultdict(list)
    d2h=defaultdict(list)
    tables={x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if {'CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_SYNCHRONIZATION'} <= tables:
        runtime={corr:end for corr,end in c.execute("""SELECT a.correlationId,a.end
            FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON a.nameId=s.id
            WHERE s.value LIKE 'cudaStreamSynchronize%'""")}
        for stream,corr in c.execute('SELECT streamId,correlationId FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION'):
            if corr in runtime:
                sync_ends[stream].append(runtime[corr])
        for ends in sync_ends.values(): ends.sort()
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        for stream,start,end,size in c.execute('SELECT streamId,start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=2 ORDER BY start'):
            d2h[stream].append((start,end,size))
    routes = []
    for stream,kernels in streams.items():
        for i,k in enumerate(kernels):
            if not k['name'].startswith('moe_router_dots'):
                continue
            end = i
            while end+1 < len(kernels) and kernels[end+1]['name'] in ('moe_router_select_kernel','moe_hash_routes_kernel'):
                end += 1
            final = kernels[end]
            after = kernels[end+1] if end+1 < len(kernels) else None
            prev = kernels[i-1] if i else None
            tokens = kernels[i+1]['gx'] if i+1 < len(kernels) and kernels[i+1]['name']=='moe_router_select_kernel' else k['gy']
            routes.append(dict(start=k['start'],end=final['end'],stream=stream,tokens=tokens,
                hash=final['name']=='moe_hash_routes_kernel',
                producer_end=prev['end'] if prev and prev['name'] in ('rmsnorm_kernel','mhc_dots_kernel') else None,
                producer=prev['name'] if prev else None, node=k['node'],
                gate=after if after and after['name']=='demand_kernel' else None))
    routes.sort(key=lambda r:r['start'])
    route_starts=[r['start'] for r in routes]
    # Graph replay assigns virtual streams to branches. The fused FFN site's
    # dots producer is two capture nodes before routing (deferred comb in
    # between), and need not have the router's stream ID. Match its actual
    # execution by node identity and time, not by a previous arbitrary kernel.
    producers=defaultdict(list)
    for start,end,stream,name,gx,gy,node in raw:
        if name=='mhc_dots_kernel' and node:
            producers[node].append((start,end))
    graph_starts=[r['start'] for r in routes if r['gate']]
    for r in routes:
        if r['gate'] and r['node']:
            candidates=producers.get(r['node']-2,[])
            at=bisect.bisect_left(candidates,(r['start'],))-1
            prior=bisect.bisect_left(graph_starts,r['start'])-1
            previous_route=graph_starts[prior] if prior>=0 else raw[0][0]-1
            if at >= 0 and previous_route < candidates[at][0] and candidates[at][1] <= r['start']:
                r['producer_end']=candidates[at][1]
                r['producer']='mhc_dots_kernel (graph node -2)'
    gates=[r for r in routes if r['gate']]
    rows=[]
    excluded=Counter()
    paired_services=set()
    for r in routes:
        r=dict(r)
        if r['gate']:
            service=[s for s in routes if s['stream']!=r['stream'] and
                     r['gate']['start'] <= s['start'] and s['end'] <= r['gate']['end'] and s['tokens']==r['tokens']]
            if len(service)>1:
                raise ValueError('ambiguous service router in demand gate')
            if not service:
                excluded['graph_gate_without_service_router'] += 1
                continue  # Usually a hit; also retain clipped/unmatched gates.
            s=service[0]
            service_key=(s['stream'],s['start'])
            if service_key in paired_services:
                raise ValueError('service router paired more than once')
            paired_services.add(service_key)
            r['kind']='decode_graph_miss'
            r['service_start']=s['start'];r['service_end']=s['end'];r['service_stream']=s['stream']
            search_start=s['end'];limit=r['gate']['end']
        elif any(g['gate']['start']<=r['start']<g['gate']['end'] and g['stream']!=r['stream'] for g in gates):
            continue  # Count graph+service as one demand, not two.
        else:
            r['kind']='prefill' if r['tokens']>64 else 'host_small_batch'
            at=bisect.bisect_right(route_starts,r['start'])
            # A later graph router can execute on another virtual stream.
            # Bound globally so a resident draft cannot steal its first read.
            search_start=r['end'];limit=route_starts[at] if at<len(route_starts) else raw[-1][1]
        at=bisect.bisect_left(read_starts,search_start)
        if at>=len(reads) or reads[at]['start']>=limit:
            excluded[r['kind']+'_without_read_in_interval'] += 1
            continue
        first=reads[at]
        absolute=r['start']+epoch
        window=next((w['name'] for w in windows if w['start_ns']+5_000_000<=absolute<w['end_ns']-5_000_000),None)
        if windows and window is None:
            excluded['outside_windows'] += 1
            continue
        r['window']=window;r['first_read_thread']=first['thread'];r['first_read_start']=first['start']
        ends=sync_ends[r.get('service_stream',r['stream'])]
        ix=bisect.bisect_left(ends,search_start)
        if (ix<len(ends) and ends[ix]<first['start'] and
                sum(search_start <= start and end <= ends[ix] and size==r['tokens']*6*4
                    for start,end,size in d2h[r.get('service_stream',r['stream'])]) >= 2):
            r['selection_to_host_routes_ns']=ends[ix]-search_start
            r['host_routes_to_read_ns']=first['start']-ends[ix]
        r['router_ns']=r['end']-r['start']
        r['selection_to_read_ns']=first['start']-r['end']
        if r['producer_end'] is not None:
            r['input_to_router_start_ns']=r['start']-r['producer_end']
            r['input_to_read_ns']=first['start']-r['producer_end']
        if 'service_start' in r:
            r['selection_to_service_router_ns']=r['service_start']-r['end']
            r['service_router_ns']=r['service_end']-r['service_start']
            r['service_selection_to_read_ns']=first['start']-r['service_end']
        rows.append(r)
    groups={}
    for kind in sorted(set(r['kind'] for r in rows)):
        subset=[r for r in rows if r['kind']==kind]
        groups[kind]=dict(observations=len(subset), tokens=sorted(set(r['tokens'] for r in subset)),
            hash_observations=sum(r['hash'] for r in subset),
            metrics={key:summarize([r[key] for r in subset if key in r])
                     for key in ('input_to_router_start_ns','router_ns','selection_to_read_ns','input_to_read_ns',
                                 'selection_to_service_router_ns','service_router_ns','service_selection_to_read_ns',
                                 'selection_to_host_routes_ns','host_routes_to_read_ns')})
    window_groups={}
    for window in sorted(set(r['window'] for r in rows),key=str):
        window_groups[str(window)]={kind:dict(observations=len(subset),
            selection_to_read=summarize([r['selection_to_read_ns'] for r in subset]))
            for kind in groups if (subset := [r for r in rows if r['window']==window and r['kind']==kind])}
    shapes={}
    for kind,tokens in sorted(set((r['kind'],r['tokens']) for r in rows)):
        subset=[r for r in rows if r['kind']==kind and r['tokens']==tokens]
        shapes[f'{kind}:{tokens}']=dict(observations=len(subset),metrics={
            key:summarize([r[key] for r in subset if key in r]) for key in groups[kind]['metrics']})
    return dict(format='dgpp-router-window-v1',session_epoch_unix_ns=epoch,groups=groups,
        windows=window_groups, shapes=shapes, process_device_context=identity,excluded=dict(excluded),
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        read_attribution=dict(worker_threads=sorted(workers), thread_call_counts={str(t):counts[t] for t in sorted(workers)},
                              captured_pread_calls=len(reads),total_target_trace_records=target_records),
        assumptions='One rank/process per database; same-process target reads. Kernel completion checkpoints; fixed profiler timeline. First OSRT pread entry on inferred expert workers, not queue acceptance or physical SSD dispatch. Worker inference requires diagnostic read tracing and near-one-to-one pread/write counts. Read-associated observations only. Next expert pread is temporally associated within one serial invocation; no cross-layer speculative prefetch is allowed. Window assignment excludes 5 ms at either request boundary. Routing fence metrics require matching IDs/weights D2H copies. Kernel spans include dependency waiting; producer completion does not measure earlier partial output readiness.',
        observations=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sqlite',type=Path,required=True)
    p.add_argument('--reads',type=Path,nargs='+',required=True)
    p.add_argument('--windows',type=Path)
    p.add_argument('--io-workers',type=int,default=8,help='Expected expert read worker count; reject ambiguous thread attribution')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=analyze(a.sqlite,a.reads,json.loads(a.windows.read_text()) if a.windows else [], a.io_workers)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='observations'},indent=2))


if __name__=='__main__':main()
