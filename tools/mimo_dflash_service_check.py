#!/usr/bin/env python3
"""Parent-run, bounded TP2 DFlash correctness evidence. No deployment actions.

Capture once with target-only mtp=false, once with DFlash7, then compare offline.
Text equality is a scoped observation: row-shape rounding can change near ties.
"""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import threading
import time
import urllib.request

PROMPTS = {
    'math': 'Solve 17x + 23 = 210. Give the exact fraction and verify it.',
    'code': 'Write a Python function merging overlapping intervals. Include two concise tests.',
    'json': 'Return only a JSON array of 12 objects with id, square, and parity for integers 1 through 12.',
    'prose': 'Describe a quiet coastal town in winter in one paragraph with specific sensory details.',
}
EXPECTED = [*(f'c1-{k}' for k in PROMPTS), 'c2-code', 'c2-json',
            'prefix-seed', 'prefix-warm', 'prefix-recompute', 'cancel', 'recovery']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def first_difference(a, b):
    if a == b:
        return None
    index = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return {'character_index': index, 'left_length': len(a), 'right_length': len(b),
            'left_context': a[max(0, index-48):index+96],
            'right_context': b[max(0, index-48):index+96]}


def extract(events):
    out = {'content': '', 'reasoning': '', 'finish_reason': None, 'usage': None}
    for event in events:
        if event.get('error'):
            raise RuntimeError(event['error'])
        if event.get('usage'):
            out['usage'] = event['usage']
        for choice in event.get('choices', []):
            delta = choice.get('delta', {})
            out['content'] += delta.get('content') or ''
            out['reasoning'] += delta.get('reasoning_content') or ''
            if choice.get('finish_reason') is not None:
                out['finish_reason'] = choice['finish_reason']
    return out


def payload(model, text, pair, limit=96, cache=False):
    return {'model': model, 'messages': [{'role': 'user', 'content': f'Evaluation {pair}.\n{text}'}],
            'temperature': 0, 'max_tokens': limit, 'stream': True,
            'stream_options': {'include_usage': True}, 'prefix_cache': cache}


class Capture:
    def __init__(self, args):
        self.args = args
        self.out = pathlib.Path(args.out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.deadline = self.started + args.budget_seconds
        self.records = {}
        self.metric_records = {}

    def remaining(self):
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError('Stage wall-clock budget exhausted')
        return value

    def metrics(self, label, drain=False):
        until = min(self.deadline, time.monotonic()+15)
        samples = []
        while True:
            with urllib.request.urlopen(self.args.url.rstrip('/')+'/metrics',
                                        timeout=min(5, self.remaining())) as response:
                m = json.load(response)
            samples.append({'elapsed': time.monotonic()-self.started, 'metrics': m})
            write(self.out/(label+'.metrics.json'), samples)
            self.metric_records[label] = m
            if not drain or (m['scheduler']['active'] == m['scheduler']['queued'] == 0):
                return m
            if time.monotonic() >= until:
                raise TimeoutError('Server did not drain after '+label)
            time.sleep(0.25)

    def call(self, name, request, barrier=None, cancel=False):
        directory = self.out/name
        directory.mkdir()
        write(directory/'request.json', request)
        result = {'name': name, 'request': request, 'events': [], 'cancel_requested': cancel,
                  'cancelled_by_client': False, 'done': False, 'status': 'error'}
        try:
            if barrier:
                barrier.wait(timeout=min(10, self.remaining()))
            begin = time.monotonic()
            result['start_elapsed'] = begin-self.started
            deadline = min(self.deadline, begin+self.args.request_timeout)
            req = urllib.request.Request(self.args.url.rstrip('/')+'/chat/completions',
                                         data=json.dumps(request).encode(),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=min(self.args.request_timeout, self.remaining())) as response:
                result['http_status'] = response.status
                # urllib's socket timeout otherwise applies per read, not per request.
                sock = response.fp.raw._sock
                text_events = 0
                with (directory/'stream.jsonl').open('w') as saved:
                    while True:
                        left = deadline-time.monotonic()
                        if left <= 0:
                            raise TimeoutError('Streaming request deadline')
                        sock.settimeout(left)
                        line = response.readline()
                        if not line:
                            break
                        if not line.startswith(b'data:'):
                            continue
                        data = line[5:].strip()
                        saved.write(json.dumps({'elapsed': time.monotonic()-begin,
                                                'data': data.decode('utf-8')}, ensure_ascii=False)+'\n')
                        saved.flush()
                        if data == b'[DONE]':
                            result['done'] = True
                            break
                        event = json.loads(data)
                        result['events'].append(event)
                        if event.get('error'):
                            raise RuntimeError(event['error'])
                        if any(c.get('delta', {}).get('content') or
                               c.get('delta', {}).get('reasoning_content') for c in event.get('choices', [])):
                            text_events += 1
                            result.setdefault('ttft', time.monotonic()-begin)
                        if cancel and text_events >= 3:
                            result['cancelled_by_client'] = True
                            break  # response.close() disconnects the stream here.
            result.update(extract(result['events']))
            if cancel:
                if not result['cancelled_by_client'] or result['finish_reason'] is not None:
                    raise RuntimeError('Cancellation was not early; no cancellation evidence')
            elif not result['done'] or not result['usage'] or result['finish_reason'] is None:
                raise RuntimeError('Incomplete SSE terminal/usage evidence')
            result['status'] = 'ok'
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
            result.update(extract([e for e in result['events'] if not e.get('error')]))
        finally:
            result['end_elapsed'] = time.monotonic()-self.started
            result['seconds'] = result['end_elapsed']-result.get('start_elapsed', result['end_elapsed'])
            write(directory/'result.json', result)
        return result

    def save(self, result):
        self.records[result['name']] = result
        write(self.out/'results.json', self.records)
        print(result['name'], result['status'], round(result['seconds'], 2), flush=True)
        if result['status'] != 'ok':
            raise RuntimeError(result.get('error', 'Request failed'))

    def run(self):
        args = self.args
        config_bytes = pathlib.Path(args.deployment).read_bytes()
        config = json.loads(config_bytes)
        engine = config['engine']
        if args.mode == 'control' and engine.get('mtp') is not False:
            raise ValueError('Control deployment must explicitly set engine.mtp=false')
        if args.mode == 'dflash' and not (engine.get('mtp') is True and engine.get('mtp_depth') == 7):
            raise ValueError('DFlash deployment must set mtp=true, mtp_depth=7')
        manifest = {'protocol': 'mimo-dflash-service-v1', 'mode': args.mode, 'pair_id': args.pair_id, 'started_unix_seconds': time.time(),
                    'model': args.model, 'url': args.url, 'runtime_label': args.runtime_label,
                    'deployment_sha256': hashlib.sha256(config_bytes).hexdigest(), 'engine': engine,
                    'budget_seconds': args.budget_seconds, 'expected_requests': EXPECTED,
                    'note': 'Supplied configuration is recorded, not live process attestation.'}
        write(self.out/'manifest.json', manifest)
        summary = {'status': 'error', 'checks': {}, 'missing': EXPECTED}
        try:
            before = self.metrics('before', drain=True)
            make = lambda text, limit=96, cache=False: payload(args.model, text, args.pair_id, limit, cache)
            for name, text in PROMPTS.items():
                self.save(self.call('c1-'+name, make(text)))
            self.metrics('after-c1', drain=True)
            barrier = threading.Barrier(2)
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(self.call, 'c2-'+name, make(PROMPTS[name]), barrier)
                           for name in ['code', 'json']]
                # Save both outcomes even if one fails.
                records = [f.result() for f in futures]
                for r in records:
                    self.records[r['name']] = r
                write(self.out/'results.json', self.records)
                for r in records:
                    self.save(r)
            self.metrics('after-c2', drain=True)
            text = ''.join(f'Record {i}: value {i%17}.\n' for i in range(400))
            text += 'Summarize the dataset and give a concise Python function computing value frequencies.'
            for name, cache in [('prefix-seed', True), ('prefix-warm', True), ('prefix-recompute', False)]:
                self.save(self.call(name, make(text, 64, cache)))
                self.metrics('after-'+name, drain=True)
            self.metrics('before-cancel', drain=True)
            self.save(self.call('cancel', make('List integers from 1 through 1000, one per line.', 256), cancel=True))
            self.metrics('after-cancel', drain=True)
            self.save(self.call('recovery', make(PROMPTS['code'])))
            after = self.metrics('after', drain=True)
            spec = after['scheduler']['spec_decode']
            drafts = spec['num_drafts_total']-before['scheduler']['spec_decode']['num_drafts_total']
            warm = self.records['prefix-warm']['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
            cancelled = after['service']['requests_cancelled']-before['service']['requests_cancelled']
            checks = {'request_accounting': after['service']['requests_total']-before['service']['requests_total'] == 11,
                      'no_server_failures': after['service']['requests_failed'] == before['service']['requests_failed'] and not after['service']['engine_failed'],
                      'cancel_observed': cancelled == 1,
                      'cache_hit_observed': warm > 0,
                      'mode_metrics': drafts == 0 if args.mode == 'control' else drafts > 0 and spec['depth'] == 7,
                      'c2_client_overlap': max(r['start_elapsed'] for r in records) < min(r['end_elapsed'] for r in records)}
            c2_replays = {key: value-self.metric_records['after-c1']['scheduler']['decode_batch']['replays_by_slots'].get(key, 0)
                          for key, value in self.metric_records['after-c2']['scheduler']['decode_batch']['replays_by_slots'].items()}
            summary.update(status='ok' if all(checks.values()) else 'inconclusive', checks=checks, c2_replays_by_slots=c2_replays,
                           drafts_delta=drafts, cancelled_delta=cancelled, warm_cached_tokens=warm)
        except Exception as error:
            summary['error'] = f'{type(error).__name__}: {error}'
            if self.deadline-time.monotonic() > 1:
                try:
                    self.metrics('failure', drain=False)
                except Exception as metrics_error:
                    summary['metrics_error'] = str(metrics_error)
        finally:
            summary['missing'] = sorted(set(EXPECTED)-self.records.keys())
            summary['seconds'] = time.monotonic()-self.started
            write(self.out/'summary.json', summary)
            print(json.dumps(summary, indent=2))
        return summary['status'] == 'ok'


def compare(args):
    dirs = [pathlib.Path(args.control), pathlib.Path(args.dflash)]
    runs = [json.loads((p/'results.json').read_text()) for p in dirs]
    manifests = [json.loads((p/'manifest.json').read_text()) for p in dirs]
    summaries = [json.loads((p/'summary.json').read_text()) for p in dirs]
    report = {'scope': 'Decoded content/reasoning equality for this fixed small panel only. Not token-ID parity or generalization. Row-shape GEMM rounding may change near ties.',
              'capture_valid': all(s['status'] == 'ok' for s in summaries) and all(set(EXPECTED) <= r.keys() for r in runs), 'comparisons': []}
    e0, e1 = (m['engine'] for m in manifests)
    report['engine_differences'] = {k: [e0.get(k), e1.get(k)] for k in e0.keys() | e1.keys() if e0.get(k) != e1.get(k)}
    report['configuration_matched'] = (manifests[0]['mode'] == 'control' and manifests[1]['mode'] == 'dflash' and
        manifests[0]['pair_id'] == manifests[1]['pair_id'] and manifests[0]['model'] == manifests[1]['model'] and
        set(report['engine_differences']) <= {'mtp', 'mtp_depth'})

    def check(label, left, right, compare_request=True):
        differences = {field: first_difference(left.get(field, ''), right.get(field, '')) for field in ['content', 'reasoning']}
        differences = {k: v for k, v in differences.items() if v is not None}
        report['comparisons'].append({'label': label, 'requests_equal': left['request'] == right['request'] if compare_request else None,
            'both_complete': left['status'] == right['status'] == 'ok' and left['done'] and right['done'],
            'differences': differences, 'finish_reasons': [left.get('finish_reason'), right.get('finish_reason')],
            'completion_tokens': [(r.get('usage') or {}).get('completion_tokens') for r in [left, right]],
            'prompt_tokens': [(r.get('usage') or {}).get('prompt_tokens') for r in [left, right]]})
    for name in EXPECTED:
        if name != 'cancel' and all(name in r for r in runs):
            check('control-vs-dflash/'+name, runs[0][name], runs[1][name])
    for mode, run in zip(['control', 'dflash'], runs):
        for a, b in [('c1-code', 'c2-code'), ('c1-json', 'c2-json'), ('c1-code', 'recovery'),
                     ('prefix-seed', 'prefix-warm'), ('prefix-warm', 'prefix-recompute')]:
            if a in run and b in run:
                check(mode+'/'+a+'-vs-'+b, run[a], run[b], compare_request=False)
    report['observed_agreement'] = report['capture_valid'] and report['configuration_matched'] and all(
        c['both_complete'] and not c['differences'] and c['requests_equal'] is not False and
        c['finish_reasons'][0] == c['finish_reasons'][1] and c['completion_tokens'][0] == c['completion_tokens'][1] and
        c['prompt_tokens'][0] == c['prompt_tokens'][1] for c in report['comparisons'])
    write(pathlib.Path(args.out), report)
    print(json.dumps(report, indent=2))
    return report['observed_agreement']


def self_test():
    assert first_difference('abc', 'abc') is None
    assert first_difference('abc', 'abX')['character_index'] == 2
    assert first_difference('abc', 'abcd')['character_index'] == 3
    x = extract([{'choices': [{'delta': {'reasoning_content': 'r'}}]},
                 {'choices': [{'delta': {'content': 'a'}, 'finish_reason': 'length'}]},
                 {'choices': [], 'usage': {'completion_tokens': 2}}])
    assert x == {'reasoning': 'r', 'content': 'a', 'finish_reason': 'length', 'usage': {'completion_tokens': 2}}
    assert len(EXPECTED) == 11 and len(set(EXPECTED)) == 11
    assert payload('model', 'prompt', 'pair') == payload('model', 'prompt', 'pair')
    import contextlib
    import io
    import tempfile
    with tempfile.TemporaryDirectory(prefix='dflash-check-', dir=pathlib.Path(__file__).resolve().parent) as temp:
        root = pathlib.Path(temp)
        for mode in ['control', 'dflash']:
            directory = root/mode
            directory.mkdir()
            write(directory/'manifest.json', {'mode': mode, 'model': 'm', 'pair_id': 'p',
                  'engine': {'mtp': mode == 'dflash', 'mtp_depth': 7 if mode == 'dflash' else 1}})
            write(directory/'summary.json', {'status': 'ok'})
            records = {name: {'request': {}, 'status': 'ok', 'done': name != 'cancel',
                       'content': 'same', 'reasoning': '', 'finish_reason': 'length',
                       'usage': {'prompt_tokens': 3, 'completion_tokens': 2}} for name in EXPECTED}
            write(directory/'results.json', records)
        args = argparse.Namespace(control=str(root/'control'), dflash=str(root/'dflash'), out=str(root/'report.json'))
        with contextlib.redirect_stdout(io.StringIO()):
            assert compare(args)
            candidate = json.loads((root/'dflash/results.json').read_text())
            candidate['c1-code']['content'] = 'sXme'
            write(root/'dflash/results.json', candidate)
            assert not compare(args)
            report = json.loads((root/'report.json').read_text())
            assert next(c for c in report['comparisons'] if c['label'] == 'control-vs-dflash/c1-code')['differences']['content']['character_index'] == 1
            candidate['c1-code']['usage'] = None
            candidate['c1-code']['status'] = 'error'
            write(root/'dflash/results.json', candidate)
            assert not compare(args)
            del candidate['recovery']
            write(root/'dflash/results.json', candidate)
            assert not compare(args)
    print('PASS offline SSE extraction, divergence, complete/missing/failed comparisons and protocol count; no network access')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('capture')
    for name in ['url', 'mode', 'out', 'deployment', 'pair-id', 'runtime-label']:
        c.add_argument('--'+name, required=True, **({'choices': ['control', 'dflash']} if name == 'mode' else {}))
    c.add_argument('--model', default='XiaomiMiMo/MiMo-V2.6-Flash-RL')
    c.add_argument('--budget-seconds', type=float, default=300)
    c.add_argument('--request-timeout', type=float, default=45)
    c = sub.add_parser('compare')
    for name in ['control', 'dflash', 'out']:
        c.add_argument('--'+name, required=True)
    sub.add_parser('self-test')
    args = p.parse_args()
    ok = self_test() if args.command == 'self-test' else compare(args) if args.command == 'compare' else Capture(args).run()
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
