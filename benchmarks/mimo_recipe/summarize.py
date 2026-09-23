"""Summarize matched service measurements without equating kernel and service gains."""
import json
import pathlib
import statistics

root = pathlib.Path(__file__).resolve().parent
base = {r['name']: r for r in json.loads((root / 'raw/baseline/results.json').read_text())}
summary = {}
for path in sorted((root / 'raw').glob('*/results.json')):
    tag = path.parent.name
    if tag == 'baseline':
        continue
    rows = json.loads(path.read_text())
    groups = {}
    for row in rows:
        ref = base[row['name']]
        name = row['name'].rsplit('-r', 1)[0]
        values = groups.setdefault(name, [])
        values.append(dict(ttft_seconds=row['ttft'], baseline_ttft_seconds=ref['ttft'],
                           decode_seconds=row['decode_seconds'], baseline_decode_seconds=ref['decode_seconds'],
                           completion_tokens=row['usage']['completion_tokens'], baseline_completion_tokens=ref['usage']['completion_tokens'],
                           same_output=(row['content'], row['reasoning']) == (ref['content'], ref['reasoning'])))
    aggregate = {}
    for name, values in groups.items():
        result = dict(samples=len(values), same_output_count=sum(v['same_output'] for v in values))
        for key in ('ttft_seconds', 'baseline_ttft_seconds', 'decode_seconds', 'baseline_decode_seconds', 'completion_tokens', 'baseline_completion_tokens'):
            result[key] = statistics.mean(v[key] for v in values)
        result['ttft_change_percent'] = 100 * (result['ttft_seconds'] / result['baseline_ttft_seconds'] - 1)
        result['decode_time_change_percent'] = 100 * (result['decode_seconds'] / result['baseline_decode_seconds'] - 1)
        aggregate[name] = result
    summary[tag] = dict(completed_matched_requests=len(rows), complete_suite=(path.parent / 'metrics.json').exists(), groups=aggregate)
(root / 'service-summary.json').write_text(json.dumps(summary, indent=2))
for tag, result in summary.items():
    print(tag, result['completed_matched_requests'], 'suite complete:', result['complete_suite'])
    for name, row in result['groups'].items():
        print(' ', name, 'TTFT %.2fs (%+.1f%%), decode %.2fs (%+.1f%%), equal outputs %d/%d' %
              (row['ttft_seconds'], row['ttft_change_percent'], row['decode_seconds'], row['decode_time_change_percent'], row['same_output_count'], row['samples']))
