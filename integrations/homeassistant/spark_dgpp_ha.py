"""DGPP JSON metrics to Home Assistant; rates are wall-clock poll deltas."""
import json
import math
import os
from spark_vllm_ha import VllmSensors


class DgppSensors(VllmSensors):
    # key, JSON group, field, display name, unit, cumulative counter
    FIELDS = [
        ('active', 'scheduler', 'active', 'Active requests', None, False),
        ('queued', 'scheduler', 'queued', 'Queued requests', None, False),
        ('prefilling', 'scheduler', 'prefilling', 'Prefilling requests', None, False),
        ('pool_total', 'scheduler', 'pool_blocks_total', 'KV pool total blocks', 'blocks', False),
        ('pool_used', 'scheduler', 'pool_blocks_in_use', 'KV pool used blocks', 'blocks', False),
        ('generated', 'scheduler', 'tokens_generated', 'Generated tokens', 'tokens', True),
        ('prompt', 'scheduler', 'prompt_tokens', 'Prompt tokens', 'tokens', True),
        ('computed', 'scheduler', 'prompt_tokens_computed', 'Computed prompt tokens', 'tokens', True),
        ('requests', 'service', 'requests_total', 'Requests', None, True),
        ('failed', 'service', 'requests_failed', 'Failed requests', None, True),
        ('cancelled', 'service', 'requests_cancelled', 'Cancelled requests', None, True),
        ('shed', 'service', 'requests_shed', 'Shed requests', None, True),
        ('bad', 'service', 'rejects_bad', 'Invalid requests', None, True),
        ('cache_slots', 'prefix_cache', 'slots', 'Prefix snapshot slots', None, False),
        ('cache_entries', 'prefix_cache', 'entries', 'Prefix cache entries', None, False),
        ('cache_hits', 'prefix_cache', 'hits', 'Prefix cache hits', None, True),
        ('cache_misses', 'prefix_cache', 'misses', 'Prefix cache misses', None, True),
        ('cache_saved', 'prefix_cache', 'tokens_saved', 'Cached prompt tokens', 'tokens', True),
        ('cache_evictions', 'prefix_cache', 'evictions', 'Prefix cache evictions', None, True),
        ('ttft_hit', 'prefix_cache', 'ttft_hit_ms_avg', 'Mean TTFT cache hit', 'ms', False),
        ('ttft_miss', 'prefix_cache', 'ttft_miss_ms_avg', 'Mean TTFT cache miss', 'ms', False),
    ]

    def __init__(self, device_id, hostname, machine_base, interval):
        super().__init__(device_id, hostname, machine_base, interval)
        self.parent_id = device_id
        self.device_id = device_id + '_dgpp'
        self.base = self.device_id + '/metrics'
        self.url = os.getenv('DGPP_METRICS_URL', 'http://127.0.0.1:30001/v1/metrics')

    def config(self, key, definition):
        cfg = super().config(key, definition)
        cfg['device'] = {'identifiers': [self.device_id], 'name': self.hostname + '-dgpp',
                         'manufacturer': 'DGPP', 'model': 'Inference metrics',
                         'via_device': self.parent_id}
        return cfg

    def parse(self, exposition, now):
        data = json.loads(exposition)
        if not isinstance(data, dict) or any(not isinstance(data.get(g), dict)
                                             for g in ('scheduler', 'service', 'prefix_cache')):
            raise ValueError('Expected DGPP scheduler, service and prefix_cache objects')
        def number(group, field):
            v = data[group].get(field)
            return v if type(v) in (int, float) and math.isfinite(v) and v >= 0 else None
        if number('scheduler', 'pool_blocks_total') is None:
            raise ValueError('Missing DGPP pool capacity')
        values, definitions, counters = {}, {}, {}
        for key, group, field, title, unit, counter in self.FIELDS:
            value = number(group, field)
            if key.startswith('ttft_') and not number('prefix_cache', key + '_count'):
                value = None
            values[key] = value
            definitions[key] = (title, unit, 'total_increasing' if counter else 'measurement')
            if counter and value is not None:
                counters[key] = (now, value)
        batch = data['scheduler'].get('decode_batch', {})
        if not isinstance(batch, dict):
            batch = {}
        batch_fields = [
            ('last_slots', 'Last decode bucket slots', 'slots', False),
            ('last_active', 'Last decode active requests', None, False),
            ('last_rows_per_request', 'Last decode rows per request', 'rows', False),
            ('replays', 'Decode graph replays', None, True),
            ('rows', 'Decode verification rows', 'rows', True),
            ('padded_rows', 'Decode padded rows', 'rows', True),
        ]
        def add_batch(field, value, title, unit, counter=False):
            key = 'decode_batch_' + field
            value = value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None
            values[key] = value
            definitions[key] = (title, unit, 'total_increasing' if counter else 'measurement')
            if counter and value is not None:
                counters[key] = (now, value)
        for field, title, unit, counter in batch_fields:
            add_batch(field, batch.get(field), title, unit, counter)
        histogram = batch.get('replays_by_slots', {})
        if not isinstance(histogram, dict):
            histogram = {}
        for slots in (1, 2, 3, 4, 6, 8, 12, 16):
            add_batch('replays_slots_' + str(slots), histogram.get(str(slots)),
                      'Decode replays ' + str(slots) + ' slots', None, True)
        slots, active, rows = (values['decode_batch_' + k] for k in
                              ('last_slots', 'last_active', 'last_rows_per_request'))
        add_batch('last_padded_rows', (slots - active) * rows
                  if slots is not None and active is not None and rows is not None and slots >= active else None,
                  'Last decode padded rows', 'rows')
        # A drop in any observed counter indicates a server reset. Do not
        # derive rates across that boundary, or across a failed scrape.
        reset = any(k in self.previous and v[1] < self.previous[k][1]
                    for k, v in counters.items())
        for key in ('generated', 'prompt', 'computed'):
            old, new = self.previous.get(key), counters.get(key)
            rate = None
            if not reset and old and new and now > old[0]:
                rate = (new[1] - old[1]) / (now - old[0])
            values[key + '_rate'] = rate
            definitions[key + '_rate'] = (definitions[key][0] + ' per second', 'tokens/s', 'measurement')
        padding_percent = None
        row_old = self.previous.get('decode_batch_rows')
        pad_old = self.previous.get('decode_batch_padded_rows')
        row_new = counters.get('decode_batch_rows')
        pad_new = counters.get('decode_batch_padded_rows')
        if not reset and row_old and pad_old and row_new and pad_new and now > row_old[0]:
            delta_rows = row_new[1] - row_old[1]
            delta_pad = pad_new[1] - pad_old[1]
            if delta_rows > 0 and 0 <= delta_pad <= delta_rows:
                padding_percent = 100 * delta_pad / delta_rows
        add_batch('padding_percent', padding_percent, 'Decode padding since previous poll', '%')
        total, used = values['pool_total'], values['pool_used']
        values['pool_percent'] = 100 * used / total if total and used is not None else None
        definitions['pool_percent'] = ('KV pool utilisation', '%', 'measurement')
        hits, misses = values['cache_hits'], values['cache_misses']
        values['cache_hit_percent'] = 100 * hits / (hits + misses) if hits is not None and misses is not None and hits + misses else None
        definitions['cache_hit_percent'] = ('Prefix cache hit rate since start', '%', 'measurement')
        self.previous = counters
        return values, definitions


class CompatibleInferenceSensors(VllmSensors):
    """Keep the installed vLLM discovery identities when DGPP is serving."""
    def __init__(self, device_id, hostname, machine_base, interval):
        from pathlib import Path
        super().__init__(device_id, hostname, machine_base, interval)
        self.dgpp = DgppSensors(device_id, hostname, machine_base, interval)
        path = Path.home()/'.local/share/spark1-ha/vllm-discovery-backup.json'
        self.legacy = json.loads(path.read_text())
        self.backend = None

    def compatible_values(self, raw, now):
        v, dgpp_definitions = self.dgpp.parse(raw, now)
        self.batch_definitions = {k: d for k, d in dgpp_definitions.items() if k.startswith("decode_batch_")}
        d = json.loads(raw)
        direct = {
            'num requests running': 'active', 'num requests waiting': 'queued',
            'KV cache utilisation': 'pool_percent',
            'prompt tokens total': 'prompt', 'prompt tokens total per second': 'prompt_rate',
            'generation tokens total': 'generated', 'generation tokens total per second': 'generated_rate',
            'prompt tokens by source total (source=local_compute)': 'computed',
            'prompt tokens by source total (source=local_cache_hit)': 'cache_saved',
            'prompt tokens cached total': 'cache_saved',
        }
        spec = d['scheduler'].get('spec_decode', {})
        if not isinstance(spec, dict):
            spec = {}
        def spec_count(value):
            return value if type(value) is int and value >= 0 else None
        speculative = {
            'spec decode ' + field.replace('_', ' '): spec_count(spec.get(field))
            for field in ('num_drafts_total', 'num_draft_tokens_total',
                          'num_accepted_tokens_total')
        }
        positions = spec.get('num_accepted_tokens_per_pos_total', [])
        if isinstance(positions, list):
            for position, value in enumerate(positions):
                speculative['spec decode num accepted tokens per pos total '
                            f'(position={position})'] = spec_count(value)
        p=d['prefix_cache']
        count=sum(p.get(k,0) for k in ('ttft_hit_count','ttft_miss_count'))
        ttft=(sum(p.get('ttft_'+kind+'_ms_avg',0)*p.get('ttft_'+kind+'_count',0)
                  for kind in ('hit','miss'))/count/1000) if count else None
        values={}
        for topic,cfg in self.legacy.items():
            name=cfg['name'];key=topic.split('/')[-2]
            value=v.get(direct.get(name,''))
            if name in speculative:value=speculative[name]
            if name=='time to first token seconds mean':value=ttft
            # Unknown metrics remain unknown. In particular, DGPP cache hits
            # count requests, whereas vLLM prefix hits/queries count tokens;
            # do not relabel these counters or fabricate latency histograms.
            values[key]=value
        values.update({k: v[k] for k in self.batch_definitions})
        return values

    def update(self, publish, prefix):
        def fetch(url):
            with urllib.request.urlopen(url,timeout=2) as response:raw=response.read(4*1024*1024+1)
            if len(raw)>4*1024*1024:raise ValueError('Metrics response too large')
            return raw.decode('utf-8')
        import time
        import urllib.request
        now=time.monotonic()
        try:
            raw=fetch(self.url)
            if self.backend!='vllm':self.previous.clear()
            values, definitions=self.parse(raw,now)
            backend='vllm'
        except (OSError,ValueError,UnicodeError):
            try:
                if self.backend!='dgpp':self.dgpp.previous.clear()
                values=self.compatible_values(fetch(self.dgpp.url),now)
                definitions=None;backend='dgpp'
            except (OSError,ValueError,UnicodeError,TypeError):
                self.previous.clear();self.dgpp.previous.clear()
                publish(self.base+'/availability','offline')
                return None
        if backend!=self.backend:self.force_discovery=True
        self.backend=backend
        if backend=='dgpp':
            for key, definition in self.batch_definitions.items():
                if self.force_discovery or self.known.get(key) != definition:
                    publish(f'{prefix}/sensor/{self.device_id}/{key}/config', json.dumps(self.config(key, definition)))
                    self.known[key] = definition
            if self.force_discovery:
                for topic,cfg in self.legacy.items():publish(topic,json.dumps(cfg))
        else:
            for key,definition in definitions.items():
                if self.force_discovery or self.known.get(key)!=definition:
                    publish(f'{prefix}/sensor/{self.device_id}/{key}/config',json.dumps(self.config(key,definition)))
                    self.known[key]=definition
        self.force_discovery=False
        publish(self.base+'/state',json.dumps(values,allow_nan=False),retain=False)
        publish(self.base+'/availability','online')
        return sum(value is not None for value in values.values())
