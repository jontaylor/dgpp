"""Prometheus-to-Home-Assistant sensors for the local inference server."""
import hashlib
import json
import math
import os
import re
import time
import urllib.request
from prometheus_client.parser import text_string_to_metric_families


class VllmSensors:
    def __init__(self, device_id, hostname, machine_base, interval):
        self.device_id = device_id + '_vllm'
        self.base = self.device_id + '/metrics'
        self.machine_base = machine_base
        self.hostname = hostname
        self.interval = interval
        self.url = os.getenv('VLLM_METRICS_URL', 'http://127.0.0.1:30001/metrics')
        self.known = {}
        self.previous = {}
        self.force_discovery = True

    @staticmethod
    def key(name, labels):
        identity = json.dumps([name, sorted(labels.items())], separators=(',', ':'))
        return re.sub('[^a-zA-Z0-9_]', '_', name) + '_' + hashlib.sha256(identity.encode()).hexdigest()[:16]

    def parse(self, exposition, now):
        values, definitions, counters = {}, {}, {}
        for family in text_string_to_metric_families(exposition):
            samples = {(s.name, tuple(sorted(s.labels.items()))): s.value for s in family.samples}
            for sample in family.samples:
                name, labels, value = sample.name, sample.labels, sample.value
                # Export the user-facing metrics, not Prometheus storage details.
                if name.endswith(('_created', '_bucket', '_count')) or name in (
                    'python_info', 'vllm:cache_config_info', 'vllm:engine_sleep_state',
                    'vllm:tool_call_parser_invocations_total', 'process_start_time_seconds',
                    'vllm:mm_cache_queries_total', 'vllm:mm_cache_hits_total',
                    'vllm:request_params_n_sum', 'vllm:iteration_tokens_total_sum',
                    'vllm:request_prefill_kv_computed_tokens_sum',
                    'http_request_size_bytes_sum', 'http_response_size_bytes_sum',
                ):
                    continue
                if family.type in ('histogram', 'summary') and not name.endswith('_sum'):
                    continue
                if not math.isfinite(value):
                    value = None
                key = self.key(name, labels)
                suffix = ', '.join(f'{k}={v}' for k, v in sorted(labels.items()) if k not in ('engine', 'model_name'))
                title = name.removeprefix('vllm:').replace('_', ' ')
                if suffix:
                    title += ' (' + suffix + ')'
                # Keep raw Prometheus units and distinguish counts from measured values.
                count = name.endswith(('_bucket', '_count')) or (family.type == 'counter')
                unit = None
                if not name.endswith(('_bucket', '_count')):
                    if name.endswith('_created'):
                        unit = 's'
                    elif 'bytes' in name or name.startswith('vllm:kv_offload_') and ('size' in name):
                        unit = 'B'
                    elif 'seconds' in name or name.endswith('_created') or name.startswith('vllm:kv_offload_') and '_time' in name:
                        unit = 's'
                state_class = 'total_increasing' if count else 'measurement'
                definitions[key] = (title, unit, state_class)
                values[key] = value
                if name in ('vllm:prompt_tokens_total', 'vllm:generation_tokens_total') and value is not None:
                    counters[key] = (now, value)
                    old = self.previous.get(key)
                    rate = None
                    if old and now > old[0] and value >= old[1]:
                        rate = (value - old[1]) / (now - old[0])
                    rate_key = key + '_rate'
                    values[rate_key] = rate
                    rate_unit = 'B/s' if unit == 'B' else 'tokens/s' if 'tokens' in name else '1/s'
                    definitions[rate_key] = (title + ' per second', rate_unit, 'measurement')
                if name.endswith('_sum') and family.type in ('histogram', 'summary'):
                    count_value = samples.get((name[:-4] + '_count', tuple(sorted(labels.items()))), 0)
                    mean_key = key + '_mean'
                    values[mean_key] = value / count_value if value is not None and count_value else None
                    definitions[mean_key] = (title.replace(' sum', '', 1) + ' mean', unit, 'measurement')
                if name == 'vllm:kv_cache_usage_perc':
                    values[key + '_percent'] = value * 100 if value is not None else None
                    definitions[key + '_percent'] = ('KV cache utilisation', '%', 'measurement')
        # Histograms are represented by their means; raw sums have little dashboard value.
        for key in list(values):
            if key + '_mean' in values or key + '_percent' in values:
                del values[key]
                del definitions[key]
        if not values:
            raise ValueError('Metrics endpoint returned no numeric samples')
        self.previous = counters
        return values, definitions

    def config(self, key, definition):
        name, unit, state_class = definition
        config = {
            'name': name, 'unique_id': self.device_id + '_' + key,
            'state_topic': self.base + '/state',
            'value_template': '{{ value_json.get("' + key + '") }}',
            'state_class': state_class,
            'availability': [
                {'topic': self.machine_base + '/availability'},
                {'topic': self.base + '/availability'},
            ],
            'availability_mode': 'all',
            'expire_after': max(30, int(self.interval * 4 + 10)),
            'device': {'identifiers': [self.device_id], 'name': self.hostname + '-vllm',
                       'manufacturer': 'vLLM', 'model': 'Prometheus inference metrics',
                       'via_device': self.device_id.removesuffix('_vllm')},
        }
        if unit:
            config['unit_of_measurement'] = unit
        return config

    def update(self, publish, prefix):
        try:
            with urllib.request.urlopen(self.url, timeout=2) as response:
                exposition = response.read(4 * 1024 * 1024 + 1)
            if len(exposition) > 4 * 1024 * 1024:
                raise ValueError('Metrics response exceeds 4 MiB')
            values, definitions = self.parse(exposition.decode('utf-8'), time.monotonic())
        except (OSError, ValueError, UnicodeError):
            self.previous.clear()
            publish(self.base + '/availability', 'offline')
            return None
        for key, definition in definitions.items():
            if self.force_discovery or self.known.get(key) != definition:
                publish(f'{prefix}/sensor/{self.device_id}/{key}/config', json.dumps(self.config(key, definition)))
                self.known[key] = definition
        self.force_discovery = False
        publish(self.base + '/state', json.dumps(values, allow_nan=False), retain=False)
        publish(self.base + '/availability', 'online')
        return len(values)
