#!/usr/bin/env python3
"""Generate small MiMo eager-attention goldens with CPU PyTorch.

Mirrors the published eager attention and rotary arithmetic, without importing
or executing checkpoint code. No weights or GPU are needed. Output goes to
stdout; pin the generator's torch version in the emitted fixture.
"""
import json
import torch


def bits(x):
    return x.contiguous().view(torch.int16).to(torch.int32).bitwise_and(65535).flatten().tolist()


def generate():
    torch.set_num_threads(1)
    result = {'torch_version': torch.__version__, 'q_heads': 2, 'kv_heads': 1, 'cases': [], 'rotary_cases': []}
    positions = [0, 1, 126, 127, 128, 129, 255, 256]
    # Integer arithmetic makes the projection inputs exactly representable
    # in BF16, independent of host RNG/library implementation.
    width = 2 * 192 + 192 + 128
    indices = torch.arange(width, dtype=torch.int64)
    inputs = torch.stack([((indices * 7 + p * 13) % 101 - 50).float() / 64 for p in range(257)]).bfloat16()
    for window, theta in [(0, 1e7), (128, 1e4)]:
        frequency = 1 / (theta ** (torch.arange(0, 64, 2).float() / 64))
        q, k, v = torch.split(inputs, [384, 192, 128], dim=-1)
        q, k = q.reshape(257, 2, 192), k.reshape(257, 1, 192)
        v = v.reshape(257, 1, 128) * 0.707
        angles = torch.arange(257).float().unsqueeze(1) * frequency.unsqueeze(0)
        cos = torch.cat([angles, angles], -1).cos().bfloat16().unsqueeze(1)
        sin = torch.cat([angles, angles], -1).sin().bfloat16().unsqueeze(1)
        def rotate(x):
            partial = x[..., :64]
            other = torch.cat([-partial[..., 32:], partial[..., :32]], -1)
            return torch.cat([partial * cos + other * sin, x[..., 64:]], -1)
        q, k = rotate(q), rotate(k)
        for pos in [8191, 32767, 1048575]:
            raw = (((indices * 7 + pos * 13) % 101 - 50).float() / 64).bfloat16()
            angle = float(pos) * frequency
            c = torch.cat([angle, angle]).cos().bfloat16()
            s = torch.cat([angle, angle]).sin().bfloat16()
            def high_rotate(x):
                partial = x[..., :64]
                partner = torch.cat([-partial[..., 32:], partial[..., :32]], -1)
                return torch.cat([partial * c + partner * s, x[..., 64:]], -1)
            result['rotary_cases'].append({'window': window, 'position': pos,
                                          'q': bits(high_rotate(raw[:384].reshape(2, 192))),
                                          'k': bits(high_rotate(raw[384:576].reshape(1, 192)))})
        sinks = torch.tensor([0.25, -0.5], dtype=torch.bfloat16) if window else None
        for pos in positions:
            start = max(0, pos - window + 1) if window else 0
            keys = k[start:pos+1].transpose(0, 1).repeat_interleave(2, dim=0)
            values = v[start:pos+1].transpose(0, 1).repeat_interleave(2, dim=0)
            scores = (q[pos].unsqueeze(1) @ keys.transpose(1, 2)) * (192 ** -0.5)
            if sinks is not None:
                scores = torch.cat([scores, sinks[:, None, None]], -1)
            scores = scores - scores.max(dim=-1, keepdim=True).values
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).bfloat16()
            if sinks is not None:
                probs = probs[..., :-1]
            output = probs @ values
            result['cases'].append({'window': window, 'position': pos,
                                    'q': bits(q[pos]), 'k': bits(k[pos]), 'v': bits(v[pos]),
                                    'output': bits(output)})
    return result


if __name__ == '__main__':
    print(json.dumps(generate(), separators=(',', ':')))
