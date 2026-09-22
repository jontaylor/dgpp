#!/usr/bin/env python3
"""CPU PyTorch real-weight MiMo decoder-layer oracle, no checkpoint code import.

Usage: SNAPSHOT LAYER OUTPUT_JSON. Bounded to one layer, three sequential
positions, and the selected experts. Reads the quantized checkpoint directly.
"""
import json
import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from mimo_reference_sample import Reader


def generate(root, layer):
    torch.set_num_threads(8)
    r = Reader(root)
    cfg = json.loads((Path(root) / 'config.json').read_text())
    draft = layer >= cfg['num_hidden_layers']
    p = f"model.mtp.layers.{layer - cfg['num_hidden_layers']}." if draft else f'model.layers.{layer}.'
    sliding = draft or bool(cfg['hybrid_layer_pattern'][layer])
    # The config uses 1 for sliding; verify against the published head layout.
    kv = 8 if sliding else 4
    def tensor(name, dtype, shape):
        n = math.prod(shape)
        raw = bytearray(r.read(name, 0, n * torch.empty((), dtype=dtype).element_size()))
        return torch.frombuffer(raw, dtype=dtype).reshape(shape)
    def bf(name, shape):
        return tensor(name, torch.bfloat16, shape)
    def dense(name, rows, cols, qkv=False):
        w = tensor(name + '.weight', torch.float8_e4m3fn, (rows, cols))
        chunks = 4 if qkv else 1
        cr = rows // chunks
        scales = tensor(name + '.weight_scale_inv', torch.float32,
                        (chunks * math.ceil(cr / 128), math.ceil(cols / 128)))
        out = []
        for chunk in range(chunks):
            s = scales[chunk * math.ceil(cr / 128):(chunk + 1) * math.ceil(cr / 128)]
            expanded = s.repeat_interleave(128, 0)[:cr].repeat_interleave(128, 1)[:, :cols]
            out.append((w[chunk*cr:(chunk+1)*cr].float() * expanded).bfloat16())
        if qkv:
            parts = [x.split([3072, kv//4*192, kv//4*128], 0) for x in out]
            return torch.cat([torch.cat([x[kind] for x in parts]) for kind in range(3)])
        return torch.cat(out)
    def expert(e, proj):
        rows, cols = (4096, 2048) if proj == 'down' else (2048, 4096)
        base = p + f'mlp.experts.{e}.{proj}_proj'
        packed = tensor(base+'.weight', torch.uint8, (rows, cols//2))
        scale = tensor(base+'.weight_scale', torch.uint8, (rows, cols//32))
        lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
        codes = torch.stack([packed & 15, packed >> 4], -1).reshape(rows, cols).long()
        return (lut[codes] * torch.exp2(scale.float()-127).repeat_interleave(32, 1)).bfloat16()
    def linear(x, w):
        return F.linear(x.float(), w.float()).bfloat16()
    def norm(x, w):
        f = x.float()
        return (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)).bfloat16() * w
    x = torch.stack([((torch.arange(4096)*7+t*13)%101-50).float()/64 for t in range(3)]).bfloat16()
    if draft:
        embedding = torch.stack([((torch.arange(4096)*11+t*19)%97-48).float()/64 for t in range(3)]).bfloat16()
        fused_input = torch.cat([norm(embedding, bf(p+'enorm.weight',(4096,))),
                                norm(x,bf(p+'hnorm.weight',(4096,)))],-1)
        x = linear(fused_input,bf(p+'eh_proj.weight',(4096,8192)))
    y = norm(x, bf(p+'input_layernorm.weight', (4096,)))
    fused = linear(y, dense(p+'self_attn.qkv_proj', 12288+kv*320, 4096, True))
    q, k, v = fused.split([12288, kv*192, kv*128], -1)
    q, k = q.reshape(3,64,192), k.reshape(3,kv,192)
    v = v.reshape(3,kv,128) * .707
    freq = 1 / ((1e4 if sliding else 1e7) ** (torch.arange(0,64,2).float()/64))
    angle = torch.arange(3).float()[:,None] * freq[None,:]
    cs = torch.cat([angle,angle], -1).cos().bfloat16()[:,None,:]
    sn = torch.cat([angle,angle], -1).sin().bfloat16()[:,None,:]
    def rotate(t):
        a = t[...,:64]
        return torch.cat([a*cs+torch.cat([-a[...,32:],a[...,:32]],-1)*sn,t[...,64:]],-1)
    q,k = rotate(q),rotate(k)
    sink = bf(p+'self_attn.attention_sink_bias',(64,)) if sliding else None
    att = []
    for t in range(3):
        keys = k[:t+1].transpose(0,1).repeat_interleave(64//kv,0)
        vals = v[:t+1].transpose(0,1).repeat_interleave(64//kv,0)
        score = (q[t,:,None,:].float() @ keys.transpose(1,2).float()).bfloat16() * 192**-.5
        if sliding: score = torch.cat([score,sink[:,None,None]],-1)
        score = score - score.max(-1,keepdim=True).values
        prob = score.float().softmax(-1).bfloat16()
        if sliding: prob=prob[...,:-1]
        att.append((prob.float()@vals.float()).bfloat16().reshape(8192))
    x = x + linear(torch.stack(att),bf(p+'self_attn.o_proj.weight',(4096,8192)))
    y = norm(x,bf(p+('pre_mlp_layernorm.weight' if draft else 'post_attention_layernorm.weight'),(4096,)))
    routes = []
    if layer == 0 or draft:
        gate = linear(y,dense(p+'mlp.gate_proj',16384,4096))
        up = linear(y,dense(p+'mlp.up_proj',16384,4096))
        act = F.silu(gate.float()).bfloat16() * up
        result = x + linear(act,dense(p+'mlp.down_proj',4096,16384))
    else:
        score = F.linear(y.float(),bf(p+'mlp.gate.weight',(256,4096)).float()).sigmoid()
        bias = tensor(p+'mlp.gate.e_score_correction_bias',torch.float32,(256,))
        ids = (score+bias).argsort(dim=-1,descending=True,stable=True)[:,:8]
        weights = score.gather(1,ids)
        weights /= weights.sum(-1,keepdim=True)+1e-20
        acc = torch.zeros((3,4096))
        for e in sorted(set(ids.flatten().tolist())):
            token, slot = torch.where(ids == e)
            gate = linear(y[token],expert(e,'gate'))
            up = linear(y[token],expert(e,'up'))
            act = F.silu(gate.float()).bfloat16()*up
            out = linear(act,expert(e,'down'))
            acc[token] += out.float()*weights[token,slot,None]
        result = x + acc.bfloat16()
        routes = ids.tolist()
    if draft:
        result = norm(result,bf(p+'final_layernorm.weight',(4096,)))
    bits = result.contiguous().view(torch.int16).to(torch.int32).bitwise_and(65535).flatten().tolist()
    return dict(revision=Path(root).name,torch_version=torch.__version__,layer=layer,
                positions=[0,1,2],routes=routes,output_bf16=bits)


if __name__ == '__main__':
    Path(sys.argv[3]).write_text(json.dumps(generate(sys.argv[1],int(sys.argv[2]))))
