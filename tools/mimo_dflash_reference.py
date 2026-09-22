#!/usr/bin/env python3
"""Independent CPU equation oracle for mimo_dflash_check dumps.

Matches config + recipe vLLM semantics (Q/K norm, partial RoPE, sink, V scale).
The published dflash.py omits sink and value scale; it is not this oracle.
Requires torch only; maps the actual BF16 drafter file, no transformers needed.
"""
import argparse
import json
import mmap
import struct
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('dump');args=p.parse_args()
    torch.set_num_threads(8)
    f=open(Path(args.checkpoint)/'dflash/dflash_draft_model.safetensors','rb');m=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_COPY)
    size=struct.unpack('<Q',m[:8])[0];header=json.loads(m[8:8+size]);base=8+size
    def weight(name):
        t=header[name];assert t['dtype']=='BF16'
        lo,hi=t['data_offsets'];return torch.frombuffer(m,dtype=torch.bfloat16,count=(hi-lo)//2,offset=base+lo).reshape(t['shape'])
    root=Path(args.dump)
    def read(name,shape):
        return torch.frombuffer(bytearray((root/name).read_bytes()),dtype=torch.bfloat16).reshape(shape)
    def norm(x,w):
        v=x.float();return (v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-6)).bfloat16()*w
    def rope(x,positions):
        angle=positions.float()[:,None]*10000.**(-torch.arange(32).float()/32)
        c=angle.cos().bfloat16()[:,None,:];s=angle.sin().bfloat16()[:,None,:]
        a,b=x[...,:32],x[...,32:64]
        return torch.cat((a*c-b*s,b*c+a*s,x[...,64:]),-1)
    features=read('features.bf16',(4,20480))
    ctx=norm(torch.nn.functional.linear(features,weight('fc.weight')),weight('hidden_norm.weight')).reshape(2,2,4096)
    mask=read('mask.bf16',(4096,))
    x=mask.expand(2,8,4096).clone();x[0,0]=read('embedding.bf16',(4096,));x[1,0]=read('embedding2.bf16',(4096,))
    for layer in range(5):
        prefix=f'layers.{layer}.'
        def w(name):return weight(prefix+name)
        n=norm(x,w('input_layernorm.weight'))
        def proj(z,kind):return torch.nn.functional.linear(z,w('self_attn.'+kind+'_proj.weight'))
        q=proj(n,'q').reshape(2,8,64,128)
        kc=proj(ctx,'k').reshape(2,2,8,128);vc=proj(ctx,'v').reshape(2,2,8,128)
        k=proj(n,'k').reshape(2,8,8,128);v=proj(n,'v').reshape(2,8,8,128)
        q=norm(q,w('self_attn.q_norm.weight'));kc=norm(kc,w('self_attn.k_norm.weight'));k=norm(k,w('self_attn.k_norm.weight'))
        q=torch.stack([rope(y,torch.arange(2,10)) for y in q]);k=torch.stack([rope(y,torch.arange(2,10)) for y in k]);kc=torch.stack([rope(y,torch.arange(2)) for y in kc])
        k=torch.cat((kc,k),1).repeat_interleave(8,2).transpose(1,2).float()
        v=(torch.cat((vc,v),1)*0.612).repeat_interleave(8,2).transpose(1,2).float()
        scores=q.transpose(1,2).float()@k.transpose(-1,-2)*128**-0.5
        sinks=w('self_attn.attention_sink_bias').float()[None,:,None,None].expand(2,64,8,1)
        probability=torch.softmax(torch.cat((scores,sinks),-1),-1)[...,:10]
        attn=(probability@v).transpose(1,2).reshape(2,8,8192).bfloat16()
        x=x+torch.nn.functional.linear(attn,w('self_attn.o_proj.weight'))
        n=norm(x,w('post_attention_layernorm.weight'))
        gate=torch.nn.functional.linear(n,w('mlp.gate_proj.weight'));up=torch.nn.functional.linear(n,w('mlp.up_proj.weight'))
        # Native SwiGLU accumulates activation and product before one BF16 cast.
        act=(torch.nn.functional.silu(gate.float())*up.float()).bfloat16()
        x=x+torch.nn.functional.linear(act,w('mlp.down_proj.weight'))
    expected=norm(x,weight('norm.weight')).reshape(16,4096).float();actual=read('hidden.bf16',(16,4096)).float()
    error=(actual-expected).abs();cos=torch.nn.functional.cosine_similarity(actual,expected,dim=-1)
    result={'max_abs':error.max().item(),'rms':error.square().mean().sqrt().item(),'min_cosine':cos.min().item(), 'semantics':'config+vLLM sink and V scale; float softmax/PV'}
    print(json.dumps(result,indent=2));assert result['min_cosine']>0.999 and result['rms']<0.08,result

if __name__=='__main__':main()
