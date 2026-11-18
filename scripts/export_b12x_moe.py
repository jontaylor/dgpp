"""Export DGPP's optional TP2 GB10 native FP4 W4A16 specialization.

Requires FlashInfer 0.7.4's b12x, Cutlass DSL 4.8.1 with CUDA 13 and an
SM121 GPU during generation. Produces an AOT C header, ARM object and runtime;
Python is not used by the server. The specialization has H5120, I1152,
384 experts, top6 and capacity64, with BF16 activations and FP32 slot output.

The generated code derives from FlashInfer/b12x's Apache-2.0 W4A16 kernel.
Source transformations are confined to this output directory; installed
package files are never modified.
"""
import argparse, ast, shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--format', choices=('nvfp4', 'mxfp4', 'mxfp4-nibble'), default='nvfp4')
args = parser.parse_args()
nibble_scales = args.format == 'mxfp4-nibble'
is_mx = args.format != 'nvfp4'
symbol = 'dgpp_b12x_mxfp4_fp32' if is_mx else 'dgpp_b12x_raw_fp32'
if nibble_scales: symbol = 'dgpp_b12x_mxfp4_nibble_fp32'
args.output.mkdir(parents=True, exist_ok=True)

import torch
import b12x.moe._shared.kernels.w4a16.kernel as kernel
source=Path(kernel.__file__).read_text();lines=source.splitlines(keepends=True);tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='W4A16GemmKernel')
start=cls.lineno-1;body=lines[start:cls.end_lineno]
# Method replacements are derived from the installed source, retained as their
# own source file for CuTe's source inspection. No installed files are changed.
edits=[]
for method in cls.body:
 if not isinstance(method,ast.FunctionDef):continue
 begin=method.body[0].lineno-1;end=method.end_lineno
 old=''.join(lines[begin:end]);new=old
 if method.name=='_run_tile':
  prefix='''        table = cute.recast_tensor(b_i32_flat, cutlass.Int64)
        payload_addr = table[expert_idx * Int32(3)]
        scales_addr = table[expert_idx * Int32(3) + Int32(1)]
        global_addr = table[expert_idx * Int32(3) + Int32(2)]
        b_i32_flat = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, payload_addr, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((self.size_n * self.size_k // 8,), stride=(1,)))
        scales_i32_flat = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, scales_addr, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((self.size_n * self.size_k // 64,), stride=(1,)))
        global_scale = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, global_addr, cute.AddressSpace.gmem, assumed_align=4),
            cute.make_layout((2,), stride=(1,)))
        expert_idx = Int32(0)
'''
  if is_mx: prefix = prefix.replace("self.size_n * self.size_k // 64", "self.size_n * self.size_k // 128")
  if nibble_scales: prefix = prefix.replace("// 128", "// 256")
  new=prefix+old
 elif method.name=='_stage_k_tile_async' and nibble_scales:
  # Keep the original staging ring and async-copy groups. Each thread now
  # stages eight bytes and the MMA consumer expands a word in registers.
  a = 'Int32(self.sh_s_off) + pipe * Int32(self.s_sh_stage) + tid,'
  b = 'Int32(self.sh_s_off) + pipe * Int32(self.s_sh_stage),'
  scale_start = old.index('                s_src_int4 = scales_expert_off')
  prefix, tail = old[:scale_start], old[scale_start:]
  if tail.count('cp_async4_shared_global(') != 2 or tail.count(a) != 1:
   raise RuntimeError("unsupported b12x scale staging")
  tail = tail.replace(a, b).replace('                if cutlass.const_expr(self.has_logical_tail):',
      '                s_dst = s_dst + tid * Int32(8)\n                if cutlass.const_expr(self.has_logical_tail):')
  tail = tail.replace('cp_async4_shared_global(', 'cp_async_u64_shared_global(')
  tail = tail.replace('s_src_int4 * Int32(4)', 's_src_int4 * Int32(2)')
  tail = tail.replace('st_shared_v4_u32(\n                            s_dst, Uint32(0), Uint32(0), Uint32(0), Uint32(0)\n                        )',
      'st_shared_u32(s_dst, Uint32(0))\n                        st_shared_u32(s_dst + Int32(4), Uint32(0))')
  if 'st_shared_v4_u32' in tail: raise RuntimeError("unsupported scale-tail fill")
  new = prefix + tail
 elif method.name=='_load_b_scale_registers' and nibble_scales:
  addr = '(s_sh_rd + scale_group_id * Int32(2 * self.s_sh_stride)) * Int32(8)'
  read = 's_pack0, s_pack1 = ld_shared_v2_u32(s_addr)'
  if old.count(addr) != 1 or old.count(read) != 1: raise RuntimeError("unsupported b12x scale reader")
  new = old.replace(addr, addr.replace('Int32(8)', 'Int32(4)')).replace(read, '''pair = ld_shared_u32(s_addr)
            s_pack0 = (pair & Uint32(0x0f0f0f0f)) | Uint32(0x70707070)
            s_pack1 = ((pair >> Uint32(4)) & Uint32(0x0f0f0f0f)) | Uint32(0x70707070)''')
 elif method.name=='__cache_key__':
  version = "dgpp_mx_nibble_fp32_v1" if nibble_scales else "dgpp_native_fp32_v4"
  new = old.replace('return (', f'return ("{version}",', 1)
 elif method.name=='_dequant_scale_x4_to_elem2x2' and not is_mx:
  new='''        s0, s1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(packed)
        lift = Uint32(0x7B007B00)
        # Packed scale words interleave even/odd channels; native widening
        # yields adjacent pairs. Match the original packed decoder's lanes.
        even = (s0 & Uint32(0xFFFF)) | (s1 << Uint32(16))
        odd = (s0 >> Uint32(16)) | (s1 & Uint32(0xFFFF0000))
        return bfloat2_mul(even, lift), bfloat2_mul(odd, lift)
'''
 elif method.name=='_tile_common_prologue' and is_mx:
  # MX has no per-matrix global. Retain b12x's native E8M0 compensation.
  new=old.replace('global_scale[expert_idx].to(cutlass.Float32)', 'cutlass.Float32(1.0)')
 elif method.name=='_tile_common_prologue':
  new=old.replace('        global_scale_f32 = global_scale[expert_idx].to(cutlass.Float32)', '''        global_idx = Int32(0)
        if cutlass.const_expr(not self.mul_topk_weights):
            global_idx = (output_n_tile * Int32(self.tile_n)) // Int32(self.size_n // 2)
        global_scale_f32 = cutlass.Float32(128.0) / global_scale[global_idx].to(cutlass.Float32)''')
 elif method.name=='_read_moe_block_data':
  # The existing DGPP accumulation applies FP32 routing weights after the
  # down dot. No BF16 routing multiplier is read by the custom FC2 store.
  new=old.replace('if cutlass.const_expr(self.mul_topk_weights):','if cutlass.const_expr(False):')
 elif method.name=='_store_tile_m8':
  prefix='''        if cutlass.const_expr(self.mul_topk_weights):
            dst = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, get_ptr_as_int64(c_bf16_flat, Int64(0)),
                              cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((self.size_m * (self.top_k + 1) * self.size_n,), stride=(1,)))
            lane = tid % Int32(32)
            warp = tid // Int32(32)
            if warp < Int32(self.tb_n_warps):
                for jj in cutlass.range_constexpr(4):
                    for part in cutlass.range_constexpr(4):
                        row = (lane % Int32(4)) * Int32(2) + Int32(part % 2)
                        col = output_n_tile * Int32(self.tile_n) + warp * Int32(64) + Int32(jj * 16) + lane // Int32(4) + Int32((part // 2) * 8)
                        if row < block_valid_rows and col < Int32(self.size_n):
                            route = ld_shared_i32_relaxed(smem_base + Int32(self.sh_route_off * 16) + row * Int32(4))
                            # FC2 internally has top_k=1; output still uses target top6.
                            slot = (route // Int32(6)) * Int32(7) + route % Int32(6)
                            dst[Int64(slot) * Int64(self.size_n) + Int64(col)] = acc[(jj * 4 + part) // _SCALAR_ACC_FRAGMENT_WIDTH][(jj * 4 + part) % _SCALAR_ACC_FRAGMENT_WIDTH] * global_scale_f32
            cute.arch.sync_threads()
            return
'''
  new=prefix+old
 if new!=old:edits.append((begin-start,end-start,new))
for a,b,new in sorted(edits,reverse=True):body[a:b]=[new]
path=args.output / 'raw_fp32_kernel_class.py'
path.write_text(''.join(body)+'\n')
if nibble_scales:
 generated = ast.parse(path.read_text()).body[0]
 stage = next(n for n in generated.body if getattr(n, 'name', '') == '_stage_k_tile_async')
 if 'cp_async_u64_shared_global' not in ast.unparse(stage).split('s_src_int4 = scales_expert_off')[-1]:
  raise RuntimeError("packed MX staging was not installed in the exported method")
exec(compile(path.read_text(),str(path),'exec'),kernel.__dict__)
fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compile_w4a16_fused_moe')
body=''.join(lines[fun.lineno-1:fun.end_lineno])
old='cutlass.Float32 if kernel.prefill_fused_sum_fp32 else cutlass_dtype,'
assert body.count(old)==1
body=body.replace(old,'cutlass.Float32,')
path=args.output / 'raw_fp32_compile.py';path.write_text(body+'\n')
exec(compile(path.read_text(),str(path),'exec'),kernel.__dict__)
props=torch.cuda.get_device_properties(0)
compiled=kernel.compile_w4a16_fused_moe(size_m=64,hidden_size=5120,intermediate_size=1152,num_experts=384,top_k=6,
 activation='silu',apply_router_weight_on_input=False,zero_fc2_output=False,moe_block_size=8,max_m_blocks=384,
 element_dtype='bf16',sms=48,max_shared_mem=props.shared_memory_per_block_optin,swiglu_limit=10.,
 weight_layout='packed',scale_format='e8m0_k32' if is_mx else 'e4m3_k16',w13_layout='packed',force_tile_config=(128,128,64,256))
compiled.compiled.export_to_c(str(args.output), symbol)
with (args.output / (symbol + '.h')).open('a') as header:
 header.write('\n#define DGPP_B12X_CAPACITY 64\n')
from cutlass.cute.export.aot_config import get_libdir
shutil.copy2(Path(get_libdir()) / 'libcuda_dialect_runtime_static.a', args.output)

print('RAW_FP32_AOT_EXPORTED',flush=True)
