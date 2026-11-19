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
parser.add_argument('--split-projections', action='store_true', help='Also export MX nibble gate/up and down entry points')
parser.add_argument('--pipeline-stages', type=int, choices=(2, 3, 4, 5),
                    help='Override the b12x asynchronous weight-copy pipeline depth for both projections')
parser.add_argument('--down-pipeline-stages', type=int, choices=(2, 3, 4, 5),
                    help='Override the copy depth for the down projection only')
parser.add_argument('--stream-weight-codes', action='store_true',
                    help='Give packed expert weight codes low L2 retention priority on both projections')
parser.add_argument('--weight-lookahead-tiles', type=int, choices=(0, 1, 2), default=0,
                    help='Prefetch packed weight lines into L2 this many K tiles beyond the asynchronous copy ring')
parser.add_argument('--warp-route-load', action='store_true',
                    help='Read the eight-entry DGPP route block directly in each warp before streaming weights')
parser.add_argument('--nibble-decoder', choices=('standard', 'packed-scale', 'native-fp4'), default='standard',
                    help='Use paired exponent arithmetic or SM121a native FP4 widening with packed MX scales')
parser.add_argument('--stage-weights-first', action='store_true',
                    help='Issue packed weight and scale copies before staging the routed activations')
parser.add_argument('--reuse-route-offset', action='store_true',
                    help='Resolve each activation row once per output tile and reuse its address along K')
parser.add_argument('--down-prefetch-tiles', type=int, choices=(0, 1, 4, 8), default=0,
                    help='At the gate/up barrier, prefetch the first K tiles of each block’s selected down projection')
args = parser.parse_args()
if args.split_projections and args.format != 'mxfp4-nibble':
 raise ValueError('split projection export requires mxfp4-nibble')
nibble_scales = args.format == 'mxfp4-nibble'
if args.nibble_decoder != 'standard' and not nibble_scales:
 raise ValueError('the packed scale decoders require mxfp4-nibble')
is_mx = args.format != 'nvfp4'
symbol = 'dgpp_b12x_mxfp4_fp32' if is_mx else 'dgpp_b12x_raw_fp32'
if nibble_scales: symbol = 'dgpp_b12x_mxfp4_nibble_fp32'
args.output.mkdir(parents=True, exist_ok=True)

import torch
import b12x.moe._shared.kernels.w4a16.kernel as kernel
source=Path(kernel.__file__).read_text();lines=source.splitlines(keepends=True);tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='W4A16GemmKernel')
class_begin=cls.lineno-1;body=lines[class_begin:cls.end_lineno]
# Method replacements are derived from the installed source, retained as their
# own source file for CuTe's source inspection. No installed files are changed.
edits=[]
if args.weight_lookahead_tiles or args.down_prefetch_tiles:
 path = args.output / 'prefetch_weight_line.py'
 path.write_text('''@dsl_user_op
def prefetch_weight_line(address: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [address.ir_value(loc=loc, ip=ip)], "prefetch.global.L2 [$0];", "l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
''')
 exec(compile(path.read_text(), str(path), 'exec'), kernel.__dict__)
if args.stream_weight_codes:
 # Only the packed B operand is consumed once per output tile. Keep the
 # existing policy on reused activations, scales and the route packet.
 path = args.output / 'stream_weight_codes.py'
 path.write_text('''@dsl_user_op
def copy_streaming_weight(smem_addr: Int32, gmem_addr: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [smem_addr.ir_value(loc=loc, ip=ip), gmem_addr.ir_value(loc=loc, ip=ip)],
        "{ .reg .b64 p; createpolicy.fractional.L2::evict_first.b64 p, 1.0; "
        "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, 16, p; }",
        "r,l", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
''')
 exec(compile(path.read_text(), str(path), 'exec'), kernel.__dict__)
for method in cls.body:
 if not isinstance(method,ast.FunctionDef):continue
 method_begin=method.body[0].lineno-1;method_end=method.end_lineno
 old=''.join(lines[method_begin:method_end]);new=old
 if method.name=='__init__' and args.down_pipeline_stages is not None:
  # This export applies routing weights after FC2. b12x passes this marker
  # only to the down projection. Set its depth before computing resources.
  anchor = '        self.stages = _w4a16_pipeline_stages('
  if old.count(anchor) != 1: raise RuntimeError('unsupported b12x pipeline setup')
  new = old.replace(anchor, f'        if mul_topk_weights:\n            pipeline_stages = {args.down_pipeline_stages}\n' + anchor)
 elif method.name=='_run_tile':
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
  if args.nibble_decoder != 'standard':
   # The nibble is an E8M0 exponent minus 112. Each byte holds two
   # independent exponents. Adding the bias in parallel cannot carry
   # between bytes. Native FP4 values omit the old reader's 2^-126 factor.
   if '_E8M0_K32_BF16_SCALE_BIAS' not in kernel.__dict__:
    raise RuntimeError('missing b12x E8M0 bias')
   bias = 112 + kernel._E8M0_K32_BF16_SCALE_BIAS
   if args.nibble_decoder == 'native-fp4':
    bias -= 126
   if not 0 < bias <= 239:
    raise RuntimeError('packed exponents exceed the BF16 normal range')
   start_read = new.index('            pair = ld_shared_u32(s_addr)')
   stop_read = new.index('        return q0, q1, q2, q3, s0, s1, s2, s3', start_read)
   new = new[:start_read] + f'''            pair = ld_shared_u32(s_addr)
            lo = (pair & Uint32(0x0f0f0f0f)) + Uint32({bias * 0x01010101})
            hi = ((pair >> Uint32(4)) & Uint32(0x0f0f0f0f)) + Uint32({bias * 0x01010101})
            s0 = (lo & Uint32(0x00ff00ff)) << Uint32(7)
            s1 = (lo >> Uint32(8) & Uint32(0x00ff00ff)) << Uint32(7)
            s2 = (hi & Uint32(0x00ff00ff)) << Uint32(7)
            s3 = (hi >> Uint32(8) & Uint32(0x00ff00ff)) << Uint32(7)
''' + new[stop_read:]
 elif method.name == '_scaled_dequant_b_fragment' and args.nibble_decoder == 'native-fp4':
  guard = 'self.native_nvfp4_scales and self.is_fp16'
  if old.count(guard) != 1: raise RuntimeError('unsupported b12x FP4 fragment')
  new = old.replace(guard, 'True')
 elif method.name=='__cache_key__':
  version = "dgpp_mx_nibble_fp32_v1" if nibble_scales else "dgpp_native_fp32_v4"
  if args.stream_weight_codes: version += "_stream_b"
  if args.weight_lookahead_tiles: version += f"_b_lookahead_{args.weight_lookahead_tiles}"
  if args.warp_route_load: version += "_warp_routes"
  if args.nibble_decoder != 'standard': version += "_" + args.nibble_decoder
  if args.stage_weights_first: version += "_weights_first"
  if args.reuse_route_offset: version += "_route_offset"
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
  if args.warp_route_load:
   # Packed blocks contain eight slot IDs including the end sentinel. Keep
   # the alternate schedulers' route conventions in the upstream function.
   new = '''        if cutlass.const_expr(self.moe_block_size == 8 and not self.dense_route_fast_path
                              and not self.direct_topk_routes and not self.single_token_route_fast_path):
            lane = tid % Int32(32)
            live = Int32(0)
            if lane < Int32(8):
                slot = packed_route_indices[route_block_idx * Int32(8) + lane].to(Int32)
                live = (slot < active_size_m * Int32(self.top_k)).to(Int32)
                if tid < Int32(8):
                    read_row = slot // Int32(self.top_k)
                    if cutlass.const_expr(self.route_major_a):
                        read_row = slot
                    st_shared_i32(smem_base + Int32(self.sh_route_off * 16) + lane * Int32(4), slot)
                    st_shared_i32(smem_base + Int32(self.sh_rd_route_off * 16) + lane * Int32(4), read_row)
            count = cute.arch.warp_redux_sync(live, "add")
            cute.arch.sync_threads()
            return count
''' + new
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
 if method.name == '_tile_common_prologue' and args.reuse_route_offset:
  # Both compiled projections assign at most one activation vector to a lane
  # per K tile. Bake the stable route into that lane's initial vector address.
  anchor = '        ) = self._tile_stream_offsets(tid, expert_idx, output_n_tile)\n'
  if new.count(anchor) != 1: raise RuntimeError('unsupported b12x tile stream offsets')
  new = new.replace(anchor, anchor + '''        if cutlass.const_expr(self.a_sh_wr_iters == 1 and not self.has_k_tile_tail):
            if a_gl_rd_row < block_valid_rows:
                source_row = ld_shared_i32_relaxed(
                    smem_base + Int32(self.sh_rd_route_off * 16) + a_gl_rd_row * Int32(4))
                a_gl_rd_col0 += source_row * a_gl_stride
''')
 if method.name == '_stage_k_tile_async' and args.reuse_route_offset:
  guard = '            if row < Int32(self.moe_block_size):'
  if new.count(guard) != 1: raise RuntimeError('unsupported b12x activation staging')
  start = new.index(guard)
  end = new.index('            a_int4 =', start)
  fallback = new[start:end]
  import textwrap
  new = (new[:start] + '            if cutlass.const_expr(self.a_sh_wr_iters != 1 or self.has_k_tile_tail):\n' +
         textwrap.indent(fallback, '    ') + new[end:])
 if method.name == '_stage_k_tile_async' and args.stage_weights_first:
  # A, B and scales occupy distinct regions of the same asynchronous group.
  # No consumer can see any of them until the existing commit/wait.
  start = new.index('        for i in cutlass.range_constexpr(self.a_sh_wr_iters):')
  end = new.index('        if cutlass.const_expr(self.weight_layout_block):', start)
  commit = '        cute.arch.cp_async_commit_group()'
  if new.count(commit) != 1: raise RuntimeError('unsupported b12x staging group')
  activation = new[start:end]
  new = (new[:start] + new[end:]).replace(commit, activation + commit)
 if method.name == '_prefetch_lookahead_tile' and args.weight_lookahead_tiles:
  marker = '        if fetch_tile < k_tiles:'
  if new.count(marker) != 1: raise RuntimeError('unsupported b12x weight lookahead')
  hint = f'''        ahead = fetch_tile + Int32({args.weight_lookahead_tiles})
        if ahead < k_tiles and (tid % Int32(8)) == Int32(0):
            # Each participating lane covers one aligned 128-byte packed B line.
            for i in cutlass.range_constexpr(self.b_sh_wr_iters):
                block = (reduce_k_tile + ahead) * Int32(self.cta_k_blocks)
                row = Int32(i * (self.cta_threads // self.b_sh_stride))
                offset = b_gl_rd_base + (block + row) * b_gl_stride
                prefetch_weight_line(get_ptr_as_int64(b_i32_flat, offset * Int32(4)))
'''
  new = new.replace(marker, hint + marker)
 if method.name == '_stage_k_tile_async' and args.stream_weight_codes:
  packed = 'if cutlass.const_expr(self.weight_layout == "packed"):\n                cp_async4_shared_global('
  if new.count(packed) != 1: raise RuntimeError('unsupported b12x packed-weight copy')
  new = new.replace(packed, packed.replace('cp_async4_shared_global', 'copy_streaming_weight'))
 if new!=old:edits.append((method_begin-class_begin,method_end-class_begin,new))
for a,b,new in sorted(edits,reverse=True):body[a:b]=[new]
path=args.output / 'raw_fp32_kernel_class.py'
path.write_text(''.join(body)+'\n')
if nibble_scales:
 generated = ast.parse(path.read_text()).body[0]
 stage = next(n for n in generated.body if getattr(n, 'name', '') == '_stage_k_tile_async')
 if 'cp_async_u64_shared_global' not in ast.unparse(stage).split('s_src_int4 = scales_expert_off')[-1]:
  raise RuntimeError("packed MX staging was not installed in the exported method")
exec(compile(path.read_text(),str(path),'exec'),kernel.__dict__)
if args.down_prefetch_tiles:
 fused = next(n for n in tree.body if getattr(n, 'name', '') == 'W4A16FusedMoeKernel')
 body = ''.join(lines[fused.lineno-1:fused.end_lineno])
 anchor = '                fc1_emit_tile,\n            )\n            self._grid_barrier(locks_i32_flat, tid, grid_x)'
 if body.count(anchor) != 1: raise RuntimeError('unsupported fused gate/up barrier')
 # Packed weights store all N channels for each K16 before the next K16.
 # The first FC2 wave traverses output tiles within each packed route block.
 # Ignore negative expert IDs, including the masked blocks on a paged miss.
 hint = f'''            tile = cta % Int32(self.fc2.n_tiles)
            route_block = cta // Int32(self.fc2.n_tiles)
            if route_block < packed_route_count[Int32(0)].to(Int32) // Int32(self.moe_block_size):
                expert = block_expert_ids[route_block].to(Int32)
                if expert >= Int32(0):
                    table = cute.recast_tensor(w2_i32_flat, cutlass.Int64)
                    base = table[expert * Int32(3)]
                    lines_per_strip = self.fc2.tile_n * 8 // 128
                    lines = {args.down_prefetch_tiles} * self.fc2.cta_k_blocks * lines_per_strip
                    for i in cutlass.range_constexpr((lines + self.cta_threads - 1) // self.cta_threads):
                        line = tid + Int32(i * self.cta_threads)
                        if line < Int32(lines):
                            strip = line // Int32(lines_per_strip)
                            within = line % Int32(lines_per_strip)
                            offset = strip * Int32(self.hidden_size * 8) + tile * Int32(self.fc2.tile_n * 8) + within * Int32(128)
                            prefetch_weight_line(base + Int64(offset))
'''
 body = body.replace(anchor, anchor.replace('            self._grid_barrier', hint + '            self._grid_barrier'))
 key = 'def __cache_key__(self) -> tuple[object, ...]:\n        return ('
 if body.count(key) != 1: raise RuntimeError('unsupported fused b12x cache key')
 body = body.replace(key, key + f'"dgpp_down_prefetch_{args.down_prefetch_tiles}",')
 path = args.output / 'down_prefetch_fused_class.py'
 path.write_text(body)
 exec(compile(body,str(path),'exec'),kernel.__dict__)
fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compile_w4a16_fused_moe')
body=''.join(lines[fun.lineno-1:fun.end_lineno])
old='cutlass.Float32 if kernel.prefill_fused_sum_fp32 else cutlass_dtype,'
assert body.count(old)==1
body=body.replace(old,'cutlass.Float32,')
if args.down_pipeline_stages is not None:
 # The forced tile resource check must evaluate the same depth as its GEMM.
 anchor = '                pipeline_stages=pipeline_stages,'
 start = body.index('        for name, forced_pn, forced_pk, forced_tn, forced_tk in (')
 end = body.index('    kernel = W4A16FusedMoeKernel(', start)
 check = body[start:end]
 if check.count(anchor) != 1: raise RuntimeError('unsupported b12x forced tile check')
 body = body[:start] + check.replace(anchor, f'                pipeline_stages={args.down_pipeline_stages} if name == "fc2" else pipeline_stages,') + body[end:]
path=args.output / 'raw_fp32_compile.py';path.write_text(body+'\n')
exec(compile(path.read_text(),str(path),'exec'),kernel.__dict__)
props=torch.cuda.get_device_properties(0)
def export(name):
 compiled=kernel.compile_w4a16_fused_moe(size_m=64,hidden_size=5120,intermediate_size=1152,num_experts=384,top_k=6,
 activation='silu',apply_router_weight_on_input=False,zero_fc2_output=False,moe_block_size=8,max_m_blocks=384,
 element_dtype='bf16',sms=48,max_shared_mem=props.shared_memory_per_block_optin,swiglu_limit=10.,
 weight_layout='packed',scale_format='e8m0_k32' if is_mx else 'e4m3_k16',w13_layout='packed',force_tile_config=(128,128,64,256),
 **({'pipeline_stages': args.pipeline_stages} if args.pipeline_stages is not None else {}))
 compiled.compiled.export_to_c(str(args.output), name)
export(symbol)
with (args.output / (symbol + '.h')).open('a') as header:
 header.write('\n#define DGPP_B12X_CAPACITY 64\n')
if args.split_projections:
 import textwrap
 fused = next(n for n in tree.body if getattr(n, 'name', '') == 'W4A16FusedMoeKernel')
 method = next(n for n in fused.body if getattr(n, 'name', '') == '_moe_body')
 body = ''.join(lines[fused.lineno-1:fused.end_lineno])
 begin = body.index('        if cutlass.const_expr(self.activation_is_gated):', body.index('    def _moe_body('))
 down = body.index('        fc2_store = fc2_bf16_flat', begin)
 end = sum(len(line) for line in lines[fused.lineno-1:method.end_lineno])
 # layer_idx is unused by the no-amax, single-layer DGPP specialization.
 # The separate optional object leaves ordinary resident replay untouched.
 body = (body[:begin] + '        if layer_idx == Int32(1):\n' +
         textwrap.indent(body[begin:down], '    ') + '        else:\n' +
         textwrap.indent(body[down:end], '    ') + body[end:])
 # The compiler caches fused kernels by this property, including on disk.
 key = 'def __cache_key__(self) -> tuple[object, ...]:\n        return ('
 if body.count(key) != 1:
  raise RuntimeError('unsupported fused b12x cache key')
 body = body.replace(key, key + '"dgpp_mx_projection_v1",')
 path = args.output / 'split_fused_class.py'
 path.write_text(body)
 exec(compile(body,str(path),'exec'),kernel.__dict__)
 export(symbol + '_projection')
from cutlass.cute.export.aot_config import get_libdir
shutil.copy2(Path(get_libdir()) / 'libcuda_dialect_runtime_static.a', args.output)

print('RAW_FP32_AOT_EXPORTED',flush=True)
