# Exact FP8 load conversion experiment

Base: `b4636ed`. The original FP8 cache mode remains available. New CMake option `DGPP_MIMO_FP8_KV_FAST_LOAD` defaults **OFF**. With it ON, E4M3 loads still expand through CUDA's native E4M3-to-half and half-to-float conversion, but omit redundant FP32-to-BF16 rounding: every finite E4M3 value is exactly representable in BF16. Signed zeros and the legacy signed canonical-NaN policy are retained. The BF16 cache route, cache/snapshot storage, audit conversion/accumulation, attention algorithms and runtime format selection remain unchanged.

This is a compile-time experiment, not an environment variable. Configure separate build directories so original and candidate binaries remain available:

```
cmake --preset spark-cross -B build-fp8-original -DDGPP_MIMO_FP8_KV_FAST_LOAD=OFF
cmake --build build-fp8-original --parallel 2 --target dgpp_serve_app mimo_fp8_load_test mimo_fp8_test mimo_attn_test mimo_snapshot_test
cmake --preset spark-cross -B build-fp8-exact -DDGPP_MIMO_FP8_KV_FAST_LOAD=ON
cmake --build build-fp8-exact --parallel 2 --target dgpp_serve_app mimo_fp8_load_test mimo_fp8_test mimo_attn_test mimo_snapshot_test
```

Record each binary SHA256, Git revision and CMake option. Startup reports `FP8 cache load build=original-RNE` or `exact-finite-truncation`, with the flag value, on each rank. That report calls the linked kernel library, avoiding a host-header-only assumption. `DGPP_MIMO_FP8_KV=1` is still required to enable FP8 storage. The existing `DGPP_MIMO_FP8_KV_AUDIT=1` remains usable independently; keep it off for timed measurements.

## Compiled evidence (CUDA 13.0.88, SM121a)

Evidence is **PTX and ptxas resource reporting**, not SASS or measured speedup. `cuobjdump`/`nvdisasm` are unavailable in the cross image. Regenerate with `bash tools/mimo_fp8_load_codegen.sh /absolute/experiment/output` inside that image.

Original scalar expansion emits `cvt.rn.f16x2.e4m3x2` then `cvt.f32.f16`. Its finite BF16 conversion adds a shift, mask, two additions and high-half extraction; its NaN handling introduces control flow. The candidate keeps both native expansion instructions and extracts the exact high half; NaN canonicalization compiles to a predicated select. The original and final sign-preserving candidate microkernels both use 12 physical registers and no spills. A discarded raw-NaN shortcut used fewer registers but is not the implemented candidate.

Whole-attention ptxas registers, original → candidate:

| Kernel | Registers |
|---|---:|
| Scalar attention | 39 → 39 |
| Decode score tiles | 54 → 54 |
| Prefill score tiles | 42 → 42 |
| Ordinary PV tiles | 68 → 68 |
| Wide PV | 52 → 52 |
| Fused-probability wide PV | 53 → 52 |
| Bounded prefill | 62 → 62 |
| Online prefill | 72 → 72 |
| Split online decode | 70 → 70 |
| Audit reduction | 38 → 38 |
| Append | 32 → 32 |

All kernels report zero spill loads/stores in both builds; append's 32-byte stack frame is unchanged. Thus reduced conversion instructions do not generally reduce register pressure or establish an occupancy improvement.

The runtime BF16/FP8 format branch remains. For example, decode-score PTX loads the format from the shape and branches at the tile loop into distinct BF16/FP8 load paths. The compiler hoists some checks; it is inaccurate to assert one branch per element everywhere. Compile-time cache-type kernel specialization could investigate remaining generalized-kernel overhead, but is outside this candidate. Neither exact output equivalence nor the unchanged BF16 source path guarantees baseline-neutral BF16 performance after recompilation. Parent must measure FP8-off and FP8-on separately against retained controls.

Direct `cvt.rn.bf16x2.e4m3x2` was rejected by CUDA13 ptxas for SM121a (`Unexpected instruction types specified for 'cvt'`); removing the rounding modifier also fails. No unsupported instruction, paired load/vectorization, lookup table, or new attention algorithm is included.

## Correctness and hardware gates

Host test exhausts all 256 E4M3 codes plus positive/negative FP32 NaN payloads against the legacy BF16 converter. GPU `mimo_fp8_load_test` compares original, exact and selected-build conversion for all 256 raw codes, including both NaN codes and signed zero; finite outputs additionally match the independent CPU representation. It checks BF16 bit passthrough, graph replay with reordered codes, and linked-library/probe build provenance. GPU NaN sign must match the original CUDA decoder's result (CUDA may canonicalize an FP8 NaN during expansion); the optimization does not reinterpret the raw FP8 sign.

Run both builds' `mimo_fp8_load_test` and existing `mimo_fp8_test` (including audit and combined online/bounded coverage), `mimo_attn_test`, and `mimo_snapshot_test`. Run the small conversion probe under compute-sanitizer memcheck. No GPU execution is claimed by the implementation agent. Then perform matched real-model BF16-off, original-FP8, and exact-load-FP8 measurements with identical context, prefix-slot count, MTP, concurrency, prompt and active optimization flags. Compare output/logit evidence separately from timing; retain the original FP8 capacity-savings result independently of performance.

Implementation validation: host exhaustive-code/signed-NaN test passed. Both original and exact whole-attention CUDA variants compiled and assembled; standalone `mimo_fp8_load_test` and existing `mimo_fp8_test` linked for each variant. CMake ON configuration generated the expected source/probe definitions, and its model startup-report and GPU-probe objects compiled. These are workstation cross-build results only; target execution, latency and quality gates remain outstanding.
