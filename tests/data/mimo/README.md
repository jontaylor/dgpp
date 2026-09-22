# MiMo metadata fixtures

Source: `XiaomiMiMo/MiMo-V2.6-Flash-RL` at
`3b38d063180c3e4aed9691fdc735f3d10b266ee4` (MIT model repository).

`config.json` is the unmodified published configuration, SHA256
`61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621`.

`tensor_headers.json` contains names, dtypes and shapes extracted using HTTP
range reads from `model_pp0_ep0_shard0.safetensors`. It retains globals and
layers 0, 1 and 47, with expert 0 only in MoE layers. Payload bytes were not
downloaded. Tensor offsets were omitted. This deliberately small fixture
provides independent evidence for both attention geometries, the dense FFN,
the expert format and global tensors; it is not whole-checkpoint validation.

The full published index was additionally checked with the index-only CLI;
it is not included as a multi-megabyte test fixture.

`weight_samples.json` contains 44 QKV row windows (704 values) from layers
0/1 and four expert windows (256 values) from layer 1 expert 0's gate/down
projections. Generated directly from Spark-1's completed snapshot using
`tools/mimo_reference_sample.py`, it preserves payload bytes, scales, source
coordinates and independently calculated BF16 results. QKV windows cross a
128-column scale boundary and cover all projection types, source-chunk
boundaries and both global/sliding attention. Expert windows cover low/high
nibble ordering, signed zero, nonzero column offsets and first/last rows.

The standard-library generator follows the pinned vLLM/SGLang storage mapping
recorded in the port plan. It is a storage-decoding oracle, not a PyTorch
forward-pass or full-model equivalence fixture. The initial generator used
integer zero in its e2m1 table, dropping negative zero; the retained generator
and fixture use floating zero and preserve its sign.

`attention_goldens.json` is generated with CPU PyTorch 2.14.0 by
`tools/mimo_attention_reference.py`. No checkpoint Python is imported or
executed. The generator mirrors the published rotary/eager-attention tensor
operations with deterministic BF16 projection inputs. It includes Q/K/V and
output for 16 global/sliding boundary cases, and Q/K for six high-position
RoPE-only cases. The normal host tests read this file and need no PyTorch
installation. Regeneration requires the optional CPU PyTorch environment;
it does not read weights or use a GPU.

`layer_weight_samples.json` contains 110 bounded windows of ordinary text
weights from the same pinned snapshot: 75 BF16, 30 FP8 dense-MLP and five
FP32 router-bias samples. `tools/mimo_layer_weight_sample.py` reads source
coordinates directly and independently decodes dense FP8 using its original
scale grid. Samples cover global tensors, layer 0/1 norms and output
projections, sliding sinks, router weights and correction bias, and all three
dense projections. Partition endpoints and scale boundaries are included.
`mimo_weight_check` accepts this fixture as an optional third argument and
checks both values and per-world sample multiplicity through the file-backed
reader. This remains sampled storage evidence, not layer-output evidence.


`layer0_goldens.json` and `layer1_goldens.json` contain three sequential
real-weight decoder outputs at positions 0, 1 and 2, generated with CPU
PyTorch 2.14.0 from the pinned snapshot. Layer 0 covers global attention and
the dense FFN; layer 1 covers sliding attention, sinks and top-8 MXFP4 MoE.
The deterministic input is specified in `mimo_layer_reference.py`; the JSON
retains BF16 output bits and selected routes. The generator directly reads
and dequantizes checkpoint tensors without importing checkpoint Python.

Regenerate and compare on idle test hardware:

```bash
python3 tools/mimo_layer_reference.py /path/to/snapshot 0 tests/data/mimo/layer0_goldens.json
python3 tools/mimo_layer_reference.py /path/to/snapshot 1 tests/data/mimo/layer1_goldens.json
mimo_layer_check /path/to/snapshot 0 /tmp/mimo-layer0.bf16
mimo_layer_check /path/to/snapshot 1 /tmp/mimo-layer1.bf16
python3 tools/mimo_layer_compare.py tests/data/mimo/layer0_goldens.json /tmp/mimo-layer0.bf16
python3 tools/mimo_layer_compare.py tests/data/mimo/layer1_goldens.json /tmp/mimo-layer1.bf16
```

These single-layer checks do not establish full-model or distributed logit
parity. DGPP accumulates weighted MoE partials in FP32 before the final BF16
fold; the reference rounds each expert down projection before weighting.


The layer probe also accepts optional `CHUNK TOKENS` arguments. For a
chunking regression, run each of layers 0 and 1 with `1 259` and `128 259`,
then pass the first raw `.bf16` file as the reference to
`tools/mimo_layer_compare.py`. This covers multiple 128-slot ring wraps and
a partial final chunk, comparing every row with the existing tolerance.
