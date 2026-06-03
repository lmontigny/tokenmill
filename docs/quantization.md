# Quantization and mixed precision

The simulator tracks three precision knobs on `LlmConfig`:

| Field | Meaning | Used by |
|---|---|---|
| `weight_bits` | Bits per stored model weight | model memory, decode weight traffic, prefill FLOPS tier |
| `activation_bits` | Bits per activation sent through TP/PP/EP collectives | all-reduce, pipeline transfer, MoE all-to-all |
| `kv_bits` | Bits per KV cache entry | KV cache capacity and decode KV traffic |

`kv_bits = 0` falls back to `weight_bits`. `activation_bits = 0` also falls
back to `weight_bits`, but production mixed-precision presets set it explicitly.

## Preset naming

| Preset suffix | Meaning |
|---|---|
| `-fp8` | W8A8KV8 by default |
| `-w4a16` | 4-bit weights, 16-bit activations, KV follows weights |
| `-w4a8kv4` | 4-bit weights, 8-bit activations, 4-bit KV cache |
| `-nvfp4-sparse` | W4A8KV4 plus 2:4 structured-sparsity speedup |

Current quantized presets:

- `llama-8b-w4a16`
- `llama-70b-w4a16`
- `llama-8b-w4a8kv4`
- `llama-70b-w4a8kv4`
- `llama-70b-nvfp4-sparse`
- `kimi-k2-nvfp4-sparse`

## Hardware behavior

Prefill is compute-bound and selects the fastest matching hardware tier:

```text
weight_bits <= 4 and flops_fp4 > 0  -> flops_fp4
weight_bits <= 8 and flops_fp8 > 0  -> flops_fp8
otherwise                           -> flops_bf16
```

Decode is memory-bandwidth-bound:

```text
decode_bytes = active_weight_bytes + kv_bytes(batch, seq)
```

`active_weight_bytes` follows `weight_bits`; `kv_bytes` follows `kv_bits`.
When a model has `weight_sparsity > 1` and the accelerator has
`supports_2to4_sparsity = true`, the simulator divides active-weight traffic
and active prefill FLOPs by that sparsity factor.

## Caveats

- FP4 and sparse FP4 paths are roofline estimates unless overridden by
  `--kernel-table`.
- Sparse speedup is ignored on accelerators without `supports_2to4_sparsity`.
- The model does not simulate quantization accuracy loss, scale metadata,
  outlier channels, or dequantization overhead.
- W4A16 keeps activation collectives at 16 bits even though weights are 4 bits.
