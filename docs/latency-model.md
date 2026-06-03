# Latency model

## Roofline

Without `--kernel-table`: **roofline** (compute-bound prefill, memory-BW-bound decode).
With `--kernel-table`: table lookup with linear interpolation on seq_len, roofline fallback on miss.

**FP8 dispatch**: when `model.dtype_bytes == 1`, prefill uses `flops_fp8` (H100: 1978 TFLOPS, 2× BF16)
instead of `flops_bf16`. Decode is memory-BW bound regardless of dtype — the speedup there comes from
loading half as many bytes (FP8 weight bytes = BF16/2).

**On-chip SRAM (`on_chip_sram`)**: in the decode roofline, if the per-chip KV working set (KV bytes ÷ TP)
fits in SRAM, those bytes are counted at **1/10** the HBM cost. The benefit is biggest for low-latency
serving on TPU 8i (384 MB Vmem), where KV cache of small/moderate-batch workloads stays on-chip and the
chip approaches its compute-not-memory regime. Weights are orders of magnitude larger than any on-chip
SRAM and always hit HBM. Set `on_chip_sram = 0` in a preset to disable this model and recover the
HBM-only behaviour.

To improve accuracy, add rows to `data/kernel_table.csv` from your own profiling.
CSV format: `gpu,model,op,batch_size,seq_len,latency_ms`.

## MoE accuracy model

**Prefill** (compute-bound, active FLOPs only):
```
active_flops = base_flops × active_param_fraction
active_param_fraction = active_weight_bytes / weight_bytes   (when set)
                      ≈ 1/3 (attn) + 2/3 × (dense_layers + moe_layers × top_K/n_experts)
```

**Decode** (memory-BW bound, active weights):
```
latency = (weight_bytes_active + kv_bytes × batch) / (HBM_BW × mfu_decode)
weight_bytes_active = active weight bytes for one forward pass (presets use published values)
```

**DeepSeek V3 KV cache** (MLA compression):
```
kv_bytes = n_layers × kv_lora_rank × seq_len × dtype_bytes   (vs 2 × n_layers × n_kv_heads × head_dim × seq_len)
```

**EP decode bandwidth** (per GPU, EP > 1):
```
bytes_per_gpu = (weight_attn + weight_dense_ffn + kv_bytes) / tp   ← TP-sharded
              + weight_expert_active / ep                            ← EP-sharded
weight_expert_active = n_moe_layers × (n_active_experts + n_shared_experts) × 2 × d_model × expert_hidden × dtype
```
When TP = EP (e.g. TP=8, EP=8) per-GPU BW is the same as EP=1. The difference appears when TP ≠ EP.

**EP all-to-all** (2 per MoE layer: token dispatch + result gather, over NVLink switch fabric):
```
expert_activations = batch_tokens × top_K            ← each token fans out to top_K experts
tokens_per_gpu     = expert_activations / ep
data_per_gpu       = (ep-1)/ep × tokens_per_gpu × d_model × dtype_bytes
latency            = data_per_gpu / scale_up_bw         ← full bisection BW on NVSwitch
```
