use serde::Deserialize;

use crate::model::llm_config::LlmConfig;

use super::cluster::ClusterConfig;
use super::kernel_table::{KernelOp, KernelTable};

#[derive(Debug, Clone, Deserialize)]
pub struct GpuSpec {
    pub name: String,
    pub flops_bf16: f64, // peak BF16 FLOPS (e.g. 989e12 for H100)
    #[serde(default)]
    pub flops_fp8: f64, // peak FP8 FLOPS (e.g. 1978e12 for H100; 0 = same as bf16)
    /// Main memory bandwidth in bytes/sec. HBM on NVIDIA/AMD/TPU; on-chip SRAM
    /// bandwidth on SRAM-only architectures like Groq.
    pub memory_bandwidth: f64,
    /// Main memory capacity in bytes per chip.
    pub memory_capacity: u64,
    /// Faster on-chip memory tier (L2 cache on NVIDIA/AMD, Vmem scratchpad on TPU).
    /// When the per-chip KV working set fits in this budget, decode KV traffic is
    /// served from SRAM at ~10× the cost of main memory. Set to 0 on SRAM-only
    /// architectures where there's no faster tier (the "memory" IS the SRAM).
    #[serde(default)]
    pub on_chip_sram: u64,
    /// Scale-up fabric bandwidth per direction (NVLink, Infinity Fabric, TPU ICI, Groq C2C).
    pub scale_up_bandwidth: f64,
    /// Per-hop latency on the scale-up fabric in seconds. Dominates collective cost at
    /// very high TP. Typical values: NVLink/IF ~1 µs, TPU ICI ~500 ns, Groq C2C ~100 ns.
    /// 0 = pure bandwidth model (the pre-existing behaviour).
    #[serde(default)]
    pub scale_up_latency: f64,
    /// Peak power draw per chip in watts (TDP). Used by `hardware::power` to
    /// compute energy/avg-power metrics from per-GPU prefill / decode busy time.
    /// Default 0 disables power reporting.
    #[serde(default)]
    pub tdp_watts: f64,
    pub mfu_prefill: f64,
    pub mfu_decode: f64,
}

impl GpuSpec {
    /// Prefill latency in seconds, accounting for TP, PP, and EP (MoE expert parallelism).
    ///
    /// For MoE with EP: expert FLOPs split by EP, attention/dense FLOPs split by TP.
    /// Each MoE layer adds two EP all-to-alls (dispatch × top_K tokens + combine).
    /// All-to-all runs over NVLink (scale-up domain only; no Infiniband/Ethernet modeled).
    pub fn prefill_latency(
        &self,
        batch: u32,
        seq_len: u32,
        model: &LlmConfig,
        ktable: Option<&KernelTable>,
        cluster: &ClusterConfig,
    ) -> f64 {
        let tp = cluster.tp.max(1);
        let pp = cluster.pp.max(1);

        // Per-TP-group latency: kernel table gives single-GPU numbers (÷TP); roofline has TP+EP baked in.
        let tp_group_lat = match ktable.and_then(|kt| {
            kt.lookup_nearest_batch(&self.name, &model.name, KernelOp::Prefill, batch, seq_len)
        }) {
            Some(single_gpu_lat) => single_gpu_lat / tp as f64,
            None => self.roofline_prefill(batch, seq_len, model, tp, cluster.ep),
        };

        let ar_msg =
            batch as u64 * seq_len as u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let ar_cost = model.n_layers as f64 * 2.0 * cluster.all_reduce_latency(ar_msg);

        let act_bytes =
            batch as u64 * seq_len as u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let pp_cost = (pp - 1) as f64 * cluster.pp_transfer_latency(act_bytes, true);

        // EP all-to-all: 2 per MoE layer. Each token dispatches to top_K experts → multiply by top_K.
        let ep_cost = if model.is_moe() && model.n_moe_layers > 0 {
            let batch_tokens = batch as u64 * seq_len as u64;
            let top_k = (model.n_active_experts + model.n_shared_experts).max(1) as u64;
            model.n_moe_layers as f64
                * 2.0
                * cluster.ep_all_to_all_latency(
                    batch_tokens * top_k,
                    model.d_model,
                    model.dtype_bytes,
                )
        } else {
            0.0
        };

        // All per-layer costs (compute, all-reduce, EP all-to-all) divide evenly across PP stages.
        (tp_group_lat + ar_cost + ep_cost) / pp as f64 + pp_cost
    }

    /// Batch decode latency in seconds for one token step.
    ///
    /// Memory-BW bound: expert weights split by EP, attention/KV by TP.
    /// MoE adds two EP all-to-alls per MoE layer (× top_K for dispatch volume).
    pub fn decode_latency(
        &self,
        batch: u32,
        avg_kv_len: u32,
        model: &LlmConfig,
        ktable: Option<&KernelTable>,
        cluster: &ClusterConfig,
    ) -> f64 {
        let tp = cluster.tp.max(1);
        let pp = cluster.pp.max(1);

        let tp_group_lat = match ktable.and_then(|kt| {
            kt.lookup_nearest_batch(&self.name, &model.name, KernelOp::Decode, batch, avg_kv_len)
        }) {
            Some(single_gpu_lat) => single_gpu_lat / tp as f64,
            None => self.roofline_decode(batch, avg_kv_len, model, tp, cluster.ep),
        };

        let ar_msg = batch as u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let ar_cost = model.n_layers as f64 * 2.0 * cluster.all_reduce_latency(ar_msg);

        let act_bytes = batch as u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let pp_cost = (pp - 1) as f64 * cluster.pp_transfer_latency(act_bytes, true);

        let ep_cost = if model.is_moe() && model.n_moe_layers > 0 {
            let top_k = (model.n_active_experts + model.n_shared_experts).max(1) as u64;
            model.n_moe_layers as f64
                * 2.0
                * cluster.ep_all_to_all_latency(
                    batch as u64 * top_k,
                    model.d_model,
                    model.dtype_bytes,
                )
        } else {
            0.0
        };

        (tp_group_lat + ar_cost + ep_cost) / pp as f64 + pp_cost
    }

    /// Prefill roofline: 2 × batch × seq × active_params FLOPs.
    /// With EP: expert FLOPs split by EP; attention/dense FLOPs split by TP.
    /// Returns per-TP-group latency (already accounts for parallelism).
    fn roofline_prefill(
        &self,
        batch: u32,
        seq_len: u32,
        model: &LlmConfig,
        tp: u32,
        ep: u32,
    ) -> f64 {
        let tokens = batch as f64 * seq_len as f64;
        let flops = if ep > 1 && model.is_moe() {
            let expert_params =
                model.expert_weight_bytes_active() as f64 / model.dtype_bytes as f64;
            let other_params = model
                .weight_bytes_active()
                .saturating_sub(model.expert_weight_bytes_active())
                as f64
                / model.dtype_bytes as f64;
            2.0 * tokens * (other_params / tp as f64 + expert_params / ep as f64)
        } else {
            let active_params = model.weight_bytes_active() as f64 / model.dtype_bytes as f64;
            2.0 * tokens * active_params / tp as f64
        };
        let peak_flops = if model.dtype_bytes == 1 && self.flops_fp8 > 0.0 {
            self.flops_fp8
        } else {
            self.flops_bf16
        };
        flops / (peak_flops * self.mfu_prefill)
    }

    /// Decode roofline: memory-BW bound — load active weights + KV per request.
    /// With EP: expert weights split by EP; attention weights and KV split by TP.
    /// Returns per-TP-group latency (already accounts for parallelism).
    ///
    /// On-chip SRAM: when the per-chip KV footprint fits in `on_chip_sram`, KV
    /// traffic is served from SRAM (~10× faster than HBM in practice). We model
    /// this by counting SRAM-resident bytes at 1/10 the HBM cost. Weights are
    /// orders of magnitude larger than any SRAM and always hit HBM.
    fn roofline_decode(
        &self,
        batch: u32,
        avg_kv_len: u32,
        model: &LlmConfig,
        tp: u32,
        ep: u32,
    ) -> f64 {
        let kv_bytes = model.kv_bytes(avg_kv_len) * batch as u64;
        let kv_bytes_per_chip = kv_bytes / tp as u64;

        let weight_bytes_per_chip = if ep > 1 && model.is_moe() {
            let expert_bytes = model.expert_weight_bytes_active();
            let other_bytes = model.weight_bytes_active().saturating_sub(expert_bytes);
            other_bytes / tp as u64 + expert_bytes / ep as u64
        } else {
            model.weight_bytes_active() / tp as u64
        };

        // SRAM benefit: if the per-chip KV fits in on-chip SRAM, it costs ~1/10 of HBM.
        let kv_effective_bytes = if self.on_chip_sram > 0 && kv_bytes_per_chip <= self.on_chip_sram
        {
            kv_bytes_per_chip / 10
        } else {
            kv_bytes_per_chip
        };

        let bytes_per_gpu = weight_bytes_per_chip + kv_effective_bytes;
        bytes_per_gpu as f64 / (self.memory_bandwidth * self.mfu_decode)
    }

    pub fn preset(name: &str) -> Option<Self> {
        match name {
            // NVIDIA Blackwell B200 SXM (NVL72/HGX). FP4 not modeled — listed for reference: 9000 TFLOPS dense.
            "b200" => Some(Self {
                name: "B200-SXM".into(),
                flops_bf16: 2250e12,              // 2.25 PFLOPS dense BF16/FP16
                flops_fp8: 4500e12,               // 4.5 PFLOPS dense FP8 (2× H100)
                memory_bandwidth: 8.0e12,         // 8 TB/s HBM3e (2.4× H100)
                memory_capacity: 192_000_000_000, // 192 GB HBM3e
                on_chip_sram: 100_000_000,        // ~100 MB L2 (estimate)
                scale_up_bandwidth: 1800e9,       // NVLink 5: 1.8 TB/s aggregate (2× H100 NVLink 4)
                scale_up_latency: 1e-6,           // NVSwitch hop ~1 µs
                tdp_watts: 1000.0,                // B200 SXM TDP 1 kW
                mfu_prefill: 0.70, // slightly lower than H100 — new gen, real-world kernels less mature
                mfu_decode: 0.75,
            }),
            "h100" => Some(Self {
                name: "H100-SXM5".into(),
                flops_bf16: 989e12,
                flops_fp8: 1978e12,
                memory_bandwidth: 3.35e12,
                memory_capacity: 80_000_000_000,
                on_chip_sram: 50_000_000, // 50 MB L2
                scale_up_bandwidth: 900e9,
                scale_up_latency: 1e-6, // NVSwitch hop ~1 µs
                tdp_watts: 700.0,       // H100 SXM5
                mfu_prefill: 0.75,
                mfu_decode: 0.80,
            }),
            "a100" => Some(Self {
                name: "A100-80GB".into(),
                flops_bf16: 312e12,
                flops_fp8: 0.0, // A100 does not have FP8 tensor cores
                memory_bandwidth: 2.0e12,
                memory_capacity: 80_000_000_000,
                on_chip_sram: 40_000_000, // 40 MB L2
                scale_up_bandwidth: 600e9,
                scale_up_latency: 1.5e-6, // NVLink 3 hop ~1.5 µs
                tdp_watts: 400.0,         // A100 80GB SXM
                mfu_prefill: 0.75,
                mfu_decode: 0.75,
            }),
            // (Google TPU presets live in `tpu.rs` and are picked up by the fall-through below.)
            // AMD Instinct MI300X (CDNA 3, 2023) — H100 competitor with 2.4× more HBM at 1.6× BW.
            // Infinity Fabric stored in `scale_up_bandwidth` (scale-up fabric is treated uniformly).
            // MFU is conservative vs H100 — ROCm/vLLM kernel maturity gap.
            "mi300x" => Some(Self {
                name: "MI300X".into(),
                flops_bf16: 1307e12,      // 1.307 PFLOPS BF16 matrix (dense)
                flops_fp8: 2614e12,       // 2.614 PFLOPS FP8 matrix (dense)
                memory_bandwidth: 5.3e12, // 5.3 TB/s HBM3 (1.58× H100)
                memory_capacity: 192_000_000_000, // 192 GB HBM3
                on_chip_sram: 256_000_000, // ~256 MB Infinity Cache + L2
                scale_up_bandwidth: 896e9, // Infinity Fabric: 896 GB/s aggregate per GPU
                scale_up_latency: 1.2e-6, // IF hop ~1.2 µs
                tdp_watts: 750.0,         // MI300X
                mfu_prefill: 0.65,
                mfu_decode: 0.72,
            }),
            // AMD Instinct MI325X (CDNA 3 refresh, 2024) — same compute as MI300X, more memory.
            "mi325x" => Some(Self {
                name: "MI325X".into(),
                flops_bf16: 1307e12,
                flops_fp8: 2614e12,
                memory_bandwidth: 6.0e12,         // 6.0 TB/s HBM3e
                memory_capacity: 256_000_000_000, // 256 GB HBM3e
                on_chip_sram: 256_000_000,
                scale_up_bandwidth: 896e9,
                scale_up_latency: 1.2e-6,
                tdp_watts: 1000.0, // MI325X TDP 1 kW
                mfu_prefill: 0.65,
                mfu_decode: 0.72,
            }),
            // AMD Instinct MI355X (CDNA 4, 2025) — B200 competitor. FP4/FP6 not modeled.
            "mi355x" => Some(Self {
                name: "MI355X".into(),
                flops_bf16: 2500e12,              // ~2.5 PFLOPS BF16 dense
                flops_fp8: 5000e12,               // ~5.0 PFLOPS FP8 dense (slightly ahead of B200)
                memory_bandwidth: 8.0e12,         // 8 TB/s HBM3e
                memory_capacity: 288_000_000_000, // 288 GB HBM3e (50% more than B200)
                on_chip_sram: 256_000_000,
                scale_up_bandwidth: 1075e9, // Infinity Fabric Gen 4: 1.075 TB/s
                scale_up_latency: 1e-6,     // IF Gen 4 hop ~1 µs
                tdp_watts: 1400.0,          // MI355X 1.4 kW
                mfu_prefill: 0.65,
                mfu_decode: 0.72,
            }),
            "a10g" => Some(Self {
                name: "A10G".into(),
                flops_bf16: 125e12,
                flops_fp8: 0.0, // A10G does not have FP8 tensor cores
                memory_bandwidth: 600e9,
                memory_capacity: 24_000_000_000,
                on_chip_sram: 6_000_000, // 6 MB L2
                scale_up_bandwidth: 0.0,
                scale_up_latency: 0.0, // no scale-up fabric (single-GPU card)
                tdp_watts: 300.0,      // A10G
                mfu_prefill: 0.55,
                mfu_decode: 0.65,
            }),
            // Fall through to vendor-specific accelerator modules.
            other => super::tpu::preset(other).or_else(|| super::groq::preset(other)),
        }
    }
}

pub struct GpuState {
    pub id: u32,
    pub spec: GpuSpec,
    pub busy_until: f64,
    pub kernel_table: Option<KernelTable>,
}

impl GpuState {
    pub fn new(id: u32, spec: GpuSpec) -> Self {
        Self {
            id,
            spec,
            busy_until: 0.0,
            kernel_table: None,
        }
    }

    pub fn with_kernel_table(mut self, kt: KernelTable) -> Self {
        self.kernel_table = Some(kt);
        self
    }

    pub fn is_free(&self, now: f64) -> bool {
        now >= self.busy_until
    }
}
