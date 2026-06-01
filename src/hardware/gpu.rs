use serde::Deserialize;

use crate::model::llm_config::LlmConfig;

use super::cluster::ClusterConfig;
use super::kernel_table::{KernelOp, KernelTable};

#[derive(Debug, Clone, Deserialize)]
pub struct GpuSpec {
    pub name: String,
    pub flops_bf16: f64,       // peak BF16 FLOPS (e.g. 989e12 for H100)
    #[serde(default)] pub flops_fp8: f64, // peak FP8 FLOPS (e.g. 1978e12 for H100; 0 = same as bf16)
    pub hbm_bandwidth: f64,    // bytes/sec (e.g. 3.35e12 for H100)
    pub hbm_capacity: u64,     // bytes
    pub nvlink_bandwidth: f64, // bytes/sec per direction
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
        let tp_group_lat = match ktable.and_then(|kt| kt.lookup_nearest_batch(&self.name, &model.name, KernelOp::Prefill, batch, seq_len)) {
            Some(single_gpu_lat) => single_gpu_lat / tp as f64,
            None => self.roofline_prefill(batch, seq_len, model, tp, cluster.ep),
        };

        let ar_msg = batch as u64 * seq_len as u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let ar_cost = model.n_layers as f64 * 2.0 * cluster.all_reduce_latency(ar_msg);

        let act_bytes = batch as u64 * seq_len as u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let pp_cost = (pp - 1) as f64 * cluster.pp_transfer_latency(act_bytes, true);

        // EP all-to-all: 2 per MoE layer. Each token dispatches to top_K experts → multiply by top_K.
        let ep_cost = if model.is_moe() && model.n_moe_layers > 0 {
            let batch_tokens = batch as u64 * seq_len as u64;
            let top_k = (model.n_active_experts + model.n_shared_experts).max(1) as u64;
            model.n_moe_layers as f64
                * 2.0
                * cluster.ep_all_to_all_latency(batch_tokens * top_k, model.d_model, model.dtype_bytes)
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

        let tp_group_lat = match ktable.and_then(|kt| kt.lookup_nearest_batch(&self.name, &model.name, KernelOp::Decode, batch, avg_kv_len)) {
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
                * cluster.ep_all_to_all_latency(batch as u64 * top_k, model.d_model, model.dtype_bytes)
        } else {
            0.0
        };

        (tp_group_lat + ar_cost + ep_cost) / pp as f64 + pp_cost
    }

    /// Prefill roofline: 2 × batch × seq × active_params FLOPs.
    /// With EP: expert FLOPs split by EP; attention/dense FLOPs split by TP.
    /// Returns per-TP-group latency (already accounts for parallelism).
    fn roofline_prefill(&self, batch: u32, seq_len: u32, model: &LlmConfig, tp: u32, ep: u32) -> f64 {
        let tokens = batch as f64 * seq_len as f64;
        let flops = if ep > 1 && model.is_moe() {
            let expert_params = model.expert_weight_bytes_active() as f64 / model.dtype_bytes as f64;
            let other_params = model.weight_bytes_active().saturating_sub(model.expert_weight_bytes_active()) as f64 / model.dtype_bytes as f64;
            2.0 * tokens * (other_params / tp as f64 + expert_params / ep as f64)
        } else {
            let active_params = model.weight_bytes_active() as f64 / model.dtype_bytes as f64;
            2.0 * tokens * active_params / tp as f64
        };
        let peak_flops = if model.dtype_bytes == 1 && self.flops_fp8 > 0.0 { self.flops_fp8 } else { self.flops_bf16 };
        flops / (peak_flops * self.mfu_prefill)
    }

    /// Decode roofline: memory-BW bound — load active weights + KV per request.
    /// With EP: expert weights split by EP; attention weights and KV split by TP.
    /// Returns per-TP-group latency (already accounts for parallelism).
    fn roofline_decode(&self, batch: u32, avg_kv_len: u32, model: &LlmConfig, tp: u32, ep: u32) -> f64 {
        let kv_bytes = model.kv_bytes(avg_kv_len) * batch as u64;
        let bytes_per_gpu = if ep > 1 && model.is_moe() {
            let expert_bytes = model.expert_weight_bytes_active();
            let other_bytes = model.weight_bytes_active().saturating_sub(expert_bytes);
            // Attention + dense FFN + KV → split by TP; expert weights → split by EP.
            other_bytes / tp as u64 + expert_bytes / ep as u64 + kv_bytes / tp as u64
        } else {
            (model.weight_bytes_active() + kv_bytes) / tp as u64
        };
        bytes_per_gpu as f64 / (self.hbm_bandwidth * self.mfu_decode)
    }

    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "h100" => Some(Self {
                name: "H100-SXM5".into(),
                flops_bf16: 989e12,
                flops_fp8: 1978e12,
                hbm_bandwidth: 3.35e12,
                hbm_capacity: 80_000_000_000,
                nvlink_bandwidth: 900e9,
                mfu_prefill: 0.75,
                mfu_decode: 0.80,
            }),
            "a100" => Some(Self {
                name: "A100-80GB".into(),
                flops_bf16: 312e12,
                flops_fp8: 0.0, // A100 does not have FP8 tensor cores
                hbm_bandwidth: 2.0e12,
                hbm_capacity: 80_000_000_000,
                nvlink_bandwidth: 600e9,
                mfu_prefill: 0.75,
                mfu_decode: 0.75,
            }),
            "a10g" => Some(Self {
                name: "A10G".into(),
                flops_bf16: 125e12,
                flops_fp8: 0.0, // A10G does not have FP8 tensor cores
                hbm_bandwidth: 600e9,
                hbm_capacity: 24_000_000_000,
                nvlink_bandwidth: 0.0,
                mfu_prefill: 0.55,
                mfu_decode: 0.65,
            }),
            _ => None,
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
        Self { id, spec, busy_until: 0.0, kernel_table: None }
    }

    pub fn with_kernel_table(mut self, kt: KernelTable) -> Self {
        self.kernel_table = Some(kt);
        self
    }

    pub fn is_free(&self, now: f64) -> bool {
        now >= self.busy_until
    }
}
