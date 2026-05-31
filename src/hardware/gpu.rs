use serde::Deserialize;

use crate::model::llm_config::LlmConfig;

use super::cluster::ClusterConfig;
use super::kernel_table::{KernelOp, KernelTable};

#[derive(Debug, Clone, Deserialize)]
pub struct GpuSpec {
    pub name: String,
    pub flops_bf16: f64,       // peak FLOPS (e.g. 989e12 for H100)
    pub hbm_bandwidth: f64,    // bytes/sec (e.g. 3.35e12 for H100)
    pub hbm_capacity: u64,     // bytes
    pub nvlink_bandwidth: f64, // bytes/sec per direction
    pub mfu_prefill: f64,
    pub mfu_decode: f64,
}

impl GpuSpec {
    /// Prefill latency in seconds, accounting for TP, PP, and EP (MoE expert parallelism).
    ///
    /// For MoE models: active FLOPs are scaled by active_param_fraction, and each MoE layer
    /// adds two EP all-to-alls (dispatch tokens to expert GPUs + gather results).
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

        let base = if let Some(kt) = ktable {
            kt.lookup_nearest_batch(&self.name, &model.name, KernelOp::Prefill, batch, seq_len)
                .unwrap_or_else(|| self.roofline_prefill(batch, seq_len, model))
        } else {
            self.roofline_prefill(batch, seq_len, model)
        };

        let compute = base / tp as f64;
        let ar_msg = batch as u64 * seq_len as u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let ar_cost = model.n_layers as f64 * 2.0 * cluster.all_reduce_latency(ar_msg);

        let pp_transfers = (pp - 1) as f64;
        let act_bytes = batch as u64 * seq_len as u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let pp_cost = pp_transfers * cluster.pp_transfer_latency(act_bytes, true);

        // EP all-to-all: two per MoE layer (dispatch + combine), run in parallel with TP.
        let ep_cost = if model.is_moe() && model.n_moe_layers > 0 {
            let batch_tokens = batch as u64 * seq_len as u64;
            model.n_moe_layers as f64
                * 2.0
                * cluster.ep_all_to_all_latency(batch_tokens, model.d_model, model.dtype_bytes)
        } else {
            0.0
        };

        (compute + ar_cost) / pp as f64 + pp_cost + ep_cost
    }

    /// Batch decode latency in seconds for one token step.
    ///
    /// Memory-BW bound: load active weights (1/tp per GPU, sparse for MoE) + read KV caches.
    /// MoE adds two EP all-to-alls per MoE layer.
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

        let base = if let Some(kt) = ktable {
            kt.lookup_nearest_batch(&self.name, &model.name, KernelOp::Decode, batch, avg_kv_len)
                .unwrap_or_else(|| self.roofline_decode(batch, avg_kv_len, model))
        } else {
            self.roofline_decode(batch, avg_kv_len, model)
        };

        let compute = base / tp as f64;
        let ar_msg = batch as u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let ar_cost = model.n_layers as f64 * 2.0 * cluster.all_reduce_latency(ar_msg);

        let pp_transfers = (pp - 1) as f64;
        let act_bytes = batch as u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let pp_cost = pp_transfers * cluster.pp_transfer_latency(act_bytes, true);

        // EP all-to-all: one new token per request in the batch.
        let ep_cost = if model.is_moe() && model.n_moe_layers > 0 {
            model.n_moe_layers as f64
                * 2.0
                * cluster.ep_all_to_all_latency(batch as u64, model.d_model, model.dtype_bytes)
        } else {
            0.0
        };

        (compute + ar_cost) / pp as f64 + pp_cost + ep_cost
    }

    fn roofline_prefill(&self, batch: u32, seq_len: u32, model: &LlmConfig) -> f64 {
        let base_flops =
            2.0 * batch as f64 * seq_len as f64 * model.d_model as f64 * model.n_layers as f64 * 12.0;
        // Scale active FLOPs for MoE (only top-K experts run per token).
        let active_flops = base_flops * model.active_param_fraction();
        active_flops / (self.flops_bf16 * self.mfu_prefill)
    }

    fn roofline_decode(&self, batch: u32, avg_kv_len: u32, model: &LlmConfig) -> f64 {
        let kv_bytes = model.kv_bytes(avg_kv_len) * batch as u64;
        // Use active weight bytes for MoE (only loaded experts contribute to BW).
        (model.weight_bytes_active() + kv_bytes) as f64 / (self.hbm_bandwidth * self.mfu_decode)
    }

    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "h100" => Some(Self {
                name: "H100-SXM5".into(),
                flops_bf16: 989e12,
                hbm_bandwidth: 3.35e12,
                hbm_capacity: 80_000_000_000,
                nvlink_bandwidth: 900e9,
                mfu_prefill: 0.50,
                mfu_decode: 0.30,
            }),
            "a100" => Some(Self {
                name: "A100-80GB".into(),
                flops_bf16: 312e12,
                hbm_bandwidth: 2.0e12,
                hbm_capacity: 80_000_000_000,
                nvlink_bandwidth: 600e9,
                mfu_prefill: 0.50,
                mfu_decode: 0.30,
            }),
            "a10g" => Some(Self {
                name: "A10G".into(),
                flops_bf16: 125e12,
                hbm_bandwidth: 600e9,
                hbm_capacity: 24_000_000_000,
                nvlink_bandwidth: 0.0,
                mfu_prefill: 0.45,
                mfu_decode: 0.25,
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
