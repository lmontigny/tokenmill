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
    /// Prefill latency in seconds, accounting for TP and PP.
    ///
    /// TP splits compute across tp GPUs (each does 1/tp of FLOPs),
    /// then pays n_layers × 2 all-reduces (after attention + after FFN).
    /// PP splits the layer stack across pp stages executed serially.
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

        // Base latency for the full model (table or roofline)
        let base = if let Some(kt) = ktable {
            kt.lookup_nearest_batch(&self.name, &model.name, KernelOp::Prefill, batch, seq_len)
                .unwrap_or_else(|| self.roofline_prefill(batch, seq_len, model))
        } else {
            self.roofline_prefill(batch, seq_len, model)
        };

        // TP: compute scales by 1/tp; each of n_layers has 2 all-reduces
        // (one after attention projection, one after FFN projection).
        // msg per all-reduce = batch * seq_len * d_model * dtype_bytes
        let compute = base / tp as f64;
        let ar_msg = batch as u64 * seq_len as u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let ar_cost = model.n_layers as f64 * 2.0 * cluster.all_reduce_latency(ar_msg);

        // PP: stages are serial; total latency ≈ base / pp (each stage handles 1/pp of layers).
        // Inter-stage transfer: activation = batch * seq_len * d_model * dtype_bytes.
        let pp_transfers = (pp - 1) as f64;
        let act_bytes = batch as u64 * seq_len as u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let pp_cost = pp_transfers * cluster.pp_transfer_latency(act_bytes, true);

        (compute + ar_cost) / pp as f64 + pp_cost
    }

    /// Batch decode latency in seconds for one token step across all decode requests.
    ///
    /// Memory-BW bound: load weights (1/tp per GPU) + read KV caches.
    /// Then pay all-reduces and PP transfers same as prefill but for batch=1 token.
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

        // TP: weight load and KV load divided by tp; all-reduce on each layer output.
        // For decode, seq_len=1 per new token.
        let compute = base / tp as f64;
        let ar_msg = batch as u64 * 1_u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let ar_cost = model.n_layers as f64 * 2.0 * cluster.all_reduce_latency(ar_msg);

        // PP: same stage-serial + transfer model as prefill, but for 1 output token.
        let pp_transfers = (pp - 1) as f64;
        let act_bytes = batch as u64 * 1_u64 * model.d_model as u64 * model.dtype_bytes as u64;
        let pp_cost = pp_transfers * cluster.pp_transfer_latency(act_bytes, true);

        (compute + ar_cost) / pp as f64 + pp_cost
    }

    fn roofline_prefill(&self, batch: u32, seq_len: u32, model: &LlmConfig) -> f64 {
        let flops = 2.0 * batch as f64 * seq_len as f64 * model.d_model as f64 * model.n_layers as f64 * 12.0;
        flops / (self.flops_bf16 * self.mfu_prefill)
    }

    fn roofline_decode(&self, batch: u32, avg_kv_len: u32, model: &LlmConfig) -> f64 {
        let kv_bytes = model.kv_bytes(avg_kv_len) * batch as u64;
        (model.weight_bytes + kv_bytes) as f64 / (self.hbm_bandwidth * self.mfu_decode)
    }

    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "h100" => Some(Self {
                name: "H100-SXM5".into(),
                flops_bf16: 989e12,
                hbm_bandwidth: 3.35e12,
                hbm_capacity: 80 * 1024 * 1024 * 1024,
                nvlink_bandwidth: 900e9,
                mfu_prefill: 0.50,
                mfu_decode: 0.30,
            }),
            "a100" => Some(Self {
                name: "A100-80GB".into(),
                flops_bf16: 312e12,
                hbm_bandwidth: 2.0e12,
                hbm_capacity: 80 * 1024 * 1024 * 1024,
                nvlink_bandwidth: 600e9,
                mfu_prefill: 0.50,
                mfu_decode: 0.30,
            }),
            "a10g" => Some(Self {
                name: "A10G".into(),
                flops_bf16: 125e12,
                hbm_bandwidth: 600e9,
                hbm_capacity: 24 * 1024 * 1024 * 1024,
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
