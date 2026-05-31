use serde::Deserialize;

use crate::model::llm_config::LlmConfig;

use super::kernel_table::{KernelOp, KernelTable};

#[derive(Debug, Clone, Deserialize)]
pub struct GpuSpec {
    pub name: String,
    pub flops_bf16: f64,       // peak FLOPS (e.g. 989e12 for H100)
    pub hbm_bandwidth: f64,    // bytes/sec (e.g. 3.35e12 for H100)
    pub hbm_capacity: u64,     // bytes
    pub nvlink_bandwidth: f64, // bytes/sec per direction
    pub mfu_prefill: f64,      // model flop utilization for prefill (~0.5)
    pub mfu_decode: f64,       // model flop utilization for decode (~0.3)
}

impl GpuSpec {
    /// Prefill latency in seconds. Uses kernel table if available, else roofline.
    pub fn prefill_latency(
        &self,
        batch: u32,
        seq_len: u32,
        model: &LlmConfig,
        ktable: Option<&KernelTable>,
    ) -> f64 {
        if let Some(kt) = ktable {
            if let Some(v) = kt.lookup_nearest_batch(&self.name, &model.name, KernelOp::Prefill, batch, seq_len) {
                return v;
            }
        }
        // Roofline fallback: 2 × batch × seq_len × d_model × n_layers × 12 FLOPs
        let flops = 2.0 * batch as f64 * seq_len as f64 * model.d_model as f64 * model.n_layers as f64 * 12.0;
        flops / (self.flops_bf16 * self.mfu_prefill)
    }

    /// Batch decode latency in seconds. Uses kernel table if available, else roofline.
    pub fn decode_latency(
        &self,
        batch: u32,
        avg_kv_len: u32,
        model: &LlmConfig,
        ktable: Option<&KernelTable>,
    ) -> f64 {
        if let Some(kt) = ktable {
            if let Some(v) = kt.lookup_nearest_batch(&self.name, &model.name, KernelOp::Decode, batch, avg_kv_len) {
                return v;
            }
        }
        // Roofline fallback: memory-BW bound, weights + KV cache
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
