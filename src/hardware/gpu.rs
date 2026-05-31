use serde::Deserialize;

use crate::model::llm_config::LlmConfig;

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
    /// Roofline prefill latency in seconds.
    pub fn prefill_latency(&self, batch: u32, seq_len: u32, model: &LlmConfig) -> f64 {
        // 2 × batch × seq_len × d_model × n_layers × 12 FLOPs (attn + FFN approximation)
        let flops = 2.0
            * batch as f64
            * seq_len as f64
            * model.d_model as f64
            * model.n_layers as f64
            * 12.0;
        flops / (self.flops_bf16 * self.mfu_prefill)
    }

    /// Roofline decode latency in seconds for one token step.
    pub fn decode_latency(&self, _batch: u32, _kv_seq_len: u32, model: &LlmConfig) -> f64 {
        // Memory-BW bound: load all weights once per step
        model.weight_bytes as f64 / (self.hbm_bandwidth * self.mfu_decode)
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
    pub busy_until: f64, // sim time when GPU becomes free
}

impl GpuState {
    pub fn new(id: u32, spec: GpuSpec) -> Self {
        Self { id, spec, busy_until: 0.0 }
    }

    pub fn is_free(&self, now: f64) -> bool {
        now >= self.busy_until
    }
}
