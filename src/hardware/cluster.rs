/// Parallelism configuration for multi-GPU inference.
///
/// TP (tensor parallelism): all tp_degree GPUs work on every layer together.
///   - Each GPU holds 1/tp_degree of weights → compute and memory BW scale linearly.
///   - After attention and after FFN: one all-reduce collective per layer.
///   - All-reduce cost: ring-allreduce = 2*(tp-1)/tp * msg_bytes / nvlink_bw
///
/// PP (pipeline parallelism): model split across pp_degree stage groups.
///   - Each stage handles n_layers/pp_degree layers.
///   - Stages execute sequentially for one request; overlap emerges from multiple requests
///     in flight (pipeline fill), which the DES models naturally via event timing.
///   - Inter-stage transfer cost: activation_bytes / nvlink_bw (intranode) or ibw (cross-node).
///
/// EP (expert parallelism): MoE experts sharded across ep_degree GPUs within a single NVLink
/// scale-up domain (NVL72, HGX, DGX). No cross-node Ethernet/InfiniBand modeled.
///   - Each GPU holds n_experts/ep experts; tokens are dispatched via NVLink all-to-all.
///   - Two all-to-alls per MoE layer: dispatch (send top_K token activations out) + combine (gather).
///   - All-to-all data per GPU per direction ≈ (ep-1)/ep × (top_K × batch / ep) × d_model × dtype_bytes.
///   - NVSwitch fabric provides full bisection BW; formula is bandwidth-dominated (large-message regime).
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub tp: u32,
    pub pp: u32,
    /// Expert parallelism degree for MoE models (1 = no EP, all experts on one GPU).
    pub ep: u32,
    /// NVLink bandwidth per direction in bytes/sec (all-reduce, EP all-to-all, intranode PP).
    pub nvlink_bw: f64,
    /// Cross-node bandwidth in bytes/sec (PP inter-server and KV transfer in disaggregated PD).
    pub internode_bw: f64,
    /// If true, prefill and decode run on separate GPU pools connected over internode_bw.
    pub disaggregate: bool,
}

impl ClusterConfig {
    pub fn single_gpu() -> Self {
        Self { tp: 1, pp: 1, ep: 1, nvlink_bw: 0.0, internode_bw: 0.0, disaggregate: false }
    }

    /// KV transfer latency from prefill node to decode node (seconds).
    pub fn kv_transfer_latency(&self, kv_bytes: u64) -> f64 {
        if !self.disaggregate || self.internode_bw <= 0.0 {
            return 0.0;
        }
        kv_bytes as f64 / self.internode_bw
    }

    /// All-reduce latency for ring-allreduce over NVLink (seconds).
    pub fn all_reduce_latency(&self, msg_bytes: u64) -> f64 {
        if self.tp <= 1 || self.nvlink_bw <= 0.0 {
            return 0.0;
        }
        let factor = 2.0 * (self.tp - 1) as f64 / self.tp as f64;
        factor * msg_bytes as f64 / self.nvlink_bw
    }

    /// Point-to-point activation transfer latency between PP stages (seconds).
    pub fn pp_transfer_latency(&self, activation_bytes: u64, intranode: bool) -> f64 {
        if self.pp <= 1 {
            return 0.0;
        }
        let bw = if intranode { self.nvlink_bw } else { self.internode_bw };
        if bw <= 0.0 { return 0.0; }
        activation_bytes as f64 / bw
    }

    /// All-to-all latency for one direction of expert dispatch or combine (seconds).
    /// Called twice per MoE layer: once to send tokens to expert GPUs, once to gather results.
    ///
    /// `batch_tokens` must already be multiplied by top_K at the call site (caller knows the model).
    /// Per-GPU send volume = (ep-1)/ep × (batch_tokens/ep) × d_model × dtype_bytes.
    /// Uses nvlink_bw (NVLink switch fabric; all sends happen in parallel, bandwidth-dominated).
    pub fn ep_all_to_all_latency(&self, batch_tokens: u64, d_model: u32, dtype_bytes: u32) -> f64 {
        if self.ep <= 1 || self.nvlink_bw <= 0.0 {
            return 0.0;
        }
        let tokens_per_gpu = (batch_tokens / self.ep as u64).max(1);
        let msg_bytes = tokens_per_gpu * d_model as u64 * dtype_bytes as u64;
        (self.ep - 1) as f64 / self.ep as f64 * msg_bytes as f64 / self.nvlink_bw
    }
}
