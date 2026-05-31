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
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub tp: u32,
    pub pp: u32,
    /// NVLink bandwidth per direction in bytes/sec (all-reduce and intranode PP transfers).
    pub nvlink_bw: f64,
    /// Cross-node bandwidth in bytes/sec (PP inter-server and KV transfer in disaggregated PD).
    pub internode_bw: f64,
    /// If true, prefill and decode run on separate GPU pools connected over internode_bw.
    pub disaggregate: bool,
}

impl ClusterConfig {
    pub fn single_gpu() -> Self {
        Self { tp: 1, pp: 1, nvlink_bw: 0.0, internode_bw: 0.0, disaggregate: false }
    }

    /// KV transfer latency from prefill node to decode node (seconds).
    /// kv_bytes: size of the KV cache for the completed prompt.
    pub fn kv_transfer_latency(&self, kv_bytes: u64) -> f64 {
        if !self.disaggregate || self.internode_bw <= 0.0 {
            return 0.0;
        }
        kv_bytes as f64 / self.internode_bw
    }

    /// All-reduce latency for ring-allreduce over NVLink (seconds).
    /// msg_bytes is the per-GPU message size before reduction.
    pub fn all_reduce_latency(&self, msg_bytes: u64) -> f64 {
        if self.tp <= 1 || self.nvlink_bw <= 0.0 {
            return 0.0;
        }
        let factor = 2.0 * (self.tp - 1) as f64 / self.tp as f64;
        factor * msg_bytes as f64 / self.nvlink_bw
    }

    /// Point-to-point activation transfer latency between PP stages (seconds).
    /// activation_bytes: size of the tensor passed between stages.
    /// intranode: true if both stages are on the same server (use NVLink), false for IB.
    pub fn pp_transfer_latency(&self, activation_bytes: u64, intranode: bool) -> f64 {
        if self.pp <= 1 {
            return 0.0;
        }
        let bw = if intranode { self.nvlink_bw } else { self.internode_bw };
        if bw <= 0.0 { return 0.0; }
        activation_bytes as f64 / bw
    }
}
