/// Parallelism configuration for multi-GPU inference.
///
/// TP (tensor parallelism): all tp_degree GPUs work on every layer together.
///   - Each GPU holds 1/tp_degree of weights → compute and memory BW scale linearly.
///   - After attention and after FFN: one all-reduce collective per layer.
///   - Ring-allreduce cost = 2*(N-1) * (α + (msg/N) / β)
///     where α = per-hop link latency and β = scale-up fabric bandwidth.
///
/// PP (pipeline parallelism): model split across pp_degree stage groups.
///   - Each stage handles n_layers/pp_degree layers.
///   - Stages execute sequentially for one request; overlap emerges from multiple requests
///     in flight (pipeline fill), which the DES models naturally via event timing.
///   - Inter-stage transfer cost = α + activation_bytes / β.
///
/// EP (expert parallelism): MoE experts sharded across ep_degree GPUs. When
/// `scale_out_bw` is set and EP exceeds `gpus_per_node`, cross-node expert
/// traffic uses the scale-out network rather than the scale-up fabric.
///   - Each GPU holds n_experts/ep experts; tokens are dispatched over scale-up or scale-out.
///   - Two all-to-alls per MoE layer: dispatch (send top_K token activations out) + combine (gather).
///   - All-to-all data per GPU per direction ≈ (ep-1)/ep × (top_K × batch / ep) × d_model × activation_bytes.
///   - Fully-connected fabrics (NVSwitch, OCS) give full bisection BW; ring/torus fabrics
///     get the same form with per-hop latency dominating at small messages.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub tp: u32,
    pub pp: u32,
    /// Expert parallelism degree for MoE models (1 = no EP, all experts on one GPU).
    pub ep: u32,
    /// Scale-up fabric bandwidth per direction in bytes/sec (NVLink, Infinity Fabric, ICI, Groq C2C).
    /// Used for all-reduce, EP all-to-all, and intra-node PP transfers.
    pub scale_up_bw: f64,
    /// Per-hop latency on the scale-up fabric in seconds (typically 100 ns – 2 µs).
    /// Dominates collective cost for small messages and very large TP (e.g. Groq at TP=358).
    /// 0 = pure bandwidth model (the old behaviour).
    pub scale_up_latency: f64,
    /// Number of accelerators in one scale-up node/server. Parallel groups larger than this
    /// cross the scale-out network when `scale_out_bw` is configured.
    pub gpus_per_node: u32,
    /// Scale-out network bandwidth in bytes/sec (InfiniBand, RoCE, Ethernet). 0 keeps the
    /// legacy uniform scale-up model for all collectives.
    pub scale_out_bw: f64,
    /// One-way scale-out network latency in seconds.
    pub scale_out_latency: f64,
    /// Cross-node bandwidth in bytes/sec (PP inter-server and KV transfer in disaggregated PD).
    pub internode_bw: f64,
    /// If true, prefill and decode run on separate GPU pools connected over internode_bw.
    pub disaggregate: bool,
}

impl ClusterConfig {
    pub fn single_gpu() -> Self {
        Self {
            tp: 1,
            pp: 1,
            ep: 1,
            scale_up_bw: 0.0,
            scale_up_latency: 0.0,
            gpus_per_node: 8,
            scale_out_bw: 0.0,
            scale_out_latency: 0.0,
            internode_bw: 0.0,
            disaggregate: false,
        }
    }

    fn ring_latency(n: u32, msg_bytes: u64, bw: f64, latency: f64) -> f64 {
        if n <= 1 || bw <= 0.0 {
            return 0.0;
        }
        let n = n as f64;
        let hops = 2.0 * (n - 1.0);
        let chunk = msg_bytes as f64 / n;
        hops * (latency + chunk / bw)
    }

    fn spans_scale_out(&self, degree: u32) -> bool {
        self.scale_out_bw > 0.0 && degree > self.gpus_per_node.max(1)
    }

    /// KV transfer latency from prefill node to decode node (seconds).
    pub fn kv_transfer_latency(&self, kv_bytes: u64) -> f64 {
        if !self.disaggregate || self.internode_bw <= 0.0 {
            return 0.0;
        }
        kv_bytes as f64 / self.internode_bw
    }

    /// Ring-allreduce latency (seconds): 2*(N-1) hops, each paying α + chunk/β.
    /// At small message sizes, the α term dominates and scales linearly with N.
    pub fn all_reduce_latency(&self, msg_bytes: u64) -> f64 {
        if self.tp <= 1 || self.scale_up_bw <= 0.0 {
            return 0.0;
        }

        if !self.spans_scale_out(self.tp) {
            return Self::ring_latency(self.tp, msg_bytes, self.scale_up_bw, self.scale_up_latency);
        }

        // Hierarchical approximation: reduce within each node, all-reduce across nodes,
        // then broadcast within each node. This captures the large Ethernet/IB penalty
        // without requiring a full topology graph.
        let local = self.gpus_per_node.max(1).min(self.tp);
        let nodes = self.tp.div_ceil(self.gpus_per_node.max(1));
        2.0 * Self::ring_latency(local, msg_bytes, self.scale_up_bw, self.scale_up_latency)
            + Self::ring_latency(nodes, msg_bytes, self.scale_out_bw, self.scale_out_latency)
    }

    /// Point-to-point activation transfer latency between PP stages (seconds).
    /// Single hop: α + bytes/β.
    pub fn pp_transfer_latency(&self, activation_bytes: u64, intranode: bool) -> f64 {
        if self.pp <= 1 {
            return 0.0;
        }
        let scale_out_bw = if self.scale_out_bw > 0.0 {
            self.scale_out_bw
        } else {
            self.internode_bw
        };
        let (bw, latency) = if intranode {
            if self.spans_scale_out(self.pp) {
                let boundaries = (self.pp - 1) as f64;
                let cross_boundaries = ((self.pp - 1) / self.gpus_per_node.max(1)) as f64;
                let local_boundaries = boundaries - cross_boundaries;
                let local =
                    self.scale_up_latency + activation_bytes as f64 / self.scale_up_bw.max(1.0);
                let remote =
                    self.scale_out_latency + activation_bytes as f64 / scale_out_bw.max(1.0);
                return (local_boundaries * local + cross_boundaries * remote) / boundaries;
            }
            (self.scale_up_bw, self.scale_up_latency)
        } else {
            (scale_out_bw, self.scale_out_latency)
        };
        if bw <= 0.0 {
            return 0.0;
        }
        latency + activation_bytes as f64 / bw
    }

    /// All-to-all latency for one direction of expert dispatch or combine (seconds).
    /// Called twice per MoE layer: once to send tokens to expert GPUs, once to gather results.
    ///
    /// `batch_tokens` must already be multiplied by top_K at the call site (caller knows the model).
    /// Per-GPU send volume = (ep-1)/ep × (batch_tokens/ep) × d_model × activation_bytes.
    /// One α per all-to-all (dominant at very small messages or very high EP).
    pub fn ep_all_to_all_latency(
        &self,
        batch_tokens: u64,
        d_model: u32,
        activation_bytes: u32,
    ) -> f64 {
        if self.ep <= 1 || self.scale_up_bw <= 0.0 {
            return 0.0;
        }
        let tokens_per_gpu = (batch_tokens / self.ep as u64).max(1);
        let msg_bytes = tokens_per_gpu * d_model as u64 * activation_bytes as u64;
        let traffic_frac = (self.ep - 1) as f64 / self.ep as f64;

        if !self.spans_scale_out(self.ep) {
            return self.scale_up_latency + traffic_frac * msg_bytes as f64 / self.scale_up_bw;
        }

        let local_ep = self.gpus_per_node.max(1).min(self.ep);
        let local_frac = (local_ep - 1) as f64 / self.ep as f64;
        let remote_frac = (self.ep - local_ep) as f64 / self.ep as f64;
        self.scale_up_latency
            + local_frac * msg_bytes as f64 / self.scale_up_bw
            + self.scale_out_latency
            + remote_frac * msg_bytes as f64 / self.scale_out_bw
    }
}
