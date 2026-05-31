use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    pub name: String,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub ffn_hidden: u32,
    pub vocab_size: u32,
    pub dtype_bytes: u32,
    pub weight_bytes: u64,

    // MoE topology (all 0 = dense model)
    #[serde(default)] pub n_experts: u32,        // total routable experts per MoE layer
    #[serde(default)] pub n_active_experts: u32, // top-K selected per token
    #[serde(default)] pub n_shared_experts: u32, // always-active (DeepSeek-style shared experts)
    #[serde(default)] pub n_moe_layers: u32,     // layers using MoE FFN; rest are dense
    #[serde(default)] pub expert_hidden: u32,    // per-expert FFN hidden dim (0 = same as ffn_hidden)

    // Multi-head Latent Attention KV compression (DeepSeek V3/R1).
    // When > 0, KV cache is compressed to kv_lora_rank dims per layer instead of n_kv_heads × head_dim.
    #[serde(default)] pub kv_lora_rank: u32,

    // Exact active weight bytes accessed per decode step (0 = derive from active_param_fraction).
    // Set this for MoE presets to avoid the 1/3-attn / 2/3-FFN approximation error.
    #[serde(default)] pub active_weight_bytes: u64,
}

impl LlmConfig {
    pub fn is_moe(&self) -> bool {
        self.n_experts > 1
    }

    /// Fraction of model weights active per forward pass per token.
    /// For dense: 1.0. For MoE: uses active_weight_bytes/weight_bytes when available,
    /// otherwise falls back to a formula (attn always-on + sparse FFN routing).
    pub fn active_param_fraction(&self) -> f64 {
        if !self.is_moe() {
            return 1.0;
        }
        if self.active_weight_bytes > 0 && self.weight_bytes > 0 {
            return self.active_weight_bytes as f64 / self.weight_bytes as f64;
        }
        // Fallback: rough split — ~1/3 attention (always active) + ~2/3 FFN (sparse in MoE layers).
        let total_experts = (self.n_experts + self.n_shared_experts) as f64;
        let active_experts = (self.n_active_experts + self.n_shared_experts) as f64;
        let expert_ratio = active_experts / total_experts;
        let moe_frac = self.n_moe_layers as f64 / self.n_layers as f64;
        let dense_frac = 1.0 - moe_frac;
        1.0 / 3.0 + 2.0 / 3.0 * (dense_frac + moe_frac * expert_ratio)
    }

    /// Weight bytes accessed per decode step (uses sparsity for MoE).
    pub fn weight_bytes_active(&self) -> u64 {
        if self.active_weight_bytes > 0 {
            self.active_weight_bytes
        } else if self.is_moe() {
            (self.weight_bytes as f64 * self.active_param_fraction()) as u64
        } else {
            self.weight_bytes
        }
    }

    /// KV cache bytes for a single request with `seq_len` tokens.
    /// Uses MLA compression when kv_lora_rank > 0 (e.g. DeepSeek V3).
    pub fn kv_bytes(&self, seq_len: u32) -> u64 {
        if self.kv_lora_rank > 0 {
            // Compressed latent KV: one vector of kv_lora_rank per layer per token.
            self.n_layers as u64
                * self.kv_lora_rank as u64
                * seq_len as u64
                * self.dtype_bytes as u64
        } else {
            2 * self.n_layers as u64
                * self.n_kv_heads as u64
                * self.head_dim as u64
                * seq_len as u64
                * self.dtype_bytes as u64
        }
    }

    pub fn preset(name: &str) -> Option<Self> {
        match name {
            // ── dense models ──────────────────────────────────────────────────
            "llama-70b" => Some(Self {
                name: "llama-70b".into(),
                n_layers: 80, d_model: 8192, n_heads: 64, n_kv_heads: 8,
                head_dim: 128, ffn_hidden: 28672, vocab_size: 128256,
                dtype_bytes: 2, weight_bytes: 140_000_000_000,
                n_experts: 0, n_active_experts: 0, n_shared_experts: 0,
                n_moe_layers: 0, expert_hidden: 0, kv_lora_rank: 0,
                active_weight_bytes: 0,
            }),
            "llama-8b" => Some(Self {
                name: "llama-8b".into(),
                n_layers: 32, d_model: 4096, n_heads: 32, n_kv_heads: 8,
                head_dim: 128, ffn_hidden: 14336, vocab_size: 128256,
                dtype_bytes: 2, weight_bytes: 16_000_000_000,
                n_experts: 0, n_active_experts: 0, n_shared_experts: 0,
                n_moe_layers: 0, expert_hidden: 0, kv_lora_rank: 0,
                active_weight_bytes: 0,
            }),
            // ── MoE models ────────────────────────────────────────────────────
            // Mixtral 8×7B: 8 experts, top-2, all 32 layers are MoE.
            // 93 GB bf16 total; ~27 GB active (attn + 2/8 expert sets).
            "mixtral-8x7b" => Some(Self {
                name: "mixtral-8x7b".into(),
                n_layers: 32, d_model: 4096, n_heads: 32, n_kv_heads: 8,
                head_dim: 128, ffn_hidden: 14336, vocab_size: 32000,
                dtype_bytes: 2, weight_bytes: 93_000_000_000,
                n_experts: 8, n_active_experts: 2, n_shared_experts: 0,
                n_moe_layers: 32, expert_hidden: 14336, kv_lora_rank: 0,
                active_weight_bytes: 26_800_000_000,
            }),
            // Llama 4 Maverick: 400 B total (fp8), 17 B active/token.
            // 128 experts top-1 + 1 shared, 36 of 48 layers are MoE.
            // GQA with n_kv_heads=8.
            "llama4-maverick" => Some(Self {
                name: "llama4-maverick".into(),
                n_layers: 48, d_model: 5120, n_heads: 40, n_kv_heads: 8,
                head_dim: 128, ffn_hidden: 8192, vocab_size: 128256,
                dtype_bytes: 1, weight_bytes: 400_000_000_000,
                n_experts: 128, n_active_experts: 1, n_shared_experts: 1,
                n_moe_layers: 36, expert_hidden: 2048, kv_lora_rank: 0,
                active_weight_bytes: 17_000_000_000,
            }),
            // DeepSeek V3: 671 B total (fp8), 37 B active/token.
            // 256 experts top-8 + 1 shared, 58 of 61 layers are MoE.
            // Multi-head Latent Attention (MLA) compresses KV to rank-512 per layer (~64× smaller KV cache).
            "deepseek-v3" => Some(Self {
                name: "deepseek-v3".into(),
                n_layers: 61, d_model: 7168, n_heads: 128, n_kv_heads: 128,
                head_dim: 128, ffn_hidden: 18432, vocab_size: 129280,
                dtype_bytes: 1, weight_bytes: 671_000_000_000,
                n_experts: 256, n_active_experts: 8, n_shared_experts: 1,
                n_moe_layers: 58, expert_hidden: 2048, kv_lora_rank: 512,
                active_weight_bytes: 37_000_000_000,
            }),
            _ => None,
        }
    }
}
