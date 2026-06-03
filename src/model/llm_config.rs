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
    /// Bits per weight (4 = NVFP4 / W4 quant, 8 = FP8 / INT8, 16 = BF16/FP16, 32 = FP32).
    /// Drives weight memory + decode HBM traffic + selects the matching FLOPS tier on GpuSpec.
    pub weight_bits: u32,
    /// Bits per activation used for collectives and pipeline/expert transfers.
    /// Defaults to weight_bits when 0, but production configs often use W4A16 or W4A8.
    #[serde(default)]
    pub activation_bits: u32,
    /// Bits per KV cache entry. Defaults to weight_bits if 0 (most production configs
    /// quantise weights and KV together, but mixed schemes like W4A8KV4 set this separately).
    #[serde(default)]
    pub kv_bits: u32,
    /// Structured-sparsity speedup factor on supporting hardware (1.0 = dense, 2.0 = 2:4 sparse).
    /// Only applies when GpuSpec.supports_2to4_sparsity is true and the model has been pruned.
    /// Defaults to 1.0 (dense).
    #[serde(default = "default_sparsity")]
    pub weight_sparsity: f64,
    pub weight_bytes: u64,

    // MoE topology (all 0 = dense model)
    #[serde(default)]
    pub n_experts: u32, // total routable experts per MoE layer
    #[serde(default)]
    pub n_active_experts: u32, // top-K selected per token
    #[serde(default)]
    pub n_shared_experts: u32, // always-active (DeepSeek-style shared experts)
    #[serde(default)]
    pub n_moe_layers: u32, // layers using MoE FFN; rest are dense
    #[serde(default)]
    pub expert_hidden: u32, // per-expert FFN hidden dim (0 = same as ffn_hidden)

    // Multi-head Latent Attention KV compression (DeepSeek V3/R1).
    // When > 0, KV cache is compressed to kv_lora_rank dims per layer instead of n_kv_heads × head_dim.
    #[serde(default)]
    pub kv_lora_rank: u32,

    // Exact active weight bytes accessed per decode step (0 = derive from active_param_fraction).
    // Set this for MoE presets to avoid the 1/3-attn / 2/3-FFN approximation error.
    #[serde(default)]
    pub active_weight_bytes: u64,
}

fn default_sparsity() -> f64 {
    1.0
}

impl LlmConfig {
    /// Effective KV bits per entry (falls back to weight_bits when kv_bits is unset).
    pub fn effective_kv_bits(&self) -> u32 {
        if self.kv_bits > 0 {
            self.kv_bits
        } else {
            self.weight_bits
        }
    }

    /// Effective activation bits per value (falls back to weight_bits when unset).
    pub fn effective_activation_bits(&self) -> u32 {
        if self.activation_bits > 0 {
            self.activation_bits
        } else {
            self.weight_bits
        }
    }

    /// Bytes per parameter, derived from weight_bits. FP4 → 0.5, FP8 → 1.0, BF16 → 2.0.
    pub fn weight_bytes_per_param(&self) -> f64 {
        self.weight_bits as f64 / 8.0
    }

    /// Bytes per KV entry, derived from effective_kv_bits.
    pub fn kv_bytes_per_entry(&self) -> f64 {
        self.effective_kv_bits() as f64 / 8.0
    }

    /// Byte width used for activations and collective messages.
    ///
    /// The current model tracks weight and KV precision separately; activations
    /// still need an integer byte width for communication formulas. FP4 weights
    /// therefore use 1 byte here until activation precision is modelled directly.
    pub fn activation_bytes(&self) -> u32 {
        ((self.effective_activation_bits() as f64 / 8.0).ceil() as u32).max(1)
    }

    pub fn with_quantization(
        &self,
        name: &str,
        weight_bits: u32,
        activation_bits: u32,
        kv_bits: u32,
        weight_sparsity: f64,
    ) -> Self {
        let mut model = self.clone();
        let old_weight_bytes_per_param = self.weight_bytes_per_param();
        let new_weight_bytes_per_param = weight_bits as f64 / 8.0;
        let scale = new_weight_bytes_per_param / old_weight_bytes_per_param;

        model.name = name.into();
        model.weight_bits = weight_bits;
        model.activation_bits = activation_bits;
        model.kv_bits = kv_bits;
        model.weight_sparsity = weight_sparsity;
        model.weight_bytes = (self.weight_bytes as f64 * scale) as u64;
        if self.active_weight_bytes > 0 {
            model.active_weight_bytes = (self.active_weight_bytes as f64 * scale) as u64;
        }
        model
    }
}

impl LlmConfig {
    pub fn is_moe(&self) -> bool {
        self.n_experts > 1
    }

    /// Active expert-weight bytes per forward pass — the portion sharded by EP, not TP.
    /// For dense models returns 0 (all weights shard by TP).
    /// Formula: n_moe_layers × (n_active_experts + n_shared_experts) × 2 × d_model × expert_hidden × weight_bytes_per_param.
    /// The factor of 2 covers gate+down projections (consistent with the 2N FLOPs roofline rule).
    pub fn expert_weight_bytes_active(&self) -> u64 {
        if !self.is_moe() || self.n_moe_layers == 0 {
            return 0;
        }
        let hidden = if self.expert_hidden > 0 {
            self.expert_hidden
        } else {
            self.ffn_hidden
        };
        let per_expert_params = 2u64 * self.d_model as u64 * hidden as u64;
        let total_params = self.n_moe_layers as u64
            * (self.n_active_experts + self.n_shared_experts) as u64
            * per_expert_params;
        (total_params as f64 * self.weight_bytes_per_param()) as u64
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
    /// Bytes per entry come from `effective_kv_bits()` — supports KV quantisation
    /// (e.g. W4A8KV4 sets kv_bits=4 separately from weight_bits).
    pub fn kv_bytes(&self, seq_len: u32) -> u64 {
        let entries = if self.kv_lora_rank > 0 {
            // Compressed latent KV: one vector of kv_lora_rank per layer per token.
            self.n_layers as u64 * self.kv_lora_rank as u64 * seq_len as u64
        } else {
            2 * self.n_layers as u64
                * self.n_kv_heads as u64
                * self.head_dim as u64
                * seq_len as u64
        };
        (entries as f64 * self.kv_bytes_per_entry()) as u64
    }

    pub fn preset(name: &str) -> Option<Self> {
        match name {
            // ── dense models ──────────────────────────────────────────────────
            "llama-70b" => Some(Self {
                name: "llama-70b".into(),
                n_layers: 80,
                d_model: 8192,
                n_heads: 64,
                n_kv_heads: 8,
                head_dim: 128,
                ffn_hidden: 28672,
                vocab_size: 128256,
                weight_bits: 16,
                activation_bits: 0,
                kv_bits: 0,
                weight_sparsity: 1.0,
                weight_bytes: 140_000_000_000,
                n_experts: 0,
                n_active_experts: 0,
                n_shared_experts: 0,
                n_moe_layers: 0,
                expert_hidden: 0,
                kv_lora_rank: 0,
                active_weight_bytes: 0,
            }),
            "llama-8b" => Some(Self {
                name: "llama-8b".into(),
                n_layers: 32,
                d_model: 4096,
                n_heads: 32,
                n_kv_heads: 8,
                head_dim: 128,
                ffn_hidden: 14336,
                vocab_size: 128256,
                weight_bits: 16,
                activation_bits: 0,
                kv_bits: 0,
                weight_sparsity: 1.0,
                weight_bytes: 16_000_000_000,
                n_experts: 0,
                n_active_experts: 0,
                n_shared_experts: 0,
                n_moe_layers: 0,
                expert_hidden: 0,
                kv_lora_rank: 0,
                active_weight_bytes: 0,
            }),
            // ── FP8 dense variants (for validation against NIM/TRT-LLM benchmarks) ──
            // Same architecture as BF16 counterparts; weight_bits=8 halves weight size.
            "llama-70b-fp8" => Some(Self {
                name: "llama-70b-fp8".into(),
                n_layers: 80,
                d_model: 8192,
                n_heads: 64,
                n_kv_heads: 8,
                head_dim: 128,
                ffn_hidden: 28672,
                vocab_size: 128256,
                weight_bits: 8,
                activation_bits: 0,
                kv_bits: 0,
                weight_sparsity: 1.0,
                weight_bytes: 70_000_000_000,
                n_experts: 0,
                n_active_experts: 0,
                n_shared_experts: 0,
                n_moe_layers: 0,
                expert_hidden: 0,
                kv_lora_rank: 0,
                active_weight_bytes: 0,
            }),
            "llama-8b-fp8" => Some(Self {
                name: "llama-8b-fp8".into(),
                n_layers: 32,
                d_model: 4096,
                n_heads: 32,
                n_kv_heads: 8,
                head_dim: 128,
                ffn_hidden: 14336,
                vocab_size: 128256,
                weight_bits: 8,
                activation_bits: 0,
                kv_bits: 0,
                weight_sparsity: 1.0,
                weight_bytes: 8_000_000_000,
                n_experts: 0,
                n_active_experts: 0,
                n_shared_experts: 0,
                n_moe_layers: 0,
                expert_hidden: 0,
                kv_lora_rank: 0,
                active_weight_bytes: 0,
            }),
            // ── MoE models ────────────────────────────────────────────────────
            // Mixtral 8×7B: 8 experts, top-2, all 32 layers are MoE.
            // 93 GB bf16 total; ~27 GB active (attn + 2/8 expert sets).
            "mixtral-8x7b" => Some(Self {
                name: "mixtral-8x7b".into(),
                n_layers: 32,
                d_model: 4096,
                n_heads: 32,
                n_kv_heads: 8,
                head_dim: 128,
                ffn_hidden: 14336,
                vocab_size: 32000,
                weight_bits: 16,
                activation_bits: 0,
                kv_bits: 0,
                weight_sparsity: 1.0,
                weight_bytes: 93_000_000_000,
                n_experts: 8,
                n_active_experts: 2,
                n_shared_experts: 0,
                n_moe_layers: 32,
                expert_hidden: 14336,
                kv_lora_rank: 0,
                active_weight_bytes: 26_800_000_000,
            }),
            // Llama 4 Maverick: 400 B total (fp8), 17 B active/token.
            // 128 experts top-1 + 1 shared, 36 of 48 layers are MoE.
            // GQA with n_kv_heads=8.
            "llama4-maverick" => Some(Self {
                name: "llama4-maverick".into(),
                n_layers: 48,
                d_model: 5120,
                n_heads: 40,
                n_kv_heads: 8,
                head_dim: 128,
                ffn_hidden: 8192,
                vocab_size: 128256,
                weight_bits: 8,
                activation_bits: 0,
                kv_bits: 0,
                weight_sparsity: 1.0,
                weight_bytes: 400_000_000_000,
                n_experts: 128,
                n_active_experts: 1,
                n_shared_experts: 1,
                n_moe_layers: 36,
                expert_hidden: 2048,
                kv_lora_rank: 0,
                active_weight_bytes: 17_000_000_000,
            }),
            // DeepSeek V3: 671 B total (fp8), 37 B active/token.
            // 256 experts top-8 + 1 shared, 58 of 61 layers are MoE.
            // Multi-head Latent Attention (MLA) compresses KV to rank-512 per layer (~64× smaller KV cache).
            "deepseek-v3" => Some(Self {
                name: "deepseek-v3".into(),
                n_layers: 61,
                d_model: 7168,
                n_heads: 128,
                n_kv_heads: 128,
                head_dim: 128,
                ffn_hidden: 18432,
                vocab_size: 129280,
                weight_bits: 8,
                activation_bits: 0,
                kv_bits: 0,
                weight_sparsity: 1.0,
                weight_bytes: 671_000_000_000,
                n_experts: 256,
                n_active_experts: 8,
                n_shared_experts: 1,
                n_moe_layers: 58,
                expert_hidden: 2048,
                kv_lora_rank: 512,
                active_weight_bytes: 37_000_000_000,
            }),
            // ── Frontier-class (≥ 1 T total parameters) ──────────────────────
            // Kimi K2 (Moonshot AI, July 2025): 1.026 T total (fp8), 32 B active/token.
            // 384 experts top-8 + 1 shared, 60 of 61 layers are MoE.
            // MLA KV (kv_lora_rank = 512) — same compression trick as DeepSeek V3.
            "kimi-k2" => Some(Self {
                name: "kimi-k2".into(),
                n_layers: 61,
                d_model: 7168,
                n_heads: 64,
                n_kv_heads: 64,
                head_dim: 128,
                ffn_hidden: 18432,
                vocab_size: 163840,
                weight_bits: 8,
                activation_bits: 0,
                kv_bits: 0,
                weight_sparsity: 1.0,
                weight_bytes: 1_026_000_000_000,
                n_experts: 384,
                n_active_experts: 8,
                n_shared_experts: 1,
                n_moe_layers: 60,
                expert_hidden: 2048,
                kv_lora_rank: 512,
                active_weight_bytes: 32_000_000_000,
            }),
            // Llama 4 Behemoth (Meta, announced 2025 — not publicly released):
            // 2 T total (fp8), 288 B active/token. 16 experts top-1 + 1 shared.
            // Architecture extrapolated from Llama 4 Scout/Maverick family;
            // verify against final spec when released.
            "llama4-behemoth" => Some(Self {
                name: "llama4-behemoth".into(),
                n_layers: 80,
                d_model: 8192,
                n_heads: 64,
                n_kv_heads: 8,
                head_dim: 128,
                ffn_hidden: 28672,
                vocab_size: 128256,
                weight_bits: 8,
                activation_bits: 0,
                kv_bits: 0,
                weight_sparsity: 1.0,
                weight_bytes: 2_000_000_000_000,
                n_experts: 16,
                n_active_experts: 1,
                n_shared_experts: 1,
                n_moe_layers: 60,
                // Each expert is large (~120 B params); per-expert hidden is correspondingly wide.
                expert_hidden: 65536,
                kv_lora_rank: 0,
                active_weight_bytes: 288_000_000_000,
            }),
            // ── mixed-precision / quantized serving variants ────────────────
            // W4A16: 4-bit weights, BF16/FP16 activations, KV follows weights.
            "llama-8b-w4a16" => Self::preset("llama-8b")
                .map(|m| m.with_quantization("llama-8b-w4a16", 4, 16, 0, 1.0)),
            "llama-70b-w4a16" => Self::preset("llama-70b")
                .map(|m| m.with_quantization("llama-70b-w4a16", 4, 16, 0, 1.0)),
            // W4A8KV4: 4-bit weights, 8-bit activations, 4-bit KV cache.
            "llama-8b-w4a8kv4" => Self::preset("llama-8b")
                .map(|m| m.with_quantization("llama-8b-w4a8kv4", 4, 8, 4, 1.0)),
            "llama-70b-w4a8kv4" => Self::preset("llama-70b")
                .map(|m| m.with_quantization("llama-70b-w4a8kv4", 4, 8, 4, 1.0)),
            // NVFP4 sparse: dense FP4 storage plus 2:4 structured-sparsity speedup
            // on hardware that advertises `supports_2to4_sparsity`.
            "llama-70b-nvfp4-sparse" => Self::preset("llama-70b")
                .map(|m| m.with_quantization("llama-70b-nvfp4-sparse", 4, 8, 4, 2.0)),
            "kimi-k2-nvfp4-sparse" => Self::preset("kimi-k2")
                .map(|m| m.with_quantization("kimi-k2-nvfp4-sparse", 4, 8, 4, 2.0)),
            _ => None,
        }
    }
}
