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
}

impl LlmConfig {
    pub fn kv_bytes(&self, seq_len: u32) -> u64 {
        2 * self.n_layers as u64
            * self.n_kv_heads as u64
            * self.head_dim as u64
            * seq_len as u64
            * self.dtype_bytes as u64
    }

    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "llama-70b" => Some(Self {
                name: "llama-70b".into(),
                n_layers: 80,
                d_model: 8192,
                n_heads: 64,
                n_kv_heads: 8,
                head_dim: 128,
                ffn_hidden: 28672,
                vocab_size: 128256,
                dtype_bytes: 2,
                weight_bytes: 140_000_000_000,
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
                dtype_bytes: 2,
                weight_bytes: 16_000_000_000,
            }),
            "mixtral-8x7b" => Some(Self {
                name: "mixtral-8x7b".into(),
                n_layers: 32,
                d_model: 4096,
                n_heads: 32,
                n_kv_heads: 8,
                head_dim: 128,
                ffn_hidden: 14336,
                vocab_size: 32000,
                dtype_bytes: 2,
                weight_bytes: 93_000_000_000,
            }),
            _ => None,
        }
    }
}
