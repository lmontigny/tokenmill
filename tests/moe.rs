//! MoE accuracy model — active params, MLA KV compression, expert weight maths.

use tokenmill::hardware::cluster::ClusterConfig;
use tokenmill::hardware::gpu::GpuSpec;
use tokenmill::model::llm_config::LlmConfig;

#[test]
fn dense_models_have_full_active_param_fraction() {
    let llama8b = LlmConfig::preset("llama-8b").unwrap();
    assert!(!llama8b.is_moe());
    assert_eq!(llama8b.active_param_fraction(), 1.0);
    assert_eq!(llama8b.weight_bytes_active(), llama8b.weight_bytes);
}

#[test]
fn mixtral_active_params_match_published_value() {
    // Mixtral 8×7B is published as ~12.9 B active per token (attn + 2 of 8 experts).
    let mix = LlmConfig::preset("mixtral-8x7b").unwrap();
    assert!(mix.is_moe());

    let active_bytes = mix.weight_bytes_active();
    let active_b = active_bytes as f64 / 1e9;
    assert!(
        active_b > 25.0 && active_b < 28.0,
        "Mixtral active bytes ≈ 26.8 GB (12.9B × 2 bytes), got {:.2} GB",
        active_b
    );
}

#[test]
fn deepseek_v3_uses_mla_kv() {
    // MLA KV: n_layers × kv_lora_rank × seq_len × dtype, vs 2 × n_layers × n_kv_heads × head_dim × seq_len.
    // For DSV3: 61 × 512 × T × 1 bytes  vs  2 × 61 × 128 × 128 × T × 1
    // Ratio = 2 × 128 × 128 / 512 = 64×.
    let dsv3 = LlmConfig::preset("deepseek-v3").unwrap();
    assert_eq!(dsv3.kv_lora_rank, 512);

    let mla_kv = dsv3.kv_bytes(1024);
    // Equivalent MHA would be 2 × 61 × 128 × 128 × 1024 × 1
    let equiv_mha = 2u64 * 61 * 128 * 128 * 1024;
    let ratio = equiv_mha as f64 / mla_kv as f64;

    assert!(
        (ratio - 64.0).abs() < 0.5,
        "expected 64× smaller KV vs MHA, got {:.2}×",
        ratio
    );
}

#[test]
fn ep_shards_expert_weights() {
    // Decode latency on DSV3 should decrease meaningfully with EP — expert weights are
    // ~15/37 ≈ 40% of active bytes, so EP=8 should give ~1.4× speedup.
    let gpu = GpuSpec::preset("h100").unwrap();
    let model = LlmConfig::preset("deepseek-v3").unwrap();

    let mut ep1 = ClusterConfig::single_gpu();
    ep1.scale_up_bw = gpu.scale_up_bandwidth;

    let mut ep8 = ClusterConfig::single_gpu();
    ep8.ep = 8;
    ep8.scale_up_bw = gpu.scale_up_bandwidth;

    let t_ep1 = gpu.decode_latency(1, 256, &model, None, &ep1);
    let t_ep8 = gpu.decode_latency(1, 256, &model, None, &ep8);

    assert!(t_ep8 < t_ep1, "EP=8 should be faster than EP=1");
    assert!(
        t_ep1 / t_ep8 > 1.3,
        "EP=8 vs EP=1 on DSV3: expected >1.3× speedup, got {:.2}×",
        t_ep1 / t_ep8
    );
}

#[test]
fn kimi_k2_uses_mla_kv() {
    // Kimi K2 reuses DeepSeek V3's MLA trick — kv_lora_rank=512.
    let k2 = LlmConfig::preset("kimi-k2").unwrap();
    assert!(k2.is_moe());
    assert_eq!(k2.kv_lora_rank, 512);
    // 1T total, 32B active, ~3% activation.
    let total_t = k2.weight_bytes as f64 / 1e12;
    let active_b = k2.weight_bytes_active() as f64 / 1e9;
    assert!(
        total_t > 1.0 && total_t < 1.1,
        "expected ~1.0 T total, got {:.2}",
        total_t
    );
    assert!(
        active_b > 30.0 && active_b < 35.0,
        "expected ~32 B active, got {:.1}",
        active_b
    );
}

#[test]
fn behemoth_active_params_dominate_decode_cost() {
    // Behemoth (2 T / 288 B active) vs Kimi K2 (1 T / 32 B active) on the same cluster.
    // Decode is memory-bound on active params → Behemoth should be ~9× slower per step.
    let gpu = GpuSpec::preset("b200").unwrap();
    let k2 = LlmConfig::preset("kimi-k2").unwrap();
    let beh = LlmConfig::preset("llama4-behemoth").unwrap();
    let mut c = ClusterConfig::single_gpu();
    c.tp = 16;
    c.ep = 16;
    c.scale_up_bw = gpu.scale_up_bandwidth;

    let t_k2 = gpu.decode_latency(1, 256, &k2, None, &c);
    let t_beh = gpu.decode_latency(1, 256, &beh, None, &c);

    let ratio = t_beh / t_k2;
    assert!(
        ratio > 4.0,
        "Behemoth (288 B active) should be much slower than K2 (32 B); got {:.2}×",
        ratio
    );
}

#[test]
fn ep_all_to_all_zero_when_ep_eq_one() {
    let mut c = ClusterConfig::single_gpu();
    c.scale_up_bw = 900e9;
    c.ep = 1;
    assert_eq!(c.ep_all_to_all_latency(1024, 4096, 2), 0.0);
}

#[test]
fn ep_all_to_all_scales_with_token_count() {
    let mut c = ClusterConfig::single_gpu();
    c.scale_up_bw = 900e9;
    c.ep = 8;

    let small = c.ep_all_to_all_latency(64, 4096, 2);
    let big = c.ep_all_to_all_latency(640, 4096, 2);

    let ratio = big / small;
    assert!(
        ratio > 9.0 && ratio < 11.0,
        "expected ~10× scaling, got {:.3}",
        ratio
    );
}

#[test]
fn expert_weight_bytes_active_is_subset_of_total_active() {
    // Active expert bytes must not exceed total active bytes.
    for name in &["mixtral-8x7b", "llama4-maverick", "deepseek-v3"] {
        let m = LlmConfig::preset(name).unwrap();
        assert!(
            m.expert_weight_bytes_active() <= m.weight_bytes_active(),
            "{}: expert weights exceed total active weights",
            name
        );
        assert!(
            m.expert_weight_bytes_active() > 0,
            "{}: expert weights should be non-zero for MoE",
            name
        );
    }
}
