//! Roofline GPU latency model.
//!
//! Cross-checks the closed-form latency against hand-derived numbers and
//! verifies the scaling behaviour we rely on (FP8 dispatch, TP scaling,
//! all-reduce overhead).

use tokenmill::hardware::cluster::ClusterConfig;
use tokenmill::hardware::gpu::GpuSpec;
use tokenmill::model::llm_config::LlmConfig;

fn close(actual: f64, expected: f64, rel_tol: f64) -> bool {
    (actual - expected).abs() / expected.abs() <= rel_tol
}

#[test]
fn h100_llama8b_bf16_decode_matches_hand_calculation() {
    // weight=16 GB, batch=1 means KV is negligible.
    // latency = 16e9 / (3.35e12 * 0.80) ≈ 5.97 ms
    let gpu = GpuSpec::preset("h100").unwrap();
    let model = LlmConfig::preset("llama-8b").unwrap();
    let cluster = ClusterConfig::single_gpu();

    let ms = gpu.decode_latency(1, 256, &model, None, &cluster) * 1000.0;
    assert!(close(ms, 5.97, 0.05), "expected ~5.97 ms, got {:.3}", ms);
}

#[test]
fn fp8_decode_is_half_of_bf16_for_same_param_count() {
    // FP8 halves weight bytes → decode is memory-bound → should be ~2× faster.
    let gpu = GpuSpec::preset("h100").unwrap();
    let bf16 = LlmConfig::preset("llama-8b").unwrap();
    let fp8 = LlmConfig::preset("llama-8b-fp8").unwrap();
    let c = ClusterConfig::single_gpu();

    let bf16_ms = gpu.decode_latency(1, 256, &bf16, None, &c) * 1000.0;
    let fp8_ms = gpu.decode_latency(1, 256, &fp8, None, &c) * 1000.0;

    let ratio = bf16_ms / fp8_ms;
    assert!(
        close(ratio, 2.0, 0.02),
        "expected ~2.0× speedup, got {:.3}",
        ratio
    );
}

#[test]
fn fp8_prefill_uses_fp8_flops_on_h100() {
    // H100 has 989 TFLOPS BF16 vs 1978 TFLOPS FP8. Same param count → 2× faster prefill.
    let gpu = GpuSpec::preset("h100").unwrap();
    let bf16 = LlmConfig::preset("llama-8b").unwrap();
    let fp8 = LlmConfig::preset("llama-8b-fp8").unwrap();
    let c = ClusterConfig::single_gpu();

    let bf16_ms = gpu.prefill_latency(1, 512, &bf16, None, &c) * 1000.0;
    let fp8_ms = gpu.prefill_latency(1, 512, &fp8, None, &c) * 1000.0;

    let ratio = bf16_ms / fp8_ms;
    assert!(
        close(ratio, 2.0, 0.02),
        "expected 2× from FP8 tensor cores, got {:.3}",
        ratio
    );
}

#[test]
fn tp_scales_decode_inversely_with_degree() {
    // With TP=N, single-GPU latency should drop by ~N (memory-bound; ignoring all-reduce).
    let gpu = GpuSpec::preset("h100").unwrap();
    let model = LlmConfig::preset("llama-70b").unwrap();
    let tp1 = ClusterConfig::single_gpu();
    let mut tp4 = ClusterConfig::single_gpu();
    tp4.tp = 4;
    tp4.nvlink_bw = gpu.nvlink_bandwidth;

    let t1 = gpu.decode_latency(1, 256, &model, None, &tp1);
    let t4 = gpu.decode_latency(1, 256, &model, None, &tp4);

    // Allow 5% slack for all-reduce overhead at small message sizes.
    let speedup = t1 / t4;
    assert!(
        speedup > 3.8 && speedup < 4.1,
        "expected ~4× speedup, got {:.3}",
        speedup
    );
}

#[test]
fn b200_is_faster_than_h100_for_fp8() {
    // B200 has 2× FP8 TFLOPS and 2.4× HBM BW; expect ~2-2.5× faster on FP8 workloads.
    let h100 = GpuSpec::preset("h100").unwrap();
    let b200 = GpuSpec::preset("b200").unwrap();
    let model = LlmConfig::preset("llama-8b-fp8").unwrap();
    let c = ClusterConfig::single_gpu();

    let h100_decode = h100.decode_latency(1, 256, &model, None, &c);
    let b200_decode = b200.decode_latency(1, 256, &model, None, &c);
    let h100_prefill = h100.prefill_latency(1, 512, &model, None, &c);
    let b200_prefill = b200.prefill_latency(1, 512, &model, None, &c);

    assert!(b200_decode < h100_decode, "B200 should beat H100 on decode");
    assert!(
        b200_prefill < h100_prefill,
        "B200 should beat H100 on prefill"
    );
}

#[test]
fn mi300x_beats_h100_on_decode_despite_lower_mfu() {
    // MI300X: 5.3 TB/s × 0.72 MFU = 3.82 TB/s effective. H100: 3.35 × 0.80 = 2.68. ~1.4× faster.
    let h100 = GpuSpec::preset("h100").unwrap();
    let mi300x = GpuSpec::preset("mi300x").unwrap();
    let model = LlmConfig::preset("llama-70b-fp8").unwrap();
    let c = ClusterConfig::single_gpu();

    let h100_ms = h100.decode_latency(1, 256, &model, None, &c);
    let mi300x_ms = mi300x.decode_latency(1, 256, &model, None, &c);

    let speedup = h100_ms / mi300x_ms;
    assert!(
        speedup > 1.3 && speedup < 1.6,
        "MI300X vs H100 decode: expected 1.3-1.6× speedup, got {:.2}×",
        speedup
    );
}

#[test]
fn mi355x_has_fp8_tensor_cores() {
    let mi355x = GpuSpec::preset("mi355x").unwrap();
    assert!(mi355x.flops_fp8 > 0.0);
    assert!(
        mi355x.flops_fp8 > mi355x.flops_bf16,
        "FP8 should be at least 2× BF16"
    );
}

#[test]
fn tpu_8i_decode_close_to_b200() {
    // Official TPU 8i specs: 8.6 TB/s HBM, 288 GB. B200: 8.0 TB/s HBM, 192 GB.
    // With slightly higher HBM BW and slightly higher mfu_decode (0.80 vs 0.75 — CAE / on-chip SRAM),
    // TPU 8i should beat B200 on memory-bound decode by ~10-15%, not run away from it.
    let b200 = GpuSpec::preset("b200").unwrap();
    let tpu = GpuSpec::preset("tpu-v8i").unwrap();
    let model = LlmConfig::preset("llama-70b-fp8").unwrap();
    let c = ClusterConfig::single_gpu();

    let b200_ms = b200.decode_latency(1, 256, &model, None, &c);
    let tpu_ms = tpu.decode_latency(1, 256, &model, None, &c);

    let speedup = b200_ms / tpu_ms;
    assert!(
        speedup > 1.0 && speedup < 1.3,
        "TPU 8i vs B200 decode: expected within 1.0-1.3×, got {:.2}×",
        speedup
    );
}

#[test]
fn tpu_8i_has_larger_hbm_than_b200() {
    // 288 GB vs 192 GB — key advantage for fitting big MoE on fewer chips.
    let b200 = GpuSpec::preset("b200").unwrap();
    let tpu = GpuSpec::preset("tpu-v8i").unwrap();
    assert!(tpu.hbm_capacity > b200.hbm_capacity);
    assert_eq!(tpu.hbm_capacity, 288_000_000_000);
}

#[test]
fn tpu_8i_sram_helps_small_batch_kv() {
    // TPU 8i has 384 MB Vmem. For DSV3-MLA (~30 KB/token) at TP=8, batch=4, seq=1024,
    // KV per chip = 30 KB × 4 × 1024 / 8 ≈ 15 MB — well within 384 MB SRAM.
    // Expect TPU 8i decode to be measurably faster than a counter-factual no-SRAM variant.
    let tpu = GpuSpec::preset("tpu-v8i").unwrap();
    assert_eq!(tpu.on_chip_sram, 384_000_000);

    let mut tpu_no_sram = tpu.clone();
    tpu_no_sram.on_chip_sram = 0;

    let model = LlmConfig::preset("deepseek-v3").unwrap();
    let mut c = ClusterConfig::single_gpu();
    c.tp = 8;
    c.ep = 8;
    c.nvlink_bw = tpu.nvlink_bandwidth;

    let with_sram = tpu.decode_latency(4, 1024, &model, None, &c);
    let no_sram = tpu_no_sram.decode_latency(4, 1024, &model, None, &c);

    assert!(
        with_sram < no_sram,
        "SRAM should reduce decode latency when KV fits ({:.3} ms vs {:.3} ms)",
        with_sram * 1000.0,
        no_sram * 1000.0
    );
}

#[test]
fn groq_chip_is_sram_only_with_tiny_capacity() {
    // Groq LPU has no off-chip HBM — 230 MB on-chip SRAM at 80 TB/s.
    // The "HBM" fields represent that on-chip memory; on_chip_sram=0 because
    // it would otherwise double-count (the HBM IS the SRAM).
    let groq = GpuSpec::preset("groq-lpu-v1").unwrap();
    assert_eq!(groq.hbm_capacity, 230_000_000);
    assert!(
        (groq.hbm_bandwidth - 80e12).abs() < 1e6,
        "expected 80 TB/s SRAM BW"
    );
    assert_eq!(groq.on_chip_sram, 0);
}

#[test]
fn groq_needs_high_tp_for_large_models() {
    // llama-70b-fp8 (70 GB) divided across 230 MB chips ≈ 305 chips minimum.
    // Sanity check: per-chip latency drops as TP rises.
    let groq = GpuSpec::preset("groq-lpu-v1").unwrap();
    let model = LlmConfig::preset("llama-70b-fp8").unwrap();
    let mut c = ClusterConfig::single_gpu();
    c.tp = 358;
    c.nvlink_bw = groq.nvlink_bandwidth;

    // At TP=358, per-chip weights ≈ 195 MB — fits in 230 MB SRAM.
    let t = groq.decode_latency(1, 256, &model, None, &c);
    assert!(
        t > 0.0 && t < 0.001,
        "expected sub-millisecond decode, got {:.3} ms",
        t * 1000.0
    );
}

#[test]
fn sram_no_benefit_when_kv_doesnt_fit() {
    // For large batch × long-context MHA (e.g. llama-70b at batch=32, seq=4096), KV per chip
    // is hundreds of MB — exceeds even TPU 8i's 384 MB SRAM. Latency should match no-SRAM case.
    let tpu = GpuSpec::preset("tpu-v8i").unwrap();
    let mut tpu_no_sram = tpu.clone();
    tpu_no_sram.on_chip_sram = 0;

    let model = LlmConfig::preset("llama-70b-fp8").unwrap();
    let mut c = ClusterConfig::single_gpu();
    c.tp = 4;
    c.nvlink_bw = tpu.nvlink_bandwidth;

    // batch=64, seq=4096: KV per chip = 160 KB × 64 × 4096 / 4 ≈ 10 GB, way over SRAM.
    let with_sram = tpu.decode_latency(64, 4096, &model, None, &c);
    let no_sram = tpu_no_sram.decode_latency(64, 4096, &model, None, &c);

    assert!(
        (with_sram - no_sram).abs() < 1e-9,
        "Latencies should be identical when KV exceeds SRAM"
    );
}

#[test]
fn tpu_ironwood_has_fp8_and_torus_ici() {
    let tpu = GpuSpec::preset("tpu-v7-ironwood").unwrap();
    assert!(tpu.flops_fp8 > 0.0, "Ironwood has FP8 tensor cores");
    // ICI is in the scale-up fabric field (~1.2 TB/s).
    assert!(tpu.nvlink_bandwidth > 1_000e9);
}

#[test]
fn a100_has_no_fp8_tensor_cores() {
    // A100 preset sets flops_fp8 = 0; FP8 model should fall back to bf16 path.
    let a100 = GpuSpec::preset("a100").unwrap();
    assert_eq!(a100.flops_fp8, 0.0);
}

#[test]
fn all_reduce_scales_with_message_size() {
    let mut c = ClusterConfig::single_gpu();
    c.tp = 8;
    c.nvlink_bw = 900e9;

    let small = c.all_reduce_latency(1_000_000);
    let big = c.all_reduce_latency(10_000_000);

    // Ring-allreduce is linear in message size at this regime.
    assert!(
        close(big / small, 10.0, 0.001),
        "expected 10× scaling, got {:.4}",
        big / small
    );
}
