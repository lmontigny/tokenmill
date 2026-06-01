//! Roofline GPU latency model.
//!
//! Cross-checks the closed-form latency against hand-derived numbers and
//! verifies the scaling behaviour we rely on (FP8 dispatch, TP scaling,
//! all-reduce overhead).

use inference_sim::hardware::cluster::ClusterConfig;
use inference_sim::hardware::gpu::GpuSpec;
use inference_sim::model::llm_config::LlmConfig;

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
