use std::fs;
use std::path::Path;

use clap::Parser;
use rayon::prelude::*;

use tokenmill::engine::sim::{MtpConfig, SchedulerKind, Simulator, SpecConfig};
use tokenmill::hardware::cluster::ClusterConfig;
use tokenmill::hardware::gpu::{GpuSpec, GpuState};
use tokenmill::hardware::kernel_table::KernelTable;
use tokenmill::metrics::report::RunSummary;
use tokenmill::model::kv_cache::KvCacheManager;
use tokenmill::model::llm_config::LlmConfig;
use tokenmill::scheduler::chunked_prefill::ChunkedPrefillScheduler;
use tokenmill::scheduler::continuous_batch::ContinuousBatchScheduler;
use tokenmill::workload::synthetic::SyntheticWorkload;
use tokenmill::workload::trace_replay::TraceReplay;

#[derive(Parser, Debug, Clone)]
#[command(name = "tokenmill", about = "LLM inference discrete-event simulator")]
struct Args {
    /// Accelerator preset: rubin | b200 | h200 | h100 | a100 | a10g | mi355x | mi325x | mi300x | tpu-v8i | tpu-v8t | tpu-v7-ironwood | groq-lpu-v1 | cerebras-cs3
    #[arg(long, default_value = "h100")]
    gpu: String,

    /// System preset: none | dgx-h100 | dgx-h200 | dgx-b200. Sets GPU and node defaults.
    #[arg(long, default_value = "none")]
    system: String,

    /// Model preset: llama-70b | llama-8b | (-fp8 variants) | mixtral-8x7b | llama4-maverick | deepseek-v3 | kimi-k2 | llama4-behemoth
    #[arg(long, default_value = "llama-70b")]
    model: String,

    /// Scheduler: continuous-batch | chunked-prefill
    #[arg(long, default_value = "continuous-batch")]
    scheduler: String,

    /// Chunk size for chunked-prefill (tokens)
    #[arg(long, default_value_t = 512)]
    chunk_size: u32,

    /// Workload: synthetic | trace:<path.csv>
    #[arg(long, default_value = "synthetic")]
    workload: String,

    /// Arrival rate for synthetic workload (req/s)
    #[arg(long, default_value_t = 5.0)]
    arrival_rate: f64,

    /// Simulation duration in seconds
    #[arg(long, default_value_t = 60.0)]
    duration: f64,

    /// Mean prompt length for synthetic workload (tokens, log-normal)
    #[arg(long, default_value_t = 512.0)]
    prompt_mean: f64,

    /// Mean output length for synthetic workload (tokens, log-normal)
    #[arg(long, default_value_t = 128.0)]
    output_mean: f64,

    /// Max tokens in flight across all in-flight requests
    #[arg(long, default_value_t = 8192)]
    max_batch_tokens: u32,

    /// KV cache block size (tokens per block)
    #[arg(long, default_value_t = 16)]
    kv_block_size: u32,

    /// KV cache total blocks (0 = auto from accelerator memory)
    #[arg(long, default_value_t = 0)]
    kv_blocks: u32,

    /// Tensor parallelism degree
    #[arg(long, default_value_t = 1)]
    tp: u32,

    /// Pipeline parallelism degree
    #[arg(long, default_value_t = 1)]
    pp: u32,

    /// Expert parallelism degree (MoE models only)
    #[arg(long, default_value_t = 1)]
    ep: u32,

    /// Accelerators per scale-up node/server before traffic crosses scale-out networking
    #[arg(long, default_value_t = 8)]
    gpus_per_node: u32,

    /// Scale-out fabric preset: none | hdr-200 | ndr-400 | quantum-x800 | spectrum-x400 | spectrum-x800
    #[arg(long, default_value = "none")]
    scale_out_fabric: String,

    /// Scale-out network bandwidth for cross-node TP/PP/EP collectives (GB/s). 0 = legacy uniform scale-up model.
    #[arg(long, default_value_t = 0.0)]
    scale_out_bw_gbps: f64,

    /// One-way scale-out network latency for cross-node TP/PP/EP collectives (microseconds)
    #[arg(long, default_value_t = 5.0)]
    scale_out_latency_us: f64,

    /// Disaggregate prefill and decode onto separate GPUs
    #[arg(long)]
    disaggregate: bool,

    /// Network bandwidth for KV transfer in disaggregated mode (GB/s)
    #[arg(long, default_value_t = 200.0)]
    internode_bw_gbps: f64,

    /// Kernel time table CSV
    #[arg(long)]
    kernel_table: Option<String>,

    /// Output format: text | json | csv
    #[arg(long, default_value = "text")]
    output: String,

    /// Study mode: comma-separated model presets for matrix comparison
    #[arg(long)]
    study_models: Option<String>,

    /// Study mode: comma-separated GPU presets for matrix comparison
    #[arg(long)]
    study_gpus: Option<String>,

    /// Study mode: comma-separated system presets for matrix comparison
    #[arg(long)]
    study_systems: Option<String>,

    /// Study mode: comma-separated TP degrees for matrix comparison
    #[arg(long)]
    study_tps: Option<String>,

    /// Study mode: comma-separated arrival rates for matrix comparison
    #[arg(long)]
    study_arrival_rates: Option<String>,

    /// Study mode: write self-contained HTML report to this path
    #[arg(long)]
    html: Option<String>,

    /// Study mode: write normalized JSON results to this path
    #[arg(long)]
    json_out: Option<String>,

    /// Sweep over arrival rates (comma-separated, e.g. 1,5,10,20). Runs in parallel.
    #[arg(long)]
    sweep_arrival_rates: Option<String>,

    /// Speculative decoding: number of draft tokens per step (0 = disabled)
    #[arg(long, default_value_t = 0)]
    spec_tokens: u32,

    /// Speculative decoding: per-token acceptance rate γ (0.0–1.0)
    #[arg(long, default_value_t = 0.7)]
    spec_acceptance_rate: f64,

    /// Draft model preset for speculative decoding (e.g. llama-8b when main is llama-70b)
    #[arg(long)]
    draft_model: Option<String>,

    /// Multi-token prediction: number of MTP heads (0 = disabled). Cannot combine with --spec-tokens.
    #[arg(long, default_value_t = 0)]
    mtp_heads: u32,

    /// Multi-token prediction: per-token acceptance rate γ (0.0–1.0)
    #[arg(long, default_value_t = 0.9)]
    mtp_acceptance_rate: f64,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Validate roofline model against a reference CSV (gpu,model,op,batch_size,seq_len,latency_ms).
    /// Prints MAPE per category and suggests MFU adjustments. Does not run a simulation.
    #[arg(long)]
    validate_kernels: Option<String>,
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn load_kernel_table(path: &str) -> Option<KernelTable> {
    match KernelTable::from_csv(Path::new(path)) {
        Ok(kt) => Some(kt),
        Err(e) => {
            eprintln!("Warning: could not load kernel table '{}': {}", path, e);
            None
        }
    }
}

/// Compute total logical KV cache blocks for the cluster.
///
/// With TP/PP, each GPU holds 1/(tp×pp) of the model weights and 1/(tp×pp) of
/// the KV per token (KV heads sharded by TP, KV layers sharded by PP).
/// Total logical blocks = (tp×pp × available_per_gpu) / bytes_per_block.
fn auto_kv_blocks(
    gpu_spec: &GpuSpec,
    model: &LlmConfig,
    kv_block_size: u32,
    tp: u32,
    pp: u32,
) -> u32 {
    let n_gpus = (tp * pp).max(1) as u64;
    let weight_per_gpu = model.weight_bytes / n_gpus;
    let hbm = gpu_spec.memory_capacity as f64;
    let available_per_gpu = if weight_per_gpu < gpu_spec.memory_capacity {
        (hbm - weight_per_gpu as f64) * 0.80
    } else {
        hbm * 0.20
    };
    let bytes_per_block = (model.kv_bytes(1) * kv_block_size as u64).max(1) as f64;
    let total_blocks = n_gpus as f64 * available_per_gpu / bytes_per_block;
    (total_blocks as u32).max(64)
}

fn fmt_bytes(bytes: u64) -> String {
    let gb = bytes as f64 / 1e9;
    if gb >= 1.0 {
        format!("{:.1} GB", gb)
    } else {
        format!("{:.0} MB", bytes as f64 / 1e6)
    }
}

fn fmt_params(bytes: u64, bytes_per_param: f64) -> String {
    let params = bytes as f64 / bytes_per_param;
    if params >= 1e12 {
        format!("{:.1} T", params / 1e12)
    } else if params >= 1e9 {
        format!("{:.0} B", params / 1e9)
    } else {
        format!("{:.0} M", params / 1e6)
    }
}

fn dtype_label(bits: u32) -> &'static str {
    match bits {
        4 => "fp4/int4",
        8 => "fp8/int8",
        16 => "bf16/fp16",
        32 => "fp32",
        _ => "custom",
    }
}

/// Print a model/cluster info card before running the simulation.
fn print_model_card(args: &Args) {
    let gpu = GpuSpec::preset(&args.gpu).unwrap_or_else(|| panic!("Unknown GPU '{}'", args.gpu));
    let model =
        LlmConfig::preset(&args.model).unwrap_or_else(|| panic!("Unknown model '{}'", args.model));

    let n_gpus = (args.tp * args.pp).max(1) as u64;
    let weight_per_gpu = model.weight_bytes / n_gpus;
    let fits = weight_per_gpu < gpu.memory_capacity;

    let kv_total_blocks = if args.kv_blocks > 0 {
        args.kv_blocks
    } else {
        auto_kv_blocks(&gpu, &model, args.kv_block_size, args.tp, args.pp)
    };
    let kv_total_tokens = kv_total_blocks as u64 * args.kv_block_size as u64;
    let kv_total_bytes = kv_total_tokens * model.kv_bytes(1);

    let sep = "─".repeat(60);
    println!("Model {}", sep);

    // Parameters
    let bytes_per_param = model.weight_bytes_per_param();
    let total_params = fmt_params(model.weight_bytes, bytes_per_param);
    if model.is_moe() {
        let active_params = fmt_params(model.weight_bytes_active(), bytes_per_param);
        let active_pct = model.active_param_fraction() * 100.0;
        println!("  Type           MoE (sparse activation)");
        println!(
            "  Parameters     {} params ({})  |  {} active/token ({:.1}%)",
            total_params,
            fmt_bytes(model.weight_bytes),
            active_params,
            active_pct
        );
    } else {
        println!("  Type           dense");
        println!(
            "  Parameters     {} params ({})",
            total_params,
            fmt_bytes(model.weight_bytes)
        );
    }

    println!(
        "  Dtype          {} weights / A{} / KV{}  ({:.1} byte/param, {} activation byte{})",
        dtype_label(model.weight_bits),
        model.effective_activation_bits(),
        model.effective_kv_bits(),
        bytes_per_param,
        model.activation_bytes(),
        if model.activation_bytes() == 1 {
            ""
        } else {
            "s"
        }
    );
    println!(
        "  Architecture   {} layers  |  d_model={}  |  {} kv-heads × {} head-dim",
        model.n_layers, model.d_model, model.n_kv_heads, model.head_dim
    );

    if model.is_moe() {
        let layer_str = if model.n_moe_layers < model.n_layers {
            format!(
                "{} MoE + {} dense",
                model.n_moe_layers,
                model.n_layers - model.n_moe_layers
            )
        } else {
            format!("all {} MoE", model.n_moe_layers)
        };
        let shared_str = if model.n_shared_experts > 0 {
            format!(" + {} shared (always active)", model.n_shared_experts)
        } else {
            String::new()
        };
        println!("  MoE layers     {}", layer_str);
        println!(
            "  Experts        {} routable{}  |  top-{} per token",
            model.n_experts, shared_str, model.n_active_experts
        );
    }

    // KV cache
    let kv_per_token_kb = model.kv_bytes(1) as f64 / 1024.0;
    if model.kv_lora_rank > 0 {
        let std_kv_kb = 2.0
            * model.n_layers as f64
            * model.n_kv_heads as f64
            * model.head_dim as f64
            * model.kv_bytes_per_entry()
            / 1024.0;
        let compression = std_kv_kb / kv_per_token_kb;
        println!("  KV cache (MLA) {} L × rank-{}  =  {:.1} KB/token  ({:.0}× smaller than MHA {:.0} KB)",
            model.n_layers, model.kv_lora_rank, kv_per_token_kb, compression, std_kv_kb);
    } else {
        println!(
            "  KV cache       {} L × {} H × {}  =  {:.1} KB/token",
            model.n_layers, model.n_kv_heads, model.head_dim, kv_per_token_kb
        );
    }

    // Cluster section
    let cluster_desc = build_cluster_desc(args, &gpu);
    println!("Cluster {}", sep);
    println!("  {}", cluster_desc);
    let peak_flops = gpu.peak_flops_for(&model);
    let dtype_tag = gpu.precision_label_for(&model);
    println!(
        "  Memory/accel   {}  ({:.0} TFLOPS {} peak)",
        fmt_bytes(gpu.memory_capacity),
        peak_flops / 1e12,
        dtype_tag
    );

    let weight_fit = if fits {
        format!(
            "✓ fits  ({}/accelerator < {} memory)",
            fmt_bytes(weight_per_gpu),
            fmt_bytes(gpu.memory_capacity)
        )
    } else {
        let min_tp = ((model.weight_bytes as f64 / (gpu.memory_capacity as f64 * 0.85)).ceil()
            as u32)
            .max(2);
        format!(
            "✗ EXCEEDS HBM  ({}/GPU > {} HBM)  — use --tp {} or higher",
            fmt_bytes(weight_per_gpu),
            fmt_bytes(gpu.memory_capacity),
            min_tp
        )
    };
    println!(
        "  Weights        {}  total  →  {}",
        fmt_bytes(model.weight_bytes),
        weight_fit
    );

    let cluster_flops_pflops = peak_flops * n_gpus as f64 / 1e15;
    println!(
        "  Peak FLOPS     {:.1} TFLOPS {} × {} GPUs  =  {:.1} PFLOPS",
        peak_flops / 1e12,
        dtype_tag,
        n_gpus,
        cluster_flops_pflops
    );

    println!(
        "  KV budget      {} blocks × {} tok  =  {:>8} tokens  ({})",
        kv_total_blocks,
        args.kv_block_size,
        fmt_with_commas(kv_total_tokens),
        fmt_bytes(kv_total_bytes)
    );

    if args.ep > 1 {
        println!(
            "  Expert par.    EP={}  ({} experts/GPU)",
            args.ep,
            (model.n_experts / args.ep).max(1)
        );
    }
    if args.spec_tokens > 0 {
        let draft_name = args
            .draft_model
            .as_deref()
            .unwrap_or("auto (1/10 main model)");
        let expected_tok = {
            let k = args.spec_tokens as f64;
            let g = args.spec_acceptance_rate.clamp(0.0, 1.0 - 1e-9);
            (1.0 - g.powf(k + 1.0)) / (1.0 - g)
        };
        println!(
            "  Spec decode    K={}  γ={:.2}  draft={}  → {:.2} tok/step expected",
            args.spec_tokens, args.spec_acceptance_rate, draft_name, expected_tok
        );
    }
    if args.mtp_heads > 0 {
        let k = args.mtp_heads as f64;
        let g = args.mtp_acceptance_rate.clamp(0.0, 1.0 - 1e-9);
        let expected_tok = (1.0 - g.powf(k + 1.0)) / (1.0 - g);
        let overhead_pct = k / model.n_layers as f64 * 100.0;
        println!(
            "  MTP            K={}  γ={:.2}  overhead={:.1}%/step  → {:.2} tok/step expected",
            args.mtp_heads, args.mtp_acceptance_rate, overhead_pct, expected_tok
        );
    }

    println!("{}", sep);
    println!();
}

fn build_cluster_desc(args: &Args, gpu: &GpuSpec) -> String {
    let n_gpus = args.tp * args.pp;
    let mut parts = vec![format!("{}× {}", n_gpus, gpu.name)];
    if args.system != "none" {
        parts.push(args.system.to_uppercase());
    }
    if args.tp > 1 {
        parts.push(format!("TP={}", args.tp));
    }
    if args.pp > 1 {
        parts.push(format!("PP={}", args.pp));
    }
    if args.ep > 1 {
        parts.push(format!("EP={}", args.ep));
    }
    let scale_out = scale_out_config(args);
    if scale_out.bw_gbps > 0.0 {
        parts.push(format!(
            "{} GPU/node, {} {:.0} GB/s",
            args.gpus_per_node, scale_out.label, scale_out.bw_gbps
        ));
    }
    if args.disaggregate {
        parts.push("disaggregated P/D".to_string());
    }
    parts.join("  |  ")
}

#[derive(Clone)]
struct ScaleOutConfig {
    label: String,
    bw_gbps: f64,
    latency_us: f64,
}

fn scale_out_config(args: &Args) -> ScaleOutConfig {
    let preset = match args.scale_out_fabric.as_str() {
        "none" => ScaleOutConfig {
            label: "scale-out".into(),
            bw_gbps: 0.0,
            latency_us: args.scale_out_latency_us,
        },
        "hdr-200" => ScaleOutConfig {
            label: "HDR/200G".into(),
            bw_gbps: 25.0,
            latency_us: 3.0,
        },
        "ndr-400" => ScaleOutConfig {
            label: "NDR/400G".into(),
            bw_gbps: 50.0,
            latency_us: 2.0,
        },
        "quantum-x800" | "xdr-800" => ScaleOutConfig {
            label: "Quantum-X800/800G".into(),
            bw_gbps: 100.0,
            latency_us: 2.0,
        },
        "spectrum-x400" => ScaleOutConfig {
            label: "Spectrum-X/400G".into(),
            bw_gbps: 50.0,
            latency_us: 5.0,
        },
        "spectrum-x800" => ScaleOutConfig {
            label: "Spectrum-X800/800G".into(),
            bw_gbps: 100.0,
            latency_us: 5.0,
        },
        other => panic!("Unknown scale-out fabric '{}'", other),
    };

    ScaleOutConfig {
        label: preset.label,
        bw_gbps: if args.scale_out_bw_gbps > 0.0 {
            args.scale_out_bw_gbps
        } else {
            preset.bw_gbps
        },
        latency_us: if args.scale_out_latency_us != 5.0 {
            args.scale_out_latency_us
        } else {
            preset.latency_us
        },
    }
}

fn apply_system_preset(args: &mut Args) {
    match args.system.as_str() {
        "none" => {}
        "dgx-h100" => {
            args.gpu = "h100".into();
            args.gpus_per_node = 8;
            if args.scale_out_fabric == "none" && args.scale_out_bw_gbps == 0.0 {
                args.scale_out_fabric = "ndr-400".into();
            }
        }
        "dgx-h200" => {
            args.gpu = "h200".into();
            args.gpus_per_node = 8;
            if args.scale_out_fabric == "none" && args.scale_out_bw_gbps == 0.0 {
                args.scale_out_fabric = "ndr-400".into();
            }
        }
        "dgx-b200" => {
            args.gpu = "b200".into();
            args.gpus_per_node = 8;
            if args.scale_out_fabric == "none" && args.scale_out_bw_gbps == 0.0 {
                args.scale_out_fabric = "ndr-400".into();
            }
        }
        other => panic!("Unknown system preset '{}'", other),
    }
}

fn split_csv(s: Option<&str>, fallback: &str) -> Vec<String> {
    s.unwrap_or(fallback)
        .split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn split_csv_f64(s: Option<&str>, fallback: f64) -> Vec<f64> {
    match s {
        Some(values) => values
            .split(',')
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(|v| {
                v.parse::<f64>()
                    .unwrap_or_else(|_| panic!("Invalid floating-point value '{}'", v))
            })
            .collect(),
        None => vec![fallback],
    }
}

fn split_csv_u32(s: Option<&str>, fallback: u32) -> Vec<u32> {
    match s {
        Some(values) => values
            .split(',')
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(|v| {
                v.parse::<u32>()
                    .unwrap_or_else(|_| panic!("Invalid integer value '{}'", v))
            })
            .collect(),
        None => vec![fallback],
    }
}

fn is_study_mode(args: &Args) -> bool {
    args.study_models.is_some()
        || args.study_gpus.is_some()
        || args.study_systems.is_some()
        || args.study_tps.is_some()
        || args.study_arrival_rates.is_some()
        || args.html.is_some()
        || args.json_out.is_some()
}

fn run_study(args: &Args, kt: Option<&KernelTable>) -> Vec<RunSummary> {
    let models = split_csv(args.study_models.as_deref(), &args.model);
    let gpus = split_csv(args.study_gpus.as_deref(), &args.gpu);
    let systems = split_csv(args.study_systems.as_deref(), "");
    let tps = split_csv_u32(args.study_tps.as_deref(), args.tp);
    let arrival_rates = split_csv_f64(args.study_arrival_rates.as_deref(), args.arrival_rate);

    let mut configs = Vec::new();
    for model in &models {
        for tp in &tps {
            for arrival_rate in &arrival_rates {
                if systems.is_empty() {
                    for gpu in &gpus {
                        let mut c = args.clone();
                        c.model = model.clone();
                        c.gpu = gpu.clone();
                        c.system = "none".into();
                        c.tp = *tp;
                        c.arrival_rate = *arrival_rate;
                        configs.push((c, *arrival_rate));
                    }
                } else {
                    for system in &systems {
                        let mut c = args.clone();
                        c.model = model.clone();
                        c.system = system.clone();
                        c.tp = *tp;
                        c.arrival_rate = *arrival_rate;
                        apply_system_preset(&mut c);
                        configs.push((c, *arrival_rate));
                    }
                }
            }
        }
    }

    configs
        .par_iter()
        .map(|(config, arrival_rate)| run_once(config, *arrival_rate, kt))
        .collect()
}

fn write_study_outputs(
    args: &Args,
    summaries: &[RunSummary],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = &args.json_out {
        write_report_file(path, &serde_json::to_string_pretty(summaries)?)?;
    }
    if let Some(path) = &args.html {
        write_report_file(path, &render_html_report(args, summaries)?)?;
    }
    Ok(())
}

fn write_report_file(path: &str, contents: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, contents)?;
    Ok(())
}

fn render_html_report(
    args: &Args,
    summaries: &[RunSummary],
) -> Result<String, Box<dyn std::error::Error>> {
    let data = serde_json::to_string_pretty(summaries)?;
    let generated = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok(format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tokenmill study report</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f8fb;
  --panel: #ffffff;
  --ink: #17202a;
  --muted: #5d6b7a;
  --line: #d9e0e8;
  --accent: #0f766e;
  --accent2: #2563eb;
}}
body {{
  margin: 0;
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--ink);
}}
header {{
  padding: 28px 32px 18px;
  background: #0f172a;
  color: white;
}}
h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
h2 {{ margin: 0 0 14px; font-size: 18px; }}
main {{ padding: 24px 32px 40px; }}
.sub {{ color: #cbd5e1; }}
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 12px;
  margin-bottom: 20px;
}}
.card {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
}}
.label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
.value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
.small {{ color: var(--muted); margin-top: 3px; }}
.controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 14px 0 18px;
}}
input, select {{
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px 10px;
  background: white;
  min-width: 170px;
}}
section {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 18px;
}}
.charts {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 14px;
}}
svg {{ width: 100%; height: 260px; display: block; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
th {{ color: var(--muted); font-size: 12px; cursor: pointer; white-space: nowrap; }}
td {{ white-space: nowrap; }}
.table-wrap {{ overflow-x: auto; }}
.note {{ color: var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>tokenmill study report</h1>
  <div class="sub">{rows} runs · generated unix {generated}</div>
</header>
<main>
  <div class="cards" id="cards"></div>
  <section>
    <h2>Filters</h2>
    <div class="controls">
      <input id="search" placeholder="Search model, GPU, scheduler">
      <select id="modelFilter"><option value="">All models</option></select>
      <select id="gpuFilter"><option value="">All GPUs</option></select>
      <select id="precisionFilter"><option value="">All precision</option></select>
    </div>
  </section>
  <section>
    <h2>Charts</h2>
    <div class="charts">
      <div><div class="label">Cost vs TPOT p95</div><svg id="scatter"></svg></div>
      <div><div class="label">Throughput by run</div><svg id="throughput"></svg></div>
      <div><div class="label">Energy per token by run</div><svg id="energy"></svg></div>
    </div>
  </section>
  <section>
    <h2>Results</h2>
    <div class="table-wrap">
      <table id="results"></table>
    </div>
  </section>
  <section>
    <h2>Study Configuration</h2>
    <p class="note">duration={duration}s, prompt_mean={prompt}, output_mean={output}, scheduler={scheduler}, latency model is reported per row.</p>
  </section>
</main>
<script id="data" type="application/json">{data}</script>
<script>
const rows = JSON.parse(document.getElementById('data').textContent);
let sortKey = 'cost_per_million_tokens_usd';
let sortDir = 1;

function precision(model) {{
  if (model.includes('nvfp4')) return 'NVFP4 sparse';
  if (model.includes('w4a8kv4')) return 'W4A8KV4';
  if (model.includes('w4a16')) return 'W4A16';
  if (model.includes('fp8')) return 'FP8';
  return 'BF16/FP16';
}}
function fmt(v, d=1) {{
  if (!Number.isFinite(v)) return '';
  return Number(v).toLocaleString(undefined, {{ maximumFractionDigits: d, minimumFractionDigits: d }});
}}
function uniq(values) {{ return [...new Set(values)].sort(); }}
function fillSelect(id, values) {{
  const el = document.getElementById(id);
  for (const v of values) {{
    const o = document.createElement('option');
    o.value = v; o.textContent = v; el.appendChild(o);
  }}
}}
fillSelect('modelFilter', uniq(rows.map(r => r.model)));
fillSelect('gpuFilter', uniq(rows.map(r => r.gpu)));
fillSelect('precisionFilter', uniq(rows.map(r => precision(r.model))));

function filtered() {{
  const q = document.getElementById('search').value.toLowerCase();
  const m = document.getElementById('modelFilter').value;
  const g = document.getElementById('gpuFilter').value;
  const p = document.getElementById('precisionFilter').value;
  return rows.filter(r =>
    (!q || `${{r.model}} ${{r.gpu}} ${{r.scheduler}}`.toLowerCase().includes(q)) &&
    (!m || r.model === m) &&
    (!g || r.gpu === g) &&
    (!p || precision(r.model) === p)
  );
}}
function bestBy(data, key, lower=true) {{
  const valid = data.filter(r => Number.isFinite(r[key]) && r[key] > 0);
  valid.sort((a,b) => lower ? a[key] - b[key] : b[key] - a[key]);
  return valid[0];
}}
function renderCards(data) {{
  const cards = [
    ['Cheapest', bestBy(data, 'cost_per_million_tokens_usd'), r => `$${{fmt(r.cost_per_million_tokens_usd, 2)}}/Mtok`],
    ['Lowest TPOT p95', bestBy(data, 'tpot_p95_ms'), r => `${{fmt(r.tpot_p95_ms, 1)}} ms`],
    ['Highest throughput', bestBy(data, 'token_throughput', false), r => `${{fmt(r.token_throughput, 0)}} tok/s`],
    ['Lowest energy/token', bestBy(data, 'energy_per_token_mj'), r => `${{fmt(r.energy_per_token_mj, 1)}} mJ`],
  ];
  document.getElementById('cards').innerHTML = cards.map(([label, row, val]) => `
    <div class="card"><div class="label">${{label}}</div>
    <div class="value">${{row ? val(row) : 'n/a'}}</div>
    <div class="small">${{row ? `${{row.model}} · ${{row.gpu}} · ${{row.scheduler}}` : ''}}</div></div>
  `).join('');
}}
function renderTable(data) {{
  const cols = [
    ['model','Model'], ['gpu','GPU/System'], ['precision','Precision'], ['tp','TP'], ['pp','PP'],
    ['arrival_rate','RPS in'], ['completions','Done'], ['token_throughput','Tok/s'],
    ['ttft_p95_ms','TTFT p95 ms'], ['tpot_p95_ms','TPOT p95 ms'],
    ['cost_per_million_tokens_usd','$/Mtok'], ['energy_per_token_mj','mJ/token'], ['kv_util_mean_pct','KV %']
  ];
  const sorted = [...data].sort((a,b) => {{
    const av = sortKey === 'precision' ? precision(a.model) : a[sortKey];
    const bv = sortKey === 'precision' ? precision(b.model) : b[sortKey];
    return (typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv))) * sortDir;
  }});
  const head = '<tr>' + cols.map(([k,n]) => `<th data-k="${{k}}">${{n}}</th>`).join('') + '</tr>';
  const body = sorted.map(r => `<tr>
    <td>${{r.model}}</td><td>${{r.gpu}}</td><td>${{precision(r.model)}}</td>
    <td>${{r.tp}}</td><td>${{r.pp}}</td><td>${{fmt(r.arrival_rate, 2)}}</td><td>${{r.completions}}</td>
    <td>${{fmt(r.token_throughput, 0)}}</td><td>${{fmt(r.ttft_p95_ms, 1)}}</td><td>${{fmt(r.tpot_p95_ms, 1)}}</td>
    <td>${{fmt(r.cost_per_million_tokens_usd, 2)}}</td><td>${{fmt(r.energy_per_token_mj, 1)}}</td><td>${{fmt(r.kv_util_mean_pct, 1)}}</td>
  </tr>`).join('');
  const table = document.getElementById('results');
  table.innerHTML = head + body;
  table.querySelectorAll('th').forEach(th => th.onclick = () => {{
    const k = th.dataset.k;
    if (sortKey === k) sortDir *= -1; else {{ sortKey = k; sortDir = 1; }}
    render();
  }});
}}
function drawBars(id, data, key, color) {{
  const svg = document.getElementById(id);
  const w = svg.clientWidth || 420, h = 260, pad = 34;
  const vals = data.map(r => r[key]).filter(v => Number.isFinite(v));
  const max = Math.max(...vals, 1);
  const barW = Math.max(3, (w - pad * 2) / Math.max(data.length, 1) - 3);
  svg.innerHTML = data.map((r,i) => {{
    const x = pad + i * ((w - pad * 2) / Math.max(data.length, 1));
    const bh = (h - pad * 2) * ((r[key] || 0) / max);
    const y = h - pad - bh;
    return `<rect x="${{x}}" y="${{y}}" width="${{barW}}" height="${{bh}}" fill="${{color}}"><title>${{r.model}} · ${{r.gpu}}: ${{fmt(r[key], 2)}}</title></rect>`;
  }}).join('') + `<line x1="${{pad}}" y1="${{h-pad}}" x2="${{w-pad}}" y2="${{h-pad}}" stroke="#94a3b8"/>`;
}}
function drawScatter(data) {{
  const svg = document.getElementById('scatter');
  const w = svg.clientWidth || 420, h = 260, pad = 36;
  const valid = data.filter(r => r.cost_per_million_tokens_usd > 0 && r.tpot_p95_ms > 0);
  const maxX = Math.max(...valid.map(r => r.tpot_p95_ms), 1);
  const maxY = Math.max(...valid.map(r => r.cost_per_million_tokens_usd), 1);
  svg.innerHTML = valid.map(r => {{
    const x = pad + (w - pad * 2) * r.tpot_p95_ms / maxX;
    const y = h - pad - (h - pad * 2) * r.cost_per_million_tokens_usd / maxY;
    return `<circle cx="${{x}}" cy="${{y}}" r="5" fill="#0f766e"><title>${{r.model}} · ${{r.gpu}}: ${{fmt(r.tpot_p95_ms,1)}} ms, $${{fmt(r.cost_per_million_tokens_usd,2)}}/Mtok</title></circle>`;
  }}).join('') + `<line x1="${{pad}}" y1="${{h-pad}}" x2="${{w-pad}}" y2="${{h-pad}}" stroke="#94a3b8"/><line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{h-pad}}" stroke="#94a3b8"/>`;
}}
function render() {{
  const data = filtered();
  renderCards(data);
  renderTable(data);
  drawScatter(data);
  drawBars('throughput', data, 'token_throughput', '#2563eb');
  drawBars('energy', data, 'energy_per_token_mj', '#dc2626');
}}
['search','modelFilter','gpuFilter','precisionFilter'].forEach(id => document.getElementById(id).addEventListener('input', render));
render();
</script>
</body>
</html>"##,
        rows = summaries.len(),
        generated = generated,
        data = data,
        duration = args.duration,
        prompt = args.prompt_mean,
        output = args.output_mean,
        scheduler = args.scheduler,
    ))
}

fn fmt_with_commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

// ── run_once ──────────────────────────────────────────────────────────────────

fn run_once(args: &Args, arrival_rate: f64, kt: Option<&KernelTable>) -> RunSummary {
    let gpu_spec =
        GpuSpec::preset(&args.gpu).unwrap_or_else(|| panic!("Unknown GPU '{}'", args.gpu));
    let model =
        LlmConfig::preset(&args.model).unwrap_or_else(|| panic!("Unknown model '{}'", args.model));

    let kv_total_blocks = if args.kv_blocks > 0 {
        args.kv_blocks
    } else {
        auto_kv_blocks(&gpu_spec, &model, args.kv_block_size, args.tp, args.pp)
    };

    let scheduler = match args.scheduler.as_str() {
        "continuous-batch" => {
            SchedulerKind::Continuous(ContinuousBatchScheduler::new(args.max_batch_tokens))
        }
        "chunked-prefill" => SchedulerKind::Chunked(ChunkedPrefillScheduler::new(
            args.chunk_size,
            args.max_batch_tokens,
        )),
        other => panic!("Unknown scheduler '{}'", other),
    };

    let scale_out = scale_out_config(args);
    let cluster = ClusterConfig {
        tp: args.tp,
        pp: args.pp,
        ep: args.ep,
        scale_up_bw: gpu_spec.scale_up_bandwidth,
        scale_up_latency: gpu_spec.scale_up_latency,
        gpus_per_node: args.gpus_per_node,
        scale_out_bw: scale_out.bw_gbps * 1e9,
        scale_out_latency: scale_out.latency_us * 1e-6,
        internode_bw: args.internode_bw_gbps * 1e9,
        disaggregate: args.disaggregate,
    };

    let mut prefill_gpu = GpuState::new(0, gpu_spec.clone());
    if let Some(k) = kt {
        prefill_gpu = prefill_gpu.with_kernel_table(k.clone());
    }

    let decode_gpu = if args.disaggregate {
        let mut g = GpuState::new(1, gpu_spec.clone());
        if let Some(k) = kt {
            g = g.with_kernel_table(k.clone());
        }
        Some(g)
    } else {
        None
    };

    let kv = KvCacheManager::new(kv_total_blocks, args.kv_block_size);
    let latency_mode = if kt.is_some() {
        "kernel-table+roofline"
    } else {
        "roofline"
    };

    let spec = if args.spec_tokens > 0 {
        let draft = match args.draft_model.as_deref().and_then(LlmConfig::preset) {
            Some(m) => m,
            None => {
                // No draft model specified: synthesise one at 1/10 the weight size of the main model.
                let mut d = model.clone();
                d.name = format!("{}-draft", d.name);
                d.weight_bytes = (d.weight_bytes / 10).max(1);
                d.active_weight_bytes =
                    (d.active_weight_bytes / 10).max(if d.active_weight_bytes > 0 { 1 } else { 0 });
                d
            }
        };
        Some(SpecConfig {
            draft_tokens: args.spec_tokens,
            acceptance_rate: args.spec_acceptance_rate,
            draft_model: draft,
        })
    } else {
        None
    };

    if args.spec_tokens > 0 && args.mtp_heads > 0 {
        eprintln!("Error: --spec-tokens and --mtp-heads are mutually exclusive.");
        std::process::exit(1);
    }

    let mtp = if args.mtp_heads > 0 {
        Some(MtpConfig {
            num_heads: args.mtp_heads,
            acceptance_rate: args.mtp_acceptance_rate,
        })
    } else {
        None
    };

    let mut sim = if args.workload.starts_with("trace:") {
        let path = &args.workload["trace:".len()..];
        let mut trace = TraceReplay::from_csv(Path::new(path))
            .unwrap_or_else(|e| panic!("Cannot load trace '{}': {}", path, e));
        Simulator::new(
            prefill_gpu,
            decode_gpu,
            model.clone(),
            cluster,
            scheduler,
            &mut trace,
            kv,
        )
    } else {
        let mut w = SyntheticWorkload::new(
            arrival_rate,
            args.prompt_mean,
            args.output_mean,
            args.duration,
            args.seed,
        );
        Simulator::new(
            prefill_gpu,
            decode_gpu,
            model.clone(),
            cluster,
            scheduler,
            &mut w,
            kv,
        )
    };
    if let Some(s) = spec {
        sim = sim.with_spec(s);
    }
    if let Some(m) = mtp {
        sim = sim.with_mtp(m);
    }

    sim.run(args.duration);

    let parallelism = match (args.tp, args.pp, args.disaggregate) {
        (1, 1, false) => "single-gpu".to_string(),
        (1, 1, true) => "disaggregated".to_string(),
        (tp, 1, false) => format!("TP={tp}"),
        (1, pp, false) => format!("PP={pp}"),
        (tp, pp, false) => format!("TP={tp} PP={pp}"),
        (tp, pp, true) => format!("TP={tp} PP={pp} disaggregated"),
    };

    sim.metrics.to_summary(
        &model.name,
        &gpu_spec.name,
        &format!("{} {}", args.scheduler, parallelism),
        args.tp,
        args.pp,
        args.ep,
        args.disaggregate,
        arrival_rate,
        latency_mode,
        gpu_spec.tdp_watts,
        gpu_spec.cost_per_hour_usd,
    )
}

// ── validate_kernels ──────────────────────────────────────────────────────────

struct RefRow {
    gpu: String,
    model: String,
    op: String,
    batch: u32,
    seq_len: u32,
    ref_ms: f64,
}

fn load_reference_csv(path: &str) -> Result<Vec<RefRow>, Box<dyn std::error::Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .comment(Some(b'#'))
        .trim(csv::Trim::All)
        .from_path(path)?;
    let mut rows = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        rows.push(RefRow {
            gpu: rec.get(0).ok_or("missing gpu")?.to_string(),
            model: rec.get(1).ok_or("missing model")?.to_string(),
            op: rec.get(2).ok_or("missing op")?.to_string(),
            batch: rec.get(3).ok_or("missing batch_size")?.parse()?,
            seq_len: rec.get(4).ok_or("missing seq_len")?.parse()?,
            ref_ms: rec.get(5).ok_or("missing latency_ms")?.parse()?,
        });
    }
    Ok(rows)
}

fn validate_kernels(path: &str) {
    let rows = match load_reference_csv(path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Cannot load reference CSV '{}': {}", path, e);
            return;
        }
    };

    let sep = "─".repeat(72);
    println!("Roofline vs reference  {}", sep);
    println!(
        "{:<16} {:<16} {:<8} {:>6} {:>7}  {:>9} {:>9} {:>8}",
        "GPU", "Model", "Op", "Batch", "SeqLen", "Ref(ms)", "Sim(ms)", "Err%"
    );
    println!("{}", sep);

    struct GroupStats {
        sum_abs_err: f64,
        count: u32,
    }
    let mut groups: std::collections::BTreeMap<(String, String), GroupStats> =
        std::collections::BTreeMap::new();

    for row in &rows {
        let gpu_spec = match GpuSpec::preset(
            &row.gpu
                .to_lowercase()
                .replace("h100-sxm5", "h100")
                .replace("a100-80gb", "a100"),
        ) {
            Some(g) => g,
            None => {
                // Try lowercase match more carefully
                let key = if row.gpu.contains("H100") {
                    "h100"
                } else if row.gpu.contains("A100") {
                    "a100"
                } else if row.gpu.contains("A10G") {
                    "a10g"
                } else {
                    eprintln!("Unknown GPU '{}', skipping", row.gpu);
                    continue;
                };
                match GpuSpec::preset(key) {
                    Some(g) => g,
                    None => {
                        eprintln!("Unknown GPU '{}', skipping", row.gpu);
                        continue;
                    }
                }
            }
        };
        let model = match LlmConfig::preset(&row.model) {
            Some(m) => m,
            None => {
                eprintln!("Unknown model '{}', skipping", row.model);
                continue;
            }
        };

        let cluster = ClusterConfig::single_gpu();
        let sim_ms = match row.op.as_str() {
            "prefill" => {
                gpu_spec.prefill_latency(row.batch, row.seq_len, &model, None, &cluster) * 1000.0
            }
            "decode" => {
                gpu_spec.decode_latency(row.batch, row.seq_len, &model, None, &cluster) * 1000.0
            }
            other => {
                eprintln!("Unknown op '{}', skipping", other);
                continue;
            }
        };

        let err_pct = (sim_ms - row.ref_ms) / row.ref_ms * 100.0;
        let marker = if err_pct.abs() > 20.0 {
            " !"
        } else if err_pct.abs() > 10.0 {
            " ?"
        } else {
            ""
        };
        println!(
            "{:<16} {:<16} {:<8} {:>6} {:>7}  {:>9.2} {:>9.2} {:>+7.1}%{}",
            row.gpu, row.model, row.op, row.batch, row.seq_len, row.ref_ms, sim_ms, err_pct, marker
        );

        let key = (row.gpu.clone(), row.op.clone());
        let entry = groups.entry(key).or_insert(GroupStats {
            sum_abs_err: 0.0,
            count: 0,
        });
        entry.sum_abs_err += err_pct.abs();
        entry.count += 1;
    }

    println!("{}", sep);
    println!("MAPE by group:");
    for ((gpu, op), stats) in &groups {
        let mape = stats.sum_abs_err / stats.count as f64;
        let flag = if mape > 20.0 {
            " ← HIGH"
        } else if mape > 10.0 {
            " ← marginal"
        } else {
            ""
        };
        println!(
            "  {:<16} {:>7}: {:>5.1}% MAPE  (n={}){}",
            gpu, op, mape, stats.count, flag
        );
    }
    println!("{}", sep);
    println!();
    println!("To reduce error: tune mfu_prefill/mfu_decode in GpuSpec::preset().");
    println!(
        "  If sim > ref: increase the corresponding MFU (GPU is more efficient than modeled)."
    );
    println!("  If sim < ref: decrease MFU. Target < 10% MAPE for production accuracy.");
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let mut args = Args::parse();
    apply_system_preset(&mut args);

    if let Some(ref path) = args.validate_kernels {
        validate_kernels(path);
        return;
    }

    let kt: Option<KernelTable> = args.kernel_table.as_deref().and_then(load_kernel_table);

    if is_study_mode(&args) {
        let summaries = run_study(&args, kt.as_ref());
        if let Err(e) = write_study_outputs(&args, &summaries) {
            eprintln!("Error writing study outputs: {e}");
            std::process::exit(1);
        }
        if args.html.is_none() && args.json_out.is_none() {
            println!("{}", serde_json::to_string_pretty(&summaries).unwrap());
        } else {
            if let Some(path) = &args.html {
                eprintln!("Wrote HTML report to {path}");
            }
            if let Some(path) = &args.json_out {
                eprintln!("Wrote JSON results to {path}");
            }
        }
        return;
    }

    // Print model card unless machine-readable output is requested.
    if args.output == "text" {
        print_model_card(&args);
    }

    // ── sweep mode ────────────────────────────────────────────────────────────
    if let Some(rates_str) = &args.sweep_arrival_rates {
        let rates: Vec<f64> = rates_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if rates.is_empty() {
            eprintln!("No valid rates in --sweep-arrival-rates");
            std::process::exit(1);
        }

        let summaries: Vec<RunSummary> = rates
            .par_iter()
            .map(|&rate| run_once(&args, rate, kt.as_ref()))
            .collect();

        match args.output.as_str() {
            "json" => println!("{}", serde_json::to_string_pretty(&summaries).unwrap()),
            "csv" => {
                RunSummary::print_csv_header();
                for s in &summaries {
                    s.print_csv_row();
                }
            }
            _ => {
                for s in &summaries {
                    println!("--- arrival_rate={} req/s ---", s.arrival_rate);
                    s.print_text();
                    println!();
                }
            }
        }
        return;
    }

    // ── single run ────────────────────────────────────────────────────────────
    let summary = run_once(&args, args.arrival_rate, kt.as_ref());

    match args.output.as_str() {
        "json" => summary.print_json(),
        "csv" => {
            RunSummary::print_csv_header();
            summary.print_csv_row();
        }
        _ => summary.print_text(),
    }
}
