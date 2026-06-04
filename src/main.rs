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
    /// Accelerator preset: rubin | b200 | h100 | a100 | a10g | mi355x | mi325x | mi300x | tpu-v8i | tpu-v8t | tpu-v7-ironwood | groq-lpu-v1 | cerebras-cs3
    #[arg(long, default_value = "h100")]
    gpu: String,

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
    let args = Args::parse();

    if let Some(ref path) = args.validate_kernels {
        validate_kernels(path);
        return;
    }

    let kt: Option<KernelTable> = args.kernel_table.as_deref().and_then(load_kernel_table);

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
