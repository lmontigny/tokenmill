mod engine;
mod hardware;
mod metrics;
mod model;
mod scheduler;
mod workload;

use std::path::Path;

use clap::Parser;
use rayon::prelude::*;

use engine::sim::{Simulator, SchedulerKind};
use hardware::cluster::ClusterConfig;
use hardware::gpu::{GpuSpec, GpuState};
use hardware::kernel_table::KernelTable;
use metrics::report::RunSummary;
use model::kv_cache::KvCacheManager;
use model::llm_config::LlmConfig;
use scheduler::chunked_prefill::ChunkedPrefillScheduler;
use scheduler::continuous_batch::ContinuousBatchScheduler;
use workload::synthetic::SyntheticWorkload;
use workload::trace_replay::TraceReplay;

#[derive(Parser, Debug, Clone)]
#[command(name = "inference-sim", about = "LLM inference discrete-event simulator")]
struct Args {
    /// GPU preset: h100 | a100 | a10g
    #[arg(long, default_value = "h100")]
    gpu: String,

    /// Model preset: llama-70b | llama-8b | mixtral-8x7b
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

    /// KV cache total blocks (0 = auto from GPU HBM)
    #[arg(long, default_value_t = 0)]
    kv_blocks: u32,

    /// Tensor parallelism degree
    #[arg(long, default_value_t = 1)]
    tp: u32,

    /// Pipeline parallelism degree
    #[arg(long, default_value_t = 1)]
    pp: u32,

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

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn load_kernel_table(path: &str) -> Option<KernelTable> {
    match KernelTable::from_csv(Path::new(path)) {
        Ok(kt) => Some(kt),
        Err(e) => { eprintln!("Warning: could not load kernel table '{}': {}", path, e); None }
    }
}

fn auto_kv_blocks(gpu_spec: &GpuSpec, model: &LlmConfig, kv_block_size: u32) -> u32 {
    let bytes_per_block = (model.kv_bytes(1) * kv_block_size as u64).max(1);
    let hbm = gpu_spec.hbm_capacity as f64;
    let available = if model.weight_bytes < gpu_spec.hbm_capacity {
        (hbm - model.weight_bytes as f64) * 0.80
    } else {
        hbm * 0.20
    };
    ((available / bytes_per_block as f64) as u32).max(64)
}

fn run_once(args: &Args, arrival_rate: f64, kt: Option<&KernelTable>) -> RunSummary {
    let gpu_spec = GpuSpec::preset(&args.gpu)
        .unwrap_or_else(|| panic!("Unknown GPU '{}'", args.gpu));
    let model = LlmConfig::preset(&args.model)
        .unwrap_or_else(|| panic!("Unknown model '{}'", args.model));

    let kv_total_blocks = if args.kv_blocks > 0 {
        args.kv_blocks
    } else {
        auto_kv_blocks(&gpu_spec, &model, args.kv_block_size)
    };

    let scheduler = match args.scheduler.as_str() {
        "continuous-batch" => SchedulerKind::Continuous(ContinuousBatchScheduler::new(args.max_batch_tokens)),
        "chunked-prefill"  => SchedulerKind::Chunked(ChunkedPrefillScheduler::new(args.chunk_size, args.max_batch_tokens)),
        other => panic!("Unknown scheduler '{}'", other),
    };

    let cluster = ClusterConfig {
        tp: args.tp, pp: args.pp,
        nvlink_bw: gpu_spec.nvlink_bandwidth,
        internode_bw: args.internode_bw_gbps * 1e9,
        disaggregate: args.disaggregate,
    };

    let mut prefill_gpu = GpuState::new(0, gpu_spec.clone());
    if let Some(k) = kt { prefill_gpu = prefill_gpu.with_kernel_table(k.clone()); }

    let decode_gpu = if args.disaggregate {
        let mut g = GpuState::new(1, gpu_spec.clone());
        if let Some(k) = kt { g = g.with_kernel_table(k.clone()); }
        Some(g)
    } else {
        None
    };

    let kv = KvCacheManager::new(kv_total_blocks, args.kv_block_size);

    let latency_mode = if kt.is_some() { "kernel-table+roofline" } else { "roofline" };

    let mut sim = if args.workload.starts_with("trace:") {
        let path = &args.workload["trace:".len()..];
        let mut trace = TraceReplay::from_csv(Path::new(path))
            .unwrap_or_else(|e| panic!("Cannot load trace '{}': {}", path, e));
        // Duration = last timestamp in trace + some buffer
        let duration = args.duration;
        Simulator::new(prefill_gpu, decode_gpu, model.clone(), cluster, scheduler, &mut trace, kv)
    } else {
        let mut w = SyntheticWorkload::new(arrival_rate, args.prompt_mean, args.output_mean, args.duration, args.seed);
        Simulator::new(prefill_gpu, decode_gpu, model.clone(), cluster, scheduler, &mut w, kv)
    };

    sim.run(args.duration);

    let parallelism = match (args.tp, args.pp, args.disaggregate) {
        (1, 1, false) => "single-gpu".to_string(),
        (1, 1, true)  => "disaggregated".to_string(),
        (tp, 1, false) => format!("TP={tp}"),
        (1, pp, false) => format!("PP={pp}"),
        (tp, pp, false) => format!("TP={tp} PP={pp}"),
        (tp, pp, true)  => format!("TP={tp} PP={pp} disaggregated"),
    };

    sim.metrics.to_summary(
        &model.name, &gpu_spec.name, &format!("{} {}", args.scheduler, parallelism),
        args.tp, args.pp, args.disaggregate, arrival_rate, latency_mode,
    )
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    // Load kernel table once (shared across sweep runs).
    let kt: Option<KernelTable> = args.kernel_table.as_deref().and_then(load_kernel_table);

    // Announce config (single-run header; sweep prints its own header).
    if args.sweep_arrival_rates.is_none() {
        let latency_mode = if kt.is_some() { "kernel-table+roofline" } else { "roofline" };
        println!(
            "Running: {} on {} | scheduler={} | latency={} | workload={}",
            args.model, args.gpu, args.scheduler, latency_mode, args.workload
        );
    }

    // ── sweep mode ───────────────────────────────────────────────────────────
    if let Some(rates_str) = &args.sweep_arrival_rates {
        let rates: Vec<f64> = rates_str.split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if rates.is_empty() {
            eprintln!("No valid rates in --sweep-arrival-rates");
            std::process::exit(1);
        }

        // Run all rates in parallel with rayon.
        let summaries: Vec<RunSummary> = rates.par_iter()
            .map(|&rate| run_once(&args, rate, kt.as_ref()))
            .collect();

        // Output
        match args.output.as_str() {
            "json" => println!("{}", serde_json::to_string_pretty(&summaries).unwrap()),
            "csv"  => {
                RunSummary::print_csv_header();
                for s in &summaries { s.print_csv_row(); }
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

    // ── single run ───────────────────────────────────────────────────────────
    let summary = run_once(&args, args.arrival_rate, kt.as_ref());

    match args.output.as_str() {
        "json" => summary.print_json(),
        "csv"  => { RunSummary::print_csv_header(); summary.print_csv_row(); }
        _      => summary.print_text(),
    }
}
