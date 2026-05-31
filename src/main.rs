mod engine;
mod hardware;
mod metrics;
mod model;
mod scheduler;
mod workload;

use std::path::Path;

use clap::Parser;

use engine::sim::{Simulator, SchedulerKind};
use hardware::cluster::ClusterConfig;
use hardware::gpu::GpuState;
use hardware::kernel_table::KernelTable;
use model::kv_cache::KvCacheManager;
use model::llm_config::LlmConfig;
use scheduler::chunked_prefill::ChunkedPrefillScheduler;
use scheduler::continuous_batch::ContinuousBatchScheduler;
use workload::synthetic::SyntheticWorkload;

#[derive(Parser, Debug)]
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

    /// Chunk size for chunked-prefill scheduler (tokens)
    #[arg(long, default_value_t = 512)]
    chunk_size: u32,

    /// Request arrival rate (req/s)
    #[arg(long, default_value_t = 5.0)]
    arrival_rate: f64,

    /// Simulation duration (seconds)
    #[arg(long, default_value_t = 60.0)]
    duration: f64,

    /// Mean prompt length (tokens)
    #[arg(long, default_value_t = 512.0)]
    prompt_mean: f64,

    /// Mean output length (tokens)
    #[arg(long, default_value_t = 128.0)]
    output_mean: f64,

    /// Max tokens in flight across all requests
    #[arg(long, default_value_t = 8192)]
    max_batch_tokens: u32,

    /// KV cache block size (tokens per block)
    #[arg(long, default_value_t = 16)]
    kv_block_size: u32,

    /// KV cache total blocks (0 = auto-size to 80% of GPU HBM after weights)
    #[arg(long, default_value_t = 0)]
    kv_blocks: u32,

    /// Tensor parallelism degree
    #[arg(long, default_value_t = 1)]
    tp: u32,

    /// Pipeline parallelism degree
    #[arg(long, default_value_t = 1)]
    pp: u32,

    /// Disaggregate prefill and decode onto separate GPUs connected over internode_bw
    #[arg(long)]
    disaggregate: bool,

    /// Internode bandwidth in GB/s (used for KV transfer in disaggregated mode)
    #[arg(long, default_value_t = 200.0)]
    internode_bw_gbps: f64,

    /// Kernel time table CSV (enables profiled latencies; falls back to roofline on miss)
    #[arg(long)]
    kernel_table: Option<String>,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() {
    let args = Args::parse();

    let gpu_spec = hardware::gpu::GpuSpec::preset(&args.gpu)
        .unwrap_or_else(|| panic!("Unknown GPU '{}'. Use: h100, a100, a10g", args.gpu));

    let model = LlmConfig::preset(&args.model)
        .unwrap_or_else(|| panic!("Unknown model '{}'. Use: llama-70b, llama-8b, mixtral-8x7b", args.model));

    let kv_total_blocks = if args.kv_blocks > 0 {
        args.kv_blocks
    } else {
        let bytes_per_token = model.kv_bytes(1);
        let bytes_per_block = (bytes_per_token * args.kv_block_size as u64).max(1);
        let hbm = gpu_spec.hbm_capacity as f64;
        let available = if model.weight_bytes < gpu_spec.hbm_capacity {
            (hbm - model.weight_bytes as f64) * 0.80
        } else {
            hbm * 0.20
        };
        ((available / bytes_per_block as f64) as u32).max(64)
    };

    let scheduler_name = args.scheduler.as_str();
    let scheduler = match scheduler_name {
        "continuous-batch" => SchedulerKind::Continuous(ContinuousBatchScheduler::new(args.max_batch_tokens)),
        "chunked-prefill"  => SchedulerKind::Chunked(ChunkedPrefillScheduler::new(args.chunk_size, args.max_batch_tokens)),
        other => panic!("Unknown scheduler '{}'. Use: continuous-batch, chunked-prefill", other),
    };

    let latency_mode = if args.kernel_table.is_some() { "kernel-table+roofline" } else { "roofline" };
    let parallelism = match (args.tp, args.pp, args.disaggregate) {
        (1, 1, false) => "single-gpu".to_string(),
        (1, 1, true)  => "disaggregated".to_string(),
        (tp, 1, false) => format!("TP={tp}"),
        (1, pp, false) => format!("PP={pp}"),
        (tp, pp, false) => format!("TP={tp} PP={pp}"),
        (tp, pp, true)  => format!("TP={tp} PP={pp} disaggregated"),
    };
    println!(
        "Running: {} on {} | {} | scheduler={} | latency={} | arrival={} req/s | duration={}s",
        model.name, gpu_spec.name, parallelism, scheduler_name, latency_mode, args.arrival_rate, args.duration
    );
    println!(
        "KV cache: {} blocks × {} tokens/block = {} total token slots",
        kv_total_blocks, args.kv_block_size, kv_total_blocks * args.kv_block_size
    );

    let cluster = ClusterConfig {
        tp: args.tp,
        pp: args.pp,
        nvlink_bw: gpu_spec.nvlink_bandwidth,
        internode_bw: args.internode_bw_gbps * 1e9,
        disaggregate: args.disaggregate,
    };

    let kt = args.kernel_table.as_ref().and_then(|path| {
        match KernelTable::from_csv(Path::new(path)) {
            Ok(kt) => { println!("Kernel table loaded: {}", path); Some(kt) }
            Err(e) => { eprintln!("Warning: could not load kernel table '{}': {}", path, e); None }
        }
    });

    let mut prefill_gpu = GpuState::new(0, gpu_spec.clone());
    if let Some(ref k) = kt { prefill_gpu = prefill_gpu.with_kernel_table(k.clone()); }

    // In disaggregated mode, spin up a second GPU for decode (same spec, id=1).
    let decode_gpu = if args.disaggregate {
        let mut g = GpuState::new(1, gpu_spec);
        if let Some(k) = kt { g = g.with_kernel_table(k); }
        Some(g)
    } else {
        None
    };

    let kv = KvCacheManager::new(kv_total_blocks, args.kv_block_size);
    let workload = SyntheticWorkload::new(
        args.arrival_rate, args.prompt_mean, args.output_mean, args.duration, args.seed,
    );

    let mut sim = Simulator::new(prefill_gpu, decode_gpu, model, cluster, scheduler, workload, kv);
    sim.run(args.duration);
    sim.metrics.print_report();
}
