mod engine;
mod hardware;
mod metrics;
mod model;
mod scheduler;
mod workload;

use clap::Parser;

use engine::sim::Simulator;
use hardware::gpu::GpuState;
use model::llm_config::LlmConfig;
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

    /// Max tokens in flight (scheduler budget)
    #[arg(long, default_value_t = 8192)]
    max_batch_tokens: u32,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() {
    let args = Args::parse();

    let gpu_spec = hardware::gpu::GpuSpec::preset(&args.gpu)
        .unwrap_or_else(|| panic!("Unknown GPU preset '{}'. Use: h100, a100, a10g", args.gpu));

    let model = LlmConfig::preset(&args.model)
        .unwrap_or_else(|| panic!("Unknown model preset '{}'. Use: llama-70b, llama-8b, mixtral-8x7b", args.model));

    println!(
        "Running: {} on {} | arrival={} req/s | duration={}s | seed={}",
        model.name, gpu_spec.name, args.arrival_rate, args.duration, args.seed
    );

    let gpu = GpuState::new(0, gpu_spec);
    let scheduler = ContinuousBatchScheduler::new(args.max_batch_tokens);
    let workload = SyntheticWorkload::new(
        args.arrival_rate,
        args.prompt_mean,
        args.output_mean,
        args.duration,
        args.seed,
    );

    let mut sim = Simulator::new(gpu, model, scheduler, workload);
    sim.run(args.duration);
    sim.metrics.print_report();
}
