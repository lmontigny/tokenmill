use serde::Serialize;

/// Flat summary of one simulation run — serialises to JSON or CSV.
#[derive(Debug, Clone, Serialize)]
pub struct RunSummary {
    // config
    pub model: String,
    pub gpu: String,
    pub scheduler: String,
    pub tp: u32,
    pub pp: u32,
    pub disaggregate: bool,
    pub arrival_rate: f64,
    pub duration_s: f64,
    pub latency_mode: String,

    // results
    pub completions: u64,
    pub preemptions: u64,
    pub throughput_rps: f64,
    pub token_throughput: f64,
    pub kv_util_mean_pct: f64,

    pub ttft_p50_ms: f64,
    pub ttft_p95_ms: f64,
    pub ttft_p99_ms: f64,

    pub prefill_p50_ms: f64,
    pub prefill_p95_ms: f64,
    pub prefill_p99_ms: f64,

    pub kv_transfer_p50_ms: f64,
    pub kv_transfer_p95_ms: f64,
    pub kv_transfer_p99_ms: f64,

    pub tpot_p50_ms: f64,
    pub tpot_p95_ms: f64,
    pub tpot_p99_ms: f64,
}

impl RunSummary {
    pub fn print_text(&self) {
        println!("=== tokenmill results ===");
        println!("Requests completed : {}", self.completions);
        if self.preemptions > 0 {
            println!("Preemptions        : {}", self.preemptions);
        }
        println!("Throughput         : {:.2} req/s", self.throughput_rps);
        println!("Token throughput   : {:.0} tok/s", self.token_throughput);
        println!("KV utilization     : mean={:.1}%", self.kv_util_mean_pct);
        println!();
        println!(
            "TTFT (ms)           p50={:.1}  p95={:.1}  p99={:.1}",
            self.ttft_p50_ms, self.ttft_p95_ms, self.ttft_p99_ms
        );
        println!(
            "Prefill time (ms)   p50={:.1}  p95={:.1}  p99={:.1}",
            self.prefill_p50_ms, self.prefill_p95_ms, self.prefill_p99_ms
        );
        if self.disaggregate {
            println!(
                "KV transfer (ms)    p50={:.1}  p95={:.1}  p99={:.1}",
                self.kv_transfer_p50_ms, self.kv_transfer_p95_ms, self.kv_transfer_p99_ms
            );
        }
        println!(
            "TPOT (ms)           p50={:.1}  p95={:.1}  p99={:.1}",
            self.tpot_p50_ms, self.tpot_p95_ms, self.tpot_p99_ms
        );
    }

    pub fn print_json(&self) {
        println!("{}", serde_json::to_string_pretty(self).unwrap());
    }

    /// Print CSV header row.
    pub fn print_csv_header() {
        println!(
            "model,gpu,scheduler,tp,pp,disaggregate,arrival_rate,completions,\
                  throughput_rps,token_throughput,kv_util_mean_pct,\
                  ttft_p50_ms,ttft_p95_ms,ttft_p99_ms,\
                  prefill_p50_ms,prefill_p95_ms,prefill_p99_ms,\
                  kv_transfer_p50_ms,tpot_p50_ms,tpot_p95_ms,tpot_p99_ms"
        );
    }

    pub fn print_csv_row(&self) {
        println!("{},{},{},{},{},{},{},{},{:.2},{:.0},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            self.model, self.gpu, self.scheduler, self.tp, self.pp, self.disaggregate,
            self.arrival_rate, self.completions,
            self.throughput_rps, self.token_throughput, self.kv_util_mean_pct,
            self.ttft_p50_ms, self.ttft_p95_ms, self.ttft_p99_ms,
            self.prefill_p50_ms, self.prefill_p95_ms, self.prefill_p99_ms,
            self.kv_transfer_p50_ms,
            self.tpot_p50_ms, self.tpot_p95_ms, self.tpot_p99_ms);
    }
}
