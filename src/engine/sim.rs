use rustc_hash::FxHashMap;

use crate::hardware::cluster::ClusterConfig;
use crate::hardware::gpu::GpuState;
use crate::metrics::collector::MetricsCollector;
use crate::model::kv_cache::KvCacheManager;
use crate::model::llm_config::LlmConfig;
use crate::scheduler::chunked_prefill::ChunkedPrefillScheduler;
use crate::scheduler::continuous_batch::ContinuousBatchScheduler;
use crate::workload::request::{InferenceRequest, RequestPhase, RequestState};
use crate::workload::traits::WorkloadSource;

use super::event::{EventPayload, SimTime};
use super::queue::SimQueue;

pub enum SchedulerKind {
    Continuous(ContinuousBatchScheduler),
    Chunked(ChunkedPrefillScheduler),
}

/// Speculative decoding configuration.
///
/// Each decode "step" runs K draft tokens through a small model, then verifies
/// with the main model in one pass. Expected tokens produced per step:
///   E[tokens] = (1 − γ^{K+1}) / (1 − γ)
/// where γ is the per-token acceptance rate (empirically 0.6–0.8 for domain-matched drafts).
pub struct SpecConfig {
    /// Number of draft tokens to speculate per step (K).
    pub draft_tokens: u32,
    /// Per-token acceptance probability (γ).
    pub acceptance_rate: f64,
    /// Draft model — runs on the same GPU as the main model.
    pub draft_model: LlmConfig,
}

impl SpecConfig {
    pub fn tokens_per_step(&self) -> f64 {
        let k = self.draft_tokens as f64;
        let g = self.acceptance_rate.clamp(0.0, 1.0 - 1e-9);
        (1.0 - g.powf(k + 1.0)) / (1.0 - g)
    }
}

/// Multi-token prediction configuration.
///
/// The main model has K lightweight MTP heads (≈ 1 transformer layer each) appended after
/// the final layer. A single forward pass produces the main token plus K speculative tokens.
/// Overhead per step: K / n_layers × base decode cost.
/// Expected tokens per step: (1 − γ^{K+1}) / (1 − γ)
/// γ is typically higher than speculative decode (0.85–0.95) because the heads share the
/// same residual stream as the main model and are trained jointly.
pub struct MtpConfig {
    /// Number of MTP prediction heads (K).
    pub num_heads: u32,
    /// Per-token acceptance rate (γ).
    pub acceptance_rate: f64,
}

impl MtpConfig {
    pub fn tokens_per_step(&self) -> f64 {
        let k = self.num_heads as f64;
        let g = self.acceptance_rate.clamp(0.0, 1.0 - 1e-9);
        (1.0 - g.powf(k + 1.0)) / (1.0 - g)
    }

    /// Fraction of extra compute added by MTP heads relative to one base decode pass.
    pub fn overhead_fraction(&self, n_layers: u32) -> f64 {
        self.num_heads as f64 / n_layers.max(1) as f64
    }
}

pub struct Simulator {
    pub clock: SimTime,
    queue: SimQueue,
    /// Prefill GPU (or the only GPU in non-disaggregated mode).
    prefill_gpu: GpuState,
    /// Dedicated decode GPU — only present in disaggregated mode.
    /// In non-disaggregated mode this is None and prefill_gpu handles decode too.
    decode_gpu: Option<GpuState>,
    model: LlmConfig,
    cluster: ClusterConfig,
    scheduler: SchedulerKind,
    pub metrics: MetricsCollector,
    kv: KvCacheManager,
    spec: Option<SpecConfig>,
    mtp: Option<MtpConfig>,

    waiting: Vec<InferenceRequest>,
    prefilling: FxHashMap<u64, RequestState>,
    /// Requests in decode: req_id → (state, current_step).
    decoding: FxHashMap<u64, (RequestState, u32)>,
    /// Requests whose KV cache is in transit (disaggregated mode only).
    transferring: FxHashMap<u64, RequestState>,
    prefill_progress: FxHashMap<u64, u32>,
    decode_scheduled: bool,
}

impl Simulator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        prefill_gpu: GpuState,
        decode_gpu: Option<GpuState>,
        model: LlmConfig,
        cluster: ClusterConfig,
        scheduler: SchedulerKind,
        workload: &mut dyn WorkloadSource,
        kv: KvCacheManager,
    ) -> Self {
        let mut queue = SimQueue::new();
        while let Some((t, req)) = workload.next_arrival() {
            queue.push(t, EventPayload::RequestArrival { req });
        }
        queue.push(0.0, EventPayload::SchedulerTick);

        let mut metrics = MetricsCollector::new();
        metrics.disaggregated = cluster.disaggregate;

        Self {
            clock: 0.0,
            queue,
            prefill_gpu,
            decode_gpu,
            model,
            cluster,
            scheduler,
            metrics,
            kv,
            spec: None,
            mtp: None,
            waiting: Vec::new(),
            prefilling: FxHashMap::default(),
            decoding: FxHashMap::default(),
            transferring: FxHashMap::default(),
            prefill_progress: FxHashMap::default(),
            decode_scheduled: false,
        }
    }

    pub fn with_spec(mut self, spec: SpecConfig) -> Self {
        self.spec = Some(spec);
        self
    }

    pub fn with_mtp(mut self, mtp: MtpConfig) -> Self {
        self.mtp = Some(mtp);
        self
    }

    pub fn run(&mut self, until: SimTime) {
        while let Some(event) = self.queue.pop() {
            if event.time > until {
                break;
            }
            self.clock = event.time;
            let new_events = match event.payload {
                EventPayload::RequestArrival { req } => self.handle_arrival(req),
                EventPayload::PrefillStart {
                    req_id,
                    gpu_id,
                    chunk_tokens,
                } => self.handle_prefill_start(req_id, gpu_id, chunk_tokens),
                EventPayload::PrefillDone { req_id, gpu_id } => {
                    self.handle_prefill_done(req_id, gpu_id)
                }
                EventPayload::KvTransfer { req_id, kv_bytes } => {
                    self.handle_kv_transfer(req_id, kv_bytes)
                }
                EventPayload::BatchDecodeStep { gpu_id } => {
                    self.decode_scheduled = false;
                    self.handle_batch_decode_step(gpu_id)
                }
                EventPayload::SchedulerTick => self.handle_scheduler_tick(),
            };
            for (t, p) in new_events {
                self.queue.push(t, p);
            }
        }
        self.metrics.sim_duration = until;
        self.metrics.kv_util_final = self.kv.utilization();
    }

    // ── GPU helpers ──────────────────────────────────────────────────────────

    fn decode_gpu_id(&self) -> u32 {
        self.decode_gpu
            .as_ref()
            .map(|g| g.id)
            .unwrap_or(self.prefill_gpu.id)
    }

    fn decode_gpu_busy_until(&self) -> SimTime {
        self.decode_gpu
            .as_ref()
            .map(|g| g.busy_until)
            .unwrap_or(self.prefill_gpu.busy_until)
    }

    fn set_decode_gpu_busy(&mut self, until: SimTime) {
        if let Some(g) = self.decode_gpu.as_mut() {
            g.busy_until = until;
        } else {
            self.prefill_gpu.busy_until = until;
        }
    }

    // ── Event handlers ───────────────────────────────────────────────────────

    fn handle_arrival(&mut self, req: InferenceRequest) -> Vec<(SimTime, EventPayload)> {
        self.waiting.push(req);
        vec![(self.clock, EventPayload::SchedulerTick)]
    }

    fn handle_scheduler_tick(&mut self) -> Vec<(SimTime, EventPayload)> {
        let mut new_events = Vec::new();
        let running: Vec<RequestState> = self
            .prefilling
            .values()
            .chain(self.transferring.values())
            .chain(self.decoding.values().map(|(s, _)| s))
            .cloned()
            .collect();

        match &self.scheduler {
            SchedulerKind::Continuous(_) => {
                let decision = if let SchedulerKind::Continuous(s) = &self.scheduler {
                    s.schedule(
                        &self.waiting,
                        &running,
                        self.kv.free_blocks,
                        self.kv.block_size,
                        self.clock,
                    )
                } else {
                    unreachable!()
                };
                let had_preemptions = !decision.preempt.is_empty();
                for req_id in decision.preempt {
                    self.handle_preemption(req_id);
                }
                if had_preemptions {
                    // Re-tick immediately so newly freed KV is used.
                    new_events.push((self.clock, EventPayload::SchedulerTick));
                }
                for req_id in decision.admit {
                    new_events.extend(self.admit_request(req_id, None));
                }
            }
            SchedulerKind::Chunked(_) => {
                let decision = if let SchedulerKind::Chunked(s) = &self.scheduler {
                    s.schedule(
                        &self.waiting,
                        &running,
                        self.kv.free_blocks,
                        self.kv.block_size,
                        self.clock,
                    )
                } else {
                    unreachable!()
                };
                let had_preemptions = !decision.preempt.is_empty();
                for req_id in decision.preempt {
                    self.handle_preemption(req_id);
                }
                if had_preemptions {
                    new_events.push((self.clock, EventPayload::SchedulerTick));
                }
                for (req_id, chunk) in decision.admit {
                    new_events.extend(self.admit_request(req_id, Some(chunk)));
                }
            }
        }

        new_events
    }

    /// Evict a request from decode/prefill: free KV, push req back to front of waiting queue.
    /// The request will re-prefill from scratch on next admission (recompute strategy).
    fn handle_preemption(&mut self, req_id: u64) {
        let req = if let Some((state, _)) = self.decoding.remove(&req_id) {
            state.req
        } else if let Some(state) = self.prefilling.remove(&req_id) {
            self.prefill_progress.remove(&req_id);
            state.req
        } else {
            return;
        };
        self.kv.free(req_id);
        self.metrics.record_preemption();
        self.metrics.record_kv_util(self.kv.utilization());
        // Re-insert at front so the request is re-scheduled promptly (LIFO policy).
        self.waiting.insert(0, req);
    }

    fn admit_request(&mut self, req_id: u64, chunk: Option<u32>) -> Vec<(SimTime, EventPayload)> {
        let pos = match self.waiting.iter().position(|r| r.req_id == req_id) {
            Some(p) => p,
            None => return vec![],
        };
        let req = self.waiting.remove(pos);

        let total_tokens = req.prompt_tokens + req.max_output_tokens;
        if self.kv.allocate(req_id, total_tokens).is_err() {
            self.waiting.insert(0, req);
            return vec![];
        }

        let chunk_tokens = chunk.unwrap_or(req.prompt_tokens);
        let mut state = RequestState::new(req);
        state.phase = RequestPhase::Prefilling;
        state.start_time = Some(self.clock);
        state.gpu_id = Some(self.prefill_gpu.id);
        self.prefilling.insert(req_id, state);
        self.prefill_progress.insert(req_id, 0);
        self.metrics.record_kv_util(self.kv.utilization());

        vec![(
            self.clock,
            EventPayload::PrefillStart {
                req_id,
                gpu_id: self.prefill_gpu.id,
                chunk_tokens,
            },
        )]
    }

    fn handle_prefill_start(
        &mut self,
        req_id: u64,
        gpu_id: u32,
        chunk_tokens: u32,
    ) -> Vec<(SimTime, EventPayload)> {
        *self.prefill_progress.entry(req_id).or_insert(0) += chunk_tokens;
        let start = self.prefill_gpu.busy_until.max(self.clock);
        let kt = self.prefill_gpu.kernel_table.as_ref();
        let latency =
            self.prefill_gpu
                .spec
                .prefill_latency(1, chunk_tokens, &self.model, kt, &self.cluster);
        let done_time = start + latency;
        self.prefill_gpu.busy_until = done_time;
        self.metrics.add_prefill_busy(latency);
        vec![(done_time, EventPayload::PrefillDone { req_id, gpu_id })]
    }

    fn handle_prefill_done(&mut self, req_id: u64, _gpu_id: u32) -> Vec<(SimTime, EventPayload)> {
        let total_prompt = self
            .prefilling
            .get(&req_id)
            .map(|s| s.req.prompt_tokens)
            .unwrap_or(0);
        let done_so_far = self.prefill_progress.get(&req_id).copied().unwrap_or(0);
        let remaining = total_prompt.saturating_sub(done_so_far);

        if remaining > 0 {
            let chunk = match &self.scheduler {
                SchedulerKind::Chunked(s) => remaining.min(s.chunk_size),
                _ => remaining,
            };
            *self.prefill_progress.entry(req_id).or_insert(0) += chunk;
            return vec![(
                self.clock,
                EventPayload::PrefillStart {
                    req_id,
                    gpu_id: self.prefill_gpu.id,
                    chunk_tokens: chunk,
                },
            )];
        }

        // Prefill complete — record prefill latency.
        if let Some(state) = self.prefilling.get_mut(&req_id) {
            state.prefill_done_time = Some(self.clock);
            if let Some(pl) = state.prefill_latency() {
                self.metrics.record_prefill_latency(pl);
            }
        }
        self.prefill_progress.remove(&req_id);

        if self.cluster.disaggregate {
            // Disaggregated: transfer KV cache over the network before decode can start.
            let (kv_bytes, state) = if let Some(mut s) = self.prefilling.remove(&req_id) {
                let bytes = self.model.kv_bytes(s.req.prompt_tokens);
                s.phase = RequestPhase::Transferring;
                (bytes, s)
            } else {
                return vec![];
            };
            self.transferring.insert(req_id, state);
            vec![(self.clock, EventPayload::KvTransfer { req_id, kv_bytes })]
        } else {
            // Coupled: decode starts immediately on the same GPU.
            self.move_to_decoding(req_id)
        }
    }

    fn handle_kv_transfer(&mut self, req_id: u64, kv_bytes: u64) -> Vec<(SimTime, EventPayload)> {
        let latency = self.cluster.kv_transfer_latency(kv_bytes);
        let done_time = self.clock + latency;
        // Transfer completes at done_time → move request to decode pool.
        // We schedule a zero-cost "transfer done" by re-using SchedulerTick after the delay,
        // but it's cleaner to inline: push a BatchDecodeStep timed at done_time.
        // First move the state out of transferring into decoding.
        if let Some(mut state) = self.transferring.remove(&req_id) {
            state.phase = RequestPhase::Decoding;
            self.decoding.insert(req_id, (state, 0));
        }

        // Kick off (or join) the decode batch at the transfer completion time.
        if !self.decode_scheduled {
            self.decode_scheduled = true;
            return vec![(
                done_time,
                EventPayload::BatchDecodeStep {
                    gpu_id: self.decode_gpu_id(),
                },
            )];
        }
        vec![]
    }

    fn move_to_decoding(&mut self, req_id: u64) -> Vec<(SimTime, EventPayload)> {
        if let Some(mut state) = self.prefilling.remove(&req_id) {
            state.phase = RequestPhase::Decoding;
            self.decoding.insert(req_id, (state, 0));
        }
        if !self.decode_scheduled {
            self.decode_scheduled = true;
            return vec![(
                self.clock,
                EventPayload::BatchDecodeStep {
                    gpu_id: self.decode_gpu_id(),
                },
            )];
        }
        vec![]
    }

    fn handle_batch_decode_step(&mut self, gpu_id: u32) -> Vec<(SimTime, EventPayload)> {
        if self.decoding.is_empty() {
            return vec![];
        }

        let batch_size = self.decoding.len() as u32;
        let avg_kv_len = {
            let total: u32 = self
                .decoding
                .values()
                .map(|(s, step)| s.req.prompt_tokens + step)
                .sum();
            total / batch_size.max(1)
        };

        // Use decode GPU spec and kernel table (may differ from prefill GPU in disagg mode).
        let decode_spec = self
            .decode_gpu
            .as_ref()
            .map(|g| &g.spec)
            .unwrap_or(&self.prefill_gpu.spec);
        let kt = self
            .decode_gpu
            .as_ref()
            .and_then(|g| g.kernel_table.as_ref())
            .or(self.prefill_gpu.kernel_table.as_ref());

        let base_lat =
            decode_spec.decode_latency(batch_size, avg_kv_len, &self.model, kt, &self.cluster);
        let (latency, tokens_per_step) = if let Some(ref spec) = self.spec {
            // Speculative decode: K serial draft steps + one main-model verify pass.
            let draft_lat = spec.draft_tokens as f64
                * decode_spec.decode_latency(
                    batch_size,
                    avg_kv_len,
                    &spec.draft_model,
                    None,
                    &self.cluster,
                );
            (draft_lat + base_lat, spec.tokens_per_step())
        } else if let Some(ref mtp) = self.mtp {
            // MTP: single forward pass + K lightweight heads (≈1 layer each).
            // Overhead = K / n_layers of base cost; heads run serially after main model.
            let overhead = mtp.overhead_fraction(self.model.n_layers);
            (base_lat * (1.0 + overhead), mtp.tokens_per_step())
        } else {
            (base_lat, 1.0)
        };

        let start = self.decode_gpu_busy_until().max(self.clock);
        let done_time = start + latency;
        self.set_decode_gpu_busy(done_time);
        self.metrics.add_decode_busy(latency);

        // Advance all decoding requests by tokens_per_step.
        let advance = tokens_per_step.round().max(1.0) as u32;
        let req_ids: Vec<u64> = self.decoding.keys().copied().collect();
        let mut completed = Vec::new();
        let mut first_token_ids = Vec::new(); // requests getting their very first token this step

        for req_id in req_ids {
            let max_steps = self
                .decoding
                .get(&req_id)
                .map(|(s, _)| s.req.max_output_tokens)
                .unwrap_or(0);
            if let Some((_, step)) = self.decoding.get_mut(&req_id) {
                if *step == 0 {
                    first_token_ids.push(req_id);
                }
                *step = (*step + advance).min(max_steps);
                if *step >= max_steps {
                    completed.push(req_id);
                }
            }
        }

        // Record first-token time (true TTFT: includes KV transfer in disaggregated mode).
        for req_id in first_token_ids {
            if let Some((state, _)) = self.decoding.get_mut(&req_id) {
                state.first_token_time = Some(done_time);
                let ttft = done_time - state.req.arrival_time;
                self.metrics.record_ttft(ttft);
                if let Some(kv_t) = state.kv_transfer_time() {
                    self.metrics.record_kv_transfer(kv_t);
                }
            }
        }

        for req_id in completed {
            if let Some((mut state, _)) = self.decoding.remove(&req_id) {
                state.completion_time = Some(done_time);
                state.phase = RequestPhase::Done;
                if let Some(tpot) = state.tpot() {
                    self.metrics.record_tpot(tpot);
                }
                self.metrics.record_completion(state.req.max_output_tokens);
                self.kv.free(req_id);
                self.metrics.record_kv_util(self.kv.utilization());
            }
        }

        let mut new_events = Vec::new();
        if !self.decoding.is_empty() {
            self.decode_scheduled = true;
            new_events.push((done_time, EventPayload::BatchDecodeStep { gpu_id }));
        } else {
            self.decode_scheduled = false;
        }
        new_events.push((done_time, EventPayload::SchedulerTick));
        new_events
    }
}
