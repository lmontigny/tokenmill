use rustc_hash::FxHashMap;

use crate::hardware::cluster::ClusterConfig;
use crate::hardware::gpu::GpuState;
use crate::metrics::collector::MetricsCollector;
use crate::model::kv_cache::KvCacheManager;
use crate::model::llm_config::LlmConfig;
use crate::scheduler::chunked_prefill::ChunkedPrefillScheduler;
use crate::scheduler::continuous_batch::ContinuousBatchScheduler;
use crate::workload::request::{InferenceRequest, RequestPhase, RequestState};
use crate::workload::synthetic::SyntheticWorkload;

use super::event::{EventPayload, SimTime};
use super::queue::SimQueue;

pub enum SchedulerKind {
    Continuous(ContinuousBatchScheduler),
    Chunked(ChunkedPrefillScheduler),
}

pub struct Simulator {
    pub clock: SimTime,
    queue: SimQueue,
    gpu: GpuState,
    model: LlmConfig,
    cluster: ClusterConfig,
    scheduler: SchedulerKind,
    pub metrics: MetricsCollector,
    kv: KvCacheManager,

    waiting: Vec<InferenceRequest>,
    // Requests currently in prefill phase
    prefilling: FxHashMap<u64, RequestState>,
    // Requests in decode phase: req_id → (state, current step)
    decoding: FxHashMap<u64, (RequestState, u32)>,
    prefill_progress: FxHashMap<u64, u32>,

    // Whether a BatchDecodeStep is already scheduled (avoid duplicate events)
    decode_scheduled: bool,
}

impl Simulator {
    pub fn new(
        gpu: GpuState,
        model: LlmConfig,
        cluster: ClusterConfig,
        scheduler: SchedulerKind,
        mut workload: SyntheticWorkload,
        kv: KvCacheManager,
    ) -> Self {
        let mut queue = SimQueue::new();
        while let Some((t, req)) = workload.next_arrival() {
            queue.push(t, EventPayload::RequestArrival { req });
        }
        queue.push(0.0, EventPayload::SchedulerTick);

        Self {
            clock: 0.0,
            queue,
            gpu,
            model,
            cluster,
            scheduler,
            metrics: MetricsCollector::new(),
            kv,
            waiting: Vec::new(),
            prefilling: FxHashMap::default(),
            decoding: FxHashMap::default(),
            prefill_progress: FxHashMap::default(),
            decode_scheduled: false,
        }
    }

    pub fn run(&mut self, until: SimTime) {
        while let Some(event) = self.queue.pop() {
            if event.time > until {
                break;
            }
            self.clock = event.time;
            let new_events = match event.payload {
                EventPayload::RequestArrival { req } => self.handle_arrival(req),
                EventPayload::PrefillStart { req_id, gpu_id, chunk_tokens } => {
                    self.handle_prefill_start(req_id, gpu_id, chunk_tokens)
                }
                EventPayload::PrefillDone { req_id, gpu_id } => {
                    self.handle_prefill_done(req_id, gpu_id)
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

    fn handle_arrival(&mut self, req: InferenceRequest) -> Vec<(SimTime, EventPayload)> {
        self.waiting.push(req);
        vec![(self.clock, EventPayload::SchedulerTick)]
    }

    fn handle_scheduler_tick(&mut self) -> Vec<(SimTime, EventPayload)> {
        let mut new_events = Vec::new();
        let running_states: Vec<RequestState> =
            self.prefilling.values().chain(self.decoding.values().map(|(s, _)| s)).cloned().collect();

        match &self.scheduler {
            SchedulerKind::Continuous(_) => {
                let admit = if let SchedulerKind::Continuous(s) = &self.scheduler {
                    s.schedule(&self.waiting, &running_states, self.kv.free_blocks, self.kv.block_size, self.clock).admit
                } else { unreachable!() };
                for req_id in admit {
                    new_events.extend(self.admit_request(req_id, None));
                }
            }
            SchedulerKind::Chunked(_) => {
                let admit = if let SchedulerKind::Chunked(s) = &self.scheduler {
                    s.schedule(&self.waiting, &running_states, self.kv.free_blocks, self.kv.block_size, self.clock).admit
                } else { unreachable!() };
                for (req_id, chunk) in admit {
                    new_events.extend(self.admit_request(req_id, Some(chunk)));
                }
            }
        }

        new_events
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
        state.gpu_id = Some(self.gpu.id);
        self.prefilling.insert(req_id, state);
        self.prefill_progress.insert(req_id, 0);

        self.metrics.record_kv_util(self.kv.utilization());

        vec![(self.clock, EventPayload::PrefillStart { req_id, gpu_id: self.gpu.id, chunk_tokens })]
    }

    fn handle_prefill_start(
        &mut self, req_id: u64, gpu_id: u32, chunk_tokens: u32,
    ) -> Vec<(SimTime, EventPayload)> {
        *self.prefill_progress.entry(req_id).or_insert(0) += chunk_tokens;
        let start = self.gpu.busy_until.max(self.clock);
        let kt = self.gpu.kernel_table.as_ref();
        let latency = self.gpu.spec.prefill_latency(1, chunk_tokens, &self.model, kt, &self.cluster);
        let done_time = start + latency;
        self.gpu.busy_until = done_time;
        vec![(done_time, EventPayload::PrefillDone { req_id, gpu_id })]
    }

    fn handle_prefill_done(&mut self, req_id: u64, gpu_id: u32) -> Vec<(SimTime, EventPayload)> {
        let total_prompt = self.prefilling.get(&req_id).map(|s| s.req.prompt_tokens).unwrap_or(0);
        let done_so_far = self.prefill_progress.get(&req_id).copied().unwrap_or(0);
        let remaining = total_prompt.saturating_sub(done_so_far);

        if remaining > 0 {
            // Still more prefill chunks (chunked-prefill mode)
            let chunk = match &self.scheduler {
                SchedulerKind::Chunked(s) => remaining.min(s.chunk_size),
                _ => remaining,
            };
            *self.prefill_progress.entry(req_id).or_insert(0) += chunk;
            return vec![(self.clock, EventPayload::PrefillStart { req_id, gpu_id, chunk_tokens: chunk })];
        }

        // Prefill complete → move to decode
        if let Some(mut state) = self.prefilling.remove(&req_id) {
            state.first_token_time = Some(self.clock);
            state.phase = RequestPhase::Decoding;
            if let Some(ttft) = state.ttft() {
                self.metrics.record_ttft(ttft);
            }
            self.decoding.insert(req_id, (state, 0));
        }
        self.prefill_progress.remove(&req_id);

        // Kick off decode batch if not already scheduled
        if !self.decode_scheduled {
            self.decode_scheduled = true;
            return vec![(self.clock, EventPayload::BatchDecodeStep { gpu_id })];
        }
        vec![]
    }

    fn handle_batch_decode_step(&mut self, gpu_id: u32) -> Vec<(SimTime, EventPayload)> {
        if self.decoding.is_empty() {
            return vec![];
        }

        // Compute batch decode latency.
        let batch_size = self.decoding.len() as u32;
        let avg_kv_len = {
            let total: u32 = self.decoding.values().map(|(s, step)| s.req.prompt_tokens + step).sum();
            total / batch_size.max(1)
        };
        let kt = self.gpu.kernel_table.as_ref();
        let latency = self.gpu.spec.decode_latency(batch_size, avg_kv_len, &self.model, kt, &self.cluster);

        let start = self.gpu.busy_until.max(self.clock);
        let done_time = start + latency;
        self.gpu.busy_until = done_time;

        // Advance all decoding requests by one token; collect completed ones
        let req_ids: Vec<u64> = self.decoding.keys().copied().collect();
        let mut completed = Vec::new();
        for req_id in req_ids {
            // Read max_steps before taking a mutable borrow to satisfy the borrow checker.
            let max_steps = self.decoding.get(&req_id).map(|(s, _)| s.req.max_output_tokens).unwrap_or(0);
            if let Some((_, step)) = self.decoding.get_mut(&req_id) {
                *step += 1;
                if *step >= max_steps {
                    completed.push(req_id);
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

        // Schedule next batch decode if there are still decoding requests
        if !self.decoding.is_empty() {
            self.decode_scheduled = true;
            new_events.push((done_time, EventPayload::BatchDecodeStep { gpu_id }));
        } else {
            self.decode_scheduled = false;
        }

        // After decode step, check if waiting requests can be admitted
        new_events.push((done_time, EventPayload::SchedulerTick));

        new_events
    }
}
