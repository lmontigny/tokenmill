use rustc_hash::FxHashMap;

use crate::hardware::gpu::GpuState;
use crate::metrics::collector::MetricsCollector;
use crate::model::llm_config::LlmConfig;
use crate::scheduler::continuous_batch::ContinuousBatchScheduler;
use crate::workload::request::{InferenceRequest, RequestPhase, RequestState};
use crate::workload::synthetic::SyntheticWorkload;

use super::event::{EventPayload, SimTime};
use super::queue::SimQueue;

pub struct Simulator {
    pub clock: SimTime,
    queue: SimQueue,
    gpu: GpuState,
    model: LlmConfig,
    scheduler: ContinuousBatchScheduler,
    workload: SyntheticWorkload,
    pub metrics: MetricsCollector,

    waiting: Vec<InferenceRequest>,
    running: FxHashMap<u64, RequestState>,
}

impl Simulator {
    pub fn new(
        gpu: GpuState,
        model: LlmConfig,
        scheduler: ContinuousBatchScheduler,
        mut workload: SyntheticWorkload,
    ) -> Self {
        let mut queue = SimQueue::new();

        // Pre-load all arrivals into the event queue
        while let Some((t, req)) = workload.next_arrival() {
            queue.push(t, EventPayload::RequestArrival { req });
        }

        // Seed first scheduler tick
        queue.push(0.0, EventPayload::SchedulerTick);

        Self {
            clock: 0.0,
            queue,
            gpu,
            model,
            scheduler,
            workload,
            metrics: MetricsCollector::new(),
            waiting: Vec::new(),
            running: FxHashMap::default(),
        }
    }

    pub fn run(&mut self, until: SimTime) {
        while let Some(event) = self.queue.pop() {
            if event.time > until {
                break;
            }
            self.clock = event.time;
            // Handlers return new events to enqueue (avoids &mut aliasing)
            let new_events = match event.payload {
                EventPayload::RequestArrival { req } => self.handle_arrival(req),
                EventPayload::PrefillStart { req_id, gpu_id, chunk_tokens } => {
                    self.handle_prefill_start(req_id, gpu_id, chunk_tokens)
                }
                EventPayload::PrefillDone { req_id, gpu_id } => {
                    self.handle_prefill_done(req_id, gpu_id)
                }
                EventPayload::DecodeStep { req_id, gpu_id, step } => {
                    self.handle_decode_step(req_id, gpu_id, step)
                }
                EventPayload::DecodeComplete { req_id, gpu_id } => {
                    self.handle_decode_complete(req_id, gpu_id)
                }
                EventPayload::SchedulerTick => self.handle_scheduler_tick(),
            };
            for (t, p) in new_events {
                self.queue.push(t, p);
            }
        }
        self.metrics.sim_duration = until;
    }

    fn handle_arrival(&mut self, req: InferenceRequest) -> Vec<(SimTime, EventPayload)> {
        self.waiting.push(req);
        // Trigger scheduler to check if this can be dispatched immediately
        vec![(self.clock, EventPayload::SchedulerTick)]
    }

    fn handle_scheduler_tick(&mut self) -> Vec<(SimTime, EventPayload)> {
        let running: Vec<RequestState> = self.running.values().cloned().collect();
        let decision = self.scheduler.schedule(&self.waiting, &running, self.clock);

        let mut new_events = Vec::new();

        // Admit new requests to prefill
        for req_id in decision.admit {
            if let Some(pos) = self.waiting.iter().position(|r| r.req_id == req_id) {
                let req = self.waiting.remove(pos);
                let prompt_tokens = req.prompt_tokens;
                let mut state = RequestState::new(req);
                state.phase = RequestPhase::Prefilling;
                state.start_time = Some(self.clock);
                state.gpu_id = Some(self.gpu.id);
                self.running.insert(req_id, state);

                new_events.push((
                    self.clock,
                    EventPayload::PrefillStart {
                        req_id,
                        gpu_id: self.gpu.id,
                        chunk_tokens: prompt_tokens,
                    },
                ));
            }
        }

        new_events
    }

    fn handle_prefill_start(
        &mut self,
        req_id: u64,
        gpu_id: u32,
        chunk_tokens: u32,
    ) -> Vec<(SimTime, EventPayload)> {
        let latency = self.gpu.spec.prefill_latency(1, chunk_tokens, &self.model);
        let done_time = self.clock + latency;
        self.gpu.busy_until = self.gpu.busy_until.max(done_time);
        vec![(done_time, EventPayload::PrefillDone { req_id, gpu_id })]
    }

    fn handle_prefill_done(&mut self, req_id: u64, gpu_id: u32) -> Vec<(SimTime, EventPayload)> {
        if let Some(state) = self.running.get_mut(&req_id) {
            state.first_token_time = Some(self.clock);
            state.phase = RequestPhase::Decoding { step: 0 };
            if let Some(ttft) = state.ttft() {
                self.metrics.record_ttft(ttft);
            }
        }
        vec![(self.clock, EventPayload::DecodeStep { req_id, gpu_id, step: 0 })]
    }

    fn handle_decode_step(
        &mut self,
        req_id: u64,
        gpu_id: u32,
        step: u32,
    ) -> Vec<(SimTime, EventPayload)> {
        let max_steps = self.running.get(&req_id).map(|s| s.req.max_output_tokens).unwrap_or(1);
        let kv_len = self.running.get(&req_id).map(|s| s.req.prompt_tokens + step).unwrap_or(0);

        let latency = self.gpu.spec.decode_latency(1, kv_len, &self.model);
        let done_time = self.clock + latency;
        self.gpu.busy_until = self.gpu.busy_until.max(done_time);

        if step + 1 >= max_steps {
            vec![(done_time, EventPayload::DecodeComplete { req_id, gpu_id })]
        } else {
            vec![(done_time, EventPayload::DecodeStep { req_id, gpu_id, step: step + 1 })]
        }
    }

    fn handle_decode_complete(&mut self, req_id: u64, _gpu_id: u32) -> Vec<(SimTime, EventPayload)> {
        if let Some(mut state) = self.running.remove(&req_id) {
            state.completion_time = Some(self.clock);
            state.phase = RequestPhase::Done;
            if let Some(tpot) = state.tpot() {
                self.metrics.record_tpot(tpot);
            }
            self.metrics.record_completion(state.req.max_output_tokens);
        }
        // Wake scheduler to admit next waiting request
        vec![(self.clock, EventPayload::SchedulerTick)]
    }
}
