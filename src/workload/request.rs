use crate::engine::event::SimTime;

pub type RequestId = u64;

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub req_id: RequestId,
    pub prompt_tokens: u32,
    pub max_output_tokens: u32,
    pub arrival_time: SimTime,
}

#[derive(Debug, Clone)]
pub enum RequestPhase {
    Waiting,
    Prefilling,
    Decoding,
    Done,
}

#[derive(Debug, Clone)]
pub struct RequestState {
    pub req: InferenceRequest,
    pub phase: RequestPhase,
    pub start_time: Option<SimTime>,
    pub first_token_time: Option<SimTime>,
    pub completion_time: Option<SimTime>,
    pub gpu_id: Option<u32>,
}

impl RequestState {
    pub fn new(req: InferenceRequest) -> Self {
        Self {
            req,
            phase: RequestPhase::Waiting,
            start_time: None,
            first_token_time: None,
            completion_time: None,
            gpu_id: None,
        }
    }

    pub fn ttft(&self) -> Option<f64> {
        self.first_token_time.map(|t| t - self.req.arrival_time)
    }

    pub fn tpot(&self) -> Option<f64> {
        match (self.first_token_time, self.completion_time) {
            (Some(first), Some(done)) => {
                let steps = self.req.max_output_tokens.saturating_sub(1) as f64;
                if steps > 0.0 { Some((done - first) / steps) } else { None }
            }
            _ => None,
        }
    }
}
