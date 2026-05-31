use crate::workload::request::InferenceRequest;

pub type SimTime = f64;
pub type RequestId = u64;
pub type GpuId = u32;

#[derive(Debug, Clone)]
pub enum EventPayload {
    RequestArrival { req: InferenceRequest },
    PrefillStart { req_id: RequestId, gpu_id: GpuId, chunk_tokens: u32 },
    PrefillDone { req_id: RequestId, gpu_id: GpuId },
    // One batch decode iteration: all running decode requests advance one token together.
    BatchDecodeStep { gpu_id: GpuId },
    SchedulerTick,
}

#[derive(Debug, Clone)]
pub struct Event {
    pub time: SimTime,
    pub seq: u64,
    pub payload: EventPayload,
}

// Min-heap: smallest time first. BinaryHeap is max-heap, so we reverse.
impl Ord for Event {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .time
            .partial_cmp(&self.time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(other.seq.cmp(&self.seq))
    }
}
impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.seq == other.seq
    }
}
impl Eq for Event {}
