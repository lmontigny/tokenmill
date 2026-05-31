use crate::engine::event::SimTime;

use super::request::InferenceRequest;

pub trait WorkloadSource: Send {
    fn next_arrival(&mut self) -> Option<(SimTime, InferenceRequest)>;
}
