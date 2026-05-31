use std::collections::BinaryHeap;

use super::event::{Event, EventPayload, SimTime};

pub struct SimQueue {
    heap: BinaryHeap<Event>,
    next_seq: u64,
}

impl SimQueue {
    pub fn new() -> Self {
        Self { heap: BinaryHeap::new(), next_seq: 0 }
    }

    pub fn push(&mut self, time: SimTime, payload: EventPayload) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.heap.push(Event { time, seq, payload });
    }

    pub fn pop(&mut self) -> Option<Event> {
        self.heap.pop()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}
