//! The data plane seam: one ring per active generation, token ids
//! and logprobs flowing engine to platform (Section 4). A full ring
//! applies backpressure to the token loop; it never drops a token.
//!
//! The cross-process shared-memory implementation of this ring is
//! the engine/platform wiring lanes' work (it lands with the socket
//! servers that own the segments). This crate defines the contract
//! as the [`TokenRing`] trait plus [`MemRing`], a real bounded ring
//! with the contract's exact semantics (FIFO, capacity from config,
//! named `RingFull` refusal instead of a drop) that both sides test
//! against today and that serves single-process embeddings of the
//! seam. The mmap-backed ring plugs in behind the same trait without
//! the contract changing.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::error::Error;
use crate::frame::SeamLimits;
use crate::message::TokenEvent;

/// A bounded FIFO ring of token events with backpressure semantics.
pub trait TokenRing {
    /// Push one event. On a full ring this is the named
    /// [`Error::RingFull`] refusal: the producer waits and retries,
    /// the event is never dropped and never overwrites an unread one.
    fn push(&mut self, event: TokenEvent) -> Result<(), Error>;

    /// Pop the oldest event, or `None` when the ring is empty.
    fn pop(&mut self) -> Option<TokenEvent>;

    /// Events currently in the ring.
    fn len(&self) -> usize;

    /// Whether the ring currently holds no events.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum events the ring holds before backpressure.
    fn capacity(&self) -> usize;
}

/// An in-process [`TokenRing`] over a queue. Stands behind the same
/// trait the shared-memory ring will implement; used by the seam's
/// round-trip tests and any single-process embedding.
#[derive(Debug)]
pub struct MemRing {
    queue: VecDeque<TokenEvent>,
    capacity: usize,
}

impl MemRing {
    /// A ring holding at most `capacity` events. Capacity 0 is
    /// clamped to 1: a zero-capacity ring could never pass an event
    /// and would deadlock its producer by construction.
    pub fn new(capacity: usize) -> MemRing {
        MemRing {
            queue: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    /// A ring sized by the seam limits.
    pub fn from_limits(limits: &SeamLimits) -> MemRing {
        MemRing::new(limits.ring_capacity)
    }
}

impl Default for MemRing {
    fn default() -> Self {
        MemRing::from_limits(&SeamLimits::default())
    }
}

impl TokenRing for MemRing {
    fn push(&mut self, event: TokenEvent) -> Result<(), Error> {
        if self.queue.len() >= self.capacity {
            crate::diag::dbglog!("ring full at capacity {}; backpressure", self.capacity);
            return Err(Error::RingFull {
                capacity: self.capacity,
            });
        }
        self.queue.push_back(event);
        Ok(())
    }

    fn pop(&mut self) -> Option<TokenEvent> {
        self.queue.pop_front()
    }

    fn len(&self) -> usize {
        self.queue.len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

/// A ring shared between the two sides of the seam inside one
/// process: the shape the round-trip test uses to stand in for the
/// shared-memory segment until the mmap ring lands behind
/// [`TokenRing`].
pub type SharedRing = Arc<Mutex<MemRing>>;

/// Create a shared ring sized by the seam limits.
pub fn shared_ring(limits: &SeamLimits) -> SharedRing {
    Arc::new(Mutex::new(MemRing::from_limits(limits)))
}
