//! The GPU-resident target driver (TODO T-9.6): the speculative
//! loop over [`wgpu_compute::ResidentSession`], the one
//! GPU-resident forward of the workspace. The session's chain,
//! KV, and head are reused exactly as they stand; this driver
//! only names the loop's operations over them.
//!
//! Checkpoint and release are bookkeeping here, and that is
//! sound rather than lazy: the resident KV is addressed by
//! absolute position (windowed layers by `position % ring`),
//! every attention view derives from the position a chunk ends
//! at, and entries past the current position are never read, so
//! restoring an earlier committed position is the position
//! restore itself ([`wgpu_compute::ResidentSession::rewind_to`],
//! whose span bound the driver's rewind inherits). A checkpoint
//! is therefore the position at checkpoint time, and releasing
//! it has nothing to free. The verify block size is not a
//! driver input either: the resident plan machinery sizes
//! itself per chunk.

use muse_glimmer::{GlimmerConfig, LayerInputCapture, Result as GlimmerResult};
use wgpu_compute::ResidentSession;

use crate::driver::TargetDriver;

/// A [`TargetDriver`] over a GPU-resident session. The session
/// owns the device KV; this wrapper owns nothing else.
pub struct ResidentTarget<'a> {
    session: ResidentSession<'a>,
}

impl<'a> ResidentTarget<'a> {
    /// Wrap a resident session as the loop's target driver.
    pub fn new(session: ResidentSession<'a>) -> Self {
        ResidentTarget { session }
    }

    /// The wrapped resident session (its stats are the serving
    /// counters the bench reads).
    pub fn session(&self) -> &ResidentSession<'a> {
        &self.session
    }

    /// The wrapped resident session, mutably (a caller resetting
    /// between runs goes through here).
    pub fn session_mut(&mut self) -> &mut ResidentSession<'a> {
        &mut self.session
    }
}

impl TargetDriver for ResidentTarget<'_> {
    /// The position at checkpoint time: restoring it is the
    /// whole checkpoint contract on a position-addressed KV
    /// (see the module docs).
    type Checkpoint = usize;

    fn config(&self) -> &GlimmerConfig {
        self.session.config()
    }

    fn position(&self) -> usize {
        self.session.position()
    }

    fn forward(&mut self, tokens: &[u32]) -> GlimmerResult<Vec<f32>> {
        self.session.forward(tokens)
    }

    fn forward_capturing(
        &mut self,
        tokens: &[u32],
        capture_layers: &[usize],
    ) -> GlimmerResult<(Vec<f32>, Vec<LayerInputCapture>)> {
        self.session.forward_capturing(tokens, capture_layers)
    }

    fn forward_block_capturing(
        &mut self,
        tokens: &[u32],
        capture_layers: &[usize],
    ) -> GlimmerResult<(Vec<f32>, Vec<LayerInputCapture>)> {
        self.session.forward_block_capturing(tokens, capture_layers)
    }

    fn checkpoint(&mut self, _rows: usize) -> GlimmerResult<usize> {
        Ok(self.session.position())
    }

    fn rewind(&mut self, _checkpoint: &usize, to: usize) -> GlimmerResult<()> {
        self.session.rewind_to(to)
    }

    fn release(&mut self, _checkpoint: &usize) -> GlimmerResult<()> {
        Ok(())
    }

    /// Forward the completed round to the resident
    /// session's capture accumulator: at the capture diag
    /// level the session fingerprints the proposed ids and
    /// records the accepted count; at every other level the
    /// session records nothing, so this is a pure
    /// observation with no effect on the round.
    fn note_spec_round(&mut self, proposed: &[u32], accepted: usize) {
        let _ = self.session.note_spec_round(proposed, accepted);
    }
}
