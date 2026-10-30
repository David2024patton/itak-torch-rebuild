//! Diagnostic seam (David's debugging law): every refusal and handshake
//! event in this crate can emit a structured line when the `debug-log`
//! feature is enabled. With the feature off the macro expands to nothing
//! and the format arguments are never evaluated.

macro_rules! dbglog {
    ($($arg:tt)*) => {
        #[cfg(feature = "debug-log")]
        {
            eprintln!("[itak-types] {}", format!($($arg)*));
        }
        #[cfg(not(feature = "debug-log"))]
        {
            // Referenced so call sites stay warning-clean with the
            // feature off; format_args! does not evaluate Display.
            let _ = format_args!($($arg)*);
        }
    };
}

pub(crate) use dbglog;

/// Stable diagnostic codes for the seam. These strings are part of the
/// contract: they appear in `Refusal` payloads on the wire and in the
/// diagnostic dump the platform can query (ARCHITECTURE.md Section 4),
/// so they never change spelling once shipped.
pub mod codes {
    /// A peer spoke a different major protocol version. The named
    /// refusal for the Hello handshake, on both sides.
    pub const PROTOCOL_VERSION_MISMATCH: &str = "protocol_version_mismatch";
    /// A frame header announced a payload larger than the configured
    /// frame limit.
    pub const FRAME_TOO_LARGE: &str = "frame_too_large";
    /// A frame carried a message type this build does not know.
    pub const UNKNOWN_MESSAGE_TYPE: &str = "unknown_message_type";
    /// A payload did not decode as the message type its header claimed.
    pub const MALFORMED_PAYLOAD: &str = "malformed_payload";
    /// A handshake or request received a message of the wrong type.
    pub const UNEXPECTED_MESSAGE: &str = "unexpected_message";
    /// The shared-memory token ring is full; the producer must wait
    /// (backpressure), never drop.
    pub const RING_FULL: &str = "ring_full";
    /// The peer sent a `Refused` frame; the specific refusal code
    /// travels inside that frame's payload.
    pub const PEER_REFUSED: &str = "peer_refused";
    /// The peer closed the connection mid-exchange.
    pub const CONNECTION_CLOSED: &str = "connection_closed";
}
