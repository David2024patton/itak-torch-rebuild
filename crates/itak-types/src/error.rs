//! Named error types for the seam. Nothing in this crate panics on
//! malformed wire input; every refusal is one of these variants, and
//! every variant carries the stable diagnostic code from [`crate::codes`]
//! so the refusal can travel the diagnostic channel by name.

use std::fmt;
use std::io;

use crate::message::MessageType;

/// Errors returned by framing, payload decoding, the ring, and the
/// Hello handshake.
#[derive(Debug)]
pub enum Error {
    /// The underlying transport failed.
    Io(io::Error),
    /// The peer closed the connection before a complete frame arrived.
    Closed,
    /// Major protocol version mismatch, the named refusal of
    /// ARCHITECTURE.md Section 4. `local` is this side's full protocol
    /// version, `peer` is the version the other side stated.
    ProtocolVersionMismatch {
        /// This side's protocol version (major in the high byte).
        local: u16,
        /// The peer's stated protocol version.
        peer: u16,
    },
    /// The peer sent a `Refused` frame for a reason other than a
    /// version mismatch: a named refusal with its stable code.
    Refused {
        /// Stable diagnostic code naming the refusal.
        code: String,
        /// Human-readable detail from the refusing side.
        message: String,
    },
    /// A frame announced a payload larger than the configured limit.
    FrameTooLarge {
        /// Payload length the header announced.
        len: u32,
        /// Configured maximum payload length.
        max: u32,
    },
    /// A frame carried a message type code this build does not know.
    UnknownMessageType(u16),
    /// A payload did not decode as its message type requires.
    MalformedPayload(String),
    /// An exchange received a well-formed message of the wrong type.
    UnexpectedMessage {
        /// What the exchange was waiting for.
        expected: &'static str,
        /// What actually arrived.
        got: MessageType,
    },
    /// The token ring is full; the producer must apply backpressure
    /// and retry, never drop the event.
    RingFull {
        /// Ring capacity in events.
        capacity: usize,
    },
}

impl Error {
    /// The stable diagnostic code naming this error, for the diag
    /// channel and for `Refusal` payloads.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Io(_) | Error::Closed => crate::codes::CONNECTION_CLOSED,
            Error::ProtocolVersionMismatch { .. } => crate::codes::PROTOCOL_VERSION_MISMATCH,
            Error::Refused { .. } => crate::codes::PEER_REFUSED,
            Error::FrameTooLarge { .. } => crate::codes::FRAME_TOO_LARGE,
            Error::UnknownMessageType(_) => crate::codes::UNKNOWN_MESSAGE_TYPE,
            Error::MalformedPayload(_) => crate::codes::MALFORMED_PAYLOAD,
            Error::UnexpectedMessage { .. } => crate::codes::UNEXPECTED_MESSAGE,
            Error::RingFull { .. } => crate::codes::RING_FULL,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "ipc transport error: {e}"),
            Error::Closed => write!(f, "ipc connection closed by peer"),
            Error::ProtocolVersionMismatch { local, peer } => write!(
                f,
                "protocol version mismatch: local {}.{}, peer {}.{} (major versions must match)",
                crate::protocol_major(*local),
                crate::protocol_minor(*local),
                crate::protocol_major(*peer),
                crate::protocol_minor(*peer),
            ),
            Error::Refused { code, message } => write!(f, "peer refused ({code}): {message}"),
            Error::FrameTooLarge { len, max } => {
                write!(
                    f,
                    "frame payload of {len} bytes exceeds the {max} byte limit"
                )
            }
            Error::UnknownMessageType(code) => write!(f, "unknown message type code {code}"),
            Error::MalformedPayload(detail) => write!(f, "malformed payload: {detail}"),
            Error::UnexpectedMessage { expected, got } => {
                write!(f, "expected {expected}, got {got:?}")
            }
            Error::RingFull { capacity } => {
                write!(
                    f,
                    "token ring full at capacity {capacity}; apply backpressure"
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            Error::Closed
        } else {
            Error::Io(e)
        }
    }
}
