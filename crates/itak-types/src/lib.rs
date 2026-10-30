//! itak-types: the shared Go/Rust IPC contract of the iTaK Torch
//! rebuild, in code (TODO T-1.6, ARCHITECTURE.md Section 4).
//!
//! The grid slot (Section 3.1): the shared contracts mirrored from
//! the Go platform's `pkg/types`, plus the seam itself: the frame
//! header with its protocol version, the Section 4 message set as
//! types on both sides, the token ring's backpressure contract, and
//! the Hello handshake whose major-version mismatch is a named
//! refusal on both sides, never a hang, never a silent downgrade.
//!
//! Pure data and framing: this crate depends on nothing and nothing
//! in the hot loop depends on I/O policy here. The payload encoding
//! is JSON, the one agreed encoding of the seam (Go encodes with
//! `encoding/json`; see `json` module and CONTRACT.md).

#![forbid(unsafe_code)]

pub mod diag;
pub mod error;
pub mod frame;
pub mod handshake;
pub mod json;
pub mod message;
pub mod ring;
pub mod types;

pub use diag::codes;
pub use error::Error;
pub use frame::{Frame, FrameHeader, SeamLimits, HEADER_LEN};
pub use handshake::{client_hello, server_hello};
pub use json::{Value, Wire};
pub use message::{Hello, HelloAck, Message, MessageType, Refusal};
pub use ring::{MemRing, SharedRing, TokenRing};
pub use types::{
    ChatMessage, InferenceRequest, InferenceResponse, LatencyStats, ModelInfo, ModelParams, Role,
    TokenUsage, SHAPE_VERSION,
};

/// The protocol version of the seam, carried in every frame header
/// and stated in `Hello`. The major version lives in the high byte,
/// the minor in the low byte: `0x0100` is protocol 1.0. A major
/// mismatch between peers is the named refusal
/// [`codes::PROTOCOL_VERSION_MISMATCH`]; minor differences are
/// compatible by definition and never refuse.
pub const PROTOCOL_VERSION: u16 = 0x0100;

/// The major component of a protocol version.
pub const fn protocol_major(version: u16) -> u16 {
    version >> 8
}

/// The minor component of a protocol version.
pub const fn protocol_minor(version: u16) -> u16 {
    version & 0x00ff
}

/// Whether two protocol versions may speak: their majors must match.
pub const fn versions_compatible(a: u16, b: u16) -> bool {
    protocol_major(a) == protocol_major(b)
}
