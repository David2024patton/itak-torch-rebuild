//! The wire frame: a fixed 16-byte header plus a payload.
//!
//! Header layout (all integers little-endian, ARCHITECTURE.md
//! Section 4):
//!
//! ```text
//! offset  size  field
//! 0       2     protocol version (u16; major in the high byte)
//! 2       2     message type (u16; see MessageType codes)
//! 4       8     request id (u64)
//! 12      4     payload length (u32)
//! ```
//!
//! The socket never carries tensors; payloads are the JSON-encoded
//! `itak-types` shapes (see `message::Message`).

use std::io::{Read, Write};

use crate::error::Error;
use crate::message::{Message, MessageType};
use crate::PROTOCOL_VERSION;

/// The fixed header size in bytes.
pub const HEADER_LEN: usize = 16;

/// Operational limits of the seam. Every limit is config with a
/// default (Section 4); nothing here is a compiled-in constant at
/// the call sites, which pass a `SeamLimits` they loaded from config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeamLimits {
    /// Maximum payload length of one frame, in bytes.
    pub max_frame_len: u32,
    /// Capacity of one generation's token ring, in events.
    pub ring_capacity: usize,
    /// Budget for the Hello exchange, in milliseconds. Callers apply
    /// it as the socket read/write timeout around the handshake;
    /// this crate's handshake functions are transport-generic and do
    /// not set socket options themselves.
    pub hello_timeout_ms: u64,
}

impl Default for SeamLimits {
    fn default() -> Self {
        SeamLimits {
            max_frame_len: 1 << 20, // 1 MiB of JSON control payload
            ring_capacity: 1024,
            hello_timeout_ms: 5_000,
        }
    }
}

/// A frame header, decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// Protocol version the sender speaks.
    pub protocol_version: u16,
    /// The message type of the payload.
    pub message_type: MessageType,
    /// Request id; replies and refusals echo the request they answer.
    pub request_id: u64,
    /// Payload length in bytes.
    pub payload_len: u32,
}

impl FrameHeader {
    /// Encode the header to its 16 wire bytes.
    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut out = [0u8; HEADER_LEN];
        out[0..2].copy_from_slice(&self.protocol_version.to_le_bytes());
        out[2..4].copy_from_slice(&self.message_type.code().to_le_bytes());
        out[4..12].copy_from_slice(&self.request_id.to_le_bytes());
        out[12..16].copy_from_slice(&self.payload_len.to_le_bytes());
        out
    }

    /// Decode a header from its 16 wire bytes.
    pub fn from_bytes(bytes: &[u8; HEADER_LEN]) -> Result<FrameHeader, Error> {
        let protocol_version = u16::from_le_bytes([bytes[0], bytes[1]]);
        let type_code = u16::from_le_bytes([bytes[2], bytes[3]]);
        let request_id = u64::from_le_bytes([
            bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11],
        ]);
        let payload_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        Ok(FrameHeader {
            protocol_version,
            message_type: MessageType::from_code(type_code)?,
            request_id,
            payload_len,
        })
    }
}

/// One complete frame: header plus payload bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// The decoded header.
    pub header: FrameHeader,
    /// The payload bytes (JSON; decode via `Message::decode_payload`).
    pub payload: Vec<u8>,
}

impl Frame {
    /// Build a frame for a message at the current protocol version.
    pub fn of(message: &Message, request_id: u64) -> Frame {
        let payload = message.encode_payload();
        Frame {
            header: FrameHeader {
                protocol_version: PROTOCOL_VERSION,
                message_type: message.message_type(),
                request_id,
                payload_len: u32::try_from(payload.len()).unwrap_or(u32::MAX),
            },
            payload,
        }
    }

    /// Build a frame with an explicit protocol version. The version
    /// refusal paths (and their tests) use this to speak a foreign
    /// version on purpose; normal callers use [`Frame::of`].
    pub fn with_version(message: &Message, request_id: u64, protocol_version: u16) -> Frame {
        let mut frame = Frame::of(message, request_id);
        frame.header.protocol_version = protocol_version;
        frame
    }

    /// Decode the payload as the message the header declares.
    pub fn message(&self) -> Result<Message, Error> {
        Message::decode_payload(self.header.message_type, &self.payload)
    }

    /// Write the frame to a transport.
    pub fn write_to<W: Write>(&self, out: &mut W) -> Result<(), Error> {
        out.write_all(&self.header.to_bytes())?;
        out.write_all(&self.payload)?;
        out.flush()?;
        Ok(())
    }

    /// Read one frame from a transport. A header announcing a payload
    /// over the configured limit is the named `FrameTooLarge` refusal,
    /// raised before the payload is read: the connection is unusable
    /// after any framing error and both sides treat it as closed.
    pub fn read_from<R: Read>(input: &mut R, limits: &SeamLimits) -> Result<Frame, Error> {
        let mut header_bytes = [0u8; HEADER_LEN];
        input.read_exact(&mut header_bytes)?;
        let header = FrameHeader::from_bytes(&header_bytes)?;
        if header.payload_len > limits.max_frame_len {
            return Err(Error::FrameTooLarge {
                len: header.payload_len,
                max: limits.max_frame_len,
            });
        }
        let mut payload = vec![0u8; header.payload_len as usize];
        input.read_exact(&mut payload)?;
        Ok(Frame { header, payload })
    }
}
