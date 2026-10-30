//! The Hello handshake (ARCHITECTURE.md Section 4): the first
//! exchange on the socket. Each side states its protocol version and
//! capability list; a major-version mismatch is a named refusal on
//! both sides, never a hang and never a silent downgrade.
//!
//! Both functions are generic over the transport so the same code
//! runs over a Unix socket in production and in tests. Callers set
//! the socket timeouts from `SeamLimits::hello_timeout_ms` before
//! calling, which is what bounds the exchange.

use std::io::{Read, Write};

use crate::diag::{codes, dbglog};
use crate::error::Error;
use crate::frame::{Frame, SeamLimits};
use crate::message::{Hello, HelloAck, Message, MessageType, Refusal};
use crate::{protocol_major, PROTOCOL_VERSION};

/// The request id of handshake frames.
const HANDSHAKE_REQUEST_ID: u64 = 0;

/// Send a `Refused` frame naming `code`, best effort: the handshake
/// is already failing with `err`, and a failed refusal write does not
/// replace the real error.
fn send_refusal<S: Write>(stream: &mut S, code: &str, message: String, err: &Error) {
    let refusal = Refusal {
        code: code.to_string(),
        message,
        protocol_version: Some(PROTOCOL_VERSION),
    };
    let frame = Frame::of(&Message::Refused(refusal), HANDSHAKE_REQUEST_ID);
    if let Err(write_err) = frame.write_to(stream) {
        dbglog!("could not deliver refusal {code}: {write_err} (original error: {err})");
    }
}

/// Map a received `Refusal` to the typed error it names: a version
/// refusal becomes [`Error::ProtocolVersionMismatch`], any other
/// refusal becomes [`Error::Refused`].
fn refusal_error(refusal: &Refusal) -> Error {
    if refusal.code == codes::PROTOCOL_VERSION_MISMATCH {
        Error::ProtocolVersionMismatch {
            local: PROTOCOL_VERSION,
            peer: refusal.protocol_version.unwrap_or(0),
        }
    } else {
        Error::Refused {
            code: refusal.code.clone(),
            message: refusal.message.clone(),
        }
    }
}

/// The dialing side of the handshake: state our version and
/// capabilities, verify the server's answer. On any mismatch this
/// side sends the named refusal and returns it as the error.
pub fn client_hello<S: Read + Write>(
    stream: &mut S,
    hello: &Hello,
    limits: &SeamLimits,
) -> Result<HelloAck, Error> {
    let frame = Frame::with_version(
        &Message::Hello(hello.clone()),
        HANDSHAKE_REQUEST_ID,
        hello.protocol_version,
    );
    frame.write_to(stream)?;
    dbglog!(
        "hello sent: version {}.{} capabilities {:?}",
        protocol_major(hello.protocol_version),
        crate::protocol_minor(hello.protocol_version),
        hello.capabilities
    );

    let reply = Frame::read_from(stream, limits)?;
    match reply.message()? {
        Message::HelloAck(ack) => {
            if protocol_major(ack.protocol_version) != protocol_major(PROTOCOL_VERSION)
                || protocol_major(reply.header.protocol_version) != protocol_major(PROTOCOL_VERSION)
            {
                let peer =
                    if protocol_major(ack.protocol_version) != protocol_major(PROTOCOL_VERSION) {
                        ack.protocol_version
                    } else {
                        reply.header.protocol_version
                    };
                let err = Error::ProtocolVersionMismatch {
                    local: PROTOCOL_VERSION,
                    peer,
                };
                send_refusal(
                    stream,
                    codes::PROTOCOL_VERSION_MISMATCH,
                    format!("peer speaks major version {}", protocol_major(peer)),
                    &err,
                );
                dbglog!("hello refused: {err}");
                return Err(err);
            }
            dbglog!("hello accepted by {}", ack.peer_name);
            Ok(ack)
        }
        Message::Refused(refusal) => {
            let err = refusal_error(&refusal);
            dbglog!("hello refused by peer: {err}");
            Err(err)
        }
        other => Err(Error::UnexpectedMessage {
            expected: "HelloAck",
            got: other.message_type(),
        }),
    }
}

/// The listening side of the handshake: read the client's `Hello`,
/// verify its major version against ours, answer with `HelloAck`.
/// On a mismatch this side sends the named refusal and returns it as
/// the error, so both sides hold the same named failure.
pub fn server_hello<S: Read + Write>(
    stream: &mut S,
    ack: &HelloAck,
    limits: &SeamLimits,
) -> Result<Hello, Error> {
    let frame = Frame::read_from(stream, limits)?;
    match frame.message()? {
        Message::Hello(hello) => {
            // Both the header and the payload state a version; either
            // one speaking a foreign major is the named refusal.
            let mismatched = [frame.header.protocol_version, hello.protocol_version]
                .into_iter()
                .find(|v| protocol_major(*v) != protocol_major(PROTOCOL_VERSION));
            if let Some(peer) = mismatched {
                let err = Error::ProtocolVersionMismatch {
                    local: PROTOCOL_VERSION,
                    peer,
                };
                send_refusal(
                    stream,
                    codes::PROTOCOL_VERSION_MISMATCH,
                    format!(
                        "peer speaks major version {}, this side speaks {}",
                        protocol_major(peer),
                        protocol_major(PROTOCOL_VERSION)
                    ),
                    &err,
                );
                dbglog!("hello refused: {err}");
                return Err(err);
            }
            let reply = Frame::of(&Message::HelloAck(ack.clone()), frame.header.request_id);
            reply.write_to(stream)?;
            dbglog!("hello accepted from {}", hello.peer_name);
            Ok(hello)
        }
        Message::Refused(refusal) => Err(refusal_error(&refusal)),
        other => {
            let err = Error::UnexpectedMessage {
                expected: "Hello",
                got: other.message_type(),
            };
            send_refusal(
                stream,
                codes::UNEXPECTED_MESSAGE,
                format!("first frame must be Hello, got {:?}", other.message_type()),
                &err,
            );
            Err(err)
        }
    }
}

/// Whether a message type is part of the handshake itself. Small
/// helper for servers structuring their read loop.
pub fn is_handshake(ty: MessageType) -> bool {
    matches!(
        ty,
        MessageType::Hello | MessageType::HelloAck | MessageType::Refused
    )
}
