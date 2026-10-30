//! Seam proof tests for TODO T-1.6: the Hello round trip over a real
//! Unix socket with the token ring in play, and the major-version
//! mismatch producing the named refusal on both sides. Everything
//! here runs over real sockets in a temp directory; the only stand-in
//! is the in-process ring behind the `TokenRing` trait, which is the
//! documented boundary where the shared-memory ring lands.

use std::io::Cursor;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use itak_types::frame::{Frame, SeamLimits, HEADER_LEN};
use itak_types::handshake::{client_hello, server_hello};
use itak_types::message::{
    Ack, CoverageReport, CreateSession, DestroySession, Detokenize, DetokenizeResult, DiagEntry,
    Diagnostics, Embed, EmbedResult, Generate, GrammarKind, GrammarSpec, HardwareSummary, Health,
    Hello, HelloAck, KvPolicy, LoadError, LoadModel, LoadResult, LoadState, Message, MessageType,
    Refusal, ReleaseShm, SamplingParams, SaveState, SessionCreated, ShmRef, Shutdown, StateResult,
    StopRules, TokenEvent, Tokenize, TokenizeResult, UnloadModel,
};
use itak_types::ring::{MemRing, SharedRing, TokenRing};
use itak_types::types::{ChatMessage, ModelInfo, Role, SHAPE_VERSION};
use itak_types::{codes, protocol_major, Error, PROTOCOL_VERSION};

fn socket_path(name: &str) -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("itak-types-{}-{}.sock", std::process::id(), name));
    let _ = std::fs::remove_file(&path);
    path
}

fn client_hello_msg(version: u16) -> Hello {
    Hello {
        protocol_version: version,
        peer_name: "platform".to_string(),
        capabilities: vec!["generate".to_string(), "embed".to_string()],
    }
}

fn server_ack() -> HelloAck {
    HelloAck {
        protocol_version: PROTOCOL_VERSION,
        peer_name: "engine".to_string(),
        capabilities: vec!["generate".to_string(), "tokenize".to_string()],
        hardware: HardwareSummary {
            backend: "vulkan".to_string(),
            device_name: "AMD Radeon RX 7900 XTX".to_string(),
            vram_bytes: 24_000_000_000,
        },
    }
}

fn timed(stream: &UnixStream) {
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("set read timeout");
    stream
        .set_write_timeout(Some(Duration::from_secs(10)))
        .expect("set write timeout");
}

#[test]
fn hello_round_trip_over_unix_socket_with_ring() {
    let limits = SeamLimits::default();
    let path = socket_path("roundtrip");
    let listener = UnixListener::bind(&path).expect("bind listener");
    let ring: SharedRing = itak_types::ring::shared_ring(&limits);
    let server_ring = Arc::clone(&ring);

    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        timed(&stream);
        // The handshake: the server learns the client's hello and
        // answers with its hardware summary.
        let hello = server_hello(&mut stream, &server_ack(), &limits).expect("server hello");
        assert_eq!(hello.peer_name, "platform");
        assert_eq!(hello.protocol_version, PROTOCOL_VERSION);

        // One control round trip: Tokenize, answered by the engine.
        let frame = Frame::read_from(&mut stream, &limits).expect("read tokenize");
        assert_eq!(frame.header.message_type, MessageType::Tokenize);
        let request_id = frame.header.request_id;
        let tokenize = match frame.message().expect("decode tokenize") {
            Message::Tokenize(t) => t,
            other => panic!("expected Tokenize, got {other:?}"),
        };
        let ids: Vec<u32> = tokenize.text.bytes().map(u32::from).collect();
        let reply = Frame::of(
            &Message::TokenizeResult(TokenizeResult { token_ids: ids }),
            request_id,
        );
        reply.write_to(&mut stream).expect("write tokenize result");

        // The data plane: three token events through the ring, then
        // one socket wakeup announcing the final sequence.
        let events = [
            TokenEvent {
                token_id: 11,
                logprob: Some(-0.5),
                ring_seq: 1,
                finished: false,
                finish_reason: None,
            },
            TokenEvent {
                token_id: 12,
                logprob: Some(-0.25),
                ring_seq: 2,
                finished: false,
                finish_reason: None,
            },
            TokenEvent {
                token_id: 13,
                logprob: None,
                ring_seq: 3,
                finished: true,
                finish_reason: Some("stop".to_string()),
            },
        ];
        {
            let mut ring = server_ring.lock().expect("ring lock");
            for event in &events {
                ring.push(event.clone()).expect("ring push");
            }
        }
        let wakeup = Frame::of(&Message::TokenEvent(events[2].clone()), 99);
        wakeup.write_to(&mut stream).expect("write wakeup");
        hello
    });

    let mut stream = UnixStream::connect(&path).expect("connect");
    timed(&stream);
    let ack = client_hello(&mut stream, &client_hello_msg(PROTOCOL_VERSION), &limits)
        .expect("client hello");
    assert_eq!(ack.peer_name, "engine");
    assert_eq!(ack.hardware.backend, "vulkan");
    assert_eq!(ack.hardware.vram_bytes, 24_000_000_000);

    // Control round trip.
    let request = Frame::of(
        &Message::Tokenize(Tokenize {
            session_handle: None,
            text: "hi".to_string(),
        }),
        7,
    );
    request.write_to(&mut stream).expect("write tokenize");
    let reply = Frame::read_from(&mut stream, &limits).expect("read result");
    assert_eq!(reply.header.request_id, 7);
    match reply.message().expect("decode result") {
        Message::TokenizeResult(result) => {
            assert_eq!(result.token_ids, vec![u32::from(b'h'), u32::from(b'i')]);
        }
        other => panic!("expected TokenizeResult, got {other:?}"),
    }

    // The wakeup arrives; the events themselves come from the ring,
    // in order, none dropped.
    let wakeup = Frame::read_from(&mut stream, &limits).expect("read wakeup");
    assert_eq!(wakeup.header.message_type, MessageType::TokenEvent);
    let mut popped = Vec::new();
    {
        let mut ring = ring.lock().expect("ring lock");
        while let Some(event) = ring.pop() {
            popped.push(event);
        }
        assert!(ring.is_empty());
    }
    assert_eq!(popped.len(), 3);
    assert_eq!(popped[0].token_id, 11);
    assert_eq!(popped[1].ring_seq, 2);
    assert!(popped[2].finished);
    assert_eq!(popped[2].finish_reason.as_deref(), Some("stop"));

    server.join().expect("server thread");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn version_mismatch_refused_by_server_and_named_on_the_wire() {
    let limits = SeamLimits::default();
    let path = socket_path("mismatch-server");
    let listener = UnixListener::bind(&path).expect("bind listener");

    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        timed(&stream);
        server_hello(&mut stream, &server_ack(), &limits)
    });

    // A client speaking protocol 2.0 to a 1.x server.
    let mut stream = UnixStream::connect(&path).expect("connect");
    timed(&stream);
    let client_err = client_hello(&mut stream, &client_hello_msg(0x0200), &limits)
        .expect_err("client must be refused");
    assert_eq!(client_err.code(), codes::PROTOCOL_VERSION_MISMATCH);
    assert!(
        matches!(client_err, Error::ProtocolVersionMismatch { .. }),
        "client error must be the named refusal, got {client_err:?}"
    );

    // The server side holds the same named refusal, with both
    // versions pinned.
    let server_err = server
        .join()
        .expect("server thread")
        .expect_err("server must refuse");
    match server_err {
        Error::ProtocolVersionMismatch { local, peer } => {
            assert_eq!(local, PROTOCOL_VERSION);
            assert_eq!(peer, 0x0200);
            assert_ne!(protocol_major(local), protocol_major(peer));
        }
        other => panic!("expected ProtocolVersionMismatch, got {other:?}"),
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn version_mismatch_refusal_frame_carries_the_name() {
    // Speak the foreign version by hand and read the refusal frame
    // off the wire: it must be a Refused frame whose code is the
    // stable name, so a peer that is not this crate can still name
    // the refusal.
    let limits = SeamLimits::default();
    let path = socket_path("mismatch-frame");
    let listener = UnixListener::bind(&path).expect("bind listener");
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        timed(&stream);
        let _ = server_hello(&mut stream, &server_ack(), &limits);
    });

    let mut stream = UnixStream::connect(&path).expect("connect");
    timed(&stream);
    let hello = Frame::with_version(&Message::Hello(client_hello_msg(0x0900)), 0, 0x0900);
    hello.write_to(&mut stream).expect("write foreign hello");
    let refusal_frame = Frame::read_from(&mut stream, &limits).expect("read refusal");
    assert_eq!(refusal_frame.header.message_type, MessageType::Refused);
    match refusal_frame.message().expect("decode refusal") {
        Message::Refused(refusal) => {
            assert_eq!(refusal.code, codes::PROTOCOL_VERSION_MISMATCH);
            assert_eq!(refusal.protocol_version, Some(PROTOCOL_VERSION));
        }
        other => panic!("expected Refused, got {other:?}"),
    }
    server.join().expect("server thread");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn version_mismatch_refused_by_client() {
    // The mirror image: the server answers with a HelloAck speaking
    // protocol 3.0. The client must refuse by name and send the
    // refusal back, so the server side does not hang either.
    let limits = SeamLimits::default();
    let path = socket_path("mismatch-client");
    let listener = UnixListener::bind(&path).expect("bind listener");

    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        timed(&stream);
        let hello_frame = Frame::read_from(&mut stream, &limits).expect("read hello");
        assert_eq!(hello_frame.header.message_type, MessageType::Hello);
        let mut ack = server_ack();
        ack.protocol_version = 0x0300;
        let reply = Frame::with_version(&Message::HelloAck(ack), 0, 0x0300);
        reply.write_to(&mut stream).expect("write foreign ack");
        // The client's named refusal must arrive.
        let refusal = Frame::read_from(&mut stream, &limits).expect("read refusal");
        assert_eq!(refusal.header.message_type, MessageType::Refused);
        match refusal.message().expect("decode refusal") {
            Message::Refused(r) => assert_eq!(r.code, codes::PROTOCOL_VERSION_MISMATCH),
            other => panic!("expected Refused, got {other:?}"),
        }
    });

    let mut stream = UnixStream::connect(&path).expect("connect");
    timed(&stream);
    let err = client_hello(&mut stream, &client_hello_msg(PROTOCOL_VERSION), &limits)
        .expect_err("client must refuse the foreign ack");
    match err {
        Error::ProtocolVersionMismatch { local, peer } => {
            assert_eq!(local, PROTOCOL_VERSION);
            assert_eq!(peer, 0x0300);
        }
        other => panic!("expected ProtocolVersionMismatch, got {other:?}"),
    }
    server.join().expect("server thread");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn minor_version_difference_does_not_refuse() {
    // Same major, different minor: compatible by definition.
    let limits = SeamLimits::default();
    let path = socket_path("minor-ok");
    let listener = UnixListener::bind(&path).expect("bind listener");
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        timed(&stream);
        server_hello(&mut stream, &server_ack(), &limits).expect("server accepts minor difference")
    });
    let mut stream = UnixStream::connect(&path).expect("connect");
    timed(&stream);
    let ack = client_hello(&mut stream, &client_hello_msg(0x0107), &limits)
        .expect("client with minor 7 is accepted");
    assert_eq!(ack.protocol_version, PROTOCOL_VERSION);
    server.join().expect("server thread");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn frame_round_trip_and_oversize_refusal() {
    let limits = SeamLimits::default();
    let message = Message::Generate(Generate {
        session_handle: 5,
        token_ids: Some(vec![1, 2, 3]),
        prompt_ref: None,
        sampling: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            max_tokens: 16,
            repeat_penalty: 1.0,
            seed: Some(42),
        },
        grammar: None,
        stop: StopRules {
            sequences: vec!["</s>".to_string()],
            token_ids: vec![2],
        },
    });
    let frame = Frame::of(&message, 1234);
    let mut buf: Vec<u8> = Vec::new();
    frame.write_to(&mut buf).expect("write frame");
    assert_eq!(buf.len(), HEADER_LEN + frame.payload.len());

    let mut cursor = Cursor::new(buf);
    let back = Frame::read_from(&mut cursor, &limits).expect("read frame");
    assert_eq!(back.header.request_id, 1234);
    assert_eq!(back.header.protocol_version, PROTOCOL_VERSION);
    assert_eq!(back.message().expect("decode generate"), message);

    // A header announcing a payload past the limit is refused by
    // name before any payload is read.
    let mut header = itak_types::FrameHeader {
        protocol_version: PROTOCOL_VERSION,
        message_type: MessageType::Generate,
        request_id: 1,
        payload_len: limits.max_frame_len + 1,
    }
    .to_bytes()
    .to_vec();
    header.extend_from_slice(b"trailing bytes are never read as payload");
    let mut cursor = Cursor::new(header);
    let err = Frame::read_from(&mut cursor, &limits).expect_err("oversize must refuse");
    assert_eq!(err.code(), codes::FRAME_TOO_LARGE);
    assert!(matches!(err, Error::FrameTooLarge { .. }));
}

#[test]
fn header_layout_is_the_documented_16_bytes() {
    let header = itak_types::FrameHeader {
        protocol_version: 0x0102,
        message_type: MessageType::Generate,
        request_id: 0x0102_0304_0506_0708,
        payload_len: 0x0a0b_0c0d,
    };
    let bytes = header.to_bytes();
    assert_eq!(bytes.len(), HEADER_LEN);
    assert_eq!(&bytes[0..2], &[0x02, 0x01]); // version, little-endian
    assert_eq!(&bytes[2..4], &[40, 0]); // Generate code
    assert_eq!(&bytes[4..12], &[8, 7, 6, 5, 4, 3, 2, 1]); // request id LE
    assert_eq!(&bytes[12..16], &[0x0d, 0x0c, 0x0b, 0x0a]); // length LE
    let back = itak_types::FrameHeader::from_bytes(&bytes).expect("decode header");
    assert_eq!(back, header);

    // An unknown message type code is refused by name.
    let mut bad = bytes;
    bad[2] = 0xff;
    bad[3] = 0x7f;
    let err = itak_types::FrameHeader::from_bytes(&bad).expect_err("unknown type must refuse");
    assert_eq!(err.code(), codes::UNKNOWN_MESSAGE_TYPE);
}

#[test]
fn ring_backpressure_never_drops() {
    let mut ring = MemRing::new(2);
    let event = |seq: u64| TokenEvent {
        token_id: 100 + u32::try_from(seq).expect("seq fits"),
        logprob: None,
        ring_seq: seq,
        finished: false,
        finish_reason: None,
    };
    ring.push(event(1)).expect("push 1");
    ring.push(event(2)).expect("push 2");
    let err = ring.push(event(3)).expect_err("full ring must refuse");
    assert_eq!(err.code(), codes::RING_FULL);
    // Nothing was dropped or overwritten: FIFO order intact.
    assert_eq!(ring.pop().expect("pop 1").ring_seq, 1);
    ring.push(event(3)).expect("push after pop");
    assert_eq!(ring.pop().expect("pop 2").ring_seq, 2);
    assert_eq!(ring.pop().expect("pop 3").ring_seq, 3);
    assert!(ring.pop().is_none());
}

#[test]
fn whole_message_set_round_trips() {
    let model = ModelInfo {
        version: SHAPE_VERSION,
        id: "glimmer-30b".to_string(),
        architecture: "muse-glimmer".to_string(),
        context_size: 131_072,
        parameter_count: 30_000_000_000,
        quantization: "Q4_K_M".to_string(),
    };
    let messages: Vec<Message> = vec![
        Message::Hello(client_hello_msg(PROTOCOL_VERSION)),
        Message::HelloAck(server_ack()),
        Message::Refused(Refusal {
            code: codes::PROTOCOL_VERSION_MISMATCH.to_string(),
            message: "peer speaks major version 2".to_string(),
            protocol_version: Some(PROTOCOL_VERSION),
        }),
        Message::Ack(Ack {
            ok: true,
            detail: String::new(),
        }),
        Message::LoadModel(LoadModel {
            path: "/models/glimmer.gguf".to_string(),
            expected_architecture: Some("muse-glimmer".to_string()),
        }),
        Message::LoadResult(LoadResult::Loaded {
            model_handle: 3,
            model: model.clone(),
            coverage: CoverageReport {
                tensors_expected: 10,
                tensors_loaded: 10,
                missing: vec![],
                unreadable: vec![],
            },
        }),
        Message::LoadResult(LoadResult::Refused {
            error: LoadError {
                code: "tensor_missing".to_string(),
                message: "required tensor absent".to_string(),
                missing: vec!["blk.0.attn_q.weight".to_string()],
            },
        }),
        Message::UnloadModel(UnloadModel { model_handle: 3 }),
        Message::CreateSession(CreateSession {
            model_handle: 3,
            context_size: 8192,
            kv_policy: KvPolicy::Hybrid,
        }),
        Message::SessionCreated(SessionCreated { session_handle: 9 }),
        Message::DestroySession(DestroySession { session_handle: 9 }),
        Message::Tokenize(Tokenize {
            session_handle: Some(9),
            text: "héllo Ġ world".to_string(),
        }),
        Message::TokenizeResult(TokenizeResult {
            token_ids: vec![1, 2, 3, u32::MAX],
        }),
        Message::Detokenize(Detokenize {
            token_ids: vec![4, 5],
        }),
        Message::DetokenizeResult(DetokenizeResult {
            text: "done".to_string(),
        }),
        Message::Generate(Generate {
            session_handle: 9,
            token_ids: None,
            prompt_ref: Some(ShmRef {
                segment: "prompt-9".to_string(),
                offset: 0,
                length: 4096,
            }),
            sampling: SamplingParams {
                temperature: 0.7,
                top_p: 0.95,
                top_k: 40,
                max_tokens: 256,
                repeat_penalty: 1.1,
                seed: None,
            },
            grammar: Some(GrammarSpec {
                kind: GrammarKind::JsonSchema,
                source: "{\"type\":\"object\"}".to_string(),
            }),
            stop: StopRules {
                sequences: vec![],
                token_ids: vec![0],
            },
        }),
        Message::TokenEvent(TokenEvent {
            token_id: 77,
            logprob: Some(-1.25),
            ring_seq: 5,
            finished: true,
            finish_reason: Some("length".to_string()),
        }),
        Message::Abort(itak_types::message::Abort {
            target_request_id: 55,
        }),
        Message::Embed(Embed {
            texts: vec!["a".to_string(), "b".to_string()],
        }),
        Message::EmbedResult(EmbedResult {
            vectors: ShmRef {
                segment: "emb-1".to_string(),
                offset: 16,
                length: 8192,
            },
            dimension: 1024,
            count: 2,
        }),
        Message::ReleaseShm(ReleaseShm {
            segment: "emb-1".to_string(),
        }),
        Message::SaveState(SaveState {
            session_handle: 9,
            path: "s9.state".to_string(),
        }),
        Message::LoadState(LoadState {
            session_handle: 9,
            path: "s9.state".to_string(),
        }),
        Message::StateResult(StateResult::Done {
            path: "s9.state".to_string(),
        }),
        Message::Health(Health {}),
        Message::Diagnostics(Diagnostics {
            healthy: true,
            engine_version: "0.1.0".to_string(),
            entries: vec![DiagEntry {
                code: codes::RING_FULL.to_string(),
                message: "ring filled once".to_string(),
                count: 1,
            }],
        }),
        Message::Shutdown(Shutdown {
            drain_timeout_ms: 2000,
        }),
    ];
    // Every Section 4 message type appears, with LoadResult in both
    // of its shapes.
    assert_eq!(messages.len(), 27);
    for message in &messages {
        let payload = message.encode_payload();
        let back = Message::decode_payload(message.message_type(), &payload)
            .unwrap_or_else(|e| panic!("decode {:?} failed: {e}", message.message_type()));
        assert_eq!(&back, message, "round trip of {:?}", message.message_type());
    }

    // The mirrored contracts round-trip too.
    let response = itak_types::InferenceResponse {
        version: SHAPE_VERSION,
        model: "glimmer-30b".to_string(),
        message: ChatMessage {
            version: SHAPE_VERSION,
            role: Role::Assistant,
            content: "answer".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
        usage: itak_types::TokenUsage {
            version: SHAPE_VERSION,
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        },
        latency: itak_types::LatencyStats {
            version: SHAPE_VERSION,
            time_to_first_token_ms: 42.5,
            total_time_ms: 100.0,
            tokens_per_second: 50.0,
        },
    };
    let encoded = itak_types::Wire::encode(&response);
    let decoded = <itak_types::InferenceResponse as itak_types::Wire>::decode(&encoded)
        .expect("decode response");
    assert_eq!(decoded, response);
}
