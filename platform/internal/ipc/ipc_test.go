package ipc

import (
	"bytes"
	"errors"
	"net"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The header layout is the documented 16 bytes, little-endian:
// version at 0, type at 2, request id at 4, payload length at 12.
func TestHeaderLayoutAndRoundTrip(t *testing.T) {
	header := Header{
		ProtocolVersion: 0x0102,
		Type:            MsgGenerate,
		RequestID:       0x0102030405060708,
		PayloadLen:      0x0a0b0c0d,
	}
	raw := header.MarshalBinary()
	if len(raw) != HeaderSize {
		t.Fatalf("header is %d bytes, want %d", len(raw), HeaderSize)
	}
	if raw[0] != 0x02 || raw[1] != 0x01 {
		t.Fatalf("version bytes = %v, want little-endian [02 01]", raw[0:2])
	}
	if raw[2] != 40 || raw[3] != 0 {
		t.Fatalf("type bytes = %v, want Generate code [40 0]", raw[2:4])
	}
	if raw[4] != 8 || raw[11] != 1 {
		t.Fatalf("request id not little-endian: %v", raw[4:12])
	}
	var back Header
	if err := back.UnmarshalBinary(raw[:]); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back != header {
		t.Fatalf("round trip = %+v, want %+v", back, header)
	}

	// An unknown message type code is the named refusal.
	raw[2] = 0xff
	raw[3] = 0x7f
	var bad Header
	if err := bad.UnmarshalBinary(raw[:]); !errors.Is(err, ErrUnknownMessageType) {
		t.Fatalf("unknown type error = %v, want ErrUnknownMessageType", err)
	}
}

// The named refusal: a foreign major is refused, a foreign minor is
// not, and the error names both versions.
func TestCheckProtocolVersionNamedRefusal(t *testing.T) {
	if err := CheckProtocolVersion(ProtocolVersion); err != nil {
		t.Fatalf("same version refused: %v", err)
	}
	if err := CheckProtocolVersion(0x0107); err != nil {
		t.Fatalf("minor difference refused: %v", err)
	}
	err := CheckProtocolVersion(0x0200)
	if !errors.Is(err, ErrProtocolVersionMismatch) {
		t.Fatalf("major mismatch error = %v, want ErrProtocolVersionMismatch", err)
	}
	if !strings.Contains(err.Error(), "peer 2.0") {
		t.Fatalf("refusal should name the peer version: %v", err)
	}
}

// Payload JSON uses exactly the field names the Rust encoder emits,
// so the two sides agree byte for byte on names.
func TestHelloJSONFieldNames(t *testing.T) {
	payload, err := MarshalPayload(Hello{
		ProtocolVersion: ProtocolVersion,
		PeerName:        "platform",
		Capabilities:    []string{"generate"},
	})
	if err != nil {
		t.Fatalf("marshal hello: %v", err)
	}
	want := `{"protocol_version":256,"peer_name":"platform","capabilities":["generate"]}`
	if string(payload) != want {
		t.Fatalf("hello JSON = %s, want %s", payload, want)
	}
}

// A Refusal frame from the Rust side decodes with its stable code.
func TestRefusalJSONDecode(t *testing.T) {
	wire := []byte(`{"code":"protocol_version_mismatch","message":"peer speaks major version 2","protocol_version":256}`)
	var refusal Refusal
	if err := UnmarshalPayload(wire, &refusal); err != nil {
		t.Fatalf("unmarshal refusal: %v", err)
	}
	if refusal.Code != CodeProtocolVersionMismatch {
		t.Fatalf("refusal code = %q, want %q", refusal.Code, CodeProtocolVersionMismatch)
	}
	if refusal.ProtocolVersion == nil || *refusal.ProtocolVersion != ProtocolVersion {
		t.Fatalf("refusal protocol version = %v, want %d", refusal.ProtocolVersion, ProtocolVersion)
	}
}

// Frames round-trip over a stream, and an oversize announcement is
// the named refusal before any payload is read.
func TestFrameRoundTripAndOversizeRefusal(t *testing.T) {
	payload, err := MarshalPayload(Tokenize{Text: "hi"})
	if err != nil {
		t.Fatalf("marshal tokenize: %v", err)
	}
	var buf bytes.Buffer
	if err := WriteFrame(&buf, MsgTokenize, 7, payload); err != nil {
		t.Fatalf("write frame: %v", err)
	}
	frame, err := ReadFrame(&buf, 0)
	if err != nil {
		t.Fatalf("read frame: %v", err)
	}
	if frame.Header.Type != MsgTokenize || frame.Header.RequestID != 7 {
		t.Fatalf("frame header = %+v", frame.Header)
	}
	var tokenize Tokenize
	if err := UnmarshalPayload(frame.Payload, &tokenize); err != nil {
		t.Fatalf("unmarshal tokenize: %v", err)
	}
	if tokenize.Text != "hi" {
		t.Fatalf("tokenize text = %q", tokenize.Text)
	}

	header := Header{
		ProtocolVersion: ProtocolVersion,
		Type:            MsgGenerate,
		RequestID:       1,
		PayloadLen:      DefaultMaxFrameLen + 1,
	}
	raw := header.MarshalBinary()
	if _, err := ReadFrame(bytes.NewReader(raw[:]), 0); !errors.Is(err, ErrFrameTooLarge) {
		t.Fatalf("oversize error = %v, want ErrFrameTooLarge", err)
	}
}

// A Hello exchange over a real Unix socket: the platform's half of
// the T-1.6 proof. The engine half in this test is a stub peer that
// speaks the same framing; the Rust integration tests carry the
// full both-sides proof against the real crate.
func TestHelloExchangeOverUnixSocket(t *testing.T) {
	socket := filepath.Join(t.TempDir(), "engine.sock")
	listener, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer func() { _ = listener.Close() }()

	serverDone := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			serverDone <- err
			return
		}
		defer func() { _ = conn.Close() }()
		_ = conn.SetDeadline(time.Now().Add(10 * time.Second))
		frame, err := ReadFrame(conn, 0)
		if err != nil {
			serverDone <- err
			return
		}
		if frame.Header.Type != MsgHello {
			serverDone <- errors.New("first frame was not Hello")
			return
		}
		var hello Hello
		if err := UnmarshalPayload(frame.Payload, &hello); err != nil {
			serverDone <- err
			return
		}
		if err := CheckProtocolVersion(hello.ProtocolVersion); err != nil {
			serverDone <- err
			return
		}
		ack, err := MarshalPayload(HelloAck{
			ProtocolVersion: ProtocolVersion,
			PeerName:        "engine",
			Capabilities:    []string{"generate", "tokenize"},
			Hardware: HardwareSummary{
				Backend:    "vulkan",
				DeviceName: "AMD Radeon RX 7900 XTX",
				VRAMBytes:  24_000_000_000,
			},
		})
		if err != nil {
			serverDone <- err
			return
		}
		serverDone <- WriteFrame(conn, MsgHelloAck, frame.Header.RequestID, ack)
	}()

	conn, err := net.Dial("unix", socket)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = conn.Close() }()
	_ = conn.SetDeadline(time.Now().Add(10 * time.Second))

	hello, err := MarshalPayload(Hello{
		ProtocolVersion: ProtocolVersion,
		PeerName:        "platform",
		Capabilities:    []string{"generate", "embed"},
	})
	if err != nil {
		t.Fatalf("marshal hello: %v", err)
	}
	if err := WriteFrame(conn, MsgHello, 0, hello); err != nil {
		t.Fatalf("write hello: %v", err)
	}
	reply, err := ReadFrame(conn, 0)
	if err != nil {
		t.Fatalf("read ack: %v", err)
	}
	if reply.Header.Type != MsgHelloAck {
		t.Fatalf("reply type = %d, want HelloAck", reply.Header.Type)
	}
	if err := CheckProtocolVersion(reply.Header.ProtocolVersion); err != nil {
		t.Fatalf("ack version refused: %v", err)
	}
	var ack HelloAck
	if err := UnmarshalPayload(reply.Payload, &ack); err != nil {
		t.Fatalf("unmarshal ack: %v", err)
	}
	if ack.Hardware.Backend != "vulkan" || ack.Hardware.VRAMBytes != 24_000_000_000 {
		t.Fatalf("ack hardware = %+v", ack.Hardware)
	}
	if err := <-serverDone; err != nil {
		t.Fatalf("server: %v", err)
	}
}
