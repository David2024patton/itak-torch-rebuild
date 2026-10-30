package ipc

import (
	"encoding/binary"
	"fmt"
	"io"
)

// HeaderSize is the fixed frame header size in bytes.
const HeaderSize = 16

// DefaultMaxFrameLen is the default maximum payload length of one
// frame, in bytes. Call sites load the real limit from config; this
// is the default the config falls back to, matching the Rust side.
const DefaultMaxFrameLen uint32 = 1 << 20

// Header is the fixed frame header of ARCHITECTURE.md Section 4.
// All integers are little-endian on the wire: protocol version
// (u16), message type (u16), request id (u64), payload length (u32).
type Header struct {
	// ProtocolVersion is the version the sender speaks.
	ProtocolVersion uint16
	// Type is the message type of the payload.
	Type MsgType
	// RequestID correlates requests, replies, and refusals.
	RequestID uint64
	// PayloadLen is the payload length in bytes.
	PayloadLen uint32
}

// MarshalBinary encodes the header to its 16 wire bytes.
func (h Header) MarshalBinary() [HeaderSize]byte {
	var out [HeaderSize]byte
	binary.LittleEndian.PutUint16(out[0:2], h.ProtocolVersion)
	binary.LittleEndian.PutUint16(out[2:4], uint16(h.Type))
	binary.LittleEndian.PutUint64(out[4:12], h.RequestID)
	binary.LittleEndian.PutUint32(out[12:16], h.PayloadLen)
	return out
}

// UnmarshalBinary decodes a header from its 16 wire bytes. An
// unknown message type code is the named ErrUnknownMessageType
// refusal, matching the Rust side.
func (h *Header) UnmarshalBinary(data []byte) error {
	if len(data) != HeaderSize {
		return fmt.Errorf("%w: header is %d bytes, want %d", ErrMalformedPayload, len(data), HeaderSize)
	}
	typ := MsgType(binary.LittleEndian.Uint16(data[2:4]))
	if !typ.Valid() {
		return fmt.Errorf("%w: %d", ErrUnknownMessageType, uint16(typ))
	}
	h.ProtocolVersion = binary.LittleEndian.Uint16(data[0:2])
	h.Type = typ
	h.RequestID = binary.LittleEndian.Uint64(data[4:12])
	h.PayloadLen = binary.LittleEndian.Uint32(data[12:16])
	return nil
}

// Frame is one complete frame: header plus payload bytes.
type Frame struct {
	// Header is the decoded frame header.
	Header Header
	// Payload is the JSON-encoded message body.
	Payload []byte
}

// WriteFrame writes one frame for a payload already encoded as
// JSON. A payload too large to frame (over u32 length, or over the
// default limit the reader enforces) is refused with
// ErrFrameTooLarge rather than truncated into a corrupt header.
func WriteFrame(w io.Writer, typ MsgType, requestID uint64, payload []byte) error {
	size := uint64(len(payload)) // a slice length is never negative
	if size > uint64(DefaultMaxFrameLen) {
		return fmt.Errorf("%w: %d bytes over the %d byte limit",
			ErrFrameTooLarge, size, DefaultMaxFrameLen)
	}
	header := Header{
		ProtocolVersion: ProtocolVersion,
		Type:            typ,
		RequestID:       requestID,
		PayloadLen:      uint32(size), // bounded by the check above
	}
	raw := header.MarshalBinary()
	if _, err := w.Write(raw[:]); err != nil {
		return fmt.Errorf("ipc: write header: %w", err)
	}
	if _, err := w.Write(payload); err != nil {
		return fmt.Errorf("ipc: write payload: %w", err)
	}
	return nil
}

// ReadFrame reads one frame. A header announcing a payload over
// maxPayload is the named ErrFrameTooLarge refusal, raised before
// the payload is read; the connection is unusable after any framing
// error and both sides treat it as closed. A maxPayload of 0 means
// DefaultMaxFrameLen.
func ReadFrame(r io.Reader, maxPayload uint32) (Frame, error) {
	if maxPayload == 0 {
		maxPayload = DefaultMaxFrameLen
	}
	var raw [HeaderSize]byte
	if _, err := io.ReadFull(r, raw[:]); err != nil {
		return Frame{}, fmt.Errorf("ipc: read header: %w", err)
	}
	var header Header
	if err := header.UnmarshalBinary(raw[:]); err != nil {
		return Frame{}, err
	}
	if header.PayloadLen > maxPayload {
		return Frame{}, fmt.Errorf("%w: %d bytes over the %d byte limit",
			ErrFrameTooLarge, header.PayloadLen, maxPayload)
	}
	payload := make([]byte, header.PayloadLen)
	if _, err := io.ReadFull(r, payload); err != nil {
		return Frame{}, fmt.Errorf("ipc: read payload: %w", err)
	}
	return Frame{Header: header, Payload: payload}, nil
}
