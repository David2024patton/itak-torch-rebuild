// Package ipc is the Go side of the Go/Rust seam defined in
// ARCHITECTURE.md Section 4: the frame header with its protocol
// version, the control-plane message set as types, and the named
// version refusal. It is the documented pkg/ipc grid slot, placed
// under internal/ with the rest of the platform until the itak-core
// module (TODO T-1.1) exists to publish it from.
//
// The Rust counterpart is the itak-types crate; both sides compile
// against the same shapes, the same 16-byte little-endian header,
// and the same JSON payload encoding, and CONTRACT.md in the crate
// folder is the shared description. The shared contracts themselves
// (InferenceRequest and friends) live in pkg/types and are not
// duplicated here; this package carries only the IPC-facing types.
package ipc

import (
	"errors"
	"fmt"
)

// ProtocolVersion is the seam protocol version this side speaks,
// carried in every frame header and stated in Hello. The major
// version lives in the high byte, the minor in the low byte:
// 0x0100 is protocol 1.0.
const ProtocolVersion uint16 = 0x0100

// CodeProtocolVersionMismatch is the stable diagnostic code naming
// the version refusal, identical on both sides of the seam.
const CodeProtocolVersionMismatch = "protocol_version_mismatch"

// CodeFrameTooLarge names a frame that announced a payload past the
// configured limit.
const CodeFrameTooLarge = "frame_too_large"

// CodeUnknownMessageType names a frame carrying a message type code
// this build does not know.
const CodeUnknownMessageType = "unknown_message_type"

// Named seam errors. Callers switch on these with errors.Is; the
// version refusal above all is a named refusal, never a hang and
// never a silent downgrade (ARCHITECTURE.md Section 4).
var (
	// ErrProtocolVersionMismatch means the peer speaks a different
	// major protocol version.
	ErrProtocolVersionMismatch = errors.New("ipc: protocol version mismatch")
	// ErrFrameTooLarge means a frame announced a payload past the
	// configured maximum.
	ErrFrameTooLarge = errors.New("ipc: frame too large")
	// ErrUnknownMessageType means a frame carried an unknown
	// message type code.
	ErrUnknownMessageType = errors.New("ipc: unknown message type")
	// ErrMalformedPayload means a payload did not decode as its
	// message type requires.
	ErrMalformedPayload = errors.New("ipc: malformed payload")
)

// ProtocolMajor extracts the major version (the high byte).
func ProtocolMajor(version uint16) uint16 {
	return version >> 8
}

// ProtocolMinor extracts the minor version (the low byte).
func ProtocolMinor(version uint16) uint16 {
	return version & 0x00ff
}

// CheckProtocolVersion is the version rule of the seam: majors must
// match, minors may differ. A mismatch returns an error wrapping
// ErrProtocolVersionMismatch naming both versions, the same named
// refusal the Rust side produces.
func CheckProtocolVersion(peer uint16) error {
	if ProtocolMajor(peer) != ProtocolMajor(ProtocolVersion) {
		return fmt.Errorf("%w: local %d.%d, peer %d.%d",
			ErrProtocolVersionMismatch,
			ProtocolMajor(ProtocolVersion), ProtocolMinor(ProtocolVersion),
			ProtocolMajor(peer), ProtocolMinor(peer))
	}
	return nil
}
