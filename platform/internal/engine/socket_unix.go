//go:build unix

package engine

import (
	"fmt"
	"os"
	"syscall"
)

// checkSocketTrust authenticates the engine end of the socket before
// anything is dialed, mirroring the two checks the Rust engine
// enforces on its side of the same socket (crates/itak-engine/src/
// main.rs: bind_socket pins the file to 0600, and check_peer refuses
// a peer whose kernel-reported uid differs). The handshake alone
// cannot authenticate the peer: the protocol shapes are public, so
// any process that binds the predictable socket path during a
// restart window answers hello_ack and passes. The kernel-checked
// facts about the socket FILE are what a squatter cannot fake:
//
//   - the file must be a socket;
//   - its owner uid must equal this process's uid (the engine runs
//     as the same user in every supported deployment, exactly the
//     engine side's peer rule);
//   - its permissions must be exactly owner-only (0600), the mode
//     the engine pins after binding.
//
// Any mismatch refuses by name with ErrUntrustedSocket before the
// dial, so no prompt byte ever reaches an unverified peer. The
// residual race (the file swapped between this stat and the dial)
// is backstopped from the other direction: only a same-uid process
// can create the replacement (the directory entry and the inode are
// both uid-gated), and the real engine refuses other-uid peers
// outright, so a same-uid impostor is inside the trust boundary the
// engine itself already accepts.
func checkSocketTrust(socketPath string, ownUID int) error {
	fi, err := os.Stat(socketPath)
	if err != nil {
		return fmt.Errorf("%w: stat %s: %w", ErrUnavailable, socketPath, err)
	}
	uid, ok := socketFileUID(fi)
	if !ok {
		return fmt.Errorf("%w: %s: cannot read the socket file's owner", ErrUntrustedSocket, socketPath)
	}
	if !socketTrustOK(fi.Mode(), uid, ownUID) {
		return fmt.Errorf("%w: %s: owner uid %d (this process runs as uid %d), mode %04o; want the engine's own socket, uid %d, mode 0600",
			ErrUntrustedSocket, socketPath, uid, ownUID, fi.Mode().Perm(), ownUID)
	}
	return nil
}

// socketTrustOK is the pure trust predicate, split out so the uid
// arm can be pinned by unit tests: a test process runs as one user
// and cannot fabricate a socket file owned by another uid.
func socketTrustOK(mode os.FileMode, fileUID, ownUID int) bool {
	return mode&os.ModeSocket != 0 && fileUID == ownUID && mode.Perm() == 0o600
}

// socketFileUID extracts the owner's uid from stat output. On every
// unix platform the platform builds for, FileInfo.Sys() is
// *syscall.Stat_t carrying the Uid field.
func socketFileUID(fi os.FileInfo) (int, bool) {
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, false
	}
	return int(st.Uid), true
}
