//go:build unix

package engine

import (
	"net"
	"os"
	"path/filepath"
	"testing"
)

// socketTrustTestListener binds a socket owned by this process and
// pinned to 0600, the state the real engine leaves at its path.
func socketTrustTestListener(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "e.sock")
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	if err := os.Chmod(path, 0o600); err != nil {
		t.Fatalf("chmod socket: %v", err)
	}
	return path
}

// TestSocketTrustOK pins the trust predicate arm by arm. The uid
// arm is pinned here rather than end to end because a test process
// runs as one user and cannot fabricate a socket file owned by
// another uid; the mode and socket-type arms are additionally
// covered end to end through DialUnix in unix_seam_test.go.
func TestSocketTrustOK(t *testing.T) {
	const sock os.FileMode = os.ModeSocket
	cases := []struct {
		name            string
		mode            os.FileMode
		fileUID, ownUID int
		want            bool
	}{
		{"the engine's own socket", sock | 0o600, 1000, 1000, true},
		{"another uid's socket", sock | 0o600, 1001, 1000, false},
		{"own socket, group-readable", sock | 0o640, 1000, 1000, false},
		{"own socket, umask default", sock | 0o770, 1000, 1000, false},
		{"own socket, world-writable", sock | 0o602, 1000, 1000, false},
		{"regular file at 0600", 0o600, 1000, 1000, false},
		{"another uid, loose mode", sock | 0o777, 0, 1000, false},
	}
	for _, tc := range cases {
		if got := socketTrustOK(tc.mode, tc.fileUID, tc.ownUID); got != tc.want {
			t.Errorf("%s: socketTrustOK(mode=%v, fileUID=%d, ownUID=%d) = %v, want %v",
				tc.name, tc.mode, tc.fileUID, tc.ownUID, got, tc.want)
		}
	}
}

// TestCheckSocketTrustArms exercises the stat wrapper directly: a
// socket this process owns at 0600 passes under its own uid and
// refuses under any other uid (the direction a test CAN fabricate:
// claiming to be a different process than the file's owner).
func TestCheckSocketTrustArms(t *testing.T) {
	path := socketTrustTestListener(t)
	own := os.Getuid()
	if err := checkSocketTrust(path, own); err != nil {
		t.Fatalf("checkSocketTrust(own uid) = %v, want nil", err)
	}
	if err := checkSocketTrust(path, own+1); err == nil {
		t.Fatal("checkSocketTrust(other uid) = nil, want ErrUntrustedSocket")
	}
}
