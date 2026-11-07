//go:build !unix

package engine

import (
	"fmt"
	"os"
)

// checkSocketTrust on non-unix platforms cannot read the socket
// file's owner uid from stat output, so it enforces the arms it can:
// the path must be a socket with owner-only (0600) permissions.
// Supported serving targets are unix (see socket_unix.go for the
// full check); this fallback keeps the seam building everywhere the
// Go toolchain targets without silently skipping verification.
func checkSocketTrust(socketPath string, _ int) error {
	fi, err := os.Stat(socketPath)
	if err != nil {
		return fmt.Errorf("%w: stat %s: %w", ErrUnavailable, socketPath, err)
	}
	if fi.Mode()&os.ModeSocket == 0 || fi.Mode().Perm() != 0o600 {
		return fmt.Errorf("%w: %s: mode %04o; want a socket with mode 0600",
			ErrUntrustedSocket, socketPath, fi.Mode().Perm())
	}
	return nil
}
