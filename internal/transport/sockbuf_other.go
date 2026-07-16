//go:build !linux

package transport

import "net"

// effectiveBufSizes: the clamping tripwire is Linux-specific (production
// target); on the dev platforms we skip the readback rather than pretend.
func effectiveBufSizes(*net.TCPConn) (snd, rcv int, ok bool) {
	return 0, 0, false
}
