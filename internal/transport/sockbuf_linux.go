//go:build linux

package transport

import (
	"net"

	"golang.org/x/sys/unix"
)

// effectiveBufSizes reads back what the kernel actually granted for the
// send/receive buffers — the tripwire that turns "mystery 3 GB/s on an
// untuned host" into one log line pointing at net.core.{w,r}mem_max.
func effectiveBufSizes(tc *net.TCPConn) (snd, rcv int, ok bool) {
	raw, err := tc.SyscallConn()
	if err != nil {
		return 0, 0, false
	}
	var serr error
	err = raw.Control(func(fd uintptr) {
		snd, serr = unix.GetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_SNDBUF)
		if serr != nil {
			return
		}
		rcv, serr = unix.GetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_RCVBUF)
	})
	return snd, rcv, err == nil && serr == nil
}
