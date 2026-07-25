//go:build linux && kvb_uring

package nvme

import "errors"

// NewUringBackend — DEFERRED, a recorded decision: giouring (the original
// pick) is incompatible with Go 1.26's linkname rules and unmaintained
// since 2023; the threadpool engine measured 98%+ of the device fio ceiling,
// so an io_uring engine (raw syscalls or a maintained binding) is a v1.1
// candidate. This stub keeps the build tag and the IOBackend seam
// compile-checked so the engine swap stays a one-file change.
func NewUringBackend() (IOBackend, error) {
	return nil, errors.New("nvme: io_uring backend deferred: giouring incompatible with Go 1.26 linkname rules; threadpool is the measured default")
}
