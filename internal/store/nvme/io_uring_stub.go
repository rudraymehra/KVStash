//go:build linux && kvb_uring

package nvme

import "errors"

// NewUringBackend — DEFERRED per the recorded A3 decision: giouring (the
// SPEC-1 pick) is incompatible with Go 1.26's linkname rules and unmaintained
// since 2023; the threadpool engine measured 98%+ of the device fio ceiling,
// so an io_uring engine (raw syscalls or a maintained binding) is the v1.1
// spike. This stub keeps the build tag and the IOBackend seam compile-checked
// so the engine swap stays a one-file change.
func NewUringBackend() (IOBackend, error) {
	return nil, errors.New("nvme: io_uring backend deferred (A3 record): giouring incompatible with Go 1.26 linkname rules; threadpool is the measured default")
}
