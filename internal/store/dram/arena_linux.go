//go:build linux

package dram

import "golang.org/x/sys/unix"

// mmapRegion maps an anonymous private region. With huge, the length is
// rounded up to the 2 MiB hugepage size and MAP_HUGETLB is attempted first;
// ANY failure (no hugepage pool, ENOMEM, EINVAL) falls back to a normal
// mapping at the unrounded size — mirroring cmd/xferspike/arena_linux.go, the
// A2-proven shape.
func mmapRegion(bytes int64, huge bool) (region []byte, gotHuge bool, err error) {
	const prot = unix.PROT_READ | unix.PROT_WRITE
	if huge {
		const hugeSize = 2 << 20
		rounded := (bytes + hugeSize - 1) &^ (hugeSize - 1)
		region, err = unix.Mmap(-1, 0, int(rounded), prot,
			unix.MAP_PRIVATE|unix.MAP_ANONYMOUS|unix.MAP_HUGETLB)
		if err == nil {
			return region, true, nil
		}
	}
	region, err = unix.Mmap(-1, 0, int(bytes), prot,
		unix.MAP_PRIVATE|unix.MAP_ANONYMOUS)
	return region, false, err
}

// advisePopulate pre-faults the whole region in one syscall on kernels ≥5.14.
// (populated=false, err=nil) means the ADVICE is unsupported (EINVAL on
// kernels <5.14) and the caller should run the portable touch loop; any other
// error is a genuine population failure (e.g. ENOMEM) the caller must treat
// as fatal — falling back would just move the fault into the touch loop.
func advisePopulate(region []byte) (populated bool, err error) {
	switch e := unix.Madvise(region, unix.MADV_POPULATE_WRITE); e {
	case nil:
		return true, nil
	case unix.EINVAL, unix.ENOSYS:
		return false, nil // advice unsupported here — touch loop instead
	default:
		return false, e
	}
}
