// Package nvme is the log-structured NVMe tier: append-only fixed-size
// segments of 4 KiB-aligned records, sealed with a footer (entry table +
// CRC32C trailer), indexed in DRAM, checkpointed periodically, and recovered
// after a crash by checkpoint load + footer scan + forward-scan of every
// unsealed segment (normally one; crash-mid-rotation can leave more),
// truncating each at its first torn record. Lose data, never serve
// corruption.
//
// Load-bearing rules:
//
//   - Sealed segments are write-once: after the seal fdatasync no write ever
//     targets that file again, so a torn tail is confined to the single
//     unsealed segment. Segment files are fallocated to full size at create
//     (create-time fsync) — size metadata never changes afterwards.
//   - Every record carries the block's xxh3_64; the reader verifies it (plus
//     header magic and nskey) before a byte is served. A mismatch is a
//     self-heal index removal, never a served block.
//   - DELETE is not crash-durable on this tier: a checkpoint/footer replay
//     may resurrect a deleted key. Cache-legal — the crash contract binds
//     only never-acked keys (must never EXISTS) and byte-identity of
//     whatever EXISTS.
//   - I/O goes through the IOBackend seam. The default is the measured
//     pinned-thread pread/pwrite engine (the recorded A3 decision); an
//     io_uring backend remains pluggable behind the kvb_uring build tag
//     (stub only — giouring is incompatible with Go 1.26 linkname rules).
//   - darwin is a correctness backend only: F_NOCACHE is not O_DIRECT and
//     darwin fsync does not flush the drive cache (F_FULLFSYNC is
//     deliberately not used). All formats are little-endian and portable;
//     every durability claim is Linux-measured.
//   - No unsafe in this package: read buffers are plain mmap allocations
//     from the aligned pool (the arena keeps the repo's only unsafe seam).
package nvme
