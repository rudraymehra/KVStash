// Package dram is the DRAM tier: a GC-invisible anonymous mmap Arena
// (hugepage-backed on Linux, plain pages elsewhere) carved by an O(1)
// OffsetAllocator-style bin allocator, indexed by a 256-shard seeded map,
// governed by the Mooncake lease/pin/TTL ladder, and exposed through the
// server's Store surface plus the GetRef (zero-copy release) and lifecycle
// extensions.
//
// Load-bearing rules (A2 verdict; go-learning-track gotchas #3/#8):
//
//   - Arena bytes cross API boundaries ONLY as (offset, len) + refcount.
//     They are never stored in heap structures and never round-tripped
//     through uintptr — Arena.Bytes materializes a view on demand.
//   - The Allocator hands out abstract uint32 ranges. The tier converts to
//     byte offsets with AllocUnitShift (4 KiB granules), so uint32 offsets
//     span 16 TiB while BlockRef.Offset stays a uint32. The allocator itself
//     is unit-agnostic and NOT goroutine-safe; the tier serializes it.
//
// The unsafe fence: this package (and only this package) is authorized to use
// unsafe, and only in the arena seam (rules.go unsafeOutsideArena;
// .golangci.yml G103 carve-out). All unsafe lives in Arena.Bytes.
package dram
