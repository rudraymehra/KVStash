//go:build ruleguard

// Package gorules holds repo-local static rules (gocritic/ruleguard via
// golangci-lint). Each rule encodes a hard-won review lesson so it never
// again depends on reviewer stamina; every mechanically expressible SDE-3/SSE
// finding gets a rule in the same PR that fixes it.
package gorules

import "github.com/quasilyte/go-ruleguard/dsl"

// wireByteOrder: every integer on the KVB1 wire is little-endian (the magic is
// written via a LE-encoded constant). A BigEndian call in wire/transport code
// is a spec violation that round-trips cleanly against itself and only fails
// against a conforming peer — the worst kind of bug. The xferspike rig is the
// one deliberate exception (its 16B ancestor frame writes the magic BE) and
// gets a nolint there.
func wireByteOrder(m dsl.Matcher) {
	m.Match(`binary.BigEndian.$_($*_)`, `binary.BigEndian.$_`).
		Report(`KVB1 wire integers are little-endian (PROTOCOL.md §0); BigEndian in wire code round-trips locally but breaks interop`)
}

// unsafeOutsideArena: unsafe belongs to the arena/alignment code, where it is
// reviewed line-by-line (gosec G103 is globally excluded for that reason).
// Anywhere else it needs an explicit justification comment and a nolint.
func unsafeOutsideArena(m dsl.Matcher) {
	m.Match(`unsafe.Pointer($_)`, `unsafe.Sizeof($_)`, `unsafe.Offsetof($_)`, `unsafe.Alignof($_)`).
		Where(!m.File().PkgPath.Matches(`store/dram`) && !m.File().Name.Matches(`arena|align|target`)).
		Report(`unsafe outside the arena/alignment code: justify with a comment + nolint, or move it behind the arena seam`)
}
