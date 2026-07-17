// Go port of OffsetAllocator (github.com/sebbbi/OffsetAllocator).
//
// Copyright (c) 2023 Sebastian Aaltonen
// Go port Copyright (c) 2026 Rudray Mehra
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

package dram

import "math/bits"

// The allocator hands out non-overlapping [offset, offset+size) uint32 ranges
// from a fixed capacity in O(1): free ranges are binned by a tiny 8-bit
// float-log size code (256 bins = 32 top × 8 leaf), a two-level bitmask scan
// finds the first bin that guarantees a fit, and freed neighbors coalesce via
// an offset-sorted doubly linked list.
//
// LOAD-BEARING INVARIANT (the classic port bug — do not "fix"): a size is
// rounded UP to a bin on Alloc but DOWN on free/insert. A node parked in bin B
// has size in [floatToUint(B), floatToUint(B+1)); rounding the request UP to
// B' guarantees every node in bins ≥ B' fits without a size re-check.
//
// Not goroutine-safe: the DRAM tier serializes access under its own mutex.

// Allocation units for the tier boundary (the allocator itself is
// unit-agnostic). With 4 KiB granules, uint32 offsets span 16 TiB and
// worst-case rounding waste on a 0.4 MB block is <1%.
const (
	AllocUnitShift = 12
	AllocUnit      = 1 << AllocUnitShift
)

// Constants mirroring the C++ original (SmallFloat + Allocator).
const (
	numTopBins        = 32
	binsPerLeaf       = 8
	topBinsIndexShift = 3
	leafBinsIndexMask = 0x7
	numLeafBins       = numTopBins * binsPerLeaf // 256

	mantissaBits  = 3
	mantissaValue = 1 << mantissaBits // 8
	mantissaMask  = mantissaValue - 1 // 7

	// unusedIndex is the C++ Node::unused sentinel (also "no bin found").
	unusedIndex = ^uint32(0)

	// defaultMaxAllocs mirrors the C++ constructor default. The node pool is
	// 2× because every split can park one extra free node.
	defaultMaxAllocs = 128 * 1024

	// Allocation.Meta packs (generation << metaSlotBits) | slot. The
	// generation is bumped every time a slot is handed out as a live
	// allocation, so a stale Free whose slot was recycled (the ABA the C++
	// original does not defend against) fails the generation check instead of
	// silently freeing a LIVE later allocation. 18 slot bits cover the
	// default pool (2×128Ki); 14 generation bits wrap after 16K reuses of one
	// slot — astronomically unlikely to collide with a still-held stale
	// handle, and kvbdebug asserts loudly long before it matters.
	metaSlotBits = 18
	metaSlotMask = (1 << metaSlotBits) - 1
	genMask      = (1 << (32 - metaSlotBits)) - 1
)

// uintToFloatRoundUp is C++ SmallFloat::uintToFloatRoundUp: the 8-bit
// float-log code (5-bit exponent, 3-bit mantissa, denormalized below 8),
// rounding UP so floatToUint(code) >= size. The '+' (not '|') lets a mantissa
// overflow carry into the exponent, exactly like the original.
func uintToFloatRoundUp(size uint32) uint32 {
	if size < mantissaValue {
		return size // denormalized: bin == size
	}
	highestSetBit := uint32(31 - bits.LeadingZeros32(size)) //nolint:gosec // G115: size>=8 here, so LeadingZeros32<=28
	mantissaStartBit := highestSetBit - mantissaBits
	exp := mantissaStartBit + 1
	mantissa := (size >> mantissaStartBit) & mantissaMask
	lowBitsMask := (uint32(1) << mantissaStartBit) - 1
	if size&lowBitsMask != 0 {
		mantissa++ // round up; may carry into exp via the '+'
	}
	return (exp << mantissaBits) + mantissa
}

// uintToFloatRoundDown is C++ SmallFloat::uintToFloatRoundDown — as above
// without the round-up step, so floatToUint(code) <= size.
func uintToFloatRoundDown(size uint32) uint32 {
	if size < mantissaValue {
		return size
	}
	highestSetBit := uint32(31 - bits.LeadingZeros32(size)) //nolint:gosec // G115: size>=8 here, so LeadingZeros32<=28
	mantissaStartBit := highestSetBit - mantissaBits
	exp := mantissaStartBit + 1
	mantissa := (size >> mantissaStartBit) & mantissaMask
	return (exp << mantissaBits) | mantissa
}

// floatToUint is C++ SmallFloat::floatToUint: the bin's smallest size.
func floatToUint(f uint32) uint32 {
	exp := f >> mantissaBits
	mantissa := f & mantissaMask
	if exp == 0 {
		return mantissa // denormalized
	}
	return (mantissa | mantissaValue) << (exp - 1)
}

// findLowestSetBitAfter mirrors the C++ helper: index of the lowest set bit
// at position >= start, or unusedIndex if none.
func findLowestSetBitAfter(mask uint32, start uint32) uint32 {
	maskAfter := mask &^ ((uint32(1) << start) - 1)
	if maskAfter == 0 {
		return unusedIndex
	}
	return uint32(bits.TrailingZeros32(maskAfter)) //nolint:gosec // G115: TrailingZeros32 of a nonzero mask is 0..31
}

// node is one contiguous range, either free (parked in a bin's doubly linked
// list) or live (handed out; Allocation.Meta is its pool index). The C++
// original packs `used` into dataSize's top bit; a separate bool costs ~4
// bytes/node and reads clearly — documented divergence.
type node struct {
	dataOffset   uint32
	dataSize     uint32
	binListPrev  uint32 // free-list links within the bin (unusedIndex = end)
	binListNext  uint32
	neighborPrev uint32 // offset-sorted neighbor links (coalescing)
	neighborNext uint32
	gen          uint32 // live-handout generation (see metaSlotBits)
	used         bool
}

// Allocation is a live range handed out by Alloc. Meta is opaque
// (generation-tagged pool slot); pass it back to Free EXACTLY ONCE, and
// before any later Alloc could recycle it — a stale Free is rejected by the
// generation check (no-op in release, panic under -tags kvbdebug).
type Allocation struct {
	Offset uint32
	Meta   uint32
}

// StorageReport summarizes free capacity. LargestFreeRegion is the bin FLOOR
// of the highest non-empty bin — a guaranteed-allocatable lower bound, not an
// exact maximum (original behavior).
type StorageReport struct {
	TotalFreeSpace    uint32
	LargestFreeRegion uint32
}

// Allocator is the O(1) bin allocator. Zero value is not usable; construct
// with NewAllocator. Not goroutine-safe.
type Allocator struct {
	size        uint32
	maxAllocs   uint32
	freeStorage uint32

	usedBinsTop uint32              // bit t set ⇒ usedBins[t] != 0
	usedBins    [numTopBins]uint8   // bit l set ⇒ binIndices[t<<3|l] != unusedIndex
	binIndices  [numLeafBins]uint32 // bin head node index

	nodes      []node
	freeNodes  []uint32 // stack of unused node indices
	freeOffset uint32   // number of entries live on the freeNodes stack
}

// NewAllocator returns an allocator over [0, size) with the default node-pool
// budget (~128Ki live allocations). All memory is allocated here; Alloc/Free
// never allocate.
func NewAllocator(size uint32) *Allocator {
	return newAllocator(size, defaultMaxAllocs)
}

// NewAllocatorMax is NewAllocator with an explicit live-allocation budget —
// the Day-5 tier sizes this from arenaUnits/minBlockUnits so the node pool
// can never be the binding constraint before capacity is. maxAllocs is capped
// so the 2× node pool still fits the Meta slot field.
func NewAllocatorMax(size, maxAllocs uint32) *Allocator {
	if maxAllocs*2 > 1<<metaSlotBits {
		maxAllocs = 1 << (metaSlotBits - 1)
	}
	return newAllocator(size, maxAllocs)
}

// MaxAllocs reports the effective live-allocation budget after the Meta
// slot-field clamp — the evictor's node-pool watermark denominator.
func (a *Allocator) MaxAllocs() uint32 { return a.maxAllocs }

// newAllocator lets tests shrink the node pool to force exhaustion.
func newAllocator(size, maxAllocs uint32) *Allocator {
	a := &Allocator{size: size, maxAllocs: maxAllocs}
	a.reset()
	return a
}

// reset mirrors the C++ ctor/reset: empty bins, full node stack, one free
// node spanning the whole capacity.
func (a *Allocator) reset() {
	a.freeStorage = 0
	a.usedBinsTop = 0
	for i := range a.usedBins {
		a.usedBins[i] = 0
	}
	for i := range a.binIndices {
		a.binIndices[i] = unusedIndex
	}
	// The pool needs headroom for split remainders: 2 nodes per allocation
	// mirrors the C++ sizing (maxAllocs nodes + freelist of the same length —
	// we allocate maxAllocs*2 total node slots to keep the same behavior of
	// "an alloc can always split" until maxAllocs live allocations exist).
	poolLen := a.maxAllocs * 2
	if a.nodes == nil {
		a.nodes = make([]node, poolLen)
		a.freeNodes = make([]uint32, poolLen)
	}
	for i := uint32(0); i < poolLen; i++ {
		a.nodes[i] = node{
			binListPrev:  unusedIndex,
			binListNext:  unusedIndex,
			neighborPrev: unusedIndex,
			neighborNext: unusedIndex,
		}
		// Stack pops from the top; push indices reversed so node 0 pops first
		// (cosmetic parity with the original's ordering).
		a.freeNodes[i] = poolLen - i - 1
	}
	a.freeOffset = poolLen
	if a.size > 0 {
		a.insertNodeIntoBin(a.size, 0)
	}
}

// popNode takes an unused node index off the stack. Caller checks capacity.
func (a *Allocator) popNode() uint32 {
	a.freeOffset--
	return a.freeNodes[a.freeOffset]
}

// pushNode returns a node index to the stack.
func (a *Allocator) pushNode(idx uint32) {
	a.freeNodes[a.freeOffset] = idx
	a.freeOffset++
}

// insertNodeIntoBin carves a FRESH node (no neighbors — only reset uses this)
// for a free range and parks it in its (round-DOWN) bin as the new list head.
func (a *Allocator) insertNodeIntoBin(size, dataOffset uint32) {
	nodeIndex := a.popNode()
	a.nodes[nodeIndex] = node{
		dataOffset:   dataOffset,
		dataSize:     size,
		binListPrev:  unusedIndex,
		binListNext:  unusedIndex,
		neighborPrev: unusedIndex,
		neighborNext: unusedIndex,
	}
	a.insertExistingNodeIntoBin(nodeIndex)
}

// insertExistingNodeIntoBin re-parks an existing node (keeping its neighbor
// links) after a coalesce or split shrank/grew it.
func (a *Allocator) insertExistingNodeIntoBin(nodeIndex uint32) {
	n := &a.nodes[nodeIndex]
	binIndex := uintToFloatRoundDown(n.dataSize)
	topBinIndex := binIndex >> topBinsIndexShift
	leafBinIndex := binIndex & leafBinsIndexMask

	if a.binIndices[binIndex] == unusedIndex {
		a.usedBins[topBinIndex] |= 1 << leafBinIndex
		a.usedBinsTop |= 1 << topBinIndex
	}
	head := a.binIndices[binIndex]
	n.binListPrev = unusedIndex
	n.binListNext = head
	n.used = false
	if head != unusedIndex {
		a.nodes[head].binListPrev = nodeIndex
	}
	a.binIndices[binIndex] = nodeIndex
	a.freeStorage += n.dataSize
}

// removeNodeFromBin unlinks a FREE node from its bin list and clears bitmask
// bits when the bin empties. Does not touch neighbor links or the node pool.
func (a *Allocator) removeNodeFromBin(nodeIndex uint32) {
	n := &a.nodes[nodeIndex]
	if n.binListPrev != unusedIndex {
		a.nodes[n.binListPrev].binListNext = n.binListNext
		if n.binListNext != unusedIndex {
			a.nodes[n.binListNext].binListPrev = n.binListPrev
		}
	} else {
		// Node is the bin head.
		binIndex := uintToFloatRoundDown(n.dataSize)
		a.binIndices[binIndex] = n.binListNext
		if n.binListNext != unusedIndex {
			a.nodes[n.binListNext].binListPrev = unusedIndex
		}
		if a.binIndices[binIndex] == unusedIndex {
			topBinIndex := binIndex >> topBinsIndexShift
			leafBinIndex := binIndex & leafBinsIndexMask
			a.usedBins[topBinIndex] &^= 1 << leafBinIndex
			if a.usedBins[topBinIndex] == 0 {
				a.usedBinsTop &^= 1 << topBinIndex
			}
		}
	}
	a.freeStorage -= n.dataSize
}

// Alloc returns a range of exactly n units, or ok=false when n is 0, no free
// range fits, or the node pool is exhausted. Never panics.
func (a *Allocator) Alloc(n uint32) (Allocation, bool) {
	if n == 0 {
		return Allocation{Meta: unusedIndex}, false
	}
	// A successful alloc may split, consuming one extra node.
	if a.freeOffset < 1 {
		return Allocation{Meta: unusedIndex}, false
	}

	// Round UP: every node in bins >= minBinIndex is guaranteed to fit n.
	minBinIndex := uintToFloatRoundUp(n)
	minTopBinIndex := minBinIndex >> topBinsIndexShift
	minLeafBinIndex := minBinIndex & leafBinsIndexMask

	topBinIndex := minTopBinIndex
	leafBinIndex := unusedIndex

	// Same top bin as the request: scan its leaves from the request's leaf.
	if a.usedBinsTop&(1<<topBinIndex) != 0 {
		leafBinIndex = findLowestSetBitAfter(uint32(a.usedBins[topBinIndex]), minLeafBinIndex)
	}
	// Otherwise take the lowest non-empty top bin above, any leaf.
	if leafBinIndex == unusedIndex {
		topBinIndex = findLowestSetBitAfter(a.usedBinsTop, minTopBinIndex+1)
		if topBinIndex == unusedIndex {
			return Allocation{Meta: unusedIndex}, false // out of space
		}
		leafBinIndex = uint32(bits.TrailingZeros32(uint32(a.usedBins[topBinIndex]))) //nolint:gosec // G115: nonzero uint8 mask → 0..7
	}
	binIndex := (topBinIndex << topBinsIndexShift) | leafBinIndex

	// Pop the bin head.
	nodeIndex := a.binIndices[binIndex]
	nd := &a.nodes[nodeIndex]
	totalSize := nd.dataSize
	a.binIndices[binIndex] = nd.binListNext
	if nd.binListNext != unusedIndex {
		a.nodes[nd.binListNext].binListPrev = unusedIndex
	}
	if a.binIndices[binIndex] == unusedIndex {
		a.usedBins[topBinIndex] &^= 1 << leafBinIndex
		if a.usedBins[topBinIndex] == 0 {
			a.usedBinsTop &^= 1 << topBinIndex
		}
	}
	a.freeStorage -= totalSize

	nd.dataSize = n
	nd.used = true
	nd.gen = (nd.gen + 1) & genMask // new live handout → stale Metas invalidated
	nd.binListPrev = unusedIndex
	nd.binListNext = unusedIndex

	// Split: park the remainder as a new free node immediately after us in
	// the offset-sorted neighbor list (C++ does the same).
	remainder := totalSize - n
	if remainder > 0 {
		newNodeIndex := a.popNode()
		a.nodes[newNodeIndex] = node{
			dataOffset:   nd.dataOffset + n,
			dataSize:     remainder,
			binListPrev:  unusedIndex,
			binListNext:  unusedIndex,
			neighborPrev: nodeIndex,
			neighborNext: nd.neighborNext,
			used:         false,
		}
		if nd.neighborNext != unusedIndex {
			a.nodes[nd.neighborNext].neighborPrev = newNodeIndex
		}
		nd.neighborNext = newNodeIndex
		a.insertExistingNodeIntoBin(newNodeIndex)
	}

	return Allocation{Offset: nd.dataOffset, Meta: (nd.gen << metaSlotBits) | nodeIndex}, true
}

// Free releases a live allocation, coalescing with free neighbors (prev
// first, then next — original order). An invalid or repeated Meta is a no-op
// in release builds and panics under -tags kvbdebug.
func (a *Allocator) Free(al Allocation) {
	if al.Meta == unusedIndex {
		assertf(false, "dram: Free of a failed-Alloc sentinel")
		return
	}
	idx := al.Meta & metaSlotMask
	gen := al.Meta >> metaSlotBits
	if idx >= uint32(len(a.nodes)) || !a.nodes[idx].used || a.nodes[idx].gen != gen { //nolint:gosec // G115: len(nodes) = maxAllocs*2 ≤ 2^18
		assertf(false, "dram: Free of invalid, stale, or already-freed allocation (meta=%#x)", al.Meta)
		return
	}
	nd := &a.nodes[idx]

	offset := nd.dataOffset
	size := nd.dataSize

	// Merge a free previous neighbor into our range.
	if nd.neighborPrev != unusedIndex && !a.nodes[nd.neighborPrev].used {
		prevIdx := nd.neighborPrev
		prev := &a.nodes[prevIdx]
		offset = prev.dataOffset
		size += prev.dataSize
		a.removeNodeFromBin(prevIdx)
		// Splice prev out of the neighbor list.
		nd.neighborPrev = prev.neighborPrev
		if prev.neighborPrev != unusedIndex {
			a.nodes[prev.neighborPrev].neighborNext = idx
		}
		a.pushNode(prevIdx)
	}
	// Merge a free next neighbor.
	if nd.neighborNext != unusedIndex && !a.nodes[nd.neighborNext].used {
		nextIdx := nd.neighborNext
		next := &a.nodes[nextIdx]
		size += next.dataSize
		a.removeNodeFromBin(nextIdx)
		nd.neighborNext = next.neighborNext
		if next.neighborNext != unusedIndex {
			a.nodes[next.neighborNext].neighborPrev = idx
		}
		a.pushNode(nextIdx)
	}

	nd.dataOffset = offset
	nd.dataSize = size
	a.insertExistingNodeIntoBin(idx)
}

// StorageReport returns total free space and the guaranteed-allocatable
// largest-region lower bound (highest non-empty bin's floor).
func (a *Allocator) StorageReport() StorageReport {
	// Mirror the C++ original: with the node pool exhausted no Alloc can
	// succeed, so the report must not advertise allocatable space.
	if a.freeOffset == 0 {
		return StorageReport{}
	}
	r := StorageReport{TotalFreeSpace: a.freeStorage}
	if a.usedBinsTop != 0 {
		topBinIndex := uint32(31 - bits.LeadingZeros32(a.usedBinsTop))          //nolint:gosec // G115: usedBinsTop != 0 here → 0..31
		leafBinIndex := uint32(7 - bits.LeadingZeros8(a.usedBins[topBinIndex])) //nolint:gosec // G115: leaf mask nonzero when its top bit is set → 0..7
		r.LargestFreeRegion = floatToUint((topBinIndex << topBinsIndexShift) | leafBinIndex)
	}
	return r
}
