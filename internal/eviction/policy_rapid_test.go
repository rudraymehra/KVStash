package eviction

import (
	"testing"

	"pgregory.net/rapid"
)

// TestPolicyRapidConsistency drives both policies through random op
// streams against a mirror map, pinning the bookkeeping contract:
//   - Usage totals equal the mirror's per-domain byte sums at every step.
//   - Every victim was mirror-resident and is dequeued exactly once.
//   - A removed key never resurfaces as a victim.
//   - The ghost ring never exceeds its capacity (S3-FIFO).
func TestPolicyRapidConsistency(t *testing.T) {
	for _, name := range []string{"s3fifo", "sampled-lru"} {
		t.Run(name, func(t *testing.T) {
			rapid.Check(t, func(rt *rapid.T) {
				p, err := New(name, 4096)
				if err != nil {
					rt.Fatal(err)
				}
				mirror := make(map[Key]int64)
				nsGen := rapid.SampledFrom([]uint32{1, 2, 3})
				byteGen := rapid.Byte()
				sizeGen := rapid.Int64Range(1, 1<<20)
				var now int64

				keyGen := func(rt *rapid.T) Key {
					return ek(nsGen.Draw(rt, "ns"), byteGen.Draw(rt, "kb"))
				}
				checkUsage := func() {
					want := make(map[uint32]int64)
					for k, sz := range mirror {
						want[k.NS] += sz
					}
					got := make(map[uint32]int64)
					for _, u := range p.Usage(nil) {
						got[u.NS] = u.Bytes
					}
					for ns, b := range want {
						if got[ns] != b {
							rt.Fatalf("usage ns%d = %d, mirror says %d", ns, got[ns], b)
						}
					}
					for ns, b := range got {
						if want[ns] != b {
							rt.Fatalf("usage reports ns%d=%d the mirror doesn't have", ns, b)
						}
					}
				}

				rt.Repeat(map[string]func(*rapid.T){
					"admit": func(rt *rapid.T) {
						k := keyGen(rt)
						sz := sizeGen.Draw(rt, "size")
						p.Admit(k, sz, now)
						if _, dup := mirror[k]; !dup {
							mirror[k] = sz
						} // double-admit: policy no-ops, mirror keeps the original
					},
					"touch": func(rt *rapid.T) {
						now++
						p.Touch(keyGen(rt), now)
					},
					"remove": func(rt *rapid.T) {
						k := keyGen(rt)
						p.Remove(k)
						delete(mirror, k)
					},
					"victims": func(rt *rapid.T) {
						ns := nsGen.Draw(rt, "vns")
						need := rapid.Int64Range(1, 4<<20).Draw(rt, "need")
						for _, c := range p.Victims(ns, need, now, nil) {
							if c.Key.NS != ns {
								rt.Fatalf("cross-tenant victim: %+v for ns%d", c, ns)
							}
							sz, ok := mirror[c.Key]
							if !ok {
								rt.Fatalf("victim %v not mirror-resident (double-dequeue or removed key)", c.Key)
							}
							if sz != c.Size {
								rt.Fatalf("victim size %d, mirror %d", c.Size, sz)
							}
							delete(mirror, c.Key)
						}
					},
					"": func(rt *rapid.T) {
						checkUsage()
						if s3, ok := p.(*S3FIFO); ok {
							for _, ns := range []uint32{1, 2, 3} {
								if size, capacity := s3.ghostStats(ns); capacity > 0 && size > capacity {
									rt.Fatalf("ghost ns%d: size %d > capacity %d", ns, size, capacity)
								}
							}
						}
					},
				})
			})
		})
	}
}
