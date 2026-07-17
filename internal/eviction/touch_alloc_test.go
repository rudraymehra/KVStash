package eviction

import "testing"

// TestTouchZeroAllocs is the GET-hot-path gate: Policy.Touch must not
// allocate, for BOTH policies (known key, unknown key, unknown domain).
func TestTouchZeroAllocs(t *testing.T) {
	for _, tc := range []struct {
		name string
		p    Policy
	}{
		{"s3fifo", NewS3FIFO(0)},
		{"sampled-lru", NewSampledLRU()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			known := ek(7, 1)
			tc.p.Admit(known, 100, 0)
			cases := map[string]Key{
				"known":          known,
				"unknown-key":    ek(7, 99),
				"unknown-domain": ek(42, 1),
			}
			for name, k := range cases {
				if allocs := testing.AllocsPerRun(1000, func() {
					tc.p.Touch(k, 1)
				}); allocs != 0 {
					t.Errorf("%s: Touch allocated %.1f/op, want 0", name, allocs)
				}
			}
		})
	}
}
