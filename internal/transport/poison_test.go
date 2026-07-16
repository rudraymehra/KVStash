//go:build kvbdebug

package transport

import "testing"

// TestPoisonOnReturn (kvbdebug builds only): a retained alias of a Returned
// buffer reads 0xDE — the loud failure the debug contract promises.
func TestPoisonOnReturn(t *testing.T) {
	bs := HeapSource{}
	b := bs.Lend(64)
	for i := range b {
		b[i] = byte(i)
	}
	alias := b[10:20]
	bs.Return(b)
	for i, v := range alias {
		if v != 0xDE {
			t.Fatalf("byte %d of retained alias = %#02x, want poison 0xDE", i, v)
		}
	}
}
