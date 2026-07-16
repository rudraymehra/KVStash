//go:build kvbdebug

package transport

// Poison overwrites b with 0xDE in debug builds (-tags kvbdebug), so any code
// that kept a view of a Returned buffer reads garbage immediately instead of
// silently reading recycled bytes. BufferSource implementations call it first
// thing in Return.
func Poison(b []byte) {
	for i := range b {
		b[i] = 0xDE
	}
}
