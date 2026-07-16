package server

import (
	"testing"
	"time"
)

// TestSweepStreams pins the reaper pass with a fixed clock: an idle live
// stream is tombstoned (staging freed), a fresh one is untouched, and a
// tombstone is deleted only after the grace period.
func TestSweepStreams(t *testing.T) {
	s := &session{streams: make(map[uint64]*putStream)}
	base := time.Unix(1_000_000, 0)
	const timeout = 5 * time.Second

	s.streams[1] = &putStream{buf: make([]byte, 100), lastActive: base}
	s.streams[2] = &putStream{buf: make([]byte, 50), lastActive: base.Add(4 * time.Second)}
	s.stagedBytes = 150

	// t=+5s: stream 1 is exactly at the timeout boundary (not past) — kept.
	s.sweepStreams(base.Add(timeout), timeout)
	if s.streams[1].tombstoned {
		t.Fatal("stream tombstoned at exactly timeout (want strictly-greater)")
	}

	// t=+6s: stream 1 idle >timeout → tombstoned, its 100 staged bytes freed;
	// stream 2 (active at +4s) survives.
	s.sweepStreams(base.Add(6*time.Second), timeout)
	if !s.streams[1].tombstoned {
		t.Fatal("idle stream not tombstoned")
	}
	if s.streams[1].buf != nil {
		t.Fatal("tombstoned stream still holds staging")
	}
	if s.stagedBytes != 50 {
		t.Fatalf("stagedBytes = %d, want 50", s.stagedBytes)
	}
	if s.streams[2].tombstoned {
		t.Fatal("fresh stream tombstoned")
	}

	// Within the grace period the tombstone stays (a late COMMIT still gets
	// its deterministic ERR_STALE_STREAM).
	s.sweepStreams(base.Add(8*time.Second), timeout)
	if _, ok := s.streams[1]; !ok {
		t.Fatal("tombstone reclaimed inside grace period")
	}

	// Past the grace period the tombstone is reclaimed (map bounded).
	s.sweepStreams(base.Add(12*time.Second), timeout)
	if _, ok := s.streams[1]; ok {
		t.Fatal("tombstone not reclaimed after grace period")
	}
}
