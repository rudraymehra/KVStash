package main

import (
	"errors"
	"fmt"
	"testing"
)

// TestCleanupStackRunsInReverse: release order must be the exact reverse of
// acquire order (the spiller once outlived the store it serves because a
// defer landed on the wrong side of an explicit close).
func TestCleanupStackRunsInReverse(t *testing.T) {
	var c cleanupStack
	var got []int
	for i := 0; i < 4; i++ {
		c.push(func() { got = append(got, i) })
	}
	c.run()
	want := []int{3, 2, 1, 0}
	if len(got) != len(want) {
		t.Fatalf("ran %d steps, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("cleanup order %v, want %v", got, want)
		}
	}
	// A second run must be a no-op — steps execute exactly once.
	c.run()
	if len(got) != len(want) {
		t.Fatal("cleanup steps ran twice")
	}
}

// TestExitCodeDistinguishesDegradedShutdown: systemd/monitoring must be able
// to tell a degraded shutdown (resources abandoned to process exit) from
// both a clean exit and a startup failure.
func TestExitCodeDistinguishesDegradedShutdown(t *testing.T) {
	if got := exitCode(nil); got != 0 {
		t.Fatalf("clean exit code %d, want 0", got)
	}
	if got := exitCode(errors.New("bind: address in use")); got != 1 {
		t.Fatalf("startup failure exit code %d, want 1", got)
	}
	wrapped := fmt.Errorf("%w: data-plane drain timed out", errDegradedShutdown)
	if got := exitCode(wrapped); got != exitDegradedShutdown {
		t.Fatalf("degraded shutdown exit code %d, want %d", got, exitDegradedShutdown)
	}
}
