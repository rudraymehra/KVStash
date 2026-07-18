package store

import (
	"sync/atomic"
	"testing"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/nvme"
)

// spyBackend counts every device READ and panics while armed — the tripwire
// proving BATCH_EXISTS (and Contains, and Stats) never touch NVMe. The
// panic fires on a reader-pool goroutine, so a violation crashes the test
// binary — loud by construction. Injected through VolumeParams.Backend, the
// seam that exists exactly for this.
type spyBackend struct {
	inner nvme.IOBackend
	reads atomic.Int64
	armed atomic.Bool
}

type spyFile struct {
	nvme.File
	b *spyBackend
}

func (sb *spyBackend) Open(path string, forWrite bool) (nvme.File, error) {
	f, err := sb.inner.Open(path, forWrite)
	if err != nil {
		return nil, err
	}
	return &spyFile{File: f, b: sb}, nil
}

func (sf *spyFile) ReadAt(p []byte, off int64) error {
	sf.b.reads.Add(1)
	if sf.b.armed.Load() {
		panic("BATCH_EXISTS touched the NVMe device")
	}
	return sf.File.ReadAt(p, off)
}

func TestExistsNeverTouchesNVMe(t *testing.T) {
	spy := &spyBackend{inner: nvme.DefaultBackend()}
	fx := newFixture(t, 64<<20, spy)
	fx.fill(t)
	if fx.t.DemoteNow() == 0 {
		t.Fatal("no demotion — nothing NVMe-resident to guard")
	}

	keys := make([][32]byte, fillN)
	for i := range keys {
		keys[i] = tk(i)
	}

	spy.armed.Store(true)
	// The guarded surface: EXISTS (both shapes), Contains, Stats. Any device
	// read panics — the test fails loudly, not statistically.
	if n, _ := fx.t.ExistsPrefix(1, keys, false); n != fillN {
		t.Fatalf("ExistsPrefix (no bitmap) = %d, want fillN", n)
	}
	n, perKey := fx.t.ExistsPrefix(1, keys, true)
	if n != fillN || len(perKey) != fillN {
		t.Fatalf("ExistsPrefix (bitmap) = %d/%d", n, len(perKey))
	}
	for i, st := range perKey {
		if st != protocol.StatusOK {
			t.Fatalf("key %d: %s", i, st)
		}
	}
	for i := 0; i < fillN; i++ {
		if !fx.t.Contains(1, keys[i]) {
			t.Fatalf("Contains(%d) false", i)
		}
	}
	_ = fx.t.Stats()
	readsDuringGuard := spy.reads.Load()
	spy.armed.Store(false)

	// Prove the spy is not vacuous: a real NVMe GET must pass through it and
	// bump the read counter — if it doesn't, the guard above tested nothing.
	var nvmeKey [32]byte
	found := false
	for i := 0; i < fillN; i++ {
		if !fx.t.d.Contains(1, keys[i]) {
			nvmeKey, found = keys[i], true
			break
		}
	}
	if !found {
		t.Fatal("no nvme-only block to validate the spy with")
	}
	before := spy.reads.Load()
	if before != readsDuringGuard {
		t.Fatalf("device reads happened during the guarded surface: %d", before-readsDuringGuard)
	}
	if _, _, rel, tier, st := fx.t.GetRefTier(1, nvmeKey); st != protocol.StatusOK || tier != "nvme" {
		t.Fatalf("GET through the spy: %s tier=%s", st, tier)
	} else {
		rel()
	}
	if spy.reads.Load() <= before {
		t.Fatal("spy saw no read on a served NVMe GET — the guard is vacuous")
	}
}
