//go:build crashtest

// Command crash is the kill -9 torture harness (SPEC-3 §5): boot a real
// kvblockd child on a scratch dir with the NVMe tier enabled, storm it with
// PUTs (a deliberate fraction never COMMITted) and concurrent GETs, journal
// every COMMIT ack on the PARENT side strictly AFTER the ack arrives, then
// SIGKILL mid-storm, restart, and hold the recovered daemon to the crash
// contract (the cache posture, ruling 5):
//
//	(a) recovery-to-ready < 5s;
//	(b) every key the recovered daemon says EXISTS must GET byte-identical
//	    (regenerated content + xxh3 — the full client-side scrub);
//	(c) a key whose COMMIT was never acked must NEVER exist;
//	(d) the daemon serves fresh traffic after recovery.
//
// A journaled key that is GONE is legal (DRAM contents die with the
// process; only sealed/flushed NVMe records survive) — loss is honest,
// corruption never.
//
// Run: go run -tags crashtest ./test/crash -loops 10 -dir /tmp/kvbtort
package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/pkg/client"
)

const (
	nsName  = "tort"
	nsToken = "tort-token"
	workers = 16
)

func main() {
	loops := flag.Int("loops", 10, "kill -9 cycles")
	dir := flag.String("dir", "", "scratch dir (default: a temp dir)")
	bin := flag.String("bin", "", "kvblockd binary (default: go build ./cmd/kvblockd)")
	seed := flag.Int64("seed", 1, "content PRNG seed base")
	stormMS := flag.Int("storm-ms", 0, "fixed storm duration before the kill (0 = random 200–3000ms; the demo uses a long fixed storm)")
	flag.Parse()

	if err := run(*loops, *dir, *bin, *seed, *stormMS); err != nil {
		fmt.Fprintln(os.Stderr, "crash-torture: FAIL:", err)
		os.Exit(1)
	}
}

type journal struct {
	mu          sync.Mutex
	acked       map[[32]byte]uint64 // key → xxh3 (written strictly AFTER the COMMIT ack)
	uncommitted map[[32]byte]bool   // keys whose COMMIT was deliberately never sent
	seq         int
}

func run(loops int, dir, bin string, seed int64, stormMS int) error {
	start := time.Now()
	if dir == "" {
		d, err := os.MkdirTemp("", "kvbtort-*")
		if err != nil {
			return err
		}
		dir = d
		defer os.RemoveAll(dir)
	}
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return err
	}
	if bin == "" {
		bin = filepath.Join(dir, "kvblockd")
		build := exec.Command("go", "build", "-o", bin, "./cmd/kvblockd")
		build.Stderr = os.Stderr
		if err := build.Run(); err != nil {
			return fmt.Errorf("build kvblockd: %w", err)
		}
	}

	dataPort, metricsPort, err := freePorts()
	if err != nil {
		return err
	}
	cfgPath, err := writeConfigs(dir, dataPort, metricsPort)
	if err != nil {
		return err
	}

	j := &journal{acked: map[[32]byte]uint64{}, uncommitted: map[[32]byte]bool{}}
	rng := rand.New(rand.NewSource(seed)) //nolint:gosec // G404: deterministic torture schedule, not crypto

	for loop := 0; loop < loops; loop++ {
		child, err := spawn(bin, cfgPath)
		if err != nil {
			return fmt.Errorf("loop %d: spawn: %w", loop, err)
		}
		readyIn, err := waitReady(metricsPort, 10*time.Second)
		if err != nil {
			_ = child.Process.Kill()
			return fmt.Errorf("loop %d: daemon never ready: %w", loop, err)
		}
		if loop > 0 && readyIn > 5*time.Second {
			return fmt.Errorf("loop %d: recovery-to-ready %v exceeds the 5s gate", loop, readyIn)
		}

		// Post-restart verification BEFORE the new storm: the whole contract.
		if loop > 0 {
			if err := verify(dataPort, j, seed); err != nil {
				return fmt.Errorf("loop %d: %w", loop, err)
			}
		}

		// The storm: 16 committers (+ GET readers inside), plus raw
		// abandoned streams (~30% of the PUT attempts never COMMIT).
		stormCtx, stopStorm := context.WithCancel(context.Background())
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(w int) {
				defer wg.Done()
				storm(stormCtx, dataPort, j, seed, loop, w)
			}(w)
		}

		wait := time.Duration(200+rng.Intn(2800)) * time.Millisecond
		if stormMS > 0 {
			wait = time.Duration(stormMS) * time.Millisecond
		}
		time.Sleep(wait)
		if err := child.Process.Signal(syscall.SIGKILL); err != nil {
			return fmt.Errorf("loop %d: SIGKILL: %w", loop, err)
		}
		_ = child.Wait()
		stopStorm()
		wg.Wait()
		fmt.Printf("[torture] loop %d/%d: killed mid-storm (journal %d acked, %d uncommitted)\n",
			loop+1, loops, len(j.acked), len(j.uncommitted))
	}

	// Final restart + verdict.
	child, err := spawn(bin, cfgPath)
	if err != nil {
		return err
	}
	defer func() { _ = child.Process.Kill(); _, _ = child.Process.Wait() }()
	readyIn, err := waitReady(metricsPort, 10*time.Second)
	if err != nil {
		return fmt.Errorf("final restart never ready: %w", err)
	}
	if readyIn > 5*time.Second {
		return fmt.Errorf("final recovery-to-ready %v exceeds the 5s gate", readyIn)
	}
	if err := verify(dataPort, j, seed); err != nil {
		return err
	}
	fmt.Printf("[torture] PASS: %d kill -9 cycles, %d acked keys journaled, 0 corrupt reads, 0 phantom keys, %.1fs total\n",
		loops, len(j.acked), time.Since(start).Seconds())
	return nil
}

// storm runs one worker: committed PUTs journaled after ack, ~30% raw
// abandoned streams, and GETs of already-journaled keys. Every error is
// tolerated (the daemon dies mid-call by design) — correctness is judged
// only at verify time.
func storm(ctx context.Context, dataPort int, j *journal, seed int64, loop, w int) {
	addr := fmt.Sprintf("127.0.0.1:%d", dataPort)
	cl, err := client.Dial(ctx, addr, client.Options{
		Streams: 1, Namespace: nsName, Token: nsToken, DialTimeout: 2 * time.Second,
	})
	if err != nil {
		return
	}
	defer cl.Close()
	rng := rand.New(rand.NewSource(seed<<32 ^ int64(loop)<<8 ^ int64(w))) //nolint:gosec // G404: schedule, not crypto

	for i := 0; ctx.Err() == nil; i++ {
		j.mu.Lock()
		seq := j.seq
		j.seq++
		j.mu.Unlock()

		key, data := blockFor(seed, seq)
		if rng.Intn(100) < 30 {
			// Deliberately uncommitted: BEGIN + CHUNK, never COMMIT. The raw
			// conn stays open until the kill.
			if abandonStream(ctx, addr, key, data) {
				j.mu.Lock()
				j.uncommitted[key] = true
				j.mu.Unlock()
			}
			continue
		}
		if err := cl.Put(ctx, key, data); err != nil {
			return // daemon likely dead — the ack never arrived, so no journal entry
		}
		j.mu.Lock()
		j.acked[key] = xxh3.Hash(data) // strictly AFTER the ack
		j.mu.Unlock()

		if rng.Intn(4) == 0 { // concurrent GET pressure (also drives demotion hits)
			k := key
			if _, err := cl.BatchGet(ctx, [][32]byte{k}, make([][]byte, 1)); err != nil {
				return
			}
		}
	}
}

// abandonStream opens a raw connection, HELLOs, sends BEGIN + one CHUNK for
// the block and then goes silent — the §5 stream the reaper (or the kill)
// cleans up. Returns true when the BEGIN was accepted (the key is then a
// must-never-exist).
func abandonStream(ctx context.Context, addr string, key [32]byte, data []byte) bool {
	d := net.Dialer{Timeout: 2 * time.Second}
	nc, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return false
	}
	// Leaked deliberately: the socket dies with the parent loop iteration's
	// GC or the process; closing it early would let the reaper run sooner,
	// which is fine either way.
	hello := protocol.AppendHelloReq(nil, protocol.HelloReq{
		ProtoMin: protocol.Version1, ProtoMax: protocol.Version1,
		Token: []byte(nsToken), Namespace: nsName, ClientName: "tort-abandon",
	})
	if !rawFrame(nc, protocol.OpHello, 0, [32]byte{}, 1, hello) {
		_ = nc.Close()
		return false
	}
	if _, ok := rawRead(nc); !ok {
		_ = nc.Close()
		return false
	}
	begin := protocol.AppendPutBegin(nil, protocol.PutBeginBody{
		TotalLen: uint32(len(data)), //nolint:gosec // G115: block sizes ≤ 2.5 MiB
		XXH3Hint: xxh3.Hash(data),
	})
	if !rawFrame(nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutBegin), key, 2, begin) {
		_ = nc.Close()
		return false
	}
	body, ok := rawRead(nc)
	if !ok {
		_ = nc.Close()
		return false
	}
	if p, err := protocol.DecodePreamble(body); err != nil || p.Status != protocol.StatusOK {
		_ = nc.Close()
		return false // OK_EXISTS/quota refusals never staged anything
	}
	// Half the payload, then silence — the torn-stream shape.
	_ = rawFrame(nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutChunk), key, 2, data[:len(data)/2])
	return true
}

func rawFrame(nc net.Conn, op protocol.Opcode, flags uint16, key [32]byte, reqID uint64, body []byte) bool {
	h := protocol.Header{
		Opcode: op, Flags: flags, Key: key, RequestID: reqID,
		PayloadLen: uint32(len(body)), //nolint:gosec // G115: bodies ≤ 2.5 MiB here
	}
	var hdr [protocol.HeaderSize]byte
	h.MarshalTo(hdr[:])
	if _, err := nc.Write(hdr[:]); err != nil {
		return false
	}
	_, err := nc.Write(body)
	return err == nil
}

func rawRead(nc net.Conn) ([]byte, bool) {
	_ = nc.SetReadDeadline(time.Now().Add(5 * time.Second))
	var hdr [protocol.HeaderSize]byte
	if _, err := readFull(nc, hdr[:]); err != nil {
		return nil, false
	}
	h, err := protocol.ParseHeader(hdr[:], protocol.DefaultMaxFrameLen)
	if err != nil {
		return nil, false
	}
	body := make([]byte, h.PayloadLen)
	if _, err := readFull(nc, body); err != nil {
		return nil, false
	}
	return body, true
}

func readFull(nc net.Conn, p []byte) (int, error) {
	n := 0
	for n < len(p) {
		m, err := nc.Read(p[n:])
		n += m
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

// verify holds the recovered daemon to the contract: (b) EXISTS ⇒ GET
// byte-identical (full scrub), (c) uncommitted keys never exist, (d) fresh
// traffic round-trips.
func verify(dataPort int, j *journal, seed int64) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	cl, err := client.Dial(ctx, fmt.Sprintf("127.0.0.1:%d", dataPort), client.Options{
		Streams: 4, Namespace: nsName, Token: nsToken, DialTimeout: 2 * time.Second,
	})
	if err != nil {
		return fmt.Errorf("verify dial: %w", err)
	}
	defer cl.Close()

	j.mu.Lock()
	acked := make(map[[32]byte]uint64, len(j.acked))
	for k, v := range j.acked {
		acked[k] = v
	}
	uncommitted := make([][32]byte, 0, len(j.uncommitted))
	for k := range j.uncommitted {
		uncommitted = append(uncommitted, k)
	}
	j.mu.Unlock()

	// (c) no phantom keys — an unacked stream must never surface a block.
	for i := 0; i < len(uncommitted); i += 64 {
		end := min(i+64, len(uncommitted))
		batch := uncommitted[i:end]
		_, perKey, err := cl.BatchExists(ctx, batch)
		if err != nil {
			return fmt.Errorf("verify exists: %w", err)
		}
		for x, st := range perKey {
			if st == protocol.StatusOK {
				return fmt.Errorf("PHANTOM: uncommitted key %x EXISTS after recovery", batch[x][:8])
			}
		}
	}

	// (b) the scrub, EXISTS-first: every key the daemon ADMITS TO must GET
	// byte-identical. EXISTS=true followed by a persistent GET miss is a
	// PHANTOM (index/storage divergence — the exact failure class of the
	// ladder's checkpoint blocker, which the GET-only oracle passed
	// vacuously). A single EXISTS→GET disagreement is re-probed once:
	// legal eviction/reclaim can race the two calls, divergence cannot.
	survived, lost, phantomRaces := 0, 0, 0
	for key, sum := range acked {
		n, perKey, err := cl.BatchExists(ctx, [][32]byte{key})
		if err != nil {
			return fmt.Errorf("verify exists(acked): %w", err)
		}
		exists := n > 0 || (len(perKey) > 0 && perKey[0] == protocol.StatusOK)
		into := make([][]byte, 1)
		statuses, err := cl.BatchGet(ctx, [][32]byte{key}, into)
		if err != nil {
			return fmt.Errorf("verify get: %w", err)
		}
		st := statuses[0]
		if st == protocol.StatusErrBusy { // reader saturation: retry once
			if statuses, err = cl.BatchGet(ctx, [][32]byte{key}, into); err == nil {
				st = statuses[0]
			}
		}
		switch st {
		case protocol.StatusOK, protocol.StatusOKExists:
			// Regenerate the exact content from the key's sequence and compare.
			seq, ok := seqOf(key)
			if !ok {
				return fmt.Errorf("CORRUPT: served key %x is not ours", key[:8])
			}
			_, want := blockFor(seed, seq)
			if xxh3.Hash(into[0]) != sum || !bytes.Equal(into[0], want) {
				return fmt.Errorf("CORRUPT: key %x served %d bytes that do not match the acked content", key[:8], len(into[0]))
			}
			survived++
		case protocol.StatusNotFound:
			if exists {
				// EXISTS said yes, GET said no: re-probe — if it STILL
				// claims existence, the index has diverged from storage.
				n2, pk2, err2 := cl.BatchExists(ctx, [][32]byte{key})
				if err2 != nil {
					return fmt.Errorf("verify re-exists: %w", err2)
				}
				if n2 > 0 || (len(pk2) > 0 && pk2[0] == protocol.StatusOK) {
					return fmt.Errorf("PHANTOM: key %x EXISTS but cannot be served (index/storage divergence)", key[:8])
				}
				phantomRaces++ // evicted between the probes — legal
			}
			lost++ // honest loss — DRAM died with the process
		default:
			return fmt.Errorf("verify: key %x answered %s", key[:8], st)
		}
	}

	// (d) fresh traffic.
	fresh, freshData := blockFor(seed, 1<<30)
	if err := cl.Put(ctx, fresh, freshData); err != nil {
		return fmt.Errorf("fresh put after recovery: %w", err)
	}
	into := make([][]byte, 1)
	if sts, err := cl.BatchGet(ctx, [][32]byte{fresh}, into); err != nil || sts[0] != protocol.StatusOK || !bytes.Equal(into[0], freshData) {
		return fmt.Errorf("fresh get after recovery: %v %v", sts, err)
	}
	if _, err := cl.Delete(ctx, [][32]byte{fresh}, true); err != nil {
		return fmt.Errorf("fresh delete: %w", err)
	}

	fmt.Printf("[torture] verify: %d/%d acked keys survived byte-identical, %d honestly lost, 0 corrupt, 0 phantom (%d benign probe races)\n",
		survived, survived+lost, lost, phantomRaces)
	return nil
}

// blockFor generates key + deterministic content for a sequence number.
// Sizes cycle through the 0.4–2.5 MiB band.
func blockFor(seed int64, seq int) ([32]byte, []byte) {
	var key [32]byte
	copy(key[:8], "KVBTORT\x00")
	key[8] = byte(seq)
	key[9] = byte(seq >> 8)
	key[10] = byte(seq >> 16)
	key[11] = byte(seq >> 24)
	sizes := []int{400 << 10, 700 << 10, 1 << 20, 1600 << 10, 2560 << 10}
	n := sizes[seq%len(sizes)]
	data := make([]byte, n)
	rng := rand.New(rand.NewSource(seed ^ int64(seq)*0x9E3779B9)) //nolint:gosec // G404: content pattern, not crypto
	rng.Read(data)
	return key, data
}

// seqOf inverts blockFor's key layout.
func seqOf(key [32]byte) (int, bool) {
	if string(key[:8]) != "KVBTORT\x00" {
		return 0, false
	}
	return int(key[8]) | int(key[9])<<8 | int(key[10])<<16 | int(key[11])<<24, true
}

func spawn(bin, cfg string) (*exec.Cmd, error) {
	cmd := exec.Command(bin, "-config", cfg) //nolint:gosec // G204: our own freshly built binary
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

func waitReady(metricsPort int, timeout time.Duration) (time.Duration, error) {
	start := time.Now()
	url := fmt.Sprintf("http://127.0.0.1:%d/healthz", metricsPort)
	for time.Since(start) < timeout {
		resp, err := http.Get(url) //nolint:noctx,gosec // G107: local loopback poll
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return time.Since(start), nil
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	return timeout, fmt.Errorf("healthz not ready within %v", timeout)
}

func freePorts() (data, metrics int, err error) {
	grab := func() (int, error) {
		l, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			return 0, err
		}
		defer l.Close()
		return l.Addr().(*net.TCPAddr).Port, nil
	}
	if data, err = grab(); err != nil {
		return 0, 0, err
	}
	if metrics, err = grab(); err != nil {
		return 0, 0, err
	}
	return data, metrics, nil
}

// writeConfigs materializes the daemon + namespaces YAML for the scratch
// dir: a small arena and an aggressive demote watermark so the storm drives
// real demotion, small segments so seals/checkpoints/rotations all happen
// within one loop.
func writeConfigs(dir string, dataPort, metricsPort int) (string, error) {
	nsPath := filepath.Join(dir, "ns.yaml")
	if err := os.WriteFile(nsPath, []byte(
		"namespaces:\n  - { name: "+nsName+", id: 7, token: "+nsToken+" }\n",
	), 0o600); err != nil {
		return "", err
	}
	nvmeDir := filepath.Join(dir, "nvme")
	cfg := fmt.Sprintf(`listen_addr: "127.0.0.1:%d"
metrics_addr: "127.0.0.1:%d"
namespaces_path: %q
max_blob_len: 4194304          # 4 MiB — the torture band tops at 2.5 MiB
dram_arena_bytes: 67108864     # 64 MiB — small so the storm hits the demote watermark fast
pinned_bytes_cap: 16777216
nvme_paths: [%q]
nvme_max_bytes: 2147483648     # 2 GiB scratch budget
nvme_segment_bytes: 8388608    # 8 MiB segments — rotations/seals/ckpts every loop
nvme_read_workers: 8
nvme_demote_watermark_pct: 60  # demote early and often
nvme_demote_batch_pct: 10
nvme_sync_every_bytes: 1048576 # 1 MiB group commit — tight durability windows for the kill
nvme_ckpt_every_segments: 2
`, dataPort, metricsPort, nsPath, nvmeDir)
	cfgPath := filepath.Join(dir, "kvblockd.yaml")
	if err := os.WriteFile(cfgPath, []byte(cfg), 0o600); err != nil {
		return "", err
	}
	return cfgPath, nil
}
