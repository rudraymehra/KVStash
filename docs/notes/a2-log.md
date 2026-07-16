# A2 soak log — off-heap arena / GC-pause proof

**Kill-gate A2:** GC p99 pause < 5 ms with off-heap arenas under load. <br>
**Status:** Mac mechanism-proof done. The 24 h *verdict of record* runs on the standing Linux box (with huge pages + GOGC/GOMEMLIMIT tuning). These Mac numbers prove the mechanism, not the gate.

## Mechanism (why this works)
KV blobs live in a `unix.Mmap` anonymous region obtained straight from the kernel, bypassing the Go allocator — so they never appear in `HeapAlloc` and the GC never scans them. Only a tiny index (`[]blobRef`) is heap-resident. The serving path writes `header + arena-subslice` via `net.Buffers` (writev), so blob bytes are never copied onto the Go heap even when served.

## Run 1 — Apple M2, macOS, 1 GiB arena, 8 streams, 10 s
```
arena_bytes        1,073,741,824   (1 GiB)
rss_bytes          1,220,952,064   (~1.14 GiB)  → tracks the arena ✓
heap_alloc_bytes      63,615,680   (~61 MiB)    → TINY vs RSS: blobs are off-heap ✓
blobs                        782   (0.44/1/2.5 MiB band)
hugepages                  false   (none reserved on Mac; expected)
bytes_served      97,542,197,984   (~9.7 GB/s from arena via writev)
gc_pauses_observed           422
gc_pause_p50_ms          0.08192
gc_pause_p99_ms          0.65536   → A2 gate (p99 < 5 ms): PASS with margin ✓
gc_pause_p999_ms         5.24288   → tail outlier at the 5 ms line; note honestly
```
`GODEBUG=gctrace=1` confirmation: across 211 GC cycles the heap stayed ~40–100 MB (`101->103->32 MB` typical) while process RSS held ~1.2 GB — the 1 GiB arena is entirely outside the GC's view. GC CPU ~2%.

## Reading it
- **The claim holds:** 1 GiB of blobs, heap stays ~61 MB, p99 pause 0.66 ms — Go's GC is a non-issue for a large cache *when the cache lives off-heap*. This is the "why Go for a 100 GB cache?" answer, with our own numbers.
- **p999 = 5.24 ms** sits right at the gate line — driven by the deliberately aggressive `churn` goroutine on a laptop under `gctrace`. The cloud run (bigger arena, longer window, GOGC/GOMEMLIMIT tuned, no gctrace overhead) is where p99/p999 get their real verdict.
- Metric used: `/sched/pauses/total/gc:seconds` (current; `/gc/pauses:seconds` is deprecated). Percentiles are the conservative upper bucket bound (never under-report).

## Next
- Next: launch the 24 h soak on a standing Linux box (Hetzner/c7g) with `vm.nr_hugepages` set (expect `hugepages:true`), larger arena, GOGC/GOMEMLIMIT sweep → the A2 verdict of record in `docs/DESIGN.md`.

## Review-ladder outcome (2026-07-15, 7-agent full ladder + CTO gate)

Verdict: FIX-FIRST → all applied. **Core A2 mechanism verified HONEST** by the gate: off-heap is genuine (5× runs: heap ~60–110 MB while RSS tracks the arena), and the GC-pause math (correct `/sched/pauses/total/gc:seconds` metric, deep-copied counts, conservative upper-bucket-bound) can only false-FAIL, never false-PASS. The lies lived on the *reporting* side — now fixed:

- **[HIGH] empty/churn-only runs could emit `p99:0` → false PASS.** Percentiles are now `*float64` + `omitempty` behind a `valid` flag; a run with no window, no bytes served, or <100 pauses emits NO percentile and `valid:false` (proven: `--duration=0` → `valid:false`, note, no p99). Added explicit `gc_pause_max_ms` and a p999<1000-pauses caveat.
- **[MED] munmap vs in-flight arena readers (use-after-free).** Now `ln.Close(); srvWG.Wait()` before the deferred unmap.
- **[MED] teardown falsely claimed "$0 residue" with missing state + non-default region.** Now sweeps ALL regions for the tag when state is absent and refuses to assert clean on assumption.
- **[MED] rig would hang & idle-bill with no SSH ingress.** provision.sh now creates a tagged security group (SSH from this machine + intra-node), passed to run-instances, deleted in teardown.
- **[LOW] added first real unit tests** (`soak_test.go`): `TestServeArenaIntegrity` (served bytes are byte-identical to the arena — closes "off-heap AND correct") + `TestPctileConservative`; plus metric-Kind guard, counter-underflow guard, hugetlb-RSS note, doc fixes.

Corrected honest run (512 MiB, 8 streams, 8 s): heap 63 MB, RSS 672 MB, 460 pauses, **p99 0.92 ms (A2 PASS)**, max 4.19 ms.
