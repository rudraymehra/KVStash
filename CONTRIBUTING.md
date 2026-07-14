# Contributing to kvblockd

Pre-v0.1: external contributions are welcome as issues/discussions; PRs will make sense after v0.1.0 stabilizes the core.

## Developer Certificate of Origin (DCO)

All commits must be signed off (`git commit -s`), certifying you wrote the code or have the right to submit it under Apache-2.0 (https://developercertificate.org/).

## The Merge Rule (project-internal)

No line merges — AI-written or otherwise — until the maintainer can explain it back cold. Every substantive PR carries a 5–10 line description written *before* reading any generated explanation. Review runs a staged pipeline (style → correctness → systems → spec/security → adversarial breakers → evidence gate); disagreements are settled by a failing test, never by seniority.

## The 10 Go gotchas this codebase will punish

1. **Slice aliasing** — a subslice shares the backing array; writing through one corrupts the other. Pooled buffers make this lethal.
2. **Nil maps** — reads are fine, writes panic. Always `make` before write.
3. **Goroutine leaks** — every spawned goroutine needs a documented exit path; blocked-forever readers hold buffers forever.
4. **Loop-variable capture** — goroutines capturing the loop var see its final value (pre-1.22 semantics linger in examples online).
5. **defer in hot loops** — defers cost; never inside the per-frame path.
6. **Unbuffered-channel deadlocks** — send blocks until received; a dead peer freezes the sender.
7. **False sharing** — adjacent atomic counters on one cache line serialize cores; pad hot per-shard counters.
8. **`time.After` in loops** — allocates a timer per iteration that lives until it fires; use `time.NewTimer` + Reset.
9. **Interface nil-ness** — a typed nil inside an interface is not `== nil`; error returns bite here.
10. **Copying sync types** — copying a struct containing a `sync.Mutex` (or passing by value) silently forks the lock.

## Style

`gofmt`/`golangci-lint` clean (config in `.golangci.yml`); comments state constraints the code can't show, never narration. Package docs live in `doc.go`.
