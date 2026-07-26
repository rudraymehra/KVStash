# Claims ledger — every headline number, its exact conditions, and what would falsify it

One section per claim we put in front of strangers. Each states the rig and
the exact commands, defines its terms (cold/warm mean different things in
different mouths), lists what is disclosed alongside the number, and ends
with a **falsification line** — the concrete observation that would make the
claim false. If you can produce that observation, file an issue; the claim
comes down before the fix goes up.

Raw artifacts for everything here are committed under `bench/results/`
(methodology rule 12: charts regenerate from JSONL alone). The honesty rules
themselves: [bench/METHODOLOGY.md](../bench/METHODOLOGY.md).

---

## 1. Wire path: ~102% of the iperf3 ceiling on 100 GbE, verify ON

**Claim.** kvblockd serves batched GETs at **12.67 GB/s (101.4 Gbit/s) ≈
102% of the same-pair iperf3 ceiling** on a real 100 GbE network with
end-to-end xxh3 verification on; on 50 GbE it saturates the NIC the same way
(6.37 GB/s ≈ 102% of the 49.8 Gbit/s ceiling).

**Rig + commands.** 2× c7gn.8xlarge (100 GbE) and 2× c6in.8xlarge (50 GbE),
us-east-1, cluster placement group, private IPs. Ceiling measured FIRST with
iperf3 on the same pair, then the store measured over the same link:
`bench/rigs/aws-transport/provision.sh && bench/rigs/aws-transport/run-chart1.sh`
(then `teardown.sh`); server `cmd/kvblockd`, load generator
`bench/kvbench` (GET-only, batch 32, closed-loop, warmed, best stream count,
median of 3). Numbers and stream curves: [bench/BENCHMARKS.md](../bench/BENCHMARKS.md).

**Definitions.** "GB/s" is decimal, payload-only goodput (headers and
protocol framing excluded). "% of ceiling" divides by the measured iperf3
number on the same pair minutes earlier, never a datasheet figure.

**Disclosed.** >100% is possible because the multi-stream store workload can
edge out a single iperf3 configuration on the same link; both numbers are
published and the ratio is quoted as "~102%", i.e. "saturates the NIC", not
"beats TCP". Verify ON == OFF on a real network (the ~12% loopback verify
cost was a loopback artifact — also published). Laptop/loopback numbers are
never quoted as this claim.

**This claim is false if** re-running the pair (same instance types, same
placement, iperf3 first) shows kvblockd GET goodput below ~95% of that
pair's iperf3 ceiling with verify ON at the published stream counts, or if
the committed JSONL in `bench/results/rig-t/` does not reproduce the chart
via `bench/report/plot.py chart1`.

---

## 2. TTFT, Llama-3.1-8B @16k: 9.1× vs pure recompute (16.2× vs serving-shape recompute)

**Claim.** Reloading a 16k-token prefix's KV from kvblockd into a FRESH vLLM
engine takes **552 ms** vs **5,045 ms** to recompute it with no connector
installed (9.1×), and vs **8,925 ms** for the connector-on cold arm that
also pays the synchronous store-on-miss write (16.2×). Full sweep
(1k/4k/8k/16k): 3.5–9.1× vs pure, 6.6–16.2× vs serving.

**Rig + commands.** NVIDIA A10G (HF Jobs `a10g-large`), vLLM 0.25.1, native
connector `vllm_kvblockd` 0.1.0, bf16, loopback:
```
MODEL=meta-llama/Llama-3.1-8B-Instruct bench/rigs/hf-gpu/submit.sh      # two-phase run
BASELINE_ONLY=1 MODEL=meta-llama/Llama-3.1-8B-Instruct bench/rigs/hf-gpu/submit.sh
python3 bench/report/plot.py chart2 --in bench/results/rig-e/chart2-ttft-run5.jsonl \
    bench/results/rig-e/chart2-ttft-baseline.jsonl --out chart2.png
```
Raw JSONL: `bench/results/rig-e/chart2-ttft-{run5,baseline}.jsonl`.

**Definitions.** TTFT = clock start immediately before the streaming request
is sent, stop at the first SSE event carrying a completion token. **Warm** =
the exact populated prompt served by a vLLM that was RESTARTED after
populate, `--no-enable-prefix-caching` in both phases — the KV's only
possible source is kvblockd over TCP, and claim 5's counter gate proves it
per rep. **Cold (serving)** = fresh-nonce prompt on the connector-on engine:
full prefill plus the synchronous store-on-miss write. **Pure recompute** =
fresh-nonce prompt on a separate engine with NO connector configured.
p50 of n=5 reps per point, warmup discarded.

**Disclosed.** The link is loopback — network transfer time ≈ 0; a real NIC
adds wire time (~0.7 s for the 16k prefix's ~2 GB at 25 GbE — inside the 5 s
recompute budget, but that number must be measured, not asserted). The table
is a SINGLE run of 5 reps per point until the pre-registered n≥3 reruns land
(`bench/rigs/hf-gpu/submit-n.sh`); per-rep values are published in
`ttft_all_ms`. Quote the conservative "vs pure" column unless the serving
context is explicit — the 16.2× column includes the miss's store cost.

**This claim is false if** rendering the committed JSONL yields different
medians; if a warm rep's `kvb_hits_total` delta did not equal its expected
block count (the record would say `warm_hits_verified: false` and the run
exits nonzero); or if a same-config rerun's warm p50 at any length lands
outside the 10% cross-run spread gate (`bench/report/aggregate.py`) without
that being disclosed next to the number.

---

## 3. TTFT, Qwen2.5-7B long context: 14.3×/22.6× @16k and 17.2×/25.2× @32k

**Claim.** Same harness, same gates, KV-lighter model (56 KiB/token GQA-4)
at its NATIVE 32k context (no rope scaling, no config overrides): warm
reload **321 ms vs 4,588 ms** pure recompute @16k (**14.3×**; 22.6× vs
serving-shape 7,271 ms) and **636 ms vs 10,923 ms** @32k (**17.2×**; 25.2×
vs 15,998 ms).

**Rig + commands.** Identical to claim 2 with
`MODEL=Qwen/Qwen2.5-7B-Instruct LENGTHS=16384,32000`. Raw JSONL:
`bench/results/rig-e/chart2-ttft-qwen32k{,-baseline}.jsonl`; chart
`bench/results/rig-e/chart2-qwen32k.png`.

**Definitions.** As claim 2. Why the multiple grows with context: speedup =
`prefill(L) / reload(L)`; prefill grows superlinearly, reload linearly with
KV bytes, and this model carries less than half the KV per token of
Llama-3.1-8B. That is the same physics every vendor's 100k-context headline
runs on — stated with the formula, not hidden behind it.

**Disclosed.** Loopback and single-run, exactly as claim 2. Every warm rep
passed the exact-count hit gate (1024/1024 and 2000/2000 blocks) and every
record is path-stamped `chunked-slab`.

**This claim is false if** any condition in claim 2's falsification line
holds for these files, or if the model config used differs from the stock HF
config (the JSONL would carry an `hf_overrides` stamp — these records have
none).

---

## 4. Durability: kill -9 torture, 0 corrupt / 0 phantom over 18,160 acked commits

**Claim.** SIGKILL a live daemon mid-write-storm, 100 loops on Linux: every
acknowledged commit either survives recovery byte-identical or is honestly
reported gone — **0 corrupt, 0 phantom over 18,160 journaled acks**.

**Rig + commands.** Linux (CI class hardware; the contract is functional,
not a throughput claim):
```
go run -tags crashtest ./test/crash -loops 100
```
Harness and crash contract: `test/crash/`, [docs/DESIGN.md](DESIGN.md).

**Definitions.** **Corrupt** = a post-recovery GET returns bytes that fail
the stored checksum or differ from what was acked. **Phantom** = recovery
resurrects a key whose commit was never acknowledged. Acks are counted from
the client's side — only writes the client was TOLD succeeded are held
against the store.

**Disclosed.** This is a single-node crash contract: an un-acked in-flight
write is allowed to vanish (a cache of recomputable data loses latency, not
data). It says nothing about disk firmware lying about flushes beyond what
fsync guarantees.

**This claim is false if** any loop of the published command on Linux
reports a corrupt or phantom entry — one is enough; the harness exits
nonzero and prints the offending key.

---

## 5. Hit attribution: warm numbers are counter-verified, block-exact, per rep

**Claim.** A published warm TTFT number cannot have come from anywhere but
kvblockd: for every measured warm rep, kvblockd's own `kvb_hits_total` must
grow by EXACTLY that rep's expected block count
(`((prompt_tokens-1)//block_size)*block_size // block_size` — the
connector's alignment rule), or the rep is recorded
`warm_hits_verified: false`, drawn red on the chart, and the job exits
nonzero. Isolation is structural: populate → vLLM restart → measure, with
`--no-enable-prefix-caching` in both phases, so no engine-local cache can
serve the warm arm.

**Rig + commands.** Any Chart-2 run (claims 2–3). The gates themselves are
proven for free against a controllable stub:
```
python3 bench/rigs/hf-gpu/run_ttft.py --selftest
```
which must show: populate FAILS LOUDLY on a severed store path; a no-hit
measure exits 3 with the warm records flagged; a partial hit delta (8 of 12
expected blocks) is UNVERIFIED with its own reason, never silently green.

**Definitions.** "Verified" = the delta equals the expectation exactly; both
a shortfall (some blocks recomputed) and an excess (someone else's reads
caught in the window) are distinct UNVERIFIED reasons in the JSONL.

**Disclosed.** An earlier run's 16× "speedup" with zero store hits was
rejected under exactly this rule and never published. The gate reads the
daemon's own metrics endpoint, not connector logs.

**This claim is false if** you can construct a run where a warm rep's TTFT
is published unflagged while `kvb_hits_total` grew by anything other than
the expected block count — the selftest above is the standing executable
check that the harness cannot do this.

---

*Changing any number above requires updating its section in the same commit
— a claim without its conditions and falsification line does not ship.*
