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

**Committed artifacts, stated exactly.** What is reproducible from this repo
today is the 50 GbE Chart-1 session: `bench/results/rig-t/*.jsonl` renders
via `bench/report/plot.py chart1` to 6.22–6.23 GB/s at 100% of that pair's
measured 6.225 GB/s iperf3 ceiling, verify ON. The **12.67 GB/s / ~102%
100 GbE figure is currently table-only** (bench/BENCHMARKS.md records the
session and its stream curve, but its per-run JSONL is not committed), and
the 6.37 GB/s transport-gate 50 GbE figure is table-only the same way. Until
that JSONL lands, the 100 GbE number's independent check is re-running the
published rig commands, not rendering a committed artifact.

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
the committed 50 GbE JSONL in `bench/results/rig-t/` stops reproducing
6.22–6.23 GB/s ≈ 100% of its recorded ceiling via
`bench/report/plot.py chart1`. Note the asymmetry honestly: the 100 GbE
half of this claim has no committed JSONL yet, so only the re-run — not an
artifact render — can currently falsify or confirm it.

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

## 3. TTFT, Qwen2.5-7B long context: 14.3×/22.6× @16k and 17.2×/25.1× @32k

**Claim.** Same harness, same gates, KV-lighter model (56 KiB/token GQA-4)
at its NATIVE 32k context (no rope scaling, no config overrides): warm
reload **321 ms vs 4,588 ms** pure recompute @16k (**14.3×**; 22.6× vs
serving-shape 7,271 ms) and **636 ms vs 10,923 ms** @32k (**17.2×**; 25.1×
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
passed the exact-count hit gate — 1024 blocks per rep @16k and 1999–2000
per rep @32k (per-prompt calibration lands within a token, so reps differ
by one block; the gate is exact against each rep's own measured count, never
a nominal target) — and every record is path-stamped `chunked-slab`.

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

## 6. fp8 KV arm — pre-registered disclosure checklist (no headline number yet)

**Status: PRE-REGISTERED.** No fp8 TTFT multiple ships until a measured
campaign run satisfies every statement below from committed JSONL. The
planning projections (warm ~350–450 ms @16k, tax ~1.2–1.3×) are sizing
estimates, never quotable claims. Submission is one command:
`FP8_CAMPAIGN=1 bench/rigs/hf-gpu/submit.sh` (two jobs: the two-phase
cold+warm run and the pure-recompute baseline, all under one
`kv_cache_dtype`).

**The five statements every published fp8 number must carry** — each names
its machine-checkable witness, greppable from the run's own artifacts:

1. **Static scales, worst case.** The run used vLLM's static fp8 scale path
   (scale=1.0 default, or checkpoint-provided scales) — the conservative
   configuration for the cited accuracy. `calculate_kv_scales` is REFUSED at
   connector boot: first-forward-pass-derived scales bake into the page
   bytes with no scale metadata in the blob, so two boots would produce
   byte-incompatible blobs under identical keys. *Witness:* the engine boots
   at all (`tests/test_config.py::test_calculate_kv_scales_refused_at_boot`
   is the executable form), plus the `kv_cache_dtype` stamp on every record.
2. **Accuracy cited, not remeasured.** fp8-e4m3 KV-cache model quality is
   vLLM's published accuracy campaign's result, cited as such; this repo
   proves BYTE and TOKEN fidelity of the store, not model quality. e4m3 is
   the default because that campaign backs it.
3. **Key isolation by fingerprint.** fp8 and bf16 blobs can never
   cross-serve: the resolved kv-cache dtype folds into the config
   fingerprint, alias-normalized (`fp8` == `fp8_e4m3`) so one semantic mints
   exactly one keyspace. *Witness:*
   `tests/test_config.py::test_fp8_alias_normalized_before_fingerprinting`.
4. **Same-dtype arms, fresh stores.** All three chart arms — baseline (no
   connector), cold (connector, store-on-miss), warm (kvblockd reload) — ran
   the same `kv_cache_dtype`, each job against its own fresh daemon, so the
   multiple measures the store and never the quantizer. *Witness:* the
   `kv_cache_dtype` stamp on every JSONL record; `bench/report/plot.py` and
   `bench/report/aggregate.py` REFUSE mixed-dtype inputs outright.
5. **Machine-checked token identity, fp8-vs-fp8.** `bench/e2e/equivalence.py`
   runs around the same engine restart (greedy decode, batch=1
   `--max-num-seqs 1`, hard fail before anything is measured); every record
   carries `equivalence_scope: "fp8-vs-fp8 (...)"`, and the compare phase
   REFUSES a state/engine dtype mismatch — "fp8 outputs match bf16
   recompute" is a claim the harness cannot emit. *Witness:* the
   `token_identity` stamp in the TTFT JSONL and the run's EQUIVJSONL
   records.

**Load-bearing phrasing rule.** The fp8 multiple is NEVER phrased "vs bf16
recompute". It is fp8-warm vs fp8-recompute — a store claim inside vLLM's
own fp8 KV mode, with that mode's (cited) accuracy trade owned by the
engine, not hidden inside our speedup.

**This section is violated if** an fp8 number appears anywhere without all
five witnesses in its committed artifacts, or phrased against a bf16 arm —
that number comes down before any fix goes up.

---

## 7. Codec quality gate — pre-registered (no codec has landed)

**Status: PRE-REGISTERED, registered before any codec measurement exists.**
The blob prefix carries a codec field for forward compatibility, but only
`codec=raw` (bit-exact bytes) is valid today: the connector REFUSES every
non-raw `kvblockd_codec` at boot and cites this section (*witness:*
`tests/test_config.py::test_codec_knob_defaults_raw_and_refuses_everything_else`).
No lossy codec serde merges — let alone publishes a number — until it passes
every condition below, all fixed in advance so the gate cannot drift toward
whatever the codec happens to score:

1. **Fixed prompt set.** A committed, checksummed prompt file
   (RULER/needle-in-a-haystack style retrieval prompts) at context lengths
   16384 and 32768, run on the existing two-phase rig
   (`bench/rigs/hf-gpu/`). The set is frozen when the harness lands —
   changing it after a codec has been measured voids the gate.
2. **Deterministic exact-match scoring, NO LLM judge.** Greedy decode,
   batch=1 (`--max-num-seqs 1`), pinned seeds; a prompt scores 1 iff the
   generated answer string exactly matches the expected needle, else 0.
   Nothing subjective, nothing model-graded, nothing a rerun can move.
3. **Pre-registered pass threshold.** At EACH length (16k and 32k
   separately): the codec arm's exact-match score must be within
   **2 percentage points absolute** of the raw arm's score from the same
   run, same prompt set, same engine config. Registered now, before any
   codec bytes exist; if a future codec needs a looser bar, that is a FAIL,
   not a renegotiation.
4. **Lossy arms publish as separately-labeled rows.** A codec arm never
   merges into, averages with, or silently replaces a raw row. It appears as
   its own row, labeled lossy (e.g. `codec=fp8-cast (lossy)`), citing its
   measured quality delta from this gate in the same table row as its speed
   number.
5. **No codec on top of engine fp8.** `kvblockd_codec != raw` combined with
   `kv_cache_dtype=fp8*` is refused at boot — double quantization is
   unvalidated anywhere (*witness:* the stacking refusal in the same
   test).

**This section is violated if** a lossy-codec number ships without a passing
gate run in its committed artifacts, without the lossy label and quality
delta on its own row, or against a threshold that changed after the first
codec measurement.

---

*Changing any number above requires updating its section in the same commit
— a claim without its conditions and falsification line does not ship.*
