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

## 2. TTFT, Llama-3.1-8B @16k: 10.2× vs pure recompute (10.6× vs serving-shape recompute)

**Claim.** Reloading a 16k-token prefix's KV from kvblockd into a FRESH vLLM
engine takes **497 ms** (median of n=3 independent runs) vs **5,045 ms** to
recompute it with no connector installed (10.2×), and vs **5,261 ms** for
the connector-on cold arm that also pays the store-on-miss write (10.6×).
Full sweep (1k/4k/8k/16k): 3.4–10.2× vs pure, 3.6–10.6× vs serving.
Supersedes the single-run 552 ms/9.1×/16.2× (2026-07-26) and the
gathered-slots-wave 576 ms/8.8× n=3 table — the serving-shape column shrank
because store-on-miss now costs 1.04–1.06× of pure recompute (the cold arm
got faster, not the reload slower); history stays in BENCHMARKS.md and git.

**Rig + commands.** NVIDIA A10G (HF Jobs `a10g-large`), vLLM 0.25.1, native
connector `vllm_kvblockd` 0.1.0 — pipelined-slab load path, stamped
`path=pipelined-slab` in every committed record's `connector` field; the
gathered-slots store path is attributed in each run's job log, not the
committed JSONL — bf16, loopback:
```
N=3 MODEL=meta-llama/Llama-3.1-8B-Instruct bench/rigs/hf-gpu/submit-n.sh # 3 independent two-phase runs
BASELINE_ONLY=1 MODEL=meta-llama/Llama-3.1-8B-Instruct bench/rigs/hf-gpu/submit.sh
python3 bench/report/plot.py chart2 \
    --in bench/results/rig-e/chart2-ttft-a10g-final-run1.jsonl \
         bench/results/rig-e/chart2-ttft-a10g-final-run2.jsonl \
         bench/results/rig-e/chart2-ttft-a10g-final-run3.jsonl \
         bench/results/rig-e/chart2-ttft-baseline.jsonl --out chart2.png
```
Raw JSONL: `bench/results/rig-e/chart2-ttft-a10g-final-run{1,2,3}.jsonl`
plus `chart2-ttft-baseline.jsonl`.

**Definitions.** TTFT = clock start immediately before the streaming request
is sent, stop at the first SSE event carrying a completion token. **Warm** =
the exact populated prompt served by a vLLM that was RESTARTED after
populate, `--no-enable-prefix-caching` in both phases — the KV's only
possible source is kvblockd over TCP, and claim 5's counter gate proves it
per rep. **Cold (serving)** = fresh-nonce prompt on the connector-on engine:
full prefill plus the store-on-miss write. **Pure recompute** = fresh-nonce
prompt on a separate engine with NO connector configured. p50 of n=5 reps
per point per run, median across n=3 runs, warmup discarded.

**Disclosed.** The link is loopback — network transfer time ≈ 0; a real NIC
adds wire time (~0.7 s for the 16k prefix's ~2 GB at 25 GbE — inside the 5 s
recompute budget, but that number must be measured, not asserted). The pure
baseline is n=1 (rep spread <1%; an n=3 baseline rides the next paid
session). Cross-run p50 spread is gated at ≤10% by
`bench/report/aggregate.py`; the worst cell in this table is 7.4% (16k
warm), per-run values and min/max published in the aggregate output. Quote
the conservative "vs pure" column unless the serving context is explicit.

**This claim is false if** rendering the committed JSONL yields different
medians; if a warm rep's `kvb_hits_total` delta did not equal its expected
block count (the record would say `warm_hits_verified: false` and the run
exits nonzero); or if a same-config rerun's warm p50 at any length lands
outside the 10% cross-run spread gate (`bench/report/aggregate.py`) without
that being disclosed next to the number.

---

## 3. TTFT, Qwen2.5-7B long context: 15.3× @16k and 20.4× @32k (15.7×/20.4× vs serving)

**Claim.** Same harness, same gates, KV-lighter model (56 KiB/token GQA-4)
at its NATIVE 32k context (no rope scaling, no config overrides), median of
n=3 independent runs: warm reload **300 ms vs 4,588 ms** pure recompute
@16k (**15.3×**; 15.7× vs serving-shape 4,701 ms) and **535 ms vs
10,923 ms** @32k (**20.4×**; 20.4× vs 10,941 ms — the two ratios agree to
one decimal from the unrounded medians). Supersedes the single-run
321/636 ms cells and their 22.6×/25.1× serving-shape columns — the serving
baseline stopped paying a cache tax (store-on-miss is now 1.00–1.02× on
this model), which is why "vs serving" collapsed toward "vs pure".

**Rig + commands.** Identical to claim 2 with
`MODEL=Qwen/Qwen2.5-7B-Instruct LENGTHS=16384,32000`. Raw JSONL:
`bench/results/rig-e/chart2-ttft-qwen-final-run{1,2,3}.jsonl` plus
`chart2-ttft-qwen32k-baseline.jsonl`; chart
`bench/results/rig-e/chart2-qwen32k.png`.

**Definitions.** As claim 2. Why the multiple grows with context: speedup =
`prefill(L) / reload(L)`; prefill grows superlinearly, reload linearly with
KV bytes, and this model carries less than half the KV per token of
Llama-3.1-8B. That is the same physics every vendor's 100k-context headline
runs on — stated with the formula, not hidden behind it.

**Disclosed.** Loopback and n=1 pure baseline, exactly as claim 2. Worst
cross-run p50 spread: 8.6% @16k warm (2.7% @32k), inside the 10% gate.
Every warm rep passed the exact-count hit gate — 1024 blocks per rep @16k
and 1999–2000 per rep @32k (per-prompt calibration lands within a token, so
reps differ by one block; the gate is exact against each rep's own measured
count, never a nominal target). The load path is stamped
`path=pipelined-slab` in every committed record; the gathered-slots store
path is attributed in each run's job log, not the committed JSONL.

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

## 6. fp8 KV arm: 26.4× @32k vs same-dtype recompute — under the pre-registered disclosure checklist

**Status: MEASURED AND CERTIFIED (2026-07-27), against the gate below,
which was pre-registered before any fp8 measurement existed.** The standing
rule is unchanged: no fp8 TTFT multiple ships unless a campaign run
satisfies every statement below from committed JSONL. The measured result,
n=3 independent runs plus a same-dtype baseline
(`bench/results/rig-e/chart2-ttft-qwen-fp8-{run1,run2,run3,baseline}.jsonl`,
Qwen2.5-7B, `kv_cache_dtype=fp8_e4m3`): warm reload **216 ms vs 4,536 ms**
fp8 recompute @16k (**21.0×**) and **399 ms vs 10,538 ms** @32k
(**26.4×**), all five witnesses plus the amendment's kernel-determinism
control satisfied per-run — one prompt in one run was stamped
kernel-nondeterministic, excluded and disclosed per the amendment; that
run certified 7/7 gated prompts token-identical. Submission is one command:
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

**Amendment (2026-07-27): the kernel-determinism control.** This amendment
was written AFTER one failed certification run and BEFORE any re-run; the
original text above is kept unedited. The failed run (HF job
`6a6752f3c6272310d46cb761`) hit statement 5's 100% gate at 7/8: prompt i=7
(64 tokens, degenerate repetition text) diverged at generated token 10 —
while every loaded block was xxh3-verified and hit-attributed, so the
delivered bytes provably matched the stored bytes, and the identical harness
passed 20/20 on bf16. The leading hypothesis is fp8 kernel nondeterminism at
a near-tie logit — a property of the ENGINE, not of the store — and the gate
correctly refused to publish. The response is NOT to loosen the gate after
seeing a failure; it is to make the suite separate blame mechanically, and
the resulting claim is STRICTER to state:

1. **The control.** On the record engine, `bench/e2e/equivalence.py`
   generates every prompt TWICE back-to-back with identical greedy
   parameters. If the same engine disagrees with itself, the prompt is
   stamped `kernel_deterministic: false` in its EQUIVJSONL record, with the
   control's own divergence index.
2. **The gate narrows; it never loosens.** The 100% token-identity gate
   (`--min-match 100`) applies ONLY to kernel-deterministic prompts.
   Nondeterministic prompts are still replayed on the restarted engine and
   their outcome is disclosed in the summary (`n_kernel_nondet` plus each
   one's record and replay divergence indices) — they can neither fail NOR
   pass the store gate. A run where ALL prompts are kernel-nondeterministic
   FAILS loudly: the gate judged zero prompts, so nothing was certified.
3. **Claim wording.** The certification claim becomes exactly
   *"token-identical on all kernel-deterministic prompts (N of M; K excluded
   as kernel-nondeterministic, disclosed)"* — never a bare "100%". The
   summary record carries it verbatim in `certification`, alongside
   `n_gated` and `n_kernel_nondet`; `matched`/`match_rate_pct` are computed
   over the gated set.
4. **Disclosed limitation.** The record engine serves with the connector on,
   so the control's second generation may load its prefix from the store the
   first generation just wrote (those bytes are xxh3-verified identical to
   what the same engine produced moments earlier — a divergence still means
   the engine disagreed with itself over bit-identical KV). A hypothetical
   load-path defect that corrupts COMPUTE while delivering byte-identical
   blobs would therefore also trip the control and EXCLUDE the prompt from
   the hard gate. Exclusion is the conservative direction — an excluded
   prompt can never PASS the gate either — and it is never silent: the
   excluded prompt's replay divergence is published in the same summary,
   where a store-shaped pattern (e.g. every exclusion diverging identically
   on replay) remains visible to any reader. Appended power limits (same
   rule — added, not edited in): passing the control is evidence, not
   proof, of determinism — two back-to-back samples in the same engine
   process can only prove NONdeterminism, and a control-passing prompt
   that is secretly nondeterministic remains gated where it can only FAIL
   the gate, never falsely pass it. And the control certifies
   within-engine-instance replay only: a near-tie resolved stably within
   one process (e.g. kernel autotune fixed at boot) passes the control yet
   can still diverge across the restart, so a gated mismatch still admits
   cross-restart engine nondeterminism as a cause — the gate stays failed
   and is never reassigned to the store without further evidence, and a
   second 7/8-style failure on the re-run would not mean the control is
   broken.
5. **Witnesses.** `python3 bench/e2e/equivalence.py --selftest` proves, on a
   driver-controlled stub: a prompt that answers differently back-to-back is
   stamped nondeterministic, excluded and disclosed while the gate still
   passes on the deterministic prompts (even when the excluded prompt's
   replay diverges); and an all-nondeterministic run exits nonzero. The
   run's own receipt is the greppable `EQUIVCONTROL
   n=... n_gated=... n_kernel_nondet=...` line in the job log, and the
   `token_identity` stamp in the TTFT JSONL carries the N-of-M wording.

**This amendment is violated if** an fp8 number ships whose token-identity
claim omits the exclusion count, or if a prompt is excluded from the gate
without its `kernel_deterministic: false` stamp and disclosed divergences in
the committed EQUIVJSONL.

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

## 8. EC2 long-context sessions — pre-registered, first session MEASURED

**Status: PRE-REGISTERED, frozen before any metered run.** The next
measurement campaign (single-node L40S at 96k-262k tokens, then a two-node
real-NIC session) publishes ONLY under the conditions below, registered in
advance so the gate cannot drift toward whatever the session happens to
score. Predictions for every cell were frozen in advance and the
measured-vs-predicted table publishes for every cell, hit or miss.

**Status update (2026-07-29, same-commit rule): session 1 MEASURED — on an
A10G (g5.2xlarge), not the planned L40S** (g6e was capacity-refused
region-wide at run time; the GPU substitution is disclosed per PR-4 and the
A10G is the multiple-friendlier, slower-prefill box). Measured, all
conditions per this section, raw JSONL in
`bench/results/rig-g/chart2-ttft-a10g-long-*.jsonl`: **31.2× @64k (n=3,
0.8% spread) · 39.4× @96k (n=2, 0.5%) · 46.4× @131k (n=2, 1.9%; best rep
47.0×) · 54.2× @160k (n=1, reps within 1.4%; best rep 55.0×)** — fp8_e4m3,
Qwen2.5-7B-Instruct-1M in its card-sanctioned standard-attention mode
(surgery stamped per PR-6), loopback, exact-count hit gate per rep.
**PR-9 scoring, hits and misses:** the frozen prefill predictions hit
within 5% at every length (27.7/51.8/82.9 s measured vs 29.0/54.3/86.8 s
predicted); the frozen reload predictions MISSED optimistic (predicted
multiples 40/52/64× at 64k/96k/128k assumed ~3.0 GB/s effective reload —
realized ~2.0–2.1 GB/s at multi-GiB transfers on this host). Four run attempts (three run slots) were
REFUSED by the PR-7 token-identity gate (refusal records banked under
`bench/results/rig-g/refusals/`) (a different certification prompt
diverging on engine replay each time — engine-side; store bytes
xxh3-verified exact in every run) and are disclosed in BENCHMARKS.md rather
than re-rolled. 192k was refused by VRAM at boot (vLLM: max 179,392 tokens
at these settings). **Second status update (2026-07-30, same-commit rule): the L40S session
MEASURED.** Calibration: L40S/A10G prefill ratio 2.8× (the pre-registered
band's largest unknown, now fixed). Cells (fp8, same conditions):
21.7× @131k · 25.0× @160k · **34.6× @262,144 (n=2, 0.3% spread; best rep
34.7×; a third run refused by the PR-7 gate, record banked)** — the 262,144
point sits at the card's certified standard-attention ceiling: a
quarter-million-token session resumed in 2.47 s vs 85.7 s recompute (actual prefix 262,146 tokens — a +2 tokenizer overshoot over the certified 262,144, disclosed), on
plain TCP. Raw JSONL: `bench/results/rig-g/chart2-ttft-l40s-*.jsonl`. The
two-node real-NIC session is now MEASURED (2026-07-30): a separate
r6in.2xlarge store node served a g6e.2xlarge GPU node across a real VPC
NIC, iperf3 ceiling measured first (2.40 GB/s burst, store→GPU), bf16
(fp8 refused by the token-identity gate on these GPUs — refusal banked),
token-identity certified: 3.7× @32k and **7.6× @131k on the burst link,
2.4× @131k shaped to 5 Gbit** — the multiple compresses as the link
narrows exactly as `speedup = prefill / reload` predicts, and that
compression is the point: this is the honest deployable number a customer
reproduces on their own NIC, where the loopback tables (10–54×) are the
software ceiling. Raw JSONL: `bench/results/rig-g/chart2-ttft-twonode-bf16-*.jsonl`; iperf3
ceilings banked at `bench/results/rig-g/iperf3-twonode-*-store2gpu.json`.

> **PR-1 (identical-engine-config).** Within every run, the recompute and reload arms boot the same vLLM 0.25.1 image digest with byte-identical flags, env (incl. PYTHONHASHSEED=0), hf snapshot, max-model-len, max-num-batched-tokens, gpu-memory-utilization, kv-cache-dtype, sampling params, and torch.compile cache state. The canonical enumerate-and-pin string of every divergence-capable engine knob (model, dtype, kv-cache dtype, max-model-len, max-num-seqs, gpu-mem-util, max-num-batched-tokens, prefix-caching, hybrid-kv, hf-overrides, PYTHONHASHSEED, gen-tokens) is sha256-stamped into every JSONL row as `engine_args_sha`; hash inequality between arms voids the run. The baseline is never crippled: no `--enforce-eager` asymmetry, no backend overrides, prefix caching off in ALL series.
> **PR-2 (chunked-prefill disclosure).** Chunked prefill is on (V1 default; effectively cannot be off at these lengths). max_num_batched_tokens was tuned in a pre-registered sweep {8192, 16384, 32768} to MINIMIZE THE BASELINE's TTFT at the largest context, then frozen identically in both arms and stamped per-row. The warm arm is insensitive to chunk size because it computes only the tail block. This is the inverse of the vendor move. The previously published A10G multiples (10.2x/20.4x/26.4x) were measured with the engine default chunking at that pin (no explicit max_num_batched_tokens was set; the archived engine boot logs are the evidence of record and will be cited in the restating footnote); they are restated/footnoted with this methodology difference in the same release.
> **PR-3 (loopback-vs-NIC split).** Every multiple is published twice where measured: loopback (single-host software ceiling, labeled) and real-NIC (deployable, labeled), each beside its same-rig iperf3 ceiling (METHODOLOGY rule 2), with tc state, ENA allowance counters, burst-window vs baseline-floor rows, and the shaped-egress asymmetry (GET shaped, PUT unshaped) disclosed on-chart.
> **PR-4 (GPU-change disclosure).** The multiple scales inversely with GPU speed: a faster GPU shrinks it (reload is wire/CPU-bound; prefill is compute-bound). L40S numbers are not comparable to A10G numbers and are never mixed in one aggregate (aggregate.py's cell key omits gpu — rig-e and rig-g directories are never co-globbed). fp8 arms are served by FLASHINFER, bf16 arms by FLASH_ATTN, both by vLLM's own auto-selection; backend is stamped per-row and multiples are never quoted across the backend boundary without the label.
> **PR-5 (baseline n and spread gates).** Headline (Tier A) cells: n=3 independent runs (fresh daemon + fresh engine boots per run); median reported; `report --check-repeat --tolerance 0.02` enforced; a cell still >2% spread after 5 total runs publishes WITH its spread disclosed, not re-rolled. Tier B curve cells: one run × 3 reps, labeled as such. The published multiple's denominator is the NO-connector baseline series (n=3); cold-with-connector is the disclosed secondary series (store-on-miss cost 1.00–1.06x published).
> **PR-6 (model surgery disclosure).** Qwen2.5-7B-Instruct-1M runs with dual_chunk_attention_config deleted: DCA is impossible on mainline vLLM ≥0.11 (V0 backend removed), the vendor card certifies standard attention ≤262,144 tokens, all measured points ≤262,144, and both arms share the identical edited snapshot. The unmodified-model TypeError crash is archived as evidence. Any Llama point >98,304 that required rope handling would be labeled; none is planned (top Llama point 129,024, native).
> **PR-7 (verification).** xxh3 per-block verify is ON in every published cell; one verify-off control cell is published, labeled "control, not a claim". Token-identity (equivalence) certification is fp8-vs-fp8 scoped, EQUIV_N=8 per fp8 job, with the EQUIVCONTROL receipt.
> **PR-8 (competitor framing).** WEKA's 41x (105k, Llama-3.1-70B, 23.97→0.58 s, config only via NAND Research/later posts, no raw data, no n, no replication) is superseded by WEKA's own "up to 75x" @128k (TRT-LLM+GDS, 2025-06-20); both are cited whenever either is. DDN 27x→"up to 55x" (zero config) and Pure KVA 20x-NFS/6x-S3 (no token counts) are cited with their disclosure gaps. RDMA/GDS-tier numbers appear only in a "different league" table (rule 10). Pure's 6x-S3 is the honest direct comparison for the S3-tier story. All cited pages archived on publication day.
> **PR-9 (pre-registered predictions).** The arm table's two predictor columns and the adopted conservative expectation were frozen before launch; A0-B/A0-F calibration re-anchors expectations and is itself logged; measured-vs-predicted is published for every cell, hit or miss.

---

*Changing any number above requires updating its section in the same commit
— a claim without its conditions and falsification line does not ship.*
