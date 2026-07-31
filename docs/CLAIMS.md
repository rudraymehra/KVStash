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

**Amendment 2 (2026-07-30): the frozen high-margin prompt corpus.** Written
BEFORE any measurement with it, for the same reason as the determinism
control: after it, the change would be indistinguishable from choosing the
prompts that pass. The problem it fixes is measured, not hypothetical — the
original certification prompts were one pangram repeated to a target length,
whose continuations are near-tied, so an fp8 kernel's last-bit wobble flipped
the argmax and the ENGINE could not replay its own greedy output. The exact,
checkable fraction: **5 equivalence refusal records
(`bench/results/rig-g/refusals/equiv-*.jsonl`) against 22 committed result
files** — count them yourself (the sixth file in that directory is an
unrelated VRAM boot refusal). The gate's own kernel-determinism control
excluded 1-4 of 8 prompts per run as engine-nondeterministic, and the
diverging prompt indices across the five records are 2, 2, 0, 6 and 1 — not a
single fixed prompt, which is what an engine-side effect looks like. The refusals were never about the store: each
reloaded block carried its xxh3 digest and the hit counters attributed every
warm rep, which is why the refused runs are published rather than hidden.

1. **The corpus is frozen and checksummed.**
   `bench/e2e/equiv-prompts-high-margin.txt`, committed with this amendment,
   sha256 `b76ad0c4c4c194aa851d202b3cd9ba991a21428aa50b9cf8838b65af0c68d0dc` (the first 16 hex chars are stamped into every
   record as `prompt_corpus_sha256`; recompute the file's digest to check). Editing it after seeing a result voids the
   certification — the digest in the artifacts is how a reader detects that.
2. **The screen NARROWS, it never loosens.** `--min-match` stays 100. A prompt
   whose smallest top1-vs-top2 logprob gap falls under the pre-registered
   floor (**2.0 nats**, frozen here) is a near-tie the engine may re-decide
   across a restart, so it cannot discriminate a store fault from numerical
   noise: it is EXCLUDED from the gate and DISCLOSED
   (`low_margin: true`, `min_margin`, `n_low_margin_excluded`, and the
   exclusion named inside the `certification` string), exactly as the
   kernel-determinism control already treats a self-inconsistent prompt. An
   excluded prompt can never convert a mismatch into a pass.
3. **Coverage is preserved, not traded away.** Corpus prompts are
   LENGTH-CALIBRATED onto the same boundary trio the legacy builder used
   (m*B-1 / m*B / m*B+1, exact to the token), with the padding in the middle so
   the high-margin entry is always what the model reads last. A pre-registered
   **coverage floor** (`--min-blocks-covered`, default 16) makes the record
   phase REFUSE (rc 2) a prompt set too shallow to exercise the block-index
   path — checked against the CONFIGURED lengths BEFORE any completion is
   issued, so a doomed set costs nothing — with `max_blocks_covered` stamped in
   every summary. The floor applies to corpus mode, which is where it was
   registered. Without this the
   screen would have silently narrowed certification to the first block or two
   — a real regression, caught by review before any measurement.
4. **The margin is judged over a frozen horizon, and the identity window
   MATCHES it.** `--margin-horizon` (default **8** steps): a min() over a long
   free-running continuation shrinks as the text drifts and would exclude
   nearly every prompt, defeating the fix while looking strict. The
   certification window is therefore also **8 generated tokens** in corpus mode
   (`EQUIV_GEN_TOKENS=8`, the pairing pre-registered with this corpus).
   Comparing 32 generated tokens while screening margins over 8 is internally
   inconsistent: it charges the store for engine drift occurring outside the
   screened region. **Measured on 2026-07-30 (HF job
   `6a6b7a5623ed89c748ec7f73`, A10G, fp8_e4m3, this corpus, window 32):** of 8
   prompts, 5 were kernel-nondeterministic with divergences at generated tokens
   1, 4, 6, **9 and 29** — the last two outside the screened horizon — and 3
   were low-margin (gaps 0.19-1.0 nats) yet all three replay-MATCHED. Zero
   prompts were gated, so nothing certified. That run's summary is the evidence
   for this pairing; the window is fixed here BEFORE the re-run.
   **The margin floor stays 2.0 nats.** Lowering it after seeing that the
   excluded prompts happened to match would be exactly the cherry-picking this
   pre-registration exists to prevent, so it is not done. Any published fp8
   claim states its window explicitly: "token-identical over the first 8
   generated tokens on all gated prompts".
5. **The two exclusions are disjoint and separately reported.**
   `n_kernel_nondet` counts only self-inconsistent prompts;
   `n_low_margin_excluded` counts only near-ties; `n_gated + n_kernel_nondet +
   n_low_margin_excluded == n` is arithmetic the rig re-checks from the record.
   `low_margin` is RE-MEASURED at compare time: the replay requests its own
   top-2 margins over the frozen horizon, and an exclusion stands only if the
   REPLAY's margin is also under the floor (a stored flag that contradicts the
   recorded arithmetic aborts the compare, rc 2). The state file is data on
   disk and therefore forgeable; the live engine is not — so editing
   `min_margin` cannot launder a real divergence into a disclosed "near-tie".
   The partition is enforced with kernel-nondeterminism taking precedence, and
   the compare REFUSES to emit a summary whose own bucket arithmetic does not
   close.
6. **Tripwire.** If EVERY excluded low-margin prompt also diverges on replay,
   that is a store-shaped pattern rather than independent coin-flips: the run
   prints a loud TRIPWIRE line and stamps
   `tripwire_all_low_margin_diverged: true` into the summary, so the signal
   survives into the artifact instead of scrolling past in a log.
7. **A store fault still fails the gate.** Wrong bytes, a wrong slot, or a
   stale block changes the output regardless of logit margin — the screen
   removes only the engine's coin-flips, not the store's discriminating power.
   *Witnesses (all no-GPU, in `--selftest`):* a near-tied prompt is excluded and
   disclosed while the gate still certifies the wide-margin prompts at 100% and
   the corpus digest reaches the summary; a set too shallow for the coverage
   floor is REFUSED rather than certified; and a store-fault-shaped divergence
   under the frozen corpus still FAILS the gate.
8. **Disclosure wording.** An fp8 claim certified under this corpus reads
   "token-identical on all gated prompts (N of M; K excluded as
   kernel-nondeterministic; L excluded as low-margin, all disclosed)" — never
   a bare "100%".

**Amendment 3 (2026-07-30): what measurement said about fp8 reuse, and corpus
v2.** Amendment 2's corpus was written from intuition and a measured
calibration pass disproved it. The calibration (HF job
`6a6b7eb523ed89c748ec7fef`; A10G, vLLM 0.25.1, Qwen2.5-7B-Instruct, fp8_e4m3,
8-token horizon; full per-candidate data committed at
`bench/results/rig-e/margin-calibration-a10g-fp8.jsonl`) measured 48 candidate
prompts on two engine properties only — the top1-top2 logprob gap and whether
the engine reproduces itself back-to-back — with no daemon, no connector and no
restart in the loop, so it cannot select on any store outcome.

1. **The measured result.** **48 of 48 candidates were self-consistent**: with
   both generations computing their KV fresh, this engine reproduces its own
   greedy output every time. But only **2 of 48** cleared the 2.0-nat margin
   floor, and the **median candidate margin was 0.25 nats** — this model's
   greedy choices are usually near-ties.
2. **What the data isolates: MARGIN — and what it does not isolate.** The
   variable this measurement cleanly identifies is the top1-top2 logit gap.
   Near-tie prompts fail certification and wide-margin prompts pass it: the two
   measured-eligible prompts certified on the first attempt at 16k and again at
   131k, while the v1 corpus — whose prompts measured 0.06-1.0 nats — produced
   10 of 16 low-margin exclusions and 6 of 16 self-divergences
   (`bench/results/rig-g/refusals/equiv-fp8-corpus-v1-16prompts.jsonl`, digest
   `b76ad0c4c4c194aa`).

   **The tempting explanation is NOT established, and this correction is
   published rather than quietly dropped.** An earlier draft of this amendment
   asserted a mechanism: that at fp8 precision, attention over byte-identical KV
   selects a different argmax depending on whether the KV was computed in place
   or reloaded, and that this generalizes to any fp8 KV reuse including an
   engine's own prefix cache. Review against the artifacts refuted that as an
   isolated finding, on three grounds:
   (a) the certified 131k run's own control shows **0 of 6 prompts
   self-divergent WITH the store in the loop — including 4 prompts measured
   below the 2.0-nat floor**, which under that mechanism were the likeliest to
   flip and did not
   (`bench/results/rig-e/equiv-fp8-131k-certified-run1.jsonl`);
   (b) the calibration that showed 48/48 self-consistency changed **three**
   variables at once versus the certification runs — no store in the loop, a
   different checkpoint (`Qwen2.5-7B-Instruct` vs the DCA-stripped
   `Qwen2.5-7B-Instruct-1M`), and a disjoint prompt set — so "remove the store
   and divergence disappears" is an uncontrolled comparison;
   (c) the committed refusals show 1-4 of 8 self-divergent, and in each the
   refusal was actually caused by a GATED, self-consistent prompt mismatching
   after the restart.
   So the honest statement is: **fp8 greedy decoding on this stack is
   reproducible on wide-margin tokens and fragile on near-ties, and this
   experiment does not establish what perturbs the near-ties.** Reload is one
   candidate; kernel scheduling under a different batch shape is another. The
   experiment that would isolate it — same checkpoint, same prompts, store in
   versus out as the only varied factor — is not run, and until it is, no claim
   here attributes the effect to reload or generalizes it to other systems.

   **Follow-up datum (2026-07-31, PR-10 session; labeled data, not a claim.)**
   The construction-matched pair pre-registered in PR-10 measured: across the
   six connector runs' store-attached padded prescreens, **14 of 144 records
   were self-inconsistent (~9.7%)**, versus **0 of 144** across all store-free
   records on the same box and checkpoint — 96 of the same padded
   construction plus the 48 raw-construction calibration records
   (`bench/results/rig-g/prescreen-attached-*.jsonl` vs
   `mech-pre-l40s-fp8-padded.jsonl` + `mech-cal-l40s-fp8-raw48.jsonl`).
   In these arms, self-inconsistency appeared only with the store in the
   loop. This narrows
   the open question; it does not settle it — the near-tie candidate pool
   still has no store-attached leg, and the store-attached engine differs
   from the baseline engine in more than the reload path (the connector's
   host-side work shares the box), so cause remains OPEN as stated above.

3. **Corpus v2.** `bench/e2e/equiv-prompts-high-margin-v2.txt`, sha256
   `8d8388a3800e95464b41f5c4357b36629652ed620743d13e8a15f8e14ce0bf06`, containing exactly the 2 measured-eligible prompts
   (margins 4.50 and 3.75 nats). Corpus v1 (digest `b76ad0c4c4c194aa`) is
   SUPERSEDED, not deleted. Note the provenance precisely: v1 was disproven by
   the **16-prompt certification run against v1 itself**
   (`bench/results/rig-g/refusals/equiv-fp8-corpus-v1-16prompts.jsonl`: 10 of 16
   low-margin, 6 of 16 self-divergent), not by the 48-candidate calibration,
   whose pool is disjoint from v1 and never measured a v1 prompt. Both v1
   refusal records stay committed.
4. **What an fp8 number certified under v2 may and may not say.** It may say
   "token-identical over the first 8 generated tokens on all gated prompts (N of
   M; the rest excluded as near-ties, disclosed)". It may NOT say or imply that
   fp8 reload is bit-exact against recompute in general — item 2 measured that
   it is not on near-tie tokens. Any published fp8 cell carries a pointer to
   this amendment, and the per-rep exact-count hit gate (which is unaffected by
   any of this) remains the primary proof that the KV came from the store.
5. **Floors unchanged.** The 2.0-nat floor and the 8-token window stay exactly
   as pre-registered. Two prompts is thin coverage and is stated as such rather
   than fixed by lowering a threshold after seeing the data.

**This amendment is violated if** an fp8 number ships whose corpus digest does
not match the committed file, whose margin floor differs from 2.0 nats without
its own dated amendment, or whose exclusions are not published beside it.

**This amendment is violated if** an fp8 number ships whose token-identity
claim omits the exclusion count, or if a prompt is excluded from the gate
without its `kernel_deterministic: false` stamp and disclosed divergences in
the committed EQUIVJSONL.

---

## 6b. fp8 @131k, CERTIFIED: 50.9× vs same-dtype recompute (measured 2026-07-30)

**Claim.** Reloading a 131,072-token prefix's fp8 KV from kvblockd into a FRESH
vLLM engine takes **1,631 ms** vs **82,945 ms** to recompute it with no
connector installed under the same `kv_cache_dtype=fp8_e4m3` — **50.9×** — with
token identity CERTIFIED on the gated prompts of the measurement-selected frozen
corpus v2.

**Rig + commands.** NVIDIA A10G (HF Jobs a10g-large), vLLM 0.25.1 pinned,
Qwen2.5-7B-Instruct-1M served in the card-sanctioned standard-attention mode
(`STRIP_DCA=1`), loopback, `max_num_batched_tokens=8192` frozen across arms:
```
MODEL=Qwen/Qwen2.5-7B-Instruct-1M STRIP_DCA=1 KV_CACHE_DTYPE=fp8_e4m3 \
  KV_BYTES_PER_TOKEN=28672 LENGTHS=131072 REPS=2 EQUIV_N=6 \
  GPU_MEM_UTIL=0.92 MAX_NUM_BATCHED_TOKENS=8192 bench/rigs/hf-gpu/submit.sh
BASELINE_ONLY=1 (same env, EQUIV_N dropped) bench/rigs/hf-gpu/submit.sh
```
Raw JSONL: `bench/results/rig-e/chart2-ttft-fp8-131k-certified-{run1,baseline}.jsonl`;
corpus selection data: `bench/results/rig-e/margin-calibration-a10g-fp8.jsonl`.

**Definitions.** As claim 2 (warm = the exact populated prompt on a RESTARTED
engine, prefix caching off in both phases). Both arms ran one
`kv_cache_dtype`, so the multiple measures the store and never the quantizer.

**Disclosed, and this is the load-bearing part.**
- **Certification scope:** *token-identical over the first 8 generated tokens on
  all gated prompts (2 of 6; 4 excluded as near-ties, disclosed)*. Two gated
  prompts is THIN coverage. It is stated as thin rather than widened by moving a
  threshold after seeing results — the 2.0-nat floor and 8-token window are
  exactly as pre-registered in Amendment 2.
- **Gated depth (added 2026-07-31, after an adversarial re-read of the committed
  artifact):** the exclusions were not random in depth — every 15–16-block
  prompt (targets 255–257) measured a near-tie margin (0.25–0.75 nats; the
  fourth exclusion, 0.125, was the shallow target-64 prompt) and was
  excluded, so the gate judged nothing deeper than **4 blocks** even though the
  built set reached 16 (its coverage-floor receipt reflects the BUILT set).
  Padding collapses margins: the same entries measured 4.50/3.75 nats raw.
  Byte-correctness to the full 8,192-block depth of this cell is carried by the
  per-rep exact-count hit gate and per-block xxh3 verify, not by the token gate;
  equivalence.py now stamps `gated_max_blocks` into every summary and the
  certified wording itself so this distinction can never again ride silently.
- **fp8 equivalence is certified on wide-margin tokens only.** Amendment 3's
  calibration (48 candidates, committed) measured that only 2 of 48 prompts
  clear the 2.0-nat margin floor (median 0.25): fp8 greedy decoding on this
  stack is reproducible on wide-margin tokens and fragile on near-ties. What
  perturbs the near-ties is NOT established here — an earlier draft blamed
  reload-vs-recompute and Amendment 3 records why the artifacts refuted that.
  So this claim asserts a measured SPEED result plus equivalence on the gated
  wide-margin prompts, and asserts nothing about near-tie tokens or about other
  systems. bf16 certifies without exclusions.
- Loopback (network time ≈ 0), single run × (2+1) reps per arm; baseline rep
  spread 0.01%. An independent cross-session baseline on the same GPU, dtype,
  model and length gives 82,883 ms → 50.8×, agreeing to 0.1×.
- Per-rep exact-count hit attribution (`warm_hits_verified: true`) is unaffected
  by any of the above and remains the proof the KV came from the store.

**This claim is false if** rendering the committed JSONL yields different
medians; if a warm rep's `kvb_hits_total` delta did not equal its expected block
count; if the certification's corpus digest is not `8d8388a3800e9546`; or if the
claim is ever quoted without its certification scope and the Amendment 3 caveat.

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
(that session's fp8 attempt was refused by the legacy no-corpus gate —
refusal banked; superseded by the PR-10 fp8 session below, 6/6 certified),
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
> **PR-10 (two-node fp8 session — frozen 2026-07-31, before any provisioning; this paragraph was rewritten once, same day, after a FULL review ladder over the uncommitted draft and before any instance existed — the commit carrying it predates the session ledger's first box).** *Motivation.* The published real-NIC cells are bf16 (7.58× @131k over an unshaped 19.15 Gbit/s measured link; 2.43× shaped to 4.87 Gbit/s; iperf3 JSONs committed under `bench/results/rig-g/`). Provenance of that choice, stated precisely: the banked two-node fp8 refusal (`bench/results/rig-g/refusals/equiv-twonode-fp8-32k.jsonl`) ran the LEGACY no-corpus prompt builder — its record carries no prompt_corpus key; that code predates corpus support — and corpus v1 was separately disproven by its own 16-prompt certification run (Amendment 3 item 3), not by the 48-candidate calibration and not on any two-node arm. The measurement-selected corpus v2 (digest `8d8388a3800e9546`, Amendments 2–3 under claim 6) plus the padded-margin prescreen (below) address the refusal mechanism; Gate A addresses whether fp8's halved bytes should even help. *Gate A evidence.* One banked run (`bench/results/gate-a/`, dev Mac, BLOCKS=1024): time ratio 3.32× for 2× bytes at equal block count ⇒ byte-bound per the script's pre-registered ≥1.70 threshold, with the artifact's own caveats (8 GiB host; magnitude not a scaling law; CLIENT path only — the script's verdict explicitly disclaims the GPU scatter stage). Session step 0, before any GPU spend: rerun on the store node (r6in.2xlarge, 64 GiB) at BLOCKS=2048 and BLOCKS=16384 — the 262k arm's exact block count — banked to the same directory. Decision rule: ratio < 1.70 at BLOCKS=16384 ⇒ the session's premise fails, the GPU node is never provisioned. A previously quoted "t ~ bytes^0.98" had no committed artifact and is withdrawn (the script computes a two-point ratio, not an exponent). *Design.* Same instance pair as the bf16 session (g6e.2xlarge GPU + r6in.2xlarge store, us-east-1), fp8_e4m3 in BOTH phases of every pair (never fp8-vs-bf16), gpu-mem-util 0.92 everywhere. Denominators are the no-connector fp8 baseline series on the same box, ONE LENGTH PER BASELINE ARM (`nic-base-32k/64k/128k/262k`) so every denominator boots the exact max-model-len pin of its cell (PR-1 hashes max-model-len; the bf16 session's multi-length-baseline convention is not repeated). Chunk size: 32k/64k/131k arms and their denominators freeze MNBT 8192 for same-pin comparability with the banked bf16 two-node cells, disclosed against PR-2 as follows — the a0c sweep winner on this box class was 16384 (measured for bf16 loopback at 262k), a non-minimal denominator inflates a multiple, and the banked check cell `nic-base-128k-m16` (16384) quantifies that bias beside the published 131k cell. The 262k pair has no bf16 counterpart, so PR-2 rules it outright: both `nic-base-262k` (8192) and `nic-base-262k-m16` (16384) run FIRST and the full pair (`nic-262k` or `nic-262k-m16`) runs at whichever pin gave the LOWER baseline median. *Certification, scoped honestly.* Frozen v2 corpus, 8-token window, 2.0-nat floor, EQUIV_N=8 per connector arm (consistent with PR-7), exact-count hit gate per rep. What the token-identity gate certifies is a same-dtype boundary-prompt round-trip (prompts built to the 63–257-token boundary trio) on the same box alongside the cell — it is NOT a depth-proportional proof at 131k/262k, and the certified wording now states the depth the gate actually judged (`gated_max_blocks`, equivalence.py) beside the built depth, because padding provably collapses margins (the certified 131k artifact excluded every 15–16-block prompt as a near-tie and gated only 3–4-block prompts). Byte-correctness to FULL depth is carried per-cell by the exact-count hit gate plus per-block xxh3 verify. A deliberate one-block corruption probe under corpus v2 on a real engine remains unrun and is pre-registered as follow-up work, not assumed. Refusal handling: a token-identity refusal banks its record and publishes no claim for that cell; the measured-vs-predicted table still lists that cell's speed number annotated "refused certification — not a claim" so refusals cannot silently prune the table (non-random missingness disclosed). *Prescreen (new, both uses pre-registered).* equivalence.py `--phase prescreen` measures the PADDED certification construction's margins + self-consistency on the live engine. (a) Every connector arm runs it on engine #1 before the paid populate and refuses on zero eligible prompts — raw margins provably do not transfer to padded prompts (4.50 nats raw → 0.125 padded in the certified artifact), and this rig has paid for post-populate refusals twice. Selection-independence: the gate re-measures its own margins at record/compare; the prescreen can stop a run, never steer it. (b) `nic-mech-pre` runs it store-free. *Frozen predictions* (model basis stated: the shaped bf16 @131k cell pins fixed overhead ≈ 240–250 ms, the unshaped @131k cell then pins the host path ≈ 2.0 GB/s; the banked 32k bf16 cell runs ~20% FASTER than this two-parameter model predicts — 967.8 ms measured vs ~1180 modeled — so the model over-predicts warm time where the fixed term dominates, small-L bands widened accordingly): @32k 4.4× (3.4–6.5), @64k 7.5× (6–10.5), @131k unshaped 12.5× (10–14), @131k shaped-5 4.2× (3.8–4.7), @262k unshaped 21× (17–24; fp8 @262k moves exactly the 7.516 GB the bf16 @131k cell moved, through 2× the block count — a free block-vs-byte probe at scale). *Falsifiers, written before the run.* Ratio falsifiers: fp8 @131k-unshaped < 9× ⇒ publish the host-path localization, not a multiple; fp8 @131k-shaped-5 < 3.4× ⇒ the byte-bound premise fails on a wire-bound stage — halt the fp8 real-NIC story pending investigation, adjudicated ONLY if the drift bracket (below) reproduces. Denominator-free falsifiers on the directly modeled quantity: warm fp8 @131k-unshaped > 3.0 s ⇒ the ~2.0 GB/s host model fails regardless of any baseline; warm fp8 @262k-unshaped > 5.2 s (>30% over the same-bytes bf16 @131k anchor, 4.03 s) ⇒ per-block cost is visible at 2× block count — publish the block-bound localization. Drift bracket: after the shaped run, `nic-128k` reruns unshaped; if its warm p50 is not within 10% of the earlier unshaped run, the session's shaped-vs-unshaped contrast is confounded by drift and the shaped falsifier is not adjudicated. Misses publish as misses (PR-9). *Arm order (store restarted before every connector arm — machine-enforced: run-arm.sh refuses a non-empty store, checks the recorded tc state and the arena fit, and refuses an instance-type/annotation mismatch via IMDS; job.sh stamps IMDS instance_type and a -dirty git marker per row).* Step 0 Gate A on the store node → iperf3 unshaped (banked) → nic-base-32k → nic-base-64k → nic-base-128k → nic-base-128k-m16 → nic-base-262k → nic-base-262k-m16 → nic-mech-cal → nic-mech-pre → nic-32k → nic-64k → nic-128k → nic-262k[-m16 per the sweep] → shape 5 Gbit → iperf3 shaped (banked) → nic-128k(shaped) → unshape → iperf3 (banked) → nic-128k drift bracket. Halt rules before connector arms: nic-mech-cal (raw 48-pool calibration on this box+checkpoint) showing BOTH v2 entries ineligible raw ⇒ corpus-transfer failure, halt; nic-mech-pre (padded, store-free) showing zero eligible v2 prompts ⇒ same, halt (its pool leg is '?'-tolerated: zero eligible there is an expected, recorded finding). *Mechanism experiment (Amendment 3 follow-up), stated at its true strength.* The construction-matched pair this session banks is: store-FREE padded self-consistency + margins (nic-mech-pre, PRESCREENJSONL, baseline engine) vs store-ATTACHED padded self-consistency + margins (each connector arm's pre-populate prescreen on engine #1 plus the record phase's EQUIVCONTROL double-generation) — same box, same checkpoint, same builder, same lengths pin (all mech arms boot LENGTHS=131072 to match the 128k anchor arm; max-model-len is a PR-1-hashed knob). Store-in-loop is the only varied factor for the v2 construction. What it does NOT do, disclosed: the 46 non-v2 pool candidates get both store-free constructions (raw via nic-mech-cal, padded via nic-mech-pre's pool leg) but no store-attached side this session, so the near-tie population's store sensitivity — Amendment 3's open question — is narrowed, not settled.
> **PR-10 RESULTS (2026-07-31, session complete; every number independently re-derived from the banked JSONL before this paragraph was written).** Step 0 held: Gate A on the store node measured 2.25× (BLOCKS=2048) and 2.95× (BLOCKS=16384 — the 262k arm's exact block count), both byte-bound ≥ the frozen 1.70 (`bench/results/gate-a/`). (EQUIV_N=24 and the retry rule are per the dated addendum below, which predates every connector arm.) Measured-vs-predicted, PR-9 discipline, **5/5 in-band**: @32k 5.96× (pred 4.4, band 3.4–6.5), @64k 8.36× (7.5, 6–10.5), @131k unshaped 12.07× (12.5, 10–14), @131k shaped-4.84 4.02× (4.2, 3.8–4.7), @262k 19.21× (21, 17–24; MNBT-16384 pin — the sweep's baseline-minimizing winner, 84,916 < 85,270 ms, honored as frozen and proven from the engine_args_sha preimages). All four falsifiers PASS: 12.07 ≥ 9; warm 2.162 s ≤ 3.0 s; warm 4.420 s ≤ 5.2 s (the same-bytes probe: fp8 @262k moved bf16 @131k's exact 7.516 GB through 2× the blocks taking 9.7% longer — within the disclosed ~10% run-to-run spread, so an UPPER BOUND on per-block cost; byte-dominant at scale); shaped 4.02 ≥ 3.4, validly adjudicated — the drift bracket reproduced the unshaped warm at −9.894%, inside the 10% window by ~0.1 point (disclosed as thin). Certification: 6/6 connector runs certified on the FIRST draw (gated 2–4 of 24, gated depth ≤ 15–16 blocks stamped per summary — full-depth byte proof stays with the hit gate + xxh3; zero gated mismatches anywhere; the retry rule was never exercised; every equivalence summary banked). Hit attribution 19/19 pairs exact. Ceilings banked: 18.84 / 4.84 / 19.11 Gbit/s (sum_received). Scope notes: this pair was cross-AZ (different default subnets, artifact-proven; AZ letters rest on operator attestation), each cell quotes its own banked ceiling; the session ledger's $2.28 is the sum of logged per-cell GPU-node minutes only (one baseline row missing, store node untracked in it) — the honest session figure is the box-lifetime ledger. Mechanism data (labeled data, not a claim): across the six connector runs' store-attached prescreens, 14 of 144 padded records were self-INconsistent vs 0 of 144 on the store-free engine — recorded under Amendment 3 as narrowing, not settling, its open question (the pool near-tie population still has no store-attached leg).
> **PR-10 addendum (2026-07-31, after the store-free mech arms, BEFORE any connector arm).** The banked go/no-go evidence (nic-mech-cal-run1-calib.jsonl, nic-mech-pre-run1-prescreen.jsonl) measured: the A10G-calibrated raw margins transfer poorly to this L40S — one v2 entry collapsed 3.75 → 0.00 nats raw, the other held (4.50 → 3.00); padded, 4 of 48 v2-construction prompts cleared the frozen 2.0-nat floor (~8% per-prompt; 48/48 self-consistent both raw and padded at the frozen window). Neither frozen halt rule fired, but at EQUIV_N=8 the probability a connector arm's record draw contains ZERO gate-eligible prompts is ~0.92^8 ≈ 50%, and that refusal lands post-populate. Amendment, before any connector measurement exists: (a) EQUIV_N is raised 8 → 24 for this session's connector arms — a SAMPLE-SIZE change justified entirely by store-free evidence that cannot select on a store outcome; the 2.0-nat floor, 8-token window, 100% min-match, and every other threshold stay exactly as frozen (PR-7's "EQUIV_N=8 per fp8 job" is superseded for this session only, disclosed here). (b) Retry rule, stated before it is ever exercised: a compare-phase refusal whose gated set is EMPTY (nothing was judged — the nonce lottery drew only near-ties) may be retried with fresh nonces, EVERY refusal record banked and counted in the session ledger; a refusal containing ANY gated-prompt MISMATCH is store evidence and HALTS the session for investigation — actual evidence is never re-rolled. (Banking note, same day: the two go/no-go artifacts named above were banked as `mech-cal-l40s-fp8-raw48.jsonl` / `mech-pre-l40s-fp8-padded.jsonl` under `bench/results/rig-g/`.)

> **PR-11 (reload-path knob A/B + code-delta measurement — frozen 2026-07-31, before any provisioning; the connector/client code under test merged and CI-green at 92e733e before this paragraph was written).** *Motivation, from banked PR-10 artifacts only.* Every unshaped fp8 two-node cell reloaded at ~70–75% of its session's banked iperf3 ceiling (warm ÷ pure-wire floor: 0.699/0.746/0.724/0.708 across an 8× byte range; the shaped cell sat at ~96% = link-bound, correctly excluded from this story). The overhead fits ≈0 fixed + ~42% per-byte — it lives IN the transfer. A laddered code change (merged reads folding the next block's 32B prefix into the body's final recv; per-conn xxh3 reuse; a `kvblockd_pipeline_half_bytes` knob whose 512 MiB setting halves the 262k pass count 29→15; one idx-staging write per pass) rides into every arm below at the node clone's HEAD. Disclosed for the record: the original diff's fence-relocation rationale was REFUTED in review (the pre-change code already submitted the next drain before the scatter; the relocation was reverted), so the ~30 ms/pass boundary attribution is an OPEN question this session instruments (py-spy arm) rather than a claim. *Design.* Same pair class as PR-10 (g6e.2xlarge GPU + r6in.2xlarge store, us-east-1; each cell quotes its own session-banked iperf3 ceiling). Every knob under test is `kv_connector_extra_config` — never an engine arg; all 131k arms share `nic-128k`'s ENGINE_ARGS_SHA by construction. Delta arms (`nic-ab-ctl`, `nic-ab-f8` fanout 8, `nic-ab-f8h512` fanout 8 + 512 MiB halves, `nic-ab-voff` verify-off, `nic-ab-pyspy` profiler-attached) run EQUIV_N=0: uncertified DIAGNOSTICS whose stamped configs steer the decision rule and are never published as cells (the exact-count hit gate + per-block xxh3 still enforce every warm rep; voff drops xxh3 and is stamped `verify=off (DIAGNOSTIC arm)`; pyspy is stamped and its sampling overhead disqualifies its TTFT). Publishable cells re-run CERTIFIED (corpus v2, EQUIV_N=24 per the PR-10 addendum, in-run prescreen, refusal/retry rules unchanged). MNBT: 131k freezes 8192 (comparability with the banked bf16/fp8 131k cells), 262k freezes 16384 (PR-10's sweep winner, honored — no re-sweep). Denominators are fresh same-SHA in-session baselines (`nic-base-128k`, `nic-base-262k-m16`), per PR-10's per-length pin-matching. *Decision rule, frozen, all deltas measured against ctl.* Let d(v) = (ctl_p50 − v_p50)/ctl_p50 for v ∈ {f8, f8h512}. A variant with d(v) < 0.02 is treated as EQUAL to ctl; among equals the config with fewer non-default knobs wins (ctl < f8 < f8h512). If both variants clear 2%, the one with the larger d wins, EXCEPT when |d(f8h512) − d(f8)| < 0.02, which resolves to f8 (fewer knobs). Certified arms then run at the selected config via the matching frozen variant (`nic-ab-131k`/`nic-ab-262k` for f8h512, `nic-ab-131k-f8`/`nic-ab-262k-f8` for f8, `nic-ab-131k-ctl`/`nic-ab-262k-ctl` for ctl — every selectable config has a table-frozen arm; the control arms sed-pin fan-out 4 so no session-shell variable can turn the control into a treatment). voff/pyspy are never eligible. *Cross-session honesty.* Ceilings vary per pair: A/B deltas are within-session; against PR-10 the primary metric is link utilization (bytes ÷ warm ÷ session ceiling), and if this session's unshaped ceiling differs from PR-10's banked 19.21 Gbit/s (sum_sent; 19.11 sum_received) by >5%, absolute warm-time comparisons are annotated ceiling-confounded. *Frozen predictions* (basis: PR-10 banked 131k warm 2.162 s run1 / 1.948 s bracket — ~10% spread; audit attribution ~43 ms per 268 MiB pass ≈ 30 boundary + 11 client-drag; Gate A client drain 2.15 GB/s verify-on): ctl warm @131k 1.95–2.30 s; f8 1.80–2.20 s; f8h512 1.70–2.10 s; voff delta, defined (voff_p50 − f8h512_p50)/f8h512_p50, in −12% to +2% (both outcomes informative: strongly negative ⇒ inline xxh3 is the per-byte drag and off-thread verify is the next lever; ≈0 ⇒ the hash already overlaps and the drag is elsewhere); certified cells 12.0–15.0× @131k (PR-10: 12.07×) and 19.0–23.0× @262k (PR-10: 19.21×). *Falsifiers, before the run.* F1 regression kill: ctl warm p50 > 2.40 s ⇒ the merged code REGRESSED the reload path — publish nothing, the session becomes a diagnosis session (bisect knobs/verify on the box). F2 knob refutation: min(f8, f8h512) > 0.98×ctl ⇒ the knobs are refuted on this link class; certified cells run at ctl config and the session publishes the code-delta + attribution story only. F3 consistency, SCOPED to the 131k cell (no same-config diagnostic exists at 262k, so F3 is undefined there by construction): the certified 131k cell's warm p50 differing from its same-config 131k diagnostic arm by >7% ⇒ that cell does not publish (certification contamination suspected). The 262k cell is gated by F4's floor plus its own certification/hit/xxh3 gates only — stated so no post-hoc gate can be invented for it. F4 publication floor: certified @131k < 11.5× or @262k < 18.5× ⇒ publish no multiple (worse than PR-10 beyond drift allowance); the miss publishes as a miss (PR-9). Drift bracket: `nic-ab-ctl-bracket` closes the session; if its warm p50 is not within 10% of `nic-ab-ctl`, the knob-A/B story is drift-confounded and DOES NOT PUBLISH, and any certified cells already run publish (subject to F4 and their own gates) as standalone cells annotated "config selected under measured drift" — the bracket firing voids the A/B conclusion, never the cells' own certification. *Arm order (store restarted before every connector arm; run-arm.sh's empty-store/tc-state/arena/IMDS gates all apply).* Gate A step-0 on the store node (kill rule unchanged: ratio < 1.70 at BLOCKS=16384 ⇒ halt before the GPU node exists) → iperf3 unshaped banked → `nic-base-128k` → `nic-ab-ctl` → `nic-ab-pyspy` → `nic-ab-f8` → `nic-ab-f8h512` → `nic-ab-voff` → decision rule → certified 131k at publish-config → `nic-base-262k-m16` → certified 262k at publish-config → `nic-ab-ctl-bracket` → teardown, $0 residue. Spend cap ≤ $15; any halt tears down the same hour. The py-spy speedscope artifact banks under `bench/results/rig-g/` and feeds attribution for any future pass-boundary work — it produces no numeric claim this session.

> **PR-11 RESULTS (2026-07-31, session complete; every number independently re-derived from the banked JSONL before this paragraph was written, and the verifier's four discrepancy flags are resolved IN this paragraph).** Same-AZ pair this time (us-east-1c, both nodes; PR-10 was cross-AZ); ceilings banked: 19.17 Gbit/s pre-session, 18.56 post (−3.2% credit drift, disclosed; sum_received throughout). Step 0 held: Gate A on the store node 1.96× (BLOCKS=2048) / 2.10× (16384) — byte-bound ≥ the frozen 1.70 — plus an unplanned free control: this box's drain ran ~22% under PR-10's banked 2.15 GB/s, so the OLD client was rerun on the SAME box (worktree at d0fc77a) — old 1.59/1.67 GB/s vs new 1.67/1.69: box variance, not a code regression, and the merged-read client is marginally ahead (`bench/results/gate-a/*-s3.txt`). Knob A/B, all stamped `uncertified (DIAGNOSTIC arm)`, warm p50 @131k: ctl (fanout 4) 2096.5 ms, f8 (fanout 8) 2051.4, f8h512 (+512 MiB halves) 2052.1, voff (verify off) 2042.1. Frozen bands: **4/4 in-band** (ctl 1.95–2.30 ✓, f8 1.80–2.20 ✓, f8h512 1.70–2.10 ✓ at the top edge, voff delta −0.49% inside −12%..+2%). Decision rule, mechanically applied: d(f8)=2.15%, d(f8h512)=2.12%, mutual gap 0.03% < 2% ⇒ tie ⇒ fewer knobs ⇒ **publish-config = fanout 8 alone** — the 512 MiB half bought nothing (halving the pass count did not move warm time: the per-pass-fixed-cost model is refuted at this scale), and verify-off's ≈0 delta lands the pre-registered "hash already overlaps" branch: **xxh3 verification is free at 131k on this link class** (the loopback-era prime per-byte suspect, acquitted here; the off-thread-verify lever is dead at this scale — the dropped 262k leg leaves scale dependence untested). Falsifiers: F1 PASS (2096.5 ≤ 2400 — the merged client code did not regress), F2 does not fire but thinly (f8 cleared the 2% bar by 0.15 points), F3 PASS (certified vs same-config diagnostic +2.17% < 7%), F4 PASS, drift bracket +6.29% — inside the 10% window, A/B validly adjudicated. **Certified cell: fp8 @131k = 2095.9 ms vs 25654.1 ms same-SHA pin-matched fresh baseline = 12.24×** (frozen band 12.0–15.0; PR-10 published 12.07×), certification 6 of 24 gated (the widest gated set banked so far), 6/6 token-identical, gated depth ≤ 16 blocks (matching PR-10's deepest), corpus v2 digest verified; the first draw REFUSED with an EMPTY gated set (21 low-margin + 3 kernel-nondet, zero gated mismatches — banked under `refusals/`) and the PR-10 addendum's retry rule was exercised for the first time, exactly as written: one fresh-nonce retry, refusal banked, no threshold moved. Utilization against each session's own ceiling: 74.8% (this cell) vs 73.8% (PR-10's 131k against its banked 18.84 Gbit unshaped ceiling) — **the measured delta is small: ~2% warm-time (the fan-out knob's within-session A/B; the client-code delta itself was not separable from box variance, per the Gate A same-box old-vs-new control), ~+1 point of link utilization; the ~25% gap to wire remains, and its attribution (per the banked py-spy artifact: shard-straggler tail + engine-thread scatter-issue GIL contention during drains, NOT hashing, NOT pass-count) is the finding this session actually bought.** Scope disclosures: the 262k pair was DROPPED under the spend cap and its F4 branch never adjudicates; the session breached its own pre-registered ≤$15 cap at **$16.52** (wall-clock × rate; ~2 h of operator-side idle burn from a laptop sleep window — the cap discipline failed on operator availability, not on measurement scope, and is recorded as a miss). Two frozen-paragraph errors corrected here, same-commit convention: (1) PR-11's "all 131k arms share nic-128k's ENGINE_ARGS_SHA by construction" was wrong as written — the sha preimage embeds the per-arm stripped-model PATH, so every arm hashes differently BY CONSTRUCTION of the path; the pin identity the sentence meant is now proven stronger: all 9 session shas reproduce exactly from preimages identical in every field (fp8_e4m3, max-model-len 131472, max-num-seqs 1, gpu-mem-util 0.92, MNBT 8192, prefix-caching off, gen 16) except that path. (2) PR-11 quoted PR-10's ceiling as "19.21 sum_sent; 19.11 sum_received" — no banked file carries 19.21; the actual artifacts are 18.84 (unshaped, the 131k cell's applicable ceiling) and 19.12/19.11 (post-unshape). Hit attribution: per-rep exact-count gates GREEN on every warm rep; arm TOTALS are sums over per-pair prompts whose token counts wobble at the 16-token boundary (e.g. 24575 = 8192+8192+8191), so totals are not multiples of rep count — stated so nobody reads the ±1 as tampering.

---

*Changing any number above requires updating its section in the same commit
— a claim without its conditions and falsification line does not ship.*
