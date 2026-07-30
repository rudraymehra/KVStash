#!/usr/bin/env python3
"""Output-equivalence suite: a warm kvblockd reload must produce EXACTLY the
tokens a fresh recompute produces — byte-exact store, token-exact output.

Two phases around an engine restart (the same isolation pattern as
bench/rigs/hf-gpu/run_ttft.py — a warm hit is only believable when the
serving engine cannot have the KV anywhere local):

  --phase record    vLLM #1 (connector on, fresh engine): send every prompt
                    once with GREEDY decode (temperature=0, --seed pinned;
                    the rig serves with --max-num-seqs 1) and capture the
                    full generated token sequence. With --metrics set, each
                    prompt must grow kvblockd's put counters (the store
                    receipt) or the phase FAILS LOUDLY. Then the KERNEL-
                    DETERMINISM CONTROL (pre-registered amendment, CLAIMS.md
                    §6, written after failed run 6a6752f3c6272310d46cb761
                    and before any re-run): the SAME engine generates the
                    SAME prompt a second time back-to-back with the same
                    greedy params. If the engine disagrees with ITSELF, the
                    prompt is stamped kernel_deterministic=false — a replay
                    the engine cannot reproduce can neither indict nor
                    acquit the store. Prompts + token sequences + the
                    determinism stamps persist to --state.

  (the harness restarts vLLM here: engine-local KV dies with the process;
   kvblockd keeps the blocks)

  --phase compare   vLLM #2 (fresh engine): send the EXACT recorded prompts
                    with the same greedy settings. The prefix KV now comes
                    from kvblockd (with --metrics, each prompt's
                    kvb_hits_total delta must equal its expected block count
                    — the run_ttft attribution rule). Every generated token
                    sequence must EQUAL the recorded one; any mismatch is
                    reported with its first-divergence index. The --min-match
                    gate judges ONLY the kernel-deterministic prompts (the
                    gated set): kernel-nondeterministic prompts are still
                    replayed and their outcome DISCLOSED in the summary
                    (n_kernel_nondet + per-prompt divergences) but can
                    neither fail NOR pass the store gate. A run where EVERY
                    prompt is kernel-nondeterministic FAILS loudly — the
                    gate judged nothing, so nothing was certified.

Prompt lengths are adversarial by construction: for every --boundary-multiples
entry m, prompts at exactly m*block_size - 1 / m*block_size / m*block_size + 1
tokens (the connector stores only the complete blocks of the first n-1 tokens,
so these straddle the store/recompute boundary), plus any --lengths extras.
"Exactly" is enforced: build_prompt pads/trims until the server's /tokenize
returns the target to the token (a tolerance as wide as one calibration unit
would collapse the boundary trio onto one length), or the record phase fails.
--n prompts (default 20 — the free CPU rig's budget; GPU runs pass --n 100)
round-robin across that length list, each with a unique nonce at token 0.

Exit codes: 0 all gated prompts matched and (with --metrics) all prompts
warm-attributed; 1 gated match rate below --min-match (default 100 — for a
byte-exact store ANY mismatch on a prompt the engine can replay is a
finding, not noise) OR every prompt was kernel-nondeterministic (nothing
certified); 3 tokens all matched but the proof is
weaker than claimed — some compare prompts could not be attributed to
kvblockd (equivalence proven only against recompute, not against the store)
or were compared at text level only (the server elided logprobs.tokens).
2 = phase-level failure (severed store path, bad state, prompt calibration
could not land exactly, or a cross-dtype compare — the state records the
engine kv-cache dtype and compare REFUSES a mismatch: an fp8 arm's
equivalence claim is fp8-vs-fp8, never fp8-vs-bf16, and every record is
labeled with that scope).

`--selftest` proves the exact-landing calibration, the equality gate, the
first-divergence report, the store receipt, the attribution rule, the
text-fallback accounting, and the kernel-determinism control (a stub prompt
that answers differently back-to-back is caught, excluded from the gate and
disclosed; an all-nondeterministic run fails loudly) against an in-process
stub with driver-controlled counters (same style as run_ttft.py) — run it
before trusting a green run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
import uuid

UNIT = "The quick brown fox jumps over the lazy dog. "
FILLER = " and"   # single-token pad for exact landing (verified against the
                  # server's /tokenize on every adjustment, never assumed)
PRINT_PREFIX = "EQUIVJSONL"


class StorePathError(RuntimeError):
    """The store path engine -> connector -> kvblockd is severed; comparing
    would prove recompute==recompute, which is theater."""


class PromptLandingError(RuntimeError):
    """build_prompt could not make /tokenize return EXACTLY the target — the
    record phase must fail rather than mislabel a boundary cell."""


# ------------------------------------------------------------------ HTTP bits
# (shape shared with bench/rigs/hf-gpu/run_ttft.py — its --selftest covers the
# same parse rules, incl. the dir="in" filter, against the same metric names)

def _post_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def tokenize_count(vllm: str, model: str, prompt: str) -> int:
    return int(_post_json(vllm + "/tokenize", {"model": model, "prompt": prompt})["count"])


def read_counters(metrics_url: str) -> dict:
    with urllib.request.urlopen(metrics_url + "/metrics", timeout=10) as r:
        text = r.read().decode()
    sums = {"hits": 0.0, "put_bytes": 0.0, "blocks": 0.0}
    for line in text.splitlines():
        if line.startswith("kvb_hits_total"):
            sums["hits"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("kvb_bytes_total") and 'dir="in"' in line:
            sums["put_bytes"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("kvb_blocks"):
            sums["blocks"] += float(line.rsplit(" ", 1)[1])
    return sums


def wait_counter_growth(metrics_url: str, before: dict, keys: tuple,
                        timeout_s: float, poll_s: float = 0.25):
    deadline = time.monotonic() + timeout_s
    while True:
        now = read_counters(metrics_url)
        if any(now[k] > before[k] for k in keys):
            return now
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_s)


# -------------------------------------------------------------- greedy decode

def complete_greedy(vllm: str, model: str, prompt: str, gen_tokens: int,
                    seed: int, timeout_s: float, want_margins: bool = False,
                    margin_horizon: int | None = None) -> dict:
    """One NON-streaming greedy completion; returns the generated token
    sequence. logprobs=0 makes vLLM return the chosen token strings
    (choices[0].logprobs.tokens) — token-level equality, not just text.

    want_margins asks for the top-2 alternatives instead, so the caller can
    compute each step's top1-vs-top2 logprob gap. A step with a tiny gap is a
    near-tie the ENGINE may resolve differently across restarts under fp8;
    the margin lets the harness say so up front instead of discovering it as
    an unexplained mismatch."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": gen_tokens,
        "temperature": 0.0,
        "seed": seed,
        "logprobs": 2 if want_margins else 0,
    }
    resp = _post_json(vllm + "/v1/completions", payload, timeout=timeout_s)
    ch = (resp.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    out = {
        "text": ch.get("text", ""),
        "tokens": lp.get("tokens"),  # None if the server elided logprobs
        "usage": resp.get("usage") or {},
    }
    if want_margins:
        out["min_margin"] = _min_margin(lp, margin_horizon)
    return out


def _min_margin(lp: dict, horizon: int | None = None):
    """Smallest top1-top2 logprob gap over the generated steps, or None when
    the server did not return alternatives. Large gap = the argmax is robust
    to numerical noise; near zero = a coin-flip the engine may re-decide."""
    chosen = lp.get("token_logprobs") or []
    tops = lp.get("top_logprobs") or []
    if horizon:
        chosen, tops = chosen[:horizon], tops[:horizon]
    gaps = []
    for i, ch_lp in enumerate(chosen):
        if ch_lp is None or i >= len(tops) or not tops[i]:
            continue
        alts = sorted((v for v in tops[i].values() if v is not None), reverse=True)
        if len(alts) >= 2:
            gaps.append(abs(alts[0] - alts[1]))
    return min(gaps) if gaps else None


# ------------------------------------------------------------- prompt build
# (nonce FIRST, same as run_ttft.build_prompt, so every connector block key
# differs between prompts; count from the server's own /tokenize, never
# assumed. UNLIKE run_ttft, the landing must be EXACT: the boundary targets
# m*B-1 / m*B / m*B+1 differ by single tokens, and a tolerance as wide as one
# UNIT (~11 tokens) collapses all three onto one physical length — the suite
# would test the same cell thrice and call it a boundary sweep.)

def build_prompt(vllm: str, model: str, target_tokens: int, nonce: str) -> tuple[str, int]:
    """Coarse UNIT calibration, then trim whole UNITs / pad 1-token FILLERs
    until /tokenize returns EXACTLY target_tokens; raises PromptLandingError
    otherwise (the record phase fails rather than mislabel a cell)."""
    head = f"kvstash-equiv {nonce} :: "
    base = tokenize_count(vllm, model, head)
    probe = tokenize_count(vllm, model, head + UNIT * 128)
    unit_toks = max((probe - base) / 128.0, 0.5)
    k = max(0, round((target_tokens - base) / unit_toks))
    fill = 0
    n = base
    for _ in range(24):
        prompt = head + UNIT * k + FILLER * fill
        n = tokenize_count(vllm, model, prompt)
        err = target_tokens - n
        if err == 0:
            return prompt, n
        if err > 0:                        # undershoot
            if err >= unit_toks:           # whole units while they fit...
                k += int(err // unit_toks)
            else:                          # ...then single-token fillers
                fill += err
        elif fill >= -err:                 # overshoot: peel fillers first
            fill += err
        elif fill:
            fill = 0
        else:                              # no fillers left: drop whole units
            k = max(0, k - max(1, math.ceil(-err / unit_toks)))
    raise PromptLandingError(
        f"could not land exactly on {target_tokens} tokens (got {n}; "
        f"unit={unit_toks:.2f} tok, k={k}, fill={fill}) — the boundary cells "
        "are only meaningful at their exact lengths")


# High-margin tail sentences for corpus-mode length calibration. Each is a
# DISTINCT factual statement with an unambiguous continuation, so padding to a
# boundary length does not reintroduce the near-tie problem a repeated pangram
# creates (that repetition was the measured cause of the fp8 replay failures).
CORPUS_TAIL = [
    "Water freezes at zero degrees Celsius. ",
    "A triangle has exactly three sides. ",
    "The number seven is a prime number. ",
    "There are sixty seconds in one minute. ",
    "The letter B follows the letter A. ",
    "A square has four equal sides. ",
    "Ten plus ten equals twenty. ",
    "The sun rises in the east. ",
]
CORPUS_FILLER = " and"   # single-token pad (same token as the legacy builder)


def build_prompt_corpus(vllm: str, model: str, target_tokens: int, nonce: str,
                        entry: str) -> tuple[str, int]:
    """Calibrate to EXACTLY target_tokens with the high-margin corpus entry as
    the LAST thing in the prompt.

    Two properties matter and they pull against each other:
      * exact landing — the boundary trio m*B-1 / m*B / m*B+1 differs by single
        tokens, so a tolerance as wide as one unit collapses all three onto one
        physical length (the sweep would measure one cell thrice);
      * a high-margin continuation — whatever the model reads LAST determines
        how tied its next-token choice is, which is the whole point of the
        corpus.
    So the padding goes in the MIDDLE: nonce header, then distinct factual
    filler sentences to reach the length, then the corpus entry. A dangling
    " and"-style pad at the end would invite exactly the near-tie the corpus
    exists to avoid."""
    head = f"kvstash-equiv {nonce} :: "
    tail = " " + entry
    fixed = tokenize_count(vllm, model, head + tail)
    if fixed > target_tokens:
        raise PromptLandingError(
            f"header+entry is already {fixed} tokens > target {target_tokens}: "
            "shorten the corpus entry or raise the boundary multiples")
    unit = "".join(CORPUS_TAIL)
    probe = tokenize_count(vllm, model, head + unit * 8 + tail)
    unit_toks = max((probe - fixed) / 8.0, 0.5)
    k = max(0, round((target_tokens - fixed) / unit_toks))
    fill, n = 0, fixed
    for _ in range(28):
        prompt = head + unit * k + CORPUS_FILLER * fill + tail
        n = tokenize_count(vllm, model, prompt)
        err = target_tokens - n
        if err == 0:
            return prompt, n
        if err > 0:
            if err >= unit_toks:
                k += int(err // unit_toks)
            else:
                fill += err
        elif fill >= -err:
            fill += err
        elif fill:
            fill = 0
        else:
            k = max(0, k - max(1, math.ceil(-err / unit_toks)))
    raise PromptLandingError(
        f"corpus mode could not land exactly on {target_tokens} tokens "
        f"(got {n}) — the boundary cells are only meaningful at exact lengths")


def expected_hit_blocks(prompt_tokens: int, block_size: int) -> int:
    """The connector's alignment rule (run_ttft.expected_hit_blocks): only
    ((n-1)//B)*B tokens of an n-token prompt are loadable."""
    return (prompt_tokens - 1) // block_size * block_size // block_size


def equivalence_scope(kv_cache_dtype: str) -> str:
    """The greppable disclosure label stamped into every record: WHAT was
    proven equal to what. Both phases run one engine config (the harness
    feeds the same --kv-cache-dtype to record and compare, and the compare
    phase refuses a mismatch), so the claim is always same-dtype. For fp8
    engines it is spelled out as fp8-vs-fp8 — the fp8 arm's token-identity
    story is fp8 reload vs fp8 recompute, NEVER fp8 vs bf16 recompute."""
    if kv_cache_dtype.startswith("fp8"):
        return (f"fp8-vs-fp8 (both phases kv_cache_dtype={kv_cache_dtype}; "
                "never fp8-vs-bf16)")
    return f"same-dtype ({kv_cache_dtype}-vs-{kv_cache_dtype})"


def boundary_lengths(block_size: int, multiples: list[int],
                     extra: list[int]) -> list[int]:
    """Adversarial targets: m*B-1 / m*B / m*B+1 for each multiple, plus the
    extras — deduped, order kept, and any length that stores ZERO complete
    blocks dropped (it would compare recompute to recompute)."""
    out = []
    for m in multiples:
        out.extend((m * block_size - 1, m * block_size, m * block_size + 1))
    out.extend(extra)
    seen, keep, dropped = set(), [], []
    for length in out:
        if length in seen:
            continue
        seen.add(length)
        if expected_hit_blocks(length, block_size) >= 1:
            keep.append(length)
        else:
            dropped.append(length)
    if dropped:
        print(f"[equiv] dropped lengths {dropped}: <= {block_size + 1} tokens "
              "stores zero complete blocks (nothing external to compare against)",
              flush=True)
    return keep


def first_divergence(a: list, b: list):
    """Index of the first differing element, or None when equal (length
    mismatch diverges at the shorter length)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _write_text(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def load_prompt_corpus(path: str) -> tuple[list[str], str]:
    """Frozen prompt corpus + its sha256. The digest is the pre-registration:
    it lands in every record, so a corpus edited after a measurement is
    detectable by anyone re-deriving the claim."""
    with open(path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    prompts = [ln.strip() for ln in raw.decode().splitlines()
               if ln.strip() and not ln.lstrip().startswith("#")]
    if not prompts:
        raise ValueError(f"prompt corpus {path} has no prompts")
    return prompts, digest


# ------------------------------------------------------ phase: margin-calibrate

def run_calibrate(args) -> int:
    """Measure each CANDIDATE prompt's greedy stability on this engine and
    print one CALIBJSONL line per prompt.

    Why this phase exists, and why it is not cherry-picking: the frozen corpus
    must contain prompts whose argmax is robust to fp8 numerical noise, and the
    honest way to know that is to MEASURE it rather than assume it (the first
    corpus was written from intuition; a 16-prompt run against it found 10 of 16
    near-ties on the target model). This phase looks at exactly two engine
    properties — the top1-top2 logprob gap over the frozen horizon, and whether
    the engine reproduces itself back-to-back — and never at whether a reload
    matched. It cannot select for a passing store gate because it never runs
    one: no store, no restart, no comparison. The selected set is then frozen
    and checksummed before any certification run, and the calibration output is
    committed beside it so the selection is auditable."""
    candidates, sha = load_prompt_corpus(args.prompt_set)
    print(f"[calibrate] {len(candidates)} candidates from {args.prompt_set} "
          f"(sha256={sha}), horizon={args.margin_horizon}, "
          f"gen_tokens={args.gen_tokens}, dtype={args.kv_cache_dtype}", flush=True)
    kept = 0
    for i, entry in enumerate(candidates):
        nonce = f"cal{i:03d}-{uuid.uuid4().hex[:8]}"
        prompt = f"kvstash-equiv {nonce} :: {entry}"
        try:
            a = complete_greedy(args.vllm, args.model, prompt, args.gen_tokens,
                                args.seed, args.request_timeout,
                                want_margins=True,
                                margin_horizon=args.margin_horizon)
            b = complete_greedy(args.vllm, args.model, prompt, args.gen_tokens,
                                args.seed, args.request_timeout)
        except Exception as e:                     # noqa: BLE001 - report, continue
            print(f"[calibrate] i={i} ERROR {e}", flush=True)
            continue
        if a["tokens"] is not None and b["tokens"] is not None:
            div = first_divergence(a["tokens"], b["tokens"])
        else:
            div = first_divergence(list(a["text"]), list(b["text"]))
        rec = {"schema_version": 1, "kind": "margin-calibration", "i": i,
               "candidate_sha256": sha, "prompt_text": entry,
               "min_margin": a.get("min_margin"),
               "self_consistent": div is None,
               "self_divergence_index": div,
               "kv_cache_dtype": args.kv_cache_dtype,
               "margin_horizon": args.margin_horizon,
               "gen_tokens": args.gen_tokens,
               "model": args.model}
        eligible = (rec["self_consistent"] and rec["min_margin"] is not None
                    and rec["min_margin"] >= args.margin_floor)
        rec["eligible"] = eligible
        kept += int(eligible)
        print("CALIBJSONL " + json.dumps(rec, sort_keys=True), flush=True)
        print(f"[calibrate] i={i} margin={rec['min_margin']} "
              f"self_consistent={rec['self_consistent']} eligible={eligible}",
              flush=True)
    print(f"[calibrate] {kept}/{len(candidates)} candidates are eligible "
          f"(self-consistent AND margin >= {args.margin_floor})", flush=True)
    if kept == 0:
        print("FATAL(calibrate): no candidate is both self-consistent and above "
              "the margin floor on this engine — certification cannot be made to "
              "work by prompt selection here; that is the finding.",
              file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------- phase: record

def run_prescreen(args) -> int:
    """Store-blind margin screen of the PADDED certification prompts.

    Builds exactly the prompts run_record would build (same boundary trio,
    same corpus cycling, same length-calibrated padding — but fresh nonces,
    so no cache-key overlap with a later record phase) and measures, per
    prompt, the horizon margin and back-to-back self-consistency on THIS
    engine. Raw-candidate margins provably do not transfer to padded prompts
    (measured on the certified 131k run: entries at 4.50/3.75 nats raw
    collapsed to 0.125-0.75 once padded to depth), so only a padded screen
    predicts whether record/compare will have anything to judge.

    Two pre-registered uses (CLAIMS.md PR-10):
      1. Refuse a doomed arm BEFORE the paid populate: exit 1 iff ZERO padded
         prompts are eligible (self-consistent AND >= floor). Any other count
         exits 0 — this phase never publishes, it only stops obvious waste.
      2. The mechanism pair: this same phase run on a connector-attached
         engine (store in loop) vs on a baseline engine (store-free), same
         box, same checkpoint, same construction — banked PRESCREENJSONL on
         both sides isolates store-in-loop for the padded construction.

    SELECTION INDEPENDENCE: this phase never chooses what the gate judges.
    Record/compare re-measure margins on their own engines and gate on those;
    a prescreen can only stop a run entirely, never steer its verdict.
    """
    lengths = boundary_lengths(args.block_size, _ints(args.boundary_multiples),
                               _ints(args.lengths))
    if not lengths:
        print("FATAL(prescreen): no usable lengths after the zero-block filter",
              file=sys.stderr)
        return 2
    corpus, corpus_sha = load_prompt_corpus(args.prompt_set)
    print(f"[prescreen] {len(corpus)} corpus entries from {args.prompt_set} "
          f"(sha256={corpus_sha}) x lengths {lengths}, n={args.n}, "
          f"floor={args.margin_floor:.2f} nats, horizon={args.margin_horizon}, "
          f"dtype={args.kv_cache_dtype}", flush=True)
    eligible = 0
    for i in range(args.n):
        target = lengths[i % len(lengths)]
        nonce = f"pre{uuid.uuid4().hex[:12]}-L{target}-e{i}"
        prompt, ntok = build_prompt_corpus(args.vllm, args.model, target,
                                           nonce, corpus[i % len(corpus)])
        a = complete_greedy(args.vllm, args.model, prompt, args.gen_tokens,
                            args.seed, args.request_timeout,
                            want_margins=True,
                            margin_horizon=args.margin_horizon)
        b = complete_greedy(args.vllm, args.model, prompt, args.gen_tokens,
                            args.seed, args.request_timeout)
        if a["tokens"] is not None and b["tokens"] is not None:
            div = first_divergence(a["tokens"], b["tokens"])
        else:
            div = first_divergence(list(a["text"]), list(b["text"]))
        margin = a.get("min_margin")
        ok = bool(div is None and margin is not None
                  and margin >= args.margin_floor)
        eligible += ok
        rec = {"schema_version": 1, "kind": "prescreen", "i": i,
               "target_tokens": target,
               "prompt_tokens": int(a["usage"].get("prompt_tokens", ntok)),
               "entry_text": corpus[i % len(corpus)][:80],
               "prompt_corpus": args.prompt_set,
               "prompt_corpus_sha256": corpus_sha,
               "min_margin": margin,
               "self_consistent": div is None,
               "first_divergence_index": div,
               "margin_floor": args.margin_floor,
               "margin_horizon": args.margin_horizon,
               "gen_tokens": args.gen_tokens,
               "kv_cache_dtype": args.kv_cache_dtype,
               "eligible": ok}
        print("PRESCREENJSONL " + json.dumps(rec, sort_keys=True), flush=True)
        print(f"[prescreen] i={i} target={target} "
              f"margin={margin if margin is None else round(margin, 3)} "
              f"self_consistent={div is None} eligible={ok}", flush=True)
    print(f"[prescreen] {eligible}/{args.n} padded prompts eligible on this "
          "engine", flush=True)
    if eligible == 0:
        print("FATAL(prescreen): ZERO padded prompts are gate-eligible on this "
              "engine — a record/compare pass would certify nothing; refusing "
              "before the paid populate.", file=sys.stderr)
        return 1
    return 0


def run_record(args) -> int:
    lengths = boundary_lengths(args.block_size, _ints(args.boundary_multiples),
                               _ints(args.lengths))
    if not lengths:
        print("FATAL(record): no usable lengths after the zero-block filter",
              file=sys.stderr)
        return 2
    max_blocks_covered = 0
    corpus, corpus_sha = ([], "")
    if args.prompt_set:
        corpus, corpus_sha = load_prompt_corpus(args.prompt_set)
        print(f"[record] FROZEN prompt corpus {args.prompt_set} "
              f"({len(corpus)} prompts, sha256={corpus_sha}), "
              f"margin floor {args.margin_floor:.2f} nats", flush=True)
    # Check the floor against the CONFIGURED lengths before issuing a single
    # completion: a doomed set must refuse at t=0, not after paid GPU work.
    if corpus:
        planned = max(expected_hit_blocks(L, args.block_size) for L in lengths)
        if planned < args.min_blocks_covered:
            print(f"FATAL(record): configured lengths {lengths} can reach at most "
                  f"{planned} blocks < --min-blocks-covered "
                  f"{args.min_blocks_covered}; certification would not exercise the "
                  f"block-index path. Raise --boundary-multiples/--lengths (or lower "
                  f"the floor DELIBERATELY and disclose it). Refusing before any "
                  f"completion is issued.", file=sys.stderr)
            return 2
        print(f"[record] planned block coverage: {planned} blocks vs floor "
              f"{args.min_blocks_covered} — OK before spending", flush=True)
    print(f"[record] n={args.n} prompts over lengths {lengths} "
          f"(block_size={args.block_size}, seed={args.seed}, "
          f"gen_tokens={args.gen_tokens}, kv_cache_dtype={args.kv_cache_dtype})",
          flush=True)
    state = {
        "state_schema": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "seed": args.seed,
        "gen_tokens": args.gen_tokens,
        "block_size": args.block_size,
        # The engine KV dtype this state was recorded under: the compare
        # phase REFUSES a mismatch (fp8-vs-bf16 is not an equivalence claim).
        "kv_cache_dtype": args.kv_cache_dtype,
        "prompt_corpus": args.prompt_set or "(repeated-filler builder)",
        "prompt_corpus_sha256": corpus_sha,
        "margin_floor": args.margin_floor if corpus else None,
        "margin_horizon": args.margin_horizon if corpus else None,
        "entries": [],
    }
    for i in range(args.n):
        target = lengths[i % len(lengths)]
        nonce = f"{uuid.uuid4().hex[:12]}-L{target}-e{i}"
        if corpus:
            # Frozen corpus entry, nonce-prefixed so every prompt mints its own
            # cache keys — then LENGTH-CALIBRATED onto the same m*B-1/m*B/m*B+1
            # boundary trio the legacy builder used. Coverage is not optional:
            # certification must exercise the block-index path to depth, or a
            # fault in block 5 of a reload is simply never tested.
            prompt, ntok = build_prompt_corpus(args.vllm, args.model, target,
                                               nonce, corpus[i % len(corpus)])
        else:
            prompt, ntok = build_prompt(args.vllm, args.model, target, nonce)
        c0 = read_counters(args.metrics) if args.metrics else None
        r = complete_greedy(args.vllm, args.model, prompt, args.gen_tokens,
                            args.seed, args.request_timeout,
                            want_margins=bool(corpus),
                            margin_horizon=args.margin_horizon)
        if args.metrics:
            after = wait_counter_growth(args.metrics, c0, ("put_bytes", "blocks"),
                                        args.put_wait_s)
            if after is None:
                raise StorePathError(
                    f"kvblockd received NOTHING for prompt {i} (target {target} "
                    f"tokens): put counters flat for {args.put_wait_s:.0f}s. The "
                    "store path is severed — a compare phase now would prove "
                    "recompute==recompute. Aborting.")
        if r["tokens"] is None:
            print(f"[record] WARN: prompt {i}: server returned no logprobs.tokens "
                  "— falling back to text-level equality for this prompt", flush=True)
        # KERNEL-DETERMINISM CONTROL: the SAME engine, the SAME prompt, the
        # same greedy params, back-to-back. If the two generations differ the
        # engine cannot replay its own output — blaming (or crediting) the
        # store for a later mismatch on this prompt would be attribution
        # theater, so it is stamped and EXCLUDED from the store gate, always
        # disclosed. Honest mechanics note: this engine serves with the
        # connector on, so the control rep's prefix may be loaded from the
        # store the first rep just wrote — those bytes are xxh3-verified
        # identical to what this same engine produced moments earlier, so a
        # divergence still means the engine disagreed with itself over
        # bit-identical KV. The control can only EXCLUDE a prompt (stamped,
        # disclosed); it can never convert a gated mismatch into a pass.
        r2 = complete_greedy(args.vllm, args.model, prompt, args.gen_tokens,
                             args.seed, args.request_timeout)
        if r["tokens"] is not None and r2["tokens"] is not None:
            kdiv = first_divergence(r["tokens"], r2["tokens"])
        else:
            kdiv = first_divergence(list(r["text"]), list(r2["text"]))
        kernel_det = kdiv is None
        if not kernel_det:
            print(f"[record] KERNEL-NONDET i={i} target={target}: the engine "
                  f"disagreed with itself back-to-back (first divergence at "
                  f"token {kdiv}) — excluded from the store gate, disclosed",
                  flush=True)
        ptoks = int(r["usage"].get("prompt_tokens", ntok))
        blocks_covered = expected_hit_blocks(ptoks, args.block_size)
        max_blocks_covered = max(max_blocks_covered, blocks_covered)
        margin = r.get("min_margin")
        low_margin = (corpus and margin is not None
                      and margin < args.margin_floor)
        if low_margin:
            print(f"[record] LOW-MARGIN i={i}: smallest top1-top2 gap "
                  f"{margin:.3f} < {args.margin_floor:.2f} nats — a near-tie the "
                  f"engine may re-decide across a restart; excluded from the "
                  f"store gate, disclosed", flush=True)
        state["entries"].append({
            "i": i, "target_tokens": target, "nonce": nonce, "prompt": prompt,
            "prompt_tokens": ptoks, "tokens": r["tokens"], "text": r["text"],
            "kernel_deterministic": kernel_det,
            "record_divergence_index": kdiv,
            "min_margin": margin,
            "low_margin": bool(low_margin),
        })
        print(f"[record] i={i} target={target} tokens={ptoks} "
              f"gen={len(r['tokens']) if r['tokens'] is not None else '?'} "
              f"kernel_deterministic={kernel_det}", flush=True)
    # COVERAGE FLOOR (pre-registered): the deepest prompt must reach at least
    # this many blocks, or certification never touches the block-index path
    # beyond the first block or two — a store fault at block 5 would pass
    # unnoticed. Refuse rather than certify a set that cannot discriminate.
    if corpus and max_blocks_covered < args.min_blocks_covered:
        print(f"FATAL(record): deepest prompt covers {max_blocks_covered} blocks "
              f"< --min-blocks-covered {args.min_blocks_covered}: certification "
              f"would not exercise the block-index path to depth. Raise "
              f"--boundary-multiples/--lengths (or lower the floor DELIBERATELY "
              f"and disclose it).", file=sys.stderr)
        return 2
    state["max_blocks_covered"] = max_blocks_covered
    state["min_blocks_covered"] = args.min_blocks_covered
    print(f"[record] block coverage: deepest prompt spans {max_blocks_covered} "
          f"blocks (floor {args.min_blocks_covered})", flush=True)
    tmp = args.state + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, args.state)
    print(f"[record] state -> {args.state} ({len(state['entries'])} prompts)", flush=True)
    return 0


# -------------------------------------------------------------- phase: compare

def run_compare(args, stamp: dict) -> int:
    try:
        with open(args.state) as f:
            state = json.load(f)
    except (OSError, ValueError) as e:
        print(f"FATAL(compare): cannot read --state {args.state}: {e}", file=sys.stderr)
        return 2
    if state.get("model") != args.model:
        print(f"FATAL(compare): state recorded model {state.get('model')!r}, "
              f"compare asked for {args.model!r}", file=sys.stderr)
        return 2
    # Dtype coherence gate (fp8 disclosure rule): the recorded tokens and the
    # replayed tokens must come from the SAME engine KV dtype. A cross-dtype
    # compare would either (a) mismatch and mislabel quantization drift as
    # store corruption, or worse (b) match and be quotable as "fp8 outputs
    # equal bf16 recompute" — a claim this suite must be UNABLE to produce.
    # States predating the field are the auto-bf16 era (rig convention).
    state_dtype = state.get("kv_cache_dtype", "auto-bf16")
    if state_dtype != args.kv_cache_dtype:
        print(f"FATAL(compare): state was recorded under kv_cache_dtype="
              f"{state_dtype!r} but compare is running under "
              f"{args.kv_cache_dtype!r} — cross-dtype token equivalence "
              "(fp8-vs-bf16) is not the claim this suite makes; the fp8 story "
              "is fp8-vs-fp8: rerun both phases under one engine dtype.",
              file=sys.stderr)
        return 2
    scope = equivalence_scope(args.kv_cache_dtype)
    seed, gen_tokens = state["seed"], state["gen_tokens"]
    block_size = state["block_size"]
    entries = state["entries"]
    print(f"[compare] {len(entries)} prompts from state created {state.get('created')} "
          f"(seed={seed}, gen_tokens={gen_tokens}, block_size={block_size})", flush=True)

    out_lines = []

    def emit(rec: dict):
        line = json.dumps(rec, sort_keys=True)
        out_lines.append(line)
        print(f"{args.print_prefix} {line}", flush=True)

    matched = 0
    n_gated = 0
    # Depth the GATE actually judged, distinct from depth the corpus was built
    # to: padding provably collapses margins (measured: the certified 131k run
    # excluded every 15-16-block prompt as a near-tie at 0.25-0.75 nats and
    # gated only 3-4-block ones), so the built depth can satisfy the coverage
    # floor while the gated set stays shallow. The certification string discloses both numbers; the
    # per-cell BYTE proof to full depth remains the exact-count hit gate plus
    # per-block xxh3 verify, not this token gate.
    gated_max_blocks = 0
    mismatches, unwarmed, text_fallback, kernel_nondet = [], [], [], []
    n_low_margin, low_margin_detail = 0, []
    for e in entries:
        # KERNEL-DETERMINISM CONTROL stamp from the record phase: a prompt
        # the record engine could not replay against ITSELF is excluded from
        # the store gate (it can neither fail nor pass it) and disclosed.
        # States predating the control carry no stamp — those prompts stay
        # gated (the pre-control behavior; the gate never loosens by default).
        kernel_det = bool(e.get("kernel_deterministic", True))
        # Pre-registered NARROWING (never a loosening): a prompt whose
        # smallest top1-top2 gap sits under the frozen margin floor is a
        # near-tie the ENGINE may re-decide across the restart, so it cannot
        # discriminate a store fault from numerical noise. It is excluded from
        # the gate and disclosed — it can never turn a mismatch into a pass.
        # Re-DERIVED here, never trusted from the state file: a hand-edited
        # flag could otherwise exclude an inconvenient prompt. A stored flag
        # that contradicts the arithmetic aborts the compare (rc 2).
        floor = state.get("margin_floor")
        mm = e.get("min_margin")
        recorded_low = bool(floor is not None and mm is not None and mm < floor)
        if bool(e.get("low_margin", False)) != recorded_low:
            print(f"FATAL(compare): state entry {e['i']} claims "
                  f"low_margin={e.get('low_margin')} but min_margin={mm} vs floor "
                  f"{floor} says {recorded_low} — the state file has been edited",
                  file=sys.stderr)
            return 2
        c0 = read_counters(args.metrics) if args.metrics else None
        r = complete_greedy(args.vllm, args.model, e["prompt"], gen_tokens,
                            seed, args.request_timeout,
                            want_margins=floor is not None,
                            margin_horizon=state.get("margin_horizon"))
        # The state file is data on disk and therefore forgeable; the engine is
        # not. An exclusion stands only if the REPLAY's own margin is also under
        # the frozen floor — so editing min_margin in the state cannot launder a
        # real divergence into a disclosed "near-tie".
        replay_mm = r.get("min_margin")
        low_margin = bool(recorded_low and floor is not None
                          and replay_mm is not None and replay_mm < floor)
        if recorded_low and not low_margin:
            print(f"  MARGIN RE-MEASURED i={e['i']}: state said near-tie "
                  f"(min_margin={mm}) but the replay measures {replay_mm} >= floor "
                  f"{floor} — the exclusion is REFUSED and the prompt is GATED",
                  flush=True)
        gated = kernel_det and not low_margin
        rec = {"schema_version": 1, "kind": "equivalence", "i": e["i"],
               "target_tokens": e["target_tokens"],
               "prompt_tokens": e["prompt_tokens"]}
        rec.update(stamp)
        # AFTER the stamp: these are checked against the state above, so a
        # harness stamp can never overwrite them with a prettier story.
        rec["kv_cache_dtype"] = args.kv_cache_dtype
        rec["equivalence_scope"] = scope
        rec["kernel_deterministic"] = kernel_det
        rec["min_margin"] = e.get("min_margin")
        rec["low_margin"] = low_margin
        rec["gated"] = gated
        if not kernel_det:
            rec["record_divergence_index"] = e.get("record_divergence_index")
        # equality: token-level when both sides have token lists, else text
        if e["tokens"] is not None and r["tokens"] is not None:
            div = first_divergence(e["tokens"], r["tokens"])
            rec["compare_level"] = "token"
        else:
            div = first_divergence(list(e["text"]), list(r["text"]))
            rec["compare_level"] = "text (logprobs unavailable)"
            text_fallback.append(e["i"])
        rec["match"] = div is None
        rec["first_divergence_index"] = div
        # STRICT PARTITION (CLAIMS fp8 amendment 2 item 5): a prompt can be
        # excluded for exactly ONE reported reason. Kernel-nondeterminism wins
        # — it is the stronger statement about the engine — so low-margin is
        # counted only for prompts that were otherwise gate-eligible. Without
        # this precedence a prompt that is both (the COMMON fp8 case) is
        # counted twice and the run dies at the rig's receipt arithmetic after
        # the paid work is done.
        if low_margin and kernel_det:
            n_low_margin += 1
            low_margin_detail.append({"i": e["i"], "min_margin": e.get("min_margin"),
                                      "replay_match": div is None,
                                      "replay_first_divergence_index": div})
            mm = e.get("min_margin")
            print(f"  LOW-MARGIN i={e['i']}: smallest top1-top2 gap "
                  f"{mm if mm is None else round(mm, 3)} < floor "
                  f"{state.get('margin_floor')} nats — near-tie, excluded from the "
                  f"store gate; replay "
                  f"{'matched' if div is None else f'diverged at token {div}'}"
                  " — disclosed either way", flush=True)
        if gated:
            n_gated += 1
            gated_max_blocks = max(
                gated_max_blocks,
                expected_hit_blocks(e["prompt_tokens"], block_size))
            if div is None:
                matched += 1
            else:
                mismatches.append((e["i"], e["target_tokens"], div))
        elif not kernel_det:
            # kernel-nondeterministic ONLY: a low-margin prompt is a DIFFERENT
            # exclusion and must not be reported under this label (the two sets
            # are disjoint, and job.sh checks the partition arithmetic).
            kernel_nondet.append({
                "i": e["i"], "target_tokens": e["target_tokens"],
                "record_divergence_index": e.get("record_divergence_index"),
                "replay_match": div is None,
                "replay_first_divergence_index": div,
            })
        if div is not None:
            rec["recorded"] = e["tokens"] if e["tokens"] is not None else e["text"]
            rec["replayed"] = r["tokens"] if r["tokens"] is not None else r["text"]
        # attribution: this compare rep must have READ its prefix from kvblockd
        if args.metrics:
            c1 = read_counters(args.metrics)
            delta = c1["hits"] - c0["hits"]
            expected = expected_hit_blocks(e["prompt_tokens"], block_size)
            rec["kvb_hit_delta"] = delta
            rec["expected_hit_blocks"] = expected
            rec["warm_attributed"] = (expected >= 1 and delta == expected)
            if not rec["warm_attributed"]:
                unwarmed.append((e["i"], e["target_tokens"], delta, expected))
        emit(rec)
        print(f"[compare] i={e['i']} target={e['target_tokens']} "
              f"match={rec['match']}"
              + ("" if kernel_det else " KERNEL-NONDET(ungated)")
              + (f" div@{div}" if div is not None else "")
              + (f" hits+={rec['kvb_hit_delta']:.0f}/{rec['expected_hit_blocks']}"
                 if args.metrics else ""), flush=True)

    # The gate judges ONLY the gated (kernel-deterministic) set; the excluded
    # prompts are disclosed with both divergence indices (record control +
    # replay), never averaged in and never dropped silently.
    rate = matched / n_gated * 100.0 if n_gated else 0.0
    # `certification` is the quotable claim, and EQUIVJSONL lines outlive a
    # failed run (they are the designed retrieval path from job logs), so it
    # is computed jointly with EVERY exit condition below — the certified
    # wording may only ever persist in a record whose run exits 0. Precedence
    # mirrors the exit ladder: nothing-gated, then gated mismatch (rc 1),
    # then unattributed / text-fallback (rc 3), and only then certified.
    # The invariant the rig re-checks: every prompt lands in exactly one bucket.
    if n_gated + len(kernel_nondet) + n_low_margin != len(entries):
        print(f"FATAL(compare): exclusion partition broken — n_gated={n_gated} + "
              f"n_kernel_nondet={len(kernel_nondet)} + "
              f"n_low_margin={n_low_margin} != n={len(entries)}; refusing to emit "
              "a summary whose own arithmetic does not close", file=sys.stderr)
        return 2
    tripwire = bool(n_low_margin and low_margin_detail
                    and all(not d["replay_match"] for d in low_margin_detail))
    if tripwire:
        print("  TRIPWIRE: every low-margin prompt ALSO diverged on replay — "
              "coin-flips would not all land the same way, so this pattern is "
              "store-shaped. Treat the run as suspect and investigate before "
              "quoting any cell from it.", file=sys.stderr, flush=True)
    excl = ""
    if n_low_margin:
        excl = (f"; {n_low_margin} excluded as low-margin (< "
                f"{state.get('margin_floor')} nats top1-top2, disclosed)")
    if not n_gated:
        certification = (
            "NOTHING CERTIFIED: every prompt was excluded — kernel-"
            "nondeterministic (the record engine disagreed with itself "
            "back-to-back) or low-margin; the store gate judged zero prompts")
    elif matched < n_gated or rate < args.min_match:
        certification = (
            f"NOT CERTIFIED: {n_gated - matched} of {n_gated} "
            "kernel-deterministic prompts mismatched")
    elif unwarmed:
        certification = (
            f"NOT CERTIFIED: tokens matched but {len(unwarmed)} prompt(s) "
            "unattributed to the store (equivalence proven against "
            "recompute only)")
    elif text_fallback:
        certification = (
            f"NOT CERTIFIED: {len(text_fallback)} prompt(s) matched at TEXT "
            "level only (token identity unproven — the server elided "
            "logprobs.tokens)")
    else:
        certification = (
            f"token-identical on all gated prompts "
            f"({n_gated} of {len(entries)}; {len(kernel_nondet)} excluded as "
            f"kernel-nondeterministic{excl}, all disclosed)"
            # Depth honesty: the coverage floor is a property of the BUILT set;
            # exclusions can leave the gate judging only shallow prompts. Say
            # so in the quotable string itself — byte-correctness to full depth
            # is the hit gate + xxh3's claim, not this one's.
            f"; gated depth <= {gated_max_blocks} blocks "
            f"(deepest built: {state.get('max_blocks_covered')})")
    summary = {"schema_version": 1, "kind": "equivalence-summary",
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "model": args.model, "seed": seed, "gen_tokens": gen_tokens,
               "block_size": block_size, "n": len(entries),
               "n_gated": n_gated, "n_kernel_nondet": len(kernel_nondet),
               "kernel_nondet": kernel_nondet,
               # Frozen-corpus provenance + the pre-registered margin screen:
               # a reader can tell exactly which prompt set produced the claim
               # and how many prompts were excluded as engine-tie-prone.
               "prompt_corpus": state.get("prompt_corpus"),
               "prompt_corpus_sha256": state.get("prompt_corpus_sha256"),
               "margin_floor": state.get("margin_floor"),
               "margin_horizon": state.get("margin_horizon"),
               "max_blocks_covered": state.get("max_blocks_covered"),
               "min_blocks_covered": state.get("min_blocks_covered"),
               "gated_max_blocks": gated_max_blocks,
               "n_low_margin_excluded": n_low_margin,
               "tripwire_all_low_margin_diverged": tripwire,
               "low_margin": low_margin_detail,
               "matched": matched, "match_rate_pct": rate,
               "min_match_pct": args.min_match,
               "certification": certification,
               "mismatches": [{"i": i, "target_tokens": t, "first_divergence_index": d}
                              for i, t, d in mismatches],
               "warm_attributed_all": not unwarmed if args.metrics else None,
               "text_fallback_prompts": len(text_fallback),
               # Isolation is a property of the HARNESS that ran the two
               # phases (did an engine restart actually happen in between?);
               # this driver cannot see it, so it claims nothing — the harness
               # must stamp it (--stamp isolation=vllm-restart) or the summary
               # says "unverified".
               "isolation": "unverified"}
    summary.update(stamp)
    summary["kv_cache_dtype"] = args.kv_cache_dtype  # state-checked, post-stamp
    summary["equivalence_scope"] = scope
    emit(summary)
    if args.out:
        # One shot at the end; the marker lines above are the streaming
        # retrieval path either way.
        with open(args.out, "a") as f:
            f.write("\n".join(out_lines) + "\n")

    print(f"\n== equivalence: {matched}/{n_gated} gated matched "
          f"({rate:.1f}%, gate >= {args.min_match:.1f}%); "
          f"{len(kernel_nondet)} of {len(entries)} prompt(s) excluded as "
          "kernel-nondeterministic (disclosed, ungated) ==")
    for nd in kernel_nondet:
        print(f"  KERNEL-NONDET i={nd['i']} target={nd['target_tokens']}: record "
              f"control diverged at token {nd['record_divergence_index']}; replay "
              + ("matched the recording"
                 if nd["replay_match"]
                 else f"diverged at token {nd['replay_first_divergence_index']}")
              + " — disclosed, outside the store gate either way", file=sys.stderr)
    if entries and not n_gated:
        print(f"FAIL: every prompt was excluded — {len(kernel_nondet)} kernel-"
              f"nondeterministic (the record engine disagreed with itself "
              f"back-to-back), {n_low_margin} low-margin near-ties — so the "
              "store gate judged ZERO prompts and NOTHING was certified.",
              file=sys.stderr)
        return 1
    if mismatches:
        for i, t, d in mismatches:
            print(f"  MISMATCH i={i} target={t}: first divergence at token {d}",
                  file=sys.stderr)
    if rate < args.min_match:
        print(f"FAIL: gated match rate {rate:.1f}% below --min-match "
              f"{args.min_match:.1f}% — for a byte-exact store ANY mismatch on "
              "a prompt the engine can replay is a finding.", file=sys.stderr)
        return 1
    if unwarmed:
        for i, t, delta, exp in unwarmed:
            print(f"  UNATTRIBUTED i={i} target={t}: kvb_hits_total grew "
                  f"{delta:.0f}, expected {exp}", file=sys.stderr)
        print("FAIL: tokens matched but some prompts were not attributed to "
              "kvblockd — equivalence was proven against recompute, not against "
              "the store.", file=sys.stderr)
        return 3
    if text_fallback:
        print(f"FAIL: {len(text_fallback)} prompt(s) compared at TEXT level only "
              "(the server elided logprobs.tokens) — matching text is necessary "
              "but token-exact equivalence was not proven for prompts "
              f"{text_fallback}.", file=sys.stderr)
        return 3
    return 0


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


# ------------------------------------------------------------------- selftest

def _stub_server(ctl):
    """In-process stub: /tokenize, non-streaming /v1/completions with
    logprobs.tokens, kvblockd-shaped /metrics under `ctl` control.
    ctl["on_completion"]: "store" grows put counters; "hit" grows hits by the
    CORRECT expected block count for the request's prompt (a well-behaved
    connector); "hit-short" grows them by one block too few; "none" freezes
    them. ctl["flip_at"]: index of a generated token to corrupt (None = exact
    replay). ctl["elide_logprobs"]: drop logprobs.tokens from completions (a
    server that forces the text-level fallback). ctl["nondet_markers"]: any
    prompt containing one of these substrings answers DIFFERENTLY on every
    second call (per-prompt call counter in ctl["call_counts"]) — a kernel
    that cannot replay itself, for proving the determinism control. Token
    sequences are otherwise a deterministic function of the prompt, so record
    and compare agree unless flip_at or a marker interferes."""
    import hashlib
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    def gen_tokens_for(prompt: str, n: int) -> list[str]:
        h = hashlib.sha256(prompt.encode()).hexdigest()
        return [f"tok-{h[(2 * i) % 60:(2 * i) % 60 + 2]}-{i}" for i in range(n)]

    class Stub(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/metrics":
                body = (
                    f'kvb_hits_total{{tier="dram",ns="1"}} {ctl["hits"]}\n'
                    f'kvb_bytes_total{{dir="in",ns="1"}} {ctl["put_bytes"]}\n'
                    f'kvb_bytes_total{{dir="out",ns="1"}} {ctl["get_bytes"]}\n'
                    f'kvb_blocks{{tier="dram"}} {ctl["blocks"]}\n'
                ).encode()
                self._send(200, body, "text/plain")
                return
            self._send(404, b"{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/tokenize":
                self._send(200, json.dumps(
                    {"count": max(1, len(body.get("prompt", "")) // 4)}).encode())
                return
            if self.path == "/v1/completions":
                prompt = body.get("prompt", "")
                ptoks = max(1, len(prompt) // 4)
                toks = gen_tokens_for(prompt, int(body.get("max_tokens", 4)))
                if ctl["flip_at"] is not None and ctl["flip_at"] < len(toks):
                    toks = list(toks)
                    toks[ctl["flip_at"]] = "CORRUPTED"
                if any(m in prompt for m in ctl.get("nondet_markers", ())):
                    cnt = ctl["call_counts"].get(prompt, 0)
                    ctl["call_counts"][prompt] = cnt + 1
                    if cnt % 2 == 1:  # every SECOND answer diverges
                        toks = list(toks)
                        toks[min(3, len(toks) - 1)] = "NONDET"
                choice = {"index": 0, "text": " ".join(toks)}
                if not ctl.get("elide_logprobs"):
                    lp = {"tokens": toks}
                    if int(body.get("logprobs") or 0) >= 2:
                        # top1-top2 gap: wide by default, near-tied when the
                        # prompt is marked — lets the selftest drive the
                        # pre-registered margin screen with no GPU.
                        tie = any(m in prompt for m in ctl.get("lowmargin_markers", ()))
                        gap = 0.05 if tie else 6.0
                        lp["token_logprobs"] = [-0.01] * len(toks)
                        lp["top_logprobs"] = [{"a": -0.01, "b": -0.01 - gap}
                                              for _ in toks]
                    choice["logprobs"] = lp
                resp = {"choices": [choice],
                        "usage": {"prompt_tokens": ptoks,
                                  "completion_tokens": len(toks)}}
                self._send(200, json.dumps(resp).encode())
                mode = ctl["on_completion"]
                if mode == "store":
                    ctl["put_bytes"] += 4.0e6
                    ctl["blocks"] += 4
                elif mode in ("hit", "hit-short"):
                    exp = expected_hit_blocks(ptoks, ctl["block_size"])
                    ctl["hits"] += max(exp - (1 if mode == "hit-short" else 0), 0)
                    ctl["get_bytes"] += 4.0e6
                return
            self._send(404, b"{}")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def selftest() -> int:
    """Prove (1) the boundary-length expansion incl. the zero-block filter
    AND that build_prompt lands EXACTLY on each boundary target (95/96/97
    must land 95/96/97 — one physical length per cell), (2) record fails
    loudly on a severed store path, (3) a healthy record persists prompts +
    token sequences, (4) an exact replay passes at 100% with the isolation
    field defaulting to "unverified", (5) a corrupted replay fails with the
    right first-divergence index, (6) a token-equal but hit-starved compare
    exits 3 (proven equal, unproven warm) with a NOT-CERTIFIED summary
    certification — a failing run's quotable record must never carry the
    certified wording, (7) a logprobs-elided compare exits 3 with the
    text-fallback prompts counted, a NOT-CERTIFIED certification and a
    harness-stamped isolation honored, (10) the kernel-determinism control:
    a prompt the stub
    engine answers differently back-to-back is stamped nondeterministic at
    record, EXCLUDED from the gate (which still passes on the deterministic
    prompts) and disclosed — even when its replay diverges — (11) an
    all-nondeterministic run FAILS loudly (nothing was certified), (12) the
    certified wording discloses the depth the gate actually judged
    (gated_max_blocks) beside the built depth, and (13) the prescreen phase
    screens PADDED margins on this engine, exits 0 when anything is eligible
    and refuses (rc 1) only an all-ineligible corpus — before any populate."""
    import tempfile

    ctl = {"hits": 0.0, "put_bytes": 0.0, "get_bytes": 0.0, "blocks": 0.0,
           "on_completion": "none", "flip_at": None, "block_size": 12,
           "elide_logprobs": False, "nondet_markers": [], "call_counts": {},
           "lowmargin_markers": []}
    srv, url = _stub_server(ctl)
    ok = True

    def mkargs(**kw):
        d = {"vllm": url, "metrics": url, "model": "stub", "n": 4,
             "gen_tokens": 6, "seed": 0, "block_size": 12,
             "boundary_multiples": "8", "lengths": "120", "state": "",
             "out": "", "min_match": 100.0, "put_wait_s": 0.6,
             "request_timeout": 15.0, "print_prefix": PRINT_PREFIX,
             "kv_cache_dtype": "auto-bf16",
             "prompt_set": "", "margin_floor": 2.0, "min_blocks_covered": 1,
             "margin_horizon": 8}
        d.update(kw)
        return argparse.Namespace(**d)

    try:
        # 1) boundary math: 8*12 -> 95,96,97 (+120); B+1 filter drops <=13.
        got = boundary_lengths(12, [8], [120])
        if got != [95, 96, 97, 120]:
            print(f"FAIL: boundary_lengths -> {got}", file=sys.stderr)
            ok = False
        got = boundary_lengths(128, [1], [])
        if got != [129]:
            print(f"FAIL: zero-block filter kept {got}, expected [129]", file=sys.stderr)
            ok = False
        if ok:
            print("[selftest] boundary lengths OK: [95, 96, 97, 120]; "
                  "zero-block lengths dropped")
        if expected_hit_blocks(97, 12) != 8 or expected_hit_blocks(96, 12) != 7:
            print("FAIL: expected_hit_blocks alignment", file=sys.stderr)
            ok = False

        # 1b) EXACT landing: the boundary trio differs by single tokens, so
        # each target must land precisely (the old unit-wide tolerance
        # calibrated 95, 96 and 97 all onto the same 96-token prompt — three
        # "boundary cells" that were one physical length).
        landed = {}
        for t in (95, 96, 97, 120):
            _, n = build_prompt(url, "stub", t, f"land-{t}")
            landed[t] = n
            if n != t:
                print(f"FAIL: build_prompt({t}) landed at {n}, not exactly {t}",
                      file=sys.stderr)
                ok = False
        if all(landed[t] == t for t in landed):
            print(f"[selftest] exact landing OK: targets {sorted(landed)} "
                  "each hit to the token")

        tmp = tempfile.mkdtemp(prefix="equiv-selftest-")
        state_path = os.path.join(tmp, "state.json")

        # 2) record must FAIL LOUDLY on a severed store path.
        ctl["on_completion"] = "none"
        dead = os.path.join(tmp, "dead.json")
        try:
            run_record(mkargs(n=1, state=dead))
            print("FAIL: record PASSED with frozen put counters", file=sys.stderr)
            ok = False
        except StorePathError as e:
            print(f"[selftest] record on a severed store path failed loudly: "
                  f"{str(e)[:96]}...")
        if os.path.exists(dead):
            print("FAIL: severed record still wrote state", file=sys.stderr)
            ok = False

        # 3) healthy record: receipts verified, state persisted with tokens.
        ctl["on_completion"] = "store"
        rc = run_record(mkargs(state=state_path))
        with open(state_path) as f:
            st = json.load(f)
        if rc != 0 or len(st["entries"]) != 4 or \
                any(not e["tokens"] for e in st["entries"]):
            print(f"FAIL: healthy record rc={rc} or malformed state", file=sys.stderr)
            ok = False
        else:
            print(f"[selftest] healthy record: rc=0, {len(st['entries'])} prompts "
                  "with token sequences, put receipts verified")

        # 4) exact replay (well-behaved connector) -> rc 0, 100% match; the
        # driver cannot see whether an engine restart happened, so with no
        # harness stamp the summary must say isolation "unverified".
        ctl["on_completion"] = "hit"
        out_good = os.path.join(tmp, "good.jsonl")
        rc = run_compare(mkargs(state=state_path, out=out_good),
                         {"rig": "selftest"})
        with open(out_good) as f:
            recs = [json.loads(line) for line in f]
        summ = [r for r in recs if r["kind"] == "equivalence-summary"]
        if rc != 0 or len(summ) != 1 or summ[0]["match_rate_pct"] != 100.0 \
                or summ[0]["matched"] != 4 or not summ[0]["warm_attributed_all"] \
                or summ[0]["isolation"] != "unverified" \
                or summ[0]["text_fallback_prompts"] != 0 \
                or summ[0]["n_gated"] != 4 or summ[0]["n_kernel_nondet"] != 0 \
                or any(not r["match"] for r in recs if r["kind"] == "equivalence"):
            print(f"FAIL: exact replay rc={rc} summary={summ}", file=sys.stderr)
            ok = False
        else:
            print("[selftest] exact replay: rc=0, 4/4 matched (all 4 gated, 0 "
                  "kernel-nondet), all warm-attributed, isolation honestly "
                  "'unverified' without a harness stamp")

        # 5) corrupted replay -> rc 1 with the right divergence index.
        ctl["flip_at"] = 2
        out_bad = os.path.join(tmp, "bad.jsonl")
        rc = run_compare(mkargs(state=state_path, out=out_bad),
                         {"rig": "selftest"})
        with open(out_bad) as f:
            bad = [r for r in (json.loads(line) for line in f)
                   if r["kind"] == "equivalence" and not r["match"]]
        if rc != 1 or len(bad) != 4 or \
                any(r["first_divergence_index"] != 2 for r in bad):
            print(f"FAIL: corrupted replay rc={rc} mismatches={len(bad)}",
                  file=sys.stderr)
            ok = False
        else:
            print("[selftest] corrupted replay: rc=1, every mismatch reported "
                  "with first divergence at token 2")
        ctl["flip_at"] = None

        # 6) token-equal but hit-starved -> rc 3 (equal, but unproven warm);
        # the persisted summary must NOT carry the certified wording — a
        # failed run's EQUIVJSONL is quotable, so its certification field
        # must state the weakness, never the claim the run failed to prove.
        for mode, why in (("none", "no hits"), ("hit-short", "one block short")):
            ctl["on_completion"] = mode
            out_uw = os.path.join(tmp, f"unwarm-{mode}.jsonl")
            rc = run_compare(mkargs(state=state_path, out=out_uw),
                             {"rig": "selftest"})
            with open(out_uw) as f:
                summ = [r for r in (json.loads(line) for line in f)
                        if r["kind"] == "equivalence-summary"]
            cert = summ[0]["certification"] if summ else ""
            if rc != 3 or cert.startswith("token-identical") \
                    or not cert.startswith("NOT CERTIFIED") \
                    or "unattributed to the store" not in cert:
                print(f"FAIL: {why} compare rc={rc}, expected 3 with a "
                      f"NOT-CERTIFIED summary; certification={cert!r}",
                      file=sys.stderr)
                ok = False
            else:
                print(f"[selftest] {why} compare: rc=3 — tokens equal but not "
                      f"attributed to kvblockd; certification={cert!r}")

        # 7) logprobs elided by the server -> text-level fallback: matching
        # text is necessary but not the token-exact claim, so rc 3 with the
        # fallback prompts counted; a harness-stamped isolation overrides the
        # "unverified" default.
        ctl["on_completion"] = "hit"
        ctl["elide_logprobs"] = True
        out_txt = os.path.join(tmp, "textfallback.jsonl")
        rc = run_compare(mkargs(state=state_path, out=out_txt),
                         {"rig": "selftest", "isolation": "vllm-restart"})
        with open(out_txt) as f:
            summ = [r for r in (json.loads(line) for line in f)
                    if r["kind"] == "equivalence-summary"]
        cert = summ[0]["certification"] if summ else ""
        if rc != 3 or summ[0]["text_fallback_prompts"] != 4 \
                or summ[0]["matched"] != 4 \
                or summ[0]["isolation"] != "vllm-restart" \
                or cert.startswith("token-identical") \
                or not cert.startswith("NOT CERTIFIED") \
                or "TEXT level only" not in cert:
            print(f"FAIL: text-fallback compare rc={rc} summary={summ}",
                  file=sys.stderr)
            ok = False
        else:
            print("[selftest] text-fallback compare: rc=3, 4/4 text-matched but "
                  "counted as fallback (summary NOT CERTIFIED); stamped "
                  "isolation honored")
        ctl["elide_logprobs"] = False

        # 8) fp8 disclosure labeling: a run recorded AND compared under an
        # fp8 engine dtype must label every record/summary fp8-vs-fp8 (the
        # only equivalence story the fp8 arm may tell) and carry the dtype.
        ctl["on_completion"] = "store"
        fp8_state = os.path.join(tmp, "fp8-state.json")
        rc = run_record(mkargs(state=fp8_state, kv_cache_dtype="fp8_e4m3"))
        if rc != 0:
            print(f"FAIL: fp8 record rc={rc}", file=sys.stderr)
            ok = False
        ctl["on_completion"] = "hit"
        out_fp8 = os.path.join(tmp, "fp8.jsonl")
        rc = run_compare(mkargs(state=fp8_state, out=out_fp8,
                                kv_cache_dtype="fp8_e4m3"),
                         {"rig": "selftest"})
        with open(out_fp8) as f:
            recs = [json.loads(line) for line in f]
        if rc != 0 or not recs \
                or any(r.get("kv_cache_dtype") != "fp8_e4m3" for r in recs) \
                or any(not str(r.get("equivalence_scope", "")).startswith("fp8-vs-fp8")
                       for r in recs):
            print(f"FAIL: fp8 records not labeled fp8-vs-fp8: rc={rc} "
                  f"sample={recs[:1]}", file=sys.stderr)
            ok = False
        else:
            print("[selftest] fp8 run: every record labeled "
                  f"equivalence_scope={recs[0]['equivalence_scope']!r}")

        # 9) cross-dtype compare must be REFUSED (rc 2): fp8 output matching
        # bf16 recompute is NOT the claim this suite makes — an fp8-vs-bf16
        # comparison is exactly the forbidden equivalence story, and letting
        # it run would produce mismatches labeled as store corruption.
        rc = run_compare(mkargs(state=fp8_state,
                                out=os.path.join(tmp, "cross.jsonl"),
                                kv_cache_dtype="auto-bf16"),
                         {"rig": "selftest"})
        if rc != 2:
            print(f"FAIL: cross-dtype compare rc={rc}, expected 2 (refusal) — "
                  "an fp8-vs-bf16 'equivalence' ran", file=sys.stderr)
            ok = False
        else:
            print("[selftest] cross-dtype compare refused (rc=2): the harness "
                  "cannot produce an fp8-vs-bf16 equivalence claim")

        # 10) KERNEL-DETERMINISM CONTROL: prompt e1 answers differently on
        # every second call (a kernel that cannot replay itself at a near-tie).
        # Record must stamp it kernel_deterministic=false with the control's
        # divergence index; the gate must still PASS on the 3 deterministic
        # prompts with the exclusion disclosed in the summary.
        ctl["on_completion"] = "store"
        ctl["nondet_markers"] = ["-e1 ::"]
        ctl["call_counts"] = {}
        nd_state = os.path.join(tmp, "nondet-state.json")
        rc = run_record(mkargs(state=nd_state))
        with open(nd_state) as f:
            st = json.load(f)
        flags = {e["i"]: e["kernel_deterministic"] for e in st["entries"]}
        if rc != 0 or flags != {0: True, 1: False, 2: True, 3: True} \
                or st["entries"][1]["record_divergence_index"] != 3:
            print(f"FAIL: nondet record rc={rc} flags={flags} "
                  f"entry1={st['entries'][1] if len(st['entries']) > 1 else None}",
                  file=sys.stderr)
            ok = False
        else:
            print("[selftest] determinism control: the back-to-back record rep "
                  "caught prompt 1 disagreeing with itself at token 3")
        ctl["on_completion"] = "hit"
        out_nd = os.path.join(tmp, "nondet.jsonl")
        rc = run_compare(mkargs(state=nd_state, out=out_nd), {"rig": "selftest"})
        with open(out_nd) as f:
            recs = [json.loads(line) for line in f]
        summ = [r for r in recs if r["kind"] == "equivalence-summary"]
        nd = summ[0].get("kernel_nondet") if summ else None
        if rc != 0 or len(summ) != 1 or summ[0]["n"] != 4 \
                or summ[0]["n_gated"] != 3 or summ[0]["n_kernel_nondet"] != 1 \
                or summ[0]["matched"] != 3 or summ[0]["match_rate_pct"] != 100.0 \
                or not nd or nd[0]["i"] != 1 \
                or nd[0]["record_divergence_index"] != 3 \
                or "3 of 4" not in summ[0]["certification"] \
                or "1 excluded as kernel-nondeterministic" not in summ[0]["certification"]:
            print(f"FAIL: nondet compare rc={rc} summary={summ}", file=sys.stderr)
            ok = False
        else:
            print("[selftest] nondet-excluded compare: rc=0, gate judged 3/3 "
                  f"gated prompts, certification={summ[0]['certification']!r}")

        # 10b) the SAME nondet prompt now DIVERGES on replay (the stub flips
        # every second call; this compare is its next odd call). The gate
        # must still pass — the prompt can neither fail nor pass it — but the
        # replay divergence must be DISCLOSED, never hidden.
        out_nd2 = os.path.join(tmp, "nondet-replay-div.jsonl")
        rc = run_compare(mkargs(state=nd_state, out=out_nd2), {"rig": "selftest"})
        with open(out_nd2) as f:
            summ = [r for r in (json.loads(line) for line in f)
                    if r["kind"] == "equivalence-summary"]
        nd = summ[0].get("kernel_nondet") if summ else None
        if rc != 0 or not nd or nd[0]["replay_match"] is not False \
                or nd[0]["replay_first_divergence_index"] != 3 \
                or summ[0]["matched"] != 3:
            print(f"FAIL: nondet replay-divergence rc={rc} summary={summ}",
                  file=sys.stderr)
            ok = False
        else:
            print("[selftest] nondet replay divergence: rc=0 (cannot fail the "
                  "gate) with the divergence disclosed at token "
                  f"{nd[0]['replay_first_divergence_index']}")

        # 11) ALL prompts kernel-nondeterministic -> the gate judged nothing,
        # so the run must FAIL LOUDLY (rc 1): a 'pass' over zero gated
        # prompts would certify nothing while looking green.
        ctl["on_completion"] = "store"
        ctl["nondet_markers"] = ["kvstash-equiv"]
        ctl["call_counts"] = {}
        all_state = os.path.join(tmp, "all-nondet-state.json")
        rc = run_record(mkargs(state=all_state))
        if rc != 0:
            print(f"FAIL: all-nondet record rc={rc}", file=sys.stderr)
            ok = False
        ctl["on_completion"] = "hit"
        out_all = os.path.join(tmp, "all-nondet.jsonl")
        rc = run_compare(mkargs(state=all_state, out=out_all), {"rig": "selftest"})
        with open(out_all) as f:
            summ = [r for r in (json.loads(line) for line in f)
                    if r["kind"] == "equivalence-summary"]
        if rc != 1 or not summ or summ[0]["n_gated"] != 0 \
                or summ[0]["n_kernel_nondet"] != 4 \
                or not summ[0]["certification"].startswith("NOTHING CERTIFIED"):
            print(f"FAIL: all-nondet compare rc={rc} (expected 1) summary={summ}",
                  file=sys.stderr)
            ok = False
        else:
            print("[selftest] all-nondeterministic run: rc=1 — loud fail, "
                  "nothing was certified")
        ctl["nondet_markers"] = []

        # 9) FROZEN CORPUS + PRE-REGISTERED MARGIN SCREEN. A near-tie prompt
        # (top1-top2 gap under the floor) must be EXCLUDED and DISCLOSED, the
        # gate must still certify the wide-margin prompts, and the corpus
        # sha256 must land in the records — the pre-registration that stops a
        # corpus from being edited after seeing a result.
        corpus = os.path.join(tempfile.mkdtemp(prefix="equiv-corpus-"), "prompts.txt")
        _write_text(corpus, "# frozen for the selftest\n"
                          "wide margin prompt number one about arithmetic\n"
                          "TIEPROMPT near tied continuation\n"
                          "wide margin prompt number two about geography\n")
        ctl["lowmargin_markers"] = ["TIEPROMPT"]
        st = os.path.join(tempfile.mkdtemp(prefix="equiv-margin-"), "state.json")
        ctl["on_completion"] = "store"      # record must see put counters grow
        rc_rec = run_record(mkargs(n=3, state=st, prompt_set=corpus,
                                   margin_floor=2.0))
        ctl["on_completion"] = "hit"        # compare must see warm attribution
        out_m = os.path.join(tempfile.mkdtemp(prefix="equiv-mout-"), "m.jsonl")
        rc_cmp = run_compare(mkargs(n=3, state=st, prompt_set=corpus,
                                    margin_floor=2.0, out=out_m),
                             {"rig": "selftest"})
        with open(out_m) as f:
            mrecs = [json.loads(line) for line in f]
        msum = [r for r in mrecs if r["kind"] == "equivalence-summary"]
        s0 = msum[0] if msum else {}
        if rc_rec != 0 or rc_cmp != 0:
            print(f"FAIL: frozen-corpus run rc=({rc_rec},{rc_cmp}) — expected 0/0",
                  file=sys.stderr)
            ok = False
        elif s0.get("n_low_margin_excluded") != 1 or s0.get("n_gated") != 2:
            print(f"FAIL: margin screen gated={s0.get('n_gated')} "
                  f"excluded={s0.get('n_low_margin_excluded')} — expected 2 gated, "
                  f"1 excluded", file=sys.stderr)
            ok = False
        elif not s0.get("prompt_corpus_sha256"):
            print("FAIL: corpus sha256 missing from the summary — the "
                  "pre-registration is unverifiable", file=sys.stderr)
            ok = False
        elif "low-margin" not in (s0.get("certification") or ""):
            print(f"FAIL: certification hides the exclusion: "
                  f"{s0.get('certification')!r}", file=sys.stderr)
            ok = False
        else:
            print(f"[selftest] margin screen OK: 2 gated + 1 low-margin excluded "
                  f"and disclosed, corpus sha256={s0['prompt_corpus_sha256']}, "
                  f"certification names it")
        # 9b) COVERAGE FLOOR bites: the same corpus run with a floor above
        # what the lengths can cover must REFUSE (rc 2), not certify a shallow
        # set that cannot exercise the block-index path.
        st2 = os.path.join(tempfile.mkdtemp(prefix="equiv-cov-"), "state.json")
        ctl["on_completion"] = "store"
        rc_cov = run_record(mkargs(n=2, state=st2, prompt_set=corpus,
                                   margin_floor=2.0, min_blocks_covered=999))
        if rc_cov != 2:
            print(f"FAIL: coverage floor did not refuse a shallow set (rc={rc_cov}, "
                  "expected 2)", file=sys.stderr)
            ok = False
        else:
            print("[selftest] coverage floor OK: a set too shallow to exercise the "
                  "block-index path is REFUSED (rc 2), never certified")

        # 9c) A DEEP STORE FAULT still fails under the frozen corpus: corrupt a
        # token late in the replayed output and the gate must NOT certify.
        st3 = os.path.join(tempfile.mkdtemp(prefix="equiv-fault-"), "state.json")
        ctl["on_completion"] = "store"
        rc_r3 = run_record(mkargs(n=2, state=st3, prompt_set=corpus,
                                  margin_floor=2.0, min_blocks_covered=1))
        ctl["on_completion"] = "hit"
        ctl["flip_at"] = 4                      # a store-fault-shaped divergence
        out_f = os.path.join(tempfile.mkdtemp(prefix="equiv-fout-"), "f.jsonl")
        rc_f = run_compare(mkargs(n=2, state=st3, prompt_set=corpus,
                                  margin_floor=2.0, out=out_f), {"rig": "selftest"})
        ctl["flip_at"] = None
        with open(out_f) as f:
            frecs = [json.loads(line) for line in f]
        fsum = [r for r in frecs if r["kind"] == "equivalence-summary"]
        if rc_r3 != 0 or rc_f == 0 or not fsum \
                or "NOT CERTIFIED" not in (fsum[0].get("certification") or ""):
            print(f"FAIL: a store-fault-shaped divergence was NOT caught under the "
                  f"frozen corpus (record rc={rc_r3}, compare rc={rc_f}, "
                  f"certification={fsum[0].get('certification') if fsum else None!r})",
                  file=sys.stderr)
            ok = False
        else:
            print("[selftest] store fault under corpus OK: divergence still FAILS "
                  f"the gate (rc={rc_f}, '{fsum[0]['certification'][:38]}...')")
        # 9d) FORGERY: a hand-edited state (min_margin + low_margin flipped on a
        # prompt whose replay diverges) must NOT be able to launder a store-fault
        # divergence into a disclosed "near-tie". The replay-side margin is
        # re-measured against the frozen floor, so the exclusion is refused.
        st4 = os.path.join(tempfile.mkdtemp(prefix="equiv-forge-"), "state.json")
        ctl["on_completion"] = "store"
        run_record(mkargs(n=2, state=st4, prompt_set=corpus, margin_floor=2.0,
                          min_blocks_covered=1))
        with open(st4) as f:
            forged = json.load(f)
        # entry 0 is a wide-margin prompt; forge it into a "near-tie"...
        forged["entries"][0]["min_margin"] = 0.1
        forged["entries"][0]["low_margin"] = True
        # ...and corrupt its recorded tokens so the replay MUST diverge.
        if forged["entries"][0].get("tokens"):
            forged["entries"][0]["tokens"] = ["FORGED"] + forged["entries"][0]["tokens"][1:]
        _write_text(st4, json.dumps(forged))
        ctl["on_completion"] = "hit"
        out_g = os.path.join(tempfile.mkdtemp(prefix="equiv-gout-"), "g.jsonl")
        rc_g = run_compare(mkargs(n=2, state=st4, prompt_set=corpus,
                                  margin_floor=2.0, out=out_g), {"rig": "selftest"})
        with open(out_g) as f:
            grecs = [json.loads(line) for line in f]
        gsum = [r for r in grecs if r["kind"] == "equivalence-summary"]
        if rc_g == 0 or not gsum \
                or "NOT CERTIFIED" not in (gsum[0].get("certification") or ""):
            print(f"FAIL: a forged state laundered a divergence into a pass "
                  f"(rc={rc_g}, certification="
                  f"{gsum[0].get('certification') if gsum else None!r})",
                  file=sys.stderr)
            ok = False
        else:
            print("[selftest] forgery refused OK: the replay-side margin "
                  "re-measurement rejects a hand-edited near-tie, the divergence "
                  f"still FAILS the gate (rc={rc_g})")

        # 9e) GATED-DEPTH DISCLOSURE: the certified wording from case 9 must
        # state the depth the gate actually judged, separately from the depth
        # the corpus was built to — the two provably diverge under padding
        # (the certified 131k run gated nothing deeper than 4 blocks while
        # the receipt implied 16).
        if not msum or not isinstance(s0.get("gated_max_blocks"), int) \
                or s0.get("gated_max_blocks", 0) < 1 \
                or "gated depth <=" not in (s0.get("certification") or ""):
            print(f"FAIL: certified wording lacks the gated-depth disclosure "
                  f"(gated_max_blocks={s0.get('gated_max_blocks')!r}, "
                  f"certification={s0.get('certification')!r})", file=sys.stderr)
            ok = False
        else:
            print(f"[selftest] gated-depth disclosure OK: summary carries "
                  f"gated_max_blocks={s0['gated_max_blocks']} and the certified "
                  "wording names it")

        # 9f) PRESCREEN: the padded-margin screen must count eligibility on
        # THIS engine (2 of 3 with one marked near-tie -> rc 0) and refuse
        # outright only when NOTHING is eligible (rc 1) — the pre-populate
        # guard that stops a doomed paid arm before it spends.
        rc_p = run_prescreen(mkargs(n=3, prompt_set=corpus, margin_floor=2.0))
        if rc_p != 0:
            print(f"FAIL: prescreen rc={rc_p} with 2 of 3 prompts eligible "
                  "(expected 0)", file=sys.stderr)
            ok = False
        else:
            print("[selftest] prescreen OK: eligible prompts found, rc 0")
        ctl["lowmargin_markers"] = ["prompt number", "TIEPROMPT"]  # all near-tie
        rc_p0 = run_prescreen(mkargs(n=3, prompt_set=corpus, margin_floor=2.0))
        if rc_p0 != 1:
            print(f"FAIL: all-ineligible prescreen rc={rc_p0} (expected 1 — it "
                  "must refuse before the paid populate)", file=sys.stderr)
            ok = False
        else:
            print("[selftest] prescreen refusal OK: zero eligible prompts "
                  "-> rc 1 before any populate")
        ctl["lowmargin_markers"] = []
    finally:
        srv.shutdown()
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ----------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("record", "compare", "calibrate",
                                        "prescreen"))
    ap.add_argument("--vllm", default="http://127.0.0.1:8000")
    ap.add_argument("--metrics", default="",
                    help="kvblockd metrics endpoint; empty skips the store "
                         "receipt + warm attribution (equality-only mode, "
                         "clearly weaker — the compare summary says so)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--state", default="equivalence-state.json")
    ap.add_argument("--n", type=int, default=20,
                    help="prompts total, round-robin over the length list "
                         "(default 20 for the free CPU rig; GPU runs use 100)")
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0, help="greedy decode seed, pinned")
    ap.add_argument("--block-size", type=int, default=16,
                    help="engine KV block size (16 = vLLM CUDA default; the "
                         "CPU backend uses 128)")
    ap.add_argument("--boundary-multiples", default="4,16",
                    help="comma list; each m adds prompts at m*B-1, m*B, m*B+1 "
                         "tokens — the store/recompute boundary cells")
    ap.add_argument("--lengths", default="",
                    help="comma list of extra non-boundary target lengths")
    ap.add_argument("--min-match", type=float, default=100.0,
                    help="compare: minimum match rate %% (default 100 — any "
                         "mismatch from a byte-exact store is a finding)")
    ap.add_argument("--kv-cache-dtype", default="auto-bf16",
                    help="the engine KV dtype BOTH phases run under (the rig's "
                         "stamp convention: auto-bf16, fp8_e4m3, ...). Recorded "
                         "into --state; compare REFUSES a mismatch and every "
                         "record is labeled with its equivalence scope — an fp8 "
                         "run is fp8-vs-fp8, never fp8-vs-bf16")
    ap.add_argument("--put-wait-s", type=float, default=60.0)
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--out", default="",
                    help="compare: JSONL report (also echoed behind the "
                         f"{PRINT_PREFIX} marker)")
    ap.add_argument("--print-prefix", default=PRINT_PREFIX)
    ap.add_argument("--stamp", action="append", default=[], metavar="K=V")
    ap.add_argument("--margin-horizon", type=int, default=8,
                    help="pre-registered: judge the top1-top2 gap over the FIRST N "
                         "generated steps. A min() over a long free-running "
                         "continuation shrinks as the text drifts and would "
                         "exclude nearly every prompt — the horizon keeps the "
                         "screen about the decision the store influences.")
    ap.add_argument("--min-blocks-covered", type=int, default=16,
                    help="pre-registered floor: the deepest certification prompt "
                         "must span at least this many blocks, so the gate "
                         "exercises the block-index path instead of only the "
                         "first block or two. Record REFUSES below it.")
    ap.add_argument("--prompt-set", default="",
                    help="path to a FROZEN high-margin prompt corpus (one prompt "
                         "per line). Replaces the repeated-pangram filler prompts, "
                         "whose near-tied continuations made fp8 engines fail to "
                         "replay themselves (a large share of runs refused for a reason "
                         "that was never about the store). The corpus sha256 is "
                         "stamped into every record: editing it after seeing a "
                         "result voids the certification.")
    ap.add_argument("--margin-floor", type=float, default=2.0,
                    help="pre-registered minimum top1-top2 logprob gap (nats) for "
                         "a prompt to be GATED. Below it the step is a near-tie the "
                         "engine may re-decide across a restart, so the prompt is "
                         "excluded and DISCLOSED — the same narrowing the "
                         "kernel-determinism control performs, never a loosening "
                         "of --min-match.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.phase:
        ap.error("--phase record|compare|calibrate is required (or --selftest); "
                 "record/compare run around an engine restart — see the docstring")

    if args.phase == "calibrate":
        if not args.prompt_set:
            ap.error("--phase calibrate needs --prompt-set (the CANDIDATE pool)")
        return run_calibrate(args)

    if args.phase == "prescreen":
        if not args.prompt_set:
            ap.error("--phase prescreen needs --prompt-set (the corpus whose "
                     "PADDED margins are being screened)")
        try:
            return run_prescreen(args)
        except PromptLandingError as e:
            print(f"FATAL(prescreen): {e}", file=sys.stderr)
            return 2

    if args.phase == "record":
        try:
            return run_record(args)
        except (StorePathError, PromptLandingError) as e:
            print(f"FATAL(record): {e}", file=sys.stderr)
            return 2

    stamp = {"model": args.model}
    for kv in args.stamp:
        k, _, v = kv.partition("=")
        if not v:
            ap.error(f"--stamp needs K=V, got {kv!r}")
        stamp[k] = v
    return run_compare(args, stamp)


if __name__ == "__main__":
    sys.exit(main())
