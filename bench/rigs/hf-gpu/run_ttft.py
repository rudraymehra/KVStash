#!/usr/bin/env python3
"""Chart-2 TTFT driver: cold (recompute) vs warm (kvblockd reload) per prefix length.

Connector-agnostic by construction: every honesty gate reads kvblockd's OWN
metrics (put receipt per prompt, hit growth per warm rep), so the driver works
identically whether the engine reaches kvblockd through the native
vllm_kvblockd connector (the GPU rig's path, proven to move bytes by
bench/e2e/cpu/local-docker.sh) or through LMCache's plugin framework.

Two-phase design (run-3 post-mortem). A warm hit is only believable if the
engine serving it CANNOT have the KV anywhere local. Run 3 tried to force that
by disabling the connector's local tier (`local_cpu: false`); LMCache stages
remote writes THROUGH that buffer, so it silently severed the store path —
kvblockd received nothing, and the "warm" arm measured LMCache's own cache.
The fix mirrors the CI-proven approach (bench/e2e/cpu/verify.py property (d):
hits persist across a vLLM restart): never fight the connector's internals —
get isolation from a vLLM RESTART between storing and measuring.

  --phase populate   vLLM #1: send every sweep prompt once — one prompt per
                     (length, rep), unique nonce at token 0 — so the engine's
                     connector saves the KV to kvblockd. After each prompt the
                     driver polls kvblockd's metrics until
                     kvb_bytes_total{dir="in"} / kvb_blocks GREW; if kvblockd
                     received nothing the whole premise is dead and we FAIL
                     LOUDLY before any measurement. Ends by waiting for any
                     async put queue to drain (put bytes stable for --drain-s),
                     then writes the exact prompts to --state.

  (job.sh restarts vLLM here: any engine-local KV — vLLM's own or a
   connector tier's — dies with the process; kvblockd keeps the blocks.)

  --phase measure    vLLM #2 (fresh engine): first ALL warm requests — the
                     exact prompts from --state; each rep's kvb_hits_total
                     delta must grow or the rep is recorded UNVERIFIED
                     (`warm_hits_verified: false`, red on the chart, exit 3).
                     Then ALL cold requests — fresh-nonce prompts (guaranteed
                     miss, full prefill). Warm runs before cold ON PURPOSE:
                     cold prefills store junk into kvblockd, and once the
                     arena fills, eviction targets unread blocks — exactly the
                     not-yet-measured populated ones. Reading every populated
                     block before writing any junk makes measure-phase
                     eviction harmless by ordering, not by hoping about
                     eviction-policy internals.

vLLM runs with --no-enable-prefix-caching in BOTH phases, so a warm hit can
never come from vLLM's own prefix cache; after the restart it cannot come from
any engine-side connector tier either — kvblockd over TCP is the only source
left, and the per-rep kvb_hits_total growth is the proof.

TTFT is measured on the OpenAI-compatible STREAMING endpoint: the clock starts
immediately before the request is sent and stops on the first SSE event
carrying a non-empty completion token (never on response completion).
`--selftest` proves the timing logic AND the two-phase honesty gates against
an in-process stub with a driver-controlled metrics endpoint — run it before
spending GPU money.

Output (measure phase): one JSONL record per (length, arm) with median/p95
over the reps (warmup pairs discarded), written to --out AND echoed to stdout
behind the CHART2JSONL marker so `hf jobs logs` is a sufficient retrieval
path. Rendering: bench/report/plot.py chart2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.request
import uuid

UNIT = "The quick brown fox jumps over the lazy dog. "
PRINT_PREFIX = "CHART2JSONL"


class PopulateError(RuntimeError):
    """The store path vLLM -> connector -> kvblockd is severed; measuring would be theater."""


# ---------------------------------------------------------------- HTTP bits

def _post_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def tokenize_count(vllm: str, model: str, prompt: str) -> int:
    """vLLM's POST /tokenize — exact server-side token count."""
    return int(_post_json(vllm + "/tokenize", {"model": model, "prompt": prompt})["count"])


def read_counters(metrics_url: str) -> dict:
    """Sum kvblockd counters across labels (parse shape follows
    bench/e2e/cpu/verify.py):

      hits      kvb_hits_total            — the warm-arm proof
      misses    kvb_misses_total          — informational
      put_bytes kvb_bytes_total{dir="in"} — committed PUT payload bytes
                (dir="out" is served GETs and must NOT count as a store)
      blocks    kvb_blocks                — committed blocks resident

    put_bytes/blocks growth is the populate-phase receipt that kvblockd
    actually received data.
    """
    with urllib.request.urlopen(metrics_url + "/metrics", timeout=10) as r:
        text = r.read().decode()
    sums = {"hits": 0.0, "misses": 0.0, "put_bytes": 0.0, "blocks": 0.0}
    for line in text.splitlines():
        if line.startswith("kvb_hits_total"):
            sums["hits"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("kvb_misses_total"):
            sums["misses"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("kvb_bytes_total") and 'dir="in"' in line:
            sums["put_bytes"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("kvb_blocks"):
            sums["blocks"] += float(line.rsplit(" ", 1)[1])
    return sums


def wait_counter_growth(metrics_url: str, before: dict, keys: tuple,
                        timeout_s: float, poll_s: float = 0.25):
    """Poll until ANY of `keys` grew past its value in `before`; returns the
    counters snapshot, or None on timeout (= kvblockd received nothing)."""
    deadline = time.monotonic() + timeout_s
    while True:
        now = read_counters(metrics_url)
        if any(now[k] > before[k] for k in keys):
            return now
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_s)


def wait_put_quiesce(metrics_url: str, stable_s: float, timeout_s: float,
                     poll_s: float = 0.5):
    """Wait until kvb_bytes_total{dir="in"} / kvb_blocks stop growing for
    stable_s. A connector's remote put path may be asynchronous (LMCache's is;
    the native connector's is sync) and job.sh kills vLLM right after populate
    — anything still queued would be dropped with the process. Returns the
    final counters, or None if puts were still trickling in at timeout_s."""
    deadline = time.monotonic() + timeout_s
    last = read_counters(metrics_url)
    stable_since = time.monotonic()
    while True:
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_s)
        now = read_counters(metrics_url)
        if now["put_bytes"] != last["put_bytes"] or now["blocks"] != last["blocks"]:
            last, stable_since = now, time.monotonic()
        elif time.monotonic() - stable_since >= stable_s:
            return now


# ------------------------------------------------------------- measurement

def measure_stream(vllm: str, model: str, prompt: str, gen_tokens: int,
                   timeout_s: float) -> dict:
    """One streaming completion; returns TTFT (first-token) + total time.

    The clock starts immediately before urlopen() (which sends the request)
    and TTFT stops on the FIRST SSE event with a non-empty `text` — never on
    stream completion. usage comes from the final include_usage chunk.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": gen_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        vllm + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    ttft = None
    usage = None
    n_token_events = 0
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:  # readline-driven: yields each SSE line as it arrives
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            choices = ev.get("choices") or []
            if choices and choices[0].get("text"):
                if ttft is None:
                    ttft = time.monotonic() - t0
                n_token_events += 1
            if ev.get("usage"):
                usage = ev["usage"]
    total = time.monotonic() - t0
    if ttft is None:
        raise RuntimeError("stream ended without a completion token")
    return {"ttft_s": ttft, "total_s": total, "usage": usage,
            "token_events": n_token_events}


# ------------------------------------------------------------ prompt build

def build_prompt(vllm: str, model: str, target_tokens: int, nonce: str) -> tuple[str, int]:
    """Filler prompt of ~target_tokens, nonce FIRST so every connector block
    key (a prefix chain from token 0 — both the native connector's BLAKE3
    chain and LMCache's chunk hashes work this way) differs between reps →
    cold is a guaranteed miss. Calibrated against the server's own /tokenize;
    the ACTUAL count is returned and stamped, never assumed."""
    head = f"kvstash-ttft {nonce} :: "
    base = tokenize_count(vllm, model, head)
    probe = tokenize_count(vllm, model, head + UNIT * 128)
    unit_toks = max((probe - base) / 128.0, 0.5)
    k = max(1, round((target_tokens - base) / unit_toks))
    # Tolerance can never be finer than one filler unit (k is an integer),
    # so: one unit, or 0.5% of the target, whichever is coarser.
    tol = max(math.ceil(unit_toks), target_tokens // 200)
    prompt, n = head, base
    for _ in range(10):
        prompt = head + UNIT * k
        n = tokenize_count(vllm, model, prompt)
        err = target_tokens - n
        if abs(err) <= tol:
            break
        k = max(1, k + (round(err / unit_toks) or (1 if err > 0 else -1)))
    return prompt, n


def parse_lengths(s: str) -> list[int]:
    lengths = [int(x) for x in s.split(",") if x.strip()]
    if not lengths:
        raise ValueError(f"no lengths in {s!r}")
    return lengths


# -------------------------------------------------------------------- stats

def _p95(xs):
    s = sorted(xs)
    if not s:
        return 0.0
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]  # nearest-rank


# ---------------------------------------------------------- phase: populate

def run_populate(args) -> int:
    """Phase 1 (vLLM #1): prefill every sweep prompt once so the engine's
    connector stores the KV into kvblockd; assert per-prompt that kvblockd
    RECEIVED data, then drain any async put queue and persist the prompts to
    --state. Raises PopulateError (fail loudly, nothing measured yet) on a
    severed store path."""
    lengths = parse_lengths(args.lengths)
    n_pairs = args.warmup + args.reps
    state = {
        "state_schema": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "lengths": lengths,
        "reps": args.reps,
        "warmup": args.warmup,
        "entries": {str(L): [] for L in lengths},
    }
    t_start = read_counters(args.metrics)
    total_tokens = 0
    for L in lengths:
        for rep in range(n_pairs):
            nonce = f"{uuid.uuid4().hex[:12]}-L{L}-r{rep}"
            prompt, ntok = build_prompt(args.vllm, args.model, L, nonce)
            c0 = read_counters(args.metrics)
            # max_tokens=1: populate only needs the PREFILL — the warm arm
            # matches on the prompt's chunk hashes, not on generated text.
            measure_stream(args.vllm, args.model, prompt, 1, args.request_timeout)
            after = wait_counter_growth(args.metrics, c0, ("put_bytes", "blocks"),
                                        args.put_wait_s, poll_s=args.poll_s)
            if after is None:
                raise PopulateError(
                    f"kvblockd received NOTHING for L={L} rep={rep}: "
                    f'kvb_bytes_total{{dir="in"}} and kvb_blocks stayed flat for '
                    f"{args.put_wait_s:.0f}s after the prefill. The store path "
                    "vLLM->connector->kvblockd is severed (the run-3 failure mode); "
                    "a warm arm measured now would be meaningless. Aborting "
                    "before any measurement.")
            put_mb = (after["put_bytes"] - c0["put_bytes"]) / 1e6
            print(f"[populate] L={L} rep={rep} tokens={ntok} stored={put_mb:.1f}MB "
                  f"blocks+={after['blocks'] - c0['blocks']:.0f}", flush=True)
            state["entries"][str(L)].append(
                {"rep": rep, "nonce": nonce, "prompt": prompt, "prompt_tokens": ntok})
            total_tokens += ntok

    final = wait_put_quiesce(args.metrics, args.drain_s, args.drain_timeout_s,
                             poll_s=args.poll_s)
    if final is None:
        raise PopulateError(
            f"kvblockd's put counters were still moving after {args.drain_timeout_s:.0f}s "
            "— the connector's async put queue never drained; restarting vLLM "
            "now would drop queued blocks.")
    put_total = final["put_bytes"] - t_start["put_bytes"]
    blocks_total = final["blocks"] - t_start["blocks"]
    if put_total <= 0 or blocks_total <= 0:
        raise PopulateError(
            f"populate ran but kvblockd's totals did not grow "
            f"(put_bytes+={put_total:.0f}, blocks+={blocks_total:.0f})")
    bpt = put_total / max(total_tokens, 1)
    print(f"[populate] DONE: {put_total / 1e9:.2f}GB in {blocks_total:.0f} blocks stored to "
          f"kvblockd for {total_tokens} prompt tokens (~{bpt / 1024:.0f}KiB/token)",
          flush=True)
    if bpt < 8 * 1024:
        print(f"[populate] WARN: ~{bpt / 1024:.1f}KiB/token is suspiciously low for a "
              "7-8B model (~56-128KiB/token expected) — partial stores?", flush=True)
    tmp = args.state + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, args.state)
    print(f"[populate] state -> {args.state} "
          f"({sum(len(v) for v in state['entries'].values())} prompts)", flush=True)
    return 0


# ----------------------------------------------------------- phase: measure

def run_measure(args, stamp: dict) -> int:
    """Phase 2 (vLLM #2, freshly restarted): all WARM reps first (exact
    populated prompts; per-rep kvb_hits_total growth is the proof), then all
    COLD reps (fresh nonces, full prefill). See the module docstring for why
    warm precedes cold. Exit 3 if any measured warm rep is unverified."""
    try:
        with open(args.state) as f:
            state = json.load(f)
    except (OSError, ValueError) as e:
        print(f"FATAL(measure): cannot read --state {args.state}: {e}", file=sys.stderr)
        return 2
    if state.get("model") != args.model:
        print(f"FATAL(measure): state was populated for model {state.get('model')!r}, "
              f"measure asked for {args.model!r}", file=sys.stderr)
        return 2
    lengths = state["lengths"]
    warmup, reps = state["warmup"], state["reps"]
    n_pairs = warmup + reps
    for L in lengths:
        n = len(state["entries"].get(str(L), []))
        if n != n_pairs:
            print(f"FATAL(measure): state has {n} prompts for L={L}, expected {n_pairs}",
                  file=sys.stderr)
            return 2
    print(f"[measure] lengths={lengths} reps={reps}(+{warmup} warmup) from state "
          f"populated at {state.get('created')}", flush=True)

    # ---- pass 1: WARM — read every populated block before storing any junk.
    warm_res = {}
    for L in lengths:
        for e in state["entries"][str(L)]:
            c0 = read_counters(args.metrics)
            m = measure_stream(args.vllm, args.model, e["prompt"],
                               args.gen_tokens, args.request_timeout)
            c1 = read_counters(args.metrics)
            hit_delta = c1["hits"] - c0["hits"]
            verified = hit_delta > 0
            usage = m.get("usage") or {}
            r = {"ttft_ms": m["ttft_s"] * 1e3, "total_ms": m["total_s"] * 1e3,
                 "hit_delta": hit_delta, "verified": verified,
                 "prompt_tokens": int(usage.get("prompt_tokens", e["prompt_tokens"]))}
            warm_res[(L, e["rep"])] = r
            print(f"[warm] L={L} rep={e['rep']} tokens={r['prompt_tokens']} "
                  f"ttft={r['ttft_ms']:.0f}ms hits+={hit_delta:.0f} verified={verified}",
                  flush=True)

    # ---- pass 2: COLD — fresh nonce at token 0 -> guaranteed miss.
    cold_res = {}
    for L in lengths:
        for rep in range(n_pairs):
            nonce = f"{uuid.uuid4().hex[:12]}-L{L}-c{rep}"
            prompt, ntok = build_prompt(args.vllm, args.model, L, nonce)
            c0 = read_counters(args.metrics)
            m = measure_stream(args.vllm, args.model, prompt,
                               args.gen_tokens, args.request_timeout)
            c1 = read_counters(args.metrics)
            usage = m.get("usage") or {}
            r = {"ttft_ms": m["ttft_s"] * 1e3, "total_ms": m["total_s"] * 1e3,
                 "miss_delta": c1["misses"] - c0["misses"],
                 "prompt_tokens": int(usage.get("prompt_tokens", ntok))}
            cold_res[(L, rep)] = r
            print(f"[cold] L={L} rep={rep} tokens={r['prompt_tokens']} "
                  f"ttft={r['ttft_ms']:.0f}ms misses+={r['miss_delta']:.0f}", flush=True)

    # ---- aggregate + emit (schema unchanged: plot.py / log extraction as-is)
    out = open(args.out, "a")

    def emit(rec: dict):
        line = json.dumps(rec, sort_keys=True)
        out.write(line + "\n")
        out.flush()
        print(f"{args.print_prefix} {line}", flush=True)

    failures = 0
    summary = []
    for L in lengths:
        pairs = []
        for rep in range(n_pairs):
            w, c = warm_res[(L, rep)], cold_res[(L, rep)]
            pairs.append({
                "warmup": rep < warmup, "prompt_tokens": w["prompt_tokens"],
                "cold_ttft_ms": c["ttft_ms"], "warm_ttft_ms": w["ttft_ms"],
                "cold_total_ms": c["total_ms"], "warm_total_ms": w["total_ms"],
                "kvb_hit_delta": w["hit_delta"], "kvb_miss_delta": c["miss_delta"],
                "warm_verified": w["verified"],
            })
        used = [p for p in pairs if not p["warmup"]]
        colds = [p["cold_ttft_ms"] for p in used]
        warms = [p["warm_ttft_ms"] for p in used]
        verified = all(p["warm_verified"] for p in used)
        if not verified:
            failures += 1
        base = {
            "schema_version": 1, "kind": "ttft",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "store": "kvblockd",
            "target_prefix_tokens": L,
            "prefix_tokens": int(statistics.median(p["prompt_tokens"] for p in used)),
            "gen_tokens": args.gen_tokens,
            "reps": len(used), "warmup_reps_discarded": warmup,
            "warm_isolation": "vllm-restart",
        }
        base.update(stamp)
        cold_p50 = statistics.median(colds)
        warm_p50 = statistics.median(warms)
        emit({**base, "arm": "cold", "series": "recompute (cold)",
              "ttft_ms": cold_p50, "ttft_p50_ms": cold_p50,
              "ttft_p95_ms": _p95(colds), "ttft_all_ms": [round(x, 3) for x in colds]})
        emit({**base, "arm": "warm", "series": "kvblockd reload (warm)",
              "ttft_ms": warm_p50, "ttft_p50_ms": warm_p50,
              "ttft_p95_ms": _p95(warms), "ttft_all_ms": [round(x, 3) for x in warms],
              "warm_hits_verified": verified,
              "kvb_hit_delta_total": sum(p["kvb_hit_delta"] for p in used),
              "speedup_p50_vs_cold": (cold_p50 / warm_p50) if warm_p50 > 0 else None})
        summary.append((L, base["prefix_tokens"], cold_p50, warm_p50, verified))
    out.close()

    print("\n== TTFT summary (p50 over reps, warmup discarded; warm arm = fresh "
          "engine reading kvblockd over TCP) ==")
    print(f"{'target':>8} {'tokens':>8} {'cold ms':>10} {'warm ms':>10} {'speedup':>8} verified")
    for target, ntok, c, w, v in summary:
        sp = f"{c / w:.2f}x" if w > 0 else "n/a"
        print(f"{target:>8} {ntok:>8} {c:>10.0f} {w:>10.0f} {sp:>8} {v}")
    if failures:
        print(f"\nFAIL: {failures} length cell(s) had warm reps whose kvb_hits_total "
              "did not grow — the warm arm is UNVERIFIED there (flagged in the JSONL).",
              file=sys.stderr)
        return 3
    return 0


# ----------------------------------------------------------------- selftest

def _stub_server(ctl):
    """In-process stub: vLLM endpoints (/tokenize, streaming /v1/completions)
    plus a kvblockd-shaped /metrics whose counters the selftest CONTROLS via
    `ctl`. ctl["on_completion"] decides what each completion does to them:
    "store" (put_bytes/blocks grow — a working populate path), "hit"
    (hits/get-bytes grow — a working warm read), "none" (frozen — a severed
    path, the run-3 failure mode)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Stub(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):  # noqa: ARG002 - quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/metrics":
                # dir="out" is deliberately present and growing: the parser
                # must never count served GETs as stores.
                body = (
                    f'kvb_hits_total{{tier="dram",ns="1"}} {ctl["hits"]}\n'
                    f'kvb_misses_total{{ns="1"}} {ctl["misses"]}\n'
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
                out = json.dumps({"count": max(1, len(body.get("prompt", "")) // 4)}).encode()
                self._send(200, out)
                return
            if self.path == "/v1/completions":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.flush()
                time.sleep(ctl["first_delay"])
                for i in range(ctl["ntok"]):
                    if i:
                        time.sleep(ctl["inter_delay"])
                    ev = {"choices": [{"index": 0, "text": f"tok{i} "}]}
                    self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                    self.wfile.flush()
                ptoks = max(1, len(body.get("prompt", "")) // 4)
                usage = {"choices": [],
                         "usage": {"prompt_tokens": ptoks, "completion_tokens": ctl["ntok"]}}
                self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                mode = ctl["on_completion"]
                if mode == "store":
                    ctl["put_bytes"] += 4.0e6
                    ctl["blocks"] += 4
                elif mode == "hit":
                    ctl["hits"] += 8
                    ctl["get_bytes"] += 4.0e6
                # "none": counters frozen — kvblockd never touched.
                return
            self._send(404, b"{}")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def selftest() -> int:
    """Prove, against the controllable stub, (1) the first-token timer, (2)
    prompt calibration + stats helpers, (3) the metrics parse (incl. the
    dir="in" filter), and the two-phase honesty gates: (4) populate FAILS
    LOUDLY when kvblockd receives nothing, (5) a healthy populate persists
    state, (6) measure marks reps UNVERIFIED (rc 3) when hits do not grow,
    (7) a healthy measure yields verified, well-formed CHART2JSONL records."""
    import contextlib
    import io
    import tempfile

    FIRST, INTER, NTOK = 0.35, 0.06, 4
    ctl = {"hits": 0.0, "misses": 0.0, "put_bytes": 0.0, "get_bytes": 0.0,
           "blocks": 0.0, "on_completion": "none",
           "first_delay": FIRST, "inter_delay": INTER, "ntok": NTOK}
    srv, url = _stub_server(ctl)
    ok = True

    def mkargs(**kw):
        d = dict(vllm=url, metrics=url, model="stub", gen_tokens=2,
                 request_timeout=15.0, print_prefix=PRINT_PREFIX,
                 put_wait_s=0.6, drain_s=0.1, drain_timeout_s=3.0, poll_s=0.05,
                 lengths="96,160", reps=1, warmup=1, state="", out="")
        d.update(kw)
        return argparse.Namespace(**d)

    try:
        # 1) first-token timer: a whole-response timer would read
        #    ~first+inter-token delays; a correct one reads ~the first only.
        m = measure_stream(url, "stub", "x" * 168, NTOK, 30)
        tail = (NTOK - 1) * INTER
        print(f"[selftest] ttft={m['ttft_s'] * 1e3:.0f}ms total={m['total_s'] * 1e3:.0f}ms "
              f"(stub: first token at {FIRST * 1e3:.0f}ms, {NTOK} tokens {INTER * 1e3:.0f}ms apart)")
        if not (FIRST - 0.05 <= m["ttft_s"] <= FIRST + 0.25):
            print(f"FAIL: ttft {m['ttft_s']:.3f}s not ≈ first-token delay {FIRST}s", file=sys.stderr)
            ok = False
        if m["total_s"] - m["ttft_s"] < tail * 0.6:
            print("FAIL: total-ttft gap too small — timer stopped at stream end, "
                  "not at the first token?", file=sys.stderr)
            ok = False
        if (m["usage"] or {}).get("prompt_tokens") != 42:  # 168 chars // 4
            print(f"FAIL: usage not parsed from the include_usage chunk: {m['usage']}", file=sys.stderr)
            ok = False
        if m["token_events"] != NTOK:
            print(f"FAIL: saw {m['token_events']} token events, expected {NTOK}", file=sys.stderr)
            ok = False

        # 2) prompt calibration + 3) stats helpers
        prompt, n = build_prompt(url, "stub", 1024, "selftest")
        print(f"[selftest] build_prompt(1024) -> {n} tokens ({len(prompt)} chars)")
        if abs(n - 1024) > 16:  # one stub "unit" (46 chars // 4) + slack
            print(f"FAIL: build_prompt converged to {n}, target 1024", file=sys.stderr)
            ok = False
        if _p95(list(range(1, 21))) != 19 or statistics.median([1, 2, 3]) != 2:
            print("FAIL: percentile helpers", file=sys.stderr)
            ok = False

        # 3b) metrics parse: dir="out" (7e6, growing) must NOT count as puts.
        ctl.update(hits=3.0, misses=2.0, put_bytes=1.0e6, get_bytes=7.0e6, blocks=5.0)
        c = read_counters(url)
        if c != {"hits": 3.0, "misses": 2.0, "put_bytes": 1.0e6, "blocks": 5.0}:
            print(f"FAIL: read_counters parsed {c} (dir filter broken?)", file=sys.stderr)
            ok = False
        else:
            print('[selftest] read_counters OK (kvb_bytes_total dir="out" excluded from put_bytes)')

        # ---- two-phase honesty gates (fast stub timings) --------------------
        ctl.update(first_delay=0.01, inter_delay=0.0, ntok=2)
        tmp = tempfile.mkdtemp(prefix="chart2-selftest-")
        state_path = os.path.join(tmp, "state.json")

        # 4) populate must FAIL LOUDLY when kvblockd receives nothing.
        ctl["on_completion"] = "none"
        dead_state = os.path.join(tmp, "dead.json")
        try:
            run_populate(mkargs(lengths="96", reps=1, warmup=0, state=dead_state))
            print("FAIL: populate PASSED although kvblockd's counters never grew "
                  "(the run-3 failure mode went undetected)", file=sys.stderr)
            ok = False
        except PopulateError as e:
            print(f"[selftest] populate on a severed store path failed loudly, as required: "
                  f"{str(e)[:110]}...")
        if os.path.exists(dead_state):
            print("FAIL: severed populate still wrote a state file", file=sys.stderr)
            ok = False

        # 5) healthy populate: receipt verified per prompt, state persisted.
        ctl["on_completion"] = "store"
        rc = run_populate(mkargs(state=state_path))
        st = json.load(open(state_path))
        if rc != 0 or sorted(st["entries"]) != ["160", "96"] or \
                any(len(v) != 2 for v in st["entries"].values()):
            print(f"FAIL: healthy populate rc={rc} or malformed state", file=sys.stderr)
            ok = False
        else:
            print(f"[selftest] healthy populate: rc=0, state has "
                  f"{sum(len(v) for v in st['entries'].values())} prompts, put receipt verified")

        # 6) measure must mark reps UNVERIFIED (rc 3) when hits do not grow.
        ctl["on_completion"] = "none"
        out_bad = os.path.join(tmp, "bad.jsonl")
        rc = run_measure(mkargs(state=state_path, out=out_bad),
                         {"model": "stub", "rig": "selftest"})
        bad_warm = [r for r in (json.loads(l) for l in open(out_bad)) if r["arm"] == "warm"]
        if rc != 3 or len(bad_warm) != 2 or any(r["warm_hits_verified"] for r in bad_warm) \
                or any(r["kvb_hit_delta_total"] != 0 for r in bad_warm):
            print(f"FAIL: no-hit measure not flagged: rc={rc} warm={bad_warm}", file=sys.stderr)
            ok = False
        else:
            print(f"[selftest] no-hit measure: rc=3, {len(bad_warm)} warm records flagged "
                  "warm_hits_verified=false, kvb_hit_delta_total=0")

        # 7) healthy measure: verified records, sane schema, CHART2JSONL marker.
        ctl["on_completion"] = "hit"
        out_good = os.path.join(tmp, "good.jsonl")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = run_measure(mkargs(state=state_path, out=out_good),
                             {"model": "stub", "rig": "selftest"})
        sys.stdout.write(buf.getvalue())  # captured only to verify the marker lines
        good = [json.loads(l) for l in open(out_good)]
        marked = [json.loads(l.split(PRINT_PREFIX + " ", 1)[1])
                  for l in buf.getvalue().splitlines()
                  if l.startswith(PRINT_PREFIX + " ")]
        warm_recs = [r for r in good if r["arm"] == "warm"]
        cold_recs = [r for r in good if r["arm"] == "cold"]
        checks = {
            "rc==0": rc == 0,
            "4 records (2 lengths x 2 arms)": len(good) == 4 and len(warm_recs) == 2
                                              and len(cold_recs) == 2,
            "warm verified with hit deltas": all(r["warm_hits_verified"] for r in warm_recs)
                                             and all(r["kvb_hit_delta_total"] > 0 for r in warm_recs),
            "isolation stamped": all(r.get("warm_isolation") == "vllm-restart" for r in good),
            "speedup + timings present": all("speedup_p50_vs_cold" in r for r in warm_recs)
                                         and all(r["ttft_ms"] > 0 for r in good),
            "CHART2JSONL mirrors the file": marked == good,
        }
        for name, passed in checks.items():
            if not passed:
                print(f"FAIL: healthy measure: {name}", file=sys.stderr)
                ok = False
        if all(checks.values()):
            print("[selftest] healthy measure: rc=0, 4 verified records, "
                  "CHART2JSONL lines mirror the JSONL file")
    finally:
        srv.shutdown()
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("populate", "measure"),
                    help="populate: store the sweep prompts via vLLM #1 and verify "
                         "kvblockd received them; measure: on a RESTARTED vLLM, time "
                         "warm (kvblockd reload) then cold (recompute) arms")
    ap.add_argument("--vllm", default="http://127.0.0.1:8000")
    ap.add_argument("--metrics", default="http://127.0.0.1:9442",
                    help="kvblockd metrics endpoint (put receipt + hit verification)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--state", default="chart2-state.json",
                    help="prompt state handed from populate to measure")
    ap.add_argument("--lengths", default="1024,4096,8192,16384",
                    help="comma-separated target prefix token counts (populate phase; "
                         "measure reads them from --state)")
    ap.add_argument("--reps", type=int, default=5, help="measured rep pairs per length")
    ap.add_argument("--warmup", type=int, default=1, help="discarded warmup pairs per length")
    ap.add_argument("--gen-tokens", type=int, default=16)
    ap.add_argument("--put-wait-s", type=float, default=120.0,
                    help="populate: max wait for kvblockd's put counters to grow per prompt")
    ap.add_argument("--drain-s", type=float, default=5.0,
                    help="populate: put counters must be stable this long before exit "
                         "(a connector's remote puts may be async; vLLM is killed next)")
    ap.add_argument("--drain-timeout-s", type=float, default=300.0)
    ap.add_argument("--poll-s", type=float, default=0.25, help="metrics poll interval")
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--out", default="chart2-ttft.jsonl")
    ap.add_argument("--print-prefix", default=PRINT_PREFIX)
    ap.add_argument("--stamp", action="append", default=[], metavar="K=V",
                    help="stamped into every record (gpu=, model=, vllm=, connector=, "
                         "tc_link=, rig=, git_sha=) — plot.py reads conditions from these")
    ap.add_argument("--selftest", action="store_true",
                    help="verify the timing logic and the two-phase honesty gates "
                         "against a local stub, then exit")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.phase:
        ap.error("--phase populate|measure is required (or --selftest). The old "
                 "single-process cold/warm flow was removed after run 3: without a "
                 "vLLM restart between store and read, a 'warm' hit can come from "
                 "an engine-side cache tier and never touch kvblockd.")

    if args.phase == "populate":
        try:
            return run_populate(args)
        except PopulateError as e:
            print(f"FATAL(populate): {e}", file=sys.stderr)
            return 2

    stamp = {"model": args.model}
    for kv in args.stamp:
        k, _, v = kv.partition("=")
        if not v:
            ap.error(f"--stamp needs K=V, got {kv!r}")
        stamp[k] = v
    return run_measure(args, stamp)


if __name__ == "__main__":
    sys.exit(main())
