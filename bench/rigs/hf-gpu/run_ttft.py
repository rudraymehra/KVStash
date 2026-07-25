#!/usr/bin/env python3
"""Chart-2 TTFT driver: cold (recompute) vs warm (kvblockd reload) per prefix length.

Starts NOTHING itself — it assumes a vLLM OpenAI server (with the
LMCacheConnectorV1 → lmcache_kvblockd kv-transfer config) and a kvblockd
daemon are already up (bench/rigs/hf-gpu/job.sh does that). For each prefix
length in the sweep it runs rep pairs:

  cold  a FRESH prompt (unique nonce at position 0, so every LMCache chunk
        hash differs — guaranteed miss) → vLLM does the full prefill and
        LMCache stores the KV blocks into kvblockd.
  warm  the SAME prompt again. vLLM runs with --no-enable-prefix-caching and
        LMCache runs with local_cpu: false, so the only place the KV can come
        from is kvblockd over TCP. The kvb_hits_total delta is checked per
        rep; a warm rep whose hits did not grow is recorded as UNVERIFIED
        (and the process exits nonzero) — honesty over completeness.

TTFT is measured on the OpenAI-compatible STREAMING endpoint: the clock
starts immediately before the request is sent and stops on the first SSE
event carrying a non-empty completion token (never on response completion).
`--selftest` proves that against an in-process stub server whose first token
is deliberately delayed — run it before spending GPU money.

Output: one JSONL record per (length, arm) with median/p95 over the reps
(first warmup pair discarded), written to --out AND echoed to stdout behind
the CHART2JSONL marker so `hf jobs logs` is a sufficient retrieval path.
Rendering: bench/report/plot.py chart2.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid

UNIT = "The quick brown fox jumps over the lazy dog. "
PRINT_PREFIX = "CHART2JSONL"


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
    """Sum kvb_hits_total / kvb_misses_total across labels (same parse shape
    as bench/e2e/cpu/verify.py)."""
    with urllib.request.urlopen(metrics_url + "/metrics", timeout=10) as r:
        text = r.read().decode()
    sums = {"hits": 0.0, "misses": 0.0}
    for line in text.splitlines():
        if line.startswith("kvb_hits_total"):
            sums["hits"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("kvb_misses_total"):
            sums["misses"] += float(line.rsplit(" ", 1)[1])
    return sums


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
    """Filler prompt of ~target_tokens, nonce FIRST so every LMCache chunk
    hash (a prefix chain from token 0) differs between reps → cold is a
    guaranteed miss. Calibrated against the server's own /tokenize; the
    ACTUAL count is returned and stamped, never assumed."""
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


# -------------------------------------------------------------------- stats

def _p95(xs):
    s = sorted(xs)
    if not s:
        return 0.0
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]  # nearest-rank


# -------------------------------------------------------------------- sweep

def run_sweep(args, stamp: dict) -> int:
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    out = open(args.out, "a")
    failures = 0

    def emit(rec: dict):
        line = json.dumps(rec, sort_keys=True)
        out.write(line + "\n")
        out.flush()
        print(f"{args.print_prefix} {line}", flush=True)

    summary = []
    for target in lengths:
        pairs = []
        for rep in range(args.warmup + args.reps):
            is_warmup = rep < args.warmup
            nonce = f"{uuid.uuid4().hex[:12]}-L{target}-r{rep}"
            prompt, ntok = build_prompt(args.vllm, args.model, target, nonce)

            c0 = read_counters(args.metrics)
            cold = measure_stream(args.vllm, args.model, prompt,
                                  args.gen_tokens, args.request_timeout)
            c1 = read_counters(args.metrics)
            time.sleep(args.settle_s)  # let LMCache finish storing to kvblockd
            c2 = read_counters(args.metrics)
            warm = measure_stream(args.vllm, args.model, prompt,
                                  args.gen_tokens, args.request_timeout)
            c3 = read_counters(args.metrics)

            hit_delta = c3["hits"] - c2["hits"]
            miss_delta = c1["misses"] - c0["misses"]
            usage = warm.get("usage") or cold.get("usage") or {}
            actual = int(usage.get("prompt_tokens", ntok))
            pair = {
                "warmup": is_warmup, "prompt_tokens": actual,
                "cold_ttft_ms": cold["ttft_s"] * 1e3,
                "warm_ttft_ms": warm["ttft_s"] * 1e3,
                "cold_total_ms": cold["total_s"] * 1e3,
                "warm_total_ms": warm["total_s"] * 1e3,
                "kvb_hit_delta": hit_delta, "kvb_miss_delta": miss_delta,
                "warm_verified": hit_delta > 0,
            }
            pairs.append(pair)
            print(f"[ttft] L={target} rep={rep}{' (warmup, discarded)' if is_warmup else ''} "
                  f"tokens={actual} cold={pair['cold_ttft_ms']:.0f}ms "
                  f"warm={pair['warm_ttft_ms']:.0f}ms hits+={hit_delta:.0f} "
                  f"misses+={miss_delta:.0f} verified={pair['warm_verified']}",
                  flush=True)

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
            "target_prefix_tokens": target,
            "prefix_tokens": int(statistics.median(p["prompt_tokens"] for p in used)),
            "gen_tokens": args.gen_tokens,
            "reps": len(used), "warmup_reps_discarded": args.warmup,
            "settle_s": args.settle_s,
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
        summary.append((target, base["prefix_tokens"], cold_p50, warm_p50, verified))

    out.close()
    print("\n== TTFT summary (p50 over reps, warmup discarded) ==")
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

def selftest() -> int:
    """Prove the TTFT logic against a local stub whose FIRST token is
    deliberately delayed: a whole-response timer would read ~first+inter
    token delays; a correct first-token timer reads ~the first delay only."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    FIRST, INTER, NTOK = 0.35, 0.06, 4

    class Stub(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):  # noqa: ARG002 - quiet
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/tokenize":
                out = json.dumps({"count": max(1, len(body.get("prompt", "")) // 4)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            if self.path == "/v1/completions":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.flush()
                time.sleep(FIRST)
                for i in range(NTOK):
                    if i:
                        time.sleep(INTER)
                    ev = {"choices": [{"index": 0, "text": f"tok{i} "}]}
                    self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                    self.wfile.flush()
                usage = {"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": NTOK}}
                self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            self.send_response(404)
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    ok = True
    try:
        m = measure_stream(url, "stub", "hello", NTOK, 30)
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
        if (m["usage"] or {}).get("prompt_tokens") != 42:
            print(f"FAIL: usage not parsed from the include_usage chunk: {m['usage']}", file=sys.stderr)
            ok = False
        if m["token_events"] != NTOK:
            print(f"FAIL: saw {m['token_events']} token events, expected {NTOK}", file=sys.stderr)
            ok = False
        prompt, n = build_prompt(url, "stub", 1024, "selftest")
        print(f"[selftest] build_prompt(1024) -> {n} tokens ({len(prompt)} chars)")
        if abs(n - 1024) > 16:  # one stub "unit" (46 chars // 4) + slack
            print(f"FAIL: build_prompt converged to {n}, target 1024", file=sys.stderr)
            ok = False
        if _p95(list(range(1, 21))) != 19 or statistics.median([1, 2, 3]) != 2:
            print("FAIL: percentile helpers", file=sys.stderr)
            ok = False
    finally:
        srv.shutdown()
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm", default="http://127.0.0.1:8000")
    ap.add_argument("--metrics", default="http://127.0.0.1:9442",
                    help="kvblockd metrics endpoint (hit verification)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lengths", default="1024,4096,8192,16384,32000",
                    help="comma-separated target prefix token counts")
    ap.add_argument("--reps", type=int, default=5, help="measured rep pairs per length")
    ap.add_argument("--warmup", type=int, default=1, help="discarded warmup pairs per length")
    ap.add_argument("--gen-tokens", type=int, default=16)
    ap.add_argument("--settle-s", type=float, default=3.0,
                    help="wait after the cold rep for LMCache's store to land in kvblockd")
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--out", default="chart2-ttft.jsonl")
    ap.add_argument("--print-prefix", default=PRINT_PREFIX)
    ap.add_argument("--stamp", action="append", default=[], metavar="K=V",
                    help="stamped into every record (gpu=, model=, vllm=, lmcache=, "
                         "tc_link=, rig=, git_sha=) — plot.py reads conditions from these")
    ap.add_argument("--selftest", action="store_true",
                    help="verify the TTFT timing logic against a local stub and exit")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    stamp = {"model": args.model}
    for kv in args.stamp:
        k, _, v = kv.partition("=")
        if not v:
            ap.error(f"--stamp needs K=V, got {kv!r}")
        stamp[k] = v
    return run_sweep(args, stamp)


if __name__ == "__main__":
    sys.exit(main())
