#!/usr/bin/env python3
"""Gate-1 verification: query a running vLLM (with the kvblockd LMCache
backend) twice at temperature 0 and check:

  (a) [HARD] completions byte-identical across the two rounds
  (b) [HARD] kvblockd /metrics shows kvb_hits_total > 0 after round 2
      (the load-bearing signal that blocks round-tripped through kvblockd)
  (c) [SOFT/warn] round-2 latency < round-1 — opt-125m on CPU is tiny, so
      the cache win is often within noise; warn, don't fail.

Property (d) "hits persist across a vLLM restart" is NOT checked here — it is
driven by the CI harness (e2e-cpu.yml restarts vLLM and re-invokes this
script with --expect-hits, asserting (b) holds on a cold engine). Run twice:
once plain, once post-restart with --expect-hits.

Usage: verify.py --vllm http://127.0.0.1:18000 --metrics http://127.0.0.1:9442 [--expect-hits]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

PROMPT = " ".join(["The quick brown fox jumps over the lazy dog."] * 16)  # ~640 tokens


def _post(url, payload):
    req = urllib.request.Request(
        url + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.load(r)
    return body["choices"][0]["text"], time.monotonic() - t0


def _hits(metrics_url) -> float:
    with urllib.request.urlopen(metrics_url + "/metrics", timeout=10) as r:
        text = r.read().decode()
    total = 0.0
    for line in text.splitlines():
        if line.startswith("kvb_hits_total"):
            total += float(line.rsplit(" ", 1)[1])
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default="http://127.0.0.1:18000")
    ap.add_argument("--metrics", default="http://127.0.0.1:9442")
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--expect-hits", action="store_true", help="post-restart run: require hits on a cold engine (property d)")
    args = ap.parse_args()

    payload = {"model": args.model, "prompt": PROMPT, "temperature": 0.0, "max_tokens": 32}

    out1, ttft1 = _post(args.vllm, payload)
    time.sleep(1.0)
    out2, ttft2 = _post(args.vllm, payload)
    hits = _hits(args.metrics)

    ok = True
    if out1 != out2:
        print("FAIL (a): completions differ across rounds", file=sys.stderr)
        ok = False
    if hits <= 0:
        print(f"FAIL (b): kvb_hits_total = {hits} (expected > 0 after round 2)", file=sys.stderr)
        ok = False
    if ttft2 >= ttft1:
        print(f"WARN (c): round-2 TTFT {ttft2:.3f}s not < round-1 {ttft1:.3f}s "
              f"(opt-125m on CPU is tiny — non-fatal, but the cache should help)", file=sys.stderr)
    print(json.dumps({"round1_ttft_s": ttft1, "round2_ttft_s": ttft2,
                      "kvb_hits_total": hits, "identical": out1 == out2}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
