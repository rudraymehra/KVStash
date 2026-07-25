#!/usr/bin/env python3
"""Render both headline charts FROM COMMITTED JSONL ALONE (methodology rule
12, bench/METHODOLOGY.md): a stranger with the results file regenerates the
README images, no rig access needed.

  Chart 1  throughput bars vs the field, iperf3 ceiling drawn in, %-of-
           ceiling + cores annotated, two panels (0.44 / 2.5 MiB).
  Chart 2  two record shapes, auto-detected:
           - kind:"ttft" (hf-gpu rig): TTFT vs prefix length, cold-recompute
             vs warm-kvblockd-reload, p50 solid / p95 dashed, per-length
             speedup annotated, unverified warm cells flagged.
           - replay/hit_rate records (aws rig-e plan): TTFT vs hit rate,
             p50 solid / p99 dashed, shaded "recompute wins here" region.

Usage:
  plot.py chart1 --in results/rig-t/*.jsonl --out chart1.png
  plot.py chart2 --in results/rig-e/*.jsonl --out chart2.png
"""
import argparse
import glob
import json
import sys

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib not installed (pip install matplotlib)")


def load(patterns):
    recs = []
    for pat in patterns:
        for path in glob.glob(pat):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
    return recs


# Bar order + display names for Chart 1 (kvblockd first, floor last).
CHART1_ORDER = [
    ("kvblockd", "kvblockd (DRAM)"),
    ("kvblockd-nvme", "kvblockd (NVMe)"),
    ("mooncake", "Mooncake TCP"),
    ("valkey", "Valkey 8 (go-redis)"),
    ("redis", "Redis 7 (go-redis)"),
    ("redis-py", "Redis 7 (redis-py)"),
    ("nvmefs", "local NVMe-fs floor"),
]


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def chart1(recs, out):
    # Headline: GET-only, batch 32, closed-loop, best-streams-per-store,
    # median of runs. Two panels by blob size.
    panels = [(462848, "0.44 MiB block"), (2621440, "2.5 MiB block")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=False)
    all_nruns = []  # every bar's run count, across both panels
    for ax, (blob, title) in zip(axes, panels):
        labels, goodputs, ratios, cores, nruns, streams = [], [], [], [], [], []
        ceiling = 0.0
        for store, disp in CHART1_ORDER:
            cell = [
                r for r in recs
                if r.get("store") == store
                and r.get("cell", {}).get("mode") == "closed"
                and r.get("cell", {}).get("mix", "get") == "get"
                and r.get("cell", {}).get("blob_bytes") == blob
            ]
            if not cell:
                continue
            # best streams = the max goodput cell; median across its runs.
            best = max(cell, key=lambda r: r.get("goodput_gbytes_s", 0))
            same = [
                r["goodput_gbytes_s"] for r in cell
                if r.get("cell", {}).get("streams") == best["cell"]["streams"]
            ]
            gp = _median(same)
            labels.append(disp)
            goodputs.append(gp)
            nruns.append(len(same))
            streams.append(best.get("cell", {}).get("streams", 0))
            # ONLY annotate %-of-ceiling when a real measured ceiling exists —
            # never fabricate one by dividing by 1 (an earlier review caught a
            # fabricated annotation here). ratio is None → no "% ceil" text.
            c = best.get("ceiling_gbytes_s", 0) or 0
            ratios.append((gp / c) if c > 0 else None)
            # CPU: daemon cores and client cores are different claims — label
            # whichever this record actually has, never one as the other.
            cpu = best.get("cpu", {}) or {}
            if cpu.get("daemon_cores"):
                cores.append((cpu["daemon_cores"], "daemon"))
            elif cpu.get("client_cores"):
                cores.append((cpu["client_cores"], "client"))
            else:
                cores.append((0, ""))
            ceiling = max(ceiling, c)

        y = range(len(labels))
        ax.barh(list(y), goodputs, color="#2b6cb0")
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("GB/s (payload-only, decimal)")
        ax.set_title(title)
        # Bars may sit at different best-stream points and run counts, so
        # each bar carries its own streams and n — a single figure-wide
        # number would overstate the thin bars.
        for i, (gp, ra, co, nr, st) in enumerate(zip(goodputs, ratios, cores, nruns, streams)):
            note = f"{gp:.1f} GB/s"
            if ra is not None:
                note += f"  ({ra * 100:.0f}% ceil)"
            cval, cwho = co
            if cval:
                note += f"  {cval:.2f} {cwho} cores"
            if st:
                note += f"  {st}str"
            note += f"  [n={nr}]"
            all_nruns.append(nr)
            ax.text(gp, i, "  " + note, va="center", fontsize=8)
        # Stores in CHART1_ORDER with no data are absent by omission; note it
        # so "missing" never reads as "benchmarked at zero".
        present = set(labels)
        missing = [d for _, d in CHART1_ORDER if d not in present]
        if missing:
            ax.text(0.98, 0.02, "no data: " + ", ".join(missing),
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="grey")
        if ceiling > 0:
            ax.axvline(ceiling, color="crimson", linestyle="--")
            ax.text(ceiling, len(labels) - 0.5, f" iperf3 ceiling {ceiling:.1f} GB/s",
                    color="crimson", fontsize=8, rotation=90, va="bottom")
    # The title quotes the WORST-covered bar (min runs), never the best; when
    # coverage differs per bar the exact n is on each bar.
    lo, hi = (min(all_nruns), max(all_nruns)) if all_nruns else (0, 0)
    runs_label = f"median of {lo}" if lo == hi else f"median per bar, n={lo}–{hi} (per-bar n shown)"
    if lo < 3:
        runs_label += " — below the median-of-3 rule"
    fig.suptitle(f"Throughput vs the field — same wire, same ops (GET-only, batch 32, warmed, {runs_label};\n"
                 "best streams per store, shown per bar as Nstr)  "
                 "Repro: bench/rigs/aws-transport/run-chart1.sh — ~$12 of spot", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def _conditions(recs):
    """Conditions box read FROM the JSONL (never hardcoded — review rule).
    Rigs stamp gpu/model/vllm/lmcache/tc_link into each record."""
    cond = {}
    for r in recs:
        for k in ("gpu", "model", "vllm", "lmcache", "tc_link"):
            if r.get(k):
                cond.setdefault(k, r[k])
    box = ", ".join(f"{cond[k]}" for k in ("gpu", "model", "vllm", "lmcache") if k in cond)
    return box, cond.get("tc_link", "link undisclosed")


def chart2_ttft(recs, out):
    # TTFT vs prefix length, cold (recompute) vs warm (kvblockd reload).
    # One record per (length, arm); duplicates (re-runs) collapse by median.
    cells = {}
    for r in recs:
        x = r.get("target_prefix_tokens") or r.get("prefix_tokens")
        if not x or r.get("arm") not in ("cold", "warm"):
            continue
        cells.setdefault((r["arm"], r.get("series", r["arm"]), x), []).append(r)

    series, unverified, nreps = {}, [], []
    for (arm, label, x), rs in sorted(cells.items()):
        p50 = _median([r.get("ttft_p50_ms", r.get("ttft_ms", 0)) for r in rs])
        p95 = _median([r.get("ttft_p95_ms", 0) for r in rs])
        series.setdefault((arm, label), []).append((x, p50, p95))
        nreps.extend(r.get("reps", 0) for r in rs)
        if arm == "warm" and any(r.get("warm_hits_verified") is False for r in rs):
            unverified.append(x)

    fig, ax = plt.subplots(figsize=(9, 6))
    by_arm = {}
    for (arm, label), pts in sorted(series.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        p50s = [p[1] for p in pts]
        p95s = [p[2] for p in pts]
        line, = ax.plot(xs, p50s, marker="o", label=f"{label} p50")
        ax.plot(xs, p95s, marker="x", linestyle="--", color=line.get_color(),
                label=f"{label} p95")
        by_arm[arm] = dict(zip(xs, p50s))

    # Per-length speedup (the chart's whole point) annotated at the warm p50;
    # <1x means recompute won there — printed just as honestly.
    cold, warm = by_arm.get("cold", {}), by_arm.get("warm", {})
    for x in sorted(set(cold) & set(warm)):
        if warm[x] > 0:
            note = f"{cold[x] / warm[x]:.1f}x"
            if x in unverified:
                note += " (UNVERIFIED)"
            ax.annotate(note, (x, warm[x]), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8,
                        color="crimson" if x in unverified else "black")

    ax.set_xscale("log", base=2)
    xs_all = sorted({x for pts in series.values() for x, _, _ in pts})
    ax.set_xticks(xs_all)
    ax.set_xticklabels([f"{round(x / 1000)}k" if x >= 1000 else str(x) for x in xs_all])
    ax.minorticks_off()
    ax.set_yscale("log")
    ax.set_xlabel("prefix length (tokens)")
    ax.set_ylabel("TTFT (ms, log)")

    box, link = _conditions(recs)
    lo, hi = (min(nreps), max(nreps)) if nreps else (0, 0)
    runs = f"n={lo}" if lo == hi else f"n={lo}–{hi}"
    ax.set_title("TTFT: reload KV from kvblockd vs recompute — same box, same model\n"
                 f"median/p95 of {runs} reps per point, warmup discarded\n"
                 f"{box or 'conditions from JSONL'}; {link} — disclosed", fontsize=9)
    if unverified:
        ax.text(0.02, 0.02, "warm hit UNVERIFIED at: "
                + ", ".join(str(x) for x in unverified),
                transform=ax.transAxes, fontsize=8, color="crimson")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def chart2(recs, out):
    # hf-gpu TTFT-sweep records render the prefix-length chart; otherwise
    # fall through to the original TTFT-vs-hit-rate rendering.
    ttft_recs = [r for r in recs if r.get("kind") == "ttft"]
    if ttft_recs:
        return chart2_ttft(ttft_recs, out)
    # TTFT vs hit rate. Series by store/config; p50 solid, p99 dashed.
    series = {}
    for r in recs:
        if r.get("kind") != "replay" and "hit_rate" not in r:
            continue
        label = r.get("series") or r.get("store", "?")
        hr = r.get("hit_rate")
        g = r.get("ops", {}).get("get", {})
        p50 = g.get("p50_us", 0) / 1000.0  # → ms
        p99 = g.get("p99_us", 0) / 1000.0
        ttft = r.get("ttft_ms")  # GPU harness writes this directly
        if ttft is not None:
            p50 = ttft
            p99 = r.get("ttft_p99_ms", ttft)
        if hr is None:
            continue
        series.setdefault(label, []).append((hr, p50, p99))

    fig, ax = plt.subplots(figsize=(9, 6))
    crossover = None
    recompute = None
    for label, pts in sorted(series.items()):
        pts.sort()
        xs = [p[0] * 100 for p in pts]
        p50s = [p[1] for p in pts]
        p99s = [p[2] for p in pts]
        line, = ax.plot(xs, p50s, marker="o", label=f"{label} p50")
        ax.plot(xs, p99s, marker="x", linestyle="--", color=line.get_color(), label=f"{label} p99")
        if "recompute" in label.lower():
            recompute = _median(p50s)

    # Shade "recompute wins here": the region left of where the best remote
    # series drops below the recompute flat line.
    if recompute:
        ax.axhline(recompute, color="grey", linestyle=":", label="recompute baseline")
        # crossover = smallest hit rate where SOME non-recompute series beats it
        best_x = None
        for label, pts in series.items():
            if "recompute" in label.lower():
                continue
            for hr, p50, _ in sorted(pts):
                if p50 < recompute:
                    best_x = hr * 100 if best_x is None else min(best_x, hr * 100)
                    break
        crossover = best_x
        if crossover is not None:
            ax.axvspan(0, crossover, color="crimson", alpha=0.08)
            ax.text(crossover / 2, ax.get_ylim()[1] * 0.9,
                    "recompute wins here —\ndon't deploy a remote\ncache for this workload",
                    ha="center", va="top", fontsize=9, color="crimson")

    # Conditions box read FROM the JSONL (was once hardcoded — a review
    # caught it; never hardcode conditions):
    # the GPU harness stamps gpu/model/vllm/lmcache/tc_link into each record.
    cond = {}
    for r in recs:
        for k in ("gpu", "model", "vllm", "lmcache", "tc_link"):
            if r.get(k):
                cond.setdefault(k, r[k])
    box = ", ".join(f"{cond[k]}" for k in ("gpu", "model", "vllm", "lmcache") if k in cond)
    link = cond.get("tc_link", "tc-emulated link")
    ax.set_xlabel("KV-cache hit rate (%)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT vs hit rate — when a remote cache helps, and when it doesn't\n"
                 f"({box or 'conditions from JSONL'}; {link} — disclosed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}" + (f" (crossover ≈ {crossover:.0f}%)" if crossover else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chart", choices=["chart1", "chart2"])
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = load(args.inputs)
    if not recs:
        sys.exit("no records loaded")
    (chart1 if args.chart == "chart1" else chart2)(recs, args.out)


if __name__ == "__main__":
    main()
