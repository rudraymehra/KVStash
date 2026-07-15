#!/usr/bin/env python3
"""A7 economics model: does loading KV cache beat recomputing it, and does the
cost pencil out?  Pure stdlib, every formula printed — no hidden math.

Decides kill-gate A7: same-AZ KV fetch cost < recompute cost at >=40% hit rate.

Sources (verified 2026-07):
  KV formula & MLA:  arxiv.org/html/2505.09343v2 (DeepSeek-V3)
  break-even refs:   arxiv.org/abs/2510.09665 (LMCache), arxiv.org/abs/2410.03065 (Cake),
                     tensormesh.ai/blog-posts/blog-redis-llm-kv-cache-optimization-30x-throughput
  GPU $/hr:          runpod.io/pricing  (A100 $1.39, H100 $2.89/hr)
  AWS transfer:      aws.amazon.com/ec2/pricing/on-demand
                     same-AZ private IP $0/GB; cross-AZ $0.01/GB each way; public/EIP $0.01/GB
"""

MB = 1 << 20
GB = 1 << 30


class Model:
    def __init__(self, name, layers, kv_heads, head_dim, dtype_bytes, prefill_tok_s, kv_mult=2, note=""):
        self.name = name
        self.layers = layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.dtype_bytes = dtype_bytes
        self.prefill_tok_s = prefill_tok_s  # rate the GPU can (re)produce tokens
        # kv_mult = 2 for standard MHA/GQA (separate K and V); 1 for MLA, which
        # stores a single shared compressed latent per token/layer (not K+V).
        self.kv_mult = kv_mult
        self.note = note

    def bytes_per_token(self):
        # KV = kv_mult * layers * kv_heads * head_dim * dtype_bytes
        return self.kv_mult * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes

    def kv_production_gbps(self):
        # bytes/token * tokens/s = the GB/s the GPU emits during prefill.
        return self.bytes_per_token() * self.prefill_tok_s / GB


# GQA/MLA byte counts are the HONEST ones. The MHA row is the widely-miscited
# figure (Cake's "2.5 MB/token") kept ONLY to show how wrong it is — Llama-3-70B
# actually uses 8 KV heads (GQA), so the real number is ~8x smaller.
MODELS = [
    Model("Llama-3.1-8B (GQA)",  layers=32, kv_heads=8,  head_dim=128, dtype_bytes=2, prefill_tok_s=15000),
    Model("Llama-3-70B (GQA)",   layers=80, kv_heads=8,  head_dim=128, dtype_bytes=2, prefill_tok_s=2000),
    Model("DeepSeek-V3 (MLA)",   layers=61, kv_heads=1,  head_dim=576, dtype_bytes=2, prefill_tok_s=8000, kv_mult=1,
          note="MLA: single shared latent 512+64=576 per token/layer, NOT separate K+V (kv_mult=1)"),
    Model("Llama-70B [MHA-MISCOUNT]", layers=80, kv_heads=64, head_dim=128, dtype_bytes=2, prefill_tok_s=1000,
          note="WRONG: Cake's 2.5MB/token uses 64 heads; real Llama-3-70B is GQA (8 heads). Do NOT cite."),
]

# Cost basis
GPU_USD_PER_HR = 1.39          # RunPod A100 80GB PCIe
GPU_USD_PER_SEC = GPU_USD_PER_HR / 3600
XFER_SAME_AZ = 0.0             # $/GB, private IP in-VPC
XFER_CROSS_AZ = 0.01           # $/GB each direction
# Deliverable bandwidth used for the A7 bandwidth check. This is the target /
# loopback figure (loopback already hit ~14); REPLACE with the measured cloud
# GB/s after the transport rig runs. The verdict genuinely depends on it.
DELIVERABLE_GBPS = 10.0
HIT_RATES = [0.40, 0.54, 0.62, 0.90]  # 54-62 = Alibaba Bailian band; 90 = marketing (labeled)
REUSED_TOKENS = 80_000         # a long agentic prefix (the workload we target)


def hr(title):
    print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72)


def main():
    hr("1. KV cache bytes per token  (kv_mult * layers * kv_heads * head_dim * dtype)")
    for m in MODELS:
        bpt = m.bytes_per_token()
        print(f"  {m.name:32s} {bpt:>10,d} B  = {bpt/MB:6.3f} MB/token"
              + (f"   [{m.note}]" if m.note else ""))

    hr("2. Break-even bandwidth  B* = bytes_per_token * prefill_rate")
    print("  (the rate the GPU PRODUCES KV; move finished bytes faster than this => loading wins.")
    print("   N cancels, so this is context-length-independent for O(n) prefill.)")
    for m in MODELS:
        b = m.kv_production_gbps()  # GiB/s (GB=1<<30)
        gbit = m.bytes_per_token() * m.prefill_tok_s / 1e9 * 8  # decimal Gbit/s
        print(f"  {m.name:32s} R={m.prefill_tok_s:>6,d} tok/s  ->  B* = {b:5.2f} GiB/s  ({gbit:5.1f} Gbit/s)")
    print("\n  Sanity: GQA models break even at ~0.5-1.85 GiB/s (8B is high at 1.83 due to its fast")
    print("  15k tok/s prefill); the MHA-miscount inflates to ~2.5 GiB/s,")
    print("  which is exactly why Cake/Tensormesh quote a ~2 GB/s SLO. Our 10+ GB/s target clears")
    print("  even the inflated bar with huge margin. (Long context: prefill is O(n^2), so effective R")
    print("  falls and B* drops further => loading wins even more, per LMCache's 256K crossover.)")

    hr(f"3. Per-hit cost: recompute vs fetch  (reusing {REUSED_TOKENS:,} tokens)")
    print(f"  recompute $/hit = tokens/R * $/GPU-sec   (GPU_USD_PER_SEC=${GPU_USD_PER_SEC:.6f})")
    print(f"  fetch $/hit     = bytes * $/GB (transfer)  [wait hidden by layer-wise overlap]\n")
    print(f"  {'model':32s} {'recompute$':>11s} {'sameAZ$':>9s} {'crossAZ$':>9s}")
    for m in MODELS:
        recompute = REUSED_TOKENS / m.prefill_tok_s * GPU_USD_PER_SEC
        bytes_moved = m.bytes_per_token() * REUSED_TOKENS
        same_az = bytes_moved / GB * XFER_SAME_AZ
        cross_az = bytes_moved / GB * XFER_CROSS_AZ
        print(f"  {m.name:32s} {recompute:>11.5f} {same_az:>9.5f} {cross_az:>9.5f}")

    hr("4. Amortized savings per request across hit rates")
    print("  saved/request = hit_rate * (recompute$ - fetch$)   [same-AZ, GQA 8B]")
    m = MODELS[0]
    recompute = REUSED_TOKENS / m.prefill_tok_s * GPU_USD_PER_SEC
    for h in HIT_RATES:
        saved = h * (recompute - 0.0)
        tag = "  (marketing — labeled)" if h == 0.90 else ("  (Bailian band)" if h in (0.54, 0.62) else "")
        print(f"  hit {h*100:4.0f}%  ->  ${saved:.5f}/request saved{tag}")

    hr("5. The honest failure mode: cross-AZ")
    print("  Cross-AZ transfer is $0.01/GB each way. For a CHEAP-to-prefill small model, the")
    print("  transfer $ can approach/exceed the recompute $ it saves. Rule: deploy same-AZ,")
    print("  private IP (a public/Elastic IP silently triggers $0.01/GB even within one AZ).")
    for m in MODELS[:2]:
        recompute = REUSED_TOKENS / m.prefill_tok_s * GPU_USD_PER_SEC
        cross_az = m.bytes_per_token() * REUSED_TOKENS / GB * XFER_CROSS_AZ
        verdict = "cross-AZ still cheaper than recompute" if cross_az < recompute else "cross-AZ COSTS MORE than recompute"
        print(f"  {m.name:32s} recompute ${recompute:.5f} vs cross-AZ ${cross_az:.5f}  -> {verdict}")

    hr("A7 VERDICT")
    # Real two-part gate (no longer a foregone conclusion): the deliverable
    # bandwidth must exceed EVERY honest (GQA/MLA) model's break-even, AND the
    # same-AZ fetch must be cheaper than recompute. Flip DELIVERABLE_GBPS below
    # its break-even and this FAILs, as it should.
    gqa_mla = [m for m in MODELS if "MISCOUNT" not in m.name]
    max_breakeven = max(m.kv_production_gbps() for m in gqa_mla)
    m8 = MODELS[0]
    recompute = REUSED_TOKENS / m8.prefill_tok_s * GPU_USD_PER_SEC
    same_az_fetch = XFER_SAME_AZ * (m8.bytes_per_token() * REUSED_TOKENS / GB)
    bw_ok = DELIVERABLE_GBPS > max_breakeven
    cost_ok = same_az_fetch < recompute
    ok = bw_ok and cost_ok
    print(f"  bandwidth check: deliverable {DELIVERABLE_GBPS:.1f} GiB/s > max break-even {max_breakeven:.2f} GiB/s -> {bw_ok}")
    print(f"  cost check:      same-AZ fetch ${same_az_fetch:.5f} < recompute ${recompute:.5f} -> {cost_ok}")
    print(f"  A7 {'PASS' if ok else 'FAIL'} (both must hold; verdict is computed, not assumed)")
    print("  Caveats: (1) same-AZ private IP only; (2) GQA/MLA byte counts, never the MHA 2.5MB;")
    print("           (3) <40% hit rate may not clear amortized infra cost;")
    print("           (4) DELIVERABLE_GBPS is the target/loopback figure — replace with measured cloud GB/s.")


if __name__ == "__main__":
    main()
