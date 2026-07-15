# kvblockd — Design & kill-gate results

Kill-gate verdicts and load-bearing design decisions. Kill-gates are pre-registered
(A1–A8); a FAIL executes its written consequence.

## A7 — same-AZ economics: does loading beat recompute, and does it pencil out?

**Gate:** same-AZ KV-cache fetch cost < recompute cost at ≥40% hit rate.
**Verdict: PASS** (same-AZ, GQA/MLA models). Model: `bench/e2e/economics.py` (every formula printed).

**The break-even identity:** loading beats recompute above `B* = bytes_per_token × prefill_rate` — literally the rate the GPU *produces* KV during prefill. It's context-length-independent for O(n) prefill (the token count cancels).

| Model | KV bytes/token | Break-even B* |
|---|---|---|
| Llama-3.1-8B (GQA) | 131,072 B (0.125 MB) | ~1.83 GB/s |
| Llama-3-70B (GQA) | 327,680 B (0.312 MB) | ~0.61 GB/s |
| DeepSeek-V3 (MLA) | 70,272 B (0.067 MB) | ~1.05 GB/s |
| *Llama-70B [MHA-miscount]* | *2.5 MB* | *~2.44 GB/s* |

GQA/MLA models break even at ~0.6–1.8 GB/s; kvblockd's 10+ GB/s target clears it with large margin. The commonly-cited "2.5 MB/token / ~2 GB/s SLO" (Cake, Tensormesh) is calibrated to the **MHA miscount** — real GQA Llama-3-70B is 8× smaller. We use GQA/MLA counts and never the 2.5 MB figure.

**Cost:** same-AZ private-IP transfer is **$0/GB**, so a hit's cost is ~zero while recompute burns GPU-seconds ($0.002–0.015/hit reusing 80K tokens on an A100). PASS at every hit rate ≥40%.

**Honest caveats (recorded, not hidden):**
1. **Same-AZ private IP only.** Cross-AZ transfer is $0.01/GB each way and *exceeds* recompute for cheap-to-prefill small models ($0.098 vs $0.002 for 8B) — the deployment guide must mandate same-AZ; a public/Elastic IP silently triggers $0.01/GB even within one AZ.
2. Use GQA/MLA byte counts, never the MHA 2.5 MB figure (8× overstatement).
3. Below ~40% hit rate, amortized infra cost may not clear recompute.

Sources: LMCache (arXiv 2510.09665), Cake (arXiv 2410.03065), Tensormesh Redis blog, DeepSeek-V3 (arXiv 2505.09343), AWS EC2 data-transfer pricing, RunPod pricing. Deliverable-bandwidth uses the loopback/target figure now; re-run with the real cloud GB/s after the transport rig runs.
