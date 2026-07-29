# Rig G — EC2 L40S long-context TTFT rig

The EC2 leg of the Chart-2 methodology: same two-phase harness, same gates,
same pinned engine image as `bench/rigs/hf-gpu/` (which produced every
published A10G table), driven over SSH onto a single `g6e.2xlarge`
(1× L40S 48 GB) instead of an HF Job. The measurement core —
`run_ttft.py`, `equivalence.py`, `job.sh` — is reused byte-for-byte;
`job.sh` stamps `rig=ec2-g6e-l40s` here via its `RIG`/`GPU_ANNOT` envs.

What this rig adds over the HF leg:

- **48 GB VRAM** → true long-context points: Qwen2.5-7B-Instruct-1M at
  96k–262k tokens (served in the model-card-sanctioned standard-attention
  mode: `dual_chunk_attention_config` is deleted from the local snapshot —
  DCA has no backend on vLLM 0.25.x — and the identical edited snapshot
  serves BOTH arms; disclosed in every JSONL row via the model path),
  and Llama-3.1-8B up to its native 131,072.
- **Root access** → the two-node real-NIC / tc-shaped sessions
  (`iperf3 ceiling first, quote the ratio` — methodology rule 2).

## Flow

```sh
DRY_RUN=1 bench/rigs/aws-gpu/provision.sh   # red-proof: print, launch nothing
bench/rigs/aws-gpu/provision.sh             # launch (AZ ladder, dead-man armed)
bench/rigs/aws-gpu/node-setup.sh            # GPU assert, image, models, repo
bench/rigs/aws-gpu/run-arm.sh a0b run1      # calibration block first, always
bench/rigs/aws-gpu/run-arm.sh arm9 run1     # then pre-registered arms
bench/rigs/aws-gpu/teardown.sh              # session over only at ALL-CLEAR
```

Arms are pre-registered in `arms.sh` (context points, REPS, dtype, engine
flags, connector overrides — engine args are never tuned mid-session; the
chunk-size knob is swept once to MINIMIZE THE BASELINE's TTFT, then frozen
identically in both arms of every run). `run-arm.sh` pulls all artifacts
back after every arm — a dead-man terminate never loses more than the
in-flight rep — and judges the honesty gates from the pulled logs:
exact-count hit verification, zero-occurrence deadline/degrade greps,
path-stamp presence, baseline-purity, and the fp8 backend line.

## Cost discipline

Every launch: `--instance-initiated-shutdown-behavior terminate`,
`DeleteOnTermination` on the volume (both asserted post-launch), a
`shutdown -h` dead-man armed from user-data, everything tagged `kvbench`.
`run-arm.sh` appends wall-clock × rate to a local ledger; `teardown.sh`
verifies $0 residue (instances, volumes, EIPs, capacity reservations,
placement groups, NAT) and prints the session total.

Results bank to `bench/results/rig-g/`; render/aggregate with the same
`bench/report/{plot,aggregate}.py` — rig-e (A10G) and rig-g (L40S) files
are never co-globbed into one aggregate (a faster GPU shrinks the multiple;
GPU change is always disclosed, never averaged away).

*(The pre-2026-07-29 content of this file — the LMCache-era shaped-link
plan — is superseded; its shaped-link and hit-rate-sweep goals move to the
two-node session driven from this same rig, against the native connector.)*
