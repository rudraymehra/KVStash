#!/usr/bin/env bash
# kvbench arm table — every pre-registered cell of the L40S session as an
# env emitter. `arm_env <name>` prints KEY=VALUE lines consumed by
# run-arm.sh. The table (predictions, REPS caps, fallback ladders) is
# pre-registered; engine args are never tuned mid-session.
#
# Session-level choices frozen by the calibration block and passed via env:
#   MNBT       — max_num_batched_tokens winner of the a0c sweep (default 16384;
#                chosen to MINIMIZE THE BASELINE's TTFT, then frozen in both
#                arms of every run — the inverse of the vendor move)
#   KVB_FANOUT — get_fanout winner of the a0d A/B (default 4)
set -euo pipefail

# Qwen-1M is always served via job.sh STRIP_DCA from the warm HF cache —
# node-setup no longer materializes a pre-stripped local dir.
QWEN=Qwen/Qwen2.5-7B-Instruct-1M
LLAMA=/nvme/hf/llama31
MNBT="${MNBT:-16384}"
KVB_FANOUT="${KVB_FANOUT:-4}"

# store queue bytes = max(4 GiB, 3 x MNBT x KV bytes/token): one chunked-
# prefill step stages MNBT*bytes in a burst and the write-behind queue must
# absorb three steps or the populate tail-skips (gate: dropped=0 failed=0).
queue_bytes() { # $1 = kv bytes/token
  local need=$((3 * MNBT * $1)) four=$((4 * 1073741824))
  echo $(( need > four ? need : four ))
}

common() { # $1 model path, $2 kv bytes/token, $3 L
  cat <<EOF
MODEL=$1
$( [[ "$1" == "$QWEN" ]] && printf 'STRIP_DCA=1\nHF_HOME=/nvme/hf' )
KV_BYTES_PER_TOKEN=$2
LENGTHS=$3
GEN_TOKENS=16
MAX_NUM_BATCHED_TOKENS=$MNBT
RIG=ec2-g6e-l40s
GPU_ANNOT=ec2 g6e.2xlarge
KVBD_STREAMS=8
KVBD_STORE_DRAIN_WORKERS=4
KVBD_STORE_FLUSH_TIMEOUT_S=120
KVBD_LOAD_DEADLINE_S=120
KVBD_EXISTS_TIMEOUT_S=2.0
KVBD_GET_FANOUT=$KVB_FANOUT
KVBD_STORE_QUEUE_BYTES=$(queue_bytes "$2")
REQUEST_TIMEOUT=$(( $3 >= 196608 ? 900 : 600 ))
EOF
}


# A10G arm env: fp8-only, MNBT 8192 (24 GB activation budget), util 0.92.
a10g_common() { # $1 = LENGTHS
  cat <<AEOF
MODEL=Qwen/Qwen2.5-7B-Instruct-1M
STRIP_DCA=1
HF_HOME=/nvme/hf
KV_CACHE_DTYPE=fp8_e4m3
KV_BYTES_PER_TOKEN=28672
EQUIV_PROMPT_SET=/work/kvstash/bench/e2e/equiv-prompts-high-margin-v2.txt
EQUIV_MARGIN_FLOOR=2.0
EQUIV_GEN_TOKENS=8
LENGTHS=$1
GEN_TOKENS=16
GPU_MEM_UTIL=0.92
MAX_NUM_BATCHED_TOKENS=8192
RIG=ec2-g5-a10g
GPU_ANNOT=ec2 g5.2xlarge
KVBD_STREAMS=8
KVBD_STORE_DRAIN_WORKERS=4
KVBD_STORE_FLUSH_TIMEOUT_S=120
KVBD_LOAD_DEADLINE_S=120
KVBD_EXISTS_TIMEOUT_S=2.0
KVBD_GET_FANOUT=$KVB_FANOUT
KVBD_STORE_QUEUE_BYTES=4294967296
REQUEST_TIMEOUT=900
AEOF
}

# Two-node real-NIC arms: the store lives on the store node, so the connector
# is pointed across the NIC (job.sh EXTERNAL_DAEMON mode), the arena is the
# store host's RAM, and TC_LINK carries whatever shaping the session applied.
# fp8 halves the payload the wire must carry. Certification uses the frozen
# MEASUREMENT-SELECTED corpus v2 + margin screen + 8-token window (CLAIMS.md
# fp8 Amendments 2-3). Provenance, stated precisely: session-2's banked
# two-node fp8 refusal ran the LEGACY no-corpus prompt builder (its record
# carries no prompt_corpus key — that code predates corpus support; the v1
# path these arms once named was inert env). Corpus v1 itself was disproven
# by its own 16-prompt certification run, not by the 48-candidate
# calibration (Amendment 3's precision note; both records banked under
# bench/results/rig-g/refusals/).
nic_common() { # $1 = LENGTHS
  # Refuse to emit a nic arm without its two required session facts: where
  # the store is, and what the link was shaped to ("unshaped" is explicit).
  : "${STORE_PRIVATE_IP:?nic arms need STORE_PRIVATE_IP (from kvbench-dday/store-private-ip)}"
  : "${STORE_TC_GBIT:?nic arms need STORE_TC_GBIT set to a gbit number or the word unshaped - the link rate is the independent variable}"
  cat <<NEOF
MODEL=Qwen/Qwen2.5-7B-Instruct-1M
STRIP_DCA=1
HF_HOME=/nvme/hf
KV_CACHE_DTYPE=fp8_e4m3
KV_BYTES_PER_TOKEN=28672
EQUIV_PROMPT_SET=/work/kvstash/bench/e2e/equiv-prompts-high-margin-v2.txt
EQUIV_MARGIN_FLOOR=2.0
EQUIV_GEN_TOKENS=8
EQUIV_N=24
LENGTHS=$1
GEN_TOKENS=16
GPU_MEM_UTIL=0.92
MAX_NUM_BATCHED_TOKENS=8192
RIG=ec2-twonode-nic
GPU_ANNOT=ec2 g6e.2xlarge + r6in.2xlarge store (fp8)
EXTERNAL_DAEMON=$STORE_PRIVATE_IP:9440
EXTERNAL_METRICS=$STORE_PRIVATE_IP:9442
TC_LINK=two-node real NIC fp8, store egress ${STORE_TC_GBIT} (htb+fq; iperf3 ceiling in session artifacts)
KVBD_STREAMS=8
KVBD_STORE_DRAIN_WORKERS=4
KVBD_STORE_FLUSH_TIMEOUT_S=180
KVBD_LOAD_DEADLINE_S=300
KVBD_EXISTS_TIMEOUT_S=3.0
KVBD_GET_FANOUT=$KVB_FANOUT
KVBD_STORE_QUEUE_BYTES=4294967296
REQUEST_TIMEOUT=900
NEOF
}

# bf16 variant of the two-node arms — the session-2 published cells. bf16
# certified cleanly in every run; the fp8 attempt of that session was
# refused by the LEGACY no-corpus gate (a gated, self-consistent prompt
# mismatched after restart — Amendment 3 declined to attribute a mechanism,
# and so does this comment). The fp8 arms above run under corpus v2 + the
# padded prescreen instead. bf16 stays valuable: bigger payload, the wire
# matters MORE, no dtype caveat.
nic_bf16_common() { # $1 = LENGTHS
  : "${STORE_PRIVATE_IP:?nic arms need STORE_PRIVATE_IP (from kvbench-dday/store-private-ip)}"
  : "${STORE_TC_GBIT:?nic arms need STORE_TC_GBIT set to a gbit number or the word unshaped}"
  cat <<NBF
MODEL=Qwen/Qwen2.5-7B-Instruct-1M
STRIP_DCA=1
HF_HOME=/nvme/hf
KV_BYTES_PER_TOKEN=57344
LENGTHS=$1
GEN_TOKENS=16
GPU_MEM_UTIL=0.92
MAX_NUM_BATCHED_TOKENS=8192
RIG=ec2-twonode-nic
GPU_ANNOT=ec2 g6e.2xlarge + r6in.2xlarge store (bf16)
EXTERNAL_DAEMON=$STORE_PRIVATE_IP:9440
EXTERNAL_METRICS=$STORE_PRIVATE_IP:9442
TC_LINK=two-node real NIC bf16, store egress ${STORE_TC_GBIT} (htb+fq; iperf3 ceiling in artifacts)
KVBD_STREAMS=8
KVBD_STORE_DRAIN_WORKERS=4
KVBD_STORE_FLUSH_TIMEOUT_S=180
KVBD_LOAD_DEADLINE_S=300
KVBD_EXISTS_TIMEOUT_S=3.0
KVBD_GET_FANOUT=$KVB_FANOUT
KVBD_STORE_QUEUE_BYTES=6442450944
REQUEST_TIMEOUT=900
NBF
}

# fp8 no-connector baseline on the GPU node, one length per invocation so the
# engine boots the exact max-model-len pin of the measured cell it divides
# (PR-1 hashes max-model-len; the denominator must share the fingerprint).
nic_base_fp8() { # $1 = single LENGTH
  cat <<NBEOF
MODEL=Qwen/Qwen2.5-7B-Instruct-1M
STRIP_DCA=1
HF_HOME=/nvme/hf
KV_CACHE_DTYPE=fp8_e4m3
KV_BYTES_PER_TOKEN=28672
LENGTHS=$1
GEN_TOKENS=16
GPU_MEM_UTIL=0.92
MAX_NUM_BATCHED_TOKENS=8192
RIG=ec2-twonode-nic
GPU_ANNOT=ec2 g6e.2xlarge (baseline, no connector)
BASELINE_ONLY=1
REPS=2
WARMUP=1
REQUEST_TIMEOUT=$(( $1 >= 196608 ? 900 : 600 ))
NBEOF
}

arm_env() {
  case "$1" in
    # ---- calibration block (runs first; re-anchors predictions) ----
    a0b)  common "$QWEN" 57344 16384,32768; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    a0c-*) # baseline-only chunk-size fairness sweep at the top point
          local m="${1#a0c-}"
          common "$QWEN" 57344 262144 | sed "s/^MAX_NUM_BATCHED_TOKENS=.*/MAX_NUM_BATCHED_TOKENS=$m/"
          echo "BASELINE_ONLY=1"; echo "REPS=2"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    a0d)  common "$QWEN" 57344 16384 | sed 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=8/;s/^KVBD_STREAMS=.*/KVBD_STREAMS=8/'
          echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    a0e)  common "$QWEN" 28672 16384; echo "FP8_PREFLIGHT=fp8_e4m3"; echo "GPU_MEM_UTIL=0.90" ;;
    a0f)  common "$QWEN" 28672 131072; echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=1"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    # ---- baselines (denominator series; NO connector; identical flags) ----
    base-qwen-bf16) common "$QWEN" 57344 98304,131072,163840,196608,262144
          echo "BASELINE_ONLY=1"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    base-qwen-fp8)  common "$QWEN" 28672 98304,131072,163840,196608,262144
          echo "BASELINE_ONLY=1"; echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    base-llama)     common "$LLAMA" 131072 98304,129024
          echo "BASELINE_ONLY=1"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.92" ;;
    # ---- the pre-registered arm table (numbers = runbook) ----
    arm1)  common "$QWEN" 28672 98304;  echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm2)  common "$QWEN" 28672 131072; echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm3)  common "$QWEN" 57344 98304;  echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm4)  common "$QWEN" 57344 131072; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm5)  common "$QWEN" 28672 163840; echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm6)  common "$QWEN" 57344 163840; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm7)  common "$QWEN" 28672 196608; echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm8)  common "$QWEN" 57344 196608; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm9)  common "$QWEN" 28672 262144; echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    # arm10 fallback ladder (pre-registered): 262144 -> 229376 -> 196608.
    # bf16 arena caps pairs at 3 for 256k (Appendix A) -> REPS=2+1.
    arm10) common "$QWEN" 57344 262144; echo "REPS=2"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm11) common "$LLAMA" 131072 98304; echo "REPS=2"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.92" ;;
    # arm12 VRAM knife-edge, LAST; fallback 129024 -> 112640 -> 98304;
    # arena caps REPS at 1+1, n=3 via three fresh runs.
    arm12) common "$LLAMA" 131072 129024; echo "REPS=1"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.92" ;;
    # ---- A10G (g5.2xlarge, 24 GB VRAM / 32 GiB RAM) long-context set ----
    # The multiple-friendly box: slower prefill, same reload physics (PR-4
    # discloses the GPU class on every chart). Qwen-1M fp8 only — bf16 KV
    # does not fit these lengths in 24 GB. One length per arm: the 32 GiB
    # host must hold arena+staging+engine, so arms never share a daemon.
    # STRIP_DCA does the disclosed surgery per arm from the warm HF cache.
    a10g-cal)  a10g_common 16384,32768; echo "REPS=3"; echo "WARMUP=1" ;;
    a10g-64k)  a10g_common 65536;  echo "REPS=2"; echo "WARMUP=1" ;;
    a10g-96k)  a10g_common 98304;  echo "REPS=2"; echo "WARMUP=1" ;;
    a10g-128k) a10g_common 131072; echo "REPS=2"; echo "WARMUP=1" ;;
    # 160k knife-edge probe: 4.6 GB fp8 KV + 15.2 weights + ~1.5 act on 22.3
    # usable — pre-registered fallback: OOM -> publish nothing, cost ~$0.30.
    a10g-160k) a10g_common 163840; echo "REPS=2"; echo "WARMUP=1" ;;
    a10g-base-160k) a10g_common 163840; echo "BASELINE_ONLY=1"; echo "REPS=2"; echo "WARMUP=1" ;;
    # 192k: the true VRAM edge (5.6 GB fp8 KV); util 0.93 both arms of this
    # cell, disclosed. Pre-registered fallback on OOM: publish nothing.
    a10g-192k) a10g_common 196608 | sed 's/^GPU_MEM_UTIL=.*/GPU_MEM_UTIL=0.93/'; echo "REPS=2"; echo "WARMUP=1" ;;
    a10g-base-192k) a10g_common 196608 | sed 's/^GPU_MEM_UTIL=.*/GPU_MEM_UTIL=0.93/'; echo "BASELINE_ONLY=1"; echo "REPS=2"; echo "WARMUP=1" ;;
    a10g-base) a10g_common 65536,98304,131072; echo "BASELINE_ONLY=1"; echo "REPS=2"; echo "WARMUP=1" ;;
    a10g-base-cal) a10g_common 16384,32768; echo "BASELINE_ONLY=1"; echo "REPS=3"; echo "WARMUP=1" ;;
    # fp8 baselines on THIS box for the nic multiples' denominator (same GPU,
    # same dtype, no connector) — ONE LENGTH PER ARM so every denominator
    # boots the exact max-model-len pin of the cell it divides (job.sh
    # derives max-model-len from max(LENGTHS) and hashes it into
    # engine_args_sha; a multi-length baseline sweep would put a different
    # engine fingerprint under the 32k/64k multiples than over them).
    nic-base-32k)  nic_base_fp8 32768 ;;
    nic-base-64k)  nic_base_fp8 65536 ;;
    nic-base-128k) nic_base_fp8 131072 ;;
    # PR-2 check cell: the a0c chunk-size sweep winner on this box class was
    # 16384 (measured for bf16 loopback at 262k). The nic arms freeze 8192
    # for same-pin comparability with the banked bf16 two-node cells; this
    # cell quantifies what that choice does to the fp8 denominator at 131k
    # (a slower-than-best baseline inflates the multiple — the vendor move).
    nic-base-128k-m16) nic_base_fp8 131072 | sed 's/^MAX_NUM_BATCHED_TOKENS=.*/MAX_NUM_BATCHED_TOKENS=16384/' ;;
    nicb-32k)  nic_bf16_common 32768;  echo "REPS=3"; echo "WARMUP=1" ;;
    nicb-64k)  nic_bf16_common 65536;  echo "REPS=2"; echo "WARMUP=1" ;;
    nicb-128k) nic_bf16_common 131072; echo "REPS=2"; echo "WARMUP=1" ;;
    nicb-base) cat <<NBBEOF
MODEL=Qwen/Qwen2.5-7B-Instruct-1M
STRIP_DCA=1
HF_HOME=/nvme/hf
KV_BYTES_PER_TOKEN=57344
LENGTHS=32768,65536,131072
GEN_TOKENS=16
GPU_MEM_UTIL=0.92
MAX_NUM_BATCHED_TOKENS=8192
RIG=ec2-twonode-nic
GPU_ANNOT=ec2 g6e.2xlarge (bf16 baseline, no connector)
BASELINE_ONLY=1
REPS=2
WARMUP=1
NBBEOF
          ;;
    # ---- two-node real-NIC, fp8 (PR-10 session: corpus-v2 certified path) ----
    nic-32k)  nic_common 32768;  echo "REPS=3"; echo "WARMUP=1" ;;
    nic-64k)  nic_common 65536;  echo "REPS=2"; echo "WARMUP=1" ;;
    nic-128k) nic_common 131072; echo "REPS=2"; echo "WARMUP=1" ;;
    # 262k fp8 moves EXACTLY the bytes bf16 moved at 131k — 7.5 GB per warm
    # rep, a ~22.6 GB 3-pair populate footprint the 32 GiB store arena
    # already carried green in the bf16 @131k session — against a baseline
    # ~2.8x the bf16 @131k one (~3.2x the fp8 @131k one; banked anchors
    # 85.7 s and 26.6 s). Top of the real-NIC curve, and a free block-bound
    # probe: same bytes as bf16 @131k through 2x the block count. Fallback
    # on any gate fail: publish nothing. MNBT per the pre-registered PR-10
    # sweep: the 262k pair runs at whichever of {8192, 16384} minimizes the
    # fp8 recompute baseline (PR-2's anti-vendor rule; this cell has no bf16
    # counterpart, so comparability does not constrain it).
    nic-262k)     nic_common 262144; echo "REPS=2"; echo "WARMUP=1" ;;
    nic-262k-m16) nic_common 262144 | sed 's/^MAX_NUM_BATCHED_TOKENS=.*/MAX_NUM_BATCHED_TOKENS=16384/'
                  echo "REPS=2"; echo "WARMUP=1" ;;
    # Its denominators, both sweep pins (baseline-minimizing winner divides).
    nic-base-262k)     nic_base_fp8 262144 ;;
    nic-base-262k-m16) nic_base_fp8 262144 | sed 's/^MAX_NUM_BATCHED_TOKENS=.*/MAX_NUM_BATCHED_TOKENS=16384/' ;;
    # Mechanism arms (CLAIMS.md Amendment 3 follow-up, PR-10). Two store-free
    # probes on the SAME box + checkpoint + engine pin the connector arms use
    # (LENGTHS=131072 pin-matches the 128k anchor arm — max-model-len is a
    # PR-1-hashed knob). EQUIV_N=0 because there is no store to certify
    # against (job.sh guard).
    #   nic-mech-cal: RAW-candidate margins of the 48-pool (CALIBJSONL) —
    #     the on-box go/no-go for corpus-v2 eligibility before any connector
    #     arm spends, and the raw side of the raw-vs-padded margin story.
    #   nic-mech-pre: PADDED margins + self-consistency (PRESCREENJSONL) of
    #     the frozen v2 corpus (must find eligibles: a zero here is the
    #     corpus-transfer failure — halt) and of the 48-pool ('?' = its
    #     zero-eligible outcome is a recorded finding, expected under
    #     padding). This is the store-FREE side of the construction-matched
    #     mechanism pair; the connector arms' in-run prescreen (same builder,
    #     connector attached) is the store-ATTACHED side.
    nic-mech-cal) cat <<MCEOF
MODEL=Qwen/Qwen2.5-7B-Instruct-1M
STRIP_DCA=1
HF_HOME=/nvme/hf
KV_CACHE_DTYPE=fp8_e4m3
KV_BYTES_PER_TOKEN=28672
MARGIN_CALIBRATE=/work/kvstash/bench/e2e/equiv-candidates-pool.txt
EQUIV_MARGIN_FLOOR=2.0
EQUIV_GEN_TOKENS=8
EQUIV_N=0
BASELINE_ONLY=1
LENGTHS=131072
GEN_TOKENS=16
GPU_MEM_UTIL=0.92
MAX_NUM_BATCHED_TOKENS=8192
RIG=ec2-twonode-nic
GPU_ANNOT=ec2 g6e.2xlarge (mechanism rider: store-free calibration)
MCEOF
          ;;
    nic-mech-pre) cat <<MPEOF
MODEL=Qwen/Qwen2.5-7B-Instruct-1M
STRIP_DCA=1
HF_HOME=/nvme/hf
KV_CACHE_DTYPE=fp8_e4m3
KV_BYTES_PER_TOKEN=28672
PRESCREEN_SETS=/work/kvstash/bench/e2e/equiv-prompts-high-margin-v2.txt:?/work/kvstash/bench/e2e/equiv-candidates-pool.txt
PRESCREEN_N=48
EQUIV_MARGIN_FLOOR=2.0
EQUIV_GEN_TOKENS=8
EQUIV_N=0
BASELINE_ONLY=1
LENGTHS=131072
GEN_TOKENS=16
GPU_MEM_UTIL=0.92
MAX_NUM_BATCHED_TOKENS=8192
RIG=ec2-twonode-nic
GPU_ANNOT=ec2 g6e.2xlarge (mechanism rider: store-free padded prescreen)
MPEOF
          ;;
    # ---- PR-11 reload-path knob A/B (docs/CLAIMS.md PR-11; frozen pre-spend)
    # Every knob under test is kv_connector_extra_config — never an engine
    # arg, so all 131k arms below share nic-128k's ENGINE_ARGS_SHA by
    # construction. The -ab- delta arms are DIAGNOSTICS: EQUIV_N=0 (no
    # certification — their warm medians steer the config decision and are
    # NEVER published as cells; the exact-count hit gate still enforces
    # every warm rep). Publishable cells re-run CERTIFIED at the config the
    # pre-registered decision rule selects. Overrides that change an
    # existing nic_common key go through sed (the m16 precedent) — never a
    # duplicate env line whose winner is a docker implementation detail.
    # The control's fan-out is PINNED to 4 (not inherited from the KVB_FANOUT
    # session knob): fan-out IS this A/B's independent variable, and a stray
    # session export must not be able to turn the control into a treatment
    # arm and fire F2 falsely. Same pin on pyspy/bracket/certified-ctl arms.
    nic-ab-ctl)    nic_common 131072 | sed -e 's/^EQUIV_N=.*/EQUIV_N=0/' \
                                           -e 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=4/'
                   printf 'REPS=3\nWARMUP=1\n' ;;
    nic-ab-f8)     nic_common 131072 | sed -e 's/^EQUIV_N=.*/EQUIV_N=0/' \
                                           -e 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=8/'
                   printf 'REPS=3\nWARMUP=1\n' ;;
    nic-ab-f8h512) nic_common 131072 | sed -e 's/^EQUIV_N=.*/EQUIV_N=0/' \
                                           -e 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=8/'
                   printf 'REPS=3\nWARMUP=1\nKVBD_PIPELINE_HALF_BYTES=536870912\n' ;;
    # verify-off isolation probe at the full-knob config: xxh3 cost on the
    # drain threads, the loopback memo's prime per-byte suspect. job.sh
    # stamps it 'verify=off (DIAGNOSTIC arm)' — never a headline.
    nic-ab-voff)   nic_common 131072 | sed -e 's/^EQUIV_N=.*/EQUIV_N=0/' \
                                           -e 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=8/'
                   printf 'REPS=3\nWARMUP=1\nKVBD_PIPELINE_HALF_BYTES=536870912\nKVBD_VERIFY=0\n' ;;
    # Attribution probe at the CONTROL config (the shape whose ~30ms/pass
    # boundary needs explaining): py-spy over the measure phase, stamped
    # DIAGNOSTIC (sampling overhead), speedscope JSON banked from results/.
    nic-ab-pyspy)  nic_common 131072 | sed -e 's/^EQUIV_N=.*/EQUIV_N=0/' \
                                           -e 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=4/'
                   printf 'REPS=2\nWARMUP=1\nPYSPY=1\nPYSPY_DURATION_S=120\n' ;;
    # End-of-session drift bracket: the control config re-measured after
    # every other arm (PR-10's bracket convention, ±10%% window).
    nic-ab-ctl-bracket) nic_common 131072 | sed -e 's/^EQUIV_N=.*/EQUIV_N=0/' \
                                                -e 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=4/'
                   printf 'REPS=2\nWARMUP=1\n' ;;
    # Publishable cells at the knob config the frozen decision rule selects
    # (pre-registered default: fanout8 + 512MiB halves; fallbacks in
    # CLAIMS.md PR-11). Full corpus-v2 certification, EQUIV_N=24 per the
    # dated amendment. 262k runs the PR-10 MNBT sweep winner (16384).
    nic-ab-131k)   nic_common 131072 | sed 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=8/'
                   printf 'REPS=2\nWARMUP=1\nKVBD_PIPELINE_HALF_BYTES=536870912\n' ;;
    nic-ab-131k-f8) nic_common 131072 | sed 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=8/'
                   printf 'REPS=2\nWARMUP=1\n' ;;
    nic-ab-131k-ctl) nic_common 131072 | sed 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=4/'
                   printf 'REPS=2\nWARMUP=1\n' ;;
    nic-ab-262k)   nic_common 262144 | sed -e 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=8/' \
                                           -e 's/^MAX_NUM_BATCHED_TOKENS=.*/MAX_NUM_BATCHED_TOKENS=16384/'
                   printf 'REPS=2\nWARMUP=1\nKVBD_PIPELINE_HALF_BYTES=536870912\n' ;;
    nic-ab-262k-f8) nic_common 262144 | sed -e 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=8/' \
                                            -e 's/^MAX_NUM_BATCHED_TOKENS=.*/MAX_NUM_BATCHED_TOKENS=16384/'
                   printf 'REPS=2\nWARMUP=1\n' ;;
    nic-ab-262k-ctl) nic_common 262144 | sed -e 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=4/' \
                                             -e 's/^MAX_NUM_BATCHED_TOKENS=.*/MAX_NUM_BATCHED_TOKENS=16384/'
                   printf 'REPS=2\nWARMUP=1\n' ;;
    *) echo "unknown arm: $1" >&2; return 1 ;;
  esac
}

# Called directly: print the env for one arm.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then arm_env "${1:?usage: arms.sh <arm>}"; fi
