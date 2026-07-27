#!/usr/bin/env bash
# Red-proof harness for submit.sh's per-spec env handling. Zero spend: HF_BIN
# points at a stub that only records its argv (one %q-quoted line per call).
#
# What it proves (each was a live bug or its regression guard):
#   1. FP8_CAMPAIGN=1: the TWO-PHASE submission argv carries NO BASELINE_ONLY
#      even after the full preview loop ran (the preview loop's baseline spec
#      used to leak BASELINE_ONLY=1 into the shell, turning BOTH billed jobs
#      into pure-recompute baselines — invisible in the echoed preview and
#      under DRY_RUN=1), while the baseline argv carries BASELINE_ONLY=1.
#   2. FP8_CAMPAIGN=1 with caller EQUIV_N>0: forwarded to the two-phase job
#      only — the baseline job must NOT inherit it (job.sh dies in the billed
#      container on EQUIV_N>0 + BASELINE_ONLY=1).
#   3. FP8_CAMPAIGN=1 with explicit EQUIV_N=0: refused BEFORE billing (it
#      would silently strip the token-identity certification from the
#      campaign's fp8 two-phase job).
#   4/5. Single-job path: caller's BASELINE_ONLY=1 / EQUIV_N still forward
#      byte-for-byte (the snapshot fix must not eat the old behavior).
#
#   bash bench/rigs/hf-gpu/test_submit.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$HERE/submit.sh"
TMP="$(mktemp -d /tmp/kvb-submit-test.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

CALLS="$TMP/hf-calls.log"
cat > "$TMP/hf" <<'STUB'
#!/usr/bin/env bash
printf '%q ' "$@" >> "${KVB_TEST_CALLS:?}"
printf '\n' >> "$KVB_TEST_CALLS"
echo "Job started: stub-job-$(wc -l < "$KVB_TEST_CALLS" | tr -d ' ')"
STUB
chmod +x "$TMP/hf"

FAIL=0
say()  { printf '%s\n' "$*"; }
pass() { say "PASS: $*"; }
fail() { say "FAIL: $*"; FAIL=1; }

run_submit() {  # $@ = extra env VAR=VAL pairs; returns submit.sh's exit code
  : > "$CALLS"
  env -u BASELINE_ONLY -u EQUIV_N -u FP8_CAMPAIGN -u KV_CACHE_DTYPE \
      -u DRY_RUN -u JOB_NAME -u FP8_PREFLIGHT \
      HF_BIN="$TMP/hf" KVB_TEST_CALLS="$CALLS" KVB_SUBMIT_N_CONFIRMED=1 \
      "$@" bash "$SUBMIT" > "$TMP/out.log" 2>&1
}

# argv assertions read the recorded calls, never the echoed preview — the
# original bug was exactly that the preview looked right while the
# submission argv did not.
call() { sed -n "${1}p" "$CALLS"; }

# ---- 1+2: campaign submission argv, caller EQUIV_N set ----------------------
if run_submit FP8_CAMPAIGN=1 EQUIV_N=6; then
  [[ "$(wc -l < "$CALLS" | tr -d ' ')" == "2" ]] || fail "campaign: expected 2 hf calls, got $(wc -l < "$CALLS")"
  TWOPHASE="$(call 1)"; BASELINE="$(call 2)"
  case "$TWOPHASE" in *chart2-fp8-twophase*) ;; *) fail "campaign: call 1 is not the twophase job: $TWOPHASE";; esac
  case "$BASELINE" in *chart2-fp8-baseline*) ;; *) fail "campaign: call 2 is not the baseline job: $BASELINE";; esac
  case "$TWOPHASE" in
    *BASELINE_ONLY*) fail "campaign: twophase submission argv carries BASELINE_ONLY (preview-loop leak): $TWOPHASE";;
    *) pass "campaign: twophase argv has no BASELINE_ONLY after a full preview pass";;
  esac
  case "$BASELINE" in
    *"BASELINE_ONLY=1"*) pass "campaign: baseline argv carries BASELINE_ONLY=1";;
    *) fail "campaign: baseline submission argv lost BASELINE_ONLY=1: $BASELINE";;
  esac
  case "$TWOPHASE" in
    *"EQUIV_N=6"*) pass "campaign: caller EQUIV_N=6 reaches the twophase job";;
    *) fail "campaign: twophase argv lost the caller's EQUIV_N=6: $TWOPHASE";;
  esac
  case "$BASELINE" in
    *EQUIV_N*) fail "campaign: baseline argv inherited EQUIV_N (job.sh dies in the billed container): $BASELINE";;
    *) pass "campaign: baseline argv carries no EQUIV_N";;
  esac
else
  fail "campaign submit exited nonzero: $(cat "$TMP/out.log")"
fi

# ---- 3: explicit EQUIV_N=0 refused before billing ---------------------------
if run_submit FP8_CAMPAIGN=1 EQUIV_N=0; then
  fail "campaign: EQUIV_N=0 was accepted (strips token-identity certification from the fp8 job)"
else
  if grep -q "EQUIV_N=0" "$TMP/out.log" && [[ ! -s "$CALLS" ]]; then
    pass "campaign: explicit EQUIV_N=0 refused at submit time, zero hf calls"
  else
    fail "campaign: EQUIV_N=0 exited nonzero but not via the refusal (out: $(cat "$TMP/out.log"); calls: $(cat "$CALLS"))"
  fi
fi

# ---- 4+5: single-job path regression guards (the snapshot fix must not eat
# the caller's explicit knobs; tested separately — job.sh refuses the combo) --
if run_submit BASELINE_ONLY=1; then
  [[ "$(wc -l < "$CALLS" | tr -d ' ')" == "1" ]] || fail "single: expected 1 hf call, got $(wc -l < "$CALLS")"
  ONLY="$(call 1)"
  case "$ONLY" in
    *"BASELINE_ONLY=1"*) pass "single: caller BASELINE_ONLY=1 still forwards";;
    *) fail "single: caller BASELINE_ONLY=1 was dropped: $ONLY";;
  esac
else
  fail "single-job (BASELINE_ONLY=1) submit exited nonzero: $(cat "$TMP/out.log")"
fi

if run_submit EQUIV_N=4; then
  ONLY="$(call 1)"
  case "$ONLY" in
    *"EQUIV_N=4"*) pass "single: caller EQUIV_N=4 still forwards";;
    *) fail "single: caller EQUIV_N=4 was dropped: $ONLY";;
  esac
else
  fail "single-job (EQUIV_N=4) submit exited nonzero: $(cat "$TMP/out.log")"
fi

if (( FAIL )); then say "test_submit: FAIL"; exit 1; fi
say "test_submit: OK"
