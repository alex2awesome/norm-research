#!/usr/bin/env bash
set -euo pipefail

# Completion-aware successor to the original PID-bound handoff queue.  Wait for
# every corrected PR/CW terminal chunk to seal, then run all four Legal lanes.
# Each completed Legal lane releases exactly one Peer lane.  The queue inspects
# filenames/counts and immutable pack hashes only; it never reads label content.

ROOT="outputs/silver_match_v3"
LOG_ROOT="$ROOT/content_gepa_handoff_runtime_v2"
RUNNER="scripts/tools/silver_match_v3/run_codex_pack_labels.py"
SCHEMA="scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json"
RUNNER_SHA="cb49d940baf498a813d2631ccd3f6099c1aba41bc3df1140c1f40e71f47639ab"
SCHEMA_SHA="9a67fd26e6a2c498bb591d76049a0eea02ea5bf96d41a5b37ae30b92e7e5c496"

mkdir -p "$LOG_ROOT"
LOCK_DIR="$LOG_ROOT/ACTIVE.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "handoff queue already active: $LOCK_DIR" >&2
  exit 2
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

verify_file() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "hash mismatch: $path expected=$expected actual=$actual" >&2
    return 1
  fi
}

raw_chunk_count() {
  local pack="$1"
  find "$pack/raw_labels" -maxdepth 1 -name 'part-*.json' -type f 2>/dev/null | wc -l | tr -d ' '
}

wait_for_complete_pack() {
  local pack="$1"
  local expected="$2"
  local observed
  while true; do
    observed="$(raw_chunk_count "$pack")"
    if [[ "$observed" == "$expected" ]]; then
      return 0
    fi
    if (( observed > expected )); then
      echo "too many raw chunks: $pack expected=$expected observed=$observed" >&2
      return 1
    fi
    sleep 15
  done
}

run_pack() {
  local pack="$1"
  local task="$2"
  local pass_name="$3"
  local concurrency="$4"
  local validation_sha="$5"
  local items_sha="$6"
  local bank_sha="$7"

  verify_file "$RUNNER" "$RUNNER_SHA"
  verify_file "$SCHEMA" "$SCHEMA_SHA"
  verify_file "$pack/validation.json" "$validation_sha"
  verify_file "$pack/items.jsonl" "$items_sha"
  verify_file "$pack/bank.json" "$bank_sha"

  PYTHONPATH=. python -m scripts.tools.silver_match_v3.run_codex_pack_labels \
    --pack-root "$pack" \
    --task "$task" \
    --pass-name "$pass_name" \
    --concurrency "$concurrency" \
    --model gpt-5.6-sol \
    --reasoning-effort high \
    --timeout-seconds 1800 \
    --chunk-attempts 3 \
    --output-schema "$SCHEMA"
}

PR="$ROOT/press-releases/gepa_clean_v2/label_workspaces"
CW="$ROOT/creative-writing/gepa_clean_v1/label_workspaces"
LEGAL="$ROOT/legal-outcome-prediction/gepa_clean_v1/label_workspaces"
PEER="$ROOT/peer-review/gepa_clean_v1/label_workspaces"

# Do not infer completion from process IDs: corrected terminal retries use new
# PIDs.  A raw chunk only appears after the runner has parsed and schema-checked
# the response, so exact expected counts are the completion boundary.
wait_for_complete_pack "$PR/optimize_pass_a" 5
wait_for_complete_pack "$PR/optimize_pass_b" 5
wait_for_complete_pack "$PR/select_pass_a" 10
wait_for_complete_pack "$PR/select_pass_b" 10
wait_for_complete_pack "$CW/optimize_pass_a" 5
wait_for_complete_pack "$CW/optimize_pass_b" 5
wait_for_complete_pack "$CW/select_pass_a" 10
wait_for_complete_pack "$CW/select_pass_b" 10

chain_optimize_a() {
  run_pack "$LEGAL/optimize_pass_a" legal-outcome-prediction \
    gpt-5.6-sol-high-legal-clean-optimize-pass-a-v1 2 \
    509fccc497f30b51a6400d78942056df30f2fd44493743b5a17424555070e700 \
    8e98d63daf829451095b7b63d3a7476a5fe31c0b9d61d1846d019e246c85722c \
    162409991e1adcbbbe134ccde3e07142c341c4e312acb349ef25ff4d215a63cb
  run_pack "$PEER/optimize_pass_a" peer-review \
    gpt-5.6-sol-high-peer-clean-optimize-pass-a-v1 1 \
    36877e8ac101d0e3dc1a574a1e4246235ecbd4873178b71d2d9f275897a2666d \
    164f1e2c8ca804be1b2be9ee02808c9a6165166132c6e275fc7d5f11e1cc90b7 \
    a7868fd4a7311a1eea92027c8cfd0de681af5400f596e9a3938cfdcb859083c5
}

chain_optimize_b() {
  run_pack "$LEGAL/optimize_pass_b" legal-outcome-prediction \
    gpt-5.6-sol-high-legal-clean-optimize-pass-b-v1 2 \
    ed9725d157829adc4540dd05df90f30a4cf538a03e59026a6d13861f6e5a17c6 \
    48de3176d0024145b33d0f05d670bc2954929fdde2103f9dad382c82248a331a \
    e5d2c4502d84178a17cc284bce7cd304d06e9d1e81babdabcc49d8b4c66c61b1
  run_pack "$PEER/optimize_pass_b" peer-review \
    gpt-5.6-sol-high-peer-clean-optimize-pass-b-v1 1 \
    dc203a817c20d96fd65f4d3fa7bf68d9ea54f40b9764042ee855ccae070c5e33 \
    378e8fc3da4c06aa087437af62251ef9dd675734009efec42a377fea160fded1 \
    8fa80d827527f7d5933f27aea1cf00ba874000a96c0101b10bb1485a26b746b5
}

chain_select_a() {
  run_pack "$LEGAL/select_pass_a" legal-outcome-prediction \
    gpt-5.6-sol-high-legal-clean-select-pass-a-v1 2 \
    cb41107081df9b9918a60962b2fcf81a00f4c67804f704d4ae850977251e3d7a \
    379d6c6f5fdb470ff7aff861ab1d71f345e7064df600f240024a499dbed4b3b3 \
    239006e3dfaa9b9c31048762aa72d7ce93d835f74f832182c4a64b44b6d2a7ba
  run_pack "$PEER/select_pass_a" peer-review \
    gpt-5.6-sol-high-peer-clean-select-pass-a-v1 1 \
    2448599e62b8ea2283d4143cc8b0b03d48e1a2bc7340adbe60fb6f0900d91ec8 \
    6f2dacd46d7a4d3206555837cdfe92e324538cd9db8064fd573ede9770d47251 \
    557e91a89025d59b11d2c46ee1627380593e5b1c814c7d491986115f0047762f
}

chain_select_b() {
  run_pack "$LEGAL/select_pass_b" legal-outcome-prediction \
    gpt-5.6-sol-high-legal-clean-select-pass-b-v1 2 \
    5349c8bdfc00418b078029f0c1d2f9faebfe0e8ab93ed77bc9d5135f573e6be4 \
    09993b8d03a04c73ad6e4b152fee3b129b4fece0a248b4c6ff831416f2928721 \
    e2a1873a87d2b8c2cdf5e75a325e92b21529fb83d123a333a5e605133ab92a74
  run_pack "$PEER/select_pass_b" peer-review \
    gpt-5.6-sol-high-peer-clean-select-pass-b-v1 1 \
    c2ef88fadd0f75474ea82357d2fdbc4d156e8d3755e4bc688319117bea7b7f4c \
    969284e1806bb4d57c77ce9b7421ba3704618c9b1c0afd72269f93bfe6b96d31 \
    9cd256d396006f258c712d02c90b756101dc5dd7ec20e33195b856adc0fb61f7
}

chain_optimize_a >"$LOG_ROOT/optimize_a.chain.log" 2>&1 &
chain_optimize_b >"$LOG_ROOT/optimize_b.chain.log" 2>&1 &
chain_select_a >"$LOG_ROOT/select_a.chain.log" 2>&1 &
chain_select_b >"$LOG_ROOT/select_b.chain.log" 2>&1 &
wait
