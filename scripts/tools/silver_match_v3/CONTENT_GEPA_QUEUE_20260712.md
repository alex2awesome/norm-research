# Content-task GEPA queue after the sk3 SSH interruption

This queue covers `press-releases`, `creative-writing`, `peer-review`, and
`legal-outcome-prediction`.  Its thresholds and request ceilings are frozen in
`policies/content_gepa_alltask_policy_v1.json`.  The data packs are not frozen
locally because the canonical faithful manifest and task artifacts exist only
on sk3, which is currently unreachable.  Do not substitute local/stale data.

## Fail-closed resume order

1. Restore read-only SSH and re-run `sha256sum` on the canonical manifest and
   every task artifact.  Require manifest SHA
   `b614e345a07123f9fe79d9521351886107476d34cf2b09daa50efce71dc1356f`.
2. For each task, enumerate every prior manual, teacher, adjudicator GEPA,
   verifier GEPA, resolver, tie-break, blind, frozen-test, and quarantine
   identity file.  Hash them; do not open test/outcome content.  An incomplete
   exclusion inventory is a hard stop.
3. Freeze the identity-only optimize panel from canonical `train` groups with
   `freeze_clean_gepa_panel`.  Freeze the select panel separately, adding the
   optimize identities to `--exclude-panel`.  Each selected source group may
   appear only once and is permanently excluded from gradients and MI.
4. Render independently permuted full-bank hidden-label packs with the now
   all-task `prepare_verifier_semantic_audit_pack --task TASK`.  Resolve two
   independent exact labels before revealing them to GEPA.
5. Run Gemma-4 through OpenRouter only after every input freeze/hash validates.
   Use `adjudicate_gemma_api` and `verify_gemma_api` with explicit `--split-role
   train` or `dev`, `--api-key-file ~/.openrouter-api-key.txt`, and a per-call
   hard cap equal to `ceil(1.2 * eligible_rows)`.  Never pass `--keep-raw`
   except for explicit parser debugging on optimize rows.
6. Shepherd prompts on optimize only.  Freeze the resulting prompt before any
   select prediction.  Select can choose between the predeclared strict
   two-order and all-three-order policies; it cannot cause another prompt edit.
7. A task proceeds to production only if the verifier retains at least 20
   exact matches at point precision >=.90 and Wilson lower >=.80.  Then run a
   separate uniform blind final-MATCH audit.  No threshold may be lowered.

## Press-releases first

Authoritative evidence to re-check on sk3:

```bash
MODEL=/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context
FAITH=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful
sha256sum "$FAITH/manifest.json" \
  "$MODEL/adjudicator_k50/press-releases/candidates/dev.frozen-retriever.top50.jsonl" \
  "$MODEL/adjudicator_k50/press-releases/panels/dev.manual.jsonl" \
  "$MODEL/adjudicator_k50/verifier_calibration_pr_dev_v1/press-releases/dev.candidates.jsonl" \
  "$MODEL/adjudicator_k50/verifier_calibration_pr_dev_v1/press-releases/dev.primary.jsonl" \
  "$MODEL/adjudicator_k50/verifier_calibration_pr_dev_v1/press-releases/dev.truth.jsonl" \
  "$MODEL/adjudicator_k50/press-releases/PR_ADJUDICATOR_TEST.UNAVAILABLE_PRESELECTION_MATERIALIZATION.json"
```

Expected evidence: the candidate and manual-panel hashes begin `c5e0e7e4` and
`840fd87f`; failed-verifier truth begins `a7ae30e1`; the quarantine-record hash
is exactly `595cd0d8925bd201433bfd5e6c386d7c034f1c2db998f1758beafe2c20e2e7c2`.
Any mismatch stops the queue.  PR verifier R3 is consumed and rejected
(`18/29` exact, precision `.6207`, Wilson lower `.4400`); do not tune on or
recycle that dev pack.  The quarantined test must never be opened or scored.

After the complete PR exclusion list is assembled in a newline-safe shell
array named `EXCLUDES`, the exact materialization template is:

```bash
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
OUT="$MODEL/adjudicator_k50/press-releases/gepa_clean_v2"
"$PY" -m scripts.tools.silver_match_v3.freeze_clean_gepa_panel \
  --manifest "$FAITH/manifest.json" --task press-releases \
  --role optimize --count 120 --seed 2026071201 \
  "${EXCLUDES[@]/#/--exclude-panel=}" \
  --output-root "$OUT/optimize_identity"
EXCLUDES+=("$OUT/optimize_identity/identities.jsonl")
"$PY" -m scripts.tools.silver_match_v3.freeze_clean_gepa_panel \
  --manifest "$FAITH/manifest.json" --task press-releases \
  --role select --count 240 --seed 2026071202 \
  "${EXCLUDES[@]/#/--exclude-panel=}" \
  --output-root "$OUT/select_identity"
```

The `EXCLUDES` array is intentionally not guessed here.  Populate it only
from the fresh sk3 inventory and preserve the exact command plus resulting
hashes in `$OUT/EXCLUSION_INVENTORY.json` before either invocation.

## Remaining task sizes

Use the same two-step freezer with task-specific seeds and policy-frozen
counts: CW `120/240` with at least 12 groups per corpus; Peer `160/300`; Legal
`200/360` with at least 8 groups per corpus.  Discover and hash their remote
paths first.  The local ledger does not contain authoritative filenames, so
inventing them would defeat the leakage audit.

## OpenRouter command template (only after a pack is frozen)

For a frozen candidate JSONL with exactly `N` rows, compute `CAP=ceil(1.2*N)`
outside the runner and record it.  One order is:

```bash
"$PY" -m scripts.tools.silver_match_v3.adjudicate_gemma_api \
  --manifest "$FAITH/manifest.json" --candidates "$CANDIDATES" \
  --output "$OUTPUT" --split-role "$ROLE" \
  --prompt scripts/tools/silver_match_v3/prompts/gepa_round2_candidate.txt \
  --prompt-addon "$FROZEN_TASK_ADDON" --max-candidates 50 \
  --order-mode "$ORDER" --api-base-url https://openrouter.ai/api/v1 \
  --api-key-file ~/.openrouter-api-key.txt --model google/gemma-4-31b-it \
  --concurrency 8 --batch-size 32 --max-api-requests "$CAP"
```

Run original/hashed on optimize and original/hashed/reverse on select.  The
verifier uses the analogous `verify_gemma_api` command with `--primary`,
`--max-alternatives 49`, and the same cap.  The global ceilings are 3,000
requests per task and 9,000 across this four-task queue.  Full-corpus matching
must use local batch vLLM/Gemma when GPU quota permits; it is never an
OpenRouter job.
