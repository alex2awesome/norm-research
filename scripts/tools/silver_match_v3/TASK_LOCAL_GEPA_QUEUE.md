# Task-local GEPA queue: code review and mathematics

Canonical provenance, K50 coverage, and complete exclusions are now frozen on
sk3 under
`$R/task_local_gepa_clean_v1`, where
`$R=/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora_20260712_r3_context`.
The earlier r5 GEPA panels are audit evidence only: they relabeled 39
authoritative non-train code rows (23 dev, 16 test) and 38 math rows (20 dev,
18 test) as GEPA train/dev. The identity-only v2 audit hashes are
`a9b6568243c2...` (code) and `6e70d6c5d872...` (math).

The replacement panels use the retriever/LoRA source-group policy from the
frozen run configs (`split_seed=73129`, 80/10/10), not
`make_calibration.split_for`. `freeze_upstream_role_reference` verified the
derived role exactly against 19,841 audited code rows and 3,281 audited math
rows with zero mismatches. Its K50 role-reference hashes are `3b717eb89b7f...`
(21,841 code rows) and `0b198d13d907...` (56,925 math rows).

Fresh source-disjoint, upstream-train panels are:

- code optimize100 `c33a488bc3cd...`, select60 `b46b1e132282...`;
- math optimize100 `3dfeda0b545b...`, select60 `8cc4cabe5cbc...`.

Each has two independently permuted full-bank label views (A seed 172901, B
seed 846731) with candidate proposals absent. Independence-audit hashes are
`1d64e2234ed9...`, `0ccfb75ef9f5...`, `cefa2a6b10fb...`, and
`b7b7731ee63a...` for code optimize/select and math optimize/select. Only these
fresh views may supply the next task-local prompt truth.

## Non-negotiable roles

- Only canonical upstream `train` source groups may enter task-local GEPA.
- `prompt_train` may produce prompt mutations; `prompt_dev` may select only
  among variants frozen before any dev outputs are read.
- Exact current-bank metric IDs are primary. Family credit is sensitivity only.
- Test, permanent blind audit, production, MI, and outcome rows are unavailable
  to both prompt mutation and prompt selection.
- Every prior labeled or inspected UID excludes its entire canonical source
  group, not only that row.
- A policy is eligible only with retained exact precision at least `.90`, a
  Wilson 95% lower bound at least `.80`, and support at least `30` on frozen
  prompt-dev. Prefer Wilson lower bound, then support, then point precision.
- OpenRouter plans use exact `google/gemma-4-31b-it`, zero implicit HTTP retries,
  and a hard predeclared logical-request ceiling. Resume explicitly after a
  transport failure.

## Required inputs after sk3 is reachable

For each task, copy or address all of the following from one immutable run:

1. canonical `manifest.json` and its task bank;
2. final retriever K=50 candidate rows for every candidate calibration UID;
3. audited exact/typed labels with canonical `task`, `norm_uid`, and provenance;
4. a task-specific JSONL union of every previously labeled, inspected,
   optimized, selected, tested, or blind-audited UID.

The files `labels/r4_code_manual_adjudication_overrides.jsonl` and
`labels/r4_math_manual_adjudication_overrides.jsonl` are not standalone GEPA
panels. They contain decisions but no canonical task/source/split provenance.
They may be joined to their original immutable panel, but must not be assigned
new roles from their content alone.

For Math, the failed fresh-boundary experiment is sealed. At minimum, its
source-group exclusion union must include:

- `outputs/silver_match_v3/r5_math/fresh_boundary_v1/pass1.validated.jsonl`
  (600 rows);
- `.../blind_audit_addon_200.pass1.validated.jsonl` (200 rows);
- `.../blind_audit_addon2_200.pass1.validated.jsonl` (200 rows).

Those three inputs cover the original and two add-on label universes; all later
two-pass and third-pass artifacts are subsets. The immutable failure remains
70/80 exact, point precision `.875`, Wilson lower `.7849700282`; none of these
rows may supply a prompt or model gradient.

## Freeze the clean train-only panel

Do not infer the upstream role with the older calibration split. First project
the immutable retriever split policy onto the exact K50 universe and verify it
against every overlapping audited teacher:

```bash
python -m scripts.tools.silver_match_v3.freeze_upstream_role_reference \
  --manifest /path/to/manifest.json \
  --task code-review \
  --candidates /path/to/code-review.frozen.top50.jsonl \
  --run-config /path/to/code-review/run_config.json \
  --audit-reference /path/to/code-review.audited_labels.jsonl \
  --minimum-k 50 \
  --output-root /path/to/code-review/upstream_roles_k50
```

Then use `freeze_clean_gepa_panel` twice, excluding optimize identities when
freezing select:

```bash
python -m scripts.tools.silver_match_v3.freeze_clean_gepa_panel \
  --manifest /path/to/manifest.json \
  --task code-review --role optimize --count 100 \
  --required-upstream-split train \
  --eligible-reference /path/to/upstream_roles_k50/roles.jsonl \
  --upstream-role-reference /path/to/upstream_roles_k50/roles.jsonl \
  --exclude-panel /path/to/all_prior_exclusions/identities.jsonl \
  --output-root /path/to/optimize100
```

The historical command below is retained only for compatible, already
authoritative label panels. It must not be used to reconstruct retriever roles
from `make_calibration.split_for`:

Run once per task, supplying every exclusion union available for that task:

```bash
python -m scripts.tools.silver_match_v3.split_train_only_gepa_panel \
  --manifest /path/to/manifest.json \
  --labels /path/to/audited_labels_with_provenance.jsonl \
  --task code-review \
  --exclude-reference /path/to/code-review.all_prior_exclusions.jsonl \
  --require-exclusions \
  --minimum-train 30 --minimum-dev 30 \
  --output /path/to/code-review.task_local_gepa.panel.jsonl
```

Use a distinct output path and task/exclusion union for
`math-stackexchange`. The splitter recomputes the canonical upstream split,
excludes whole source groups, verifies train/dev group disjointness, and writes
input/output hashes.

## Freeze adjudicator and verifier cells before inference

Code-review prompt variants are cumulative and must all be named before any
new task-local output is read:

```bash
python -m scripts.tools.silver_match_v3.freeze_task_gepa_api_plan \
  --task code-review \
  --predeclaration scripts/tools/silver_match_v3/locks/task_local_gepa_code_math_predeclaration_v1.json \
  --manifest /path/to/manifest.json \
  --panel /path/to/code-review.task_local_gepa.panel.jsonl \
  --candidates /path/to/code-review.final_k50.jsonl \
  --exclude-reference /path/to/code-review.all_prior_exclusions.jsonl \
  --adjudicator-variant r0=scripts/tools/silver_match_v3/prompts/gepa_round0.txt \
  --adjudicator-variant r1=scripts/tools/silver_match_v3/prompts/gepa_round0.txt,scripts/tools/silver_match_v3/prompts/gepa_code_r1_addon.txt \
  --adjudicator-variant r2=scripts/tools/silver_match_v3/prompts/gepa_round0.txt,scripts/tools/silver_match_v3/prompts/gepa_code_r1_addon.txt,scripts/tools/silver_match_v3/prompts/gepa_code_r2_addon.txt \
  --adjudicator-variant r3=scripts/tools/silver_match_v3/prompts/gepa_round0.txt,scripts/tools/silver_match_v3/prompts/gepa_code_r1_addon.txt,scripts/tools/silver_match_v3/prompts/gepa_code_r2_addon.txt,scripts/tools/silver_match_v3/prompts/gepa_code_r3_addon.txt \
  --verifier-variant v0=scripts/tools/silver_match_v3/prompts/verify_match_v1.txt \
  --verifier-variant v1=scripts/tools/silver_match_v3/prompts/verify_match_v1.txt,scripts/tools/silver_match_v3/prompts/verify_code_gepa_r1_exact_leaf.txt \
  --max-total-api-requests 5000 \
  --output-root /path/to/code-review.task_local_gepa.freeze_v1
```

For Math, freeze `r0`, cumulative `r1` and `r2` using
`gepa_math_r1_addon.txt` and `gepa_math_r2_addon.txt`, plus verifier `v0` and
`v1` using `verify_math_gepa_r1_exact_leaf.txt`. The freezer fails closed when
the requested variant cross-product could exceed the API cap.

`COMMAND_PLAN.json` is the executable DAG. Every inference cell contains an
OpenRouter `command` and a `direct_batch_command` targeting the same frozen
output. Choose exactly one backend for an entire paired cell before inference;
prefer direct batch when the server-wide quota admits a GPU, and never run an
OpenAI-compatible vLLM server. Run the selected module/argv records in order;
do not hand-edit commands after outputs arrive. The DAG includes two-order
adjudication, exact consensus proposals, proposal-only verifier projection,
three verifier orders, and both two- and three-order scoring. Dev results choose
only among the predeclared cells. A failing dev gate consumes that selection
panel; it does not authorize inspecting a blind set or relaxing thresholds.
