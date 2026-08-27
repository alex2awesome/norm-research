# Frozen hierarchy CPU batch audit (2026-07-13)

## Verdict

The repository has a sound CPU control plane for validating the 990 frozen source
cells, compiler briefs, and shared label-free item panels. It does **not** yet have one
generic executor that can honestly turn all 990 briefs into candidate programs. The
remaining gap is candidate authoring plus task-specific execution adapters, not CPU
capacity.

`hierarchy_cpu_work_ledger.py` now records that distinction and resumes from registry
changes without counting briefs, path strings, or historical artifact presence as a
scientific run. It is an operational snapshot, not a cross-task codability or stage
estimate.

## What can run honestly on CPU now

- Validate the frozen v1 panel with its original 11-task × 3-level × 30-cell contract.
- Rebuild and validate the 990 source-only compiler briefs. A brief remains an authoring
  input, never an executed program.
- Validate all 11 shared item panels as text-only, outcome-free, disjoint
  compiler-train/sealed-heldout splits.
- Replay already-authored task-specific code through its existing runner:
  code-review modules, the Math constant-L slice, and the additive Science full-article
  claim runner. These remain distinct instruments and are not a generic 990-cell fleet.
- Replay CODE probes with `contract_check_isomorphic.py`. HYBRID probes can be replayed
  only when their already-frozen prompt-field extraction artifact exists; generating that
  artifact is not CPU-only.
- Validate and execute new typed programs with `dag_schema_hardened.py`, provided the
  candidate is trusted Python running in process. This is a provenance boundary, not an
  operating-system sandbox.
- Use `ops_capability_v2.py` with explicit dependency/abstention reporting. The corrected
  counterexample replay is 17/17; frozen v1 is retained for audit and is 0/17 on that
  corrected set.

## What must not be treated as CPU-only verifiability

- `battery/agentic_run.py` reads the TRAIN LLM judge vector and historical extracted LLM
  fields. It is a reference-guided reconstruction compiler aid, not a code-only verifier
  and not a generic hierarchy runner. It also executes candidate Python in process.
- `battery/contract_check.py` is the frozen historical checker. It mixes code-only
  synthetic probes with a train-discrimination pass that may use historical extracted
  fields. New work should use the channel-faithful successor and report CODE and HYBRID
  gates separately.
- `battery/dag_schema.py` trusts author-declared provenance and gives nodes the full
  context. It is a historical WS4 artifact. New programs should use the hardened DAG;
  neither implementation is an OS sandbox against hostile Python.
- A locally present execution JSON does not establish construct fidelity, heldout
  readiness, prompt reconstruction, or isomorphism.

The channel-faithful L path is no longer wholly untested: three real frozen extraction
artifacts exist (`math a66`, `math a78`, `peer-review a25`). Their HYBRID gates are FAIL,
FAIL, and PASS respectively, but their unlabeled discrimination gates were not run. They
remain three criterion-specific instrument checks, not a prevalence estimate.

## Current resumable ledger snapshot

The additive snapshot is
`outputs/metric_seam_pilot/hierarchy_r123/hierarchy_cpu_work_ledger_v1.json`.

- 990/990 compiler briefs validate as input-only.
- 11/11 label-free item panels validate.
- 56 historical code-review candidate and generic execution declarations are locally
  present.
- The corrected code-review scientific overlay is 50 static-fidelity mappings, 27
  train-operational mappings, and 18 heldout/prompt-ready mappings. Only those 18 are
  queued for prompt-reference scoring; the generic execution artifact does not promote
  all 56.
- The other ten tasks are `not_integrated_na` in this control-plane ledger. This avoids
  turning stale global readiness into false zeros for Math, Patents, Science, or the seven
  untouched tasks.
- Validated completed deep runs: 0. A completion now needs an explicit cell-bound receipt
  reporting construct, same-byte input, executed-program, reference-instrument, and
  reference-reconstruction fidelity, plus certificate/abstention validity and sealed
  evaluation.
- Bounded non-discovery has a separate terminal-attempt receipt. It can be recorded with
  no candidate path and never licenses a tacitness claim or counts as a completed deep
  run.

## Remaining implementation gaps

1. Add task-specific registries/adapters for Math, Patents, full-article Science, and then
   the remaining domains. Until joined, their task-specific results remain NA in this
   ledger rather than zero.
2. Give each adapter a validator that can issue the completion receipt. The generic
   readiness layer checks identity and local content bindings but cannot infer a domain
   artifact's semantics.
3. Run candidate Python in a process sandbox before treating blind authoring as
   adversarially sealed. Hardened DAG provenance does not prevent filesystem or process
   access by a hostile module.
4. Preserve capability version and dependency availability per run; an abstention caused
   by a missing parser/library is an instrument result, not evidence about the construct.
5. Keep prompt/reference generation as a separate authorized stage. CPU replay can consume
   a frozen prompt artifact but may not fabricate it.

## Commands

Build or resume the control-plane snapshot (no candidate/model execution):

```bash
python -m methods.metric_seam.hierarchy_cpu_work_ledger \
  --panel outputs/metric_seam_pilot/hierarchy_r123/panel_v3.json \
  --briefs outputs/metric_seam_pilot/hierarchy_r123/compiler_briefs_v3.jsonl \
  --items-root outputs/metric_seam_pilot/hierarchy_r123/items_v2 \
  --program-registry outputs/metric_seam_pilot/hierarchy_r123/code_review_registry_v2.json \
  --code-review-corrected-funnel outputs/metric_seam_pilot/hierarchy_r123/code_review_corrected_funnel_v1.json \
  --out outputs/metric_seam_pilot/hierarchy_r123/hierarchy_cpu_work_ledger_v1.json \
  --resume
```

Focused validation:

```bash
python -m pytest \
  methods/metric_seam/test_hierarchy_panel_compat.py \
  methods/metric_seam/test_hierarchy_prevalence.py \
  methods/metric_seam/test_hierarchy_batch.py \
  methods/metric_seam/test_hierarchy_cpu_work_ledger.py \
  methods/metric_seam/test_hierarchy_code_review_registry.py -q
```

Canonical accelerator-masked verification:

```bash
python -m methods.metric_seam.run_cpu_tests
```
