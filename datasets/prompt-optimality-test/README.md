# prompt-optimality-test — ISOLATED side experiment (GEPA-standard labeled datasets)

**Purpose.** External-validity leg for the *prompt-optimality* paper only: show the
discovery-vs-value scaling results (Heaps-linear phrasing discovery; joint value saturating in a
handful of criteria) on the standard labeled datasets the GEPA literature uses, and benchmark our
in-house GEPA loop against the **official GEPA implementation** so reviewers cannot attribute
results to a nonstandard optimizer.

## ⚠ STRICT ISOLATION RULES — read before touching anything

1. **These datasets carry gold labels. They must NEVER be used in the main norm-research study.**
   The main study is label-free by standing rule (reconstruction-only: metrics are never
   label-aware; measurement targets are each metric's own verdict pattern, never an external Y).
   Nothing in this folder may be used as a target, anchor, training signal, few-shot source, or
   calibration set for any pipeline outside this folder.
2. **No cross-contamination of artifact stores.** Do not write anything derived from these
   datasets into `outputs/metric_implementer/`, the metric banks
   (`datasets/*/online-rubrics/`), silver-norm sets, `outputs/v2_db/`, or any `_sigs.npz` pool
   consumed by the main estimators. All runs, registries, caches, logs, and results live under
   THIS folder.
3. **Different estimand — keep the writing separated.** Here value is measured against gold
   labels y (label-aware by design). In the main study value is measured against M_i's own
   verdicts (label-free). Numbers from the two designs are not comparable and must never appear
   in the same table without an explicit design column.
4. **Log RAW proposal draws.** Every GEPA candidate proposal (accepted or rejected) gets appended
   to `runs/<dataset>/<arm>/proposals.jsonl` with wall-clock timestamp and lineage. This fixes
   the survivor-bias problem of the main repo's registry curves (which persist only accepted
   versions) and yields the clean draw sequence the scaling estimators want.

## Layout

```
README.md              this file (rules + protocol)
RUNBOOK.md             step-by-step comparison protocol
setup_gepa.sh          pin + install the official gepa package and record versions
download_datasets.py   fetch benchmark datasets into data/ (HF datasets)
data/                  downloaded datasets (git-ignored)
vendor/                pinned clone of github.com/gepa-ai/gepa (git-ignored)
runs/                  all experiment outputs, one dir per dataset × arm
```

## The comparison (summary; details in RUNBOOK.md)

- **Datasets** (GEPA-paper-standard): HotpotQA, HoVer, AIME-2025; optional IFBench/PUPA later.
- **Arm A — official GEPA**: `pip install gepa` (pinned), `gepa.optimize()` with a fixed seed
  prompt, fixed `max_metric_calls` budget, fixed task/reflection LMs.
- **Arm B — our in-house GEPA loop** (the one used throughout norm-research), same seed prompt,
  same budget, same LMs.
- **Readouts**: (1) final val-set accuracy per arm (sanity: our loop ≈ official); (2) the
  prompt-optimality estimators applied to BOTH trajectories' raw proposal logs — discovery
  rarefaction + Heaps fit, exchangeable joint-value rarefaction against gold y, saturation pair
  (y_inf/H(y), τ), paired front-loading D with probe-bootstrap CI.
- **Prediction under test**: phrasing discovery is Heaps-linear on these benchmarks too, while
  joint value against gold labels saturates in a handful of criteria — i.e. the low-dimensionality
  finding is a property of articulable task content generally, not of our norm metrics.

Model calls: task/reflection LMs are GLM over the z.ai subscription HTTP endpoint — a
documented deviation from the offline-batch-vLLM standing rule (these arms are 0-GPU API
runs, and DSPy in the paperexact harness requires an endpoint). Judging Sonnet-or-better or
GLM; GLM subscription API is the default proposer.
