# metrics_tree_infilling

## Purpose

LLM feature discovery over a labeled corpus via a **gap-detecting classification
tree**. Given a corpus of binary-labeled items (`y ∈ {0,1}`) and an explicit
metric set (code-based `score(text)→float` scorers + frozen LLM-judge rubrics),
this method fits a model-based recursive-partitioning tree, flags terminal
nodes where the explicit metrics under-predict, asks an LLM for the missing
feature in each gap, and folds the new feature back in. The newer-and-cleaner
sibling of `methods/metric_tree/` — see
[`project_metrics_tree_infilling`](../../.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/project_metrics_tree_infilling.md).

## Algorithm sketch

1. **Discover/test split** — honest 70/30 (`io_metrics.discover_test_split`).
   Everything below is fit on `discover`, all keep/drop guards live on `test`.
2. **Gap-detecting MOB tree** (`mob/glmtree.py`, `mob/mfluctuation.py`). Each
   node holds a logistic GLM of the label as a function of metric **levels**
   `X`; the tree splits on partitioning covariates `z` wherever that
   metric→label relationship is **unstable** across subpopulations. Instability
   is detected with the M-fluctuation parameter-instability test
   (Zeileis & Hornik 2007): sup-LM for numeric `z`, χ² for categorical, an
   outer-product covariance matching `partykit`'s default, and a **permutation
   null** in place of the asymptotic Brownian-bridge p-value.
3. **Flag gap nodes** (`gaps.py`) — terminal nodes whose held-out fit is poor
   *and* whose remaining `z` candidates fail to split.
4. **Invent the missing feature** (`contrast.py` + `feature_gen.py`). Inside
   each gap, contrast the items the metrics get **wrong** against the ones
   they get **right** (not positives vs. negatives), ask the proposer for the
   one distinguishing property not already covered, distill it to a cheap
   reproducible scorer, and materialize it over the whole corpus.
5. **Measure depth, not assign it** (`depth_dial.py`). The new feature's
   *minimal depth* on held-out data tells you how general it is: shallow =
   governs much of the population, deep = a narrow conditional feature.
6. **Guards** (`guards.py`) — redundancy R² vs. existing metrics, gap-closure on
   test, measured importance, reliability discount. The outer loop
   (`loop.py`, `run_infill`) iterates discover → detect → generate →
   materialize → reinsert → guard.

## Key files

| File | Spec § | Role |
|---|---|---|
| `config.py` | all | `InfillConfig` hyperparameters |
| `io_metrics.py` | §1 | load metrics + corpus; materialize levels `X`, covariates `z`; vLLM judge scorer; honest discover/test split |
| `mob/mfluctuation.py` | §2 | M-fluctuation instability test (sup-LM / χ², permutation p-value) |
| `mob/glmtree.py` | §2 | `GapTree`: per-node logistic GLM, exhaustive cutpoint split, routing, `minimal_depth` |
| `gaps.py` | §2.1 | flag terminal nodes with poor held-out fit |
| `contrast.py` | §3, §6 | residualized WRONG/RIGHT contrast; pooling for root-level features |
| `feature_gen.py` | §4 | proposer → `{name, description, rubric}`; distilled scorer + reliability |
| `depth_dial.py` | §6 | embed + cluster gap descriptions to decide what to pool |
| `guards.py` | §7 | redundancy R², gap-closure, measured importance, reliability discount |
| `loop.py` | §8 | outer discover→detect→generate→materialize→reinsert→guard loop (`run_infill`) |
| `run.py` | — | CLI entry point |
| `tests/test_mfluctuation.py` | §2 | engine unit tests (planted instability) |
| `tests/test_loop_smoke.py` | §1–§8 | end-to-end loop on a synthetic planted-feature corpus (no LLM) |
| `tests/validate_against_partykit.py` | §2, §10 | Python self-validation + R `partykit` parity (when R is available) |
| `tests/test_scenario/` | §1–§7 | creature-dossier corpus: recovers two tacit norms (broad + narrow) from text, rejects decoys, measures generality as coverage (see its README) |

The within-node model (`X`) and the splitting covariates (`z`) are separated by each
metric's `role` (`feature` / `context` / `both`): discovered features enter `X` only, so the
tree never fragments regions on a feature's raw value — its coefficient instability across the
context covariates still drives the splits (standard MOB practice).

**z-design matters (2026-07-01).** Every column offered as `z` divides alpha through the
Bonferroni correction, so offering the whole metric bank as `z` (the old default: levels + NA
indicators, `m_z ≈ 2×n_metrics`) taxes every real moderator ~50×. On creative-writing a real
moderator (`source_half`, label rates 0.44/0.18) sat at raw p=0.003 but adj_p=0.144 under
`m_z=48`, while curated `z = {source_half, text_cluster}` split it at adj_p=0.014 (depth-2
tree). Set `curated_z_only=True` to restrict `z` to `extra_z_columns` + `text_length`
(metrics stay in `X`) — partykit's intended small hypothesized-moderator design. A stump
under bank-wide `z` means "nothing survives an m_z≈48 Bonferroni," not "no structure."

## The tree engine

R's `partykit::glmtree` has **no faithful Python equivalent** (spec §10). The
generalized M-fluctuation test is implemented directly:

- per-observation score contributions `ψ_i = (y_i − p̂_i)·x_i`, decorrelated by
  the outer-product covariance `Ĵ` (the `partykit` default);
- **sup-LM** statistic for numeric `z`, **χ²** for categorical `z` — kept
  faithful to `strucchange`;
- the asymptotic Brownian-bridge p-value is replaced by a **permutation null**.
  The sup-LM null is shared across all numeric `z` because it depends only on
  `ψ` and `n`.

Everything stays in Python at inference time; R is only used (optionally) for
the parity validator.

## Where the metrics come from

- **LLM-judge rubrics** (most metrics):
  `datasets/<task>/online-rubrics/{gpt-parsed,claude-parsed}/**/*.json` →
  `extracted.rubrics_metrics = [{name, description, guidance}, …]`.
- **Code metrics**: a directory of `score(text)→float` modules, e.g.
  `methods/existing_metrics_runner/coded/metrics/`.

Judge metrics are materialized through an injectable `judge_scorer` (default
reuses `LLMClient` from `verification_library`; can be swapped for the
`metric_tree` `score_ternary_subset` path on sk3) or supplied as `precomputed`
levels from the v2 cells DB.

## How to run

```bash
PYTHONPATH=methods python -m metrics_tree_infilling.run \
    --task peer-review --metrics rubric --max-metrics 40 \
    --proposer-backend anthropic \
    --materialize-backend openai_compatible \
    --openai-base-url http://localhost:8000/v1
```

Flags (from `run.py`):

- `--task` — one of `press-release`, `peer-review`, `code-review`,
  `notice-and-comment` (see `DATASET_CONFIGS` in `run.py`).
- `--metrics {rubric,code,both}` — metric source(s).
- `--code-metrics-dir` — directory of `score(text)` modules
  (default `methods/existing_metrics_runner/coded/metrics`).
- `--max-metrics N` — cap the explicit metric set (rubric sources can yield
  thousands).
- `--max-outer-rounds`, `--n-permutations`, `--seed` — loop / engine controls.
- `--proposer-backend {anthropic,openai_compatible}` + `--proposer-model`.
- `--materialize-backend {vllm,openai_compatible,anthropic}` +
  `--materialize-model` + `--openai-base-url` for the bulk judge scorer.
- `--output-dir` — defaults from `InfillConfig.output_dir`.

## Validation

```bash
# fast unit + end-to-end tests (no R, no LLM)
python -m pytest methods/metrics_tree_infilling/tests/

# extensive engine validation (R parity runs iff Rscript + partykit are present)
PYTHONPATH=methods python -m metrics_tree_infilling.tests.validate_against_partykit \
    --n 100 --out report.json
```

## Dependencies

- Pure-Python tree engine — no R required at inference time.
- `numpy`, `pandas`, `scikit-learn`, `scipy`.
- `verification_library.LLMClient` for the default proposer / judge clients
  (and therefore the same API-key story).
- Optional: `Rscript` + `partykit` + `strucchange` + `jsonlite` for the
  partykit-parity validator.
- Optional: a local vLLM endpoint for bulk judge materialization
  (`--materialize-backend openai_compatible --openai-base-url ...`).

## Current state

Built 2026-06-05. All six unit / end-to-end tests pass; engine
self-validation: detection 0.81 / FP 0.0 / cutpoint err 0.008 on 20 planted
scenarios.

**Hard limitation** (spec §9 — stated, not papered over): this loop discovers
missing **main effects** — features that, once materialized, carry marginal
signal. It will **not** find a missing **interaction of absent features** (a
root-level XOR): such structure looks like noise at every node *and* survives
pooling, because neither component correlates with the label marginally even
population-wide. Catching that requires lookahead / interaction search and is
out of scope.

## Outputs

Written to `outputs/metrics_tree_infilling/<task>/`:

- `features.json` — every candidate with `status` (kept / dropped + reason),
  `reliability`, `redundancy_r2`, `gap_closure`, `minimal_depth`, etc.
- `tree_summary.json` — `rounds`, `final_gap_count`, terminal-node count,
  kept-feature names, root standardized coefficients, full nested tree dict
  (each node: `id`, `depth`, `n`, `base_rate`, `terminal`, optional `split`).
- `config.json` — full serialized `InfillConfig`.

## Related

- `methods/metric_tree/` — older partition + LLM-proposer tree (algorithm
  2/3). This package replaces the "propose-from-scratch at every node" loop
  with "start from an explicit metric set, propose only into the gaps".
- `methods/existing_metrics_runner/coded/metrics/` — default `score(text)`
  module directory for `--metrics code`.
- `methods/verification_library/` — provides `LLMClient` for both proposer
  and judge.
- Memory notes:
  [`project_metrics_tree_infilling`](../../.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/project_metrics_tree_infilling.md),
  [`project_three_algorithms`](../../.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/project_three_algorithms.md),
  [`project_metric_specificity`](../../.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/project_metric_specificity.md).
