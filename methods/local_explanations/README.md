# local_explanations

## Purpose

Extract evaluation criteria **bottom-up** from individual labeled examples (one rationale per example) rather than proposing global metrics top-down. Per-example features are clustered into a canonical vocabulary, each example is binary-scored against that vocabulary, and an L1 logistic regression on the resulting concept-bottleneck matrix predicts the label.

## Algorithm sketch

Two approaches share Steps 3–6 and differ in Steps 1–2:

**Approach A — Rationalization + Prior Calibration**
1. Estimate a content-free **prior** `p(z | y)` from ~15 generic "what makes a good X?" queries.
2. For each labeled example (label revealed), generate `features_per_example` rationale features.
3. Subtract the prior (TF-IDF–style) so generic templates downweight.
4. Cluster features → canonical vocabulary (UMAP + HDBSCAN, or K-means w/ silhouette sweep) + LLM dedup.
5. Score every (text × canonical feature) cell via vLLM batch judging.
6. Fit L1 logistic regression on the score matrix and report AUC.

**Approach B — STaR-Local (default)**
1. **Blind** extraction: the model argues FOR/AGAINST and predicts the label.
2. A 2×2 weight matrix rewards features on the winning side when the model is correct and penalises misleading features when it is wrong (`weight_{correct,incorrect}_{winning,losing}` in `config.py`). Incorrect predictions are not discarded; they downweight rather than vanish.
3–6 are identical to Approach A.

Both pipelines support either programmatic scoring (`programmatize.py` distils a clustered concept into Python) or LLM scoring.

## Key files

```
methods/local_explanations/
├── config.py                  # LocalExplanationConfig, TaskMetadata, TASK_METADATA_REGISTRY
├── runner.py                  # run_rationalization, run_star_local — main entry points
├── prior.py                   # Approach A prior estimation
├── explainer.py               # per-example LLM extraction + JSONL cache
├── parsing.py                 # parse_feature_list_json, parse_star_local_output
├── feature_weights.py         # STaR 2×2 weight matrix → per-feature score aggregation
├── clustering.py              # K-means / UMAP+HDBSCAN canonical-vocabulary build
├── cluster_features.py        # UMAP+HDBSCAN driver, optional supervised UMAP
├── refactoring_clusterer.py   # LLM-based cluster dedup / cleanup
├── similarity_data.py         # build candidate pairs + LLM-labeled triplets
├── train_similarity_model.py  # fine-tune mpnet-base on triplets
├── scorer.py                  # vLLM batch scoring of (text × feature) cells
├── predictor.py               # L1 logistic regression + AUC
├── programmatize.py           # cluster → Python checker (alternative scorer)
├── rubric_generation.py       # cluster → human-readable rubric
└── prompts.py                 # prompt templates incl. FEW_SHOT_EXAMPLES per task
```

Top-level driver: `scripts/run_local_explanations.py`.

## How to run

```bash
python scripts/run_local_explanations.py \
    --dataset peer-review \
    --approach star_local \
    --proposer-model openai/gpt-5-mini \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.93 \
    --n-canonical-features 30 \
    --max-text-tokens 512 \
    --balance-classes
```

Supported `--dataset` values: `peer-review`, `press-release`, `code-review`, `math-stackexchange`, `notice-and-comment` (with optional `--agency <name>`). Switch to `--approach rationalization` for Approach A.

Hyperparameter sweeps (Optuna over clustering + predictor variants, reusing the cached extraction) live at `scripts/optuna_sweep_local_explanations.py`.

## Dependencies

- vLLM (Llama-3.3-70B-FP8 / BF16 on sk3), `transformers`, `torch`.
- `dspy` (for the OpenAI/Anthropic-backed proposer in prior estimation, clustering, rubric generation).
- `sentence-transformers` (default embedder `all-MiniLM-L6-v2`; LoRA-merged BGE-large in production peer-review runs).
- `umap-learn`, `hdbscan`, `scikit-learn`, `pandas`, `numpy`.
- OpenAI async client (`gpt-5-mini` default for the similarity-pair labeling judge, configurable via `similarity_labeling_model`).

## Current state

In progress. Approach B is the active line.

Per `project_local_explanations_clustering_findings.md` (April 2026), on peer-review:
- 56,152 training abstracts → 429,486 unique features extracted by Llama-3.3-70B-FP8.
- K-means at K=10 collapsed everything to ~3 mega-themes → switched to UMAP + HDBSCAN.
- Default UMAP+HDBSCAN config gives ~420 clusters with 46% noise; **operating point is `umap_target_weight=0.0`, `hdbscan_cluster_selection_epsilon=0.0`, plus LLM dedup**.
- Supervised UMAP and `cluster_selection_epsilon` both failed to improve discrimination.
- 99.7% of features are pure-pos or pure-neg per label — there is no within-feature label disagreement to disentangle.

Open follow-ups (per `project_local_explanations_followups.md`): two-pass "ADDITIONAL features" extraction, anti-pattern few-shots, lift × log-coverage ranking, per-example topic-conditioned prompting.

## Outputs

- `outputs/local_explanations/<task>_<approach>_<model>/`
  - `explanation_cache/star_local_v1.jsonl` — raw per-example LLM responses (deterministic given model+prompt+dataset; reuse across sweeps).
  - `canonical_features.jsonl` — final vocabulary.
  - `cluster_assignments/`, `similarity_pairs/`, `triplets/` — intermediate clustering artifacts.
  - `feature_matrix_{train,eval,test}.parquet` — (text × canonical feature) binary scores.
  - `predictor_{train,eval,test}_metrics.json` — AUC, top L1 coefficients.

Existing concrete run dirs include `peer_review_star_local_llama33_v2`, `peer_review_optuna_sweep_v1`, `peer_review_sweep_best_trial11`.

## Related

- `project_local_explanations_design.md` — two-approach design + required baselines (no-articulation; rubric+datapoint-in-LLM).
- `project_local_explanations_followups.md` — anti-mode-collapse follow-ups.
- `project_local_explanations_hyperparam_sweep.md` — Optuna sweep design + Step 1/2 cache reuse boundary.
- `project_local_explanations_clustering_findings.md` — peer-review clustering empirics.
- `feedback_local_explanations_per_task_fewshots.md` — adding a new task requires both `TASK_METADATA_REGISTRY` (`config.py`) **and** `FEW_SHOT_EXAMPLES` (`prompts.py`).
- Sibling methods: `methods/autometrics/` (top-down metric proposal), `methods/metric_tree/` (hierarchical), `methods/verification_library/` (per-example Python programs).
