# metric_tree

Recursive, interpretable classification over a labeled corpus via a hierarchical
**binary-partition tree** whose splits are LLM-proposed rubric features. This is
"algorithm 2 / 3" in the project taxonomy (see
[`project_three_algorithms`](../../.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/project_three_algorithms.md)):
the same tree structure supports two inference modes — *full-scoring with
base-rate leaves* and *router-gated selective deepening*.

## Algorithm sketch

At each node:

1. **Propose** ~`n_rubrics_to_propose` contrastive binary rubrics on items routed
   to that node (`proposer.py` / `autometrics.ContrastiveRubricProposer`). Depths
   `< clustering_depth` use descriptive/clustering metrics; deeper depths use
   discriminative metrics.
2. **Score** the candidate rubrics on the partition (`scoring.py`,
   `build_binary_feature_matrix` / `score_binary_subset`), select the top
   `K = n_binary_metrics_per_level` by mutual information.
3. **Split** the partition into `2^K` cells by the binary scores
   (`partition.py`).
4. **Recurse** until `max_depth`, `min_partition_size`, or
   `min_contrastive_pairs` is hit; the leaf prediction is the partition
   **base rate**.

On top of this, two extensions are implemented in this package:

- **Router-gated inference** (`router.py`, `inference.predict_batch`): a per-node
  MLP on frozen sentence embeddings (`embed_texts`) decides whether each example
  should continue deeper or exit early with the current base rate.
- **Restructuring pipeline** (`restructure.py`, 4-phase: ternary YES/NO/**NA**
  scoring → embedding+LLM dedup → NA-aware rebuild → gap-fill) that lifts
  population-wide features toward the root and pushes conditional ones into
  subtrees. See
  [`project_restructuring_pipeline`](../../.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/project_restructuring_pipeline.md).

## Key files

| File | Role |
|---|---|
| `tree_builder.py` | `build_metric_tree` — the core greedy partition builder |
| `config.py` | `TreeConfig` dataclass (depth, K, balance, router, restructuring) |
| `data_structures.py` | `MetricTree`, `PartitionTreeNode`, `TreeMetric` |
| `proposer.py` | `PartitionMetricProposer` — partition-conditional rubric generator |
| `scoring.py` | binary + ternary scorers, MI ranking, NA-rate, `score_ternary_subset` |
| `partition.py` | partition assignment, contrastive pair counting, pruning |
| `router.py` | `NodeRouter` MLP for router-gated inference |
| `restructure.py` | 4-phase iterative restructuring over the global ternary matrix |
| `inference.py` | `predict_batch`, `predict_root_only` (full-scoring + router paths) |
| `ensemble.py` | `build_metric_tree_ensemble`, `ensemble_predict` |
| `analysis.py` | complexity, articulability gap, depth distribution, summary export |
| `visualization.py` | tree-structure plots, text formatting, complexity-by-depth |
| `serialization.py` | `save_tree` / load helpers |
| `example_selection.py` | clustering / representative example selection + `embed_texts` |
| `mahalanobis.py` | (removed from active path — base-rate prediction is used instead; kept for reference) |

## How to run

The CLI lives in `scripts/run_metric_tree.py` (which adds
`methods/autometrics` + `methods/` to `sys.path` and pulls in
`autometrics.backends.create_backend` for the vLLM scoring backend):

```bash
python scripts/run_metric_tree.py \
    --dataset peer-review \
    --max-depth 3 --n-binary-metrics 3 --n-propose 5 \
    --clustering-depth 2 \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.95 \
    --output-dir outputs/metric_tree/peer_review_partition_tree
```

Useful flags:

- `--dataset` — one of the entries in `DATASET_CONFIGS` in
  `scripts/run_metric_tree.py` (press-release, peer-review, code-review,
  notice-and-comment, plus per-agency `notice-and-comment-<ag>`).
- `--use-router --router-threshold 0.5` — enable router-gated inference
  (algorithm 3).
- `--restructure-iterations N` — enable the 4-phase restructuring pipeline
  (algorithm 2 in its strongest form). `--restructure-{na-threshold,k-min,k-max}`
  tune the NA-aware rebuild.
- `--n-trees N` — ensemble of trees (`ensemble.py`).
- `--proposer-model openai/...` — use an external API model for proposing while
  keeping vLLM for scoring.
- `--balance-classes` — downsample the majority class before building.
- `--min-minority-fraction` — error-based pruning so that pure partitions stop
  branching.

A resume helper lives at `scripts/resume_metric_tree_inference.py` for
restarting inference over a saved tree, and `scripts/test_metric_tree.py`
exercises a small synthetic run.

## Dependencies

- `methods/autometrics/` — supplies the contrastive proposer
  (`ContrastiveRubricProposer`), the vLLM/Anthropic/OpenAI backends
  (`autometrics.backends`), and shared helpers
  (`iterative_refinement.runner._coerce_binary_labels`, `LabelCache`,
  `task_descriptions.get_task_description`).
- `sentence-transformers` (`all-MiniLM-L6-v2` by default) for clustering and the
  router’s frozen embeddings.
- `torch` (router MLP), `scikit-learn` (MI, AUC), `numpy`, `pandas`.

## Current state

In active use; this is the "algorithm 2/3" branch of the project. Known caveats:

- The proposer tends to emit generic checklist criteria even at deep nodes —
  documented in
  [`project_metric_specificity`](../../.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/project_metric_specificity.md).
  The restructuring pipeline is the primary mitigation; `metrics_tree_infilling/`
  is the newer line of work that targets the same problem from a different
  direction.
- `mahalanobis.py` is no longer wired into the default predict path — base-rate
  prediction is used at leaves.

## Outputs

Written under `--output-dir` (default
`outputs/metric_tree/<output_subdir>/`). Subdirectories observed under
`outputs/metric_tree/` include `press_release_70b/`. Each run produces:

- the serialized tree (`save_tree` in `serialization.py`),
- analysis artifacts from `analysis.py` (`export_tree_summary`,
  `compute_articulability_gap`, depth distribution),
- visualizations from `visualization.py` (tree structure, metrics-by-depth).

## Related

- `methods/autometrics/` — algorithm 1 (flat feature generation +
  dense/logistic prediction); the proposer + backend code reused here.
- `methods/metrics_tree_infilling/` — newer gap-detecting MOB approach: starts
  from an explicit metric set and uses an LLM only to fill the holes the tree
  exposes.
- Memory notes:
  [`project_three_algorithms`](../../.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/project_three_algorithms.md),
  [`project_restructuring_pipeline`](../../.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/project_restructuring_pipeline.md),
  [`project_metric_specificity`](../../.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/project_metric_specificity.md).
