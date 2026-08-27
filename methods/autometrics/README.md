# autometrics

## Purpose

AutoMetrics automatically discovers, generates, and aggregates evaluation
metrics for NLP tasks. Given a dataset with `(input, output, [reference],
target_score)`, it proposes candidate LLM-as-a-judge / rubric / code metrics,
retrieves complementary metrics from a built-in bank of 40+, scores all
metrics on the data, and regresses to a small subset that best predicts the
target — yielding both a top-N metric list and a single aggregated metric.

This is the codebase for **Algorithm 1** in our three-algorithm comparison
("autometrics iterative" — flat feature generation, no tree). See
`/Users/spangher/.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/project_three_algorithms.md`.

## Algorithm sketch

1. **Generate** candidate metrics from a generator LLM, conditioned on
   task description + sample (input, output) pairs.
2. **Retrieve** top-K from the metric bank using BM25 / ColBERT / LLMRec
   (`PipelinedRec`). GPU vs. CPU defaults differ — see
   `_get_default_retriever_config` in `autometrics/autometrics.py`.
3. **Score** the retrieved + generated metrics on the dataset.
4. **Regress** (Lasso / Ridge / ElasticNet / PLS / HotellingPLS) to pick
   the top-N predictive metrics.
5. **Aggregate** into a single regression metric and emit a report card.

Iterative-refinement variant (`autometrics/iterative_refinement/runner.py`):
contrastive rubric proposer generates metrics, fits a logistic / MLP head,
identifies failure pairs, proposes new metrics that fix those, repeats with
dedup + self-critique signatures.

## Key files

- `autometrics/autometrics/autometrics.py` — main `Autometrics` class; the
  generate → retrieve → score → regress → aggregate pipeline.
- `autometrics/autometrics/iterative_refinement/runner.py` — the iterative
  contrastive metric-proposer loop (the one actually used in this project's
  experiments). Includes `_DedupSignature`, `_SelfCritiqueSignature`,
  failure-pair selection (`matching.py`), and label cache.
- `autometrics/autometrics/backends/` — `ScoringBackend` protocol;
  `DSPyBackend` wraps DSPy calls, `VLLMOfflineBackend` uses local
  `vllm.LLM.generate()` for high-volume judge scoring.
- `autometrics/autometrics/generator/ContrastiveRubricProposer.py` —
  proposer that takes pos/neg example pairs and emits a rubric metric.
- `autometrics/autometrics/metrics/generated/GeneratedLLMJudgeMetric.py` —
  LLM-judge metric class; high-volume `_call_llm` routes through the chosen
  backend.
- `autometrics/autometrics/recommend/` — `BM25`, `ColBERT`, `LLMRec`,
  `PipelinedRec` retrieval modules.
- `autometrics/autometrics/aggregator/regression/` — `LogisticL1`,
  `LogisticL1WithInteractions`, `GatedMLP`, `PLS`, `HotellingPLS`.
- `autometrics/autometrics/util/splits.py` — `load_fixed_split` for the
  canonical train/eval/test partitions used across this project.
- `examples/autometrics_simple_example.py` / `autometrics_example.py` /
  `tutorial.py` — runnable entrypoints.

## How to run

Simple end-to-end (defaults, OpenAI-compatible endpoint):

```bash
export OPENAI_API_KEY="sk-..."
python examples/autometrics_simple_example.py
```

Library form:

```python
import dspy
from autometrics.autometrics import Autometrics
from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer

ds = HelpSteer()
am = Autometrics()
results = am.run(
    dataset=ds,
    target_measure="helpfulness",
    generator_llm=dspy.LM("openai/gpt-4o-mini"),
    judge_llm=dspy.LM("openai/gpt-4o-mini"),
)
```

Iterative-refinement runner (used in our experiments):

```bash
python -m autometrics.iterative_refinement.runner \
    --dataset peer_review --output_dir runs/iterative_autometrics/<name>
```

For high-volume scoring on local GPUs, build a vLLM backend:

```python
from autometrics.backends import create_backend
backend = create_backend("vllm", model_name_or_path="meta-llama/Llama-3.1-70B-Instruct")
am.run(..., scoring_backend=backend)
```

## Dependencies

- Python deps: `pip install -e .` (uses `pyproject.toml`).
- Optional extras: `mauve`, `bleurt`, `bert-score`, `rouge`, `reward-models`,
  `gpu` (FlashAttention) — see README front matter for the full list.
- **Java 21** required for Pyserini-backed retrievers (BM25 / ColBERT). See
  the upstream README for install snippets.
- LLM access: an OpenAI-compatible endpoint via `OPENAI_API_KEY`, or a
  local vLLM model for the high-volume judge scoring path.
- GPU: optional for retrieval (ColBERT) and reward-model metrics; the
  iterative runner runs fine CPU-only if judging via an API.

## Current state

Working / actively used as Algorithm 1 in the three-algorithm comparison.
Bugfixes recorded in
`/Users/spangher/.claude/projects/-Users-spangher-Projects-stanford-research-norm-research/memory/autometrics_architecture.md`
(2026-03-09): `runner.py` pair-tuple indexing and `active_coef_map` ordering.
vLLM backend integration landed at the same time.

Recent experiment runs live under
`/Users/spangher/Projects/stanford-research/norm-research/runs/iterative_autometrics/`.

## Outputs

- Disk cache for all metrics: `./autometrics_cache/` (override with
  `AUTOMETRICS_CACHE_DIR`).
- Iterative-refinement runs: `runs/iterative_autometrics/<run_name>/`
  with per-iter `metrics.json`, label cache, lifecycle traces.
- Top-metric reports: `results['top_metrics']`, `results['regression_metric']`,
  and a printed report card from `am.run(...)`.

## Related

- `methods/metric_tree/` — Algorithm 2 (full-scoring tree) and Algorithm 3
  (router-gated tree), which share infrastructure with the iterative runner.
- `methods/dense/` — dense Llama-8B reward model baseline trained on the
  same per-task splits.
- `methods/articulation_star/` — STaR-trained per-datapoint articulator
  (different goal: rationale generation, not metric discovery).
- Memory:
  - `autometrics_architecture.md` — file paths, backend wiring, fixed bugs.
  - `project_three_algorithms.md` — where AutoMetrics fits in the comparison.
