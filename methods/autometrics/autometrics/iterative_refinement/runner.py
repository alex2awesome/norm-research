from __future__ import annotations

import hashlib
import json
import random
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence
import ast

import dspy
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from autometrics.aggregator.regression.LogisticL1 import LogisticL1, LogisticL1WithInteractions
from autometrics.aggregator.regression.GatedMLP import GatedInteractionMLP
from autometrics.dataset.Dataset import Dataset
from autometrics.generator.ContrastiveRubricProposer import ContrastiveRubricProposer
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import (
    GeneratedRefBasedLLMJudgeMetric,
    GeneratedRefFreeLLMJudgeMetric,
)
from autometrics.util.splits import load_fixed_split

from .label_cache import LabelCache
from .lifecycle import MetricLifecycleTracker
from .matching import normalize_pair_id, exact_match, mahalanobis_match, propensity_match, residual_select

_iter_logger = logging.getLogger("autometrics.iterative")


@dataclass
class MetricSpec:
    metric_id: str
    name: str
    rubric_text: str
    rubric: Dict[str, str]
    metric: Any


class _DedupSignature(dspy.Signature):
    """Check if a candidate metric is distinct from existing metrics."""

    existing_metrics: str = dspy.InputField(desc="Existing metrics with rubrics.")
    candidate_metric: str = dspy.InputField(desc="Candidate metric name and rubric.")
    verdict: str = dspy.OutputField(desc="Return 'distinct' or 'duplicate: <metric name>'.")


class _SelfCritiqueSignature(dspy.Signature):
    """Critique a proposed evaluation metric. Determine if it is substantive and would meaningfully distinguish between high- and low-quality items based on genuine content differences, NOT surface-level features like text length, formatting, readability, or writing style."""

    task_description: str = dspy.InputField(desc="Brief description of the task.")
    metric_name: str = dspy.InputField(desc="Name of the proposed metric.")
    metric_rubric: str = dspy.InputField(desc="The metric's full rubric description.")
    verdict: str = dspy.OutputField(
        desc="Return 'substantive' if the metric captures a genuine content-level "
             "distinction, or 'superficial: <reason>' if it is surface-level or trivial."
    )


class _MultiMetricSignature(dspy.Signature):
    """Score multiple metrics at once. Return a JSON object mapping metric names to numeric scores."""

    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    rubrics: str = dspy.InputField(desc="Metric names with rubrics. Return scores for each metric name.")
    input_text: str = dspy.InputField(desc="Input text.")
    output_text: str = dspy.InputField(desc="Output text.")
    scores_json: str = dspy.OutputField(desc="JSON object mapping metric names to numeric scores.")


def _jsonl_append(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def _iter_log(message: str, verbose_only: bool, verbose: bool) -> None:
    if verbose_only and not verbose:
        return
    _iter_logger.info(message)


def _rubric_to_text(rubric: Dict[str, str]) -> str:
    lines = []
    for i in range(1, 6):
        key = f"score{i}_description"
        if key in rubric and rubric[key]:
            lines.append(f"{i}: {rubric[key]}")
    return "\n".join(lines).strip()


def _parse_scores_json(raw: str) -> Dict[str, float]:
    text = (raw or "").strip()
    if not text:
        return {}
    # Try JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {str(k): float(v) for k, v in obj.items()}
    except Exception:
        pass
    # Try to extract JSON object from text
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return {str(k): float(v) for k, v in obj.items()}
    except Exception:
        pass
    # Fallback: literal eval
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return {str(k): float(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}


def _metric_id_from_rubric(rubric_text: str) -> str:
    normalized = _normalize_whitespace(rubric_text).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _ensure_unique_name(name: str, existing: set[str], metric_id: str) -> str:
    if name not in existing:
        return name
    suffix = metric_id[:6]
    candidate = f"{name}_{suffix}"
    if candidate not in existing:
        return candidate
    idx = 2
    while f"{candidate}_{idx}" in existing:
        idx += 1
    return f"{candidate}_{idx}"


def _resolve_column(df: pd.DataFrame, preferred: str, fallback: Optional[Sequence[str] | str]) -> str:
    if preferred in df.columns:
        return preferred
    if fallback:
        fallbacks = [fallback] if isinstance(fallback, str) else list(fallback)
        for cand in fallbacks:
            if cand in df.columns:
                return cand
    raise ValueError(f"Required column missing: {preferred} (fallback={fallback})")


def _coerce_binary_labels(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    labels = pd.to_numeric(df[label_column], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"Label column {label_column} contains non-numeric values.")
    unique_vals = sorted(labels.unique().tolist())
    if len(unique_vals) != 2:
        raise ValueError(f"Label column {label_column} must be binary; found {unique_vals}.")
    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
    df = df.copy()
    df[label_column] = labels.map(mapping).astype(int)
    return df


def _truncate_text(text: str, max_chars: int = 1500) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _compute_metric_stats(
    scored_frame: "pd.DataFrame",
    active_specs: list,
    label_column: str,
    coef_map: Optional[Dict[str, float]] = None,
) -> str:
    """Compute per-metric summary statistics from scored data.

    Returns a formatted string showing each metric's mean, std, label
    correlation, and coefficient so the proposer knows what's covered.
    """
    import numpy as np

    lines = []
    y = scored_frame[label_column].values.astype(float) if label_column in scored_frame.columns else None

    for spec in active_specs:
        if spec.name not in scored_frame.columns:
            continue
        vals = scored_frame[spec.name].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            continue

        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))

        corr_str = ""
        if y is not None and valid.sum() > 2:
            both_valid = valid & ~np.isnan(y)
            if both_valid.sum() > 2:
                corr = float(np.corrcoef(vals[both_valid], y[both_valid])[0, 1])
                corr_str = f"  corr_with_label={corr:+.3f}"

        coef_str = ""
        if coef_map and spec.metric_id in coef_map:
            coef_str = f"  coef={coef_map[spec.metric_id]:+.3f}"

        lines.append(
            f"  - {spec.name}: mean={mean:.2f} std={std:.2f}{corr_str}{coef_str}"
        )

    if not lines:
        return ""

    return (
        "\n\n--- METRIC PERFORMANCE STATISTICS ---\n"
        "Score distribution and predictive signal for each active metric "
        "(higher |corr_with_label| = more predictive; higher |coef| = more "
        "important in the current model). Prioritize proposing metrics that "
        "capture different signals from the high-|coef| metrics below.\n"
        + "\n".join(lines)
    )


def _condense_metric_description(spec: MetricSpec) -> str:
    """Condense a metric's full 5-level rubric into a 1-2 sentence summary.

    Extracts the score-1 (lowest) and score-5 (highest) descriptions and
    combines them into a concise description of what the metric measures.
    """
    rubric = spec.rubric
    low = rubric.get("score1_description", "").strip()
    high = rubric.get("score5_description", "").strip()

    if high and low:
        return f"{spec.name}: Ranges from '{low}' (1) to '{high}' (5)."
    if high:
        return f"{spec.name}: Highest score means '{high}'."
    if low:
        return f"{spec.name}: Lowest score means '{low}'."
    return f"{spec.name}: {spec.rubric_text[:150]}"


def _generate_iteration_reasoning(
    scoring_backend: Any,
    generator_llm: Any,
    task_description: str,
    iteration: int,
    contrastive_text: List[str],
    active_metric_summaries: List[str],
    prev_reasoning: Optional[str] = None,
    prev_metrics_proposed: Optional[List[str]] = None,
    misclassified_examples: Optional[str] = None,
) -> str:
    """LLM call that reasons about what to measure before proposing metrics.

    At iteration 1: reasons about task aspects and pos/neg differences.
    At iteration >1: first reflects on why previous metrics failed, then
    reasons about what new aspects to target.

    Returns 2-3 sentence reasoning summary.
    """
    parts = [f"Task: {task_description}"]

    if iteration > 1 and prev_reasoning:
        parts.append(
            f"\nIn the previous iteration, our reasoning was:\n{prev_reasoning}"
        )
        if prev_metrics_proposed:
            parts.append(
                f"\nWe proposed these metrics: {', '.join(prev_metrics_proposed)}"
            )
        if misclassified_examples:
            parts.append(
                f"\nHere are examples we still got wrong:\n{misclassified_examples}"
            )
        parts.append(
            "\nFirst, briefly explain why you think the previous metrics failed "
            "to capture the distinction. What patterns in the misclassified examples "
            "suggest our metrics missed?"
        )

    if active_metric_summaries:
        parts.append(
            "\nCurrent active metrics:\n" + "\n".join(f"- {s}" for s in active_metric_summaries)
        )

    if contrastive_text:
        parts.append(
            "\nContrastive pairs (similar scored examples with opposite labels):\n"
            + "\n\n".join(contrastive_text[:3])
        )

    parts.append(
        "\nBased on the above, write a 2-3 sentence reasoning summary that identifies: "
        "(1) the core aspects of the task that distinguish positive from negative examples, "
        "(2) what specific textual or structural differences the contrastive pairs reveal, "
        "and (3) what NEW dimensions should be measured that current metrics do NOT capture. "
        "Be specific and concrete — name the patterns you see."
    )

    prompt = "\n".join(parts)

    if scoring_backend is not None and hasattr(scoring_backend, "generate_text"):
        return scoring_backend.generate_text(prompt, max_tokens=512)
    if generator_llm is not None:
        try:
            with dspy.settings.context(lm=generator_llm):
                response = generator_llm(prompt, max_tokens=512)
            if isinstance(response, list) and response:
                return str(response[0])
            return str(response)
        except Exception:
            return ""
    return ""


def _generate_trajectory_summary(
    scoring_backend: Any,
    generator_llm: Any,
    iteration_reasonings: List[Tuple[int, str, List[str]]],
) -> str:
    """Aggregate all prior iteration reasonings into a meta-summary.

    Each entry in iteration_reasonings is (iteration_num, reasoning_text,
    list_of_metric_names_proposed).

    Returns a concise trajectory summary: what's been tried, what worked,
    what failed, and what remains unexplored.
    """
    if not iteration_reasonings:
        return ""

    history_lines = []
    for iter_num, reasoning, metric_names in iteration_reasonings:
        names_str = ", ".join(metric_names) if metric_names else "(none survived)"
        history_lines.append(
            f"Iteration {iter_num}:\n"
            f"  Reasoning: {reasoning}\n"
            f"  Metrics proposed: {names_str}"
        )

    prompt = (
        "Below is the history of our iterative metric development process. "
        "Each iteration shows our reasoning about what to measure and which "
        "metrics we proposed.\n\n"
        + "\n\n".join(history_lines)
        + "\n\nWrite a concise trajectory summary (3-5 sentences) covering:\n"
        "1. What approaches have been tried so far\n"
        "2. What seems to be working (metrics that survived and have signal)\n"
        "3. What has failed or been redundant\n"
        "4. What directions remain unexplored and should be tried next\n"
        "Be specific — reference actual metric names and patterns."
    )

    if scoring_backend is not None and hasattr(scoring_backend, "generate_text"):
        return scoring_backend.generate_text(prompt, max_tokens=512)
    if generator_llm is not None:
        try:
            with dspy.settings.context(lm=generator_llm):
                response = generator_llm(prompt, max_tokens=512)
            if isinstance(response, list) and response:
                return str(response[0])
            return str(response)
        except Exception:
            return ""
    return ""


def _cluster_metrics_thematically(
    active_specs: list,
    scoring_backend: Any = None,
    generator_llm: Any = None,
) -> str:
    """Ask the LLM to cluster active metrics into themes and identify gaps."""
    if len(active_specs) < 3:
        return ""
    metrics_text = "\n".join([f"- {s.name}: {s.rubric_text[:200]}" for s in active_specs])
    prompt = (
        "Below are the current evaluation metrics. Group them into thematic clusters "
        "(e.g., 'Writing Quality', 'Factual Content', 'Audience Impact'). For each cluster, "
        "list the metrics and indicate if that theme is OVER-REPRESENTED (has too many similar "
        "metrics) or UNDER-REPRESENTED. Suggest 2-3 theme areas that are completely MISSING "
        "and would be most valuable for distinguishing high-quality from low-quality items.\n\n"
        f"Metrics:\n{metrics_text}\n\n"
        "Format: list clusters, then missing themes. Be concise."
    )
    if scoring_backend is not None and hasattr(scoring_backend, "generate_text"):
        return scoring_backend.generate_text(prompt, max_tokens=1024)
    if generator_llm is not None:
        try:
            with dspy.settings.context(lm=generator_llm):
                response = generator_llm(prompt, max_tokens=1024)
            if isinstance(response, list) and response:
                return str(response[0])
            return str(response)
        except Exception:
            return ""
    return ""


def _compute_train_assessment(
    df: pd.DataFrame,
    label_column: str,
    probs: np.ndarray,
) -> Dict[str, float]:
    y_true = df[label_column].values
    y_pred = (probs >= 0.5).astype(int)
    correct = int((y_pred == y_true).sum())
    n = int(len(y_true))
    acc = float(correct / n) if n > 0 else float("nan")
    return {"n": n, "correct": correct, "accuracy": acc}


def _format_examples(df: pd.DataFrame, id_column: str, text_column: str, label_column: str) -> str:
    lines: List[str] = []
    for _, row in df.iterrows():
        lines.append(f"[id={row[id_column]} label={row[label_column]}]\n{_truncate_text(str(row[text_column]))}")
    return "\n\n".join(lines).strip()


def _normalize_rubric(rubric: Any, scale: str) -> Dict[str, str]:
    if isinstance(rubric, dict):
        if any(k.startswith("score") for k in rubric.keys()):
            return {k: str(v) for k, v in rubric.items()}
        if all(k in rubric for k in ["1", "2", "3", "4", "5"]):
            return {f"score{i}_description": str(rubric[str(i)]) for i in range(1, 6)}
    if isinstance(rubric, list) and len(rubric) >= 5:
        return {f"score{i}_description": str(rubric[i - 1]) for i in range(1, 6)}
    if isinstance(rubric, str):
        if scale == "binary":
            return {
                "score1_description": f"No: {rubric}",
                "score2_description": "N/A",
                "score3_description": "N/A",
                "score4_description": "N/A",
                "score5_description": f"Yes: {rubric}",
            }
        return {
            "score1_description": rubric,
            "score2_description": "",
            "score3_description": "",
            "score4_description": "",
            "score5_description": "",
        }
    # Fallback for missing or unknown formats
    if scale == "binary":
        return {
            "score1_description": "No",
            "score2_description": "N/A",
            "score3_description": "N/A",
            "score4_description": "N/A",
            "score5_description": "Yes",
        }
    return {
        "score1_description": "Very low",
        "score2_description": "Low",
        "score3_description": "Medium",
        "score4_description": "High",
        "score5_description": "Very high",
    }


def _build_metric_from_parts(
    *,
    name: str,
    rubric: Dict[str, str],
    rubric_text: str,
    judge_llm: dspy.LM,
    task_description: str,
    has_references: bool,
    existing_names: set[str],
    metric_id: Optional[str] = None,
    scoring_backend: Optional[Any] = None,
) -> MetricSpec:
    metric_id = metric_id or _metric_id_from_rubric(rubric_text)
    name = _ensure_unique_name(str(name), existing_names, metric_id)
    existing_names.add(name)

    axis = f"{name}\nRubric:\n{rubric_text}"
    metric_cls = GeneratedRefBasedLLMJudgeMetric if has_references else GeneratedRefFreeLLMJudgeMetric
    metric = metric_cls(
        name=name,
        description=f"Contrastive metric: {name}",
        axis=axis,
        rubric=rubric,
        model=judge_llm,
        task_description=task_description,
        scoring_backend=scoring_backend,
    )
    return MetricSpec(metric_id=metric_id, name=name, rubric_text=rubric_text, rubric=rubric, metric=metric)


def _build_feature_frame(
    df: pd.DataFrame,
    metrics: List[MetricSpec],
    label_cache: LabelCache,
    id_column: str,
    text_column: str,
    label_column: str,
    batch_size: int,
    verbose: bool = False,
    stage: str = "",
    score_all_metrics_together: bool = True,
    judge_llm: Optional[dspy.LM] = None,
    task_description: Optional[str] = None,
    use_tqdm: bool = False,
    llm_parallelism: int = 1,
    scoring_backend: Optional[Any] = None,
) -> pd.DataFrame:
    out = df[[id_column, label_column]].copy()
    if score_all_metrics_together and metrics:
        if task_description is None:
            raise ValueError("task_description is required for batched metric scoring.")
        if judge_llm is None and scoring_backend is None:
            raise ValueError("Either judge_llm or scoring_backend must be provided for batched metric scoring.")
        ids = df[id_column].astype(str).tolist()
        cached_sets = [label_cache.available_ids(spec.metric_id) for spec in metrics]
        cached_all = set.intersection(*cached_sets) if cached_sets else set()
        missing_ids = [i for i in ids if i not in cached_all]
        if verbose:
            _iter_log(
                f"[Autometrics][Iterative] Batched scoring for {len(metrics)} metrics: total={len(ids)} cached_all={len(cached_all)} missing={len(missing_ids)}{f' ({stage})' if stage else ''}",
                verbose_only=False,
                verbose=verbose,
            )
        if missing_ids:
            rubrics_text = "\n\n".join([f"{m.name}\n{m.rubric_text}" for m in metrics])
            missing_df = df[df[id_column].astype(str).isin(missing_ids)]
            scored_map: Dict[str, Dict[str, float]] = {m.name: {} for m in metrics}

            # ── Backend path: delegate to ScoringBackend ──
            # Send all missing samples at once — VLLM handles batching internally.
            if scoring_backend is not None:
                metric_names = [m.name for m in metrics]
                if verbose:
                    _iter_log(
                        f"[Autometrics][Iterative] Backend scoring {len(missing_df)} samples ({stage})",
                        verbose_only=False,
                        verbose=verbose,
                    )
                all_ids = missing_df[id_column].astype(str).tolist()
                all_texts = missing_df[text_column].astype(str).tolist()
                # Use single-text scoring if available (text appears once in
                # prompt instead of twice, nearly halving prompt size).
                if hasattr(scoring_backend, "score_single_text_batch"):
                    responses = scoring_backend.score_single_text_batch(
                        task_description=task_description,
                        rubrics_text=rubrics_text,
                        metric_names=metric_names,
                        texts=all_texts,
                    )
                else:
                    responses = scoring_backend.score_multi_metric_batch(
                        task_description=task_description,
                        rubrics_text=rubrics_text,
                        metric_names=metric_names,
                        inputs=all_texts,
                        outputs=all_texts,
                    )
                for row_id, resp in zip(all_ids, responses):
                    for spec in metrics:
                        scored_map[spec.name][row_id] = float(resp.scores.get(spec.name, 0.0))
            else:
                # ── Original DSPy path ──
                module = dspy.ChainOfThought(_MultiMetricSignature)
                thread_local = None
                if llm_parallelism > 1:
                    import threading
                    thread_local = threading.local()
                if llm_parallelism < 1:
                    llm_parallelism = 1
                if verbose and llm_parallelism > 1:
                    _iter_log(
                        f"[Autometrics][Iterative] Batched scoring parallelism={llm_parallelism}{f' ({stage})' if stage else ''}",
                        verbose_only=False,
                        verbose=verbose,
                    )
                progress = None
                if use_tqdm:
                    try:
                        import tqdm as _tqdm  # type: ignore
                        progress = _tqdm.tqdm(
                            total=len(missing_df),
                            desc=f"Scoring {stage or 'metrics'}",
                            dynamic_ncols=True,
                            leave=False,
                        )
                    except Exception:
                        progress = None
                for start in range(0, len(missing_df), batch_size):
                    chunk_df = missing_df.iloc[start : start + batch_size]
                    if verbose:
                        _iter_log(
                            f"[Autometrics][Iterative] Batched scoring chunk {start}:{start+len(chunk_df)} ({stage})",
                            verbose_only=False,
                            verbose=verbose,
                        )
                    rows = list(chunk_df.itertuples(index=False, name=None))
                    cols = list(chunk_df.columns)
                    try:
                        id_idx = cols.index(id_column)
                        text_idx = cols.index(text_column)
                    except ValueError:
                        id_idx = 0
                        text_idx = 0

                    def _get_module() -> Any:
                        if thread_local is None:
                            return module
                        if not hasattr(thread_local, "module"):
                            thread_local.module = dspy.ChainOfThought(_MultiMetricSignature)
                        return thread_local.module

                    def _score_row(row_tuple: tuple) -> tuple[str, Dict[str, float]]:
                        row_id = str(row_tuple[id_idx])
                        text_val = str(row_tuple[text_idx])
                        local_module = _get_module()
                        with dspy.settings.context(lm=judge_llm):
                            pred = local_module(
                                task_description=task_description,
                                rubrics=rubrics_text,
                                input_text=text_val,
                                output_text=text_val,
                            )
                        scores = _parse_scores_json(getattr(pred, "scores_json", ""))
                        return row_id, scores

                    if llm_parallelism == 1:
                        for row in rows:
                            row_id, scores = _score_row(row)
                            for spec in metrics:
                                scored_map[spec.name][row_id] = float(scores.get(spec.name, 0.0))
                            if progress is not None:
                                progress.update(1)
                    else:
                        import concurrent.futures as _futures
                        with _futures.ThreadPoolExecutor(max_workers=llm_parallelism) as executor:
                            futures = [executor.submit(_score_row, row) for row in rows]
                            for future in _futures.as_completed(futures):
                                try:
                                    row_id, scores = future.result()
                                except Exception:
                                    continue
                                for spec in metrics:
                                    scored_map[spec.name][row_id] = float(scores.get(spec.name, 0.0))
                                if progress is not None:
                                    progress.update(1)
                if progress is not None:
                    progress.close()
            # Persist to caches
            for spec in metrics:
                id_list = list(scored_map[spec.name].keys())
                score_list = [scored_map[spec.name][i] for i in id_list]
                label_cache.set_scores(spec.metric_id, id_list, score_list)
        # Build output from caches
        for spec in metrics:
            scores = label_cache.get_scores(
                metric_id=spec.metric_id,
                metric=spec.metric,
                df=df,
                id_column=id_column,
                text_column=text_column,
                batch_size=batch_size,
            )
            out[spec.name] = scores
        return out
    if verbose:
        _iter_log(
            f"[Autometrics][Iterative] Scoring {len(df)} rows for {len(metrics)} metrics{f' ({stage})' if stage else ''}",
            verbose_only=False,
            verbose=verbose,
        )
    for spec in metrics:
        stats: Dict[str, Any] = {}
        scores = label_cache.get_scores(
            metric_id=spec.metric_id,
            metric=spec.metric,
            df=df,
            id_column=id_column,
            text_column=text_column,
            batch_size=batch_size,
            stats=stats,
            verbose=verbose,
            log_prefix=f"[{stage}][{spec.name}] " if stage else f"[{spec.name}] ",
        )
        out[spec.name] = scores
        if verbose:
            _iter_log(
                f"[Autometrics][Iterative] Metric {spec.name}: total={stats.get('total')} cached={stats.get('cached')} new_scored={stats.get('new_scored')}",
                verbose_only=False,
                verbose=verbose,
            )
    return out


def _select_active_metric_ids(
    metric_specs: List[MetricSpec],
    coef_map: Dict[str, float],
    interaction_coef_map: Optional[Dict[str, Tuple[str, str, float]]] = None,
) -> List[str]:
    # A metric is active if it has a non-zero direct coefficient
    # OR participates in any non-zero interaction
    active_set: set[str] = set()
    name_to_id = {spec.name: spec.metric_id for spec in metric_specs}

    # Direct coefficients
    for spec in metric_specs:
        if abs(coef_map.get(spec.metric_id, 0.0)) > 1e-6:
            active_set.add(spec.metric_id)

    # Interaction coefficients: keep base metrics that participate
    if interaction_coef_map:
        for _col_name, (metric_a_name, metric_b_name, coef) in interaction_coef_map.items():
            if abs(coef) > 1e-6:
                if metric_a_name in name_to_id:
                    active_set.add(name_to_id[metric_a_name])
                if metric_b_name in name_to_id:
                    active_set.add(name_to_id[metric_b_name])

    if active_set:
        # Preserve original spec order
        return [spec.metric_id for spec in metric_specs if spec.metric_id in active_set]

    # Fallback: keep the strongest metric to avoid empty active set
    if not metric_specs:
        return []
    ranked = sorted(metric_specs, key=lambda s: abs(coef_map.get(s.metric_id, 0.0)), reverse=True)
    return [ranked[0].metric_id]


def _name_interaction(
    metric_a_name: str,
    metric_a_rubric: str,
    metric_b_name: str,
    metric_b_rubric: str,
    task_description: str,
    scoring_backend: Optional[Any] = None,
    generator_llm: Optional[Any] = None,
) -> str:
    """Ask the LLM to name what the interaction of two metrics captures."""
    prompt = (
        f"Task: {task_description}\n\n"
        f"Two evaluation metrics were found to be meaningful in combination (their interaction term "
        f"has a significant coefficient in a predictive model).\n\n"
        f"Metric A: {metric_a_name}\n{metric_a_rubric}\n\n"
        f"Metric B: {metric_b_name}\n{metric_b_rubric}\n\n"
        f"In one concise sentence, describe what the combination/interaction of these two metrics "
        f"captures that neither metric alone would. Focus on the semantic meaning, not statistics."
    )
    if scoring_backend is not None and hasattr(scoring_backend, "generate_text"):
        return scoring_backend.generate_text(prompt, max_tokens=128).strip()
    if generator_llm is not None:
        import dspy
        with dspy.settings.context(lm=generator_llm):
            pred = dspy.Predict("prompt -> description")(prompt=prompt)
            return getattr(pred, "description", "").strip()
    return f"Combination of {metric_a_name} and {metric_b_name}"


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["pr_auc"] = float("nan")
    try:
        if np.std(y_true) > 0 and np.std(y_prob) > 0:
            metrics["pearson_r"] = float(np.corrcoef(y_true, y_prob)[0, 1])
        else:
            metrics["pearson_r"] = float("nan")
    except Exception:
        metrics["pearson_r"] = float("nan")
    return metrics


def _gate_metric(metrics: Dict[str, float]) -> Tuple[str, float]:
    roc_auc = metrics.get("roc_auc")
    if roc_auc is not None and not np.isnan(roc_auc):
        return "roc_auc", roc_auc
    return "accuracy", metrics.get("accuracy", 0.0)


def _llm_dedup(
    generator_llm: dspy.LM,
    candidate: MetricSpec,
    existing_metrics: List[MetricSpec],
    scoring_backend: Optional[Any] = None,
) -> bool:
    if not existing_metrics:
        return True
    existing_text = "\n\n".join(
        [f"{m.name}\n{m.rubric_text}" for m in existing_metrics]
    )
    candidate_text = f"{candidate.name}\n{candidate.rubric_text}"
    if scoring_backend is not None:
        verdict = scoring_backend.check_dedup(
            existing_metrics=existing_text,
            candidate_metric=candidate_text,
        ).strip().lower()
    else:
        with dspy.settings.context(lm=generator_llm):
            prediction = dspy.Predict(_DedupSignature)(
                existing_metrics=existing_text,
                candidate_metric=candidate_text,
            )
        verdict = str(getattr(prediction, "verdict", "")).strip().lower()
    if verdict.startswith("duplicate"):
        return False
    if "duplicate" in verdict:
        return False
    return True


def _self_critique_candidates(
    candidate_defs: list,
    task_description: str,
    generator_llm: dspy.LM,
    scoring_backend: Any = None,
) -> list:
    """Return (filtered_defs, verdicts) after self-critique.

    Uses scoring_backend.critique_metrics_batch if available, else
    falls back to DSPy _SelfCritiqueSignature per candidate.
    Returns list of (candidate_def, verdict_str) tuples.
    """
    if not candidate_defs:
        return []

    results: list = []

    if scoring_backend is not None and hasattr(scoring_backend, "critique_metrics_batch"):
        verdicts = scoring_backend.critique_metrics_batch(
            task_description=task_description,
            candidates=[
                {"name": str(c.get("name", "")), "rubric": str(c.get("rubric", ""))}
                for c in candidate_defs
            ],
        )
        for cand, verdict in zip(candidate_defs, verdicts):
            results.append((cand, verdict))
    else:
        for cand in candidate_defs:
            rubric_str = cand.get("rubric", "")
            if isinstance(rubric_str, dict):
                rubric_str = "\n".join(f"{k}: {v}" for k, v in rubric_str.items())
            try:
                with dspy.settings.context(lm=generator_llm):
                    prediction = dspy.Predict(_SelfCritiqueSignature)(
                        task_description=task_description,
                        metric_name=str(cand.get("name", "")),
                        metric_rubric=str(rubric_str),
                    )
                verdict = str(getattr(prediction, "verdict", "substantive")).strip()
            except Exception:
                verdict = "substantive"
            results.append((cand, verdict))

    return results


def _log_coefficients(
    coef_records: List[Dict[str, Any]],
    iteration: int,
    metric_specs: List[MetricSpec],
    coef_map: Dict[str, float],
) -> None:
    for spec in metric_specs:
        coef_records.append(
            {
                "iteration": iteration,
                "metric_id": spec.metric_id,
                "name": spec.name,
                "coefficient": float(coef_map.get(spec.metric_id, 0.0)),
            }
        )


def _log_iteration_summary(
    *,
    iteration: int,
    phase: str,
    candidates: Sequence[MetricSpec],
    active_ids: Sequence[str],
    active_names: Sequence[str],
    coef_map: Dict[str, float],
    gate_metric: str,
    gate_value: float,
    eval_selection: Dict[str, float],
    eval_gating: Dict[str, float],
    test_metrics: Dict[str, float],
    train_assessed: Optional[Dict[str, float]] = None,
    mismatch_stats: Optional[Dict[str, float]] = None,
    eval_sizes: Optional[Dict[str, int]] = None,
    verbose: bool,
) -> None:
    if not verbose:
        return
    lines = [
        f"[Autometrics][Iterative] Iteration {iteration} summary ({phase})",
        f"- candidates: {len(candidates)}",
        f"- active: {len(active_ids)}",
        f"- gate: {gate_metric}={gate_value:.4f}",
    ]
    if active_names:
        lines.append(f"- active metrics: {', '.join(active_names)}")
    if candidates:
        ranked = sorted(
            candidates,
            key=lambda s: abs(coef_map.get(s.metric_id, 0.0)),
            reverse=True,
        )[:5]
        for spec in ranked:
            coef = float(coef_map.get(spec.metric_id, 0.0))
            kept = "kept" if spec.metric_id in set(active_ids) else "dropped"
            lines.append(f"  - {spec.name}: coef={coef:.4f} ({kept})")
    lines.append(
        f"- eval(selection): acc={eval_selection.get('accuracy', float('nan')):.4f} "
        f"prec={eval_selection.get('precision', float('nan')):.4f} "
        f"recall={eval_selection.get('recall', float('nan')):.4f} "
        f"f1={eval_selection.get('f1', float('nan')):.4f} "
        f"roc_auc={eval_selection.get('roc_auc', float('nan')):.4f}"
    )
    lines.append(
        f"- eval(gating): acc={eval_gating.get('accuracy', float('nan')):.4f} "
        f"prec={eval_gating.get('precision', float('nan')):.4f} "
        f"recall={eval_gating.get('recall', float('nan')):.4f} "
        f"f1={eval_gating.get('f1', float('nan')):.4f} "
        f"roc_auc={eval_gating.get('roc_auc', float('nan')):.4f}"
    )
    lines.append(
        f"- test: acc={test_metrics.get('accuracy', float('nan')):.4f} "
        f"prec={test_metrics.get('precision', float('nan')):.4f} "
        f"recall={test_metrics.get('recall', float('nan')):.4f} "
        f"f1={test_metrics.get('f1', float('nan')):.4f} "
        f"roc_auc={test_metrics.get('roc_auc', float('nan')):.4f} "
        f"pearson_r={test_metrics.get('pearson_r', float('nan')):.4f}"
    )
    if eval_sizes:
        lines.append(
            f"- data sizes: eval_sel={eval_sizes.get('eval_selection', 0)} "
            f"eval_gate={eval_sizes.get('eval_gating', 0)} test={eval_sizes.get('test', 0)}"
        )
    if train_assessed:
        lines.append(
            f"- train assessed: n={int(train_assessed.get('n', 0))} "
            f"correct={int(train_assessed.get('correct', 0))} "
            f"acc={train_assessed.get('accuracy', float('nan')):.4f}"
        )
    if mismatch_stats:
        lines.append(
            f"- mismatches: method={mismatch_stats.get('method', 'n/a')} "
            f"pairs={int(mismatch_stats.get('pairs', 0))} "
            f"points={int(mismatch_stats.get('points', 0))}"
        )
    _iter_log("\n".join(lines), verbose_only=False, verbose=verbose)


def _compute_marginal_contributions(
    model: LogisticL1,
    dataset: Dataset,
    metric_specs: List[MetricSpec],
) -> Dict[str, float]:
    df = dataset.get_dataframe()
    y_true = df[dataset.get_target_columns()[0]].values
    probs = model.predict_proba(dataset)
    base_metrics = _compute_metrics(y_true, probs)
    gate_key, base_value = _gate_metric(base_metrics)

    # For GatedMLP: use feature zeroing approach (set feature column to 0, re-predict)
    if isinstance(model, GatedInteractionMLP):
        import torch
        input_columns = model.get_input_columns()
        X = df[input_columns].replace([np.inf, -np.inf], np.nan).fillna(0).values
        if model.scaler is not None:
            X_scaled = model.scaler.transform(X)
        else:
            X_scaled = X
        contribs: Dict[str, float] = {}
        for idx, spec in enumerate(metric_specs):
            if idx >= X_scaled.shape[1]:
                continue
            X_masked = X_scaled.copy()
            X_masked[:, idx] = 0.0  # zero out this feature
            X_t = torch.tensor(X_masked, dtype=torch.float32)
            with torch.no_grad():
                logits = model._net(X_t).squeeze(1)
                masked_probs = torch.sigmoid(logits).cpu().numpy()
            masked_metrics = _compute_metrics(y_true, masked_probs)
            _, masked_value = _gate_metric(masked_metrics)
            contribs[spec.metric_id] = float(base_value - masked_value)
        return contribs

    coef = getattr(model.model, "coef_", None)
    if coef is None:
        return {}
    coef_vec = np.array(coef).reshape(-1)
    intercept = float(getattr(model.model, "intercept_", [0.0])[0])
    X = df[model.get_input_columns()].values
    if model.scaler is not None:
        X = model.scaler.transform(X)

    # Build map of which interaction coefficient indices involve each base metric
    interaction_indices_for_base: Dict[int, List[int]] = {}
    if isinstance(model, LogisticL1WithInteractions):
        n_base = len(metric_specs)
        spec_names = [s.name for s in metric_specs]
        for inter_idx, (_col_name, col_a, col_b) in enumerate(model.interaction_pairs):
            coef_idx = n_base + inter_idx
            if coef_idx >= len(coef_vec):
                break
            for base_idx, name in enumerate(spec_names):
                if name == col_a or name == col_b:
                    interaction_indices_for_base.setdefault(base_idx, []).append(coef_idx)

    contribs: Dict[str, float] = {}
    for idx, spec in enumerate(metric_specs):
        if idx >= len(coef_vec):
            continue
        masked_coef = coef_vec.copy()
        masked_coef[idx] = 0.0
        # Also zero out interaction terms involving this metric
        for inter_idx in interaction_indices_for_base.get(idx, []):
            masked_coef[inter_idx] = 0.0
        logits = intercept + X.dot(masked_coef)
        masked_probs = 1.0 / (1.0 + np.exp(-logits))
        masked_metrics = _compute_metrics(y_true, masked_probs)
        _, masked_value = _gate_metric(masked_metrics)
        contribs[spec.metric_id] = float(base_value - masked_value)
    return contribs


def run_iterative(
    dataset: Dataset,
    target_measure: str,
    generator_llm: dspy.LM,
    judge_llm: dspy.LM,
    num_iterations: int = 5,
    matching_only: bool = False,
    residual_only: bool = False,
    split_dir: Optional[str] = None,
    data_path: Optional[str] = None,
    id_column: str = "id",
    text_column: str = "text",
    label_column: str = "judgement",
    output_dir: str = "outputs/iterative_autometrics",
    regenerate_metrics: bool = False,
    seed: int = 42,
    k_pairs: int = 5,
    num_metrics: int = 5,
    num_rubrics: int = 5,
    caliper: float = 0.1,
    label_batch_size: int = 200,
    eval_gate_fraction: float = 0.2,
    eval_fraction: float = 0.4,
    max_text_tokens: Optional[int] = 512,
    eval_plateau_eps: float = 0.005,
    early_stop_patience: int = 2,
    disable_early_stopping: bool = False,
    churn_warning_threshold: float = 0.5,
    min_tenure: int = 0,
    score_all_metrics_together: bool = True,
    tqdm_scoring: bool = False,
    llm_parallelism: int = 1,
    scoring_backend: Optional[Any] = None,
    intermediate_test_samples: Optional[int] = 100,
    use_interactions: bool = True,
    max_interaction_metrics: int = 8,
    eval_selection_max: Optional[int] = None,
    model_type: str = "logistic",  # "logistic" or "gated_mlp"
    gated_mlp_hidden_dim: int = 64,
    gated_mlp_lambda_feature: float = 0.1,
    gated_mlp_lambda_interaction: float = 0.05,
    gated_mlp_epochs: int = 200,
    gated_mlp_gate_threshold: float = 0.1,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    del regenerate_metrics  # unused in iterative path
    del kwargs

    random.seed(seed)
    np.random.seed(seed)

    if not data_path:
        raise ValueError("data_path is required for iterative AutoMetrics (fixed splits on disk).")
    if matching_only and residual_only:
        raise ValueError("matching_only and residual_only cannot both be True.")

    orig_train_df, orig_eval_df, test_df = load_fixed_split(
        data_path=data_path,
        split_dir=split_dir,
        create_if_missing=True,
        seed=seed,
        label_column=label_column,
    )
    _iter_log(
        f"[Autometrics][Iterative] Loaded on-disk splits: train={len(orig_train_df)}, eval={len(orig_eval_df)}, test={len(test_df)}",
        verbose_only=False,
        verbose=verbose,
    )

    id_column = _resolve_column(orig_train_df, id_column, None)
    text_column = _resolve_column(orig_train_df, text_column, None)
    label_column = _resolve_column(orig_train_df, label_column, None)

    # Combine train + eval from on-disk splits, then re-split.
    # Test set stays identical for comparability with other model runs.
    combined_df = pd.concat([orig_train_df, orig_eval_df], ignore_index=True)
    combined_df = _coerce_binary_labels(combined_df, label_column)
    test_df = _coerce_binary_labels(test_df, label_column)

    stratify_combined = combined_df[label_column] if combined_df[label_column].nunique() > 1 else None
    train_df, eval_df = train_test_split(
        combined_df,
        test_size=eval_fraction,
        random_state=seed,
        stratify=stratify_combined,
    )

    # Split eval into selection (for model fitting/selection) and gating (for validation)
    stratify_eval = eval_df[label_column] if eval_df[label_column].nunique() > 1 else None
    eval_sel_df, eval_gate_df = train_test_split(
        eval_df,
        test_size=eval_gate_fraction,
        random_state=seed,
        stratify=stratify_eval,
    )
    _iter_log(
        f"[Autometrics][Iterative] Re-split: train={len(train_df)}, eval_sel={len(eval_sel_df)}, eval_gate={len(eval_gate_df)}, test={len(test_df)}",
        verbose_only=False,
        verbose=verbose,
    )

    # Truncate text to first N whitespace tokens to speed up LLM scoring
    if max_text_tokens:
        def _truncate(text):
            tokens = str(text).split()
            if len(tokens) > max_text_tokens:
                return " ".join(tokens[:max_text_tokens])
            return str(text)

        for _df in (train_df, eval_sel_df, eval_gate_df, test_df):
            _df[text_column] = _df[text_column].map(_truncate)
        _iter_log(
            f"[Autometrics][Iterative] Truncated text to first {max_text_tokens} whitespace tokens",
            verbose_only=False,
            verbose=verbose,
        )

    task_description = dataset.get_task_description()
    if not task_description:
        raise ValueError("dataset.task_description is required for iterative AutoMetrics.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    label_cache = LabelCache(cache_dir=str(output_path / "label_cache"))

    run_config = {
        "num_iterations": num_iterations,
        "matching_only": matching_only,
        "residual_only": residual_only,
        "split_dir": split_dir,
        "data_path": data_path,
        "target_measure": target_measure,
        "id_column": id_column,
        "text_column": text_column,
        "label_column": label_column,
        "output_dir": output_dir,
        "k_pairs": k_pairs,
        "num_metrics": num_metrics,
        "num_rubrics": num_rubrics,
        "caliper": caliper,
        "label_batch_size": label_batch_size,
        "eval_gate_fraction": eval_gate_fraction,
        "eval_fraction": eval_fraction,
        "max_text_tokens": max_text_tokens,
        "eval_selection_max": eval_selection_max,
        "tqdm_scoring": tqdm_scoring,
        "llm_parallelism": llm_parallelism,
        "seed": seed,
        "churn_warning_threshold": churn_warning_threshold,
        "min_tenure": min_tenure,
        "verbose": verbose,
        "early_stop_patience": early_stop_patience,
        "disable_early_stopping": disable_early_stopping,
        "use_interactions": use_interactions,
    }
    (output_path / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # Helper: subsample test_df for intermediate iterations to save scoring time
    def _get_test_df(iteration: int, final_iteration: int) -> pd.DataFrame:
        if intermediate_test_samples and iteration < final_iteration and len(test_df) > intermediate_test_samples:
            _iter_log(
                f"[Autometrics][Iterative] Subsampling test set: {intermediate_test_samples}/{len(test_df)} rows (intermediate iteration {iteration})",
                verbose_only=False,
                verbose=verbose,
            )
            stratify_test = test_df[label_column] if test_df[label_column].nunique() > 1 else None
            if stratify_test is not None:
                sampled, _ = train_test_split(
                    test_df, train_size=intermediate_test_samples, random_state=seed, stratify=stratify_test,
                )
                return sampled
            return test_df.sample(n=intermediate_test_samples, random_state=seed)
        return test_df

    final_iteration = num_iterations - 1

    proposer = ContrastiveRubricProposer(generator_llm=generator_llm, seed=seed, scoring_backend=scoring_backend)
    lifecycle = MetricLifecycleTracker()
    seen_pairs: set[Tuple[str, str]] = set()

    metric_bank: Dict[str, MetricSpec] = {}
    active_metric_ids: List[str] = []
    coef_records: List[Dict[str, Any]] = []
    cumulative_train_ids: set[str] = set()
    dropped_metrics_log: List[str] = []
    no_new_metrics_iters = 0
    eval_plateau_iters = 0
    last_eval_gate_score = None
    active_coef_map: Dict[str, float] = {}
    # Interaction tracking: {col_name: (metric_a_name, metric_b_name, coef, description)}
    active_interactions: Dict[str, Tuple[str, str, float, str]] = {}
    # Reasoning chain: list of (iteration, reasoning_text, metric_names_proposed)
    iteration_reasonings: List[Tuple[int, str, List[str]]] = []
    prev_iteration_reasoning: Optional[str] = None

    def build_dataset_from_frame(df: pd.DataFrame, metric_specs: List[MetricSpec], name: str) -> Dataset:
        metric_columns = [m.name for m in metric_specs]
        return Dataset(
            dataframe=df,
            target_columns=[label_column],
            ignore_columns=[id_column],
            metric_columns=metric_columns,
            name=name,
            data_id_column=id_column,
            input_column=text_column,
            output_column=text_column,
            reference_columns=[],
            metrics=[m.metric for m in metric_specs],
            task_description=task_description,
        )

    def fit_regression(
        df_frame: pd.DataFrame,
        metric_specs: List[MetricSpec],
        name: str,
        interaction_names: Optional[List[str]] = None,
    ) -> LogisticL1:
        ds = build_dataset_from_frame(df_frame, metric_specs, name=name)
        if model_type == "gated_mlp" and len(metric_specs) >= 2:
            base_names = [m.name for m in metric_specs]
            model = GatedInteractionMLP(
                base_feature_names=base_names,
                hidden_dim=gated_mlp_hidden_dim,
                lambda_feature=gated_mlp_lambda_feature,
                lambda_interaction=gated_mlp_lambda_interaction,
                n_epochs=gated_mlp_epochs,
                gate_threshold=gated_mlp_gate_threshold,
                dataset=ds,
                input_metrics=[m.metric for m in metric_specs],
            )
        elif use_interactions and len(metric_specs) >= 2:
            base_names = [m.name for m in metric_specs]
            model = LogisticL1WithInteractions(
                base_feature_names=base_names,
                interaction_feature_names=interaction_names,
                dataset=ds,
                input_metrics=[m.metric for m in metric_specs],
            )
        else:
            model = LogisticL1(dataset=ds, input_metrics=[m.metric for m in metric_specs])
        model.learn(ds, target_column=label_column)
        return model

    def compute_probabilities(model: LogisticL1, df_frame: pd.DataFrame, metric_specs: List[MetricSpec], name: str) -> np.ndarray:
        ds = build_dataset_from_frame(df_frame, metric_specs, name=name)
        return model.predict_proba(ds)

    def current_metric_specs() -> List[MetricSpec]:
        return [metric_bank[mid] for mid in active_metric_ids]

    def _extract_coefs_and_interactions(
        model: LogisticL1,
        metric_specs: List[MetricSpec],
        iteration: int,
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[str, str, float]]]:
        """Extract base coefficient map and interaction coefficient map from a fitted model."""
        coef_map = {
            spec.metric_id: float(coef)
            for spec, coef in zip(metric_specs, model.model.coef_.reshape(-1)[:len(metric_specs)])
        }
        interaction_coef_map: Dict[str, Tuple[str, str, float]] = {}
        if isinstance(model, GatedInteractionMLP):
            interaction_coef_map = model.get_interaction_coef_map(metric_specs)
        elif use_interactions and isinstance(model, LogisticL1WithInteractions):
            interaction_coef_map = model.get_interaction_coef_map(metric_specs)
        return coef_map, interaction_coef_map

    def _update_active_interactions(
        interaction_coef_map: Dict[str, Tuple[str, str, float]],
        metric_specs: List[MetricSpec],
    ) -> None:
        """Name new interactions and update active_interactions dict."""
        nonlocal active_interactions
        name_to_spec = {spec.name: spec for spec in metric_specs}
        new_interactions: Dict[str, Tuple[str, str, float, str]] = {}
        for col_name, (metric_a_name, metric_b_name, coef) in interaction_coef_map.items():
            # Reuse existing description if we've already named this interaction
            if col_name in active_interactions:
                _, _, _, prev_desc = active_interactions[col_name]
                new_interactions[col_name] = (metric_a_name, metric_b_name, coef, prev_desc)
            else:
                spec_a = name_to_spec.get(metric_a_name)
                spec_b = name_to_spec.get(metric_b_name)
                rubric_a = spec_a.rubric_text if spec_a else ""
                rubric_b = spec_b.rubric_text if spec_b else ""
                desc = _name_interaction(
                    metric_a_name, rubric_a,
                    metric_b_name, rubric_b,
                    task_description,
                    scoring_backend=scoring_backend,
                    generator_llm=generator_llm,
                )
                new_interactions[col_name] = (metric_a_name, metric_b_name, coef, desc)
                _iter_log(
                    f"[Autometrics][Iterative] Named interaction: {col_name} -> {desc}",
                    verbose_only=False, verbose=verbose,
                )
        active_interactions = new_interactions

    # Iteration 0: contrastive init
    positives = train_df[train_df[label_column] == 1]
    negatives = train_df[train_df[label_column] == 0]
    if positives.empty or negatives.empty:
        raise ValueError("Training split must contain both positive and negative labels.")

    pos_seed = positives.sample(n=min(k_pairs, len(positives)), random_state=seed)
    neg_seed = negatives.sample(n=min(k_pairs, len(negatives)), random_state=seed)

    seed_pairs = []
    for pos_id, neg_id in zip(pos_seed[id_column].astype(str), neg_seed[id_column].astype(str)):
        pair = normalize_pair_id(pos_id, neg_id)
        seen_pairs.add(pair)
        seed_pairs.append("|".join(pair))

    existing_names: set[str] = set()
    has_references = False
    candidate_specs: List[MetricSpec] = []
    dropped_empty = 0
    dropped_dupe = 0
    min_unique = max(1, int(num_metrics))
    max_candidate_attempts = 3
    attempt = 0
    prior_candidate_summaries: List[str] = []
    while attempt < max_candidate_attempts and len(candidate_specs) < min_unique:
        if prior_candidate_summaries:
            prior_blob = "\n".join(prior_candidate_summaries[-50:])
            if len(prior_blob) > 2000:
                prior_blob = prior_blob[-2000:]
            current_metrics_hint = (
                "None. Avoid repeating these previously proposed candidates:\n"
                f"{prior_blob}"
            )
        else:
            current_metrics_hint = "None"
        candidate_defs = proposer.propose(
            task_description=task_description,
            positive_examples=_format_examples(pos_seed, id_column, text_column, label_column),
            negative_examples=_format_examples(neg_seed, id_column, text_column, label_column),
            current_metrics=current_metrics_hint,
            contrastive_pairs="None",
            num_metrics=num_metrics,
            num_rubrics=num_rubrics,
        )
        _iter_log(
            f"[Autometrics][Iterative] Iteration 0 generated {len(candidate_defs)} candidates "
            f"(requested metrics={num_metrics}, rubrics={num_rubrics}, attempt {attempt+1}/{max_candidate_attempts})",
            verbose_only=False,
            verbose=verbose,
        )

        for cand in candidate_defs:
            name = cand.get("name") or "Metric"
            scale = str(cand.get("scale") or "ordinal").lower()
            rubric = _normalize_rubric(cand.get("rubric"), scale)
            rubric_text = _rubric_to_text(rubric)
            if rubric_text:
                short_rubric = rubric_text.replace("\n", " ").strip()
                if len(short_rubric) > 140:
                    short_rubric = short_rubric[:140] + "..."
                prior_candidate_summaries.append(f"- {name}: {short_rubric}")
            if not rubric_text:
                dropped_empty += 1
                dropped_metrics_log.append(f"DROPPED (empty rubric, iter 0): {name}")
                if verbose:
                    _iter_log(
                        f"[Autometrics][Iterative] Dropped candidate (empty rubric): {name}",
                        verbose_only=False,
                        verbose=verbose,
                    )
                continue
            metric_id = _metric_id_from_rubric(rubric_text)
            if metric_id in metric_bank:
                dropped_dupe += 1
                dropped_metrics_log.append(f"DROPPED (duplicate rubric, iter 0): {name}")
                if verbose:
                    _iter_log(
                        f"[Autometrics][Iterative] Dropped candidate (duplicate rubric): {name}",
                        verbose_only=False,
                        verbose=verbose,
                    )
                continue
            spec = _build_metric_from_parts(
                name=name,
                rubric=rubric,
                rubric_text=rubric_text,
                judge_llm=judge_llm,
                task_description=task_description,
                has_references=has_references,
                existing_names=existing_names,
                metric_id=metric_id,
                scoring_backend=scoring_backend,
            )
            metric_bank[spec.metric_id] = spec
            candidate_specs.append(spec)
            lifecycle.register_metric(spec.metric_id, spec.name, spec.rubric_text, 0, source_pairs=seed_pairs)
        attempt += 1

    if not candidate_specs:
        raise RuntimeError("No metrics generated in iteration 0.")
    _iter_log(
        f"[Autometrics][Iterative] Iteration 0 prepared {len(candidate_specs)} candidates for selection "
        f"(dropped_empty={dropped_empty}, dropped_duplicate={dropped_dupe})",
        verbose_only=True,
        verbose=verbose,
    )

    # Subsample eval_selection for seed iteration
    if eval_selection_max and len(eval_sel_df) > eval_selection_max:
        iter_eval_sel_df_0 = eval_sel_df.sample(n=eval_selection_max, random_state=seed)
        _iter_log(
            f"[Autometrics][Iterative] Iteration 0 subsampled eval_selection: {len(eval_sel_df)} → {len(iter_eval_sel_df_0)}",
            verbose_only=False, verbose=verbose,
        )
    else:
        iter_eval_sel_df_0 = eval_sel_df

    eval_sel_frame = _build_feature_frame(
        iter_eval_sel_df_0, candidate_specs, label_cache, id_column, text_column, label_column, label_batch_size,
        verbose=verbose, stage="eval_selection_full",
        score_all_metrics_together=score_all_metrics_together,
        judge_llm=judge_llm,
        task_description=task_description,
        use_tqdm=tqdm_scoring,
        llm_parallelism=llm_parallelism,
        scoring_backend=scoring_backend,
        )
    selection_model = fit_regression(eval_sel_frame, candidate_specs, name="eval_selection_full")
    selection_coef_map, selection_interaction_map = _extract_coefs_and_interactions(
        selection_model, candidate_specs, iteration=0,
    )
    lifecycle.record_coefficients(0, selection_coef_map)
    if selection_interaction_map:
        _update_active_interactions(selection_interaction_map, candidate_specs)

    active_metric_ids = _select_active_metric_ids(
        candidate_specs, selection_coef_map, selection_interaction_map,
    )
    for spec in candidate_specs:
        if spec.metric_id not in active_metric_ids:
            dropped_metrics_log.append(
                f"DROPPED (zero coefficient, iter 0): {spec.name} - {spec.rubric_text[:100]}"
            )
    active_specs = current_metric_specs()
    _iter_log(
        f"[Autometrics][Iterative] Iteration 0 selected {len(active_metric_ids)} active metrics after regression",
        verbose_only=False,
        verbose=verbose,
    )

    eval_sel_frame_active = _build_feature_frame(
        iter_eval_sel_df_0, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
        verbose=verbose, stage="eval_selection_active_0",
        score_all_metrics_together=score_all_metrics_together,
        judge_llm=judge_llm,
        task_description=task_description,
        use_tqdm=tqdm_scoring,
        llm_parallelism=llm_parallelism,
        scoring_backend=scoring_backend,
        )
    eval_gate_frame_active = _build_feature_frame(
        eval_gate_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
        verbose=verbose, stage="eval_gating_active_0",
        score_all_metrics_together=score_all_metrics_together,
        judge_llm=judge_llm,
        task_description=task_description,
        use_tqdm=tqdm_scoring,
        llm_parallelism=llm_parallelism,
        scoring_backend=scoring_backend,
        )
    model = fit_regression(eval_sel_frame_active, active_specs, name="eval_selection_active")

    eval_sel_probs = compute_probabilities(model, eval_sel_frame_active, active_specs, name="eval_selection_active")
    eval_gate_probs = compute_probabilities(model, eval_gate_frame_active, active_specs, name="eval_gating_active")
    eval_sel_metrics = _compute_metrics(eval_sel_frame_active[label_column].values, eval_sel_probs)
    eval_gate_metrics = _compute_metrics(eval_gate_frame_active[label_column].values, eval_gate_probs)
    gate_key, gate_value = _gate_metric(eval_gate_metrics)
    last_eval_gate_score = gate_value

    lifecycle.mark_active(0, active_metric_ids)
    lifecycle.record_marginal_contributions(
        0,
        _compute_marginal_contributions(
            model,
            build_dataset_from_frame(eval_gate_frame_active, active_specs, "eval_gating_active"),
            active_specs,
        ),
    )
    _log_coefficients(coef_records, 0, candidate_specs, selection_coef_map)

    iter0_test_df = _get_test_df(0, final_iteration)
    test_frame = _build_feature_frame(
        iter0_test_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
        verbose=verbose, stage="test_0",
        score_all_metrics_together=score_all_metrics_together,
        judge_llm=judge_llm,
        task_description=task_description,
        use_tqdm=tqdm_scoring,
        llm_parallelism=llm_parallelism,
        scoring_backend=scoring_backend,
        )
    test_probs = compute_probabilities(model, test_frame, active_specs, name="test")
    test_metrics = _compute_metrics(test_frame[label_column].values, test_probs)

    _log_iteration_summary(
        iteration=0,
        phase="init",
        candidates=candidate_specs,
        active_ids=active_metric_ids,
        active_names=[metric_bank[mid].name for mid in active_metric_ids],
        coef_map=selection_coef_map,
        gate_metric=gate_key,
        gate_value=gate_value,
        eval_selection=eval_sel_metrics,
        eval_gating=eval_gate_metrics,
        test_metrics=test_metrics,
        train_assessed={"n": 0, "correct": 0, "accuracy": float("nan")},
        mismatch_stats={"method": "seed", "pairs": len(seed_pairs), "points": len(seed_pairs) * 2},
        eval_sizes={
            "eval_selection": len(eval_sel_frame_active),
            "eval_gating": len(eval_gate_frame_active),
            "test": len(test_frame),
        },
        verbose=verbose,
    )

    _jsonl_append(output_path / "iterations.jsonl", {
        "iteration": 0,
        "active_metric_ids": active_metric_ids,
        "active_metric_names": [metric_bank[mid].name for mid in active_metric_ids],
        "gate_metric": gate_key,
        "eval_gate_score": gate_value,
        "eval_selection": eval_sel_metrics,
        "eval_gating": eval_gate_metrics,
        "test": test_metrics,
        "accepted": True,
        "num_active_metrics": len(active_metric_ids),
        "num_active_interactions": len(active_interactions),
        "active_interactions": {k: {"a": v[0], "b": v[1], "coef": v[2], "desc": v[3]} for k, v in active_interactions.items()},
        "train_assessed": {"n": len(seed_pairs) * 2, "correct": 0, "accuracy": float("nan"), "note": "seed pairs, no model scoring"},
        "mismatch_stats": {"method": "seed", "pairs": len(seed_pairs), "points": len(seed_pairs) * 2},
        "data_sizes": {
            "eval_selection": len(eval_sel_frame_active),
            "eval_gating": len(eval_gate_frame_active),
            "test": len(test_frame),
        },
    })
    _jsonl_append(output_path / "eval_metrics.jsonl", {"iteration": 0, **eval_gate_metrics})
    _jsonl_append(output_path / "test_metrics.jsonl", {"iteration": 0, **test_metrics})

    for pair in seed_pairs:
        _jsonl_append(output_path / "pairs.jsonl", {"iteration": 0, "pair_id": pair, "method": "seed"})

    # Iterative refinement
    for iteration in range(1, num_iterations):
        prev_active_ids = list(active_metric_ids)
        prev_gate_score = last_eval_gate_score
        active_specs = current_metric_specs()
        if not active_specs:
            break
        _iter_log(
            f"[Autometrics][Iterative] Iteration {iteration} starting with {len(active_specs)} active metrics",
            verbose_only=False,
            verbose=verbose,
        )

        # ── Subsample eval_selection for this iteration ──
        if eval_selection_max and len(eval_sel_df) > eval_selection_max:
            iter_eval_sel_df = eval_sel_df.sample(n=eval_selection_max, random_state=seed + iteration)
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} subsampled eval_selection: {len(eval_sel_df)} → {len(iter_eval_sel_df)}",
                verbose_only=False, verbose=verbose,
            )
        else:
            iter_eval_sel_df = eval_sel_df

        # ── Exact-match-driven pool growth ──
        # Strategy: keep labeling fresh batches of training data as long as
        # exact matches (identical rounded metric scores, different labels)
        # can be found.  Only fall back to near-exact / propensity once exact
        # matches are exhausted.
        all_train_ids = set(train_df[id_column].astype(str).tolist())
        feature_columns = [spec.name for spec in active_specs]
        _bff_kwargs = dict(
            label_cache=label_cache, id_column=id_column, text_column=text_column,
            label_column=label_column, batch_size=label_batch_size, verbose=verbose,
            score_all_metrics_together=score_all_metrics_together,
            judge_llm=judge_llm, task_description=task_description,
            use_tqdm=tqdm_scoring, llm_parallelism=llm_parallelism,
            scoring_backend=scoring_backend,
        )

        # Re-score existing cumulative pool with current active metrics
        # (_build_feature_frame only scores IDs missing for new metrics via cache)
        if cumulative_train_ids:
            rescore_ids = cumulative_train_ids & all_train_ids
            if rescore_ids:
                rescore_df = train_df[train_df[id_column].astype(str).isin(rescore_ids)]
                _iter_log(
                    f"[Autometrics][Iterative] Iteration {iteration} re-scoring cumulative pool: "
                    f"{len(rescore_df)} samples with {len(active_specs)} active metrics",
                    verbose_only=False, verbose=verbose,
                )
                _ = _build_feature_frame(
                    rescore_df, active_specs, stage=f"train_rescore_{iteration}", **_bff_kwargs,
                )

        # Growth loop: label batches until exact matches stop improving
        prev_exact_count = -1
        max_growth_rounds = 50
        for growth_round in range(max_growth_rounds):
            # Ensure we have at least one batch labeled
            if not cumulative_train_ids:
                fresh_ids = list(all_train_ids)
                random.shuffle(fresh_ids)
                batch_ids = fresh_ids[:label_batch_size]
                batch_df = train_df[train_df[id_column].astype(str).isin(batch_ids)]
                _iter_log(
                    f"[Autometrics][Iterative] Iteration {iteration} labeling initial batch of {len(batch_df)} samples",
                    verbose_only=False, verbose=verbose,
                )
                _ = _build_feature_frame(
                    batch_df, active_specs, stage=f"train_growth_{iteration}_{growth_round}", **_bff_kwargs,
                )
                cumulative_train_ids.update(batch_ids)

            # Get fully labeled pool (intersection of caches for all active metrics)
            labeled_ids = (set.intersection(
                *[label_cache.available_ids(spec.metric_id) for spec in active_specs]
            ) & all_train_ids) if active_specs else set()

            if len(labeled_ids) < 2:
                # Need more data to even try matching
                fresh_ids = list(all_train_ids - cumulative_train_ids)
                if not fresh_ids:
                    break
                random.shuffle(fresh_ids)
                batch_ids = fresh_ids[:label_batch_size]
                batch_df = train_df[train_df[id_column].astype(str).isin(batch_ids)]
                _ = _build_feature_frame(
                    batch_df, active_specs, stage=f"train_growth_{iteration}_{growth_round}", **_bff_kwargs,
                )
                cumulative_train_ids.update(batch_ids)
                continue

            # Build candidate frame and try exact-only matching
            candidate_df = train_df[train_df[id_column].astype(str).isin(labeled_ids)]
            candidate_frame_tmp = _build_feature_frame(
                candidate_df, active_specs, stage=f"train_growth_check_{iteration}_{growth_round}", **_bff_kwargs,
            )
            exact_pairs = exact_match(
                candidate_frame_tmp, id_column, label_column, feature_columns,
                1, k_pairs, seen_pairs, exact_only=True,
            )
            current_exact_count = len(exact_pairs)
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} pool growth round {growth_round}: "
                f"{current_exact_count} exact matches from {len(labeled_ids)} labeled samples",
                verbose_only=False, verbose=verbose,
            )

            if current_exact_count >= k_pairs:
                _iter_log(
                    f"[Autometrics][Iterative] Iteration {iteration} pool growth complete: "
                    f"found {current_exact_count} >= {k_pairs} exact matches",
                    verbose_only=False, verbose=verbose,
                )
                break

            if growth_round > 0 and current_exact_count <= prev_exact_count:
                _iter_log(
                    f"[Autometrics][Iterative] Iteration {iteration} pool growth exhausted: "
                    f"no new exact matches from last batch ({current_exact_count} <= {prev_exact_count})",
                    verbose_only=False, verbose=verbose,
                )
                break

            prev_exact_count = current_exact_count

            # Label a fresh batch to grow the pool
            fresh_ids = list(all_train_ids - cumulative_train_ids)
            if not fresh_ids:
                _iter_log(
                    f"[Autometrics][Iterative] Iteration {iteration} pool growth complete: no more unlabeled data",
                    verbose_only=False, verbose=verbose,
                )
                break

            random.shuffle(fresh_ids)
            batch_ids = fresh_ids[:label_batch_size]
            batch_df = train_df[train_df[id_column].astype(str).isin(batch_ids)]
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} labeling {len(batch_df)} fresh samples "
                f"(growth round {growth_round})",
                verbose_only=False, verbose=verbose,
            )
            _ = _build_feature_frame(
                batch_df, active_specs, stage=f"train_growth_{iteration}_{growth_round}", **_bff_kwargs,
            )
            cumulative_train_ids.update(batch_ids)
        else:
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} pool growth hit safety limit ({max_growth_rounds} rounds)",
                verbose_only=False, verbose=verbose,
            )

        # ── Final candidate frame from full labeled pool ──
        labeled_ids = (set.intersection(
            *[label_cache.available_ids(spec.metric_id) for spec in active_specs]
        ) & all_train_ids) if active_specs else set()

        if not labeled_ids:
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} stopping: no labeled data for matching.",
                verbose_only=False, verbose=verbose,
            )
            break

        _iter_log(
            f"[Autometrics][Iterative] Iteration {iteration} cumulative training pool: "
            f"{len(cumulative_train_ids)} total, {len(labeled_ids)} fully labeled for {len(active_specs)} metrics",
            verbose_only=False, verbose=verbose,
        )

        candidate_df = train_df[train_df[id_column].astype(str).isin(labeled_ids)]
        candidate_frame = _build_feature_frame(
            candidate_df, active_specs, stage=f"train_candidate_{iteration}", **_bff_kwargs,
        )
        train_probs = compute_probabilities(model, candidate_frame, active_specs, name="train")
        candidate_frame = candidate_frame.copy()
        candidate_frame["prob"] = train_probs
        train_assessed = _compute_train_assessment(candidate_frame, label_column, train_probs)
        train_assessed["cumulative_pool_size"] = len(cumulative_train_ids)

        # ── Pair selection: exact (full) -> propensity -> residual ──
        pairs = exact_match(
            candidate_frame, id_column, label_column, feature_columns, 1, k_pairs, seen_pairs
        )
        _iter_log(
            f"[Autometrics][Iterative] Iteration {iteration} exact-match pairs: {len(pairs)} (incl. near-exact)",
            verbose_only=False, verbose=verbose,
        )
        used_method = "exact"

        if len(pairs) < k_pairs:
            # Update seen_pairs with exact matches before Mahalanobis
            mahal_seen = set(seen_pairs)
            for pid, nid, _ in pairs:
                mahal_seen.add(normalize_pair_id(pid, nid))
            mahal_pairs = mahalanobis_match(
                candidate_frame, id_column, label_column, feature_columns, 1,
                k_pairs - len(pairs), mahal_seen,
                prob_column="prob", propensity_caliper=caliper,
            )
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} mahalanobis pairs: {len(mahal_pairs)} "
                f"(supplementing {len(pairs)} exact)",
                verbose_only=False, verbose=verbose,
            )
            if mahal_pairs:
                used_method = "exact+mahalanobis" if pairs else "mahalanobis"
            pairs.extend(mahal_pairs)

        if len(pairs) < k_pairs:
            # Update seen_pairs with matches so far before propensity
            prop_seen = set(seen_pairs)
            for pid, nid, _ in pairs:
                prop_seen.add(normalize_pair_id(pid, nid))
            propensity_pairs = propensity_match(
                candidate_frame, id_column, label_column, "prob", 1, k_pairs - len(pairs), caliper, prop_seen
            )
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} propensity pairs: {len(propensity_pairs)} "
                f"(supplementing {len(pairs)} exact+mahalanobis)",
                verbose_only=False, verbose=verbose,
            )
            if propensity_pairs:
                used_method = used_method + "+propensity" if "exact" in used_method or "mahalanobis" in used_method else "propensity"
            pairs.extend(propensity_pairs)

        hard_pos: List[str] = []
        hard_neg: List[str] = []
        if residual_only or (not matching_only and len(pairs) < k_pairs):
            hard_pos, hard_neg = residual_select(
                candidate_frame, id_column, label_column, "prob", 1, k_pairs
            )
            used_method = "residual"
        elif matching_only and len(pairs) < k_pairs:
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} stopping: insufficient matched pairs under caliper.",
                verbose_only=False,
                verbose=verbose,
            )
            break

        is_pair_method = used_method != "residual"
        mismatch_points = 0
        if is_pair_method:
            mismatch_points = len(set([p[0] for p in pairs] + [p[1] for p in pairs]))
        else:
            mismatch_points = len(set(hard_pos + hard_neg))
        mismatch_stats = {
            "method": used_method,
            "pairs": len(pairs),
            "points": mismatch_points,
        }

        if is_pair_method and len(pairs) < k_pairs:
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} stopping: insufficient matched pairs.",
                verbose_only=False,
                verbose=verbose,
            )
            break
        if used_method == "residual" and (len(hard_pos) < k_pairs or len(hard_neg) < k_pairs):
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} stopping: insufficient hard examples for residual selection.",
                verbose_only=False,
                verbose=verbose,
            )
            break

        # Variable pair sampling + token-based text budget
        # Budget: ~8k tokens ≈ ~32k chars for contrastive text.
        # Each iteration, randomly vary how many pairs we show the proposer
        # (more pairs = less text each, fewer pairs = more text each).
        CHARS_PER_TOKEN = 4
        contrastive_token_budget = 8000
        contrastive_char_budget = contrastive_token_budget * CHARS_PER_TOKEN
        overhead_per_pair = 300  # score dicts, IDs, labels, formatting

        available_pairs = pairs if is_pair_method else list(zip(hard_pos, hard_neg, [0.0] * min(len(hard_pos), len(hard_neg))))
        n_available = len(available_pairs) if is_pair_method else min(len(hard_pos), len(hard_neg), k_pairs)
        if n_available > 0:
            # Randomly sample how many pairs to show (between 2 and n_available)
            n_show = random.randint(min(2, n_available), n_available)
            per_text_chars = max(500, (contrastive_char_budget - n_show * overhead_per_pair) // max(1, n_show * 2))
        else:
            n_show = 0
            per_text_chars = 1500

        pair_records: List[str] = []
        contrastive_text = []
        if is_pair_method:
            # Sample which pairs to show (all go into pair_records, but only n_show get text)
            show_indices = set(random.sample(range(len(pairs)), min(n_show, len(pairs)))) if pairs else set()
            for pair_idx, (pos_id, neg_id, diff) in enumerate(pairs):
                pair_key = normalize_pair_id(pos_id, neg_id)
                seen_pairs.add(pair_key)
                pair_id = "|".join(pair_key)
                pair_records.append(pair_id)
                if pair_idx not in show_indices:
                    continue  # registered but not shown to proposer
                pos_matches = train_df[train_df[id_column].astype(str) == str(pos_id)]
                neg_matches = train_df[train_df[id_column].astype(str) == str(neg_id)]
                if pos_matches.empty or neg_matches.empty:
                    _iter_log(
                        f"[Autometrics][Iterative] Warning: pair {pair_id} ID not found in train_df (pos_empty={pos_matches.empty}, neg_empty={neg_matches.empty})",
                        verbose_only=False, verbose=verbose,
                    )
                    continue
                pos_row = pos_matches.iloc[0]
                neg_row = neg_matches.iloc[0]
                pos_score_matches = candidate_frame[candidate_frame[id_column].astype(str) == str(pos_id)]
                neg_score_matches = candidate_frame[candidate_frame[id_column].astype(str) == str(neg_id)]
                if pos_score_matches.empty or neg_score_matches.empty:
                    _iter_log(
                        f"[Autometrics][Iterative] Warning: pair {pair_id} ID not found in candidate_frame (pos_empty={pos_score_matches.empty}, neg_empty={neg_score_matches.empty})",
                        verbose_only=False, verbose=verbose,
                    )
                    continue
                pos_scores = pos_score_matches[[m.name for m in active_specs]].iloc[0].to_dict()
                neg_scores = neg_score_matches[[m.name for m in active_specs]].iloc[0].to_dict()
                contrastive_text.append(
                    "PAIR\n"
                    f"POS id={pos_id} scores={pos_scores}\n{_truncate_text(str(pos_row[text_column]), max_chars=per_text_chars)}\n\n"
                    f"NEG id={neg_id} scores={neg_scores}\n{_truncate_text(str(neg_row[text_column]), max_chars=per_text_chars)}"
                )
        else:
            used_pairs = 0
            for pos_id in hard_pos:
                for neg_id in hard_neg:
                    pair_key = normalize_pair_id(pos_id, neg_id)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    pair_id = "|".join(pair_key)
                    pair_records.append(pair_id)
                    pos_matches = train_df[train_df[id_column].astype(str) == str(pos_id)]
                    neg_matches = train_df[train_df[id_column].astype(str) == str(neg_id)]
                    if pos_matches.empty or neg_matches.empty:
                        _iter_log(
                            f"[Autometrics][Iterative] Warning: residual pair {pair_id} ID not found in train_df",
                            verbose_only=False, verbose=verbose,
                        )
                        used_pairs += 1
                        break
                    pos_row = pos_matches.iloc[0]
                    neg_row = neg_matches.iloc[0]
                    pos_score_matches = candidate_frame[candidate_frame[id_column].astype(str) == str(pos_id)]
                    neg_score_matches = candidate_frame[candidate_frame[id_column].astype(str) == str(neg_id)]
                    if pos_score_matches.empty or neg_score_matches.empty:
                        _iter_log(
                            f"[Autometrics][Iterative] Warning: residual pair {pair_id} ID not found in candidate_frame",
                            verbose_only=False, verbose=verbose,
                        )
                        used_pairs += 1
                        break
                    pos_scores = pos_score_matches[[m.name for m in active_specs]].iloc[0].to_dict()
                    neg_scores = neg_score_matches[[m.name for m in active_specs]].iloc[0].to_dict()
                    contrastive_text.append(
                        "HARD EXAMPLES\n"
                        f"POS id={pos_id} scores={pos_scores}\n{_truncate_text(str(pos_row[text_column]), max_chars=per_text_chars)}\n\n"
                        f"NEG id={neg_id} scores={neg_scores}\n{_truncate_text(str(neg_row[text_column]), max_chars=per_text_chars)}"
                    )
                    used_pairs += 1
                    break
                if used_pairs >= k_pairs:
                    break

        if used_method == "residual" and len(pair_records) < k_pairs:
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} stopping: insufficient unique residual pairs.",
                verbose_only=False,
                verbose=verbose,
            )
            break

        _iter_log(
            f"[Autometrics][Iterative] Iteration {iteration} contrastive context: "
            f"showing {len(contrastive_text)} of {len(pair_records)} pairs, "
            f"{per_text_chars} chars/text (~{per_text_chars // CHARS_PER_TOKEN} tokens/text)",
            verbose_only=False, verbose=verbose,
        )

        # ── Build enriched context for the proposer ──

        # Condensed metric summaries (1-2 sentences each, not full rubrics)
        active_metric_summaries = [_condense_metric_description(s) for s in active_specs]

        # Metric performance statistics from scored training data
        stats_section = ""
        if candidate_frame is not None and len(active_specs) > 0:
            stats_coef_map = {}
            if model is not None and hasattr(model, 'model') and hasattr(model.model, 'coef_'):
                stats_coef_map = {
                    spec.metric_id: float(coef)
                    for spec, coef in zip(active_specs, model.model.coef_.reshape(-1)[:len(active_specs)])
                }
            stats_section = _compute_metric_stats(
                candidate_frame, active_specs, label_column, coef_map=stats_coef_map,
            )

        # Build misclassified examples text for reasoning (up to 5 examples)
        misclassified_text = ""
        if candidate_frame is not None and "prob" in candidate_frame.columns:
            preds = (candidate_frame["prob"].values >= 0.5).astype(int)
            actuals = candidate_frame[label_column].values.astype(int)
            wrong_mask = preds != actuals
            wrong_ids = candidate_frame.loc[wrong_mask, id_column].astype(str)
            if len(wrong_ids) > 0:
                sample_ids = wrong_ids.sample(n=min(5, len(wrong_ids)), random_state=seed).tolist()
                # Look up text from train_df (candidate_frame doesn't have text_column)
                text_lookup = train_df.set_index(train_df[id_column].astype(str))[text_column]
                wrong_lines = []
                for sid in sample_ids:
                    row = candidate_frame[candidate_frame[id_column].astype(str) == sid].iloc[0]
                    pred_label = int(row["prob"] >= 0.5)
                    txt = str(text_lookup.get(sid, ""))
                    wrong_lines.append(
                        f"[id={sid} true_label={int(row[label_column])} "
                        f"predicted={pred_label}]\n{_truncate_text(txt, max_chars=500)}"
                    )
                misclassified_text = "\n\n".join(wrong_lines)

        # ── Iteration reasoning chain ──
        # Step 1: Generate per-iteration reasoning (before proposing metrics)
        prev_metrics_proposed = None
        if iteration_reasonings:
            prev_metrics_proposed = iteration_reasonings[-1][2]

        iteration_reasoning = _generate_iteration_reasoning(
            scoring_backend=scoring_backend,
            generator_llm=generator_llm,
            task_description=task_description,
            iteration=iteration,
            contrastive_text=contrastive_text,
            active_metric_summaries=active_metric_summaries,
            prev_reasoning=prev_iteration_reasoning,
            prev_metrics_proposed=prev_metrics_proposed,
            misclassified_examples=misclassified_text if iteration > 1 else None,
        )
        if iteration_reasoning:
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} reasoning: {iteration_reasoning[:200]}...",
                verbose_only=False, verbose=verbose,
            )

        # Step 2: Generate trajectory summary (aggregates all prior reasonings)
        trajectory_section = ""
        if iteration > 1 and iteration_reasonings:
            trajectory_summary = _generate_trajectory_summary(
                scoring_backend, generator_llm, iteration_reasonings,
            )
            if trajectory_summary:
                trajectory_section = (
                    "\n\n--- TRAJECTORY SUMMARY (what we've tried so far) ---\n"
                    f"{trajectory_summary}\n"
                    "--- END TRAJECTORY SUMMARY ---\n"
                )
                _iter_log(
                    f"[Autometrics][Iterative] Iteration {iteration} trajectory: {trajectory_summary[:200]}...",
                    verbose_only=False, verbose=verbose,
                )

        # Dropped metrics history (so proposer avoids similar themes)
        dropped_section = ""
        if dropped_metrics_log:
            recent_dropped = dropped_metrics_log[-20:]
            dropped_section = (
                "\n\n--- PREVIOUSLY DROPPED METRICS (do NOT re-propose similar themes) ---\n"
                + "\n".join(recent_dropped)
            )

        # Build interaction section for the proposer
        interaction_section = ""
        if active_interactions:
            interaction_lines = []
            for col_name, (m_a, m_b, coef, desc) in active_interactions.items():
                interaction_lines.append(
                    f"- {m_a} × {m_b} (coef={coef:.3f}): {desc}"
                )
            interaction_section = (
                "\n\n--- ACTIVE INTERACTION TERMS (combinations found to be meaningful) ---\n"
                "The following pairwise metric interactions were found to have significant "
                "predictive power. Avoid proposing metrics that duplicate what these "
                "interactions already capture, either individually or in combination.\n"
                + "\n".join(interaction_lines)
            )

        # Assemble current_metrics_text with condensed descriptions (not full rubrics)
        metrics_list_text = "\n".join(active_metric_summaries) if active_metric_summaries else ""
        current_metrics_text = trajectory_section + metrics_list_text
        current_metrics_text += stats_section + interaction_section + dropped_section

        # Build iteration-aware task description with reasoning context
        iter_task_description = task_description
        if iteration_reasoning:
            reasoning_section = (
                f"\n\n--- REASONING FOR THIS ITERATION (iteration {iteration}) ---\n"
                f"{iteration_reasoning}\n"
                "--- END REASONING ---\n\n"
                "Use the above reasoning to guide your metric proposals. Focus on the "
                "specific patterns and dimensions identified."
            )
            iter_task_description += reasoning_section
        if iteration >= 2:
            iter_task_description += (
                f"\n\nThis is iteration {iteration}. Focus on genuinely NEW dimensions "
                "identified in the reasoning above that current metrics do NOT capture."
            )

        new_specs: List[MetricSpec] = []
        dropped_empty = 0
        dropped_dupe = 0
        dropped_llm = 0
        dropped_critique = 0
        attempt = 0
        while attempt < max_candidate_attempts and len(new_specs) < min_unique:
            candidate_defs = proposer.propose(
                task_description=iter_task_description,
                positive_examples="N/A — see contrastive_pairs for labeled examples.",
                negative_examples="N/A — see contrastive_pairs for labeled examples.",
                current_metrics=current_metrics_text or "None",
                contrastive_pairs="\n\n".join(contrastive_text),
                num_metrics=num_metrics,
                num_rubrics=num_rubrics,
            )
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} generated {len(candidate_defs)} new candidates "
                f"(requested metrics={num_metrics}, rubrics={num_rubrics}, attempt {attempt+1}/{max_candidate_attempts})",
                verbose_only=True,
                verbose=verbose,
            )

            # Self-critique: filter superficial metrics before dedup
            if candidate_defs:
                critique_results = _self_critique_candidates(
                    candidate_defs, task_description, generator_llm, scoring_backend,
                )
                filtered_defs = []
                for cand, verdict in critique_results:
                    if "superficial" in verdict.lower():
                        dropped_critique += 1
                        cand_name = cand.get("name", "Metric")
                        cand_rubric_raw = cand.get("rubric", {})
                        cand_rubric_preview = _truncate_text(
                            _rubric_to_text(_normalize_rubric(cand_rubric_raw, str(cand.get("scale", "ordinal")))),
                            max_chars=200,
                        ) if cand_rubric_raw else ""
                        dropped_metrics_log.append(
                            f"DROPPED (self-critique: superficial, iter {iteration}): "
                            f"{cand_name} — {verdict.strip()}"
                            + (f"\n  Rubric: {cand_rubric_preview}" if cand_rubric_preview else "")
                        )
                        _iter_log(
                            f"[Autometrics][Iterative] Self-critique filtered: {cand_name} -- {verdict}",
                            verbose_only=False, verbose=verbose,
                        )
                    else:
                        filtered_defs.append(cand)
                candidate_defs = filtered_defs

            for cand in candidate_defs:
                name = cand.get("name") or "Metric"
                scale = str(cand.get("scale") or "ordinal").lower()
                rubric = _normalize_rubric(cand.get("rubric"), scale)
                rubric_text = _rubric_to_text(rubric)
                if not rubric_text:
                    dropped_empty += 1
                    dropped_metrics_log.append(f"DROPPED (empty rubric, iter {iteration}): {name}")
                    if verbose:
                        _iter_log(
                            f"[Autometrics][Iterative] Dropped candidate (empty rubric): {name}",
                            verbose_only=False,
                            verbose=verbose,
                        )
                    continue
                metric_id = _metric_id_from_rubric(rubric_text)
                if metric_id in metric_bank:
                    dropped_dupe += 1
                    rubric_preview = _truncate_text(rubric_text, max_chars=200)
                    existing_name = metric_bank[metric_id].name
                    dropped_metrics_log.append(
                        f"DROPPED (duplicate rubric of '{existing_name}', iter {iteration}): {name}"
                        f"\n  Rubric: {rubric_preview}"
                    )
                    if verbose:
                        _iter_log(
                            f"[Autometrics][Iterative] Dropped candidate (duplicate rubric): {name}",
                            verbose_only=False,
                            verbose=verbose,
                        )
                    continue
                spec = _build_metric_from_parts(
                    name=name,
                    rubric=rubric,
                    rubric_text=rubric_text,
                    judge_llm=judge_llm,
                    task_description=task_description,
                    has_references=has_references,
                    existing_names=existing_names,
                    metric_id=metric_id,
                    scoring_backend=scoring_backend,
                )
                if not _llm_dedup(generator_llm, spec, active_specs, scoring_backend=scoring_backend):
                    dropped_llm += 1
                    rubric_preview = _truncate_text(rubric_text, max_chars=200)
                    dropped_metrics_log.append(
                        f"DROPPED (LLM dedup, iter {iteration}): {name}"
                        f"\n  Rubric: {rubric_preview}"
                    )
                    if verbose:
                        _iter_log(
                            f"[Autometrics][Iterative] Dropped candidate (LLM dedup): {name}",
                            verbose_only=False,
                            verbose=verbose,
                        )
                    continue
                metric_bank[spec.metric_id] = spec
                new_specs.append(spec)
                lifecycle.register_metric(spec.metric_id, spec.name, spec.rubric_text, iteration, source_pairs=pair_records)
            attempt += 1

        # Record iteration reasoning for the chain
        new_metric_names = [s.name for s in new_specs]
        if iteration_reasoning:
            iteration_reasonings.append((iteration, iteration_reasoning, new_metric_names))
            prev_iteration_reasoning = iteration_reasoning

        if not new_specs:
            no_new_metrics_iters += 1
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} no new metrics survived "
                f"(dropped_empty={dropped_empty}, dropped_duplicate={dropped_dupe}, "
                f"dropped_llm={dropped_llm}, dropped_critique={dropped_critique}, counter={no_new_metrics_iters})",
                verbose_only=False,
                verbose=verbose,
            )

            eval_sel_frame_active = _build_feature_frame(
                iter_eval_sel_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
                verbose=verbose, stage=f"eval_selection_active_{iteration}",
                score_all_metrics_together=score_all_metrics_together,
                judge_llm=judge_llm,
                task_description=task_description,
                use_tqdm=tqdm_scoring,
                llm_parallelism=llm_parallelism,
                scoring_backend=scoring_backend,
                )
            eval_gate_frame_active = _build_feature_frame(
                eval_gate_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
                verbose=verbose, stage=f"eval_gating_active_{iteration}",
                score_all_metrics_together=score_all_metrics_together,
                judge_llm=judge_llm,
                task_description=task_description,
                use_tqdm=tqdm_scoring,
                llm_parallelism=llm_parallelism,
                scoring_backend=scoring_backend,
                )
            eval_sel_probs = compute_probabilities(model, eval_sel_frame_active, active_specs, name=f"eval_selection_{iteration}")
            eval_gate_probs = compute_probabilities(model, eval_gate_frame_active, active_specs, name=f"eval_gating_{iteration}")
            eval_sel_metrics = _compute_metrics(eval_sel_frame_active[label_column].values, eval_sel_probs)
            eval_gate_metrics = _compute_metrics(eval_gate_frame_active[label_column].values, eval_gate_probs)
            gate_key, gate_value = _gate_metric(eval_gate_metrics)

            iter_test_df = _get_test_df(iteration, final_iteration)
            test_frame = _build_feature_frame(
                iter_test_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
                verbose=verbose, stage=f"test_{iteration}",
                score_all_metrics_together=score_all_metrics_together,
                judge_llm=judge_llm,
                task_description=task_description,
                use_tqdm=tqdm_scoring,
                llm_parallelism=llm_parallelism,
                scoring_backend=scoring_backend,
                )
            test_probs = compute_probabilities(model, test_frame, active_specs, name=f"test_{iteration}")
            test_metrics = _compute_metrics(test_frame[label_column].values, test_probs)

            active_coef_map = {
                spec.metric_id: float(coef)
                for spec, coef in zip(active_specs, model.model.coef_.reshape(-1))
            }

            _log_iteration_summary(
                iteration=iteration,
                phase="no_new_metrics",
                candidates=active_specs,
                active_ids=active_metric_ids,
                active_names=[metric_bank[mid].name for mid in active_metric_ids],
                coef_map=active_coef_map,
                gate_metric=gate_key,
                gate_value=gate_value,
                eval_selection=eval_sel_metrics,
                eval_gating=eval_gate_metrics,
                test_metrics=test_metrics,
                train_assessed=train_assessed,
                mismatch_stats=mismatch_stats,
                eval_sizes={
                    "eval_selection": len(eval_sel_frame_active),
                    "eval_gating": len(eval_gate_frame_active),
                    "test": len(test_frame),
                },
                verbose=verbose,
            )

            if prev_gate_score is not None and gate_value - prev_gate_score < eval_plateau_eps:
                eval_plateau_iters += 1
            else:
                eval_plateau_iters = 0
            last_eval_gate_score = gate_value
            lifecycle.record_coefficients(iteration, active_coef_map)
            lifecycle.mark_active(iteration, active_metric_ids)
            lifecycle.record_marginal_contributions(
                iteration,
                _compute_marginal_contributions(
                    model,
                    build_dataset_from_frame(eval_gate_frame_active, active_specs, f"eval_gating_{iteration}"),
                    active_specs,
                ),
            )
            _log_coefficients(coef_records, iteration, active_specs, active_coef_map)

            churn = 0.0
            if prev_active_ids:
                churn = 1.0 - (len(set(prev_active_ids) & set(active_metric_ids)) / float(len(prev_active_ids)))
            churn_warning = churn > churn_warning_threshold

            _jsonl_append(output_path / "iterations.jsonl", {
                "iteration": iteration,
                "active_metric_ids": active_metric_ids,
                "active_metric_names": [metric_bank[mid].name for mid in active_metric_ids],
                "gate_metric": gate_key,
                "eval_gate_score": gate_value,
                "eval_selection": eval_sel_metrics,
                "eval_gating": eval_gate_metrics,
                "test": test_metrics,
                "accepted": False,
                "num_active_metrics": len(active_metric_ids),
                "num_active_interactions": len(active_interactions),
                "active_interactions": {k: {"a": v[0], "b": v[1], "coef": v[2], "desc": v[3]} for k, v in active_interactions.items()},
                "churn": churn,
                "churn_warning": churn_warning,
                "method": used_method,
                "note": "no_new_metrics",
                "train_assessed": train_assessed,
                "mismatch_stats": mismatch_stats,
                "iteration_reasoning": iteration_reasoning or "",
                "data_sizes": {
                    "eval_selection": len(eval_sel_frame_active),
                    "eval_gating": len(eval_gate_frame_active),
                    "test": len(test_frame),
                },
            })
            _jsonl_append(output_path / "eval_metrics.jsonl", {"iteration": iteration, **eval_gate_metrics})
            _jsonl_append(output_path / "test_metrics.jsonl", {"iteration": iteration, **test_metrics})

            for pair_id in pair_records:
                _jsonl_append(output_path / "pairs.jsonl", {"iteration": iteration, "pair_id": pair_id, "method": used_method})

            if not disable_early_stopping and (no_new_metrics_iters >= early_stop_patience or eval_plateau_iters >= early_stop_patience):
                _iter_log(
                    f"[Autometrics][Iterative] Iteration {iteration} stopping: no-new-metrics ({no_new_metrics_iters}) or eval plateau ({eval_plateau_iters}) hit patience={early_stop_patience}.",
                    verbose_only=False,
                    verbose=verbose,
                )
                break
            continue

        joint_specs = active_specs + new_specs
        eval_sel_frame_joint = _build_feature_frame(
            iter_eval_sel_df, joint_specs, label_cache, id_column, text_column, label_column, label_batch_size,
            verbose=verbose, stage=f"eval_selection_joint_{iteration}",
            score_all_metrics_together=score_all_metrics_together,
            judge_llm=judge_llm,
            task_description=task_description,
            use_tqdm=tqdm_scoring,
            llm_parallelism=llm_parallelism,
            scoring_backend=scoring_backend,
            )
        eval_gate_frame_joint = _build_feature_frame(
            eval_gate_df, joint_specs, label_cache, id_column, text_column, label_column, label_batch_size,
            verbose=verbose, stage=f"eval_gating_joint_{iteration}",
            score_all_metrics_together=score_all_metrics_together,
            judge_llm=judge_llm,
            task_description=task_description,
            use_tqdm=tqdm_scoring,
            llm_parallelism=llm_parallelism,
            scoring_backend=scoring_backend,
            )

        # Only generate interactions among previously-active metrics, not new
        # candidates.  New metrics must prove themselves via base coefficient
        # before getting interaction terms.  This prevents quadratic feature
        # explosion when many candidates are tested at once.
        # Cap to top-K by absolute coefficient to keep interactions O(K^2).
        prev_active_names = [m.name for m in active_specs]
        if max_interaction_metrics and len(prev_active_names) > max_interaction_metrics:
            name_to_coef = {m.name: abs(active_coef_map.get(m.metric_id, 0.0)) for m in active_specs}
            prev_active_names = sorted(prev_active_names, key=lambda n: name_to_coef.get(n, 0.0), reverse=True)[:max_interaction_metrics]
        joint_model = fit_regression(
            eval_sel_frame_joint, joint_specs, name=f"eval_selection_{iteration}",
            interaction_names=prev_active_names,
        )
        joint_coef_map, joint_interaction_map = _extract_coefs_and_interactions(
            joint_model, joint_specs, iteration=iteration,
        )

        eval_gate_probs = compute_probabilities(joint_model, eval_gate_frame_joint, joint_specs, name=f"eval_gating_{iteration}")
        eval_gate_metrics = _compute_metrics(eval_gate_frame_joint[label_column].values, eval_gate_probs)
        gate_key, gate_value = _gate_metric(eval_gate_metrics)

        # Check if new metrics survived: non-zero direct coef OR participate in non-zero interaction
        new_metric_names = {spec.name for spec in new_specs}
        new_survived = any(abs(joint_coef_map.get(spec.metric_id, 0.0)) > 1e-6 for spec in new_specs)
        if not new_survived and joint_interaction_map:
            for _col, (m_a, m_b, c) in joint_interaction_map.items():
                if abs(c) > 1e-6 and (m_a in new_metric_names or m_b in new_metric_names):
                    new_survived = True
                    break
        accept = new_survived or (prev_gate_score is not None and gate_value > prev_gate_score)
        _iter_log(
            f"[Autometrics][Iterative] Iteration {iteration} accept={accept} new_survived={new_survived} gate={gate_value:.4f}",
            verbose_only=False,
            verbose=verbose,
        )

        if new_survived:
            no_new_metrics_iters = 0
        else:
            no_new_metrics_iters += 1

        if accept:
            active_metric_ids = _select_active_metric_ids(
                joint_specs, joint_coef_map, joint_interaction_map,
            )
            if joint_interaction_map:
                _update_active_interactions(joint_interaction_map, joint_specs)
            for spec in joint_specs:
                if spec.metric_id not in active_metric_ids:
                    dropped_metrics_log.append(
                        f"DROPPED (zero coefficient, iter {iteration}): {spec.name} - {spec.rubric_text[:100]}"
                    )
        else:
            active_metric_ids = prev_active_ids

        if min_tenure > 0 and prev_active_ids:
            kept = list(active_metric_ids)
            for mid in prev_active_ids:
                if mid in kept:
                    continue
                lifecycle_entry = lifecycle.get(mid)
                if lifecycle_entry and len(lifecycle_entry.active_iterations) < min_tenure:
                    kept.append(mid)
            active_metric_ids = kept

        active_specs = current_metric_specs()
        eval_sel_frame_active = _build_feature_frame(
            iter_eval_sel_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
            verbose=verbose, stage=f"eval_selection_active_{iteration}",
            score_all_metrics_together=score_all_metrics_together,
            judge_llm=judge_llm,
            task_description=task_description,
            use_tqdm=tqdm_scoring,
            llm_parallelism=llm_parallelism,
            scoring_backend=scoring_backend,
            )
        eval_gate_frame_active = _build_feature_frame(
            eval_gate_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
            verbose=verbose, stage=f"eval_gating_active_{iteration}",
            score_all_metrics_together=score_all_metrics_together,
            judge_llm=judge_llm,
            task_description=task_description,
            use_tqdm=tqdm_scoring,
            llm_parallelism=llm_parallelism,
            scoring_backend=scoring_backend,
            )
        refit_interaction_names = [m.name for m in active_specs]
        if max_interaction_metrics and len(refit_interaction_names) > max_interaction_metrics:
            name_to_coef = {m.name: abs(joint_coef_map.get(m.metric_id, 0.0)) for m in active_specs}
            refit_interaction_names = sorted(refit_interaction_names, key=lambda n: name_to_coef.get(n, 0.0), reverse=True)[:max_interaction_metrics]
        model = fit_regression(eval_sel_frame_active, active_specs, name=f"eval_selection_active_{iteration}",
                               interaction_names=refit_interaction_names)

        eval_sel_probs = compute_probabilities(model, eval_sel_frame_active, active_specs, name=f"eval_selection_active_{iteration}")
        eval_gate_probs_active = compute_probabilities(model, eval_gate_frame_active, active_specs, name=f"eval_gating_active_{iteration}")
        eval_sel_metrics = _compute_metrics(eval_sel_frame_active[label_column].values, eval_sel_probs)
        eval_gate_metrics_active = _compute_metrics(eval_gate_frame_active[label_column].values, eval_gate_probs_active)
        gate_key_active, gate_value_active = _gate_metric(eval_gate_metrics_active)

        if prev_gate_score is not None and gate_value_active - prev_gate_score < eval_plateau_eps:
            eval_plateau_iters += 1
        else:
            eval_plateau_iters = 0
        last_eval_gate_score = gate_value_active

        lifecycle.record_coefficients(iteration, joint_coef_map)
        lifecycle.mark_active(iteration, active_metric_ids)

        iter_test_df = _get_test_df(iteration, final_iteration)
        test_frame = _build_feature_frame(
            iter_test_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
            verbose=verbose, stage=f"test_{iteration}",
            score_all_metrics_together=score_all_metrics_together,
            judge_llm=judge_llm,
            task_description=task_description,
            use_tqdm=tqdm_scoring,
            llm_parallelism=llm_parallelism,
            scoring_backend=scoring_backend,
            )
        test_probs = compute_probabilities(model, test_frame, active_specs, name=f"test_{iteration}")
        test_metrics = _compute_metrics(test_frame[label_column].values, test_probs)

        _log_iteration_summary(
            iteration=iteration,
            phase="iter",
            candidates=joint_specs,
            active_ids=active_metric_ids,
            active_names=[metric_bank[mid].name for mid in active_metric_ids],
            coef_map=joint_coef_map,
            gate_metric=gate_key_active,
            gate_value=gate_value_active,
            eval_selection=eval_sel_metrics,
            eval_gating=eval_gate_metrics_active,
            test_metrics=test_metrics,
            train_assessed=train_assessed,
            mismatch_stats=mismatch_stats,
            eval_sizes={
                "eval_selection": len(eval_sel_frame_active),
                "eval_gating": len(eval_gate_frame_active),
                "test": len(test_frame),
            },
            verbose=verbose,
        )

        churn = 0.0
        if prev_active_ids:
            churn = 1.0 - (len(set(prev_active_ids) & set(active_metric_ids)) / float(len(prev_active_ids)))
        churn_warning = churn > churn_warning_threshold

        lifecycle.record_marginal_contributions(
            iteration,
            _compute_marginal_contributions(
                model,
                build_dataset_from_frame(eval_gate_frame_active, active_specs, f"eval_gating_active_{iteration}"),
                active_specs,
            ),
        )
        _log_coefficients(coef_records, iteration, joint_specs, joint_coef_map)
        active_coef_map = joint_coef_map

        _jsonl_append(output_path / "iterations.jsonl", {
            "iteration": iteration,
            "active_metric_ids": active_metric_ids,
            "active_metric_names": [metric_bank[mid].name for mid in active_metric_ids],
            "gate_metric": gate_key_active,
            "eval_gate_score": gate_value_active,
            "eval_selection": eval_sel_metrics,
            "eval_gating": eval_gate_metrics_active,
            "selection_eval_gating": eval_gate_metrics,
            "test": test_metrics,
            "accepted": accept,
            "num_active_metrics": len(active_metric_ids),
            "num_active_interactions": len(active_interactions),
            "active_interactions": {k: {"a": v[0], "b": v[1], "coef": v[2], "desc": v[3]} for k, v in active_interactions.items()},
            "churn": churn,
            "churn_warning": churn_warning,
            "method": used_method,
            "train_assessed": train_assessed,
            "mismatch_stats": mismatch_stats,
            "iteration_reasoning": iteration_reasoning or "",
            "data_sizes": {
                "eval_selection": len(eval_sel_frame_active),
                "eval_gating": len(eval_gate_frame_active),
                "test": len(test_frame),
            },
        })
        _jsonl_append(output_path / "eval_metrics.jsonl", {"iteration": iteration, **eval_gate_metrics_active})
        _jsonl_append(output_path / "test_metrics.jsonl", {"iteration": iteration, **test_metrics})

        for pair_id in pair_records:
            _jsonl_append(output_path / "pairs.jsonl", {"iteration": iteration, "pair_id": pair_id, "method": used_method})

        if not disable_early_stopping and (no_new_metrics_iters >= early_stop_patience or eval_plateau_iters >= early_stop_patience):
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} stopping: no-new-metrics ({no_new_metrics_iters}) or eval plateau ({eval_plateau_iters}) hit patience={early_stop_patience}.",
                verbose_only=False,
                verbose=verbose,
            )
            break

    # Write lifecycle artifacts
    lifecycle.to_metrics_dataframe().to_csv(output_path / "metrics.csv", index=False)
    pd.DataFrame(coef_records).to_csv(output_path / "coefficients.csv", index=False)
    _iter_log(
        f"[Autometrics][Iterative] Wrote artifacts to {output_path}",
        verbose_only=False,
        verbose=verbose,
    )

    return {
        "top_metrics": [metric_bank[mid].metric for mid in active_metric_ids],
        "regression_metric": model,
        "active_metric_ids": active_metric_ids,
        "active_interactions": {
            k: {"metric_a": v[0], "metric_b": v[1], "coefficient": v[2], "description": v[3]}
            for k, v in active_interactions.items()
        } if active_interactions else {},
        "output_dir": str(output_path),
    }
