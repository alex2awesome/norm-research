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

from autometrics.aggregator.regression.LogisticL1 import LogisticL1
from autometrics.dataset.Dataset import Dataset
from autometrics.generator.ContrastiveRubricProposer import ContrastiveRubricProposer
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import (
    GeneratedRefBasedLLMJudgeMetric,
    GeneratedRefFreeLLMJudgeMetric,
)
from autometrics.util.splits import load_fixed_split

from .label_cache import LabelCache
from .lifecycle import MetricLifecycleTracker
from .matching import normalize_pair_id, propensity_match, residual_select

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
    try:
        _iter_logger.info(message)
    except Exception:
        pass
    try:
        print(message)
    except Exception:
        pass


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
            if scoring_backend is not None:
                metric_names = [m.name for m in metrics]
                for start in range(0, len(missing_df), batch_size):
                    chunk_df = missing_df.iloc[start : start + batch_size]
                    if verbose:
                        _iter_log(
                            f"[Autometrics][Iterative] Backend scoring chunk {start}:{start+len(chunk_df)} ({stage})",
                            verbose_only=False,
                            verbose=verbose,
                        )
                    chunk_ids = chunk_df[id_column].astype(str).tolist()
                    chunk_inputs = chunk_df[text_column].astype(str).tolist()
                    chunk_outputs = chunk_inputs  # same column used for both
                    responses = scoring_backend.score_multi_metric_batch(
                        task_description=task_description,
                        rubrics_text=rubrics_text,
                        metric_names=metric_names,
                        inputs=chunk_inputs,
                        outputs=chunk_outputs,
                    )
                    for row_id, resp in zip(chunk_ids, responses):
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
) -> List[str]:
    active = [spec.metric_id for spec in metric_specs if abs(coef_map.get(spec.metric_id, 0.0)) > 1e-6]
    if active:
        return active
    # Fallback: keep the strongest metric to avoid empty active set
    if not metric_specs:
        return []
    ranked = sorted(metric_specs, key=lambda s: abs(coef_map.get(s.metric_id, 0.0)), reverse=True)
    return [ranked[0].metric_id]


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

    coef = getattr(model.model, "coef_", None)
    if coef is None:
        return {}
    coef_vec = np.array(coef).reshape(-1)
    intercept = float(getattr(model.model, "intercept_", [0.0])[0])
    X = df[model.get_input_columns()].values
    if model.scaler is not None:
        X = model.scaler.transform(X)

    contribs: Dict[str, float] = {}
    for idx, spec in enumerate(metric_specs):
        if idx >= len(coef_vec):
            continue
        masked_coef = coef_vec.copy()
        masked_coef[idx] = 0.0
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
    eval_gate_fraction: float = 0.3,
    eval_plateau_eps: float = 0.005,
    churn_warning_threshold: float = 0.5,
    min_tenure: int = 0,
    score_all_metrics_together: bool = True,
    eval_sample_fraction: Optional[float] = None,
    eval_max_samples: Optional[int] = None,
    tqdm_scoring: bool = False,
    llm_parallelism: int = 1,
    scoring_backend: Optional[Any] = None,
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

    train_df, eval_df, test_df = load_fixed_split(
        data_path=data_path,
        split_dir=split_dir,
        create_if_missing=True,
        seed=seed,
        label_column=label_column,
    )
    _iter_log(
        f"[Autometrics][Iterative] Loaded splits: train={len(train_df)}, eval={len(eval_df)}, test={len(test_df)}",
        verbose_only=False,
        verbose=verbose,
    )

    id_column = _resolve_column(train_df, id_column, "press_release_id")
    text_column = _resolve_column(train_df, text_column, ["output", "text"])
    label_column = _resolve_column(train_df, label_column, ["human_score", "judgement"])

    train_df = _coerce_binary_labels(train_df, label_column)
    eval_df = _coerce_binary_labels(eval_df, label_column)
    test_df = _coerce_binary_labels(test_df, label_column)

    if eval_sample_fraction or eval_max_samples:
        n_eval = len(eval_df)
        target_n = n_eval
        if eval_sample_fraction:
            target_n = max(1, int(n_eval * eval_sample_fraction))
        if eval_max_samples:
            target_n = min(target_n, int(eval_max_samples))
        if target_n < n_eval:
            stratify_eval = eval_df[label_column] if eval_df[label_column].nunique() > 1 else None
            if stratify_eval is None:
                eval_df = eval_df.sample(n=target_n, random_state=seed, replace=False)
            else:
                eval_df, _ = train_test_split(
                    eval_df,
                    train_size=target_n,
                    random_state=seed,
                    stratify=stratify_eval,
                )
            _iter_log(
                f"[Autometrics][Iterative] Downsampled eval to {len(eval_df)} rows (fraction={eval_sample_fraction}, max={eval_max_samples})",
                verbose_only=False,
                verbose=verbose,
            )

    task_description = dataset.get_task_description()
    if not task_description:
        raise ValueError("dataset.task_description is required for iterative AutoMetrics.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    label_cache = LabelCache(cache_dir=str(output_path / "label_cache"))

    # Eval selection/gating split
    stratify_eval = eval_df[label_column] if eval_df[label_column].nunique() > 1 else None
    eval_sel_df, eval_gate_df = train_test_split(
        eval_df,
        test_size=eval_gate_fraction,
        random_state=seed,
        stratify=stratify_eval,
    )
    _iter_log(
        f"[Autometrics][Iterative] Eval split: selection={len(eval_sel_df)}, gating={len(eval_gate_df)}",
        verbose_only=False,
        verbose=verbose,
    )

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
        "eval_sample_fraction": eval_sample_fraction,
        "eval_max_samples": eval_max_samples,
        "tqdm_scoring": tqdm_scoring,
        "llm_parallelism": llm_parallelism,
        "seed": seed,
        "churn_warning_threshold": churn_warning_threshold,
        "min_tenure": min_tenure,
        "verbose": verbose,
    }
    (output_path / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    proposer = ContrastiveRubricProposer(generator_llm=generator_llm, seed=seed, scoring_backend=scoring_backend)
    lifecycle = MetricLifecycleTracker()
    seen_pairs: set[Tuple[str, str]] = set()

    metric_bank: Dict[str, MetricSpec] = {}
    active_metric_ids: List[str] = []
    coef_records: List[Dict[str, Any]] = []
    no_new_metrics_iters = 0
    eval_plateau_iters = 0
    last_eval_gate_score = None

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
    ) -> LogisticL1:
        ds = build_dataset_from_frame(df_frame, metric_specs, name=name)
        model = LogisticL1(dataset=ds, input_metrics=[m.metric for m in metric_specs])
        model.learn(ds, target_column=label_column)
        return model

    def compute_probabilities(model: LogisticL1, df_frame: pd.DataFrame, metric_specs: List[MetricSpec], name: str) -> np.ndarray:
        ds = build_dataset_from_frame(df_frame, metric_specs, name=name)
        return model.predict_proba(ds)

    def current_metric_specs() -> List[MetricSpec]:
        return [metric_bank[mid] for mid in active_metric_ids]

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

    eval_sel_frame = _build_feature_frame(
        eval_sel_df, candidate_specs, label_cache, id_column, text_column, label_column, label_batch_size,
        verbose=verbose, stage="eval_selection_full",
        score_all_metrics_together=score_all_metrics_together,
        judge_llm=judge_llm,
        task_description=task_description,
        use_tqdm=tqdm_scoring,
        llm_parallelism=llm_parallelism,
        scoring_backend=scoring_backend,
        )
    selection_model = fit_regression(eval_sel_frame, candidate_specs, name="eval_selection_full")
    selection_coef_map = {
        spec.metric_id: float(coef) for spec, coef in zip(candidate_specs, selection_model.model.coef_.reshape(-1))
    }
    lifecycle.record_coefficients(0, selection_coef_map)

    active_metric_ids = _select_active_metric_ids(candidate_specs, selection_coef_map)
    active_specs = current_metric_specs()
    _iter_log(
        f"[Autometrics][Iterative] Iteration 0 selected {len(active_metric_ids)} active metrics after regression",
        verbose_only=False,
        verbose=verbose,
    )

    eval_sel_frame_active = _build_feature_frame(
        eval_sel_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
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

    test_frame = _build_feature_frame(
        test_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
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
        "train_assessed": {"n": 0, "correct": 0, "accuracy": float("nan")},
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

        # Lazy label training data until enough matches/hard examples
        labeled_ids = None
        unlabeled_ids = set(train_df[id_column].astype(str).tolist())
        while True:
            if labeled_ids is None:
                labeled_ids = set.intersection(
                    *[label_cache.available_ids(spec.metric_id) for spec in active_specs]
                ) if active_specs else set()
            else:
                labeled_ids = set.intersection(
                    labeled_ids, *[label_cache.available_ids(spec.metric_id) for spec in active_specs]
                ) if active_specs else labeled_ids

            unlabeled_ids = set(train_df[id_column].astype(str).tolist()) - labeled_ids
            candidate_df = train_df[train_df[id_column].astype(str).isin(labeled_ids)]

            if len(candidate_df) >= 2 * k_pairs:
                break
            if not unlabeled_ids:
                break
            batch_ids = list(unlabeled_ids)[:label_batch_size]
            batch_df = train_df[train_df[id_column].astype(str).isin(batch_ids)]
            _ = _build_feature_frame(
                batch_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
                verbose=verbose, stage="train_label_batch",
                score_all_metrics_together=score_all_metrics_together,
                judge_llm=judge_llm,
                task_description=task_description,
                use_tqdm=tqdm_scoring,
                llm_parallelism=llm_parallelism,
                scoring_backend=scoring_backend,
                )
            labeled_ids.update(batch_ids)

        if not labeled_ids:
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} stopping: no labeled data for matching.",
                verbose_only=False,
                verbose=verbose,
            )
            break

        candidate_df = train_df[train_df[id_column].astype(str).isin(labeled_ids)]
        candidate_frame = _build_feature_frame(
            candidate_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
            verbose=verbose, stage="train_candidate",
            score_all_metrics_together=score_all_metrics_together,
            judge_llm=judge_llm,
            task_description=task_description,
            use_tqdm=tqdm_scoring,
            llm_parallelism=llm_parallelism,
            scoring_backend=scoring_backend,
            )
        train_probs = compute_probabilities(model, candidate_frame, active_specs, name="train")
        candidate_frame = candidate_frame.copy()
        candidate_frame["prob"] = train_probs
        train_assessed = _compute_train_assessment(candidate_frame, label_column, train_probs)

        pairs = propensity_match(
            candidate_frame, id_column, label_column, "prob", 1, k_pairs, caliper, seen_pairs
        )
        _iter_log(
            f"[Autometrics][Iterative] Iteration {iteration} propensity pairs found: {len(pairs)}",
            verbose_only=True,
            verbose=verbose,
        )

        used_method = "matching"
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

        mismatch_points = 0
        if used_method == "matching":
            mismatch_points = len(set([p[0] for p in pairs] + [p[1] for p in pairs]))
        else:
            mismatch_points = len(set(hard_pos + hard_neg))
        mismatch_stats = {
            "method": used_method,
            "pairs": len(pairs),
            "points": mismatch_points,
        }

        if used_method == "matching" and len(pairs) < k_pairs:
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} stopping: insufficient matched pairs under caliper.",
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

        pair_records: List[str] = []
        contrastive_text = []
        if used_method == "matching":
            for pos_id, neg_id, diff in pairs:
                pair_key = normalize_pair_id(pos_id, neg_id)
                seen_pairs.add(pair_key)
                pair_id = "|".join(pair_key)
                pair_records.append(pair_id)
                pos_row = train_df[train_df[id_column].astype(str) == pos_id].iloc[0]
                neg_row = train_df[train_df[id_column].astype(str) == neg_id].iloc[0]
                pos_scores = candidate_frame[candidate_frame[id_column].astype(str) == pos_id][[m.name for m in active_specs]].iloc[0].to_dict()
                neg_scores = candidate_frame[candidate_frame[id_column].astype(str) == neg_id][[m.name for m in active_specs]].iloc[0].to_dict()
                contrastive_text.append(
                    "PAIR\n"
                    f"POS id={pos_id} scores={pos_scores}\n{_truncate_text(str(pos_row[text_column]))}\n\n"
                    f"NEG id={neg_id} scores={neg_scores}\n{_truncate_text(str(neg_row[text_column]))}"
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
                    pos_row = train_df[train_df[id_column].astype(str) == pos_id].iloc[0]
                    neg_row = train_df[train_df[id_column].astype(str) == neg_id].iloc[0]
                    pos_scores = candidate_frame[candidate_frame[id_column].astype(str) == pos_id][[m.name for m in active_specs]].iloc[0].to_dict()
                    neg_scores = candidate_frame[candidate_frame[id_column].astype(str) == neg_id][[m.name for m in active_specs]].iloc[0].to_dict()
                    contrastive_text.append(
                        "HARD EXAMPLES\n"
                        f"POS id={pos_id} scores={pos_scores}\n{_truncate_text(str(pos_row[text_column]))}\n\n"
                        f"NEG id={neg_id} scores={neg_scores}\n{_truncate_text(str(neg_row[text_column]))}"
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

        current_metrics_text = "\n\n".join([f"{m.name}\n{m.rubric_text}" for m in active_specs])
        new_specs: List[MetricSpec] = []
        dropped_empty = 0
        dropped_dupe = 0
        dropped_llm = 0
        attempt = 0
        while attempt < max_candidate_attempts and len(new_specs) < min_unique:
            candidate_defs = proposer.propose(
                task_description=task_description,
                positive_examples="",
                negative_examples="",
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

            for cand in candidate_defs:
                name = cand.get("name") or "Metric"
                scale = str(cand.get("scale") or "ordinal").lower()
                rubric = _normalize_rubric(cand.get("rubric"), scale)
                rubric_text = _rubric_to_text(rubric)
                if not rubric_text:
                    dropped_empty += 1
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

        if not new_specs:
            no_new_metrics_iters += 1
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} no new metrics survived "
                f"(dropped_empty={dropped_empty}, dropped_duplicate={dropped_dupe}, "
                f"dropped_llm={dropped_llm}, counter={no_new_metrics_iters})",
                verbose_only=False,
                verbose=verbose,
            )

            eval_sel_frame_active = _build_feature_frame(
                eval_sel_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
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

            test_frame = _build_feature_frame(
                test_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
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
                "churn": churn,
                "churn_warning": churn_warning,
                "method": used_method,
                "note": "no_new_metrics",
                "train_assessed": train_assessed,
                "mismatch_stats": mismatch_stats,
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

            if no_new_metrics_iters >= 2 or eval_plateau_iters >= 2:
                _iter_log(
                    f"[Autometrics][Iterative] Iteration {iteration} stopping: no-new-metrics or eval plateau.",
                    verbose_only=False,
                    verbose=verbose,
                )
                break
            continue

        joint_specs = active_specs + new_specs
        eval_sel_frame_joint = _build_feature_frame(
            eval_sel_df, joint_specs, label_cache, id_column, text_column, label_column, label_batch_size,
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

        joint_model = fit_regression(eval_sel_frame_joint, joint_specs, name=f"eval_selection_{iteration}")
        joint_coef_map = {
            spec.metric_id: float(coef)
            for spec, coef in zip(joint_specs, joint_model.model.coef_.reshape(-1))
        }

        eval_gate_probs = compute_probabilities(joint_model, eval_gate_frame_joint, joint_specs, name=f"eval_gating_{iteration}")
        eval_gate_metrics = _compute_metrics(eval_gate_frame_joint[label_column].values, eval_gate_probs)
        gate_key, gate_value = _gate_metric(eval_gate_metrics)

        new_survived = any(abs(joint_coef_map.get(spec.metric_id, 0.0)) > 1e-6 for spec in new_specs)
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
            active_metric_ids = _select_active_metric_ids(joint_specs, joint_coef_map)
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
            eval_sel_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
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
        model = fit_regression(eval_sel_frame_active, active_specs, name=f"eval_selection_active_{iteration}")

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

        test_frame = _build_feature_frame(
            test_df, active_specs, label_cache, id_column, text_column, label_column, label_batch_size,
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
            "churn": churn,
            "churn_warning": churn_warning,
            "method": used_method,
            "train_assessed": train_assessed,
            "mismatch_stats": mismatch_stats,
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

        if no_new_metrics_iters >= 2 or eval_plateau_iters >= 2:
            _iter_log(
                f"[Autometrics][Iterative] Iteration {iteration} stopping: no-new-metrics or eval plateau.",
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
        "output_dir": str(output_path),
    }
