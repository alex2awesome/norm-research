"""End-to-end pipeline orchestration for both local explanation approaches."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .clustering import build_canonical_vocabulary
from .config import LocalExplanationConfig, TaskMetadata
from .explainer import LocalExplainer
from .feature_weights import aggregate_feature_scores, compute_star_weights
from .predictor import PredictionResult, fit_and_evaluate
from .prior import PriorEstimator
from .scorer import score_examples_on_vocabulary

logger = logging.getLogger(__name__)


def _truncate_texts(df: pd.DataFrame, text_column: str, max_tokens: int) -> pd.DataFrame:
    """Truncate text to max whitespace tokens."""
    if max_tokens <= 0:
        return df
    df = df.copy()
    df[text_column] = df[text_column].apply(
        lambda t: " ".join(str(t).split()[:max_tokens])
    )
    return df


def run_rationalization(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: LocalExplanationConfig,
    scoring_backend: Any,
    generate_fn,  # callable(prompt: str) -> str — for prior, clustering, rubrics
    task_description: str,
    task_metadata: TaskMetadata,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    tokenizer: Any = None,
    max_model_len: int = 0,
) -> Dict[str, Any]:
    """Run Approach A: Rationalization + Prior Calibration.

    Pipeline:
    1. Estimate prior
    2. Generate per-example rationalizations on train set
    3. Calibrate features against prior
    4. Build canonical vocabulary (clustering pipeline)
    5. Score all splits on vocabulary
    6. Fit & evaluate L1 logistic regression
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Truncate texts
    train_df = _truncate_texts(train_df, text_column, config.max_text_tokens)
    eval_df = _truncate_texts(eval_df, text_column, config.max_text_tokens)
    test_df = _truncate_texts(test_df, text_column, config.max_text_tokens)

    # --- Step 1: Estimate prior ---
    logger.info("=== Step 1: Estimating prior ===")
    prior_estimator = PriorEstimator(config, task_description, task_metadata)
    prior = prior_estimator.estimate_prior(generate_fn)

    # --- Step 2: Per-example rationalization ---
    logger.info("=== Step 2: Per-example rationalization ===")
    explainer = LocalExplainer(
        config, scoring_backend, task_description, task_metadata,
        cache_dir=str(output_dir / "explanation_cache"),
    )
    example_features = explainer.explain_rationalization_batch(
        texts=train_df[text_column].tolist(),
        labels=train_df[label_column].tolist(),
        doc_ids=train_df[id_column].astype(str).tolist(),
    )

    # --- Step 3: Calibrate ---
    logger.info("=== Step 3: Prior calibration ===")
    calibrated = prior_estimator.calibrate_features(example_features, prior)

    # Flatten to weighted features list
    weighted_features = []
    for doc_id, feat_weights in calibrated.items():
        weighted_features.extend(feat_weights)

    logger.info("Total weighted features: %d", len(weighted_features))

    # --- Step 4: Build canonical vocabulary ---
    logger.info("=== Step 4: Clustering pipeline ===")
    canonical = build_canonical_vocabulary(
        weighted_features, config,
        text_type=task_metadata.text_type,
        generate_fn=generate_fn,
        output_dir=str(output_dir / "clustering"),
    )

    if not canonical:
        logger.error("No canonical features produced — aborting")
        return {"error": "No canonical features"}

    # --- Step 5: Score all splits ---
    logger.info("=== Step 5: Binary scoring on canonical features ===")
    scored_train = score_examples_on_vocabulary(
        train_df, canonical, scoring_backend, task_description,
        id_column=id_column, text_column=text_column, label_column=label_column,
        label_cache_dir=str(output_dir / "score_cache"),
        tokenizer=tokenizer, max_model_len=max_model_len,
        stage="train",
    )
    scored_eval = score_examples_on_vocabulary(
        eval_df, canonical, scoring_backend, task_description,
        id_column=id_column, text_column=text_column, label_column=label_column,
        label_cache_dir=str(output_dir / "score_cache"),
        tokenizer=tokenizer, max_model_len=max_model_len,
        stage="eval",
    )
    scored_test = score_examples_on_vocabulary(
        test_df, canonical, scoring_backend, task_description,
        id_column=id_column, text_column=text_column, label_column=label_column,
        label_cache_dir=str(output_dir / "score_cache"),
        tokenizer=tokenizer, max_model_len=max_model_len,
        stage="test",
    )

    # --- Step 6: Fit & evaluate ---
    logger.info("=== Step 6: L1 logistic regression ===")
    feature_cols = [c for c in scored_train.columns if c not in (id_column, label_column, text_column)]
    result = fit_and_evaluate(
        train_X=scored_train[feature_cols].values,
        train_y=scored_train[label_column].values,
        eval_X=scored_eval[feature_cols].values,
        eval_y=scored_eval[label_column].values,
        test_X=scored_test[feature_cols].values,
        test_y=scored_test[label_column].values,
        feature_names=feature_cols,
    )

    # Save results
    _save_results(result, canonical, output_dir, approach="rationalization")
    return _format_output(result, canonical)


def run_star_local(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: LocalExplanationConfig,
    scoring_backend: Any,
    generate_fn,  # callable(prompt: str) -> str
    task_description: str,
    task_metadata: TaskMetadata,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    tokenizer: Any = None,
    max_model_len: int = 0,
) -> Dict[str, Any]:
    """Run Approach B: Amended STaR-Local.

    Pipeline:
    1. Blind extraction + deliberation + prediction on train set
    2. Compute per-feature weights from prediction correctness
    3. Build canonical vocabulary (clustering pipeline)
    4. Score all splits on vocabulary
    5. Fit & evaluate L1 logistic regression
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = _truncate_texts(train_df, text_column, config.max_text_tokens)
    eval_df = _truncate_texts(eval_df, text_column, config.max_text_tokens)
    test_df = _truncate_texts(test_df, text_column, config.max_text_tokens)

    # --- Step 1: Blind extraction ---
    logger.info("=== Step 1: STaR-Local blind extraction ===")
    explainer = LocalExplainer(
        config, scoring_backend, task_description, task_metadata,
        cache_dir=str(output_dir / "explanation_cache"),
    )
    star_results = explainer.explain_star_local_batch(
        texts=train_df[text_column].tolist(),
        doc_ids=train_df[id_column].astype(str).tolist(),
    )

    # --- Step 2: Compute weights ---
    logger.info("=== Step 2: Computing STaR weights ===")
    true_labels = dict(zip(
        train_df[id_column].astype(str),
        train_df[label_column].astype(int),
    ))
    weighted_features = compute_star_weights(
        star_results, true_labels, task_metadata.positive_label, config,
    )

    # Build per-feature polarity map: 'for' / 'against' / 'both'.
    # Used downstream to filter cross-polarity candidate pairs before LLM labeling
    # — features that argue for opposite outcomes can't be the same concept by construction.
    feature_polarity: Dict[str, str] = {}
    for _doc_id, _result in star_results.items():
        for _f in _result.features_for:
            _f = _f.strip()
            if not _f:
                continue
            feature_polarity[_f] = "both" if feature_polarity.get(_f) == "against" else "for"
        for _f in _result.features_against:
            _f = _f.strip()
            if not _f:
                continue
            feature_polarity[_f] = "both" if feature_polarity.get(_f) == "for" else "against"
    n_for = sum(1 for v in feature_polarity.values() if v == "for")
    n_against = sum(1 for v in feature_polarity.values() if v == "against")
    n_both = sum(1 for v in feature_polarity.values() if v == "both")
    logger.info(
        "Feature polarity (unique strings): %d for, %d against, %d both",
        n_for, n_against, n_both,
    )

    logger.info("Total weighted features: %d", len(weighted_features))

    # --- Step 3: Build canonical vocabulary ---
    # Clustering fine-tunes a sentence-transformer on GPU. vLLM's Qwen is
    # currently holding ~95% of the device, so release it first, then reload
    # after clustering for Step 4 binary scoring. Safe no-ops if the backend
    # doesn't expose release/reload.
    released = False
    if hasattr(scoring_backend, "release") and hasattr(scoring_backend, "reload"):
        logger.info("Releasing VLLM backend to free GPU for clustering/fine-tuning")
        try:
            scoring_backend.release()
            released = True
        except Exception as exc:
            logger.warning("Backend release failed (%s) — continuing anyway", exc)

    logger.info("=== Step 3: Clustering pipeline ===")
    canonical = build_canonical_vocabulary(
        weighted_features, config,
        text_type=task_metadata.text_type,
        generate_fn=generate_fn,
        output_dir=str(output_dir / "clustering"),
        feature_polarity=feature_polarity,
    )

    if released:
        logger.info("Reloading VLLM backend for Step 4 binary scoring")
        scoring_backend.reload()

    if not canonical:
        logger.error("No canonical features produced — aborting")
        return {"error": "No canonical features"}

    # --- Step 4: Score all splits ---
    logger.info("=== Step 4: Binary scoring on canonical features ===")
    scored_train = score_examples_on_vocabulary(
        train_df, canonical, scoring_backend, task_description,
        id_column=id_column, text_column=text_column, label_column=label_column,
        label_cache_dir=str(output_dir / "score_cache"),
        tokenizer=tokenizer, max_model_len=max_model_len,
        stage="train",
    )
    scored_eval = score_examples_on_vocabulary(
        eval_df, canonical, scoring_backend, task_description,
        id_column=id_column, text_column=text_column, label_column=label_column,
        label_cache_dir=str(output_dir / "score_cache"),
        tokenizer=tokenizer, max_model_len=max_model_len,
        stage="eval",
    )
    scored_test = score_examples_on_vocabulary(
        test_df, canonical, scoring_backend, task_description,
        id_column=id_column, text_column=text_column, label_column=label_column,
        label_cache_dir=str(output_dir / "score_cache"),
        tokenizer=tokenizer, max_model_len=max_model_len,
        stage="test",
    )

    # --- Step 5: Fit & evaluate ---
    logger.info("=== Step 5: L1 logistic regression ===")
    feature_cols = [c for c in scored_train.columns if c not in (id_column, label_column, text_column)]
    result = fit_and_evaluate(
        train_X=scored_train[feature_cols].values,
        train_y=scored_train[label_column].values,
        eval_X=scored_eval[feature_cols].values,
        eval_y=scored_eval[label_column].values,
        test_X=scored_test[feature_cols].values,
        test_y=scored_test[label_column].values,
        feature_names=feature_cols,
    )

    _save_results(result, canonical, output_dir, approach="star_local")
    return _format_output(result, canonical)


def _save_results(
    result: PredictionResult,
    canonical: list,
    output_dir: Path,
    approach: str,
):
    """Save results to disk."""
    results_dict = {
        "approach": approach,
        "train": result.train_metrics,
        "eval": result.eval_metrics,
        "test": result.test_metrics,
        "n_features": len(result.feature_names),
        "n_active_features": len(result.active_features),
        "active_features": [
            {"name": name, "coefficient": coef}
            for name, coef in result.active_features
        ],
        "canonical_features": [
            {"name": cf.name, "description": cf.description, "weight": cf.aggregate_weight}
            for cf in canonical
        ],
    }
    path = output_dir / f"results_{approach}.json"
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info("Results saved to %s", path)


def _format_output(result: PredictionResult, canonical: list) -> Dict[str, Any]:
    """Format output dict for the caller."""
    return {
        "test_metrics": result.test_metrics,
        "eval_metrics": result.eval_metrics,
        "active_features": result.active_features,
        "n_canonical_features": len(canonical),
        "model": result.model,
    }
