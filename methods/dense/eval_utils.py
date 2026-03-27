"""
Shared evaluation utilities for dense reward models.

Provides:
  - compute_metrics(): standard binary classification metrics
  - discover_dataset_evals(): find dataset-specific evals.py
  - run_dataset_evals(): compute per-slice metrics and summaries
"""

import importlib.util
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard metrics (single source of truth)
# ---------------------------------------------------------------------------

def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Compute standard binary classification metrics.

    This is the canonical implementation used by both training evaluation
    and the standalone eval script.
    """
    probs = probs.astype(np.float64)
    labels = labels.astype(int)
    preds = (probs >= 0.5).astype(int)
    probs_clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    bce_loss = -np.mean(
        labels * np.log(probs_clipped) + (1 - labels) * np.log(1.0 - probs_clipped)
    )
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return {
        "loss": float(bce_loss),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(auc),
        "n": int(len(labels)),
        "pos_rate": float(labels.mean()),
    }


# ---------------------------------------------------------------------------
# Dataset-specific eval discovery
# ---------------------------------------------------------------------------

def discover_dataset_evals(data_path: str):
    """Find and load evals.py from the dataset's directory.

    Searches the parent and grandparent of data_path for an evals.py module.

    Returns:
        Loaded module or None if not found.
    """
    p = Path(data_path).resolve()
    for candidate in [p.parent, p.parent.parent]:
        evals_path = candidate / "evals.py"
        if evals_path.exists():
            logger.info("Found dataset-specific evals at %s", evals_path)
            spec = importlib.util.spec_from_file_location("dataset_evals", str(evals_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def run_dataset_evals(
    df: pd.DataFrame,
    labels: np.ndarray,
    probs: np.ndarray,
    data_path: str,
    output_path: Optional[str] = None,
) -> dict:
    """Run dataset-specific sliced evaluation.

    Args:
        df: DataFrame with all columns (including metadata for slicing).
        labels: ground-truth binary labels (aligned with df rows).
        probs: predicted probabilities (aligned with df rows).
        data_path: path to the dataset file (used to discover evals.py).
        output_path: optional path to write results JSON.

    Returns:
        Dict with keys: overall, slices (per-slice metrics), summary (aggregates).
        Returns empty dict if no dataset-specific evals found.
    """
    evals_mod = discover_dataset_evals(data_path)
    if evals_mod is None:
        logger.info("No dataset-specific evals found for %s", data_path)
        return {}

    results = {}

    # Overall metrics
    results["overall"] = compute_metrics(labels, probs)

    # Per-slice metrics
    if hasattr(evals_mod, "get_slices"):
        slices = evals_mod.get_slices(df)
        logger.info("Running %d dataset-specific slices", len(slices))

        slice_metrics = {}
        for slice_name, mask in sorted(slices.items()):
            mask_arr = mask.to_numpy() if hasattr(mask, "to_numpy") else np.array(mask)
            s_labels = labels[mask_arr]
            s_probs = probs[mask_arr]
            m = compute_metrics(s_labels, s_probs)
            slice_metrics[slice_name] = m
            logger.info(
                "  %-40s | n=%5d | pos=%.1f%% | AUC: %.4f | F1: %.4f",
                slice_name, m["n"], m["pos_rate"] * 100, m["auc"], m["f1"],
            )

        results["slices"] = slice_metrics

        # Summary / aggregate metrics
        if hasattr(evals_mod, "get_summary_metrics"):
            summary = evals_mod.get_summary_metrics(slice_metrics)
            results["summary"] = summary
            logger.info("--- Summary metrics ---")
            for k, v in summary.items():
                if isinstance(v, float):
                    logger.info("  %-40s : %.4f", k, v)
                else:
                    logger.info("  %-40s : %s", k, v)

    # Save results
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Dataset-specific eval results written to %s", out)

    return results
