"""
Dataset-specific evaluation slices for the peer-review dataset.

The global eval runner (scripts/eval_model.py) discovers this module and calls:
  - get_slices(df) -> dict of slice_name -> boolean mask
  - get_summary_metrics(slice_metrics) -> dict of aggregate metric names -> values
"""

import numpy as np
import pandas as pd


def get_slices(df: pd.DataFrame) -> dict:
    """Return a dict of slice_name -> boolean Series over the dataframe.

    Each slice defines a subset of the data on which standard metrics
    (accuracy, precision, recall, f1, auc) will be computed independently.
    """
    slices = {}

    # --- Per-venue slices (the main diagnostic for venue-memorization) ---
    if "venue" in df.columns:
        for venue in sorted(df["venue"].dropna().unique()):
            mask = df["venue"] == venue
            # Only include slices with enough data and both classes
            if mask.sum() >= 20 and df.loc[mask, "judgement"].nunique() == 2:
                slices[f"venue/{venue}"] = mask

    # --- Per-source slices ---
    if "source" in df.columns:
        for source in sorted(df["source"].dropna().unique()):
            mask = df["source"] == source
            if mask.sum() >= 20 and df.loc[mask, "judgement"].nunique() == 2:
                slices[f"source/{source}"] = mask

    # --- Per-domain slices ---
    if "domain" in df.columns:
        for domain in sorted(df["domain"].dropna().unique()):
            mask = df["domain"] == domain
            if mask.sum() >= 20 and df.loc[mask, "judgement"].nunique() == 2:
                slices[f"domain/{domain}"] = mask

    # --- Accept-rate buckets (venues grouped by base rate) ---
    if "venue" in df.columns:
        venue_accept_rate = df.groupby("venue")["judgement"].mean()
        low_accept = venue_accept_rate[venue_accept_rate < 0.5].index
        mid_accept = venue_accept_rate[(venue_accept_rate >= 0.5) & (venue_accept_rate < 0.85)].index
        high_accept = venue_accept_rate[venue_accept_rate >= 0.85].index

        for name, venues in [("low_accept_rate(<50%)", low_accept),
                             ("mid_accept_rate(50-85%)", mid_accept),
                             ("high_accept_rate(>85%)", high_accept)]:
            mask = df["venue"].isin(venues)
            if mask.sum() >= 20 and df.loc[mask, "judgement"].nunique() == 2:
                slices[f"accept_bucket/{name}"] = mask

    return slices


def get_summary_metrics(slice_metrics: dict) -> dict:
    """Compute aggregate metrics across slices.

    Args:
        slice_metrics: dict of slice_name -> dict of metric_name -> value
                       (as returned by the global eval runner for each slice).

    Returns:
        dict of aggregate metric names -> values.
    """
    summary = {}

    # --- Macro-AUC across venues ---
    venue_aucs = {
        name: m["auc"]
        for name, m in slice_metrics.items()
        if name.startswith("venue/") and not np.isnan(m.get("auc", float("nan")))
    }
    if venue_aucs:
        summary["macro_auc_across_venues"] = float(np.mean(list(venue_aucs.values())))
        summary["std_auc_across_venues"] = float(np.std(list(venue_aucs.values())))
        summary["min_venue_auc"] = float(min(venue_aucs.values()))
        summary["max_venue_auc"] = float(max(venue_aucs.values()))
        summary["min_venue_auc_name"] = min(venue_aucs, key=venue_aucs.get)
        summary["max_venue_auc_name"] = max(venue_aucs, key=venue_aucs.get)

    # --- Macro-F1 across venues ---
    venue_f1s = {
        name: m["f1"]
        for name, m in slice_metrics.items()
        if name.startswith("venue/")
    }
    if venue_f1s:
        summary["macro_f1_across_venues"] = float(np.mean(list(venue_f1s.values())))

    # --- AUC gap between accept-rate buckets (venue memorization signal) ---
    bucket_aucs = {
        name: m["auc"]
        for name, m in slice_metrics.items()
        if name.startswith("accept_bucket/") and not np.isnan(m.get("auc", float("nan")))
    }
    if len(bucket_aucs) >= 2:
        summary["auc_gap_high_vs_low_accept_venues"] = float(
            bucket_aucs.get("accept_bucket/high_accept_rate(>85%)", float("nan"))
            - bucket_aucs.get("accept_bucket/low_accept_rate(<50%)", float("nan"))
        )

    return summary
