"""
Dataset-specific evaluation slices for the press-release dataset.

The global eval runner (scripts/eval_model.py) discovers this module and calls:
  - get_slices(df) -> dict of slice_name -> boolean mask
  - get_summary_metrics(slice_metrics) -> dict of aggregate metric names -> values
"""

import numpy as np
import pandas as pd


def get_slices(df: pd.DataFrame) -> dict:
    """Return a dict of slice_name -> boolean Series over the dataframe."""
    slices = {}

    # --- Per-company slices (top companies by volume) ---
    if "press_release_company" in df.columns:
        company_counts = df["press_release_company"].value_counts()
        top_companies = company_counts[company_counts >= 50].index
        for company in sorted(top_companies):
            mask = df["press_release_company"] == company
            if df.loc[mask, "judgement"].nunique() == 2:
                slices[f"company/{company}"] = mask

    # --- Per-news-domain slices (which outlets picked it up) ---
    if "news_article_domain" in df.columns:
        domain_counts = df["news_article_domain"].value_counts()
        top_domains = domain_counts[domain_counts >= 50].index
        for domain in sorted(top_domains):
            mask = df["news_article_domain"] == domain
            if df.loc[mask, "judgement"].nunique() == 2:
                slices[f"news_domain/{domain}"] = mask

    # --- Year slices ---
    if "press_release_date" in df.columns:
        dates = pd.to_datetime(df["press_release_date"], errors="coerce")
        years = dates.dt.year.dropna().unique()
        for year in sorted(years):
            mask = dates.dt.year == year
            if mask.sum() >= 20 and df.loc[mask, "judgement"].nunique() == 2:
                slices[f"year/{int(year)}"] = mask

    return slices


def get_summary_metrics(slice_metrics: dict) -> dict:
    """Compute aggregate metrics across slices."""
    summary = {}

    # --- Macro-AUC across companies ---
    company_aucs = {
        name: m["auc"]
        for name, m in slice_metrics.items()
        if name.startswith("company/") and not np.isnan(m.get("auc", float("nan")))
    }
    if company_aucs:
        summary["macro_auc_across_companies"] = float(np.mean(list(company_aucs.values())))
        summary["std_auc_across_companies"] = float(np.std(list(company_aucs.values())))
        summary["min_company_auc"] = float(min(company_aucs.values()))
        summary["max_company_auc"] = float(max(company_aucs.values()))

    # --- Macro-AUC across news domains ---
    domain_aucs = {
        name: m["auc"]
        for name, m in slice_metrics.items()
        if name.startswith("news_domain/") and not np.isnan(m.get("auc", float("nan")))
    }
    if domain_aucs:
        summary["macro_auc_across_news_domains"] = float(np.mean(list(domain_aucs.values())))

    return summary
