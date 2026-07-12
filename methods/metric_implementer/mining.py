"""Assemble the per-version analysis table for Stage-0 scaling-axis discovery (2026-06-18).

Joins, for every prompt version in a registry:
  - intrinsic prompt FEATURES (features.extract_prompt_features, re-read from the body),
  - provenance COVARIATES (N data budget, executor/judge model, GEPA operator, optimizer,
    round, lineage depth, token cap),
  - RECOVERY outcomes from the version's scorecard (reconstruction.behavioral — the transmission
    proxy; oracle_agreement.spearman when present; cf_validity; reliability.rho; consistency.mean;
    fidelity_scalar).

One row per (task, metric_id, version_id). This is the input to analyze_features.py, which asks
which columns vary enough and co-vary with recovery (within-metric) to be candidate scaling axes.

Reads the registry JSONs directly (versions/*.json + scorecards/<vid>__<kind>__<hash>.json) — no
LLM, no long-table parquet needed, so it runs anywhere the registry is checked out. The scorecard
carries no version_id field, so it is parsed from the filename.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .features import extract_prompt_features

# scorecard recovery fields -> output column (nested path as tuple)
_RECOVERY = {
    "recon_behavioral": ("reconstruction", "behavioral"),
    "recon_semantic": ("reconstruction", "semantic"),
    "cf_validity": ("counterfactual", "cf_validity"),
    "reliability_rho": ("reliability", "rho"),
    "consistency_mean": ("consistency", "mean"),
    "oracle_spearman": ("oracle_agreement", "spearman"),
    "fidelity_scalar": ("fidelity_scalar",),
}


def _dig(d: dict, path: tuple):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _scorecard_index(registry_dir: Path) -> Dict[tuple, dict]:
    """(metric_id, version_id) -> scorecard dict. version_id parsed from filename
    `<vid>__<kind>__<eval_hash>.json`; if multiple eval configs exist, keep the last."""
    idx: Dict[tuple, dict] = {}
    for p in sorted(registry_dir.glob("**/scorecards/*.json")):
        vid = p.name.split("__", 1)[0]
        try:
            card = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        idx[(card.get("metric_id"), vid)] = card
    return idx


def assemble_table(registry_dir, task: str = "") -> pd.DataFrame:
    """Build the per-version analysis table for one registry directory."""
    registry_dir = Path(registry_dir)
    cards = _scorecard_index(registry_dir)
    rows: List[dict] = []
    parents: Dict[str, Dict[str, Optional[str]]] = {}     # metric_id -> {vid: parent}

    for p in sorted(registry_dir.glob("**/versions/*.json")):
        try:
            v = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if v.get("kind") != "prompt":
            continue
        mid, vid = v.get("metric_id"), v.get("version_id")
        budget = v.get("budget", {}) or {}
        lineage = v.get("lineage", {}) or {}
        caps = budget.get("caps_in_force") or {}
        parents.setdefault(mid, {})[vid] = lineage.get("parent_version")

        row = {
            "task": task,
            "metric_id": mid,
            "version_id": vid,
            "name": v.get("name", ""),
            # ---- provenance covariates ----
            "optimizer": lineage.get("optimizer", "gepa"),     # populated once Phase-1 tags it
            "operator": lineage.get("operator"),
            "optimizer_round": lineage.get("optimizer_round"),
            "parent_version": lineage.get("parent_version"),
            "data_budget_n": budget.get("data_budget_n"),
            "judge_model": (budget.get("models") or {}).get("judge"),
            "token_cap": caps.get("instruction_tokens"),
            "is_seed": lineage.get("operator") == "INIT",
        }
        row.update(extract_prompt_features(v.get("body", "")))
        card = cards.get((mid, vid))
        row["has_scorecard"] = card is not None
        for col, path in _RECOVERY.items():
            row[col] = _dig(card, path) if card else None
        row["promotable"] = float(card["promotable"]) if card and "promotable" in card else None
        rows.append(row)

    df = pd.DataFrame(rows)
    # lineage depth (distance to an INIT/root), per metric
    if not df.empty:
        depths: Dict[tuple, int] = {}

        def depth(mid, vid, seen=()):
            if vid is None or (mid, vid) in seen:
                return 0
            par = parents.get(mid, {}).get(vid)
            if par is None:
                return 0
            return 1 + depth(mid, par, seen + ((mid, vid),))

        df["lineage_depth"] = [depth(r.metric_id, r.version_id) for r in df.itertuples()]
    return df


def assemble_all(outputs_dir, tasks: Optional[List[str]] = None) -> pd.DataFrame:
    """Stack the analysis tables for every task registry under outputs/metric_implementer/."""
    outputs_dir = Path(outputs_dir)
    frames = []
    task_dirs = ([outputs_dir / t for t in tasks] if tasks
                 else sorted(d for d in outputs_dir.iterdir() if (d / "registry").is_dir()))
    for td in task_dirs:
        reg = td / "registry"
        if reg.is_dir():
            frames.append(assemble_table(reg, task=td.name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    import sys
    from .config import REPO_ROOT
    out = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "outputs" / "metric_implementer")
    df = assemble_all(out)
    print(f"assembled {len(df)} versions across {df['task'].nunique()} tasks, "
          f"{df.groupby(['task','metric_id']).ngroups} metric-task groups")
    print(f"with scorecard: {int(df['has_scorecard'].sum())} ; "
          f"recon_behavioral non-null: {int(df['recon_behavioral'].notna().sum())}")
    print(df[["task", "metric_id", "version_id", "operator", "instruction_tokens",
              "is_decomposed", "recon_behavioral", "fidelity_scalar"]].head(12).to_string(index=False))
