"""
Aggregate the 11 R2 attribute labeling outputs into one long-form parquet
plus per-axis × per-task summary tables.

Multi-label semantics: one (aspect_id, axis, value) row per labeling assignment.
An aspect with 2 rule_type values produces 2 rule_type rows.

Reads:
  outputs/r2_attr_labeling/outputs/<task>.jsonl

Writes:
  outputs/r2_attr_labeling/agg/r2_attributes.parquet
  outputs/r2_attr_labeling/agg/<axis>__by_task.csv  (counts)
  outputs/r2_attr_labeling/agg/<axis>__by_task_pct.csv (percent of aspects)
  outputs/r2_attr_labeling/agg/coverage_summary.csv (per-task: # aspects labeled, distinct values per axis)
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
IN_DIR = ROOT / "outputs/r2_attr_labeling/outputs"
AGG = ROOT / "outputs/r2_attr_labeling/agg"
AGG.mkdir(parents=True, exist_ok=True)

AXES = ["rule_type", "focus", "outcome_vs_compliance", "scope", "modality"]

CLOSED_VOCAB = {
    "rule_type": {
        "substantive_quality", "procedural", "stylistic",
        "transparency_disclosure", "epistemic_integrity",
        "safety_harm_minimization", "external_standard_compliance",
    },
    "focus": {"artifact", "process", "actor", "audience"},
    "outcome_vs_compliance": {"outcome", "compliance"},
    "scope": {"piece-local", "section-specific", "whole-artifact", "corpus-level"},
    "modality": {"textual", "structural", "numerical", "visual",
                 "temporal_procedural", "behavioral"},
}


def main():
    rows = []
    aspect_rows = []
    issues = []
    for fp in sorted(IN_DIR.glob("*.jsonl")):
        task = fp.stem
        with open(fp) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception as e:
                    issues.append(f"{task}:{line_no} parse {e}"); continue
                aspect_rows.append({
                    "task": task,
                    "aspect_id": o.get("aspect_id"),
                    "name": o.get("name", ""),
                    "description": o.get("description", ""),
                })
                for axis in AXES:
                    vals = o.get(axis) or []
                    if not isinstance(vals, list):
                        issues.append(f"{task}:{line_no} {axis} not list"); continue
                    for v in vals:
                        v = (v or "").strip()
                        valid = v in CLOSED_VOCAB[axis]
                        rows.append({
                            "task": task,
                            "aspect_id": o.get("aspect_id"),
                            "axis": axis,
                            "value": v,
                            "valid": valid,
                        })
    long_df = pd.DataFrame(rows)
    aspects_df = pd.DataFrame(aspect_rows).drop_duplicates(["task", "aspect_id"])

    print(f"loaded {len(aspects_df):,} aspects across {aspects_df['task'].nunique()} tasks")
    print(f"  -> {len(long_df):,} (aspect, axis, value) rows")
    print(f"  issues: {len(issues)}; invalid vocab: {(~long_df['valid']).sum()}")
    if issues[:5]:
        for i in issues[:5]: print("   ", i)

    long_df.to_parquet(AGG / "r2_attributes.parquet")

    # Per-axis × task: counts (# of (aspect, value) pairs per task)
    for axis in AXES:
        sub = long_df[(long_df["axis"] == axis) & (long_df["valid"])]
        cnts = sub.groupby(["task", "value"]).size().unstack("value", fill_value=0)
        cnts = cnts.reindex(columns=sorted(CLOSED_VOCAB[axis]), fill_value=0)
        cnts.to_csv(AGG / f"{axis}__by_task.csv")

        # percent of aspects (per task) that have each value (multi-label-safe)
        n_per_task = aspects_df.groupby("task").size()
        pcts = (
            sub.groupby(["task", "value", "aspect_id"]).size()
               .reset_index()
               .groupby(["task", "value"]).size()
               .unstack("value", fill_value=0)
        )
        pcts = pcts.reindex(columns=sorted(CLOSED_VOCAB[axis]), fill_value=0)
        pcts = pcts.div(n_per_task, axis=0) * 100
        pcts.round(1).to_csv(AGG / f"{axis}__by_task_pct.csv")

    # coverage summary
    cov = aspects_df.groupby("task").size().to_frame("n_aspects")
    for axis in AXES:
        ax_sub = long_df[(long_df["axis"] == axis) & (long_df["valid"])]
        avg_per_aspect = (
            ax_sub.groupby(["task", "aspect_id"]).size()
                  .groupby("task").mean()
        )
        cov[f"avg_{axis}_per_aspect"] = avg_per_aspect
    cov.round(2).to_csv(AGG / "coverage_summary.csv")
    print(f"\nwrote agg outputs to {AGG}")

    # Print rule_type × task percent table since user cares most about that
    pct_path = AGG / "rule_type__by_task_pct.csv"
    if pct_path.exists():
        print("\n=== rule_type — % aspects per task (multi-label, so can exceed 100% sum) ===")
        df = pd.read_csv(pct_path, index_col=0)
        print(df.round(1).to_string())


if __name__ == "__main__":
    main()
