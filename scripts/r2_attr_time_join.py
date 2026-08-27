"""
Join R2 attribute labels with R2 time-profile data.

Both tables key on (task, aspect_id). r2_attributes.parquet stores aspect_id
with the labeling pipeline naming (task uses snake_case e.g. peer_review).
r2_time_profile.parquet stores task in kebab-case (peer-review). Aspect_ids
share the same string ('b0_a0', etc.) since both pipelines source from
structural_metrics/r2_v1_subagent.

Writes:
  outputs/r2_attr_labeling/agg/attr_time_join.parquet
  outputs/r2_attr_labeling/agg/attr_x_median_year.csv  (per-axis-value medians)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
AGG  = ROOT / "outputs/r2_attr_labeling/agg"


def main():
    attr = pd.read_parquet(AGG / "r2_attributes.parquet")
    time = pd.read_parquet(AGG / "r2_time_profile.parquet")
    # Normalise task names to snake_case in both
    attr["task_norm"] = attr["task"].str.replace("-", "_")
    time["task_norm"] = time["task"].str.replace("-", "_")

    # Drop attribute rows for invalid values (we already checked, but be safe)
    attr = attr[attr["valid"]]

    j = attr.merge(
        time[["task_norm", "aspect_id", "n_yrs", "median_y", "iqr_y", "span_y",
              "pre_2000_share", "post_2020_share", "n_halfdecades_touched",
              "n_sources_with_yr"]],
        on=["task_norm", "aspect_id"],
        how="left",
    )
    j.to_parquet(AGG / "attr_time_join.parquet")
    print(f"joined {len(j):,} (attribute row × time) records")
    print(f"  attributes with year data: {j['median_y'].notna().sum():,}")

    have = j[j["median_y"].notna()]

    # Per axis-value: median of medians across all tasks
    print("\n=== Per axis-value: median year, half-decades touched, # aspects ===")
    summ_rows = []
    for axis in ["rule_type", "focus", "outcome_vs_compliance", "scope", "modality"]:
        sub = have[have["axis"] == axis]
        for val, g in sub.groupby("value"):
            summ_rows.append({
                "axis": axis,
                "value": val,
                "n_aspect_rows": len(g),
                "median_y_of_medians": float(g["median_y"].median()),
                "avg_halfdecades_touched": float(g["n_halfdecades_touched"].mean()),
                "pct_with_pre_2000_src": float((g["pre_2000_share"] > 0).mean() * 100),
                "pct_median_post_2020": float((g["median_y"] >= 2020).mean() * 100),
            })
    s = pd.DataFrame(summ_rows).sort_values(["axis", "median_y_of_medians"])
    s.to_csv(AGG / "attr_x_median_year.csv", index=False)
    print(s.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
