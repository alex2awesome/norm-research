"""
For each R2 aspect (across all 11 tasks), compute its source-year distribution:
  median, IQR, span, n_sources, pre-2000 share, post-2020 share, longevity score.

Chain: R2 aspect -> family_ids -> R1 family.cluster_ids -> L0 keys -> page_id -> year

Writes:
  outputs/r2_attr_labeling/agg/r2_time_profile.parquet
    per-aspect rows with year stats

  outputs/r2_attr_labeling/agg/task_stability.parquet
    per-task rollup: how stable / persistent / modern are this task's aspects?
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
SM = ROOT / "outputs/analyses/structural_metrics"
ANALYSES = ROOT / "outputs/analyses"
OUT = ROOT / "outputs/r2_attr_labeling/agg"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math-stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]

# Confidence threshold for source years
MIN_CONF = 2

# Year bounds for "longevity" calculation
MIN_YR = 1900
MAX_YR = 2026


def load_years():
    """page_id -> (year, confidence)"""
    out = {}
    with open(ANALYSES / "source_years.jsonl") as f:
        for line in f:
            o = json.loads(line)
            y = o.get("year")
            c = o.get("year_confidence", 0)
            if y is not None and c >= MIN_CONF:
                # clamp insane years
                if MIN_YR - 200 <= y <= MAX_YR + 5:
                    out[o["page_id"]] = (y, c)
    return out


def load_key_to_page():
    out = {}
    with open(ANALYSES / "canon_all_real_forms.jsonl") as f:
        for line in f:
            o = json.loads(line)
            parts = o["key"].rsplit("::", 1)
            if len(parts) == 2:
                out[o["key"]] = parts[0]
    return out


def task_chain(task: str, key_to_page: dict, years: dict):
    """For one task, compute per-aspect year stats."""
    clusters = json.loads((SM / f"clusters_{task}.json").read_text())
    cluster_keys = defaultdict(list)
    for k, c in clusters.items():
        cluster_keys[c].append(k)

    r1 = json.loads((SM / "r1_v4a_lora_fork3_merge" / f"r1_families_{task}.json").read_text())
    fams = r1["families"]

    r2 = json.loads((SM / "r2_v1_subagent" / f"r2_aspects_{task}.json").read_text())
    rows = []
    for asp in r2["aspects"]:
        fam_ids = asp.get("family_ids", [])
        yrs = []
        srcs = set()
        for fid in fam_ids:
            if 0 <= fid < len(fams):
                for cid in fams[fid].get("cluster_ids", []):
                    for k in cluster_keys.get(cid, []):
                        pg = key_to_page.get(k)
                        if pg in years:
                            yrs.append(years[pg][0])
                            # source = source_file (3rd ::-segment of key)
                            kp = k.split("::")
                            if len(kp) >= 3:
                                srcs.add(kp[2])
        rec = {
            "task": task,
            "aspect_id": asp.get("aspect_id"),
            "name": asp.get("name"),
            "n_families": asp.get("n_families", len(fam_ids)),
            "n_yrs": len(yrs),
            "n_sources_with_yr": len(srcs),
        }
        if yrs:
            yarr = np.array(yrs)
            rec.update({
                "min_y":   int(yarr.min()),
                "max_y":   int(yarr.max()),
                "median_y": float(np.median(yarr)),
                "q25_y":    float(np.percentile(yarr, 25)),
                "q75_y":    float(np.percentile(yarr, 75)),
                "iqr_y":    float(np.percentile(yarr, 75) - np.percentile(yarr, 25)),
                "span_y":   int(yarr.max() - yarr.min()),
                "pre_2000_share": float((yarr < 2000).mean()),
                "post_2020_share": float((yarr >= 2020).mean()),
                # Longevity: number of distinct half-decade buckets touched within [MIN_YR, MAX_YR]
                "n_halfdecades_touched": int(len(set((yarr.clip(MIN_YR, MAX_YR)) // 5)))
            })
        rows.append(rec)
    return rows


def main():
    print("loading years and key-to-page index...")
    years = load_years()
    print(f"  pages with conf >= {MIN_CONF}: {len(years):,}")
    key_to_page = load_key_to_page()
    print(f"  canonical keys: {len(key_to_page):,}")

    all_rows = []
    for task in TASKS:
        rows = task_chain(task, key_to_page, years)
        with_yrs = sum(1 for r in rows if r["n_yrs"] > 0)
        print(f"  {task:30s} {len(rows):>4} aspects, {with_yrs:>4} with year data")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT / "r2_time_profile.parquet")
    print(f"\nwrote {len(df):,} rows -> {OUT}/r2_time_profile.parquet")

    # Per-task rollup
    have = df[df["n_yrs"] > 0]
    roll_rows = []
    for task, g in have.groupby("task"):
        roll_rows.append({
            "task": task,
            "n_aspects_total": len(df[df.task == task]),
            "n_aspects_with_yr": len(g),
            # Task stability: distribution of per-aspect median years
            "task_median_y":  float(g["median_y"].median()),
            "task_iqr_of_medians": float(g["median_y"].quantile(0.75) - g["median_y"].quantile(0.25)),
            # Per-aspect longevity (avg # half-decades touched)
            "avg_aspect_halfdecades_touched": float(g["n_halfdecades_touched"].mean()),
            "med_aspect_halfdecades_touched": float(g["n_halfdecades_touched"].median()),
            # Modern-ness
            "pct_aspects_median_post_2020": float((g["median_y"] >= 2020).mean() * 100),
            "pct_aspects_with_pre_2000_source": float((g["pre_2000_share"] > 0).mean() * 100),
            # Per-aspect IQR distribution (within-aspect spread)
            "avg_aspect_iqr_y": float(g["iqr_y"].mean()),
        })
    roll = pd.DataFrame(roll_rows).sort_values("task_median_y")
    roll.to_parquet(OUT / "task_stability.parquet")
    print(f"\n=== per-task stability/recency rollup ===")
    print(roll.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
