"""
Expose per-mention (task, aspect_id, year, source_file) data so we can
make density / span / dot-strip plots in the notebook.

Writes:
  outputs/r2_attr_labeling/agg/r2_aspect_year_mentions.parquet
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
SM = ROOT / "outputs/analyses/structural_metrics"
ANALYSES = ROOT / "outputs/analyses"
OUT = ROOT / "outputs/r2_attr_labeling/agg"

TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math-stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]
MIN_CONF = 2


def main():
    # page -> year
    years = {}
    with open(ANALYSES / "source_years.jsonl") as f:
        for line in f:
            o = json.loads(line)
            y = o.get("year")
            c = o.get("year_confidence", 0)
            if y is not None and c >= MIN_CONF and -500 <= y <= 2030:
                years[o["page_id"]] = y

    # key -> page_id, task
    key_to_page = {}
    with open(ANALYSES / "canon_all_real_forms.jsonl") as f:
        for line in f:
            o = json.loads(line)
            parts = o["key"].rsplit("::", 1)
            if len(parts) == 2:
                key_to_page[o["key"]] = parts[0]

    all_rows = []
    for task in TASKS:
        clusters = json.loads((SM / f"clusters_{task}.json").read_text())
        cluster_keys = defaultdict(list)
        for k, c in clusters.items():
            cluster_keys[c].append(k)

        r1 = json.loads((SM / "r1_v4a_lora_fork3_merge" /
                         f"r1_families_{task}.json").read_text())
        fams = r1["families"]

        r2 = json.loads((SM / "r2_v1_subagent" /
                         f"r2_aspects_{task}.json").read_text())

        for asp in r2["aspects"]:
            for fid in asp.get("family_ids", []):
                if not (0 <= fid < len(fams)):
                    continue
                for cid in fams[fid].get("cluster_ids", []):
                    for k in cluster_keys.get(cid, []):
                        pg = key_to_page.get(k)
                        if pg in years:
                            kp = k.split("::")
                            src = kp[2] if len(kp) >= 3 else pg
                            all_rows.append({
                                "task": task,
                                "aspect_id": asp["aspect_id"],
                                "aspect_name": asp.get("name", ""),
                                "year": years[pg],
                                "page_id": pg,
                                "source_file": src,
                            })
        print(f"  {task:30s} {sum(1 for r in all_rows if r['task']==task):>6,} mentions")

    df = pd.DataFrame(all_rows)
    # one row per (aspect, source_file) to count distinct sources, not raw mentions
    df_dedup = df.drop_duplicates(["task", "aspect_id", "source_file"])
    df.to_parquet(OUT / "r2_aspect_year_mentions.parquet")
    df_dedup.to_parquet(OUT / "r2_aspect_year_mentions_distinct_sources.parquet")
    print(f"\nwrote {len(df):,} (raw mention) rows and {len(df_dedup):,} (distinct source) rows")


if __name__ == "__main__":
    main()
