"""Dump {task, metric_id, name, subtask_breadth} for the rubric banks, so any recon/GEPA result
(which logs metric_id) can be joined to the metric's specificity class. metric_ids are deterministic
(f'{file_stem}_{i}'), so a map built with the current loader matches prior runs."""
import sys
import pandas as pd
from .manifest import full_manifest, load_metrics

OUT = "/lfs/skampere3/0/alexspan/tmp_vinfo/breadth_map.parquet"


def main(argv=None):
    tasks = (argv or ["math_se", "creative_writing"])
    man = full_manifest(metrics_per_task=200, metric_files_cap=600)
    rows = []
    for tname in tasks:
        entry = next((e for e in man.datasets if e.name == tname), None)
        if entry is None:
            print("skip", tname); continue
        for m in load_metrics(entry):
            rows.append({"task": tname, "metric_id": m.metric_id, "name": m.name,
                         "subtask_breadth": (m.meta or {}).get("subtask_breadth")})
    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    print(f"wrote {OUT}: {len(df)} metrics")
    print(df.groupby(["task", "subtask_breadth"]).size())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
