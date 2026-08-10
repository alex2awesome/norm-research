"""Clean audit of Qwen response health, with explicit invariants.

For each Qwen response file:
  - Try smart_parse → did it succeed?
  - If success: how many results entries? How many score entries per result?
  - Are dp_ids the expected 10 dp_ids from the prompt?
  - Are aspect_ids the expected 5-10 from the bundle?

Outputs a clean parquet of per-file health stats. Then aggregates to per-task summary.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Import smart parser from matrix builder
sys.path.insert(0, str(Path(__file__).parent))
from v2_build_feature_matrices import _smart_parse


TASKS = ["peer_review", "math", "notice_and_comment", "press_releases", "humor",
         "news_homepages", "patents", "code_review", "creative_writing"]


def audit_file(fp: Path):
    """Returns dict of health metrics for a single response file."""
    raw = fp.read_text()
    out = {
        "file": fp.name,
        "raw_len": len(raw),
        "has_think_tag": "</think>" in raw,
        "parsed_ok": False,
        "n_results": 0,
        "n_scores_total": 0,
        "n_unique_dps": 0,
        "n_unique_aspects": 0,
        "n_applicable": 0,
        "n_scored": 0,
        "parse_method": None,
    }
    obj = _smart_parse(raw)
    if obj is None:
        # Try to detect what went wrong
        if raw.strip() == "":
            out["parse_method"] = "empty"
        elif "{" not in raw:
            out["parse_method"] = "no_json_chars"
        elif "</think>" in raw and len(raw.rsplit("</think>", 1)[1].strip()) < 50:
            out["parse_method"] = "think_only_short_tail"
        else:
            out["parse_method"] = "smart_parse_failed"
        return out
    out["parsed_ok"] = True
    results = obj.get("results", [])
    out["n_results"] = len(results)
    dps = set()
    aspects = set()
    for r in results:
        if not isinstance(r, dict): continue
        dp = r.get("text_id") or r.get("dp_id")
        if dp: dps.add(dp)
        for s in r.get("scores", []):
            if not isinstance(s, dict): continue
            out["n_scores_total"] += 1
            if s.get("aspect_id"): aspects.add(s["aspect_id"])
            if s.get("applicable"):
                out["n_applicable"] += 1
                if s.get("score") is not None: out["n_scored"] += 1
    out["n_unique_dps"] = len(dps)
    out["n_unique_aspects"] = len(aspects)
    out["parse_method"] = "smart_parse_ok"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--qwen-pool", default="runs/validity_full/v2/_qwen_pool")
    ap.add_argument("--out", default="outputs/v2_analysis/qwen_audit.parquet")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    pool = repo / args.qwen_pool

    rows = []
    for task in args.tasks:
        # Try both pools (gpu3 and gpu5)
        for sub in ["gpu3_responses", "gpu5_responses"]:
            d = pool / sub
            if not d.exists(): continue
            files = list(d.glob(f"{task}__*.json"))
            if not files: continue
            for fp in files:
                r = audit_file(fp)
                r["task"] = task
                r["pool"] = sub.replace("_responses", "")
                rows.append(r)

    df = pd.DataFrame(rows)
    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Save as csv (sk3 may not have pyarrow); also save as parquet if we can
    out_csv = out.with_suffix(".csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}\n")
    try:
        df.to_parquet(out, index=False)
    except Exception:
        pass

    # Per-task summary
    print(f"{'task':<22} {'files':>6} {'parsed':>6} {'parse%':>7} "
          f"{'mean_n_results':>14} {'mean_n_dps':>11} {'mean_n_scores':>14}")
    print("-" * 84)
    for task in args.tasks:
        sub = df[df["task"] == task]
        if len(sub) == 0: continue
        n = len(sub)
        parsed = sub["parsed_ok"].sum()
        mean_r = sub[sub["parsed_ok"]]["n_results"].mean() if parsed > 0 else 0
        mean_dps = sub[sub["parsed_ok"]]["n_unique_dps"].mean() if parsed > 0 else 0
        mean_sc = sub[sub["parsed_ok"]]["n_scores_total"].mean() if parsed > 0 else 0
        print(f"{task:<22} {n:>6} {parsed:>6} {parsed/n*100:>6.1f}% "
              f"{mean_r:>14.2f} {mean_dps:>11.2f} {mean_sc:>14.2f}")

    # Failure mode breakdown
    print(f"\n=== Failure modes (across all tasks) ===")
    failed = df[~df["parsed_ok"]]
    for mode, n in failed["parse_method"].value_counts().items():
        print(f"  {mode}: {n}")


if __name__ == "__main__":
    main()
