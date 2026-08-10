"""
Sample N R2_post (or R2 if no post run) aspects per task for open-ended coding.

Writes one JSONL per task to outputs/attr_open_coding/inputs/<task>.jsonl
with rows of {task, aspect_id, name, description}.

Run:
  python scripts/build_open_coding_samples.py --n 100 --seed 11
"""
import argparse
import json
from pathlib import Path
import random

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
R2_POST_DIR = ROOT / "outputs/v2_analysis/r2_post"
R2_FALLBACK_DIR = ROOT / "outputs/analyses/structural_metrics/r2_v1_subagent"
OUT = ROOT / "outputs/attr_open_coding/inputs"
OUT.mkdir(parents=True, exist_ok=True)

# Map task slug -> (kind, path). R2_post is preferred; fall back to R2 v1 subagent.
TASK_SOURCES = {
    "code_review":              ("r2_post", R2_POST_DIR / "code_review/aspects_r2_post.json"),
    "creative_writing":         ("r2_post", R2_POST_DIR / "creative_writing/aspects_r2_post.json"),
    "humor":                    ("r2_post", R2_POST_DIR / "humor/aspects_r2_post.json"),
    "math":                     ("r2_post", R2_POST_DIR / "math/aspects_r2_post.json"),
    "news_homepages":           ("r2_post", R2_POST_DIR / "news_homepages/aspects_r2_post.json"),
    "notice_and_comment":       ("r2_post", R2_POST_DIR / "notice_and_comment/aspects_r2_post.json"),
    "patents":                  ("r2_post", R2_POST_DIR / "patents/aspects_r2_post.json"),
    "peer_review":              ("r2_post", R2_POST_DIR / "peer_review/aspects_r2_post.json"),
    "press_releases":           ("r2_post", R2_POST_DIR / "press_releases/aspects_r2_post.json"),
    # 2 tasks have no R2_post — use R2 v1 subagent directly
    "legal-outcome-prediction": ("r2_v1",   R2_FALLBACK_DIR / "r2_aspects_legal-outcome-prediction.json"),
    "grant-funding":            ("r2_v1",   R2_FALLBACK_DIR / "r2_aspects_grant-funding.json"),
}


def load_aspects(task: str, kind: str, p: Path):
    raw = json.loads(p.read_text())
    if kind == "r2_post":
        # list of {aspect_id, name, description, n_r1_total, r1_metric_ids}
        return [{"aspect_id": a["aspect_id"],
                 "name": a["name"],
                 "description": a.get("description", "")} for a in raw]
    if kind == "r2_v1":
        # {meta..., "aspects": [{name, description, source_family_ids, ...}]}
        out = []
        for i, a in enumerate(raw.get("aspects", [])):
            out.append({"aspect_id": f"r2v1_{i}",
                        "name": a.get("name", ""),
                        "description": a.get("description", "")})
        return out
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    totals = {}
    for task, (kind, p) in TASK_SOURCES.items():
        if not p.exists():
            print(f"!! missing {task}: {p}")
            continue
        rows = load_aspects(task, kind, p)
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        picks = rows[: args.n]
        out_path = OUT / f"{task}.jsonl"
        with open(out_path, "w") as f:
            for r in picks:
                f.write(json.dumps({"task": task, **r}, ensure_ascii=False) + "\n")
        totals[task] = (kind, len(rows), len(picks), str(out_path))
        print(f"  {task:30s} [{kind:7s}] {len(rows):>4} aspects total -> "
              f"sampled {len(picks)} -> {out_path.name}")

    grand = sum(v[2] for v in totals.values())
    print(f"\nwrote samples for {len(totals)} tasks, total {grand} aspects.")


if __name__ == "__main__":
    main()
