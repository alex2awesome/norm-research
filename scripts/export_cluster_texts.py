"""Export rubric-cluster + subtask-description texts to a flat JSONL so they can
be re-embedded with a local model on sk3 (the OpenAI account is out of credits).

Row order per (level, task) matches embedding_diversity.py's load order, so the
sk3-produced .npy arrays line up with the existing emb_*.npy caches.

Output: outputs/analyses/_sk3_embed_input.jsonl  -- {level, task, idx, text}
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
HIER = ROOT / "outputs" / "hierarchy"
OUT = ROOT / "outputs" / "analyses"
PAGES = ROOT / "notebooks" / "_explore_cache" / "pages.parquet"

TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math-stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]


def cluster_texts(task):
    p = HIER / f"{task}_general_r1_refined.json"
    d = json.loads(p.read_text())
    out = []
    for par in d.get("parented_trees", []):
        for ch in par.get("children", []):
            nm = ch.get("medoid_name", "") or ""
            ds = ch.get("medoid_description", "") or ""
            out.append(f"{nm}: {ds}".strip())
    return out


def main():
    df = pd.read_parquet(PAGES)
    df = df[~df.is_error]
    n = 0
    with (OUT / "_sk3_embed_input.jsonl").open("w") as f:
        for task in TASKS:
            for i, t in enumerate(cluster_texts(task)):
                f.write(json.dumps({"level": "rubric_cluster", "task": task,
                                    "idx": i, "text": t}) + "\n")
                n += 1
            sub = df[df.task == task]
            descs = [d for d in sub.subtask_description.fillna("").tolist()
                     if d.strip()]
            for i, t in enumerate(descs):
                f.write(json.dumps({"level": "subtask_desc", "task": task,
                                    "idx": i, "text": t}) + "\n")
                n += 1
    print(f"wrote {n} rows -> {OUT/'_sk3_embed_input.jsonl'}")


if __name__ == "__main__":
    main()
