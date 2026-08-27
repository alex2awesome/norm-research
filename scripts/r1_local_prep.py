"""Local data prep for the R1-via-subagents pipeline.

For a given task: load forms + clusters + locally-cached base-bge embeddings,
build cover-once batches (v4a algorithm = batch_size=40, anchor + top-(B-1)
NN), then write each batch as a complete (system, few-shot, user) message
record so the subagent invocation pattern is just "respond as the assistant
should respond to this".

Inputs (all local):
  outputs/analyses/canon_all_real_forms.jsonl
  outputs/analyses/structural_metrics/clusters_<task>.json
  notebooks/_explore_cache/bge/emb_bge_<bucket>_<task>.npy

Output:
  /tmp/r1_subagent/<task>/batches.jsonl  -- one JSON per line per batch:
      {batch_idx, cluster_ids, messages: {system, few_shots, user}}
  /tmp/r1_subagent/<task>/clusters_repr.json  -- {cluster_id: rep_text}
                                                   for downstream aggregation
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Reuse the locked Tight-R1 prompts + per-task info from sk3_build_r1.py
sys.path.insert(0, str(Path(__file__).parent))
from sk3_build_r1 import (TASK_INFO, build_messages, cluster_data, load_task,
                          make_batches, make_system, fewshot_messages)

ROOT = Path(__file__).resolve().parent.parent
FORMS_PATH = ROOT / "outputs" / "analyses" / "canon_all_real_forms.jsonl"
CLUSTERS_DIR = ROOT / "outputs" / "analyses" / "structural_metrics"
LOCAL_EMB = ROOT / "notebooks" / "_explore_cache" / "bge"


# Monkey-patch load_task's EMB path to point at the local cache.
import sk3_build_r1
sk3_build_r1.EMB = LOCAL_EMB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--output-dir", default="/tmp/r1_subagent")
    args = ap.parse_args()

    out = Path(args.output_dir) / args.task
    out.mkdir(exist_ok=True, parents=True)

    # forms for this task
    forms = [json.loads(l) for l in FORMS_PATH.open()
             if json.loads(l)["task"] == args.task]

    rows, emb = load_task(args.task, forms)
    if rows is None:
        print(f"ERR: {args.task} embeddings missing/mismatched")
        return
    cl = json.loads((CLUSTERS_DIR / f"clusters_{args.task}.json").read_text())
    reps, centroids, members = cluster_data(rows, emb, cl)
    cids = list(reps.keys())
    batches, _ = make_batches(cids, centroids, args.batch_size)
    print(f"{args.task}: {len(cids)} clusters -> {len(batches)} batches "
          f"(batch_size={args.batch_size})")

    # Save cluster representative texts for downstream aggregation
    (out / "clusters_repr.json").write_text(json.dumps(
        {str(c): reps[c] for c in cids}, indent=1))

    # For each batch, materialise the full chat messages
    with (out / "batches.jsonl").open("w") as f:
        for bi, batch in enumerate(batches):
            msgs = build_messages(args.task, batch, reps, members,
                                  step_b=False)
            # msgs is [system, fewshot_user, fewshot_asst, user]
            rec = {
                "batch_idx": bi,
                "cluster_ids": [int(c) for c in batch],
                "system": msgs[0]["content"],
                "few_shot_user": msgs[1]["content"],
                "few_shot_assistant": msgs[2]["content"],
                "user": msgs[3]["content"],
            }
            f.write(json.dumps(rec) + "\n")

    print(f"wrote {len(batches)} batches -> {out}/batches.jsonl")


if __name__ == "__main__":
    main()
