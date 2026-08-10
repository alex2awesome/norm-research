"""Per-task cross-encoder training on the v6 judge's pair labels.

For each task, fine-tune a cross-encoder (ModernBERT-base) on the task's
TRAIN pairs, regressing the judge score (0/1/2 -> 0/0.5/1.0). The result is a
fast, distilled v6-judge: it re-ranks the bi-encoder's kNN candidate pairs in
the clustering pipeline.

Output: ce_models/<task>/  (a saved CrossEncoder per task)
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"
os.environ["HF_HOME"] = "/lfs/skampere3/0/alexspan/hf_cache"
os.environ["HF_MODULES_CACHE"] = "/lfs/skampere3/0/alexspan/hf_cache/modules"
os.environ["XDG_CACHE_HOME"] = "/lfs/skampere3/0/alexspan/.cache"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from collections import defaultdict
from pathlib import Path

WORK = Path("/lfs/skampere3/0/alexspan/norm_embed")
VERDICTS = WORK / "all_verdicts.jsonl"
CE_DIR = WORK / "ce_models"
CE_BASE_DIR = ("/lfs/skampere3/0/shared_hf_cache/"
               "models--answerdotai--ModernBERT-base/snapshots")

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    tasks = args.only.split(",") if args.only else TASKS
    CE_DIR.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    ce_base = CE_BASE_DIR + "/" + sorted(os.listdir(CE_BASE_DIR))[0]

    train = defaultdict(list)
    for line in VERDICTS.open():
        v = json.loads(line)
        if v.get("split") == "train" and v.get("score") in (0, 1, 2):
            train[v["task"]].append(v)

    for task in tasks:
        pairs = train.get(task, [])
        if len(pairs) < 200:
            print(f"  {task}: only {len(pairs)} pairs -- skipped", flush=True)
            continue
        print(f"\n=== {task}: training cross-encoder on {len(pairs)} pairs ===",
              flush=True)
        examples = [InputExample(texts=[p["canonical_a"], p["canonical_b"]],
                                 label=p["score"] / 2.0) for p in pairs]
        loader = DataLoader(examples, shuffle=True, batch_size=64)
        ce = CrossEncoder(ce_base, num_labels=1, max_length=256, device="cuda")
        ce.fit(train_dataloader=loader, epochs=3,
               warmup_steps=int(0.1 * len(loader)), show_progress_bar=False)
        ce.save(str(CE_DIR / task))
        print(f"  saved cross-encoder -> {CE_DIR / task}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
