"""Per-task LoRA fine-tuning of bge-large on the v6 judge's pair labels.

For each task: train a LoRA adapter on bge-large with CoSENT loss over the
task's TRAIN pairs (judge score 0/1/2 used as the graded target), then embed
that task's canonical forms with the fine-tuned model.

CoSENT is a ranking loss -- it pushes pairs with higher judge scores to higher
cosine -- so the resulting embedding's cosine becomes a calibrated proxy for
the judge's notion of sameness. Hard negatives (score 1) drive the boundary.

Output: emb_lora_<bucket>_<task>.npy  (row order matches canon_all_real_forms)
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

import numpy as np

WORK = Path("/lfs/skampere3/0/alexspan/norm_embed")
VERDICTS = WORK / "all_verdicts.jsonl"
FORMS = WORK / "canon_all_real_forms.jsonl"
OUTD = WORK / "out"
ADAPTERS = WORK / "lora_adapters"
BGE_BASE = "/lfs/skampere3/0/shared_hf_cache/models--BAAI--bge-large-en-v1.5/snapshots"

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated task subset")
    args = ap.parse_args()
    tasks = args.only.split(",") if args.only else TASKS
    OUTD.mkdir(parents=True, exist_ok=True)
    ADAPTERS.mkdir(parents=True, exist_ok=True)
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    from peft import LoraConfig

    bge_path = BGE_BASE + "/" + sorted(os.listdir(BGE_BASE))[0]

    # train pairs by task
    train = defaultdict(list)
    for line in VERDICTS.open():
        v = json.loads(line)
        if v.get("split") == "train" and v.get("score") in (0, 1, 2):
            train[v["task"]].append(v)

    # canonical forms by (bucket, task)
    forms = defaultdict(list)
    for line in FORMS.open():
        r = json.loads(line)
        forms[(r["bucket"], r["task"])].append(r)
    for v in forms.values():
        v.sort(key=lambda r: r["idx"])

    for task in tasks:
        pairs = train.get(task, [])
        if len(pairs) < 200:
            print(f"  {task}: only {len(pairs)} train pairs -- skipped", flush=True)
            continue
        print(f"\n=== {task}: training LoRA on {len(pairs)} pairs ===", flush=True)
        model = SentenceTransformer(bge_path, device="cuda")
        model.add_adapter(LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=["query", "key", "value", "dense"],
            bias="none"))
        examples = [InputExample(texts=[p["canonical_a"], p["canonical_b"]],
                                 label=float(p["score"])) for p in pairs]
        loader = DataLoader(examples, shuffle=True, batch_size=64)
        loss = losses.CoSENTLoss(model)
        model.fit(train_objectives=[(loader, loss)], epochs=4,
                  warmup_steps=int(0.1 * len(loader)),
                  optimizer_params={"lr": 2e-4}, show_progress_bar=False)

        for bucket in ("general", "specific", "hyper_specific"):
            rows = forms.get((bucket, task))
            if not rows:
                continue
            texts = [r["canonical"] or " " for r in rows]
            emb = model.encode(texts, batch_size=256, normalize_embeddings=True,
                               show_progress_bar=False, convert_to_numpy=True)
            np.save(OUTD / f"emb_lora_{bucket}_{task}.npy", emb.astype(np.float32))
            print(f"  embedded {bucket}/{task}: {emb.shape}", flush=True)
        del model
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
