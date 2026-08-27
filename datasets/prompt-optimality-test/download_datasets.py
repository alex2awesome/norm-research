"""Fetch the GEPA-standard benchmark datasets into data/ (isolation rule: nothing leaves this
folder). Small fixed dev/val splits are materialized as JSONL so every arm sees byte-identical
data; split membership uses a stable content hash (never a seeded shuffle of a growing list).

  python download_datasets.py            # all three
  python download_datasets.py hotpotqa   # one
"""
import hashlib
import json
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data"
N_TRAIN, N_VAL = 150, 300      # GEPA-paper-scale budgets; adjust in RUNBOOK if needed


def _stable_split(key: str) -> str:
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 40 else "val"


def _dump(name, rows):
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    fhs = {s: open(out / f"{s}.jsonl", "w") for s in counts}
    for r in rows:
        s = _stable_split(r["id"])
        if counts[s] >= (N_TRAIN if s == "train" else N_VAL):
            continue
        fhs[s].write(json.dumps(r) + "\n")
        counts[s] += 1
        if all(counts[k] >= (N_TRAIN if k == "train" else N_VAL) for k in counts):
            break
    for f in fhs.values():
        f.close()
    print(f"{name}: {counts}")


def hotpotqa():
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    _dump("hotpotqa", ({"id": r["id"], "question": r["question"], "answer": r["answer"],
                        "context": r["context"]} for r in ds))


def hover():
    """Class-BALANCED splits: the raw validation set is ordered by label, so naive head-taking
    yields an all-SUPPORTED split (v1 bug — GEPA gamed the base rate with 'always SUPPORTED').
    Cap each (split, label) cell at half its split budget."""
    from datasets import load_dataset
    ds = load_dataset("hover-nlp/hover", split="validation", trust_remote_code=True)
    out = DATA / "hover"
    out.mkdir(parents=True, exist_ok=True)
    caps = {("train", 0): N_TRAIN // 2, ("train", 1): N_TRAIN // 2,
            ("val", 0): N_VAL // 2, ("val", 1): N_VAL // 2}
    counts = {k: 0 for k in caps}
    kept = {"train": [], "val": []}
    for r in ds:
        s = _stable_split(str(r["id"]))
        k = (s, int(r["label"]))
        if counts[k] >= caps[k]:
            continue
        kept[s].append({"id": str(r["id"]), "claim": r["claim"], "label": int(r["label"])})
        counts[k] += 1
        if all(counts[k2] >= caps[k2] for k2 in caps):
            break
    # The raw set is label-ordered, so file ORDER must not follow it: any head-take (val-n caps,
    # dev panels) would be single-class again. Deterministic interleave via id-hash sort.
    for s, rows in kept.items():
        rows.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        with open(out / f"{s}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print(f"hover (balanced, hash-interleaved): {counts}")


def aime():
    from datasets import load_dataset
    ds = load_dataset("MathArena/aime_2025", split="train")
    _dump("aime2025", ({"id": str(i), "problem": r["problem"], "answer": str(r["answer"])}
                       for i, r in enumerate(ds)))


ALL = {"hotpotqa": hotpotqa, "hover": hover, "aime": aime}

if __name__ == "__main__":
    picks = sys.argv[1:] or list(ALL)
    for p in picks:
        ALL[p]()
