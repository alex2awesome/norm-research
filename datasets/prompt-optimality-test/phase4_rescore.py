"""Phase 4a — re-evaluate every DISTINCT proposed candidate (both arms) on the fixed val split,
recording PER-ITEM correctness vectors. This is what the minibatch proposal log cannot give
(minibatch item identity varies), and it is the input to the value-scaling estimators: candidate ×
item binary matrices. Resumable (skips hashes already in rescore.jsonl); 8-way concurrent.

  source .venv/bin/activate && python phase4_rescore.py
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_official_gepa import EVALS, GLM, _load

HERE = Path(__file__).parent
MAX_CANDIDATES = 40          # per run; drops are LOGGED, never silent
VAL_N = {"hover": 100, "hotpotqa": 100, "aime2025": 17}


def _norm_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:16]


def rescore_run(ds: str, arm: str, task_lm: GLM):
    rundir = HERE / "runs" / ds / arm
    plog = rundir / "proposals.jsonl"
    if not plog.exists():
        print(f"[{ds}|{arm}] no proposals.jsonl — skipped")
        return
    seen, cands = set(), []
    for line in open(plog):
        c = json.loads(line)["candidate"]
        text = next(iter(c.values()))
        h = _norm_hash(text)
        if h not in seen:
            seen.add(h)
            cands.append((h, text))
    dropped = max(0, len(cands) - MAX_CANDIDATES)
    if dropped:
        print(f"[{ds}|{arm}] {len(cands)} distinct candidates; rescoring first "
              f"{MAX_CANDIDATES} in draw order, DROPPING {dropped} (cap)")
    cands = cands[:MAX_CANDIDATES]

    out_path = rundir / "rescore.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["hash"] for l in open(out_path)}
    todo = [(h, t) for h, t in cands if h not in done]
    val = _load(ds, "val", VAL_N[ds])
    evaluator = EVALS[ds]
    print(f"[{ds}|{arm}] rescoring {len(todo)}/{len(cands)} candidates x {len(val)} items",
          flush=True)

    def _one_item(cand_text, data):
        resp = task_lm([{"role": "system", "content": cand_text},
                        {"role": "user", "content": data["input"]}])
        return evaluator(data, resp).score

    for h, text in todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            scores = list(ex.map(lambda d: _one_item(text, d), val))
        with open(out_path, "a") as fh:
            fh.write(json.dumps({"hash": h, "candidate": text, "item_scores": scores,
                                 "val_mean": sum(scores) / len(scores)}) + "\n")
        print(f"  {h} val_mean={sum(scores)/len(scores):.3f}", flush=True)


def main():
    task_lm = GLM("glm-4.7", max_tokens=2048, temperature=0.2)
    for ds in ("hover", "hotpotqa", "aime2025"):
        for arm in ("official", "inhouse", "unitrecomb"):
            rescore_run(ds, arm, task_lm)
    print(f"RESCORE_DONE total task-lm calls {task_lm.n_calls}")


if __name__ == "__main__":
    main()
