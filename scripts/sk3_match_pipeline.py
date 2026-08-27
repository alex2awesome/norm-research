"""CE re-rank matching pipeline.

LoRA-bge kNN candidates -> per-task ModernBERT cross-encoder re-rank ->
hybrid affinity -> complete-linkage clustering.

For each task:
  1. LoRA-bge embeddings (pooled across buckets) give a cosine matrix.
  2. Candidate pairs = full n^2 if the task is small (n <= --full-below),
     else the top-k kNN of each rubric (broad net, --knn).
  3. The per-task CE re-scores every candidate pair -> sameness in [0,1].
  4. Hybrid distance D: 1-cos everywhere, overwritten by 1-CE on candidates.
  5. Complete-linkage clustering on D; tau swept.

Reports the held-out FP/FN frontier for CE-rerank vs LoRA-bi-encoder-only on
the same eval pairs and same linkage code, plus candidate coverage of the
true-same eval pairs (i.e. was the candidate net broad enough for the CE to
even have a chance at those merges).
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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

WORK = Path("/lfs/skampere3/0/alexspan/norm_embed")
FORMS = WORK / "canon_all_real_forms.jsonl"
VERDICTS = WORK / "all_verdicts.jsonl"
EMB = WORK / "out"
CE_DIR = WORK / "ce_models"
MATCH_OUT = WORK / "match_out"

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]
TAUS = [round(0.50 + 0.02 * i, 2) for i in range(25)]  # 0.50 .. 0.98
FN_TARGETS = [0.08, 0.10, 0.12, 0.14, 0.16]
METHODS = ("lora", "ce", "blend")
BLEND_ALPHA = 0.5  # weight on the CE score in the blend affinity


def load_task(task, by_bucket):
    """Pool a task's bucket embeddings (sorted-bucket order, idx within);
    return the global rows list and the L2-normalised LoRA embedding."""
    rows, mats = [], []
    for bucket in sorted(by_bucket):
        rs = sorted(by_bucket[bucket], key=lambda r: r["idx"])
        p = EMB / f"emb_lora_{bucket}_{task}.npy"
        if not p.exists():
            return None, None
        e = np.load(p).astype(np.float32)
        if len(e) != len(rs):
            return None, None
        rows += rs
        mats.append(e)
    emb = np.vstack(mats)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return rows, emb


def candidates(cos, n, knn, full_below):
    """(i<j) candidate pairs: full n^2 if small, else symmetrised top-k kNN."""
    if n <= full_below:
        iu = np.triu_indices(n, k=1)
        return iu[0].astype(np.int64), iu[1].astype(np.int64), "full"
    k = min(knn, n - 1)
    cs = cos.copy()
    np.fill_diagonal(cs, -2.0)
    nn = np.argpartition(-cs, k, axis=1)[:, :k]
    cset = set()
    for i in range(n):
        for j in nn[i]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a != b:
                cset.add((a, b))
    arr = np.array(sorted(cset), dtype=np.int64)
    return arr[:, 0], arr[:, 1], f"knn{k}"


def linkages(cos, ii, jj, ce_scores):
    """Complete-linkage dendrograms, one per method.

    lora  -- 1-cos everywhere (bi-encoder only).
    ce    -- 1-CE on candidate pairs, 1-cos elsewhere.
    blend -- 1-(a*CE+(1-a)*cos) on candidates, 1-cos elsewhere.
    """
    base = np.clip(1.0 - cos, 0.0, None).astype(np.float64)
    np.fill_diagonal(base, 0.0)
    out = {"lora": linkage(squareform(base, checks=False), method="complete")}

    cand_cos = cos[ii, jj].astype(np.float64)
    affinities = {
        "ce": ce_scores,
        "blend": BLEND_ALPHA * ce_scores + (1 - BLEND_ALPHA) * cand_cos,
    }
    for name, aff in affinities.items():
        D = base.copy()
        d = np.clip(1.0 - aff, 0.0, None)
        D[ii, jj] = d
        D[jj, ii] = d
        out[name] = linkage(squareform(D, checks=False), method="complete")
        del D
    return out


def fpfn(Z, keymap, pairs):
    """Per-tau (FP, FN) on this task's eval pairs."""
    out = {}
    for tau in TAUS:
        lab = fcluster(Z, t=1.0 - tau, criterion="distance")
        merged = bad = split = miss = 0
        for v in pairs:
            ia, ib = keymap.get(v["key_a"]), keymap.get(v["key_b"])
            if ia is None or ib is None:
                continue
            if lab[ia] == lab[ib]:
                merged += 1
                bad += v["score"] != 2
            else:
                split += 1
                miss += v["score"] == 2
        out[tau] = (merged, bad, split, miss)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn", type=int, default=200)
    ap.add_argument("--full-below", type=int, default=4200)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    tasks = args.only.split(",") if args.only else TASKS
    MATCH_OUT.mkdir(exist_ok=True)

    from sentence_transformers import CrossEncoder

    by_task = defaultdict(lambda: defaultdict(list))
    for line in FORMS.open():
        r = json.loads(line)
        by_task[r["task"]][r["bucket"]].append(r)

    ev = defaultdict(list)
    for line in VERDICTS.open():
        v = json.loads(line)
        if v.get("split") == "eval" and v.get("score") in (0, 1, 2):
            ev[v["task"]].append(v)

    # accumulate eval-pair tallies summed across tasks, per method per tau
    acc = {m: {tau: [0, 0, 0, 0] for tau in TAUS} for m in METHODS}
    cov_hit = cov_tot = 0

    for task in tasks:
        rows, emb = load_task(task, by_task[task])
        if rows is None:
            print(f"  {task}: embeddings missing -- skipped", flush=True)
            continue
        n = len(rows)
        keymap = {r["key"]: g for g, r in enumerate(rows)}
        cos = (emb @ emb.T).astype(np.float32)
        ii, jj, mode = candidates(cos, n, args.knn, args.full_below)
        print(f"\n=== {task}: n={n}  {mode}  {len(ii):,} candidate pairs ===",
              flush=True)

        # rubric texts are short -- max_length 96 + a large batch saturate
        # the GPU (batch_size 512 left a B200 at <7/183 GB)
        ce = CrossEncoder(str(CE_DIR / task), device="cuda", max_length=96)
        texts = [[rows[int(a)]["canonical"] or " ",
                  rows[int(b)]["canonical"] or " "] for a, b in zip(ii, jj)]
        ce_scores = np.asarray(
            ce.predict(texts, batch_size=8192, show_progress_bar=False),
            dtype=np.float64)
        print(f"  CE scored {len(ce_scores):,} pairs "
              f"(mean {ce_scores.mean():.3f})", flush=True)

        Zs = linkages(cos, ii, jj, ce_scores)
        np.save(MATCH_OUT / f"Z_ce_{task}.npy", Zs["ce"])
        np.save(MATCH_OUT / f"Z_blend_{task}.npy", Zs["blend"])
        np.savez(MATCH_OUT / f"scored_{task}.npz",
                 ii=ii, jj=jj, ce=ce_scores)
        (MATCH_OUT / f"keys_{task}.json").write_text(
            json.dumps([r["key"] for r in rows]))

        # candidate coverage of true-same eval pairs
        candset = set(zip(ii.tolist(), jj.tolist()))
        for v in ev.get(task, []):
            if v["score"] != 2:
                continue
            a, b = keymap.get(v["key_a"]), keymap.get(v["key_b"])
            if a is None or b is None:
                continue
            cov_tot += 1
            lo, hi = (a, b) if a < b else (b, a)
            cov_hit += (lo, hi) in candset

        for m in METHODS:
            for tau, (mg, bad, sp, miss) in fpfn(Zs[m], keymap,
                                                 ev.get(task, [])).items():
                acc[m][tau][0] += mg
                acc[m][tau][1] += bad
                acc[m][tau][2] += sp
                acc[m][tau][3] += miss
        print(f"  done -> Z_{{ce,blend}}_{task}.npy", flush=True)

    def rate(method, tau):
        mg, bad, sp, miss = acc[method][tau]
        fp = bad / mg * 100 if mg else 0.0
        fn = miss / sp * 100 if sp else 0.0
        return fp, fn

    def cell(method, tau):
        fp, fn = rate(method, tau)
        return f"{fp:>5.1f}/{fn:<5.1f}"

    print(f"\n{'=' * 70}")
    print("HELD-OUT FP / FN (%)  -  lora bi-encoder | ce re-rank | blend")
    print(f"{'=' * 70}")
    print(f"{'tau':>6} | {'lora FP/FN':>11} | {'ce FP/FN':>11} | "
          f"{'blend FP/FN':>11}")
    for tau in TAUS:
        print(f"{tau:>6.2f} | {cell('lora', tau):>11} | "
              f"{cell('ce', tau):>11} | {cell('blend', tau):>11}")

    print(f"\nfrontier -- min FP achievable at each FN ceiling:")
    print(f"{'FN<=':>6} | " + " | ".join(f"{m:>8}" for m in METHODS))
    for tgt in FN_TARGETS:
        cells = []
        for m in METHODS:
            fps = [rate(m, t)[0] for t in TAUS if rate(m, t)[1] <= tgt * 100]
            cells.append(f"{min(fps):>7.1f}%" if fps else f"{'--':>8}")
        print(f"{tgt * 100:>5.0f}% | " + " | ".join(cells))

    print(f"\ncandidate coverage of true-same eval pairs: "
          f"{cov_hit}/{cov_tot} = {cov_hit / cov_tot * 100:.1f}%"
          if cov_tot else "\nno true-same eval pairs")


if __name__ == "__main__":
    main()
