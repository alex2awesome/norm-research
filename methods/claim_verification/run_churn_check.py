#!/usr/bin/env python3
"""CHURNALISM spot-check: the SS103 delta-checker ported to journalism.
For (press release, coverage article) pairs: does the article add anything beyond the PR?
  TRIVIAL_DELTA     = churnalism (restates/quotes the PR without independent addition)
  SUBSTANTIVE_DELTA = independent reporting (new sources/quotes not in the PR, verification,
                      criticism, added context or consequences)
  NO_OVERLAP        = article is not about this PR
Checks: (a) matched pairs verdict dist, (b) MISMATCHED placebo (should be NO_OVERLAP),
(c) convergent CODE check: verbatim token containment should be higher for TRIVIAL verdicts,
(d) examples. Small n, 8 workers (Gemma shared with the expansion run).
Run on sk3: python -m methods.claim_verification.run_churn_check [--n 200]"""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np, pandas as pd
from claim_verification.core import Cache, _post, _parse_json, _key
from claim_verification.evidence_api import EvidenceAPI, clean_evidence_text
from claim_verification.seam_metrics import _toks
from claim_verification.run_tiered_pr import load_meta, coverage_texts

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma"}

CHURN = """Below are a PRESS RELEASE and a NEWS ARTICLE that may cover it.

PRESS RELEASE:
{pr}

NEWS ARTICLE:
{art}

Judge what the article adds BEYOND the press release:
- "TRIVIAL_DELTA": churnalism — the article restates, lightly rewrites, or quotes the press
  release without independent addition.
- "SUBSTANTIVE_DELTA": independent reporting — the article adds sources or quotes not in the
  press release, verification or skepticism, context, consequences, or original analysis.
- "NO_OVERLAP": the article is not substantially about this press release.

Return ONLY JSON: {{"added": "<one sentence: what the article adds, or 'nothing'>", "verdict": "TRIVIAL_DELTA"|"SUBSTANTIVE_DELTA"|"NO_OVERLAP"}}"""

def containment(art, pr):
    """CODE: fraction of article content tokens present in the PR (verbatim churn proxy)."""
    at, pt = _toks(art), set(_toks(pr))
    return sum(1 for w in at if w in pt) / max(len(at), 1)

def churn_check(pr, art, cache):
    k = _key("churn", CFG["model"], pr[:200], art[:200])
    hit = cache.get(k)
    if hit is not None: return hit
    raw = _post(CFG["base_url"], CFG["model"],
                CHURN.format(pr=pr[:2200], art=art[:2200]), max_tokens=220)
    obj = _parse_json(raw) or {}
    v = str(obj.get("verdict", "")).upper()
    if v not in ("TRIVIAL_DELTA", "SUBSTANTIVE_DELTA", "NO_OVERLAP"): v = "PARSE_FAIL"
    out = {"verdict": v, "added": str(obj.get("added", ""))[:200]}
    cache.put(k, out)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    d = pd.read_parquet(f"{ROOT}/datasets/press-releases/press_release_deconfounded.parquet")
    pr_text = {str(r.id): str(r.text) for r in d.itertuples()}
    _, nout, pr2art = load_meta()
    api = EvidenceAPI()
    pairs = []
    rng = np.random.default_rng(0)
    pids = [p for p in pr2art if p in pr_text]
    rng.shuffle(pids)
    for pid in pids:
        arts = coverage_texts(pr2art[pid][:1], api)
        if arts and len(arts[0]) > 600:
            pairs.append((pid, pr_text[pid], arts[0], nout.get(pid, 0)))
        if len(pairs) >= args.n: break
    print(f"[churn] {len(pairs)} matched (PR, article) pairs", flush=True)
    cache = Cache(f"{ROOT}/outputs/churn_check/cache.jsonl")
    os.makedirs(f"{ROOT}/outputs/churn_check", exist_ok=True)
    half = len(pairs) // 2
    tasks = [("matched", pid, pr, art, k_) for pid, pr, art, k_ in pairs]
    tasks += [("mismatched", pairs[i][0], pairs[(i + half) % len(pairs)][1], pairs[i][2],
               pairs[i][3]) for i in range(0, len(pairs), 2)]
    lock = Lock(); rows = []
    def work(t):
        arm, pid, pr, art, k_ = t
        prc, artc = clean_evidence_text(pr), clean_evidence_text(art)
        try: r = churn_check(prc, artc, cache)
        except Exception: return
        with lock:
            rows.append({"arm": arm, "pr_id": pid, "n_outlets": k_,
                         "containment": round(containment(artc[:2200], prc[:2200]), 3), **r})
            if len(rows) % 50 == 0: print(f"[churn] {len(rows)}/{len(tasks)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, tasks))
    F = pd.DataFrame(rows)
    for arm, g in F.groupby("arm"):
        print(f"  {arm:10} n={len(g):3d} "
              f"{g.verdict.value_counts(normalize=True).round(3).to_dict()}", flush=True)
    m = F[F.arm == "matched"]
    print("\n[churn] convergent CODE check (containment by verdict):", flush=True)
    print(m.groupby("verdict").containment.agg(["mean", "count"]).round(3).to_string(), flush=True)
    # churn vs coverage breadth (n_outlets)
    mv = m[m.verdict.isin(["TRIVIAL_DELTA", "SUBSTANTIVE_DELTA"])]
    if len(mv) > 60:
        from sklearn.metrics import roc_auc_score
        y = (mv.verdict == "SUBSTANTIVE_DELTA").astype(int)
        if y.nunique() == 2 and mv.n_outlets.std() > 0:
            print(f"\n[churn] substantive-vs-trivial ~ n_outlets AUC="
                  f"{roc_auc_score(y, mv.n_outlets):.4f} (does value-added coverage "
                  f"track broader pickup?)", flush=True)
    F.to_csv(f"{ROOT}/outputs/churn_check/results.csv", index=False)
    print("\n[churn] examples:", flush=True)
    for v in ("TRIVIAL_DELTA", "SUBSTANTIVE_DELTA"):
        for _, r in m[m.verdict == v].head(2).iterrows():
            print(f"  [{v}] pr={r.pr_id} cont={r.containment}\n    added: {r.added[:140]}", flush=True)
    print("CHURN_CHECK_DONE", flush=True)

if __name__ == "__main__":
    main()
