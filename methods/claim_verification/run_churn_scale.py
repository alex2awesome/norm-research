#!/usr/bin/env python3
"""CHURNALISM AT SCALE: run_churn_check validated the delta-checker at n=200 (placebo
100% NO_OVERLAP, containment monotone .712/.270/.139). This scales to ALL (PR, first
covering article) pairs, with an ~8% mismatched drift-guard placebo, and reads out the
churn->pickup relationship properly. Reuses outputs/churn_check/cache.jsonl (spot-check
calls free). Run on sk3: python -m methods.claim_verification.run_churn_scale"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np, pandas as pd
from claim_verification.core import Cache
from claim_verification.evidence_api import EvidenceAPI, clean_evidence_text
from claim_verification.run_tiered_pr import load_meta, coverage_texts
from claim_verification.run_churn_check import churn_check, containment, ROOT

def main():
    d = pd.read_parquet(f"{ROOT}/datasets/press-releases/press_release_deconfounded.parquet")
    pr_text = {str(r.id): str(r.text) for r in d.itertuples()}
    _, nout, pr2art = load_meta()
    api = EvidenceAPI()
    rng = np.random.default_rng(0)
    pids = [p for p in pr2art if p in pr_text]
    rng.shuffle(pids)
    print(f"[churn-scale] {len(pids)} PRs with coverage mapping", flush=True)
    pairs = []
    for j, pid in enumerate(pids):
        arts = coverage_texts(pr2art[pid][:1], api)
        if arts and len(arts[0]) > 600:
            pairs.append((pid, pr_text[pid], arts[0], nout.get(pid, 0)))
        if j % 500 == 0: print(f"[churn-scale] fetched {j}/{len(pids)}, usable {len(pairs)}", flush=True)
    print(f"[churn-scale] {len(pairs)} usable (PR, article) pairs", flush=True)
    cache = Cache(f"{ROOT}/outputs/churn_check/cache.jsonl")
    os.makedirs(f"{ROOT}/outputs/churn_check", exist_ok=True)
    half = max(1, len(pairs) // 2)
    tasks = [("matched", pid, pr, art, k_) for pid, pr, art, k_ in pairs]
    tasks += [("mismatched", pairs[i][0], pairs[(i + half) % len(pairs)][1], pairs[i][2],
               pairs[i][3]) for i in range(0, len(pairs), 12)]  # ~8% drift guard
    lock = Lock(); rows = []
    def work(t):
        arm, pid, pr, art, k_ = t
        prc, artc = clean_evidence_text(pr), clean_evidence_text(art)
        try: r = churn_check(prc, artc, cache)
        except Exception: return
        with lock:
            rows.append({"arm": arm, "pr_id": pid, "n_outlets": k_,
                         "containment": round(containment(artc[:2200], prc[:2200]), 3), **r})
            if len(rows) % 300 == 0: print(f"[churn-scale] {len(rows)}/{len(tasks)}", flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(work, tasks))
    F = pd.DataFrame(rows)
    F.to_csv(f"{ROOT}/outputs/churn_check/results_scale.csv", index=False)
    for arm, g in F.groupby("arm"):
        print(f"  {arm:10} n={len(g):5d} "
              f"{g.verdict.value_counts(normalize=True).round(3).to_dict()}", flush=True)
    m = F[F.arm == "matched"]
    print("\n[churn-scale] containment by verdict (CODE convergence):", flush=True)
    print(m.groupby("verdict").containment.agg(["mean", "count"]).round(3).to_string(), flush=True)
    mv = m[m.verdict.isin(["TRIVIAL_DELTA", "SUBSTANTIVE_DELTA"])]
    cov = m[m.verdict != "NO_OVERLAP"]
    print(f"\n[churn-scale] churn rate among covering pairs: "
          f"{(cov.verdict == 'TRIVIAL_DELTA').mean():.4f} (n={len(cov)})", flush=True)
    from sklearn.metrics import roc_auc_score
    if len(mv) > 100 and mv.n_outlets.std() > 0:
        y = (mv.verdict == "SUBSTANTIVE_DELTA").astype(int)
        if y.nunique() == 2:
            print(f"[churn-scale] substantive-vs-trivial ~ n_outlets "
                  f"AUC={roc_auc_score(y, mv.n_outlets):.4f} (n={len(mv)}, spot-check .605)", flush=True)
    print("[churn-scale] n_outlets median by verdict:", flush=True)
    print(m.groupby("verdict").n_outlets.agg(["median", "mean", "count"]).round(2).to_string(), flush=True)
    # containment-quartile x verdict agreement (CODE<->LLM convergence at scale)
    m2 = m.copy(); m2["cq"] = pd.qcut(m2.containment, 4, labels=False, duplicates="drop")
    print("\n[churn-scale] TRIVIAL rate by containment quartile:", flush=True)
    print(m2.groupby("cq").verdict.apply(lambda v: (v == "TRIVIAL_DELTA").mean()).round(3).to_string(), flush=True)
    print("CHURN_SCALE_DONE", flush=True)

if __name__ == "__main__":
    main()
