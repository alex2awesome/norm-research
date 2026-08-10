#!/usr/bin/env python3
"""Score the 5 peer-review agentic-hybrid code-metrics over extracted fields.

Loads each metric_seam module (programs_peer_review/<aspect>_h0.py), the Gemma-4-extracted
fields (peer_review_fields.jsonl), builds an Ops TF-IDF retrieval corpus over the same
abstracts (for the tier-2 internal novelty signal), and computes score(text, extracted, ops).

Reports per-aspect univariate AUC vs accept/reject, and saves a code-score matrix to fold
into the V layer. CPU-only (base env w/ sklearn).

Usage: python score_code_metrics.py --fields peer_review_fields.jsonl
"""
import argparse, importlib.util, json, pathlib, sys
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research")
PROG = BASE / "methods/metric_seam/hybrids/programs_peer_review"
sys.path.insert(0, str(BASE / "methods/metric_seam/hybrids"))
from ops import Ops  # noqa: E402

AIDS = ["a163", "a130", "a214", "a25", "a45"]

def load_module(aid):
    spec = importlib.util.spec_from_file_location(f"pr_{aid}", PROG / f"{aid}_h0.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="peer_review_fields.jsonl")
    ap.add_argument("--out", default=str(BASE / "datasets/peer-review/peer_review_code_scores.npz"))
    a = ap.parse_args()

    mods = {aid: load_module(aid) for aid in AIDS}
    rows = [json.loads(l) for l in open(a.fields) if l.strip()]
    ids = [r["id"] for r in rows]
    y = np.array([r["y"] for r in rows], dtype=int)

    # Ops needs a corpus json: [{datapoint_id, text}]; build it for TF-IDF retrieve_similar.
    corpus_path = str(pathlib.Path(a.fields).with_suffix(".corpus.json"))
    json.dump([{"datapoint_id": r["id"], "text": r["text"]} for r in rows], open(corpus_path, "w"))
    ops = Ops(corpus_path=corpus_path)

    X = np.full((len(rows), len(AIDS)), np.nan)
    for j, aid in enumerate(AIDS):
        fn = mods[aid].score
        for i, r in enumerate(rows):
            extracted = (r.get("fields") or {}).get(aid, {})
            try:
                X[i, j] = float(fn(r["text"], extracted, ops))
            except Exception:
                X[i, j] = np.nan

    print(f"[code] {len(rows)} abstracts x {len(AIDS)} code-metrics; na={np.isnan(X).mean():.3f}", flush=True)
    print(f"{'aspect':6s} {'mean':>6s} {'std':>6s} {'AUC':>6s} {'n':>5s}")
    for j, aid in enumerate(AIDS):
        col = X[:, j]; mask = ~np.isnan(col)
        auc = roc_auc_score(y[mask], col[mask]) if mask.sum() > 30 and len(set(y[mask].tolist())) == 2 else float("nan")
        print(f"{aid:6s} {col[mask].mean() if mask.sum() else float('nan'):6.3f} "
              f"{col[mask].std() if mask.sum() else float('nan'):6.3f} {auc:6.3f} {int(mask.sum()):5d}")

    np.savez_compressed(a.out, X=X, y=y, ids=np.array(ids, dtype=object),
                        code_names=np.array(AIDS, dtype=object))
    print(f"[code] saved -> {a.out}", flush=True)
    print("CODE_DONE", flush=True)

if __name__ == "__main__":
    main()
