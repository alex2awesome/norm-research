"""Text-level P/R cross-firing analysis (task #123, user 2026-07-05).

Silver assigns multiple metrics per TEXT (CE top-k per norm, gold aspects). We score every
cert-metric's M_omega rubric (name: description) over a sample of silver DOCS -> verdict matrix
V[doc, metric], and compare against the silver assignment matrix A[doc, metric]:

  1. own-AUC per metric: does V[:,m] retrieve the docs silver assigned to m?
  2. SPECIFICITY: rank of own-AUC among AUC(V[:,m], A[:,j]) over all j — does a metric fire on
     ITS texts or generically on everyone's?
  3. micro P/R@k per doc: top-k metrics by V vs the silver set (the user's "how often do we get
     those AND others").
  4. MI link: Spearman corr of OPT_Omega / T with own-AUC and with specificity.

Evaluate-only: silver is an external reference, never an optimization signal.
"""
from __future__ import annotations

import argparse, json, re
import numpy as np

from . import alpha_probe as ap
from .silver_validation import load_matches, load_cert, load_r2_index, gold_salience
from .. import config as cfgmod


def nn(s): return re.sub(r"\s+", " ", s.strip().lower())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--bge", required=True)          # bge corpus name (humor, code_review, ...)
    p.add_argument("--task", required=True)         # config/manifest task (humor, code-review, ...)
    p.add_argument("--hier", required=True)         # <task>_general_r2_expanded.json
    p.add_argument("--catalog", required=True)
    p.add_argument("--matches", required=True)      # matches_joined_<bge>.jsonl
    p.add_argument("--cert", required=True)
    p.add_argument("--gold", default=None)
    p.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--n-docs", type=int, default=400)
    p.add_argument("--topk-assign", type=int, default=3)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    idx = load_r2_index(a.catalog, a.hier)
    cert = load_cert(a.cert)
    # metric -> rubric (name: description) for every cert-scored metric with a joined description
    name2desc = {}
    for aid, rec in idx.items():
        if rec["name"] in cert and rec.get("description"):
            name2desc.setdefault(rec["name"], rec["description"])
    metrics = sorted(name2desc)
    a2name = {aid: idx[aid]["name"] for aid in idx}

    # ---- assignment matrix from matches_joined: per-doc norm-count assigned to each metric ----
    matches = load_matches(a.matches)
    per_doc = {}
    for m in matches:
        d = per_doc.setdefault(m["doc"], {})
        for aid in m["top10"][: a.topk_assign]:
            nm = a2name.get(aid)
            if nm in name2desc:
                d[nm] = d.get(nm, 0) + 1
    docs = [d for d, mm in per_doc.items() if len(mm) >= 2 and isinstance(d, str) and len(d) > 100]
    rng = np.random.default_rng(0)
    docs = list(rng.choice(docs, size=min(a.n_docs, len(docs)), replace=False))
    A = np.zeros((len(docs), len(metrics)))
    for i, d in enumerate(docs):
        for nm, c in per_doc[d].items():
            A[i, metrics.index(nm)] = c
    Ab = (A > 0).astype(float)
    print(f"[pr] {a.bge}: {len(docs)} docs x {len(metrics)} metrics; "
          f"mean assignments/doc = {Ab.sum(1).mean():.1f}")

    # ---- score V[doc, metric] ----
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    cfg0 = cfgmod.ImplementerConfig()
    from ..vllm_backend import make_judge_backend
    executor = make_judge_backend(a.target_model, cfg0, temperature=None)
    V = np.zeros((len(docs), len(metrics)))
    for j, nm in enumerate(metrics):
        rubric = f"{nm}: {name2desc[nm]}"
        V[:, j] = ap.signature(executor, rubric, docs, cfg.max_text_chars,
                               template=ap._YESNO_TEXTFIRST)
        if (j + 1) % 25 == 0:
            print(f"[pr] scored {j+1}/{len(metrics)} metrics", flush=True)
    np.savez(a.out.replace(".json", "_VA.npz"), V=V, A=A, metrics=np.array(metrics, dtype=object),
             docs_hash=np.array([hash(d) for d in docs]))

    # ---- analysis (CPU) ----
    from scipy.stats import spearmanr

    def auc(scores, labels):
        ok = np.isfinite(scores)
        s, l = scores[ok], labels[ok]
        if l.sum() < 3 or l.sum() > len(l) - 3:
            return np.nan
        r = np.argsort(np.argsort(s)) + 1
        return float((r[l > 0].mean() - (l.sum() + 1) / 2) / (len(l) - l.sum()))

    own_auc, spec_rank = {}, {}
    for j, nm in enumerate(metrics):
        aucs = np.array([auc(V[:, j], Ab[:, jj]) for jj in range(len(metrics))])
        if np.isnan(aucs[j]):
            continue
        own_auc[nm] = float(aucs[j])
        spec_rank[nm] = float(np.nanmean(aucs[j] >= aucs))     # 1.0 = own is the best of all
    # micro P/R@k
    prk = {}
    for k in (3, 5, 10):
        tp = fp = fn = 0
        for i in range(len(docs)):
            pred = set(np.argsort(-np.nan_to_num(V[i]))[:k])
            gold_set = set(np.where(Ab[i] > 0)[0])
            tp += len(pred & gold_set); fp += len(pred - gold_set); fn += len(gold_set - pred)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        prk[k] = {"P": prec, "R": rec, "F1": 2 * prec * rec / max(prec + rec, 1e-9)}
    # MI link
    opt = np.array([cert[nm]["opt"] for nm in own_auc])
    hm = np.array([cert[nm]["hm"] for nm in own_auc])
    oa = np.array(list(own_auc.values())); sr = np.array([spec_rank[nm] for nm in own_auc])
    out = {"bge": a.bge, "n_docs": len(docs), "n_metrics": len(metrics),
           "n_scored_auc": len(own_auc),
           "own_auc_mean": float(np.nanmean(oa)), "own_auc_q": list(np.nanpercentile(oa, [10, 50, 90])),
           "specificity_mean": float(np.nanmean(sr)),
           "frac_own_best_decile": float(np.mean(sr >= 0.9)),
           "micro_PR": prk,
           "corr_OPT_ownAUC": float(spearmanr(opt, oa).correlation),
           "corr_T_ownAUC": float(spearmanr(hm, oa).correlation),
           "corr_OPT_specificity": float(spearmanr(opt, sr).correlation),
           "per_metric": {nm: {"own_auc": own_auc[nm], "spec": spec_rank[nm],
                               "opt": cert[nm]["opt"]} for nm in own_auc}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[pr] {a.bge}: ownAUC mean={out['own_auc_mean']:.3f} spec={out['specificity_mean']:.3f} "
          f"P@5={prk[5]['P']:.2f} R@5={prk[5]['R']:.2f} corr(OPT,ownAUC)={out['corr_OPT_ownAUC']:+.3f}")


if __name__ == "__main__":
    main()
