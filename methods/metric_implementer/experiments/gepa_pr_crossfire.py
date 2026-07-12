"""GEPA-pipeline text-level P/R cross-firing (task #123, rebuilt on GEPA+Gemma 2026-07-05).

The old bge_pertask `signals_<task>.jsonl` index could not be traced to item text (source corpus was
replaced). The GEPA+Gemma pipeline makes that moot: `input.jsonl` (items) and the Gemma norm extraction
(`deploy_round1_full.jsonl` / `anchors_best_full.jsonl`) are BOTH keyed by `unit_id`, so item<->norm is
native. This driver builds the silver assignment there and runs the cross-firing analysis.

  A[item, metric]  silver: each item's Gemma norms matched to R2 metrics by the finetuned BGE bi-encoder.
  V[item, metric]  ours  : each certified metric's M_omega rubric scored (P(YES), Llama-8B) over the SAME item.

Questions (the user's "how often do we get the assigned metrics AND others"):
  1. own-AUC per metric   : does V[:,m] retrieve the items silver assigned to m?
  2. specificity rank      : rank of own-AUC among AUC(V[:,m], A[:,j]) over all j (fires on ITS items or generically?)
  3. micro P/R@k per item  : top-k metrics by V vs the silver set.
  4. label-free link       : Spearman of retrieval sharpness (own-AUC, specificity) with OPT_Omega (cert)
                             AND with the CUF unit-level census (n_units, dead-weight) built today.

Two phases, run as SEPARATE processes (GPU-mem safety: ST bi-encoder vs vLLM cannot share one process):
  assign : python -m ...gepa_pr_crossfire assign  --task code_review --mt code-review ... -> pr_assign_<task>.npz
  score  : python -m ...gepa_pr_crossfire score   --assign pr_assign_<task>.npz --mt code-review ... -> pr_<task>.json

Evaluate-only: silver is an external reference, never an optimization signal (reconstruction-only paradigm).
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np


# ------------------------------------------------------------------------------------------------
# shared loaders
# ------------------------------------------------------------------------------------------------
def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def load_metrics(hier_path):
    """r2_expanded merged_groups -> ordered [(merged_name, 'name. description')]."""
    mg = json.load(open(hier_path))["merged_groups"]
    if isinstance(mg, dict):
        mg = list(mg.values())
    out = []
    for g in mg:
        nm = g.get("merged_name") or ""
        desc = g.get("merged_description") or ""
        if nm:
            out.append((nm, f"{nm}. {desc}".strip()))
    return out


def load_items(input_path):
    """input.jsonl {unit_id, text} -> dict."""
    d = {}
    for l in open(input_path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        uid, tx = r.get("unit_id"), r.get("text")
        if uid and tx:
            d[uid] = tx
    return d


def load_norms(norms_path):
    """deploy_round1_full ({unit_id, signals:[{signal_text}]}) OR anchors_best_full
    ({unit_id, signal_text}) -> dict unit_id -> [signal_text]. Auto-detected per line."""
    d = {}
    for l in open(norms_path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        uid = r.get("unit_id")
        if not uid:
            continue
        bucket = d.setdefault(uid, [])
        if "signals" in r and isinstance(r["signals"], list):        # deploy schema
            for s in r["signals"]:
                t = (s or {}).get("signal_text")
                if t:
                    bucket.append(t)
        elif r.get("signal_text"):                                   # anchors schema
            bucket.append(r["signal_text"])
    return d


def load_cert(cert_path):
    """cert rows -> {merged_name: {opt, hm, g1}}."""
    rows = json.load(open(cert_path))
    out = {}
    for r in rows:
        if not r.get("gains"):
            continue
        out[r["name"]] = {"opt": float(r["opt_omega_bits"]), "hm": float(r["H_M"]),
                          "g1": float(r["gains"][0])}
    return out


def load_cuf(cuf_path):
    """CUF bank_units.jsonl -> {metric: {n_units, dead_frac, n_spans}}. Schema (run_unit_certificate
    --bank-r2): one line per metric = {metric, k, rows:[{node_id, level, span, verdict, atom,
    detect_free, detect_M, delta_free, delta_M, ...}], meta}. A span is a certified UNIT iff detected in
    either arm (detect_free|detect_M) or its verdict starts CERTIFIED / is UNIT(-IN-COMPANY). Nested
    sub-spans (level 2+) are the same address refined, so n_units counts the shallowest level only
    (avoids the known cosmetic dup-span inflation)."""
    if not cuf_path or not os.path.exists(cuf_path):
        return {}
    from collections import defaultdict
    agg = defaultdict(list)
    for l in open(cuf_path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        nm = r.get("metric") or r.get("name") or r.get("merged_name")
        if nm:
            agg[nm].extend(r.get("rows") or r.get("units") or [])

    def _vd(rw):
        return (rw.get("verdict", "") or rw.get("company_verdict", "") or "").upper()

    def _cert(rw):
        return bool(rw.get("detect_free") or rw.get("detect_M")) or \
            _vd(rw).startswith("CERTIFIED") or "UNIT" in _vd(rw) or _vd(rw) == "COMPOSITE"

    out = {}
    for nm, rows in agg.items():
        if not rows:
            out[nm] = {"n_units": 0.0, "dead_frac": 0.0, "n_spans": 0.0}
            continue
        lv = min((rw.get("level", 1) for rw in rows), default=1)
        prim = [rw for rw in rows if rw.get("level", 1) == lv] or rows
        n_cert = sum(1 for rw in prim if _cert(rw))
        n_dead = sum(1 for rw in prim if not _cert(rw) or "DEAD" in _vd(rw))
        out[nm] = {"n_units": float(n_cert), "dead_frac": float(n_dead) / len(prim),
                   "n_spans": float(len(prim))}
    return out


# ------------------------------------------------------------------------------------------------
# phase 1 : assign  (BGE bi-encoder -> A matrix)
# ------------------------------------------------------------------------------------------------
def phase_assign(a):
    metrics = load_metrics(a.hier)
    names = [m[0] for m in metrics]
    metric_texts = [m[1] for m in metrics]
    items = load_items(a.input)
    norms = load_norms(a.norms)

    # items with >= min_norms usable norms AND text, deterministic sample
    cand = [(uid, items[uid], norms[uid]) for uid in norms
            if uid in items and len(norms[uid]) >= a.min_norms and len(items[uid]) > 80]
    cand.sort(key=lambda x: x[0])                                    # stable order
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(cand))[: a.n_docs]
    sample = [cand[i] for i in sorted(idx)]
    uids = [s[0] for s in sample]
    texts = [s[1] for s in sample]
    print(f"[assign] {a.task}: {len(sample)} items x {len(names)} metrics "
          f"(pool={len(cand)}, mean norms/item={np.mean([len(s[2]) for s in sample]):.1f})", flush=True)

    from sentence_transformers import SentenceTransformer
    dev = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    model = SentenceTransformer(a.bge_model, device=dev)
    M = model.encode(metric_texts, normalize_embeddings=True, batch_size=64,
                     show_progress_bar=False, convert_to_numpy=True)
    all_sigs, owner = [], []
    for i, (_uid, _tx, sigs) in enumerate(sample):
        for s in sigs:
            all_sigs.append(s)
            owner.append(i)
    S = model.encode(all_sigs, normalize_embeddings=True, batch_size=128,
                     show_progress_bar=False, convert_to_numpy=True)
    sims = S @ M.T                                                   # (n_sig, n_metric) cosine

    if getattr(a, "ce_model", None):
        # canonical cascade: BGE retrieves top ce_cand candidates, the trained cross-encoder reranks.
        # Sharper than bi-encoder cosine (kills the generic-attractor bias) at the cost of an 8B pass.
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(a.ce_model, device=dev)
        cand = np.argsort(-sims, axis=1)[:, : a.ce_cand]            # (n_sig, ce_cand)
        pairs, flat = [], []
        for si in range(len(all_sigs)):
            for j in cand[si]:
                pairs.append((all_sigs[si], metric_texts[int(j)])); flat.append((si, int(j)))
        scores = ce.predict(pairs, batch_size=256, show_progress_bar=False)
        ce_sims = np.full_like(sims, -1e9)
        for (si, j), sc in zip(flat, scores):
            ce_sims[si, j] = float(sc)
        sims = ce_sims
        print(f"[assign] CE-reranked {len(pairs)} (sig,metric) pairs over top-{a.ce_cand} BGE cands",
              flush=True)
    topk = np.argsort(-sims, axis=1)[:, : a.topk_assign]

    A = np.zeros((len(sample), len(names)))
    for si, i in enumerate(owner):
        for j in topk[si]:
            A[i, j] += 1.0
    print(f"[assign] mean silver metrics/item = {(A > 0).sum(1).mean():.2f}; "
          f"items with >=2 = {int(((A > 0).sum(1) >= 2).sum())}", flush=True)

    # face-validity spot check: first 3 items, their top silver metric + a norm
    for i in range(min(3, len(sample))):
        top = names[int(np.argmax(A[i]))] if A[i].max() > 0 else "(none)"
        print(f"[assign]   item {uids[i]}: top-silver='{top}' | norm0='{sample[i][2][0][:70]}'", flush=True)

    np.savez(a.out, A=A, names=np.array(names, dtype=object), uids=np.array(uids, dtype=object),
             texts=np.array(texts, dtype=object))
    print(f"[assign] wrote {a.out}", flush=True)


# ------------------------------------------------------------------------------------------------
# phase 2 : score  (vLLM M_omega -> V) + analysis
# ------------------------------------------------------------------------------------------------
def _auc(scores, labels):
    ok = np.isfinite(scores)
    s, l = scores[ok], labels[ok]
    if l.sum() < 3 or l.sum() > len(l) - 3:
        return np.nan
    r = np.argsort(np.argsort(s)) + 1
    return float((r[l > 0].mean() - (l.sum() + 1) / 2) / (len(l) - l.sum()))


def _safe_pct(x, q):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    return list(np.percentile(x, q)) if x.size else [float("nan")] * len(q)


def _safe_rho(x, y):
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    return float(spearmanr(x[m], y[m]).correlation)


def phase_score(a):
    z = np.load(a.assign, allow_pickle=True)
    A = z["A"]; names = list(z["names"]); uids = list(z["uids"]); texts = list(z["texts"])
    Ab = (A > 0).astype(float)
    n_docs, n_metrics = A.shape
    metric_desc = dict(load_metrics(a.hier))                        # merged_name -> 'name. desc'

    from .. import config as cfgmod
    from ..vllm_backend import make_judge_backend
    from . import alpha_probe as ap
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.mt)
    cfg0 = cfgmod.ImplementerConfig()
    executor = make_judge_backend(a.target_model, cfg0, temperature=None)

    V = np.zeros((n_docs, n_metrics))
    for j, nm in enumerate(names):
        rubric = metric_desc.get(nm, nm)
        V[:, j] = ap.signature(executor, rubric, texts, cfg.max_text_chars, template=ap._YESNO_TEXTFIRST)
        if (j + 1) % 25 == 0:
            print(f"[score] {j + 1}/{n_metrics} metrics", flush=True)
    np.savez(a.out.replace(".json", "_VA.npz"), V=V, A=A, names=np.array(names, dtype=object))

    cert = load_cert(a.cert) if a.cert else {}
    cuf = load_cuf(a.cuf)

    own_auc, spec_rank = {}, {}
    for j, nm in enumerate(names):
        aucs = np.array([_auc(V[:, j], Ab[:, jj]) for jj in range(n_metrics)])
        if not np.isfinite(aucs[j]):
            continue
        own_auc[nm] = float(aucs[j])
        fin = np.isfinite(aucs)
        spec_rank[nm] = float(np.mean(aucs[j] >= aucs[fin])) if fin.any() else float("nan")

    prk = {}
    for k in (3, 5, 10):
        tp = fp = fn = 0
        for i in range(n_docs):
            pred = set(np.argsort(-np.nan_to_num(V[i]))[:k])
            gold = set(np.where(Ab[i] > 0)[0])
            tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        prk[k] = {"P": prec, "R": rec, "F1": 2 * prec * rec / max(prec + rec, 1e-9)}

    scored = [nm for nm in names if nm in own_auc]
    oa = np.array([own_auc[nm] for nm in scored])
    sr = np.array([spec_rank[nm] for nm in scored])
    opt = np.array([cert.get(nm, {}).get("opt", np.nan) for nm in scored])
    nun = np.array([cuf.get(nm, {}).get("n_units", np.nan) for nm in scored])
    dead = np.array([cuf.get(nm, {}).get("dead_frac", np.nan) for nm in scored])

    out = {
        "task": a.mt, "n_docs": int(n_docs), "n_metrics": int(n_metrics),
        "n_scored_auc": len(scored),
        "mean_silver_per_item": float(Ab.sum(1).mean()),
        "own_auc_mean": float(np.nanmean(oa)) if oa.size else float("nan"),
        "own_auc_q10_50_90": _safe_pct(oa, [10, 50, 90]),
        "specificity_mean": float(np.nanmean(sr)) if sr.size else float("nan"),
        "frac_own_best_decile": float(np.mean(sr >= 0.9)) if sr.size else float("nan"),
        "micro_PR": prk,
        "corr_OPT_ownAUC": _safe_rho(opt, oa),
        "corr_OPT_specificity": _safe_rho(opt, sr),
        "corr_nUnits_ownAUC": _safe_rho(nun, oa),
        "corr_deadFrac_ownAUC": _safe_rho(dead, oa),
        "n_with_cert": int(np.isfinite(opt).sum()), "n_with_cuf": int(np.isfinite(nun).sum()),
        "per_metric": {nm: {"own_auc": own_auc[nm], "spec": spec_rank[nm],
                            "opt": cert.get(nm, {}).get("opt"), "n_units": cuf.get(nm, {}).get("n_units")}
                       for nm in scored},
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[score] {a.mt}: ownAUC={out['own_auc_mean']:.3f} spec={out['specificity_mean']:.3f} "
          f"P@5={prk[5]['P']:.2f} R@5={prk[5]['R']:.2f} "
          f"corr(OPT,ownAUC)={out['corr_OPT_ownAUC']:+.3f} corr(nUnits,ownAUC)={out['corr_nUnits_ownAUC']:+.3f}",
          flush=True)


# ------------------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="phase", required=True)

    pa = sub.add_parser("assign")
    pa.add_argument("--task", required=True)          # bge underscore name (code_review)
    pa.add_argument("--mt", required=True)            # hyphen name (code-review) for hier lookup
    pa.add_argument("--input", required=True)
    pa.add_argument("--norms", required=True)
    pa.add_argument("--hier", required=True)
    pa.add_argument("--bge-model", required=True)
    pa.add_argument("--ce-model", default=None, help="optional cross_encoder_llama8b for reranking")
    pa.add_argument("--ce-cand", type=int, default=30, help="BGE candidates the CE reranks per norm")
    pa.add_argument("--n-docs", type=int, default=500)
    pa.add_argument("--min-norms", type=int, default=2)
    pa.add_argument("--topk-assign", type=int, default=3)
    pa.add_argument("--out", required=True)

    ps = sub.add_parser("score")
    ps.add_argument("--assign", required=True)
    ps.add_argument("--mt", required=True)
    ps.add_argument("--hier", required=True)
    ps.add_argument("--cert", default=None)
    ps.add_argument("--cuf", default=None)
    ps.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ps.add_argument("--out", required=True)

    a = p.parse_args(argv)
    if a.phase == "assign":
        phase_assign(a)
    else:
        phase_score(a)


if __name__ == "__main__":
    main()
