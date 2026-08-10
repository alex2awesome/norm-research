#!/usr/bin/env python3
"""Stage 0 (CPU, local) -- build every corpus + nuisance artifact the
adversarial-debiasing pilot's validation battery (V1-V4) needs.

Spec: notes/2026-08-05__taste-decomposition-design.md S9 (BINDING).
Cell: N&C responded (9,521 rows / 1,814 dockets, y = agency responded).

Everything here is deterministic and label-blind EXCEPT the two synthetic
plants, which are label-aware BY CONSTRUCTION (that is the point of a planted
check).  Nothing produced here feeds any metric or any quoted V/A number; the
plants exist only to test whether the GRL machinery does what it claims.

Outputs (build/):
  population.csv         doc_id, docket, judgement, split, text          (real corpus)
  corpus_planted.csv     ... text = "PLANT " + text, + plant column      (V1/V2)
  corpus_realtok.csv     ... text = "RSTOK " + text, + realtok column    (V3a)
  v3b_population.csv     year-balanced subsample, real text              (V3b)
  nuisance.npz           standardized nuisance matrix + names + masks
  build_report.json      every rate / diagnostic quoted in the note
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BUILD = HERE / "build"
BUILD.mkdir(parents=True, exist_ok=True)
BASE_NPZ = REPO / "methods/taste_decomposition/closure/nc_responded/nc_responded_base.npz"
EMB_CACHE = REPO / "methods/taste_decomposition/data_cache/bge_large_embed_cache.npz"

SEED = 20260806
PLANT_TOKEN = "⟦QX7⟧"      # V1/V2 synthetic y-correlated plant
REALTOK_TOKEN = "⟦RS4⟧"    # V3a real-signal (content-correlated) plant
P_PLANT_POS = 0.65
P_PLANT_NEG = 0.35
K_TOPIC = 20

# ---- split (docket-grouped stable hash; NOT the dense chain's random split) --
TRAIN_HI, EVAL_HI = 0.80, 0.90


def hash_unit(key: str) -> float:
    return int(hashlib.sha256(str(key).encode("utf-8")).hexdigest(), 16) / 2 ** 256


# ---------------------------------------------------------------- nuisance --
_MARK_RE = re.compile(r"^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|```|>\s)")  # layer2_robustness.py


def char_len(t):
    return float(len(t or ""))


def log_token_proxy(t):
    return float(np.log1p(len((t or "").split())))


def linebreak_rate(t):
    t = t or ""
    return float(t.count("\n") / max(len(t), 1))


def markdown_rate(t):
    t = t or ""
    lines = t.split("\n")
    hits = sum(1 for ln in lines if _MARK_RE.match(ln))
    return float(hits / max(len(lines), 1))


_YEAR_RE = re.compile(r"(?:^|-)((?:19|20)\d\d)(?:-|$)")


def docket_year(docket: str):
    """Era channel derived from the DOCKET ID, e.g. FSIS-2016-0021 -> 2016.

    NOT `year` from nc_vat_sample.jsonl: that field is populated only for the
    MATCHED pool, so it is a near-perfect y proxy (y=1 for 100% of rows that
    carry it) -- exactly why Layer 2 dropped date for this cell.  The docket-ID
    year is available for both classes (94.7% coverage) and carries no such leak.
    """
    m = _YEAR_RE.findall(str(docket))
    return float(m[0]) if m else np.nan


def load_embeddings(texts):
    z = np.load(EMB_CACHE, allow_pickle=True)
    cache = set(z.files)
    keys = [hashlib.sha1((t or "").encode("utf-8")).hexdigest() for t in texts]
    missing = [k for k in keys if k not in cache]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(keys)} texts missing from the Layer-2 bge cache; "
            "the two pipelines must share the same text pool -- investigate before proceeding."
        )
    return np.stack([z[k] for k in keys])


def main():
    rng = np.random.default_rng(SEED)
    rep = {"seed": SEED, "spec": "notes/2026-08-05__taste-decomposition-design.md S9"}

    z = np.load(BASE_NPZ, allow_pickle=True)
    doc_id = np.array([str(s) for s in z["doc_id"]])
    docket = np.array([str(s) for s in z["docket"]])
    y = z["y"].astype(int)
    texts = np.array([str(t) for t in z["texts"]], dtype=object)
    A = z["A"].astype(float)
    a_names = [str(s) for s in z["a_names"]]
    n = len(y)
    assert len(set(doc_id)) == n
    rep["population"] = {"n": n, "n_dockets": int(len(set(docket))), "pos_rate": float(y.mean())}

    # ---------------------------------------------------------- splits ------
    # Docket-grouped stable-hash, but cut on CUMULATIVE ROW COUNT rather than on
    # the hash value itself: dockets range from 1 to hundreds of comments, so a
    # raw hash<.80/.90 cut lands at 80.1/6.6/13.3 by row (eval n=626, pos rate
    # .66 vs train .78).  Walking dockets in hash order and cutting at 80%/90% of
    # rows keeps the rule deterministic and docket-disjoint while hitting 80/10/10
    # by row.  No seeded shuffle of a growing list is involved.
    uniq = sorted(set(docket), key=hash_unit)
    size = pd.Series(docket).value_counts()
    cum, assign = 0, {}
    for d in uniq:
        frac = cum / n
        assign[d] = "train" if frac < TRAIN_HI else ("eval" if frac < EVAL_HI else "test")
        cum += int(size[d])
    split = np.array([assign[d] for d in docket])
    tr = split == "train"
    rep["split"] = {
        "rule": "dockets ordered by sha256(docket)/2**256, cut at 80%/90% of cumulative ROWS (docket-grouped, stable-hash)",
        "counts": {s: int((split == s).sum()) for s in ("train", "eval", "test")},
        "n_dockets": {s: int(len(set(docket[split == s]))) for s in ("train", "eval", "test")},
        "pos_rate": {s: float(y[split == s].mean()) for s in ("train", "eval", "test")},
        "docket_disjoint": bool(
            not (set(docket[tr]) & set(docket[split == "eval"]))
            and not (set(docket[tr]) & set(docket[split == "test"]))
            and not (set(docket[split == "eval"]) & set(docket[split == "test"]))
        ),
    }
    assert rep["split"]["docket_disjoint"]

    # -------------------------------------------------------- nuisances -----
    nf = pd.DataFrame({
        "char_len": [char_len(t) for t in texts],
        "log_token": [log_token_proxy(t) for t in texts],
        "linebreak_rate": [linebreak_rate(t) for t in texts],
        "markdown_rate": [markdown_rate(t) for t in texts],
    })
    yr = np.array([docket_year(d) for d in docket])
    yr_missing = (~np.isfinite(yr)).astype(float)
    yr_filled = np.where(np.isfinite(yr), yr, np.nanmedian(yr[tr]))
    nf["docket_year"] = yr_filled
    nf["year_missing"] = yr_missing

    print("[embed] loading bge-large vectors from the Layer-2 cache ...")
    emb = load_embeddings(list(texts))
    km = KMeans(n_clusters=K_TOPIC, random_state=0, n_init=10).fit(emb[tr])  # TRAIN-ONLY fit
    topic = km.predict(emb)
    onehot = np.zeros((n, K_TOPIC))
    onehot[np.arange(n), topic] = 1.0

    cont = nf.values
    cont_names = list(nf.columns)
    topic_names = [f"topic_{i:02d}" for i in range(K_TOPIC)]
    raw = np.hstack([cont, onehot])
    names = cont_names + topic_names

    mu, sd = raw[tr].mean(0), raw[tr].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (raw - mu) / sd
    rep["nuisance"] = {
        "names": names,
        "n_cols": len(names),
        "standardization": "z-score on TRAIN rows only",
        "topic": {"k": K_TOPIC, "embed": "BAAI/bge-large-en-v1.5 (Layer-2 cache)",
                  "kmeans_fit": "train rows only"},
        "docket_year_coverage": float(np.isfinite(yr).mean()),
    }

    # channel groups the battery names individually
    groups = {
        "length": [names.index("char_len"), names.index("log_token")],
        "format": [names.index("linebreak_rate"), names.index("markdown_rate")],
        "date": [names.index("docket_year"), names.index("year_missing")],
        "topic": [names.index(t) for t in topic_names],
    }
    groups["standard"] = sorted(sum(groups.values(), []))
    rep["nuisance"]["groups"] = {k: v for k, v in groups.items()}

    # channel-alone AUCs (docket-grouped OOF logistic regression) -- context for the note
    def alone_auc(cols):
        X = Z[:, cols]
        oof = np.zeros(n)
        for a, b in GroupKFold(5).split(X, y, groups=docket):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            m.fit(X[a], y[a])
            oof[b] = m.predict_proba(X[b])[:, 1]
        return float(roc_auc_score(y, oof))

    rep["nuisance"]["alone_auc_oof"] = {k: alone_auc(v) for k, v in groups.items()}
    print("  nuisance-alone AUC:", rep["nuisance"]["alone_auc_oof"])

    # ------------------------------------------------- V1/V2 planted corpus --
    p = np.where(y == 1, P_PLANT_POS, P_PLANT_NEG)
    plant = (rng.random(n) < p).astype(int)
    rep["plant"] = {
        "token": PLANT_TOKEN,
        "position": "PREPENDED (text = token + ' ' + text)",
        "position_rationale": (
            "max_length=1024 truncates from the RIGHT and 25% of comments hit the "
            "4,000-char cap (~1,000 tokens), so an appended token is not guaranteed "
            "to survive tokenisation; a prepended token always does."),
        "target_rates": {"P(plant|y=1)": P_PLANT_POS, "P(plant|y=0)": P_PLANT_NEG},
        "realized_rates": {
            "P(plant|y=1)": float(plant[y == 1].mean()),
            "P(plant|y=0)": float(plant[y == 0].mean()),
            "overall": float(plant.mean()),
        },
        "realized_rates_by_split": {
            s: {"pos": float(plant[(split == s) & (y == 1)].mean()),
                "neg": float(plant[(split == s) & (y == 0)].mean())}
            for s in ("train", "eval", "test")},
        "plant_alone_auc": float(roc_auc_score(y, plant)),
        "plant_alone_auc_eval": float(roc_auc_score(y[split == "eval"], plant[split == "eval"])),
    }
    print("  plant-alone AUC:", rep["plant"]["plant_alone_auc"])

    # ------------------------------------------- V3a real-signal token -------
    # "Real signal" = the ARTICULATED-QUALITY composite (docket-grouped OOF GBM on
    # the 198-criterion A bank), residualised against the whole standard nuisance
    # matrix.  Deviation from the spec's "e.g. top-quartile evidence-specificity
    # rubric score" is FORCED: every single evidence/specificity criterion in this
    # bank has alone-AUC .49-.52 on this cell (measured, see report), so a
    # single-criterion quartile token would carry no learnable signal and V3a would
    # pass vacuously.  The composite is still a *content* channel (built only from
    # judged rubric scores, never from surface text features) and the residualisation
    # makes it near-orthogonal to the channels being debiased.
    # The A columns are orthogonalised against Z FIRST (train-fit linear residual),
    # then the composite is fit on the residual columns, then the composite itself
    # is residualised again -- a token defined on the raw composite is 15% explained
    # by the nuisance matrix (measured), which would make any V3a erosion ambiguous
    # between "debiasing broke a real channel" and "debiasing correctly removed
    # length".  Threshold = TRAIN median (a quartile cut costs too much channel
    # strength: measured alone-AUC .551 at q=.75 vs .621 at the median).
    col_med = np.nanmedian(A, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    Af = np.where(np.isfinite(A), A, col_med)
    Ar = Af - LinearRegression().fit(Z[tr], Af[tr]).predict(Z)
    oof_q = np.zeros(n)
    for a, b in GroupKFold(5).split(Ar, y, groups=docket):
        g = HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=0.06,
                                           max_iter=400, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=20,
                                           random_state=0)
        g.fit(Ar[a], y[a])
        oof_q[b] = g.predict_proba(Ar[b])[:, 1]
    q_auc = float(roc_auc_score(y, oof_q))
    resid = oof_q - LinearRegression().fit(Z[tr], oof_q[tr]).predict(Z)
    thr = np.quantile(resid[tr], 0.50)          # threshold from TRAIN rows only
    realtok = (resid >= thr).astype(int)
    # how much of the real-signal token is explained by the nuisance matrix?
    r2_tok_on_nuis = float(
        1.0 - np.var(realtok - LinearRegression().fit(Z, realtok).predict(Z)) / max(np.var(realtok), 1e-12))
    rep["realtok"] = {
        "token": REALTOK_TOKEN,
        "position": "PREPENDED",
        "definition": ("above TRAIN median of (OOF GBM quality composite fit on the 198 A-bank "
                       "criteria after each criterion was linearly residualised on the standard "
                       "nuisance matrix), itself residualised again on the nuisance matrix"),
        "composite_alone_auc": q_auc,
        "label_supervision_note": ("the composite's WEIGHTS are y-supervised (OOF GBM), so the token "
                                   "is a content-derived channel, not a y-free one; it is not a "
                                   "function of y (P(tok|y=1)=.55 vs P(tok|y=0)=.31).  What V3a tests "
                                   "is whether an UNNAMED channel survives debiasing of NAMED ones, "
                                   "which does not depend on how the unnamed channel was constructed."),
        "quartile_threshold_source": "train rows only",
        "realtok_rate": float(realtok.mean()),
        "realtok_alone_auc": float(roc_auc_score(y, realtok)),
        "realtok_alone_auc_eval": float(roc_auc_score(y[split == "eval"], realtok[split == "eval"])),
        "R2_of_realtok_explained_by_nuisance_matrix": r2_tok_on_nuis,
        "single_criterion_alternative_rejected": {
            "reason": "all evidence/specificity criteria alone-AUC in [.487,.522] on this cell",
            "examples": {a_names[j][:70]: float(roc_auc_score(y[np.isfinite(A[:, j])], A[np.isfinite(A[:, j]), j]))
                         for j in (7, 32, 35, 141, 149) if np.isfinite(A[:, j]).sum() > 9000},
        },
    }
    print("  realtok-alone AUC:", rep["realtok"]["realtok_alone_auc"],
          "| composite AUC:", q_auc, "| R2 on nuisances:", round(r2_tok_on_nuis, 4))

    # --------------------------------------------- V3b year-balanced subset --
    # Largest subsample whose date-alone AUC is ~.50 (equal positive rate in every
    # retained year).  Date is then a channel the model verifiably cannot use for y,
    # while still being readable from the text -- the spec's "verifiably unused" cell.
    # Balancing is done WITHIN EACH SPLIT: balancing the pooled subsample only makes
    # date uninformative in aggregate, and (measured) still leaves a train->eval
    # date-alone AUC of .41 because train and eval draw on different dockets and
    # dockets are nested in years.  Per-split balancing makes date uninformative on
    # the split the gate is actually read on.
    yr_int = np.where(np.isfinite(yr), yr, -1).astype(int)
    best = None
    for target in np.arange(0.60, 0.95, 0.01):
        keep = np.zeros(n, bool)
        for sp in ("train", "eval", "test"):
            for u in np.unique(yr_int):
                if u < 0:
                    continue
                idx = np.flatnonzero((yr_int == u) & (split == sp))
                pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
                if len(pos) < 3 or len(neg) < 3:
                    continue
                k_pos = min(len(pos), int(round(len(neg) * target / (1 - target))))
                k_neg = min(len(neg), int(round(len(pos) * (1 - target) / target)))
                if k_pos < 3 or k_neg < 3:
                    continue
                r = np.random.default_rng(SEED + u + 1000 * {"train": 1, "eval": 2, "test": 3}[sp])
                keep[r.choice(pos, k_pos, replace=False)] = True
                keep[r.choice(neg, k_neg, replace=False)] = True
        if keep.sum() < 500:
            continue
        auc = roc_auc_score(y[keep], yr_int[keep])
        npos, nneg = int(y[keep].sum()), int((1 - y[keep]).sum())
        # maximise COMPARABLE PAIRS (n_pos*n_neg), not raw n: AUC variance is driven
        # by min(n_pos,n_neg), and maximising n alone lands on a 92%-positive
        # subsample with ~59 eval negatives, which cannot resolve a .005 gate.
        score = npos * nneg if abs(auc - 0.5) < 0.02 else -1
        if best is None or score > best[0]:
            best = (score, float(target), keep.copy(), float(auc))
    _, v3b_target, v3b_keep, v3b_auc = best
    v3b_yearalone_oof = None
    if v3b_keep.sum() > 500:
        Xd = Z[np.ix_(v3b_keep, groups["date"])]
        yy, gg = y[v3b_keep], docket[v3b_keep]
        oof = np.zeros(v3b_keep.sum())
        for a, b in GroupKFold(5).split(Xd, yy, groups=gg):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            m.fit(Xd[a], yy[a])
            oof[b] = m.predict_proba(Xd[b])[:, 1]
        v3b_yearalone_oof = float(roc_auc_score(yy, oof))
    rep["v3b"] = {
        "construction": "per-year subsample to a common positive rate (year-balanced)",
        "target_pos_rate": v3b_target,
        "n": int(v3b_keep.sum()),
        "n_dockets": int(len(set(docket[v3b_keep]))),
        "pos_rate": float(y[v3b_keep].mean()),
        "date_alone_auc_raw_year": v3b_auc,
        "date_alone_auc_oof_logreg": v3b_yearalone_oof,
        "split_counts": {s: int(((split == s) & v3b_keep).sum()) for s in ("train", "eval", "test")},
        "full_corpus_date_alone_auc_for_contrast": rep["nuisance"]["alone_auc_oof"]["date"],
    }
    print("  V3b subsample:", rep["v3b"]["n"], "rows, date-alone AUC",
          round(v3b_auc, 4), "/ oof", v3b_yearalone_oof)

    # ------------------------------------------------------------- write ----
    base = pd.DataFrame({"doc_id": doc_id, "docket": docket, "judgement": y,
                         "split": split, "text": texts})
    base.to_csv(BUILD / "population.csv", index=False)

    pl = base.copy()
    pl["plant"] = plant
    pl["text"] = [f"{PLANT_TOKEN} {t}" if k else t for t, k in zip(texts, plant)]
    pl.to_csv(BUILD / "corpus_planted.csv", index=False)

    rt = base.copy()
    rt["realtok"] = realtok
    rt["text"] = [f"{REALTOK_TOKEN} {t}" if k else t for t, k in zip(texts, realtok)]
    rt.to_csv(BUILD / "corpus_realtok.csv", index=False)

    base[v3b_keep].to_csv(BUILD / "v3b_population.csv", index=False)

    np.savez_compressed(
        BUILD / "nuisance.npz",
        doc_id=doc_id, Z=Z, names=np.array(names, dtype=object),
        plant=plant, realtok=realtok, split=split, y=y, docket=docket,
        v3b_keep=v3b_keep, topic=topic, raw=raw, mu=mu, sd=sd,
        groups_json=json.dumps({k: [int(i) for i in v] for k, v in groups.items()}),
    )
    (BUILD / "build_report.json").write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {BUILD}")
    print(json.dumps({k: rep[k] for k in ("population", "split")}, indent=2))


if __name__ == "__main__":
    main()
