#!/usr/bin/env python3
"""Layer 2 -- robustness appendix (grouped transfer + nuisance-stratified
readouts) of the taste-residual decomposition.

Design: notes/2026-08-05__taste-decomposition-design.md S2. Ledger names,
freeze changes and cell registry: S0/S4/S6 of the same file.

Part (a) grouped-transfer table, per cell and per score column (VA_nl, and T
where a row-level T prediction exists on the SAME rows):
  - pooled AUC
  - within-group AUC: n-weighted mean of AUC computed inside each group with
    >=20 rows and both classes present (using the cell's own canonical
    grouping unit -- ntitle / docket / prompt_id / contest)
  - group-identity-alone AUC: score = the row's group's positive rate,
    estimated OUT OF FOLD via a plain (non-grouped) K-fold over rows so a
    row's own label never informs its own score; groups unseen in the train
    fold fall back to the train fold's global rate. This is deliberately NOT
    a GroupKFold (which would put an entire group in one fold and make
    "group rate" undefined/circular for the held-out fold).

Part (b) nuisance-stratified readouts (threshold-free), uniform nuisance set:
  1. length: char length + log1p(word count) proxy for token count
  2. format: linebreak rate + markdown/list-marker line rate
  3. topic: k=20 KMeans on base BAAI/bge-large-en-v1.5 embeddings, fit
     TRAIN-SIDE ONLY inside the cell's grouped folds (mirrors the Layer-1 OOF
     discipline) -> an out-of-fold cluster label per row
  4. date/era: publication year, where the corpus carries one (peer-review,
     N&C); absent for CW / caption cells (noted, not imputed)
Per nuisance dimension: nuisance-alone AUC (grouped-OOF logistic regression),
plus a joint-linear AUC over all available dimensions; then the
decile/cluster-stratified AUC of each score column (VA_nl, T where available)
-- n-weighted mean over strata with >=20 rows and both classes. Survival flag
= |stratified - pooled| <= .02.

VA_nl here is recomputed FRESH (frozen Layer-1 grid, seed 0, same grouped
outer folds) rather than reusing the saved *_va_nl_oof_*.npy files, because
several cells' Layer-1 loaders build their row order from Python set
iteration (nc_layer1_stack.NCData.valid_out/valid_agr) which is not provably
stable against an independently-constructed id list. Recomputing inside this
script guarantees ids / y / groups / VA_nl OOF are all in one, single,
internally-consistent row order. Sanity gate: the recomputed VA_nl AUC is
checked against the corresponding *_layer1.json ledger value (tolerance
.01 -- seed-0-only vs that file's 3-seed mean).

T (dense) row-level predictions are available on the SAME rows for peer
verdict ONLY (methods/taste_decomposition/closure/peer_verdict_dense_preds.csv,
restricted to dense_split in {eval,test} = the honest held-out rows, n=1244;
freeze change 2). All other priority cells have only an aggregate eval/test
AUC on file (no row-level predictions saved anywhere local or on sk3 --
checked datasets/*/dense_llama/*/rm_out on sk3: adapters only, no per-row
prediction artifact) -- those cells run part (a)/(b) with VA_nl only, T
marked missing, per the task's explicit fallback.

Text access: all 8 priority cells have LOCAL raw text (verified against the
Layer-1 matrices' own id/hash construction -- see notes/2026-08-06__layer2_
robustness.md for the per-cell join proof). No sk3 corpus fetch was needed.

CPU + one local MPS-accelerated embedding pass (BAAI/bge-large-en-v1.5, plain
transformers AutoModel, no sentence-transformers -- that package's import
chain pulls in a broken local tensorflow build). No training, no LLM judging.

Usage:
  python layer2_robustness.py --cell peer_verdict
  python layer2_robustness.py --all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
TASTE = Path(__file__).resolve().parent
RESULTS = TASTE / "results"
CACHE = TASTE / "data_cache"
CACHE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(TASTE))
import layer1_stack as L1            # noqa: E402  (peer-review verdict/curation/revealed)
import layer1_gemma_cells as L1G     # noqa: E402  (cw_community, cap_crowd, cap_finalist)

NC_DIR = REPO / "datasets" / "notice-and-comment" / "v4"
sys.path.insert(0, str(NC_DIR))
import nc_layer1_stack as NCL        # noqa: E402

PEER = REPO / "datasets" / "peer-review" / "vat_3y"
CLOSURE = TASTE / "closure"
CW_TEXT_CSV = REPO / "datasets/creative-writing/va_bank_v2/writingprompts_modeling_clean_reconstructed.csv.gz"
CAP_POOL = REPO / "datasets/humor/caption_multiy/caption_contest_v2.jsonl"

GBM_SEED = 0  # Layer 2 uses the single primary seed (matches each Layer-1
              # script's own Delta_interact point-estimate convention); the
              # 3-seed mean is the Layer-1 headline, not re-derived here.

MIN_GROUP_N = 20
MIN_STRAT_N = 20
SURVIVAL_TOL = 0.02
K_TOPIC = 20
N_SPLITS = 5


# ============================================================== generic ===
def gfolds(n, groups, n_splits=N_SPLITS):
    ns = min(n_splits, len(np.unique(groups)))
    return list(GroupKFold(n_splits=ns).split(np.zeros(n), groups=groups))


def auc_safe(y, s, idx=None):
    if idx is not None:
        y, s = y[idx], s[idx]
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(y) < MIN_STRAT_N or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def pooled_auc(y, s, mask=None):
    if mask is not None:
        return auc_safe(y[mask], s[mask])
    return auc_safe(y, s)


def within_group_auc(y, groups, s, mask=None, min_n=MIN_GROUP_N):
    if mask is not None:
        y, groups, s = y[mask], groups[mask], s[mask]
    aucs, ns, kept = [], [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < min_n:
            continue
        a = auc_safe(y[idx], s[idx])
        if a is None:
            continue
        aucs.append(a)
        ns.append(len(idx))
        kept.append(str(g))
    if not aucs:
        return {"auc": None, "n_qualifying_groups": 0, "n_groups_total": int(len(np.unique(groups)))}
    aucs, ns = np.array(aucs), np.array(ns)
    return {
        "auc": float(np.average(aucs, weights=ns)),
        "n_qualifying_groups": int(len(aucs)),
        "n_groups_total": int(len(np.unique(groups))),
        "n_rows_in_qualifying_groups": int(ns.sum()),
    }


def group_identity_oof(y, groups, mask=None, n_splits=N_SPLITS, seed=0):
    """Group positive-rate as a score, estimated out-of-fold via a plain
    (row-level, non-grouped) K-fold: a row's own label never informs its own
    score, but other rows of the same group in the train fold do. Groups
    absent from a fold's train side fall back to that fold's global rate --
    the correct, conservative behavior for near-singleton grouping units."""
    if mask is not None:
        y_full, groups_full = y[mask], groups[mask]
    else:
        y_full, groups_full = y, groups
    n = len(y_full)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_id = np.empty(n, dtype=int)
    fold_id[perm] = np.arange(n) % n_splits
    score = np.full(n, np.nan)
    for k in range(n_splits):
        te = fold_id == k
        tr = ~te
        df = pd.DataFrame({"g": groups_full[tr], "y": y_full[tr]})
        rate = df.groupby("g")["y"].mean()
        global_rate = float(y_full[tr].mean())
        score[te] = [float(rate.get(g, global_rate)) for g in groups_full[te]]
    return score, auc_safe(y_full, score)


def grouped_transfer_block(y, groups, s, mask=None):
    p = pooled_auc(y, s, mask)
    wg = within_group_auc(y, groups, s, mask)
    _, gid = group_identity_oof(y, groups, mask)
    return {"pooled_auc": p, "within_group": wg, "group_identity_alone_auc": gid}


def lr_oof_auc(Xf, y, groups, n_splits=N_SPLITS):
    Xf = np.asarray(Xf, dtype=float)
    if Xf.ndim == 1:
        Xf = Xf.reshape(-1, 1)
    if Xf.shape[1] == 0:
        return None
    folds = gfolds(len(y), groups, n_splits)
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        pipe.fit(Xf[tr], y[tr])
        oof[te] = pipe.predict_proba(Xf[te])[:, 1]
    return auc_safe(y, oof)


def decile_labels(x, n_bins=10):
    x = np.asarray(x, dtype=float)
    try:
        cats = pd.qcut(x, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        cats = pd.qcut(x.astype(float) + np.arange(len(x)) * 1e-9, q=n_bins, labels=False, duplicates="drop")
    return np.asarray(cats)


def stratified_auc(y, s, strata, mask=None, min_n=MIN_STRAT_N):
    if mask is not None:
        y, s, strata = y[mask], s[mask], strata[mask]
    aucs, ns = [], []
    for st in np.unique(strata):
        if st is None or (isinstance(st, float) and np.isnan(st)):
            continue
        idx = np.where(strata == st)[0]
        if len(idx) < min_n:
            continue
        a = auc_safe(y[idx], s[idx])
        if a is None:
            continue
        aucs.append(a)
        ns.append(len(idx))
    if not aucs:
        return {"auc": None, "n_qualifying_strata": 0, "n_strata_total": int(len(np.unique(strata)))}
    aucs, ns = np.array(aucs), np.array(ns)
    return {"auc": float(np.average(aucs, weights=ns)), "n_qualifying_strata": int(len(aucs)),
            "n_strata_total": int(len(np.unique(strata))), "n_rows_in_qualifying_strata": int(ns.sum())}


# ============================================================ text feats ===
_MARK_RE = re.compile(r"^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|```|>\s)")


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
    if not lines:
        return 0.0
    hits = sum(1 for ln in lines if _MARK_RE.match(ln))
    return float(hits / max(len(lines), 1))


def text_nuisance_frame(texts):
    return pd.DataFrame({
        "char_len": [char_len(t) for t in texts],
        "log_token": [log_token_proxy(t) for t in texts],
        "linebreak_rate": [linebreak_rate(t) for t in texts],
        "markdown_rate": [markdown_rate(t) for t in texts],
    })


# ============================================================ embeddings ===
_EMB_MODEL = None
_EMB_TOK = None
_EMB_DEVICE = None


def _get_embed_model():
    global _EMB_MODEL, _EMB_TOK, _EMB_DEVICE
    if _EMB_MODEL is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        _EMB_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
        _EMB_TOK = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        _EMB_MODEL = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").eval().to(_EMB_DEVICE)
        print(f"  [embed] BAAI/bge-large-en-v1.5 loaded on {_EMB_DEVICE}")
    return _EMB_MODEL, _EMB_TOK, _EMB_DEVICE


def embed_texts(texts, batch_size=64, max_length=256):
    """sha1-keyed disk cache so shared text pools (peer verdict/curation/
    revealed; N&C responded/outcome; cap crowd/finalist) are embedded once."""
    import torch

    cache_path = CACHE / "bge_large_embed_cache.npz"
    keys = [hashlib.sha1((t or "").encode("utf-8")).hexdigest() for t in texts]
    cache = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        cache = {k: z[k] for k in z.files}
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    if missing_idx:
        model, tok, device = _get_embed_model()
        print(f"  [embed] {len(missing_idx)} new / {len(texts)} total texts to embed")
        for bstart in range(0, len(missing_idx), batch_size):
            bidx = missing_idx[bstart:bstart + batch_size]
            btexts = [(texts[i] or "")[:4000] for i in bidx]
            enc = tok(btexts, padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc)
            emb = out.last_hidden_state[:, 0]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1).float().cpu().numpy()
            for j, i in enumerate(bidx):
                cache[keys[i]] = emb[j]
            if bstart % (batch_size * 20) == 0:
                print(f"    ... {bstart + len(bidx)}/{len(missing_idx)}")
        np.savez_compressed(cache_path, **cache)
        print(f"  [embed] cache now {len(cache)} vectors -> {cache_path}")
    return np.stack([cache[k] for k in keys])


def topic_kmeans_oof(emb, groups, k=K_TOPIC, n_splits=N_SPLITS, seed=0):
    folds = gfolds(len(groups), groups, n_splits)
    cluster = np.full(len(groups), -1, dtype=int)
    for tr, te in folds:
        kk = min(k, max(2, len(tr) // 5))
        km = KMeans(n_clusters=kk, random_state=seed, n_init=10)
        km.fit(emb[tr])
        cluster[te] = km.predict(emb[te])
    return cluster


# ======================================================= nuisance block ===
def nuisance_block(y, groups, score_map, texts, years=None, score_masks=None, cell_tag=""):
    """score_map: {"VA_nl": array, "T": array or None}. score_masks: optional
    per-score-column row mask (e.g. T restricted to dense held-out rows) --
    VA_nl always gets the FULL row set unless explicitly masked; the two must
    not share one mask or VA_nl silently loses rows it doesn't need to lose."""
    score_masks = score_masks or {}
    nf = text_nuisance_frame(texts)
    emb = embed_texts(texts)
    topic_oof = topic_kmeans_oof(emb, groups)

    dims = {
        "length": {"cols": nf[["char_len", "log_token"]].values,
                   "strat_scalar": nf["char_len"].values, "kind": "decile"},
        "format": {"cols": nf[["linebreak_rate", "markdown_rate"]].values,
                   "strat_scalar": nf["linebreak_rate"].values, "kind": "decile"},
        "topic": {"cols": pd.get_dummies(topic_oof).values.astype(float),
                  "strat_scalar": topic_oof, "kind": "cluster"},
    }
    n_date_imputed = 0
    if years is not None and np.isfinite(years).sum() >= 0.9 * len(years):
        yrs = np.asarray(years, dtype=float)
        # decile stratification uses the RAW (unimputed) years -- rows with an
        # unknown year simply fall out of every decile stratum, same as any
        # other stratified_auc NaN-skip. The alone/joint LR fit needs a
        # NaN-free matrix, so that copy gets a median impute (<=10% of rows,
        # guarded above) with the imputed count reported for transparency.
        yrs_imputed = yrs.copy()
        nan_mask = ~np.isfinite(yrs_imputed)
        n_date_imputed = int(nan_mask.sum())
        if n_date_imputed:
            yrs_imputed[nan_mask] = float(np.nanmedian(yrs))
        dims["date"] = {"cols": yrs_imputed.reshape(-1, 1),
                         "strat_scalar": yrs, "kind": "decile"}

    out = {"nuisance_alone_auc": {}, "stratified": {}, "n_topic_clusters": int(len(np.unique(topic_oof))),
           "date_n_imputed_for_alone_auc": n_date_imputed}

    joint_cols = []
    for name, d in dims.items():
        a = lr_oof_auc(d["cols"], y, groups)
        out["nuisance_alone_auc"][name] = a
        joint_cols.append(d["cols"])
    out["nuisance_alone_auc"]["joint"] = lr_oof_auc(np.column_stack(joint_cols), y, groups)

    for name, d in dims.items():
        strata = decile_labels(d["strat_scalar"]) if d["kind"] == "decile" else d["strat_scalar"]
        entry = {"pooled_note": "see part (a) pooled_auc for the same score column"}
        for score_name, s in score_map.items():
            if s is None:
                continue
            smask = score_masks.get(score_name)
            strat = stratified_auc(y, s, strata, mask=smask)
            pooled = pooled_auc(y, s, smask)
            drop = None if (strat["auc"] is None or pooled is None) else float(pooled - strat["auc"])
            entry[score_name] = {**strat, "pooled_auc": pooled,
                                  "survives": None if drop is None else bool(abs(drop) <= SURVIVAL_TOL),
                                  "drop_from_pooled": drop}
        out["stratified"][name] = entry
    return out


# ============================================================ cell loaders =
def _sanity(name, recomputed_auc, ledger_auc, tol=0.012):
    ok = (ledger_auc is None) or (recomputed_auc is not None and abs(recomputed_auc - ledger_auc) <= tol)
    print(f"  [sanity] {name}: recomputed VA_nl(seed0)={recomputed_auc}  ledger VA_nl={ledger_auc}  "
          f"{'OK' if ok else 'MISMATCH-FLAG'}")
    return ok


def load_peer_cell(cell_jsonl_name, tag):
    """peer verdict / curation / revealed: layer1_stack.load_cell() re-derived
    here with text + year captured in the SAME filter/order."""
    z = np.load(L1.NPZ, allow_pickle=True)
    X, V, nt = z["X"], z["V"], z["ntitle"]
    X_by_nt = {nt[i]: X[i] for i in range(len(nt))}
    V_by_nt = {nt[i]: V[i] for i in range(len(nt))}

    rows = [json.loads(l) for l in open(PEER / f"{cell_jsonl_name}.jsonl") if l.strip()]
    R = [r for r in rows if r.get("ntitle") in X_by_nt and L1._valid_y(r)]
    ntl = [r["ntitle"] for r in R]
    y = np.array([int(float(r["judgement"])) for r in R])
    A = np.array([X_by_nt[k] for k in ntl], dtype=float)
    Vm = np.array([V_by_nt[k] for k in ntl], dtype=float)
    groups = np.array(ntl)
    texts = [r.get("text", "") for r in R]
    years = np.array([float(r["year"]) if r.get("year") is not None else np.nan for r in R])

    Ac, _ = L1.clean_cols(A)
    Vc, _ = L1.clean_cols(Vm)
    VA = np.column_stack([Vc, Ac]) if Vc.shape[1] and Ac.shape[1] else (Vc if Vc.shape[1] else Ac)

    folds = L1.outer_folds(len(y), groups)
    r = L1.gbm_oof(VA, y, groups, folds, seed=GBM_SEED)
    va_lin, _ = L1.linear_oof(VA, y, groups, folds)

    ledger_path = RESULTS / f"{tag}_layer1.json"
    ledger_auc = None
    if ledger_path.exists():
        ledger_auc = json.loads(ledger_path.read_text())["ledger"].get("VA_nl") or \
            json.loads(ledger_path.read_text())["ledger"].get("VA_nl_mean")
    _sanity(tag, r["auc"], ledger_auc)

    T_score, T_mask = None, None
    T_note = "no row-level T found"
    if tag in ("peer_curation", "peer_revealed"):
        agg = {"peer_curation": "eval .593 / test .588", "peer_revealed": "eval .871 / test .896"}[tag]
        T_note = (f"aggregate-only T on file ({agg}, notes/2026-07-27__vat-run-registry.md 'DENSE "
                  f"CHAIN - CLEAN-EVAL FINAL'); no same-rows row-level rescore has been run for this "
                  f"cell (only peer verdict got the closure/rescore_dense_same_rows.py treatment). "
                  f"Running one is a GPU job on sk3 (design S4b cross-cutting GPU batch), out of "
                  f"Layer 2's CPU-only scope.")
    if tag == "peer_verdict":
        dp = pd.read_csv(CLOSURE / "peer_verdict_dense_preds.csv")
        assert len(dp) == len(y), f"dense_preds length {len(dp)} != cell n {len(y)}"
        # verified positionally aligned to layer1_stack.load_cell('verdict') --
        # see notes/2026-08-06__layer2_robustness.md for the check.
        T_score = dp["dense_prob"].values.astype(float)
        T_mask = dp["dense_split"].isin(["eval", "test"]).values
        T_note = ("row-level, methods/taste_decomposition/closure/peer_verdict_dense_preds.csv "
                  f"(rescore_dense_same_rows.py); restricted to dense_split in "
                  f"{{eval,test}} (n={int(T_mask.sum())}) -- train rows are in-sample/contaminated.")

    return dict(tag=tag, n=int(len(y)), y=y, groups=groups, group_column="ntitle",
                ids=ntl, texts=texts, years=years,
                va_nl=r["oof"], va_nl_auc=r["auc"], va_lin_auc=va_lin,
                T_score=T_score, T_mask=T_mask, T_note=T_note,
                matrix=str(L1.NPZ))


def load_nc_cell(cell):
    data = NCL.NCData()
    if cell == "responded":
        pairs = [(d, data.y_resp_by_id[d], data.X_m, data.docket_m, data.text_m) for d in data.X_m]
        pairs += [(d, data.y_resp_by_id[d], data.X_u, data.docket_u, data.text_u) for d in data.X_u]
    elif cell == "outcome":
        ids_sorted = sorted(data.valid_out)
        pairs = [(d, data.y_out_by_id[d], data.X_m, data.docket_m, data.text_m) for d in ids_sorted]
    elif cell == "agree":
        ids_sorted = sorted(data.valid_agr)
        pairs = [(d, data.y_agr_by_id[d], data.X_m, data.docket_m, data.text_m) for d in ids_sorted]
    else:
        raise ValueError(cell)

    ids, ys, As, Vs, groups, texts = [], [], [], [], [], []
    for d, y, Xd, docketd, textd in pairs:
        ids.append(d)
        ys.append(y)
        As.append(Xd[d].astype(float))
        t = textd.get(d, "")
        texts.append(t)
        Vs.append(np.array([NCL.v_features(t)[n] for n in NCL.V_NAMES], dtype=float))
        groups.append(docketd[d])
    y = np.array(ys)
    A = np.array(As, dtype=float)
    Vm = np.array(Vs, dtype=float)
    groups = np.array(groups)

    Ac, _ = NCL.clean_cols(A)
    Vc, _ = NCL.clean_cols(Vm)
    VA = np.column_stack([Vc, Ac]) if Vc.shape[1] and Ac.shape[1] else (Vc if Vc.shape[1] else Ac)

    folds = NCL.outer_folds(len(y), groups)
    r = NCL.gbm_oof(VA, y, groups, folds, seed=GBM_SEED)
    va_lin, _ = NCL.linear_oof(VA, y, groups, folds)

    tag = f"nc_{cell}"
    ledger_path = RESULTS / f"{tag}_layer1.json"
    ledger_auc = None
    if ledger_path.exists():
        led = json.loads(ledger_path.read_text())["ledger"]
        ledger_auc = led.get("VA_nl_mean", led.get("VA_nl"))
    _sanity(tag, r["auc"], ledger_auc)

    # year lookup (matched pool from nc_vat_sample.jsonl; unmatched pool for responded)
    year_by_id = {}
    for p in [NC_DIR / "nc_vat_sample.jsonl", NC_DIR / "nc_unmatched_sample.jsonl"]:
        for line in open(p):
            if not line.strip():
                continue
            rr = json.loads(line)
            year_by_id.setdefault(rr["doc_id"], rr.get("year"))
    years = np.array([float(year_by_id[d]) if year_by_id.get(d) is not None else np.nan for d in ids])

    tv = NCL.T_VALUES.get(cell, {})
    agree_flag = (" -- NOTE: eval/test DIVERGE for agree (n_eval=505, docket-skewed, unstable "
                  "per notes/2026-07-27__vat-run-registry.md); report both, never one."
                  if cell == "agree" else "")
    return dict(tag=tag, n=int(len(y)), y=y, groups=groups, group_column="docket",
                ids=ids, texts=texts, years=years,
                va_nl=r["oof"], va_nl_auc=r["auc"], va_lin_auc=va_lin,
                T_score=None, T_mask=None,
                T_note=f"aggregate-only T on file (eval {tv.get('eval')} / test {tv.get('test')}, "
                       f"nc_multiy_results.json){agree_flag}; no row-level dense prediction file "
                       f"found locally or on sk3 rm_out (datasets/notice-and-comment/v4/dense_llama/"
                       f"{cell}/rm_out has adapter checkpoints only, no saved per-row scores) -- "
                       f"would require a new GPU scoring pass, out of Layer 2's CPU-only scope.",
                matrix=str(NC_DIR))


def load_cw_community():
    d = L1G.load_cw_community()
    mats, y, groups = d["mats"], d["y"], d["groups"]
    ids = list(L1G.rvg.load_bank("creative_writing")[0]["item_ids"])
    assert len(ids) == len(y)

    folds = L1G.outer_folds(len(y), groups, n_splits=5)
    r = L1G.gbm_oof_family1(mats["VA"], y, groups, folds, GBM_SEED)
    va_lin, _ = L1G.linear_oof_family1(mats["VA"], y, groups, folds)

    ledger_path = RESULTS / "cw_community_layer1.json"
    ledger_auc = json.loads(ledger_path.read_text())["ledger"]["VA_nl_mean"] if ledger_path.exists() else None
    _sanity("cw_community", r["auc"], ledger_auc)

    df = pd.read_csv(CW_TEXT_CSV)
    df["cand_id"] = df.apply(
        lambda row: f"{row['prompt_id']}_{hashlib.sha1(str(row['text']).encode()).hexdigest()[:10]}", axis=1)
    text_by_id = dict(zip(df["cand_id"], df["text"]))
    texts = [text_by_id.get(i, "") for i in ids]
    n_missing = sum(1 for t in texts if not t)
    if n_missing:
        print(f"  [cw_community] WARNING: {n_missing}/{len(ids)} ids had no text match")

    return dict(tag="cw_community", n=int(len(y)), y=y, groups=groups, group_column="prompt_id",
                ids=ids, texts=texts, years=None,
                va_nl=r["oof"], va_nl_auc=r["auc"], va_lin_auc=va_lin,
                T_score=None, T_mask=None,
                T_note="aggregate eval-only AUC (eval_pass_r2_results.json, .7801); no row-level "
                       "prediction file on sk3 (wp_clean_rm_out has adapter checkpoints only).",
                matrix="outputs/va_gemma_banks/creative_writing_shard*.npz")


def _caption_meta():
    rows = [json.loads(l) for l in open(CAP_POOL) if l.strip()]
    meta = {}
    for r in rows:
        did = f"{r['contest']}_{hashlib.sha1(r['text'].encode()).hexdigest()[:12]}"
        meta[did] = r
    return meta


def load_caption_cell(slug):
    d = L1G.CELLS[slug]["loader"]()
    mats, y, groups = d["mats"], d["y"], d["groups"]
    id_key = "crowd_ids" if slug == "cap_crowd" else "hardneg_ids"
    c = L1G._caption_pools()
    ids = sorted(x for x in c[id_key] if x in c["X_by_id"])
    assert len(ids) == len(y)

    folds = L1G.outer_folds(len(y), groups, n_splits=5)
    r = L1G.gbm_oof_raw(mats["VA"], y, groups, folds, GBM_SEED)
    va_lin, _ = L1G.linear_oof_family2(mats["VA"], y, groups, folds)

    ledger_path = RESULTS / f"{slug}_layer1.json"
    ledger_auc = json.loads(ledger_path.read_text())["ledger"]["VA_nl_mean"] if ledger_path.exists() else None
    _sanity(slug, r["auc"], ledger_auc)

    meta = _caption_meta()
    texts = [meta[i]["text"] if i in meta else "" for i in ids]
    n_missing = sum(1 for t in texts if not t)
    if n_missing:
        print(f"  [{slug}] WARNING: {n_missing}/{len(ids)} ids had no text match")
    T_dense = L1G.CELLS[slug]["T"]

    return dict(tag=slug, n=int(len(y)), y=y, groups=groups, group_column="contest",
                ids=ids, texts=texts, years=None,
                va_nl=r["oof"], va_nl_auc=r["auc"], va_lin_auc=va_lin,
                T_score=None, T_mask=None,
                T_note=f"aggregate eval-only AUC ({T_dense}, eval_pass_r2_results.json); no "
                       f"row-level prediction file on sk3 (dense_llama/{'crowd' if slug=='cap_crowd' else 'finalist'}/"
                       f"rm_out has adapter checkpoints only).",
                matrix="datasets/humor/caption_multiy/cap_scores_shard*.npz")


CELL_LOADERS = {
    "peer_verdict": lambda: load_peer_cell("verdict", "peer_verdict"),
    "nc_responded": lambda: load_nc_cell("responded"),
    "nc_outcome": lambda: load_nc_cell("outcome"),
    "cw_community": load_cw_community,
    "cap_crowd": lambda: load_caption_cell("cap_crowd"),
    "cap_finalist": lambda: load_caption_cell("cap_finalist"),
    "peer_curation": lambda: load_peer_cell("curation", "peer_curation"),
    "peer_revealed": lambda: load_peer_cell("revealed", "peer_revealed"),
    "nc_agree": lambda: load_nc_cell("agree"),
}
# original design-spec priority order (peer verdict, N&C responded/outcome,
# CW community, cap crowd/finalist, peer curation/revealed); nc_agree appended
# per coordinator directive 2026-08-06 (health-check message) after the
# original 8 were mostly complete.
PRIORITY = ["peer_verdict", "nc_responded", "nc_outcome", "cw_community",
            "cap_crowd", "cap_finalist", "peer_curation", "peer_revealed", "nc_agree"]
# remaining-cells order as explicitly sequenced by the coordinator's
# health-check message (nc_outcome first since it needed the NaN-impute fix):
REMAINING_SEQUENCE = ["nc_outcome", "cw_community", "cap_crowd", "cap_finalist",
                      "peer_curation", "peer_revealed", "nc_agree"]


# ==================================================================== run =
def run_cell(slug):
    t0 = time.time()
    print(f"\n=== layer2 {slug} ===")
    d = CELL_LOADERS[slug]()
    y, groups = d["y"], d["groups"]
    score_map = {"VA_nl": d["va_nl"], "T": d["T_score"]}

    part_a = {
        "VA_nl": grouped_transfer_block(y, groups, d["va_nl"]),
    }
    if d["T_score"] is not None:
        part_a["T"] = grouped_transfer_block(y, groups, d["T_score"], mask=d["T_mask"])

    score_masks = {"VA_nl": None}
    if d["T_score"] is not None:
        score_masks["T"] = d["T_mask"]
    part_b = nuisance_block(y, groups, score_map, d["texts"], years=d.get("years"),
                             score_masks=score_masks)

    res = {
        "cell": slug, "n": d["n"], "group_column": d["group_column"],
        "n_groups": int(len(np.unique(groups))), "pos_rate": float(y.mean()),
        "matrix": d["matrix"], "sklearn_version": sklearn.__version__,
        "va_nl_auc_recomputed_seed0": d["va_nl_auc"], "va_lin_auc_recomputed": d["va_lin_auc"],
        "T_available": d["T_score"] is not None, "T_note": d["T_note"],
        "T_n_valid": int(d["T_mask"].sum()) if d["T_mask"] is not None else 0,
        "part_a_grouped_transfer": part_a,
        "part_b_nuisance_stratified": part_b,
        "runtime_sec": None,
    }
    res["runtime_sec"] = time.time() - t0
    out_path = RESULTS / f"layer2_{slug}.json"
    out_path.write_text(json.dumps(res, indent=2, default=str))
    print(f"  wrote {out_path}  ({res['runtime_sec']:.1f}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=None, choices=list(CELL_LOADERS.keys()))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    slugs = PRIORITY if (args.all or args.cell is None) else [args.cell]
    for slug in slugs:
        try:
            run_cell(slug)
        except Exception as e:
            print(f"!!! {slug} FAILED: {type(e).__name__}: {e}")
            raise


if __name__ == "__main__":
    main()
