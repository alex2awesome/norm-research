#!/usr/bin/env python3
"""ROUND-0 AUDIT (CPU leg) of the patents claim-fell dense arm.

Confirmatory dual-track closure campaign, patents cell. Prereg:
notes/2026-08-05__layer3-closure-prereg.md (FREEZE DECLARATION + ADDENDA 1-3).
The dense arm being audited: notes/2026-08-06__dense-arms-hw-si-patents.md
(T = .7965 clean-eval / .8389 test, seed 42 only, PROVISIONAL).

This leg answers the round-0 audit questions that need no GPU:
  1. prediction distribution (collapse / calibration)
  2. per-app_id decomposition (how much of the AUC lives BETWEEN applications)
  3. class / era alone-AUCs (does a metadata channel alone explain the AUC?)
  4. length + nuisance correlations and alone-AUCs
  5. CONSTRUCTION AUDIT: the candidate-reference set is label-conditional
     (memory + notes/2026-07-07 forensic audit: pos rows = 7 same-CPC FAISS
     fillers + the examiner's gold reference APPENDED LAST; neg rows = fillers
     only, 0% gold). Measures the per-position channel directly.
  6. cross-split duplicate-text contamination
  7. dumps high-confidence rows for the manual read

Everything is fit on the dense model's own TRAIN split and evaluated on its
EVAL split, so every number is apples-to-apples with T = .7965.

CPU only. Writes JSON next to itself. Run on sk3.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path("/lfs/skampere3/0/alexspan/norm-research")
JL = BASE / "datasets/patents/processed/option3_claims_gemma_scale.jsonl"
DS = BASE / "datasets/patents/dense_standard"
VA_CSV = BASE / "notebooks/data/patents_va_features.csv"
LABELS_PQ = BASE / "datasets/patents/processed/labels.parquet"
CPC_JSON = Path("/lfs/skampere3/0/alexspan/tmp/appid_probe/app_cpc.json")
PATEX_JSON = Path("/lfs/skampere3/0/alexspan/tmp/appid_probe/patex_join.json")
OUT = Path(__file__).resolve().parent
csv.field_size_limit(sys.maxsize)

V_COLS = ["v_max_lexoverlap", "v_mean_lexoverlap", "v_count_lexhit", "v_element_wordlen",
          "v_n_refs", "v_max_spanlen", "v_mean_spanlen"]
A_COLS = ["a_n_disclose", "a_any_disclose", "a_frac_disclose", "a_max_disclose_overlap"]

DEP_RE = re.compile(r"\bof claim\s+\d+|\baccording to claim\s+\d+|\bas (?:recited|claimed) in claim", re.I)
WORD_RE = re.compile(r"[a-z0-9]+")
STOP = set("""a an the of and or to in for with at by on as is are be been being that this these those
from into such wherein comprising said which whose it its not""".split())


def toks(t):
    return [w for w in WORD_RE.findall((t or "").lower()) if w not in STOP and len(w) > 2]


def build_text(r):
    """VERBATIM copy of datasets/patents/build_dense_standard_claimfell.py::build_text."""
    parts = [f"CLAIM ELEMENT:\n{r['element']}"]
    for i, ref in enumerate(r.get("refs") or []):
        spans = " ".join(ref.get("spans") or [])
        parts.append(f"REFERENCE {i + 1} (patent {ref.get('doc_id', '?')}):\n{spans}")
    return "\n\n".join(parts)


# ------------------------------------------------------------------ features
def row_features(r):
    el = r["element"] or ""
    el_t = set(toks(el))
    refs = r.get("refs") or []
    ov, sl, isg = [], [], []
    for ref in refs:
        sp = " ".join(ref.get("spans") or [])
        st = set(toks(sp))
        ov.append(len(el_t & st) / max(len(el_t), 1))
        sl.append(len(sp))
        isg.append(bool(ref.get("is_gold")))
    n = len(refs)
    f = {
        "app_id": str(r["app_id"]),
        "claim_num": int(r["claim_num"]) if str(r["claim_num"]).lstrip("-").isdigit() else -1,
        "rejection_type": str(r.get("rejection_type")),
        "y": 1 if r["label"] == "pos" else 0,
        "n_refs": n,
        "n_disclose": int(r.get("n_disclose") or 0),
        "gold_disclose": int(bool(r.get("gold_disclose"))),
        "n_gold_docs": len(r.get("gold_docs") or []),
        "n_gold_in_refs": int(sum(isg)),
        "has_gold_in_refs": int(any(isg)),
        "gold_pos_idx": (isg.index(True) if any(isg) else -1),
        "gold_is_last": int(bool(isg) and isg[-1]),
        "el_chars": len(el), "el_words": len(el.split()),
        "is_dependent": int(bool(DEP_RE.search(el))),
        "series": str(r["app_id"])[:2],
        "text_chars": len(build_text(r)),
        "span_chars_total": int(sum(sl)),
        "span_chars_mean": float(np.mean(sl)) if sl else 0.0,
        "span_chars_last": float(sl[-1]) if sl else 0.0,
        "span_chars_mean_first7": float(np.mean(sl[:-1])) if len(sl) > 1 else 0.0,
        "ov_max": float(max(ov)) if ov else 0.0,
        "ov_mean": float(np.mean(ov)) if ov else 0.0,
        "ov_last": float(ov[-1]) if ov else 0.0,
        "ov_mean_first7": float(np.mean(ov[:-1])) if len(ov) > 1 else 0.0,
        "ov_argmax_pos": int(np.argmax(ov)) if ov else -1,
        "ov_argmax_is_last": int(bool(ov) and int(np.argmax(ov)) == len(ov) - 1),
    }
    f["ov_last_minus_rest"] = f["ov_last"] - f["ov_mean_first7"]
    f["span_last_minus_rest"] = f["span_chars_last"] - f["span_chars_mean_first7"]
    for k in range(8):  # per-position channels
        f[f"ov_p{k}"] = float(ov[k]) if k < len(ov) else np.nan
        f[f"sl_p{k}"] = float(sl[k]) if k < len(sl) else np.nan
    return f


# ------------------------------------------------------------------- helpers
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def fit_eval(tr, ev, cols, y_tr, y_ev, kind="hgb", seed=0):
    """Fit on the dense TRAIN split, score the dense EVAL split. Apples-to-apples with T."""
    Xtr = tr[cols].to_numpy(dtype=float)
    Xev = ev[cols].to_numpy(dtype=float)
    if kind == "hgb":
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           max_leaf_nodes=31, random_state=seed)
    else:
        Xtr = np.nan_to_num(Xtr); Xev = np.nan_to_num(Xev)
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    m.fit(Xtr, y_tr)
    return auc(y_ev, m.predict_proba(Xev)[:, 1])


def cat_alone_auc(tr, ev, col, y_tr, y_ev, prior_w=20.0):
    """Smoothed target encoding fit on TRAIN only -> alone-AUC on EVAL."""
    gm = float(np.mean(y_tr))
    d = defaultdict(lambda: [0.0, 0.0])
    for v, y in zip(tr[col].astype(str).values, y_tr):
        d[v][0] += y; d[v][1] += 1
    enc = {v: (s + prior_w * gm) / (n + prior_w) for v, (s, n) in d.items()}
    s = np.array([enc.get(v, gm) for v in ev[col].astype(str).values])
    return auc(y_ev, s), {k: round(v, 4) for k, v in sorted(enc.items(), key=lambda kv: -d[kv[0]][1])[:15]}, \
        {k: int(d[k][1]) for k in sorted(d, key=lambda k: -d[k][1])[:15]}


def pairwise_decomposition(df, score_col="prob", y_col="y", g_col="app_id", rng_seed=0, max_pairs=4_000_000):
    """Split the AUC's own (pos, neg) pair population into WITHIN-app and CROSS-app."""
    y = df[y_col].to_numpy(); s = df[score_col].to_numpy(dtype=float); g = df[g_col].to_numpy()
    pi = np.where(y == 1)[0]; ni = np.where(y == 0)[0]
    rng = np.random.default_rng(rng_seed)
    n_all = len(pi) * len(ni)
    if n_all > max_pairs:
        a = rng.choice(pi, max_pairs); b = rng.choice(ni, max_pairs)
    else:
        a = np.repeat(pi, len(ni)); b = np.tile(ni, len(pi))
    same = g[a] == g[b]
    conc = (s[a] > s[b]).astype(float) + 0.5 * (s[a] == s[b])
    out = {
        "n_pairs_sampled": int(len(a)),
        "frac_pairs_within_app": float(same.mean()),
        "auc_all_pairs": float(conc.mean()),
        "auc_within_app": float(conc[same].mean()) if same.any() else float("nan"),
        "n_within_pairs": int(same.sum()),
        "auc_cross_app": float(conc[~same].mean()) if (~same).any() else float("nan"),
        "n_cross_pairs": int((~same).sum()),
    }
    # oracle app-identity channel: score each row by its app's LEAVE-ONE-OUT mean y
    ssum = df.groupby(g_col)[y_col].transform("sum").to_numpy(dtype=float)
    scnt = df.groupby(g_col)[y_col].transform("count").to_numpy(dtype=float)
    loo = np.where(scnt > 1, (ssum - y) / np.maximum(scnt - 1, 1), np.mean(y))
    out["auc_app_identity_oracle_LOO"] = auc(y, loo)
    out["app_size_mean"] = float(scnt.mean())
    out["frac_rows_in_singleton_apps"] = float((scnt == 1).mean())
    return out


def calibration(y, p, nbins=10):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    q = np.quantile(p, np.linspace(0, 1, nbins + 1))
    q[0] -= 1e-9; q[-1] += 1e-9
    b = np.clip(np.digitize(p, q[1:-1]), 0, nbins - 1)
    rows, ece = [], 0.0
    for k in range(nbins):
        m = b == k
        if not m.any():
            continue
        rows.append({"bin": k, "n": int(m.sum()), "mean_pred": round(float(p[m].mean()), 4),
                     "obs_rate": round(float(y[m].mean()), 4)})
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"bins": rows, "ECE": round(float(ece), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "base_rate": round(float(y.mean()), 4)}


# ---------------------------------------------------------------------- main
def main():
    print("loading jsonl ...", flush=True)
    jrows = [json.loads(l) for l in open(JL) if l.strip()]
    print(f"  {len(jrows)} rows", flush=True)

    feats = [row_features(r) for r in jrows]
    F = pd.DataFrame(feats)
    F["jidx"] = np.arange(len(F))

    # ---- technology class + filing year, 100% app_id coverage -------------
    cpc = json.load(open(CPC_JSON))          # app_id -> [seq, section, class, subclass, group]
    px = json.load(open(PATEX_JSON))         # app_id -> [filing_date, uspc_class, uspc_sub, art_unit, pgpub]
    import pyarrow.parquet as pq
    lab = pq.read_table(LABELS_PQ, columns=["application_number", "filing_year"]).to_pandas()
    lab["application_number"] = lab["application_number"].astype(str)
    yr = dict(zip(lab["application_number"], lab["filing_year"]))
    del lab
    F["cpc_section"] = [cpc.get(a, [None] * 5)[1] or "NA" for a in F["app_id"]]
    F["cpc_class"] = [cpc.get(a, [None] * 5)[2] or "NA" for a in F["app_id"]]
    F["cpc_subclass"] = [cpc.get(a, [None] * 5)[3] or "NA" for a in F["app_id"]]
    F["uspc_class"] = [str(px.get(a, [None] * 5)[1] or "NA") for a in F["app_id"]]
    F["art_unit"] = [str(px.get(a, [None] * 5)[3] or "NA") for a in F["app_id"]]
    F["art_tc"] = [s[:2] for s in F["art_unit"]]
    F["filing_year"] = [float(yr.get(a, np.nan)) if yr.get(a) == yr.get(a) else np.nan
                        for a in F["app_id"]]
    F["filing_year_s"] = [("NA" if v != v else str(int(v))) for v in F["filing_year"]]
    cov = {c: round(float((F[c] != "NA").mean()), 4)
           for c in ("cpc_section", "cpc_class", "cpc_subclass", "uspc_class", "art_unit",
                     "filing_year_s")}
    print(f"metadata coverage: {cov}", flush=True)
    thash = [hashlib.sha1(build_text(r).encode()).hexdigest() for r in jrows]
    F["thash"] = thash
    F["ehash"] = [hashlib.sha1((r["element"] or "").encode()).hexdigest() for r in jrows]

    # ---- align split rows back to jsonl rows (split preserves jsonl order) ----
    by_hash = defaultdict(list)
    for i, h in enumerate(thash):
        by_hash[h].append(i)
    ptr = defaultdict(int)
    split_idx = {}
    for split in ("train", "eval", "test"):
        d = pd.read_csv(DS / "split" / f"{split}.csv")
        idxs = []
        for t in d["text"].astype(str).values:
            h = hashlib.sha1(t.encode()).hexdigest()
            lst = by_hash.get(h)
            if not lst:
                idxs.append(-1); continue
            k = ptr[h]
            idxs.append(lst[k] if k < len(lst) else lst[-1])
            ptr[h] = k + 1
        split_idx[split] = np.array(idxs)
        miss = int((np.array(idxs) < 0).sum())
        print(f"  {split}: {len(idxs)} rows, {miss} unmatched", flush=True)
        assert miss == 0, f"{split} alignment failed"
    # sanity: labels must agree after alignment
    for split in ("train", "eval", "test"):
        d = pd.read_csv(DS / "split" / f"{split}.csv")
        assert (F["y"].to_numpy()[split_idx[split]] == d["judgement"].to_numpy()).all(), \
            f"{split} label mismatch after alignment"
    print("ALIGNMENT GATE PASS (0 unmatched, 0 label mismatches)", flush=True)

    tr = F.iloc[split_idx["train"]].reset_index(drop=True)
    ev = F.iloc[split_idx["eval"]].reset_index(drop=True)
    te = F.iloc[split_idx["test"]].reset_index(drop=True)
    y_tr, y_ev, y_te = tr["y"].to_numpy(), ev["y"].to_numpy(), te["y"].to_numpy()

    # ---- dense predictions (seed 42; extra seeds picked up if present) ----
    preds = {}
    for run in sorted((DS).glob("rm_out_seed*")):
        if not run.is_dir():
            continue
        seed = run.name.replace("rm_out_seed", "")
        for split, frame in (("eval", ev), ("test", te)):
            f = run / f"preds_{split}.csv"
            if not f.exists():
                continue
            p = pd.read_csv(f)
            assert len(p) == len(frame), f"{f} length mismatch"
            assert (p["judgement"].to_numpy() == frame["y"].to_numpy()).all(), f"{f} label order mismatch"
            preds[(seed, split)] = p["prob"].to_numpy(dtype=float)
    ev["prob"] = preds[("42", "eval")]
    te["prob"] = preds[("42", "test")]

    R = {"cell": "patents claim-fell", "n_rows": len(F),
         "splits": {k: int(len(v)) for k, v in split_idx.items()},
         "n_apps": {k: int(F.iloc[v]["app_id"].nunique()) for k, v in split_idx.items()}}

    # =============== 1. prediction distribution / collapse / calibration ====
    R["dense_seed42"] = {}
    for split, frame in (("eval", ev), ("test", te)):
        p = frame["prob"].to_numpy()
        R["dense_seed42"][split] = {
            "auc": round(auc(frame["y"], p), 4),
            "n": int(len(p)),
            "mean": round(float(p.mean()), 4), "std": round(float(p.std()), 4),
            "min": round(float(p.min()), 4), "max": round(float(p.max()), 4),
            "q": {str(q): round(float(np.quantile(p, q)), 4)
                  for q in (.01, .05, .25, .5, .75, .95, .99)},
            "frac_lt_.05": round(float((p < .05).mean()), 4),
            "frac_gt_.95": round(float((p > .95).mean()), 4),
            "n_unique": int(len(np.unique(np.round(p, 6)))),
            "hist20": np.histogram(p, bins=20, range=(0, 1))[0].tolist(),
            "calibration": calibration(frame["y"], p),
        }
    R["dense_seed42"]["seed42_training_history_note"] = (
        "checkpoint selected on the EVAL split (--selection_split eval), so the eval AUC "
        "carries a max-of-2-epochs selection bias; TEST is selection-free.")

    # =============== 2. per-app_id decomposition ===========================
    R["per_app_decomposition"] = {
        "eval": pairwise_decomposition(ev), "test": pairwise_decomposition(te)}
    for split, frame in (("eval", ev), ("test", te)):
        sz = frame.groupby("app_id").size()
        R["per_app_decomposition"][split].update({
            "n_apps": int(frame["app_id"].nunique()),
            "rows_per_app_mean": round(float(sz.mean()), 3),
            "rows_per_app_max": int(sz.max()),
            "frac_apps_label_pure": round(float(
                frame.groupby("app_id")["y"].nunique().eq(1).mean()), 4),
        })

    # =============== 2b. dense AUC STRATIFIED by class / era / rejection ====
    def strat_auc(frame, col, score="prob", min_n=30):
        rows, num, den = [], 0.0, 0
        for v, sub in frame.groupby(col):
            if len(sub) < min_n or sub["y"].nunique() < 2:
                continue
            a = auc(sub["y"], sub[score]); rows.append({"level": str(v), "n": int(len(sub)),
                                                        "pos_rate": round(float(sub["y"].mean()), 3),
                                                        "auc": round(a, 4)})
            num += a * len(sub); den += len(sub)
        rows.sort(key=lambda r: -r["n"])
        return {"n_weighted_within_stratum_auc": round(num / den, 4) if den else None,
                "n_rows_covered": int(den), "levels": rows[:12]}

    R["dense_auc_stratified"] = {
        c: strat_auc(ev, c) for c in
        ("cpc_section", "cpc_subclass", "filing_year_s", "art_tc", "rejection_type", "app_id")}
    R["dense_auc_stratified"]["_note"] = (
        "n-weighted mean of WITHIN-stratum AUCs on EVAL. If the pooled .7965 were driven by a "
        "class/era channel, these would collapse toward .5.")

    # =============== 3. class / era alone-AUCs (the leak gate) =============
    lk = {"metadata_coverage": cov,
          "GATE": "STOP after round 0 if any class/era channel alone reaches >= .75 on EVAL"}
    for col in ("cpc_section", "cpc_class", "cpc_subclass", "uspc_class", "art_tc", "art_unit",
                "filing_year_s", "series", "rejection_type"):
        a, enc, cnt = cat_alone_auc(tr, ev, col, y_tr, y_ev)
        lk[f"{col}_alone_eval"] = round(a, 4)
        if col in ("cpc_section", "filing_year_s", "rejection_type"):
            lk[f"{col}_train_target_encoding_top15"] = enc
            lk[f"{col}_train_counts_top15"] = cnt
    for a_col, b_col, nm in (("cpc_subclass", "filing_year_s", "cpcsub_x_year"),
                             ("cpc_section", "rejection_type", "cpcsec_x_rejection"),
                             ("art_unit", "filing_year_s", "artunit_x_year")):
        tr2 = tr.assign(_j=tr[a_col].astype(str) + "|" + tr[b_col].astype(str))
        ev2 = ev.assign(_j=ev[a_col].astype(str) + "|" + ev[b_col].astype(str))
        lk[f"{nm}_alone_eval"] = round(cat_alone_auc(tr2, ev2, "_j", y_tr, y_ev)[0], 4)
    lk["filing_year_numeric_alone_eval"] = round(auc(y_ev, np.nan_to_num(
        ev["filing_year"].to_numpy(dtype=float), nan=float(np.nanmean(tr["filing_year"])))), 4)
    lk["claim_num_alone_eval"] = round(fit_eval(tr, ev, ["claim_num"], y_tr, y_ev), 4)
    lk["is_dependent_alone_eval"] = round(auc(y_ev, ev["is_dependent"]), 4)
    # full non-text metadata model: the ceiling of the "no text needed" channel
    for c in ("cpc_subclass", "uspc_class", "art_unit", "filing_year_s", "rejection_type"):
        a, enc, _ = cat_alone_auc(tr, ev, c, y_tr, y_ev)
        gm = float(np.mean(y_tr))
        d = defaultdict(lambda: [0.0, 0.0])
        for v, yy in zip(tr[c].astype(str).values, y_tr):
            d[v][0] += yy; d[v][1] += 1
        e2 = {v: (s + 20.0 * gm) / (n + 20.0) for v, (s, n) in d.items()}
        tr[f"te_{c}"] = [e2.get(v, gm) for v in tr[c].astype(str).values]
        ev[f"te_{c}"] = [e2.get(v, gm) for v in ev[c].astype(str).values]
    meta_cols = [f"te_{c}" for c in ("cpc_subclass", "uspc_class", "art_unit", "filing_year_s",
                                     "rejection_type")] + ["filing_year", "claim_num",
                                                           "is_dependent", "n_refs"]
    lk["ALL_metadata_no_text_hgb_eval"] = round(fit_eval(tr, ev, meta_cols, y_tr, y_ev), 4)
    lk["ALL_metadata_no_text_note"] = (
        "target encodings fit on TRAIN only (smoothing prior 20); slight optimism from "
        "encoding-on-train is possible but splits are app_id-disjoint.")
    R["class_era_alone"] = lk

    # =============== 4. length + nuisance channels ==========================
    nu = {}
    for c in ("text_chars", "el_chars", "el_words", "n_refs", "span_chars_total",
              "span_chars_mean", "claim_num", "is_dependent"):
        nu[c] = {"alone_auc_eval": round(auc(y_ev, ev[c]), 4),
                 "spearman_with_dense_prob": round(float(
                     pd.Series(ev[c]).corr(pd.Series(ev["prob"]), method="spearman")), 4)}
    nu["_joint_length_metadata_hgb_eval"] = round(fit_eval(
        tr, ev, ["text_chars", "el_chars", "el_words", "n_refs", "span_chars_total",
                 "span_chars_mean", "claim_num", "is_dependent"], y_tr, y_ev), 4)
    R["nuisance_channels"] = nu

    # =============== 5. CONSTRUCTION AUDIT: the reference-set asymmetry =====
    ca = {}
    ca["n_refs_distribution"] = {str(k): int(v) for k, v in Counter(F["n_refs"]).most_common(10)}
    ca["has_gold_in_refs_by_label"] = {
        "pos": round(float(F.loc[F.y == 1, "has_gold_in_refs"].mean()), 4),
        "neg": round(float(F.loc[F.y == 0, "has_gold_in_refs"].mean()), 4)}
    ca["has_gold_alone_auc_population"] = round(auc(F["y"], F["has_gold_in_refs"]), 4)
    ca["gold_position_index_distribution_pos_rows"] = {
        str(k): int(v) for k, v in Counter(F.loc[F.y == 1, "gold_pos_idx"]).most_common(12)}
    ca["gold_is_last_rate_pos_rows"] = round(float(F.loc[F.y == 1, "gold_is_last"].mean()), 4)
    ca["n_gold_docs_by_label"] = {
        "pos_mean": round(float(F.loc[F.y == 1, "n_gold_docs"].mean()), 4),
        "neg_mean": round(float(F.loc[F.y == 0, "n_gold_docs"].mean()), 4)}
    # per-position channels, fit on TRAIN -> EVAL
    ca["ov_per_position_hgb_eval"] = round(fit_eval(
        tr, ev, [f"ov_p{k}" for k in range(8)], y_tr, y_ev), 4)
    ca["spanlen_per_position_hgb_eval"] = round(fit_eval(
        tr, ev, [f"sl_p{k}" for k in range(8)], y_tr, y_ev), 4)
    ca["ov_AND_spanlen_per_position_hgb_eval"] = round(fit_eval(
        tr, ev, [f"ov_p{k}" for k in range(8)] + [f"sl_p{k}" for k in range(8)], y_tr, y_ev), 4)
    ca["last_slot_contrast_features_hgb_eval"] = round(fit_eval(
        tr, ev, ["ov_last", "ov_mean_first7", "ov_last_minus_rest",
                 "span_chars_last", "span_chars_mean_first7", "span_last_minus_rest"],
        y_tr, y_ev), 4)
    ca["ov_last_minus_rest_alone_eval"] = round(auc(y_ev, ev["ov_last_minus_rest"]), 4)
    ca["span_last_minus_rest_alone_eval"] = round(auc(y_ev, ev["span_last_minus_rest"]), 4)
    ca["ov_argmax_is_last_alone_eval"] = round(auc(y_ev, ev["ov_argmax_is_last"]), 4)
    ca["ORDER_INVARIANT_control_hgb_eval"] = round(fit_eval(
        tr, ev, ["ov_max", "ov_mean", "span_chars_mean", "span_chars_total", "n_refs",
                 "el_words"], y_tr, y_ev), 4)
    ca["per_position_mean_overlap_by_label"] = {
        lbl: {f"p{k}": round(float(F.loc[F.y == v, f"ov_p{k}"].mean()), 4) for k in range(8)}
        for lbl, v in (("pos", 1), ("neg", 0))}
    ca["per_position_mean_spanlen_by_label"] = {
        lbl: {f"p{k}": round(float(F.loc[F.y == v, f"sl_p{k}"].mean()), 1) for k in range(8)}
        for lbl, v in (("pos", 1), ("neg", 0))}
    R["construction_audit"] = ca

    # =============== 6. duplicate-text contamination across splits =========
    sp_of = np.empty(len(F), dtype=object)
    for s, idxs in split_idx.items():
        sp_of[idxs] = s
    F["split"] = sp_of
    dup = {}
    for key, name in (("thash", "full_text"), ("ehash", "claim_element")):
        g = F.groupby(key)["split"].agg(lambda x: len(set(x)))
        multi = g[g > 1]
        n_rows = int(F[key].isin(multi.index).sum())
        sub = F[F[key].isin(multi.index)]
        contradict = int(sub.groupby(key)["y"].nunique().gt(1).sum())
        dup[name] = {"n_hash_groups_spanning_splits": int(len(multi)),
                     "n_rows_involved": n_rows,
                     "frac_of_corpus": round(n_rows / len(F), 4),
                     "n_such_groups_with_contradictory_labels": contradict}
        ev_dup = ev[key].isin(multi.index).to_numpy()
        dup[name]["eval_rows_with_a_twin_in_train_or_test"] = int(ev_dup.sum())
        if ev_dup.any() and (~ev_dup).any():
            dup[name]["dense_auc_on_eval_dup_rows"] = round(auc(ev["y"][ev_dup], ev["prob"][ev_dup]), 4)
            dup[name]["dense_auc_on_eval_clean_rows"] = round(
                auc(ev["y"][~ev_dup], ev["prob"][~ev_dup]), 4)
    R["duplicate_contamination"] = dup

    # =============== 7. V / A matrix comparison on the SAME eval rows =======
    va = pd.read_csv(VA_CSV)
    assert len(va) == len(F), "VA csv row count mismatch"
    assert (va["fell"].to_numpy() == F["y"].to_numpy()).all(), "VA csv label alignment mismatch"
    va_tr = va.iloc[split_idx["train"]].reset_index(drop=True)
    va_ev = va.iloc[split_idx["eval"]].reset_index(drop=True)
    R["va_on_same_split"] = {
        "V_hgb_eval": round(fit_eval(va_tr, va_ev, V_COLS, y_tr, y_ev), 4),
        "VA_hgb_eval": round(fit_eval(va_tr, va_ev, V_COLS + A_COLS, y_tr, y_ev), 4),
        "VA_linear_eval": round(fit_eval(va_tr, va_ev, V_COLS + A_COLS, y_tr, y_ev, kind="lin"), 4),
        "A_only_hgb_eval": round(fit_eval(va_tr, va_ev, A_COLS, y_tr, y_ev), 4),
        "note": ("fit on the dense TRAIN split and scored on the dense EVAL split -- NOT the "
                 "Layer-1 grouped-OOF protocol number (VA_nl .6256); this is the same-split "
                 "comparison that makes Delta_beyond apples-to-apples with T."),
    }

    # =============== 8. manual-read dump ===================================
    ev_srt = ev.assign(_i=np.arange(len(ev))).sort_values("prob")
    pick = {
        "high_conf_correct_pos": ev_srt[ev_srt.y == 1].tail(5)["_i"].tolist(),
        "high_conf_correct_neg": ev_srt[ev_srt.y == 0].head(5)["_i"].tolist(),
        "high_conf_error_pred_pos_true_neg": ev_srt[ev_srt.y == 0].tail(3)["_i"].tolist(),
        "high_conf_error_pred_neg_true_pos": ev_srt[ev_srt.y == 1].head(3)["_i"].tolist(),
    }
    ev_txt = pd.read_csv(DS / "split" / "eval.csv")
    dump = []
    for bucket, idxs in pick.items():
        for i in idxs:
            j = int(ev.loc[i, "jidx"])
            r = jrows[j]
            dump.append({
                "bucket": bucket, "eval_row": int(i), "app_id": ev.loc[i, "app_id"],
                "claim_num": int(ev.loc[i, "claim_num"]),
                "rejection_type": ev.loc[i, "rejection_type"],
                "y": int(ev.loc[i, "y"]), "dense_prob": round(float(ev.loc[i, "prob"]), 4),
                "n_refs": int(ev.loc[i, "n_refs"]),
                "has_gold_in_refs": int(ev.loc[i, "has_gold_in_refs"]),
                "gold_pos_idx": int(ev.loc[i, "gold_pos_idx"]),
                "n_disclose": int(ev.loc[i, "n_disclose"]),
                "gold_disclose": int(ev.loc[i, "gold_disclose"]),
                "element": r["element"][:900],
                "refs": [{"doc_id": q.get("doc_id"), "is_gold": bool(q.get("is_gold")),
                          "discloses": bool(q.get("discloses")),
                          "span_chars": len(" ".join(q.get("spans") or [])),
                          "span": (" ".join(q.get("spans") or []))[:400]}
                         for q in (r.get("refs") or [])],
                "text_head": str(ev_txt.loc[i, "text"])[:400],
            })
    json.dump(dump, open(OUT / "round0_manual_read.json", "w"), indent=2)
    R["manual_read_dump"] = "round0_manual_read.json (16 rows)"

    json.dump(R, open(OUT / "round0_audit_cpu.json", "w"), indent=2)
    print(json.dumps(R, indent=2)[:12000], flush=True)
    print("ROUND0_AUDIT_CPU_DONE", flush=True)


if __name__ == "__main__":
    main()
