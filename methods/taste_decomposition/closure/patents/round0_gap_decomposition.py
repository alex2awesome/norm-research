#!/usr/bin/env python3
"""ROUND-0 AUDIT, part 2: what fraction of Delta_beyond = T - VA does a NAMEABLE
structural channel already claim?

round0_audit_cpu.py found that a joint model over pure structure -- claim ordinal
number, element/span/text lengths, dependent-claim flag, reference count, with NO
text content at all -- reaches .7626 on the dense EVAL split, against the dense
model's .7965 and the V+A bank's .6214 on the same split. This script pins that
down with the freeze's own readouts, applied at round 0:

  * the claim_num channel: shape, direction, per-value positive rate
  * STACKED INCREMENT (FREEZE ADDENDUM): AUC(structure) vs AUC(structure + dense)
    and vs AUC(bank + dense) -- the dense increment over all named channels
  * bank + structure: does simply adding the structural columns to the V+A matrix
    close Delta_beyond? (the "how much of the +.17 is un-banked structure" number)
  * decile-stratified and matched-sampling discounts of T and VA on the structural
    joint score (the prereg's Track-B discount machinery)
  * the same decomposition on TEST (selection-free)

CPU only. Run on sk3.
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
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path("/lfs/skampere3/0/alexspan/norm-research")
JL = BASE / "datasets/patents/processed/option3_claims_gemma_scale.jsonl"
DS = BASE / "datasets/patents/dense_standard"
VA_CSV = BASE / "notebooks/data/patents_va_features.csv"
OUT = Path(__file__).resolve().parent
csv.field_size_limit(sys.maxsize)

V_COLS = ["v_max_lexoverlap", "v_mean_lexoverlap", "v_count_lexhit", "v_element_wordlen",
          "v_n_refs", "v_max_spanlen", "v_mean_spanlen"]
A_COLS = ["a_n_disclose", "a_any_disclose", "a_frac_disclose", "a_max_disclose_overlap"]
STRUCT = ["claim_num", "is_dependent", "el_chars", "el_words", "text_chars",
          "span_chars_total", "span_chars_mean", "n_refs"]
DEP_RE = re.compile(r"\bof claim\s+\d+|\baccording to claim\s+\d+|\bas (?:recited|claimed) in claim", re.I)


def build_text(r):
    parts = [f"CLAIM ELEMENT:\n{r['element']}"]
    for i, ref in enumerate(r.get("refs") or []):
        parts.append(f"REFERENCE {i + 1} (patent {ref.get('doc_id', '?')}):\n"
                     f"{' '.join(ref.get('spans') or [])}")
    return "\n\n".join(parts)


def auc(y, s):
    return float(roc_auc_score(np.asarray(y), np.asarray(s, dtype=float)))


def fit_score(tr, ev, te, cols, y_tr, kind="hgb", seed=0):
    Xtr, Xev, Xte = (d[cols].to_numpy(dtype=float) for d in (tr, ev, te))
    if kind == "hgb":
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           max_leaf_nodes=31, random_state=seed)
    else:
        Xtr, Xev, Xte = (np.nan_to_num(x) for x in (Xtr, Xev, Xte))
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    m.fit(Xtr, y_tr)
    return m.predict_proba(Xev)[:, 1], m.predict_proba(Xte)[:, 1]


def decile_strat_auc(y, score, strat, nbins=10):
    """n-weighted mean of within-stratum AUCs (the prereg's discount estimator)."""
    q = np.quantile(strat, np.linspace(0, 1, nbins + 1)); q[0] -= 1e-9; q[-1] += 1e-9
    b = np.clip(np.digitize(strat, q[1:-1]), 0, nbins - 1)
    num, den = 0.0, 0
    for k in range(nbins):
        m = b == k
        if m.sum() < 20 or len(set(y[m].tolist())) < 2:
            continue
        num += auc(y[m], score[m]) * m.sum(); den += m.sum()
    return round(num / den, 4) if den else None


def matched_sampling_auc(y, score, strat, n_draw=200000, seed=0, tol=None):
    """Match pos/neg pairs on the nuisance score, then measure concordance."""
    y = np.asarray(y); score = np.asarray(score, float); strat = np.asarray(strat, float)
    if tol is None:
        tol = 0.02 * (np.nanmax(strat) - np.nanmin(strat))
    rng = np.random.default_rng(seed)
    pi = np.where(y == 1)[0]; ni = np.where(y == 0)[0]
    order = np.argsort(strat[ni]); ns = ni[order]; nv = strat[ns]
    a = rng.choice(pi, n_draw)
    lo = np.searchsorted(nv, strat[a] - tol, "left")
    hi = np.searchsorted(nv, strat[a] + tol, "right")
    ok = hi > lo
    pick = np.where(ok, lo + (rng.random(n_draw) * np.maximum(hi - lo, 1)).astype(int), 0)
    b = ns[np.clip(pick, 0, len(ns) - 1)]
    a, b = a[ok], b[ok]
    conc = (score[a] > score[b]).astype(float) + 0.5 * (score[a] == score[b])
    return round(float(conc.mean()), 4), int(ok.sum())


def main():
    jrows = [json.loads(l) for l in open(JL) if l.strip()]
    thash = defaultdict(list)
    for i, r in enumerate(jrows):
        thash[hashlib.sha1(build_text(r).encode()).hexdigest()].append(i)
    ptr = defaultdict(int); split_idx = {}
    for split in ("train", "eval", "test"):
        d = pd.read_csv(DS / "split" / f"{split}.csv")
        ix = []
        for t in d["text"].astype(str).values:
            h = hashlib.sha1(t.encode()).hexdigest(); lst = thash[h]; k = ptr[h]
            ix.append(lst[k] if k < len(lst) else lst[-1]); ptr[h] = k + 1
        split_idx[split] = np.array(ix)

    rows = []
    for r in jrows:
        el = r["element"] or ""
        refs = r.get("refs") or []
        sl = [len(" ".join(q.get("spans") or [])) for q in refs]
        rows.append({"y": 1 if r["label"] == "pos" else 0, "app_id": str(r["app_id"]),
                     "claim_num": int(r["claim_num"]) if str(r["claim_num"]).lstrip("-").isdigit() else -1,
                     "is_dependent": int(bool(DEP_RE.search(el))),
                     "el_chars": len(el), "el_words": len(el.split()),
                     "text_chars": len(build_text(r)), "n_refs": len(refs),
                     "span_chars_total": int(sum(sl)),
                     "span_chars_mean": float(np.mean(sl)) if sl else 0.0})
    F = pd.DataFrame(rows)
    va = pd.read_csv(VA_CSV)
    assert (va["fell"].to_numpy() == F["y"].to_numpy()).all()
    F = pd.concat([F, va[V_COLS + A_COLS]], axis=1)

    tr = F.iloc[split_idx["train"]].reset_index(drop=True)
    ev = F.iloc[split_idx["eval"]].reset_index(drop=True)
    te = F.iloc[split_idx["test"]].reset_index(drop=True)
    y_tr, y_ev, y_te = tr.y.to_numpy(), ev.y.to_numpy(), te.y.to_numpy()
    dense_ev = pd.read_csv(DS / "rm_out_seed42/preds_eval.csv")["prob"].to_numpy()
    dense_te = pd.read_csv(DS / "rm_out_seed42/preds_test.csv")["prob"].to_numpy()

    R = {"note": "all models fit on the dense TRAIN split, scored on EVAL and TEST -- "
                 "apples-to-apples with T. Layer-1's VA_nl .6256 is a different "
                 "(grouped-OOF, full-population) protocol; the same-split VA here is the "
                 "comparable quantity."}

    # ---- the claim_num channel, described --------------------------------
    cn = {}
    cn["pos_rate_by_claim_num"] = {
        str(int(k)): [round(float(v.mean()), 3), int(len(v))]
        for k, v in F.groupby("claim_num")["y"] if len(v) >= 200}
    cn["claim_num_quantiles_by_label"] = {
        lbl: {str(q): float(np.quantile(F.loc[F.y == v, "claim_num"], q))
              for q in (.1, .25, .5, .75, .9)} for lbl, v in (("pos", 1), ("neg", 0))}
    cn["mean_claim_num"] = {"pos": round(float(F.loc[F.y == 1, "claim_num"].mean()), 2),
                            "neg": round(float(F.loc[F.y == 0, "claim_num"].mean()), 2)}
    cn["claim_num_alone_auc_eval_INVERTED"] = round(auc(y_ev, -ev["claim_num"].to_numpy()), 4)
    cn["claim_num_alone_auc_test_INVERTED"] = round(auc(y_te, -te["claim_num"].to_numpy()), 4)
    cn["is_dependent_pos_rate"] = {
        "dependent": round(float(F.loc[F.is_dependent == 1, "y"].mean()), 4),
        "independent": round(float(F.loc[F.is_dependent == 0, "y"].mean()), 4)}
    cn["spearman_claimnum_vs_dense_prob"] = {
        "eval": round(float(pd.Series(ev.claim_num).corr(pd.Series(dense_ev), method="spearman")), 4),
        "test": round(float(pd.Series(te.claim_num).corr(pd.Series(dense_te), method="spearman")), 4)}
    R["claim_num_channel"] = cn

    # ---- the ladder ------------------------------------------------------
    blocks = {
        "V": V_COLS, "A": A_COLS, "VA": V_COLS + A_COLS,
        "STRUCT": STRUCT, "claim_num_only": ["claim_num"],
        "STRUCT_minus_claimnum": [c for c in STRUCT if c != "claim_num"],
        "VA_plus_STRUCT": V_COLS + A_COLS + STRUCT,
        "VA_plus_claimnum": V_COLS + A_COLS + ["claim_num"],
    }
    scores_ev, scores_te, ladder = {}, {}, {}
    for name, cols in blocks.items():
        se, st = fit_score(tr, ev, te, cols, y_tr)
        scores_ev[name], scores_te[name] = se, st
        ladder[name] = {"eval": round(auc(y_ev, se), 4), "test": round(auc(y_te, st), 4),
                        "n_cols": len(cols)}
    ladder["T_dense_seed42"] = {"eval": round(auc(y_ev, dense_ev), 4),
                                "test": round(auc(y_te, dense_te), 4), "n_cols": None}
    R["ladder"] = ladder

    # ---- gap accounting --------------------------------------------------
    ga = {}
    for sp, yy, dd in (("eval", y_ev, dense_ev), ("test", y_te, dense_te)):
        S = scores_ev if sp == "eval" else scores_te
        T, VA = auc(yy, dd), auc(yy, S["VA"])
        gap = T - VA
        ga[sp] = {
            "T": round(T, 4), "VA_same_split": round(VA, 4), "gap_T_minus_VA": round(gap, 4),
            "VA_plus_STRUCT": round(auc(yy, S["VA_plus_STRUCT"]), 4),
            "frac_of_gap_closed_by_STRUCT": round((auc(yy, S["VA_plus_STRUCT"]) - VA) / gap, 3),
            "VA_plus_claimnum": round(auc(yy, S["VA_plus_claimnum"]), 4),
            "frac_of_gap_closed_by_claim_num_ALONE": round(
                (auc(yy, S["VA_plus_claimnum"]) - VA) / gap, 3),
            "STRUCT_alone": round(auc(yy, S["STRUCT"]), 4),
            "residual_T_minus_VA_plus_STRUCT": round(T - auc(yy, S["VA_plus_STRUCT"]), 4),
        }
    R["gap_accounting"] = ga

    # ---- stacked increment (FREEZE ADDENDUM) -----------------------------
    def stack(base_ev, base_te):
        Xe = np.column_stack([base_ev, dense_ev]); Xt = np.column_stack([base_te, dense_te])
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        # fit the stack on TEST, read on EVAL and vice versa (no in-sample stacking)
        m.fit(Xt, y_te); a_ev = auc(y_ev, m.predict_proba(Xe)[:, 1])
        m2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        m2.fit(Xe, y_ev); a_te = auc(y_te, m2.predict_proba(Xt)[:, 1])
        return round(a_ev, 4), round(a_te, 4)
    si = {}
    for nm in ("STRUCT", "VA", "VA_plus_STRUCT"):
        se, st = stack(scores_ev[nm], scores_te[nm])
        si[nm] = {"base_eval": round(auc(y_ev, scores_ev[nm]), 4), "stacked_eval": se,
                  "increment_eval": round(se - auc(y_ev, scores_ev[nm]), 4),
                  "base_test": round(auc(y_te, scores_te[nm]), 4), "stacked_test": st,
                  "increment_test": round(st - auc(y_te, scores_te[nm]), 4)}
    si["_note"] = ("stack weights fit on the OTHER split (eval<->test) so the increment is "
                   "never read in-sample.")
    R["stacked_increment"] = si

    # ---- discounts on the structural channel -----------------------------
    dc = {}
    for sp, yy, dd in (("eval", y_ev, dense_ev), ("test", y_te, dense_te)):
        S = scores_ev if sp == "eval" else scores_te
        s = S["STRUCT"]
        d = {"strat_by": "joint STRUCT score (decile) / matched sampling",
             "STRUCT_alone_auc": round(auc(yy, s), 4)}
        d["T_undiscounted"] = round(auc(yy, dd), 4)
        d["VA_undiscounted"] = round(auc(yy, S["VA"]), 4)
        d["T_decile_strat"] = decile_strat_auc(yy, dd, s)
        d["VA_decile_strat"] = decile_strat_auc(yy, S["VA"], s)
        d["Delta_decile_strat"] = round(d["T_decile_strat"] - d["VA_decile_strat"], 4)
        d["T_matched"], d["n_matched"] = matched_sampling_auc(yy, dd, s)
        d["VA_matched"], _ = matched_sampling_auc(yy, S["VA"], s)
        d["Delta_matched"] = round(d["T_matched"] - d["VA_matched"], 4)
        dc[sp] = d
    R["struct_discount"] = dc

    # ---- group-level bootstrap on the key quantities ---------------------
    def boot(yy, sa, sb, groups, n=1000, seed=0):
        rng = np.random.default_rng(seed)
        gs = pd.Series(range(len(yy))).groupby(pd.Series(groups)).apply(list).values
        out = []
        for _ in range(n):
            pick = rng.integers(0, len(gs), len(gs))
            ix = np.concatenate([gs[i] for i in pick])
            if len(set(yy[ix].tolist())) < 2:
                continue
            out.append(auc(yy[ix], sa[ix]) - auc(yy[ix], sb[ix]))
        o = np.array(out)
        return [round(float(np.quantile(o, .025)), 4), round(float(np.quantile(o, .975)), 4)]
    R["bootstrap_95CI"] = {
        "eval_T_minus_VA": boot(y_ev, dense_ev, scores_ev["VA"], ev.app_id.values),
        "eval_T_minus_VAplusSTRUCT": boot(y_ev, dense_ev, scores_ev["VA_plus_STRUCT"], ev.app_id.values),
        "test_T_minus_VA": boot(y_te, dense_te, scores_te["VA"], te.app_id.values),
        "test_T_minus_VAplusSTRUCT": boot(y_te, dense_te, scores_te["VA_plus_STRUCT"], te.app_id.values),
    }

    np.savez_compressed(OUT / "round0_gap_scores.npz",
                        y_ev=y_ev, y_te=y_te, dense_ev=dense_ev, dense_te=dense_te,
                        **{f"ev_{k}": v for k, v in scores_ev.items()},
                        **{f"te_{k}": v for k, v in scores_te.items()})
    json.dump(R, open(OUT / "round0_gap_decomposition.json", "w"), indent=2)
    print(json.dumps(R, indent=2), flush=True)
    print("ROUND0_GAP_DECOMPOSITION_DONE", flush=True)


if __name__ == "__main__":
    main()
