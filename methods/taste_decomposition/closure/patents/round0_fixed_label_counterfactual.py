#!/usr/bin/env python3
"""ROUND-0, part 5: the fixed-label counterfactual.

§4 of notes/2026-08-07__closure_patents.md shows the cell's label pools statutory
grounds the corpus text cannot speak to (§112 definiteness, §101 subject-matter
eligibility, double patenting). Recommendation §8.3 is to restrict positives to the
prior-art grounds (§102 anticipation / §103 obviousness). This script asks whether that
restriction would RESCUE the cell -- i.e. on the fixed label, is Delta_beyond still
dominated by claim ordinal position, or does a real entailment residual appear?

Everything is refit on the restricted TRAIN split and read on the restricted EVAL and
TEST splits, so the comparison is internally consistent. The dense model is NOT
retrained (that would need another 5 GPU-hours per seed); its predictions are simply
read on the restricted rows, which makes T here an optimistic stand-in -- a model
trained only on prior-art positives could do better or worse. Flagged as such.

CPU only. Run on sk3.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

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


def bt(r):
    p = [f"CLAIM ELEMENT:\n{r['element']}"]
    for i, x in enumerate(r.get("refs") or []):
        p.append(f"REFERENCE {i + 1} (patent {x.get('doc_id', '?')}):\n"
                 f"{' '.join(x.get('spans') or [])}")
    return "\n\n".join(p)


def auc(y, s):
    return float(roc_auc_score(np.asarray(y), np.asarray(s, float)))


def main():
    J = [json.loads(l) for l in open(JL) if l.strip()]
    th = defaultdict(list)
    for i, r in enumerate(J):
        th[hashlib.sha1(bt(r).encode()).hexdigest()].append(i)
    ptr = defaultdict(int); si = {}
    for sp in ("train", "eval", "test"):
        d = pd.read_csv(DS / "split" / f"{sp}.csv"); ix = []
        for t in d.text.astype(str).values:
            h = hashlib.sha1(t.encode()).hexdigest(); l = th[h]; k = ptr[h]
            ix.append(l[k] if k < len(l) else l[-1]); ptr[h] = k + 1
        si[sp] = np.array(ix)

    rows = []
    for r in J:
        el = r["element"] or ""
        refs = r.get("refs") or []
        sl = [len(" ".join(q.get("spans") or [])) for q in refs]
        rows.append({"y": 1 if r["label"] == "pos" else 0,
                     "rt": str(r.get("rejection_type")), "app_id": str(r["app_id"]),
                     "claim_num": int(r["claim_num"]) if str(r["claim_num"]).lstrip("-").isdigit() else -1,
                     "is_dependent": int(bool(DEP_RE.search(el))),
                     "el_chars": len(el), "el_words": len(el.split()),
                     "text_chars": len(bt(r)), "n_refs": len(refs),
                     "span_chars_total": int(sum(sl)),
                     "span_chars_mean": float(np.mean(sl)) if sl else 0.0})
    F = pd.DataFrame(rows)
    va = pd.read_csv(VA_CSV)
    assert (va["fell"].to_numpy() == F["y"].to_numpy()).all()
    F = pd.concat([F, va[V_COLS + A_COLS]], axis=1)
    dense = {"eval": pd.read_csv(DS / "rm_out_seed42/preds_eval.csv").prob.to_numpy(),
             "test": pd.read_csv(DS / "rm_out_seed42/preds_test.csv").prob.to_numpy()}

    R = {"caveat": ("the dense model was TRAINED on the pooled label; its predictions are "
                    "merely re-read on the restricted rows. T here is a stand-in, not a "
                    "retrained prior-art-only dense arm.")}
    for name, keep_types in (("FIXED_label_102_103_only", {"102", "103"}),
                             ("pooled_label_ALL_grounds", None)):
        tr = F.iloc[si["train"]].reset_index(drop=True)
        sub = {}
        for sp in ("eval", "test"):
            fr = F.iloc[si[sp]].reset_index(drop=True).copy()
            fr["prob"] = dense[sp]
            if keep_types is not None:
                m = (fr.y == 0) | fr.rt.isin(keep_types)
                fr = fr[m].reset_index(drop=True)
            sub[sp] = fr
        trf = tr if keep_types is None else tr[(tr.y == 0) | tr.rt.isin(keep_types)].reset_index(drop=True)
        y_tr = trf.y.to_numpy()
        out = {}
        for blk, cols in (("VA", V_COLS + A_COLS), ("STRUCT", STRUCT),
                          ("claim_num_only", ["claim_num"]),
                          ("VA_plus_STRUCT", V_COLS + A_COLS + STRUCT)):
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                               max_leaf_nodes=31, random_state=0)
            m.fit(trf[cols].to_numpy(float), y_tr)
            for sp in ("eval", "test"):
                out.setdefault(sp, {})[blk] = round(
                    auc(sub[sp].y, m.predict_proba(sub[sp][cols].to_numpy(float))[:, 1]), 4)
        for sp in ("eval", "test"):
            out[sp]["T_dense"] = round(auc(sub[sp].y, sub[sp].prob), 4)
            out[sp]["n"] = int(len(sub[sp]))
            out[sp]["pos_rate"] = round(float(sub[sp].y.mean()), 4)
            g = out[sp]["T_dense"] - out[sp]["VA"]
            out[sp]["gap_T_minus_VA"] = round(g, 4)
            out[sp]["frac_of_gap_closed_by_STRUCT"] = round(
                (out[sp]["VA_plus_STRUCT"] - out[sp]["VA"]) / g, 3) if g else None
            out[sp]["residual_beyond_VA_plus_STRUCT"] = round(
                out[sp]["T_dense"] - out[sp]["VA_plus_STRUCT"], 4)
        out["train_n"] = int(len(trf))
        R[name] = out
    json.dump(R, open(OUT / "round0_fixed_label_counterfactual.json", "w"), indent=2)
    print(json.dumps(R, indent=2), flush=True)
    print("ROUND0_FIXED_LABEL_DONE", flush=True)


if __name__ == "__main__":
    main()
