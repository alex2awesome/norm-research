#!/usr/bin/env python3
"""ROUND-0, part 4: FREEZE ADDENDUM 3 decomposition of the one MIXED channel found.

The dominating channel on this cell -- claim ordinal position -- is MIXED in exactly
the sense ADDENDUM 3 defines: its conjectured upstream parent (attorney claim-drafting
convention, which puts the broadest claims first) plausibly causes REAL merit-relevant
variation too, because breadth is precisely what makes a claim vulnerable to prior art.
So the parent channel is decomposed into components that are scored and routed
separately:

  * SURFACE component  -- the ordinal integer itself / the parent-claim number printed
    in the text; a drafting-order habit that carries no information about the claim.
  * CANDIDATE-REAL component -- claim BREADTH, proxied here by independence
    (an independent claim is broader than any claim depending from it), element brevity
    (a short limitation is a broad limitation) and limitation-marker count.

The decomposition asks: does ordinal position predict ABOVE breadth, and does breadth
predict ABOVE ordinal position? Whichever survives the other is the live channel.

NOTE ON SCOPE. The freeze's decomposition pass authors >=2 refined LLM-judged criteria
and routes each through the blind audit. This campaign stopped at round 0, so no judged
criteria were written; this is the DETERMINISTIC-PROXY version of the same pass, run to
show which side of the split the signal sits on and to specify what a judged criterion
would have to isolate. Flagged as a proxy decomposition, not an Addendum-3 pass.

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
OUT = Path(__file__).resolve().parent
csv.field_size_limit(sys.maxsize)

DEP_RE = re.compile(r"\bof claim\s+\d+|\baccording to claim\s+\d+|\bas (?:recited|claimed) in claim", re.I)
PAR_RE = re.compile(r"claims?\s+(\d+)", re.I)
LIMIT_RE = re.compile(r"\bwherein\b|\bcomprising\b|\bconfigured to\b|\bfurther\b|;")

SURFACE = ["claim_num", "parent_claim_num"]
BREADTH = ["is_independent", "el_words", "n_limit_markers", "n_commas", "n_numerals"]


def bt(r):
    p = [f"CLAIM ELEMENT:\n{r['element']}"]
    for i, x in enumerate(r.get("refs") or []):
        p.append(f"REFERENCE {i + 1} (patent {x.get('doc_id', '?')}):\n"
                 f"{' '.join(x.get('spans') or [])}")
    return "\n\n".join(p)


def auc(y, s):
    return float(roc_auc_score(np.asarray(y), np.asarray(s, float)))


def fit(tr, ev, cols, y_tr, seed=0):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                       max_leaf_nodes=31, random_state=seed)
    m.fit(tr[cols].to_numpy(float), y_tr)
    return m.predict_proba(ev[cols].to_numpy(float))[:, 1]


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
        m = PAR_RE.search(el[:200])
        rows.append({"y": 1 if r["label"] == "pos" else 0, "app_id": str(r["app_id"]),
                     "claim_num": int(r["claim_num"]) if str(r["claim_num"]).lstrip("-").isdigit() else -1,
                     "parent_claim_num": int(m.group(1)) if m else 0,
                     "is_independent": int(not bool(DEP_RE.search(el))),
                     "el_words": len(el.split()),
                     "n_limit_markers": len(LIMIT_RE.findall(el)),
                     "n_commas": el.count(","), "n_numerals": len(re.findall(r"\d", el))})
    F = pd.DataFrame(rows)
    tr, ev, te = (F.iloc[si[s]].reset_index(drop=True) for s in ("train", "eval", "test"))
    y_tr, y_ev, y_te = tr.y.to_numpy(), ev.y.to_numpy(), te.y.to_numpy()
    dense = {"eval": pd.read_csv(DS / "rm_out_seed42/preds_eval.csv").prob.to_numpy(),
             "test": pd.read_csv(DS / "rm_out_seed42/preds_test.csv").prob.to_numpy()}

    R = {"channel": "claim ordinal position",
         "conjectured_upstream_parent": ("attorney claim-drafting convention -- the broadest "
                                         "claims are drafted first, and examiners reject the "
                                         "broadest claims"),
         "mixed": True,
         "components": {
             "SURFACE (drafting-order habit)": SURFACE,
             "CANDIDATE-REAL (claim breadth)": BREADTH},
         "scope_flag": ("deterministic-proxy decomposition; the freeze's Addendum-3 pass "
                        "authors LLM-judged criteria and blind-routes them, which this "
                        "campaign did not reach")}

    lad = {}
    for nm, cols in (("SURFACE_only", SURFACE), ("BREADTH_only", BREADTH),
                     ("SURFACE+BREADTH", SURFACE + BREADTH)):
        pe, pt = fit(tr, ev, cols, y_tr), fit(tr, te, cols, y_tr)
        lad[nm] = {"eval": round(auc(y_ev, pe), 4), "test": round(auc(y_te, pt), 4)}
    lad["marginal_of_SURFACE_over_BREADTH"] = {
        k: round(lad["SURFACE+BREADTH"][k] - lad["BREADTH_only"][k], 4) for k in ("eval", "test")}
    lad["marginal_of_BREADTH_over_SURFACE"] = {
        k: round(lad["SURFACE+BREADTH"][k] - lad["SURFACE_only"][k], 4) for k in ("eval", "test")}
    R["ladder"] = lad

    # within-breadth-stratum survival of the ordinal channel, and vice versa
    st = {}
    for sp, frame, yy in (("eval", ev, y_ev), ("test", te, y_te)):
        d = {}
        # ordinal position INSIDE independent claims only (breadth partly held fixed)
        for nm, mask in (("independent_claims_only", frame.is_independent == 1),
                         ("dependent_claims_only", frame.is_independent == 0)):
            m = mask.to_numpy()
            d[nm] = {"n": int(m.sum()), "pos_rate": round(float(yy[m].mean()), 4),
                     "claim_num_alone_auc": round(auc(yy[m], -frame.claim_num.to_numpy()[m]), 4),
                     "el_words_alone_auc": round(auc(yy[m], -frame.el_words.to_numpy()[m]), 4),
                     "dense_auc": round(auc(yy[m], dense[sp][m]), 4)}
        # element-length deciles: does ordinal position survive holding brevity fixed?
        w = frame.el_words.to_numpy(float)
        q = np.quantile(w, np.linspace(0, 1, 11)); q[0] -= 1e-9; q[-1] += 1e-9
        b = np.clip(np.digitize(w, q[1:-1]), 0, 9)
        num = den = 0.0
        for k in range(10):
            m = b == k
            if m.sum() < 30 or len(set(yy[m].tolist())) < 2:
                continue
            num += auc(yy[m], -frame.claim_num.to_numpy()[m]) * m.sum(); den += m.sum()
        d["claim_num_alone_auc_stratified_by_element_length_decile"] = round(num / den, 4)
        num = den = 0.0
        cn = frame.claim_num.to_numpy(float)
        qc = np.quantile(cn, np.linspace(0, 1, 11)); qc[0] -= 1e-9; qc[-1] += 1e-9
        bc = np.clip(np.digitize(cn, qc[1:-1]), 0, 9)
        for k in range(10):
            m = bc == k
            if m.sum() < 30 or len(set(yy[m].tolist())) < 2:
                continue
            num += auc(yy[m], -w[m]) * m.sum(); den += m.sum()
        d["el_words_alone_auc_stratified_by_claim_num_decile"] = round(num / den, 4)
        st[sp] = d
    R["cross_stratification"] = st
    json.dump(R, open(OUT / "round0_mixed_decomposition.json", "w"), indent=2)
    print(json.dumps(R, indent=2), flush=True)
    print("ROUND0_MIXED_DECOMPOSITION_DONE", flush=True)


if __name__ == "__main__":
    main()
