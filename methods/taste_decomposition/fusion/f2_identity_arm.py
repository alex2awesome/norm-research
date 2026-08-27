#!/usr/bin/env python3
"""F2 IDENTITY ARM for peer_revealed (user-ordered leak audit, 2026-08-12).

Question: does the cell's deconfounded residual (+.0927/+.0969, F2 primary) survive
conditioning on AUTHOR/INSTITUTION IDENTITY — channels that were never in the 57
judged nuisance columns (nuisance_struct = 0 for this cell)?

Mirror of f2_deconf.run_cell with ONE change: the nuisance block is augmented with
four identity covariates, joined by ntitle from the OpenAlex authorship pull
(methods/taste_decomposition/peer_identity_audit/):
    auth_enc   mean y of the same authors' TRAIN-split papers (own paper excluded)
    inst_enc   same over institutions
    auth_fame  mean log1p(cited_by_count) of the authors' TRAIN-split papers
    auth_cov   indicator: any author coverage
Encoders are built from TRAIN-split rows only; E-rows are dense-held-out, so the
covariates are constructed leak-free (no E-row's own label enters its covariate).
Missing covariates imputed at the train pos-rate (auth/inst enc), train fame mean.

This is a NEW registered arm (f2_identity), never an overwrite of the frozen F2 row.
CPU. Usage:  python3 f2_identity_arm.py [--n-boot 2000]
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TD = HERE.parent
RESULTS = TD / "results"
AUD = TD / "peer_identity_audit"
REPO = TD.parents[1]

def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m

F2 = _mod(HERE / "f2_deconf.py", "f2_deconf_mod")

CELL = "peer_revealed"


def identity_columns(ids_E):
    rows = [json.loads(l) for l in open(REPO / "datasets/peer-review/vat_3y/revealed.jsonl")]
    meta = {json.loads(l)["work_id"]: json.loads(l)
            for l in open(AUD / "openalex_authorships.jsonl")}
    by_nt = {}
    for r in rows:
        try:
            yv = float(r.get("judgement"))
        except (TypeError, ValueError):
            continue
        if yv not in (0.0, 1.0):
            continue
        wid = r["id"].rsplit("/", 1)[-1]
        m = meta.get(wid, {})
        by_nt[r["ntitle"]] = {
            "y": int(yv), "split": r.get("split"),
            "authors": [a["author_id"] for a in m.get("authors", []) if a.get("author_id")],
            "insts": sorted({i["id"] for a in m.get("authors", [])
                             for i in a.get("institutions", []) if i.get("id")}),
            "cites": m.get("cited_by_count"),
        }
    auth_y, inst_y, auth_c = defaultdict(list), defaultdict(list), defaultdict(list)
    train_ys = []
    for nt, r in by_nt.items():
        if r["split"] != "train":
            continue
        train_ys.append(r["y"])
        for e in r["authors"]:
            auth_y[e].append(r["y"])
            if r["cites"] is not None:
                auth_c[e].append(np.log1p(r["cites"]))
        for e in r["insts"]:
            inst_y[e].append(r["y"])
    base_y = float(np.mean(train_ys))
    base_c = float(np.mean([v for vs in auth_c.values() for v in vs]))

    cols, joined = [], 0
    for nt in ids_E:
        r = by_nt.get(str(nt))
        if r is None:
            cols.append([base_y, base_y, base_c, 0.0])
            continue
        joined += 1
        av = [y for e in r["authors"] for y in auth_y.get(e, [])]
        iv = [y for e in r["insts"] for y in inst_y.get(e, [])]
        cv = [c for e in r["authors"] for c in auth_c.get(e, [])]
        cols.append([np.mean(av) if av else base_y,
                     np.mean(iv) if iv else base_y,
                     np.mean(cv) if cv else base_c,
                     1.0 if av else 0.0])
    X = np.array(cols, dtype=float)
    names = ["ID:author_trainenc_meany", "ID:institution_trainenc_meany",
             "ID:author_fame_logcites", "ID:author_coverage"]
    return X, names, joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()

    meta, ids_E, y, groups, dense, t0col = F2.load_E(CELL)
    ad = F2.F2C.ADAPTERS[CELL]()
    bank, nuis, join = F2.align(CELL, ad, ids_E, y, groups)
    ID, id_names, joined = identity_columns(ids_E)
    print(f"[{CELL}] n_E={len(y)} bank={bank.shape} nuis={nuis.shape} "
          f"identity joined {joined}/{len(y)}", flush=True)

    fam = meta["family"]
    nuis_id = np.column_stack([nuis, ID])
    bn = np.column_stack([bank, nuis])
    bni = np.column_stack([bank, nuis_id])

    r_id = F2.fit_arm(fam, ID, dense, y, groups)          # identity block alone
    r_bn = F2.fit_arm(fam, bn, dense, y, groups)          # (c)/(d) reference
    r_bni = F2.fit_arm(fam, bni, dense, y, groups)        # (c')/(d') identity-augmented

    oof_c, oof_d = r_bn["_oof_VA_nl0"], r_bn["_oof_VAT_nl0"]
    oof_ci, oof_di = r_bni["_oof_VA_nl0"], r_bni["_oof_VAT_nl0"]
    prim_ref = F2.gboot(y, oof_d, oof_c, groups, n_boot=a.n_boot)
    prim_id = F2.gboot(y, oof_di, oof_ci, groups, n_boot=a.n_boot)
    id_gain = F2.gboot(y, oof_ci, oof_c, groups, n_boot=a.n_boot)

    alone = {n: F2.alone_auc(y, ID[:, j]) for j, n in enumerate(id_names)}
    out = {
        "arm": "f2_identity", "cell": CELL, "env": F2.env_block(),
        "spec": "f2_deconf mirror; nuisance block += 4 identity covariates "
                "(train-split encoders, leak-free construction); NEW arm, frozen F2 row untouched",
        "n_E": int(len(y)), "identity_joined": joined,
        "identity_alone_nl": r_id["VA_nl_mean"],
        "identity_alone_auc_per_col": alone,
        "c_bank_nuis": r_bn["VA_nl_mean"], "d_plus_T": r_bn["VAT_nl_mean"],
        "c_prime_bank_nuis_identity": r_bni["VA_nl_mean"],
        "d_prime_plus_T": r_bni["VAT_nl_mean"],
        "primary_reference_d_minus_c": prim_ref,
        "primary_identity_dprime_minus_cprime": prim_id,
        "identity_increment_cprime_minus_c": id_gain,
        "runtime_s": round(time.time() - t0, 1),
    }
    (RESULTS / "f2_identity_peer_revealed.json").write_text(json.dumps(out, indent=1))
    print(f"[{CELL}] ID-alone {r_id['VA_nl_mean']:.4f} | (c) {r_bn['VA_nl_mean']:.4f} "
          f"(d) {r_bn['VAT_nl_mean']:.4f} | (c') {r_bni['VA_nl_mean']:.4f} "
          f"(d') {r_bni['VAT_nl_mean']:.4f}", flush=True)
    print(f"[{CELL}] PRIMARY ref (d)-(c) = {prim_ref['estimate']:+.4f} | "
          f"identity-conditioned (d')-(c') = {prim_id['estimate']:+.4f} "
          f"[{prim_id.get('lo', float('nan')):+.4f},{prim_id.get('hi', float('nan')):+.4f}] "
          f"P={prim_id.get('p_gt_0', float('nan')):.3f}", flush=True)
    print("F2_IDENTITY_DONE", flush=True)


if __name__ == "__main__":
    main()
