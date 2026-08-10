#!/usr/bin/env python3
"""STAGE 0 readout (round 0) for the CW-community confirmatory closure cell.

Merges the frozen 45-criterion A bank + 15 V features across
  (a) the 408 already-scored dense-held-out rows of the original 2,000-row
      Layer-1 population  (outputs/va_gemma_banks/creative_writing_shard*.npz)
  (b) the extension rows scored by stage0_score_ext_gemma.py
into one honest population, defines the closure splits, and reports the
refreshed round-0 same-rows T / VA_nl / Delta_beyond.

Splits (stable sha256 on prompt_id, never a seeded shuffle):
    h < .70  -> fit_mine  |  .70 <= h < .85 -> monitor  |  h >= .85 -> test
Every row is dense-held-out by construction, so MONITOR subset-of dense-held-out
is satisfied trivially (freeze requirement).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import closure_lib_cw as C

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
VA_OUT = REPO / "outputs" / "va_gemma_banks"

MON_LO, MON_HI = 0.70, 0.85


def load_original_bank():
    """A/V matrices for the original 2,000-row population, keyed by row id."""
    meta = json.loads((VA_OUT / "creative_writing_meta.json").read_text())
    Xs, Vs, ids = [], [], []
    si = 0
    while (VA_OUT / f"creative_writing_shard{si}.npz").exists():
        z = np.load(VA_OUT / f"creative_writing_shard{si}.npz", allow_pickle=True)
        Xs.append(z["X"])
        Vs.append(z["V"])
        ids += [str(s) for s in z["ids"]]
        a_names = [str(s) for s in z["a_names"]]
        v_names = [str(s) for s in z["v_names"]]
        si += 1
    X = np.vstack(Xs)
    V = np.vstack(Vs)
    return {i: (X[k], V[k]) for k, i in enumerate(ids)}, a_names, v_names, meta


def build_matrix(ext_npz_paths):
    pop = pd.read_csv(HERE / "cw_honest_population.csv")
    by_id, a_names, v_names, _ = load_original_bank()

    ext = {}
    for p in ext_npz_paths:
        z = np.load(p, allow_pickle=True)
        assert [str(s) for s in z["a_names"]] == a_names, f"criterion mismatch in {p}"
        assert [str(s) for s in z["v_names"]] == v_names, f"V mismatch in {p}"
        for k, i in enumerate([str(s) for s in z["ids"]]):
            ext[i] = (z["X"][k], z["V"][k])

    A, V, keep = [], [], []
    for i, rid in enumerate(pop.id.astype(str)):
        src = ext.get(rid) or by_id.get(rid)
        if src is None:
            continue
        A.append(src[0])
        V.append(src[1])
        keep.append(i)
    pop = pop.iloc[keep].reset_index(drop=True)
    return pop, np.array(A, float), np.array(V, float), a_names, v_names


def add_splits(pop):
    h = np.array([C.hash_unit(str(p)) for p in pop.prompt_id])
    split = np.where(h < MON_LO, "fit_mine",
                     np.where(h < MON_HI, "monitor", "test"))
    pop = pop.copy()
    pop["h"] = h
    pop["split"] = split
    return pop


def main():
    ext_paths = sorted(HERE.glob("cw_ext*_scores.npz"))
    print("[stage0] extension score files:", [p.name for p in ext_paths])
    pop, A, V, a_names, v_names = build_matrix(ext_paths)
    pop = add_splits(pop)
    y = pop.judgement.astype(int).values
    groups = pop.prompt_id.astype(str).values
    VA = np.column_stack([V, A])
    print(f"[stage0] matrix {VA.shape}  n={len(pop)}  pos={y.mean():.4f}")
    print(pop.split.value_counts().to_dict())

    dense = pd.read_csv(HERE / "cw_honest_dense_preds.csv")
    dmap = dict(zip(dense.id.astype(str), dense.dense_prob))
    pop["dense_prob"] = [dmap[str(i)] for i in pop.id]
    T = pop.dense_prob.values

    res = C.fit_block(VA, y, groups, pop.split.values)
    v_res = C.fit_block(V, y, groups, pop.split.values)

    out = {
        "n": int(len(pop)), "pos_rate": float(y.mean()),
        "n_groups": int(pop.prompt_id.nunique()),
        "split_counts": pop.split.value_counts().to_dict(),
        "split_rule": f"sha256(prompt_id): <{MON_LO} fit_mine, <{MON_HI} monitor, else test",
        "n_features": {"V": int(V.shape[1]), "A": int(A.shape[1]),
                       "VA": int(VA.shape[1])},
        "criteria": a_names,
        "VA": {k: v for k, v in res.items() if not k.startswith("_")},
        "V_only": {k: v for k, v in v_res.items() if not k.startswith("_")},
    }
    for nm, m in (("population", np.ones(len(pop), bool)),
                  ("fit_mine", pop.split.values == "fit_mine"),
                  ("monitor", pop.split.values == "monitor"),
                  ("test", pop.split.values == "test")):
        out[f"T_{nm}"] = float(roc_auc_score(y[m], T[m]))
        out[f"n_{nm}"] = int(m.sum())
    out["Delta_beyond_population"] = out["T_population"] - res["va_nl_population"]
    out["Delta_beyond_monitor"] = out["T_monitor"] - res["va_nl_monitor"]
    out["Delta_beyond_test"] = out["T_test"] - float(
        roc_auc_score(y[pop.split.values == "test"],
                      res["_pop_nl"][pop.split.values == "test"])
        if (pop.split.values == "test").sum() else np.nan)
    out["boot_Delta_population"] = C.group_bootstrap_delta(
        y, T, res["_pop_nl"], groups)
    out["boot_Delta_monitor"] = C.group_bootstrap_delta(
        y[pop.split.values == "monitor"], T[pop.split.values == "monitor"],
        res["_pop_nl"][pop.split.values == "monitor"],
        groups[pop.split.values == "monitor"])

    np.savez_compressed(HERE / "round0_state.npz", VA=VA, V=V, A=A, y=y,
                        groups=groups, split=pop.split.values.astype(object),
                        ids=pop.id.values.astype(object), T=T,
                        pop_nl=res["_pop_nl"], pop_lin=res["_pop_lin"],
                        a_names=np.array(a_names, dtype=object),
                        v_names=np.array(v_names, dtype=object),
                        bank_names=np.array(v_names + a_names, dtype=object),
                        NUIS=np.zeros((len(y), 0)),
                        nuis_names=np.array([], dtype=object),
                        nuis_upstream=np.array([], dtype=object),
                        nuis_mixed=np.array([], dtype=bool),
                        held_seed_preds=res["_held_seed_preds"],
                        held_mask=res["_held_mask"])
    pop.to_csv(HERE / "cw_population_with_splits.csv", index=False)
    (HERE / "round0_results.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "criteria"},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
