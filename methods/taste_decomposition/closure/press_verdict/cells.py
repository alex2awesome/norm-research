#!/usr/bin/env python3
"""Cell adapter for the Layer-3 closure campaign on the PRESS/journalism VERDICT cell
(press-release editorial pickup, k>=3 distinct tracked outlets).

Same contract as peer_revealed/cells.py: return the cell's Layer-1 A/V population in a
REPRODUCIBLE row order together with ids / groups / y / raw A,V blocks / texts / dense.

POPULATION (frozen, never rebuilt here).  `methods/taste_decomposition/results/
press_verdict_pr_A_k3_scores_CACHE.npz` IS the population definition: 2,956 rows
(1,478 pos / 1,478 topic-matched neg), 556 companies, produced by sk3's
`datasets/press-releases/run_A_layer_k3.py`.  Its `ids` order is the canonical row
order and is used verbatim.

A BLOCK.  40 Gemma-judged rubrics, applicability-gated: raw value = `levels` where
`applicable`, NaN otherwise.  Layer 1 filled the inapplicable cells with the constant
0.5 that the A-layer's own published gate used; the closure protocol's
`closure_core.clean_fit` median-imputes inside FIT+MINE instead.  Both are built here
(`A` = NaN-coded so clean_fit owns the imputation, `A_const05` = the Layer-1 variant)
and round 0 reports the readout under both, so the deviation is measured, not assumed.

V BLOCK.  The 88-column reconstructed hand-feature bank
(`methods/taste_decomposition/press_verdict_v_features_recon.py`, itself recovered from
`__pycache__/codex_pr_vrescue.cpython-312.pyc` by bytecode disassembly -- see that
module's docstring).  Cached to `press_v88.npz` on first build (it is deterministic).

DENSE (T).  `datasets/press-releases/dense_standard_k3/` on sk3: Llama-3.1-8B LoRA
dense-standard, COMPANY-GROUPED stable-hash 80/10/10 split, 3 seeds (42, 1, 2),
select-on-eval (no deviation).  Per-row probabilities live in `rm_out_seed*/preds_{eval,
test}.csv`, which carry no row key of their own but are row-aligned with
`split/{eval,test}.csv` (verified: group and judgement columns match positionally for
all three seeds, all 605 rows).  Local mirror: `press_dense_preds_3seed.csv`.

T CONVENTION.  VA_nl in this programme is the MEAN OVER SEEDS OF THE AUC (not the AUC
of the averaged prediction), so T is defined the same way: T = mean over dense seeds
{42,1,2} of AUC(y, p_seed).  `d["dense"]` is the per-row seed-mean probability, used
wherever a single dense SCORE VECTOR is required (swap algebra, stacks, strata); every
AUC-valued readout of T is computed per seed and averaged, and the ensemble figure
(AUC of the seed-mean probability, which is higher because averaging denoises) is
reported separately and never quoted as T.

SKLEARN.  `press_verdict_layer1.json` was produced under scikit-learn 1.9.0 and that
cell's module docstring documents that GroupKFold fold ASSIGNMENTS move across sklearn
releases.  Run this campaign under 1.9.0 (scratch venv) so every round shares one
splitter; `sklearn_guard()` asserts it.

CPU only.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
TD = CLOSURE.parent
REPO = TD.parents[1]
sys.path.insert(0, str(TD))
sys.path.insert(0, str(CLOSURE))

RESULTS = TD / "results"
A_CACHE = RESULTS / "press_verdict_pr_A_k3_scores_CACHE.npz"
PARQUET = REPO / "datasets" / "press-releases" / "press_release_deconfounded.parquet"
MODELING = REPO / "datasets" / "press-releases" / "press_release_modeling_dataset.csv.gz"
V_CACHE = HERE / "press_v88.npz"
DENSE_CSV = HERE / "press_dense_preds_3seed.csv"
DATE_CACHE = HERE / "press_release_dates.csv"

DENSE_SEEDS = (42, 1, 2)

CELL_META = {
    "press_verdict": dict(
        group_column="company", item="press release",
        corpus="corporate and institutional press releases issued by named organisations",
        construct="how newsworthy the release is -- whether working journalists at "
                  "independent outlets would judge it worth writing up as news",
        text_trunc=3500,
        layer1="press_verdict_layer1.json",
    ),
}


def sklearn_guard(required=(1, 9)):
    import sklearn
    got = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    assert got >= required, (
        f"scikit-learn {sklearn.__version__} < {required}: this cell's Layer-1 numbers "
        "were produced under 1.9.0 and GroupKFold fold assignments are version-sensitive "
        "(see press_verdict_layer1.py LANDMINE). Use the 1.9.0 scratch venv.")
    return sklearn.__version__


# ------------------------------------------------------------------- V bank ----
def _build_v(ids, texts):
    if V_CACHE.exists():
        z = np.load(V_CACHE, allow_pickle=True)
        if list(z["ids"].astype(str)) == list(ids):
            return z["V"], [str(s) for s in z["names"]]
    sys.path.insert(0, str(TD))
    from press_verdict_v_features_recon import extract_features  # noqa: E402
    V, names = extract_features(texts)
    np.savez_compressed(V_CACHE, V=V, names=np.array(names, dtype=object),
                        ids=np.array(ids, dtype=object))
    return V, names


# ------------------------------------------------------------ release dates ----
def _dates(ids):
    """press_release_date from the modeling dataset, keyed by press_release_id.
    85.5% coverage (2,528/2,956); used ONLY as an observed covariate for the
    position line, never as a feature in any bank."""
    if DATE_CACHE.exists():
        d = pd.read_csv(DATE_CACHE, dtype={"id": str})
        return dict(zip(d["id"], d["date"]))
    csv.field_size_limit(10 ** 9)
    want = set(ids)
    out = {}
    with gzip.open(MODELING, "rt") as fh:
        for row in csv.DictReader(fh):
            pid = row["press_release_id"]
            if pid in want and pid not in out and row["press_release_date"]:
                out[pid] = row["press_release_date"]
    pd.DataFrame({"id": list(out), "date": list(out.values())}).to_csv(DATE_CACHE, index=False)
    return out


# ------------------------------------------------------------------- dense ----
def _dense(ids):
    """Per-seed dense probabilities + dense split membership, keyed by row id."""
    assert DENSE_CSV.exists(), f"missing {DENSE_CSV}; run fetch_dense.py first"
    d = pd.read_csv(DENSE_CSV, dtype={"id": str})
    by = {r.id: r for r in d.itertuples()}
    P = np.full((len(ids), len(DENSE_SEEDS)), np.nan)
    dsplit = []
    for k, i in enumerate(ids):
        r = by.get(i)
        if r is None:
            dsplit.append("train")
            continue
        dsplit.append(r.dense_split)
        for j, s in enumerate(DENSE_SEEDS):
            P[k, j] = getattr(r, f"p{s}")
    return P, np.array(dsplit)


# -------------------------------------------------------------------- load ----
def load(cell="press_verdict"):
    assert cell == "press_verdict", cell
    z = np.load(A_CACHE, allow_pickle=True)
    ids = [str(s) for s in z["ids"]]
    y = np.asarray(z["y"]).astype(int)
    comp = np.array([str(s) for s in z["comp"]])
    topic = np.asarray(z["topic"])
    levels, applicable = np.asarray(z["levels"], float), np.asarray(z["applicable"])
    a_names = [str(s) for s in z["names"]]

    A_nan = np.where(applicable, levels, np.nan)
    A_c05 = np.where(applicable, levels, 0.5)

    from press_verdict_v_features_recon import clean_text  # noqa: E402
    df = pd.read_parquet(PARQUET, columns=["id", "text"])
    df["id"] = df["id"].astype(str)
    id2t = dict(zip(df["id"], df["text"].fillna("")))
    missing = [i for i in ids if i not in id2t]
    assert not missing, f"{len(missing)} ids missing from the parquet"
    texts = [clean_text(id2t[i]) for i in ids]

    V, v_names = _build_v(ids, texts)
    P, dsplit = _dense(ids)
    date_by = _dates(ids)
    dates = np.array([date_by.get(i, "") for i in ids], dtype=object)

    # PRIMARY A BLOCK = the Layer-1 constant-0.5 fill.  DECISION recorded at round 0,
    # before any round ran (press_verdict_r0_context.json / the campaign note): the cell's
    # OWN published A-gate and its Layer-1 production pipeline both fill inapplicable
    # rubric cells with the constant 0.5, so that matrix IS this cell's scorecard.  The
    # closure-standard median-impute variant is carried as `A_median_impute` and is a
    # SENSITIVITY, not the primary: median-imputing erases the applicability pattern and
    # costs the bank .029 of held-out AUC, which would inflate Delta_beyond by the same
    # amount.  Using the weaker bank would have been the less conservative choice.
    d = dict(tag="press_verdict", ids=ids, y=y, groups=comp, topic=topic,
             A=A_c05, A_median_impute=A_nan, A_const05=A_c05,
             V=V, a_names=a_names, v_names=list(v_names),
             texts=texts, years=np.array([float(s[:4]) if s else np.nan for s in dates]),
             dates=dates,
             dense=np.nanmean(P, axis=1), dense_seeds=P, dense_seed_ids=list(DENSE_SEEDS),
             dense_split=dsplit)
    d["meta"] = CELL_META[cell]
    led = json.loads((RESULTS / CELL_META[cell]["layer1"]).read_text())
    d["layer1"] = led
    assert len(y) == led["n"], f"n={len(y)} != layer1 n={led['n']}"
    assert len(set(comp)) == led["n_groups"]
    return d


LOADERS = {"press_verdict": load}
CELLS = ["press_verdict"]


def T_by_seed(d, mask):
    """T = mean over dense seeds of AUC on `mask` (the programme's VA_nl convention,
    applied to the dense side).  Also returns the per-seed values and the ensemble."""
    from sklearn.metrics import roc_auc_score
    y = d["y"][mask]
    per = [float(roc_auc_score(y, d["dense_seeds"][mask, j]))
           for j in range(d["dense_seeds"].shape[1])]
    return {"T": float(np.mean(per)),
            "per_seed": {str(s): p for s, p in zip(d["dense_seed_ids"], per)},
            "spread": float(max(per) - min(per)),
            "T_seed_ensemble_NOT_QUOTED": float(roc_auc_score(y, d["dense"][mask])),
            "n": int(mask.sum())}


if __name__ == "__main__":
    print("sklearn", sklearn_guard())
    d = load()
    held = np.isin(d["dense_split"], ["eval", "test"])
    print(f"press_verdict n={len(d['y'])} companies={len(set(d['groups']))} "
          f"pos={d['y'].mean():.3f} A={d['A'].shape[1]} V={d['V'].shape[1]} "
          f"held-out={held.sum()} ({len(set(d['groups'][held]))} companies) "
          f"dates={int((d['dates'] != '').sum())}")
    print("T HONEST:", json.dumps(T_by_seed(d, held), indent=1))
