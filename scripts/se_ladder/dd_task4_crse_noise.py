"""Deep dive task 4: CR.SE label-noise / answerer-consistency evidence.

(a) Parse OwnerUserId for answers from the CR.SE raw dump (Posts.xml).
(b) On the CR.SE v2 balanced file AND the SO[python] balanced file
    (comparison), compute:
      - leave-one-out answerer-history AUC: predict each row's label from
        the mean label of the SAME answerer's OTHER rows (answerers with
        >=2, >=3, >=5 labeled rows) — a replicable-expert-style signal
        that is independent of this answer's text;
      - identity-leak AUC train->test (as in task 2) for CR.SE.
(c) Analytic CIs (Hanley-McNeil) for the ladder headline AUCs, to size
    the small-n argument.

CPU only. Run on sk3.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT_DIR = REPO / "outputs/v2_analysis/se_ladder"
DS = REPO / "datasets"

OWNER_RE = re.compile(rb'OwnerUserId="(\d+)"')
ID_RE = re.compile(rb' Id="(\d+)"')
TYPE_RE = re.compile(rb'PostTypeId="(\d+)"')


def crse_owner_map():
    """Stream Posts.xml; answers are PostTypeId=2."""
    out = {}
    with open(DS / "codereview_se/raw_dump/Posts.xml", "rb") as f:
        for line in f:
            if b'PostTypeId="2"' not in line:
                continue
            mi = ID_RE.search(line)
            mo = OWNER_RE.search(line)
            if mi and mo:
                out[mi.group(1).decode()] = mo.group(1).decode()
    return out


def loo_auc(df, min_n_values=(2, 3, 5)):
    """Leave-one-out answerer-history AUC."""
    res = {}
    g = df.groupby("owner")["judgement"]
    s, c = g.transform("sum"), g.transform("count")
    loo = (s - df["judgement"]) / (c - 1).replace(0, np.nan)
    for mn in min_n_values:
        m = (c >= mn) & loo.notna()
        sub_y, sub_p = df.loc[m, "judgement"], loo[m]
        if sub_y.nunique() == 2:
            res[f"min_n_{mn}"] = {
                "n_rows": int(m.sum()),
                "frac_rows": float(m.mean()),
                "auc": float(roc_auc_score(sub_y, sub_p)),
            }
    return res


def hanley_ci(auc, n1, n2):
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    se = math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc)
                    + (n2 - 1) * (q2 - auc * auc)) / (n1 * n2))
    return [auc - 1.96 * se, auc + 1.96 * se]


def main():
    out = {}

    print("parsing CR.SE Posts.xml for OwnerUserId...", flush=True)
    om = crse_owner_map()
    print(f"  {len(om)} answer->owner entries", flush=True)

    crse = pd.read_csv(DS / "code-review/crse_balanced_v2/crse_v2_propensity_balanced.csv.gz",
                       dtype={"question_id": str, "answer_id": str})
    crse["owner"] = crse.answer_id.map(lambda a: om.get(str(int(float(a)))))
    cov = crse.owner.notna().mean()
    crse_o = crse[crse.owner.notna()]
    res = {"n": int(len(crse)), "owner_coverage": float(cov),
           "loo_answerer_auc": loo_auc(crse_o)}
    tr = crse_o[crse_o.split == "train"]
    te = crse_o[crse_o.split == "test"]
    hist = tr.groupby("owner").judgement.mean()
    te2 = te[te.owner.isin(hist.index)]
    res["test_owner_seen_in_train_frac"] = float(len(te2) / max(1, len(te)))
    if len(te2) > 50 and te2.judgement.nunique() == 2:
        res["identity_leak_auc_on_covered_test"] = float(
            roc_auc_score(te2.judgement, te2.owner.map(hist)))
        res["n_covered_test"] = int(len(te2))
    out["crse"] = res
    print(json.dumps(res, indent=2), flush=True)

    # SO[python] comparison (owner map from parquet)
    import pyarrow.parquet as pq
    t = pq.read_table(DS / "stackoverflow_python/so_python_answers.parquet",
                      columns=["Id", "OwnerUserId"]).to_pandas()
    om2 = dict(zip(t.Id.astype(np.int64).astype(str),
                   t.OwnerUserId.astype(str)))
    py = pd.read_csv(DS / "stackoverflow_python/balanced/so_python_v2_propensity_balanced.csv.gz",
                     dtype={"question_id": str, "answer_id": str})
    py["owner"] = py.answer_id.map(lambda a: om2.get(str(int(float(a)))))
    py_o = py[py.owner.notna() & (py.owner != "nan") & (py.owner != "None")]
    out["so_python"] = {"n": int(len(py)),
                        "owner_coverage": float(py.owner.notna().mean()),
                        "loo_answerer_auc": loo_auc(py_o)}
    print(json.dumps(out["so_python"], indent=2), flush=True)

    # analytic CIs for ladder headline AUCs
    cis = {}
    for name, auc, n_test in [
            ("crse_bank_ENS", 0.584, 1450), ("crse_mbert", 0.626, 1450),
            ("py_bank_ENS", 0.667, 6177), ("py_mbert", 0.695, 6177),
            ("js_mbert", 0.660, 6112), ("sql_mbert", 0.677, 3219)]:
        n1 = n2 = n_test // 2
        cis[name] = {"auc": auc, "ci95": hanley_ci(auc, n1, n2)}
    out["auc_ci95"] = cis
    print(json.dumps(cis, indent=2), flush=True)

    (OUT_DIR / "dd_task4_crse_noise.json").write_text(json.dumps(out, indent=2))
    print("wrote dd_task4_crse_noise.json", flush=True)


if __name__ == "__main__":
    main()
