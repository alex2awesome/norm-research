"""Deep dive task 3: extract the articulability residual for hand-reading.

Rows of the so_python TEST split where ModernBERT FT is confident-correct
(|p-0.5| in the top 40% among its correct predictions) but bank ENS is on
the wrong side of 0.5. Stratified sample ~150 (lib_data-heavy, where the
residual concentrates). Dumps JSONL for laptop reading.

Run on sk3 after dd_modernbert_preds.py + dd_bank_preds.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT_DIR = REPO / "outputs/v2_analysis/se_ladder"
SLICE = "so_python"


def main():
    inp = pd.read_parquet(OUT_DIR / f"{SLICE}_input.parquet")
    mb = pd.read_parquet(OUT_DIR / f"{SLICE}_modernbert_preds.parquet")
    bk = pd.read_parquet(OUT_DIR / f"{SLICE}_bank_preds.parquet")
    te = inp[inp.split == "test"].merge(mb, on="row_id").merge(
        bk, on="row_id")
    print(f"test rows joined: {len(te)}")

    te["mb_correct"] = ((te.p > 0.5) == (te.label == 1))
    te["bank_correct"] = ((te.p_en > 0.5) == (te.label == 1))
    te["mb_conf"] = (te.p - 0.5).abs()
    thr = te.loc[te.mb_correct, "mb_conf"].quantile(0.6)
    resid = te[te.mb_correct & (te.mb_conf >= thr) & (~te.bank_correct)]
    print(f"residual pool: {len(resid)} "
          f"(mb confident-correct & bank wrong); conf thr={thr:.3f}")
    print(resid.groupby("stratum").size())

    rng = np.random.default_rng(7)
    target = {"lib_data": 75, "stdlib": 50, "framework": 25}
    rows = []
    for s, n in target.items():
        sub = resid[resid.stratum == s]
        take = sub.iloc[rng.choice(len(sub), min(n, len(sub)),
                                   replace=False)]
        rows.append(take)
    samp = pd.concat(rows)
    out = OUT_DIR / "dd_anatomy_sample.jsonl"
    with open(out, "w") as f:
        for _, r in samp.iterrows():
            f.write(json.dumps({
                "row_id": int(r.row_id), "stratum": r.stratum,
                "label": int(r.label), "p_mbert": float(r.p),
                "p_bank": float(r.p_en),
                "body": str(r.body)[:3500]}) + "\n")
    print(f"wrote {out} ({len(samp)} rows)")

    # also dump overall counts for the report
    stats = {
        "n_test": int(len(te)),
        "mb_correct_frac": float(te.mb_correct.mean()),
        "bank_correct_frac": float(te.bank_correct.mean()),
        "mb_conf_thr": float(thr),
        "residual_pool": int(len(resid)),
        "residual_by_stratum": resid.groupby("stratum").size().to_dict(),
        "residual_by_label": resid.groupby("label").size().to_dict(),
    }
    (OUT_DIR / "dd_anatomy_stats.json").write_text(
        json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
