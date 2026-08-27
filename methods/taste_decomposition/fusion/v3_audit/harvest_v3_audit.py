#!/usr/bin/env python3
"""Harvest the V3 criteria-in-prompt audit arms (caption sweep + CW mirror).

Caption arms (cap_crowd_k20/k40/k10defs/augmd, cap_finalist_*): AUC on the
SAME evaluation-valid rows E as the fusion note (eval+test pooled; row sets
identical across arms by construction), plus per-split legs. Paired row-level
bootstraps vs the V3 baseline (aug_a) and vs the bank (VA_nl fullfit@E OOF
probs from the Direction-1b inputs).

CW mirror arms (cw_raw, cw_augk20): MONITOR (eval) and TEST AUCs; paired
bootstraps aug vs raw and aug vs the ORIGINAL 77K-row dense model's per-row
preds (dense_prob in cw_population_with_splits.csv — honest on every row).

Preds must be rsync'd back into fusion/dense_data/<arm>/rm_out_seed42/ first.
Usage: python3 harvest_v3_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
FUS = HERE.parent
DATA = FUS / "dense_data"
CW = FUS.parent / "closure" / "cw_community"
RNG = np.random.default_rng(0)
NBOOT = 2000

CAP_ARMS = {
    "cap_crowd": ["aug_a", "k20", "k40", "k10defs", "augmd", "moredata"],
    "cap_finalist": ["aug_a", "k20", "k40", "k10defs", "moredata"],
}


def paired_boot(y, pa, pb):
    """AUC(pa) - AUC(pb), row-level paired bootstrap."""
    y = np.asarray(y)
    pa, pb = np.asarray(pa), np.asarray(pb)
    base = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    n = len(y)
    ds = []
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        ds.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    ds = np.array(ds)
    return {"delta": float(base),
            "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
            "p_gt0": float((ds > 0).mean())}


def load_E(cell, arm):
    """eval+test preds for an arm, row-aligned frames with judgement/prob/split."""
    d = DATA / f"{cell}_{arm}"
    frames = []
    for sp in ("eval", "test"):
        pr = pd.read_csv(d / "rm_out_seed42" / f"preds_{sp}.csv")
        sf = pd.read_csv(d / "split" / f"{sp}.csv")
        assert len(pr) == len(sf) and (pr.judgement.values == sf.judgement.values).all(), \
            f"{cell}_{arm}/{sp} misaligned"
        pr = pr.copy()
        pr["split"] = sp
        if "did" in sf.columns:
            pr["did"] = sf.did.values
        else:
            # moredata splits lack did; eval/test row ORDER is byte-identical to
            # the originals and to aug_a's (asserted via judgement equality).
            ref = pd.read_csv(DATA / f"{cell}_aug_a" / "split" / f"{sp}.csv")
            assert (ref.judgement.values == sf.judgement.values).all()
            pr["did"] = ref.did.values
        frames.append(pr[["did", "judgement", "prob", "split"]])
    return pd.concat(frames, ignore_index=True)


def main():
    out = {}

    # ---------------- captions ----------------
    for cell, arms in CAP_ARMS.items():
        have = {a: load_E(cell, a) for a in arms
                if (DATA / f"{cell}_{a}" / "rm_out_seed42" / "preds_eval.csv").exists()}
        if "aug_a" not in have:
            print(f"[{cell}] aug_a baseline preds missing -- skip cell")
            continue
        base = have["aug_a"].set_index("did")
        res = {}
        for a, df in have.items():
            d = df.set_index("did").loc[base.index]
            assert (d.judgement.values == base.judgement.values).all()
            r = {
                "auc_E": float(roc_auc_score(d.judgement, d.prob)),
                "auc_eval": float(roc_auc_score(d[d.split == "eval"].judgement,
                                                d[d.split == "eval"].prob)),
                "auc_test": float(roc_auc_score(d[d.split == "test"].judgement,
                                                d[d.split == "test"].prob)),
                "n_E": int(len(d)),
            }
            if a != "aug_a":
                r["boot_vs_aug_a"] = paired_boot(d.judgement, d.prob, base.prob)
            res[a] = r
            print(cell, a, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                       for k, v in r.items() if not isinstance(v, dict)}),
                  r.get("boot_vs_aug_a", ""))
        # D2xV3 interaction: augmd vs (aug_a alone) and vs (moredata alone)
        if cell == "cap_crowd" and "augmd" in res and "moredata" in have:
            md = have["moredata"].set_index("did").loc[base.index]
            am = have["augmd"].set_index("did").loc[base.index]
            res["augmd"]["boot_vs_moredata"] = paired_boot(am.judgement, am.prob, md.prob)
        # bank comparisons on E (full-bank VA_nl seed-mean OOF, keyed by did)
        bk_path = HERE / f"bank_oof_probs_{cell}.csv"
        if bk_path.exists():
            bk = pd.read_csv(bk_path).set_index("did")
            for a, df in have.items():
                d = df.set_index("did").loc[base.index]
                b = bk.loc[base.index]
                assert (b.y.values == d.judgement.values).all()
                res[a]["boot_vs_bank_full"] = paired_boot(d.judgement, d.prob,
                                                          b.bank_full_nl)
            res["bank_full_nl"] = {"auc_E": float(
                roc_auc_score(bk.loc[base.index].y, bk.loc[base.index].bank_full_nl))}
        out[cell] = res

    # ---------------- CW mirror ----------------
    pop = pd.read_csv(CW / "cw_population_with_splits.csv")
    pop["did"] = pop.id.astype(str)
    cw = {}
    for arm in ("cw_raw", "cw_augk20", "cw_augk40"):
        d = DATA / arm / "rm_out_seed42"
        if not (d / "preds_eval.csv").exists():
            print(arm, "preds missing -- skipped")
            continue
        frames = []
        for sp in ("eval", "test"):
            pr = pd.read_csv(d / f"preds_{sp}.csv")
            sf = pd.read_csv(DATA / arm / "split" / f"{sp}.csv")
            assert len(pr) == len(sf) and (pr.judgement.values == sf.judgement.values).all()
            pr = pr.copy()
            pr["split"] = sp
            pr["did"] = sf.did.values
            frames.append(pr)
        df = pd.concat(frames, ignore_index=True).set_index("did")
        big = pop.set_index("did").loc[df.index]
        assert (big.judgement.values == df.judgement.values).all()
        cw[arm] = {
            "auc_monitor": float(roc_auc_score(df[df.split == "eval"].judgement,
                                               df[df.split == "eval"].prob)),
            "auc_test": float(roc_auc_score(df[df.split == "test"].judgement,
                                            df[df.split == "test"].prob)),
            "T_big_monitor_ref": float(roc_auc_score(big[df.split == "eval"].judgement,
                                                     big[df.split == "eval"].dense_prob)),
            "T_big_test_ref": float(roc_auc_score(big[df.split == "test"].judgement,
                                                  big[df.split == "test"].dense_prob)),
        }
        for sp, key in (("eval", "monitor"), ("test", "test")):
            m = df.split == sp
            cw[arm][f"boot_vs_Tbig_{key}"] = paired_boot(
                df[m].judgement, df[m].prob, big[m].dense_prob)
        print(arm, json.dumps({k: round(v, 4) for k, v in cw[arm].items()
                               if isinstance(v, float)}))
    # bank reference: terminal 144-col VA_nl per-row preds (round7 pop_nl)
    st = np.load(CW / "round7_state.npz", allow_pickle=True)
    pop_nl = pd.Series(st["pop_nl"], index=[str(x) for x in st["ids"]])
    for arm in list(cw):
        d = DATA / arm / "rm_out_seed42"
        if not (d / "preds_eval.csv").exists():
            continue
        frames = []
        for sp in ("eval", "test"):
            pr = pd.read_csv(d / f"preds_{sp}.csv")
            sf = pd.read_csv(DATA / arm / "split" / f"{sp}.csv")
            pr = pr.copy(); pr["split"] = sp; pr["did"] = sf.did.astype(str).values
            frames.append(pr)
        df = pd.concat(frames, ignore_index=True)
        bnk = pop_nl.loc[df.did].values
        for sp, key in (("eval", "monitor"), ("test", "test")):
            m = (df.split == sp).values
            cw[arm][f"boot_vs_bank_{key}"] = paired_boot(
                df.judgement[m], df.prob[m], bnk[m])

    if "cw_raw" in cw and "cw_augk20" in cw:
        raw = pd.concat([pd.read_csv(DATA / "cw_raw" / "rm_out_seed42" / f"preds_{sp}.csv")
                        .assign(split=sp) for sp in ("eval", "test")], ignore_index=True)
        aug = pd.concat([pd.read_csv(DATA / "cw_augk20" / "rm_out_seed42" / f"preds_{sp}.csv")
                        .assign(split=sp) for sp in ("eval", "test")], ignore_index=True)
        assert (raw.judgement.values == aug.judgement.values).all()
        for sp, key in (("eval", "monitor"), ("test", "test")):
            m = raw.split == sp
            cw[f"augk20_minus_raw_{key}"] = paired_boot(
                raw[m].judgement, aug[m].prob, raw[m].prob)
            print(f"augk20-raw {key}", cw[f"augk20_minus_raw_{key}"])
    out["cw_mirror"] = cw

    (HERE / "v3_audit_results.json").write_text(json.dumps(out, indent=2))
    print("wrote", HERE / "v3_audit_results.json")


if __name__ == "__main__":
    main()
