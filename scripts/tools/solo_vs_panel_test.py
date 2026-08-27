#!/usr/bin/env python3
"""Protocol-confound test: SOLO vs PANEL candidate scoring (2026-07-08).

Stage-1 measures each candidate scored SOLO (one rubric per prompt, loop._score_one); the
stage-2 replication scored candidates inside a 45-rubric PANEL with the bank. If panel context
attenuates subtle rubrics, stage-2's blanket nulls are a measurement artifact, not candidate
falsification. Test: on the SAME fresh replication sample, score candidates SOLO and compare
(a) score-vector correlation with the panel scores, (b) incremental-over-bank gain solo vs
panel. Diagnostic only — the corrected stage-2 protocol follows from the answer.
"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "methods")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.io_metrics import (
    MetricSpec, _stable_id, load_rubric_metrics_from_dir, make_vllm_judge_scorer)

sys.path.insert(0, "scripts/tools")
from replicate_candidates import nb_p, paired_gain, stage1_candidates  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True, help="one stage-1 leg dir")
    ap.add_argument("--data", required=True)
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--stage1-n", type=int, default=900)
    ap.add_argument("--n-replication", type=int, default=2400)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = InfillConfig(materialize_backend="vllm_offline", materialize_model=args.judge_model,
                       max_text_tokens=700, verbose=False,
                       cache_dir="outputs/ctree/B_tree/judge_cache", output_dir=str(out))
    judge = make_vllm_judge_scorer(cfg)
    bank = load_rubric_metrics_from_dir(args.bank_dir)

    cands = sorted(stage1_candidates(Path(args.leg), 0.05),
                   key=lambda c: (c["stage1_p"] or 1))[: args.top_k]
    df = pd.read_csv(args.data).dropna(subset=["text", "judgement"])
    df["judgement"] = df["judgement"].astype(int)
    leg_ids = set(df.sample(min(args.stage1_n + args.stage1_n // 2, len(df)),
                            random_state=7)[args.id_col].astype(str))
    fresh = df[~df[args.id_col].astype(str).isin(leg_ids)]
    h = fresh[args.id_col].astype(str).map(
        lambda q: hashlib.md5(f"rep2::{q}".encode()).hexdigest())
    keep_ids, ntot = [], 0
    for gid, g in fresh.assign(_h=h).sort_values("_h").groupby(args.id_col, sort=False):
        keep_ids.append(gid)
        ntot += len(g)
        if ntot >= args.n_replication:
            break
    rep = fresh[fresh[args.id_col].isin(keep_ids)].reset_index(drop=True)
    y = rep.judgement.to_numpy()
    texts = rep.text.astype(str).tolist()
    print(f"[{Path(args.leg).name}] n={len(rep)} cands={len(cands)}", flush=True)

    # PANEL scoring: identical spec list to the stage-2 run -> full cache hit
    all_cands = stage1_candidates(Path(args.leg), 0.05)
    specs = bank + [MetricSpec(metric_id=_stable_id("r2", c["name"], c["rubric"]),
                               name=c["name"], description=c["description"], kind="judge",
                               guidance=c["rubric"]) for c in all_cands]
    lv_p, ap_p = judge(specs, texts)
    nb = len(bank)
    Xb = lv_p[:, :nb]
    vi = [j for j in range(nb) if ap_p[:, j].mean() > 0.10 and np.nanstd(Xb[:, j]) > 0.05]
    Xb = Xb[:, vi]
    mu = np.nanmean(Xb, 0)
    Xb = np.where(np.isnan(Xb), np.where(np.isnan(mu), 0.5, mu), Xb)

    results = []
    name_to_panel = {c["name"]: nb + i for i, c in enumerate(all_cands)}
    for c in cands:
        spec = MetricSpec(metric_id=_stable_id("solo", c["name"], c["rubric"]), name=c["name"],
                          description=c["description"], kind="judge", guidance=c["rubric"])
        lv_s, ap_s = judge([spec], texts)          # SOLO — the stage-1 measurement context
        xs = lv_s[:, 0]
        xp = lv_p[:, name_to_panel[c["name"]]]
        both = np.isfinite(xs) & np.isfinite(xp)
        rho = float(spearmanr(xs[both], xp[both]).statistic) if both.sum() > 50 else np.nan
        def gain(x):
            ok = np.isfinite(x)
            if ok.mean() < 0.3 or np.nanstd(x[ok]) < 0.05:
                return None
            xx = np.where(ok, x, np.nanmean(x[ok]))
            da, db = paired_gain(Xb, xx, y)
            return dict(auc=float(np.mean(da)), bits=float(np.mean(db)),
                        p_auc=nb_p(da), p_bits=nb_p(db))
        gs, gp = gain(xs), gain(xp)
        rec = dict(name=c["name"], stage1_p=c["stage1_p"], stage1_bits=c["stage1_bits"],
                   solo_panel_rho=rho, solo_applic=float(np.isfinite(xs).mean()),
                   panel_applic=float(np.isfinite(xp).mean()), solo=gs, panel=gp)
        results.append(rec)
        print(f"  {c['name'][:44]:44s} rho(solo,panel)={rho:.2f} "
              f"solo_bits={gs['bits'] if gs else None} panel_bits={gp['bits'] if gp else None} "
              f"solo_p={gs['p_auc'] if gs else None}", flush=True)
    json.dump(results, open(out / "solo_vs_panel.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
