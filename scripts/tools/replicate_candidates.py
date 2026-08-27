#!/usr/bin/env python3
"""Stage-2 replication for community-infilling candidates (the two-stage discovery design).

Stage 1 (the arm runs) is deliberately conservative: Bonferroni over ALL planned proposals —
a real 0.005-0.01-bit community metric at n~945 mathematically cannot clear it. Stage 2 takes
each leg's nominal tail (kept OR confirm-dropped with confirm_p_auc < stage1-alpha and positive
confirmed bits), re-scores those candidates + the bank on a STRICTLY FRESH community sample
(the leg's seed-7 item ids are excluded), and tests incremental-over-bank gains with a
Nadeau-Bengio corrected t-test, Bonferroni over the REPLICATED SET only. Survivors are formal
stage-2 keeps.

  python scripts/tools/replicate_candidates.py \
      --legs 'outputs/ctree/arm_comparison/cw-genre-*' --skip-pooled \
      --data-template 'datasets/creative-writing/by_genre/{community}.csv.gz' \
      --leg-prefix cw-genre- --id-col prompt \
      --bank-dir datasets/creative-writing/medoid-bank-clean --judge-model <path>
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
from scipy.stats import t as tdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.io_metrics import (
    MetricSpec, _stable_id, load_rubric_metrics_from_dir, make_vllm_judge_scorer)


def paired_gain(Xb, xnew, y, seeds=(0, 1, 2, 3, 4)):
    da, db = [], []
    for seed in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xb, y):
            if len(np.unique(y[tr])) < 2:
                continue
            m0 = LogisticRegression(max_iter=2000).fit(Xb[tr], y[tr])
            m1 = LogisticRegression(max_iter=2000).fit(np.hstack([Xb[tr], xnew[tr, None]]), y[tr])
            p0 = np.clip(m0.predict_proba(Xb[te])[:, 1], 1e-6, 1 - 1e-6)
            p1 = np.clip(m1.predict_proba(np.hstack([Xb[te], xnew[te, None]]))[:, 1], 1e-6, 1 - 1e-6)
            da.append(roc_auc_score(y[te], p1) - roc_auc_score(y[te], p0))
            db.append((log_loss(y[te], p0, labels=[0, 1]) - log_loss(y[te], p1, labels=[0, 1]))
                      / np.log(2))
    return np.array(da), np.array(db)


def nb_p(diffs, n_train_frac=0.8):
    d = np.asarray(diffs)
    k = len(d)
    if k < 2 or d.std(ddof=1) == 0:
        return 1.0
    corr = 1.0 / k + (1 - n_train_frac) / n_train_frac
    t = d.mean() / (d.std(ddof=1) * np.sqrt(corr))
    return float(1 - tdist.cdf(t, df=k - 1))


def stage1_candidates(leg_dir: Path, alpha1: float):
    out = []
    for lf in glob.glob(str(leg_dir / "*" / "global_infill_ledger.json")):
        arm = Path(lf).parent.name
        for e in json.load(open(lf))["ledgers"]:
            st = str(e.get("status", ""))
            hot = st == "kept" or (st.startswith("dropped:confirm")
                                   and (e.get("confirm_p_auc") or 1) < alpha1
                                   and (e.get("confirm_bits_gain") or 0) > 0)
            if hot and e.get("rubric"):
                out.append(dict(name=e["name"], description=e.get("description", ""),
                                rubric=e["rubric"], arm=arm, stage1_status=st,
                                stage1_p=e.get("confirm_p_auc"),
                                stage1_bits=e.get("confirm_bits_gain")))
    # dedup by name (convergent proposals share evidence: keep the best-p instance)
    best = {}
    for c in out:
        k = c["name"].strip().lower()
        if k not in best or (c["stage1_p"] or 1) < (best[k]["stage1_p"] or 1):
            best[k] = c
    return list(best.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", required=True, help="glob of stage-1 leg output dirs")
    ap.add_argument("--skip-pooled", action="store_true")
    ap.add_argument("--data-template", required=True)
    ap.add_argument("--leg-prefix", required=True, help="strip from leg dirname -> community")
    ap.add_argument("--leg-suffix", default="", help="also strip this suffix (e.g. -glmprop)")
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--stage1-n", type=int, default=900, help="--n used in the stage-1 legs")
    ap.add_argument("--n-replication", type=int, default=2400)
    ap.add_argument("--alpha1", type=float, default=0.05)
    ap.add_argument("--alpha2", type=float, default=0.05)
    ap.add_argument("--salt", default="rep2", help="sample salt; use rep3 for a THIRD sample")
    ap.add_argument("--exclude-salts", default="",
                    help="comma list of prior salts whose samples to exclude (disjointness "
                         "across replication rounds; prior samples are deterministic)")
    ap.add_argument("--only-from", default=None,
                    help="stage2_ledger.json — restrict candidates to names with "
                         "rep_p_auc < --only-alpha in it (confirmation rounds)")
    ap.add_argument("--only-alpha", type=float, default=0.05)
    ap.add_argument("--include-degenerate", action="store_true",
                    help="with --only-from: also carry candidates whose prior round was "
                         "stage2_status=degenerate (rubric collapsed; use after a GEPA "
                         "rewrite gives them a scoreable rubric)")
    ap.add_argument("--solo", action="store_true",
                    help="score candidates ONE PER PROMPT (matches stage-1's loop._score_one "
                         "measurement context; the 45-rubric panel attenuates subtle rubrics)")
    ap.add_argument("--dense-rubrics", default=None,
                    help="JSON {name: dense_rubric} overriding ledger rubrics (articulation "
                         "ladder: rung-3 definition+guidance+exemplars)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = InfillConfig(materialize_backend="vllm_offline", materialize_model=args.judge_model,
                       max_text_tokens=700, verbose=False,
                       cache_dir="outputs/ctree/B_tree/judge_cache", output_dir=str(out))
    judge = make_vllm_judge_scorer(cfg)
    bank = load_rubric_metrics_from_dir(args.bank_dir)

    legs = [Path(p) for p in sorted(glob.glob(args.legs)) if Path(p).is_dir()]
    if args.skip_pooled:
        legs = [p for p in legs if "pooled" not in p.name]
    all_cands = {p.name: stage1_candidates(p, args.alpha1) for p in legs}
    if args.only_from:
        prior = json.load(open(args.only_from))
        ok_names = {r["name"] for r in prior
                    if (r.get("rep_p_auc") or 1) < args.only_alpha
                    or (r.get("rep_p_bits") or 1) < args.only_alpha
                    or (args.include_degenerate
                        and r.get("stage2_status") == "degenerate")}
        all_cands = {k: [c for c in v if c["name"] in ok_names] for k, v in all_cands.items()}
    m2 = sum(len(v) for v in all_cands.values())
    print(f"stage-2 set: {m2} candidates over {len(legs)} legs (alpha2 bar "
          f"{args.alpha2/max(m2,1):.4g})", flush=True)

    results = []
    for leg in legs:
        cands = all_cands[leg.name]
        if not cands:
            continue
        community = leg.name.replace(args.leg_prefix, "")
        if args.leg_suffix and community.endswith(args.leg_suffix):
            community = community[: -len(args.leg_suffix)]
        df = pd.read_csv(args.data_template.format(community=community)).dropna(
            subset=["text", "judgement"])
        df["judgement"] = df["judgement"].astype(int)
        # exclude the stage-1 leg's items (same seed-7 sample construction) -> strictly fresh
        leg_ids = set(df.sample(min(args.stage1_n + args.stage1_n // 2, len(df)),
                                random_state=7)[args.id_col].astype(str))
        # exclude prior replication rounds' samples (deterministic per salt) for disjointness
        for ps in [s for s in args.exclude_salts.split(",") if s.strip()]:
            avail = df[~df[args.id_col].astype(str).isin(leg_ids)]
            ph = avail[args.id_col].astype(str).map(
                lambda q: hashlib.md5(f"{ps}::{q}".encode()).hexdigest())
            got = 0
            for gid, g in avail.assign(_h=ph).sort_values("_h").groupby(args.id_col, sort=False):
                leg_ids.add(str(gid))
                got += len(g)
                if got >= args.n_replication:
                    break
        fresh = df[~df[args.id_col].astype(str).isin(leg_ids)]
        # stable-hash whole-group sample of the replication set
        h = fresh[args.id_col].astype(str).map(
            lambda q: hashlib.md5(f"{args.salt}::{q}".encode()).hexdigest())
        keep_ids, ntot = [], 0
        for gid, g in fresh.assign(_h=h).sort_values("_h").groupby(args.id_col, sort=False):
            keep_ids.append(gid)
            ntot += len(g)
            if ntot >= args.n_replication:
                break
        rep = fresh[fresh[args.id_col].isin(keep_ids)].reset_index(drop=True)
        y = rep.judgement.to_numpy()
        texts = rep.text.astype(str).tolist()
        # Pool exhaustion (aliens/topology class): a leg with no fresh items left must not
        # crash the sibling legs — record and move on.
        if len(rep) < 50 or rep.judgement.nunique() < 2:
            print(f"[{leg.name}] POOL EXHAUSTED (fresh n={len(rep)}) — leg skipped", flush=True)
            results.extend({**c, "leg": leg.name, "stage2_status": "pool-exhausted"}
                           for c in cands)
            continue
        print(f"[{leg.name}] replication n={len(rep)} (fresh; excluded {len(leg_ids)} stage-1 "
              f"groups) base={y.mean():.3f} candidates={len(cands)}", flush=True)

        dense = json.load(open(args.dense_rubrics)) if args.dense_rubrics else {}
        cspecs = [MetricSpec(
            metric_id=_stable_id("r2", c["name"], dense.get(c["name"], c["rubric"])),
            name=c["name"], description=c["description"], kind="judge",
            guidance=dense.get(c["name"], c["rubric"])) for c in cands]
        if args.solo:
            lv_b, _ = judge(bank, texts)
            cols = [judge([s], texts)[0][:, 0] for s in cspecs]
            lv = np.column_stack([lv_b] + cols) if cols else lv_b
        else:
            lv, _ = judge(bank + cspecs, texts)
        apl = np.isfinite(lv)
        nb = len(bank)
        Xb = lv[:, :nb]
        vi = [j for j in range(nb) if apl[:, j].mean() > 0.10 and np.nanstd(Xb[:, j]) > 0.05]
        Xb = Xb[:, vi]
        mu = np.nanmean(Xb, 0)
        Xb = np.where(np.isnan(Xb), np.where(np.isnan(mu), 0.5, mu), Xb)
        for ci, c in enumerate(cands):
            x = lv[:, nb + ci]
            ok = np.isfinite(x)
            if ok.mean() < 0.3 or np.nanstd(x[ok]) < 0.05:
                results.append({**c, "leg": leg.name, "stage2_status": "degenerate"})
                continue
            x = np.where(ok, x, np.nanmean(x[ok]))
            try:
                da, db = paired_gain(Xb, x, y)
            except Exception as e:  # one candidate's numeric failure must not kill the leg
                results.append({**c, "leg": leg.name, "stage2_status": f"error:{e}"[:80]})
                continue
            p_auc, p_bits = nb_p(da), nb_p(db)
            passed = (np.mean(db) >= 0.003 and np.mean(da) >= 0.0
                      and p_auc < args.alpha2 / m2 and p_bits < args.alpha2 / m2)
            rec = {**c, "leg": leg.name, "rep_n": len(rep),
                   "rep_auc_gain": float(np.mean(da)), "rep_bits_gain": float(np.mean(db)),
                   "rep_p_auc": p_auc, "rep_p_bits": p_bits,
                   "stage2_status": "KEPT" if passed else "not-replicated"}
            results.append(rec)
            print(f"  [{rec['stage2_status'][:4]:4s}] {c['name'][:48]:48s} "
                  f"rep_bits={np.mean(db):+.4f} p_auc={p_auc:.3g} p_bits={p_bits:.3g}",
                  flush=True)
    json.dump(results, open(out / "stage2_ledger.json", "w"), indent=1)
    n_kept = sum(1 for r in results if r.get("stage2_status") == "KEPT")
    print(f"STAGE-2 COMPLETE: {n_kept} formal keeps / {m2} candidates -> "
          f"{out}/stage2_ledger.json", flush=True)


if __name__ == "__main__":
    main()
