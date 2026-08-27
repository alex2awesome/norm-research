#!/usr/bin/env python3
"""Build the Direction-2 (more training data) and Direction-3 (feature-augmented
dense) datasets for the two caption cells (2026-08-07 V+A+T fusion task).

Direction 2 (moredata): train split grows, eval/test byte-identical.
  cap_crowd:    + scored rows with crowd_mean, 50 <= crowd_votes < 100, contest in
                the dense TRAIN bucket (or unseen by the split), labeled against the
                contest's existing votes>=100 median (exact-median ties dropped;
                rows without an existing contest median dropped -- 40 unseen-contest
                candidates, a per-contest own-median would be too noisy at that n).
  cap_finalist: + neg_random rows (label 0) whose contest is in the dense TRAIN
                bucket. (neg_random from eval/test-bucket contests are EXCLUDED --
                the split is contest-grouped; adding them to train would leak
                group identity across the split.)

Direction 3 (aug): same rows/splits as the original dense cells, text replaced by
  full text:
      <caption>
  VA metrics:
      <name>: <score>            (arm a)
      <name> (importance W): <score>   (arm b)
  Top-10 criteria of the cell's cleaned VA matrix ranked by TRAIN-ONLY grouped
  permutation importance (GroupKFold(3) within the dense train split, frozen
  HistGB, permutation_importance on the held-out inner fold, roc_auc, 5 repeats,
  mean over folds). Scores are label-blind judge outputs -> safe on all splits;
  y never appears in the prompt; importance never sees eval/test rows.

Outputs (local, shipped to sk3 by the launcher):
  methods/taste_decomposition/fusion/dense_data/{cell}_{moredata|aug_a|aug_b}/
      data.csv  split/{train,eval,test}.csv  manifest.json

CPU only. Usage: python3 build_direction23_data.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
TD = HERE.parent
REPO = TD.parents[1]
CAP = REPO / "datasets/humor/caption_multiy"
OUT = HERE / "dense_data"
OUT.mkdir(exist_ok=True)

MIN_VOTES_EXTRA = 50  # extras band: 50 <= votes < 100 (MIN_VOTES=100 is the modeled pool)
TOP_K = 10


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


L1 = load_module(TD / "layer1_gemma_cells.py", "l1_fusion_build")
capagg = sys.modules["capagg_taste_decomp"]


def fmt(v):
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")


def main():
    t0 = time.time()
    c = L1._caption_pools()
    meta = capagg.load_pool()
    manifest_all = {}

    # per-contest votes>=100 median (the modeled pool's own median), for extras labeling
    from collections import defaultdict
    by_contest = defaultdict(list)
    for did in c["X_by_id"]:
        m = meta.get(did)
        if m is None:
            continue
        if m.get("crowd_mean") is not None and (m.get("crowd_votes") or 0) >= capagg.MIN_VOTES:
            by_contest[m["contest"]].append(m["crowd_mean"])
    med_by_contest = {k: float(np.median(v)) for k, v in by_contest.items() if len(v) >= 6}

    for cell, dl_name, id_key, y_key in (
            ("cap_crowd", "crowd", "crowd_ids", "y_crowd"),
            ("cap_finalist", "finalist", "hardneg_ids", "y_fin")):
        dl = CAP / "dense_llama" / dl_name
        data = pd.read_csv(dl / "data.csv")
        splits = {s: pd.read_csv(dl / "split" / f"{s}.csv") for s in ("train", "eval", "test")}

        # ---- map original dense rows -> did by (contest, text); verify 0 unmatched
        by_ct = {}
        for did in c[id_key]:
            if did in c["X_by_id"]:
                by_ct[(str(c["contest_by_id"][did]), meta[did]["text"])] = did
        for nm, df in [("data", data)] + list(splits.items()):
            dids = [by_ct.get((str(g), t)) for t, g in zip(df.text, df.group)]
            assert all(d is not None for d in dids), f"{cell}/{nm}: unmatched rows"
            df["did"] = dids
        bucket_of_contest = {}
        for s in ("train", "eval", "test"):
            for g in splits[s].group.astype(str).unique():
                bucket_of_contest.setdefault(g, s)

        # ================= Direction 2: moredata =================
        extras = []
        if cell == "cap_crowd":
            for did, ro in c["role_by_id"].items():
                m = meta.get(did)
                if m is None or did in c["y_crowd"] or did not in c["X_by_id"]:
                    continue
                cm, cv = m.get("crowd_mean"), (m.get("crowd_votes") or 0)
                if cm is None or not (MIN_VOTES_EXTRA <= cv < capagg.MIN_VOTES):
                    continue
                if bucket_of_contest.get(str(m["contest"])) not in (None, "train"):
                    continue
                med = med_by_contest.get(m["contest"])
                if med is None or cm == med:
                    continue
                extras.append({"text": m["text"], "judgement": int(cm > med),
                               "group": m["contest"], "did": did})
        else:
            for did, ro in c["role_by_id"].items():
                if ro != "neg_random" or did not in c["X_by_id"]:
                    continue
                m = meta.get(did)
                if m is None:
                    continue
                if bucket_of_contest.get(str(m["contest"])) not in (None, "train"):
                    continue
                extras.append({"text": m["text"], "judgement": 0,
                               "group": m["contest"], "did": did})
        ex_df = pd.DataFrame(extras)
        md_dir = OUT / f"{cell}_moredata"
        (md_dir / "split").mkdir(parents=True, exist_ok=True)
        cols = ["text", "judgement", "group"]
        grown_train = pd.concat([splits["train"][cols], ex_df[cols]], ignore_index=True)
        grown_train.to_csv(md_dir / "split" / "train.csv", index=False)
        for s in ("eval", "test"):
            splits[s][cols].to_csv(md_dir / "split" / f"{s}.csv", index=False)
            assert open(md_dir / "split" / f"{s}.csv").read() != ""  # sanity
        pd.concat([data[cols], ex_df[cols]], ignore_index=True).to_csv(md_dir / "data.csv", index=False)
        ex_df.to_csv(md_dir / "extras_manifest.csv", index=False)
        md_manifest = {
            "cell": cell, "direction": 2,
            "n_train_orig": int(len(splits["train"])), "n_extras": int(len(ex_df)),
            "n_train_grown": int(len(grown_train)),
            "train_growth_pct": round(100 * len(ex_df) / len(splits["train"]), 1),
            "extras_pos_rate": float(ex_df.judgement.mean()) if len(ex_df) else None,
            "n_eval": int(len(splits["eval"])), "n_test": int(len(splits["test"])),
            "eval_test_unchanged": True,
            "extras_rule": ("crowd_mean, 50<=votes<100, train/unseen-bucket contest, "
                            "labeled vs existing votes>=100 contest median, ties dropped")
                           if cell == "cap_crowd" else
                           "neg_random (label 0), train-bucket contest only",
        }
        (md_dir / "manifest.json").write_text(json.dumps(md_manifest, indent=2))
        print(f"[{cell} moredata] train {len(splits['train'])} -> {len(grown_train)} "
              f"(+{md_manifest['train_growth_pct']}%), extras pos {md_manifest['extras_pos_rate']}")

        # ================= Direction 3: aug arms =================
        # cleaned VA matrix on the cell population (label-blind guard), layer1-identical
        ids_all = sorted(d for d in c[id_key] if d in c["X_by_id"])
        A = np.array([c["X_by_id"][d] for d in ids_all], dtype=float)
        V = np.array([c["V_by_id"][d] for d in ids_all], dtype=float)
        Ac, a_keep = capagg.clean_cols(A)
        Vc, v_keep = capagg.clean_cols(V)
        names = [c["v_names"][j] for j in v_keep] + [c["a_names"][j] for j in a_keep]
        VA = np.column_stack([Vc, Ac])
        row_of = {d: i for i, d in enumerate(ids_all)}
        y_by_id = c[y_key]

        # train-only grouped permutation importance
        tr_ids = [d for d in splits["train"].did]
        tr_idx = np.array([row_of[d] for d in tr_ids])
        Xtr = VA[tr_idx]
        ytr = np.array([y_by_id[d] for d in tr_ids])
        gtr = np.array([str(c["contest_by_id"][d]) for d in tr_ids])
        imps = np.zeros(VA.shape[1])
        nf = 0
        for itr, ite in GroupKFold(n_splits=3).split(Xtr, ytr, gtr):
            m = L1._fit_gbm(L1.GRID[1], seed=0)
            m.fit(Xtr[itr], ytr[itr])
            r = permutation_importance(m, Xtr[ite], ytr[ite], scoring="roc_auc",
                                       n_repeats=5, random_state=0, n_jobs=-1)
            imps += r.importances_mean
            nf += 1
        imps /= nf
        order = np.argsort(-imps)[:TOP_K]
        top = [{"name": names[j], "col": int(j), "importance": float(imps[j])} for j in order]
        wsum = sum(max(t["importance"], 0) for t in top) or 1.0
        for t in top:
            t["weight"] = round(max(t["importance"], 0) / wsum, 2)
        print(f"[{cell} aug] top-{TOP_K} by train-fold permutation importance:")
        for t in top:
            print(f"    {t['weight']:.2f}  {t['importance']:+.4f}  {t['name']}")

        def aug_text(did, arm):
            i = row_of[did]
            lines = ["full text:", f"    {meta[did]['text']}", "VA metrics:"]
            for t in top:
                v = fmt(float(VA[i, t["col"]]))
                if arm == "a":
                    lines.append(f"    {t['name']}: {v}")
                else:
                    lines.append(f"    {t['name']} (importance {t['weight']:.2f}): {v}")
            return "\n".join(lines)

        for arm in ("a", "b"):
            ad = OUT / f"{cell}_aug_{arm}"
            (ad / "split").mkdir(parents=True, exist_ok=True)
            for nm, df in [("data", data)] + list(splits.items()):
                out = df.copy()
                out["text"] = [aug_text(d, arm) for d in out.did]
                out = out[["text", "judgement", "group", "did"]]
                if nm == "data":
                    out.to_csv(ad / "data.csv", index=False)
                else:
                    out.to_csv(ad / "split" / f"{nm}.csv", index=False)
            (ad / "manifest.json").write_text(json.dumps({
                "cell": cell, "direction": 3, "arm": arm, "top_criteria": top,
                "rows_identical_to_original_split": True,
                "importance_protocol": "GroupKFold(3) within dense train split, frozen "
                                       "HistGB (leaves=31,lr=.06), permutation_importance "
                                       "roc_auc n_repeats=5 on inner held-out fold, mean",
            }, indent=2))
            print(f"[{cell} aug_{arm}] wrote {ad}")

        manifest_all[cell] = {"moredata": md_manifest, "aug_top": top}

    (OUT / "build_manifest.json").write_text(json.dumps(manifest_all, indent=2))
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
