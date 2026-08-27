"""Deep dive task 2: balancing-fidelity + leakage checks (2026-06-12).

Per slice (crse, so_python, so_js, so_sql):
  A. split fidelity: recompute split_of(question_id) (shared splits.py
     convention) on the balanced file vs the stored split column.
  B. answerer leakage: join OwnerUserId; (i) fraction of rows whose owner
     appears in both classes; (ii) identity-leak AUC — predict test labels
     from the owner's mean TRAIN label (owners seen in train), with coverage.
  C. near-duplicate answers across splits: cached bge embeddings (aligned
     with {slice}_input.parquet); 5K-row sample of eval+test rows, max
     cosine vs all train rows; quantiles, counts >=.95/.98/.99, top pairs.

CPU only. Run on sk3 from /lfs/skampere3/0/alexspan/norm-research.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT_DIR = REPO / "outputs/v2_analysis/se_ladder"
DS = REPO / "datasets"

SLICES = {
    "crse": {
        "balanced": DS / "code-review/crse_balanced_v2/crse_v2_propensity_balanced.csv.gz",
        "answers": DS / "codereview_se/posts.parquet",
        "answers_kind": "se_posts",
    },
    "so_python": {
        "balanced": DS / "stackoverflow_python/balanced/so_python_v2_propensity_balanced.csv.gz",
        "answers": DS / "stackoverflow_python/so_python_answers.parquet",
        "answers_kind": "so",
    },
    "so_js": {
        "balanced": DS / "stackoverflow_js/balanced/so_js_v2_propensity_balanced.csv.gz",
        "answers": DS / "stackoverflow_js/so_js_answers.parquet",
        "answers_kind": "so",
    },
    "so_sql": {
        "balanced": DS / "stackoverflow_sql/balanced/so_sql_v2_propensity_balanced.csv.gz",
        "answers": DS / "stackoverflow_sql/so_sql_answers.parquet",
        "answers_kind": "so",
    },
}


def split_of(question_id) -> str:
    h = int(hashlib.md5(str(question_id).encode()).hexdigest()[:8], 16)
    u = h / 0xFFFFFFFF
    if u < 0.80:
        return "train"
    if u < 0.90:
        return "eval"
    return "test"


def owner_map(cfg):
    if cfg["answers_kind"] == "so":
        t = pq.read_table(cfg["answers"], columns=["Id", "OwnerUserId"]).to_pandas()
        return dict(zip(t.Id.astype(np.int64).astype(str), t.OwnerUserId))
    # SE posts dump: answers are PostTypeId==2
    f = pq.ParquetFile(cfg["answers"])
    names = f.schema_arrow.names
    cols = [c for c in ("Id", "OwnerUserId", "PostTypeId") if c in names]
    t = pq.read_table(cfg["answers"], columns=cols).to_pandas()
    if "PostTypeId" in t.columns:
        t = t[t.PostTypeId.astype(str).isin(("2", "2.0"))]
    return dict(zip(t.Id.astype(np.int64).astype(str), t.OwnerUserId))


def near_dup(slice_name, res, rng):
    inp_p = OUT_DIR / f"{slice_name}_input.parquet"
    emb_p = OUT_DIR / f"{slice_name}_bge.npy"
    if not inp_p.exists() or not emb_p.exists():
        res["near_dup"] = "missing input/embeddings"
        return
    df = pd.read_parquet(inp_p, columns=["row_id", "question_id", "label",
                                         "split", "body"])
    E = np.load(emb_p).astype(np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    tr_idx = np.where((df.split == "train").values)[0]
    ho_idx = np.where((df.split != "train").values)[0]
    if len(ho_idx) > 5000:
        ho_idx = rng.choice(ho_idx, 5000, replace=False)
    Etr = E[tr_idx]
    best = np.full(len(ho_idx), -1.0, dtype=np.float32)
    best_j = np.zeros(len(ho_idx), dtype=np.int64)
    for s in range(0, len(ho_idx), 512):
        sl = ho_idx[s:s + 512]
        sims = E[sl] @ Etr.T
        bj = sims.argmax(axis=1)
        best[s:s + 512] = sims[np.arange(len(sl)), bj]
        best_j[s:s + 512] = tr_idx[bj]
    qs = {q: float(np.quantile(best, q)) for q in (0.5, 0.9, 0.99, 0.999)}
    counts = {t: int((best >= t).sum()) for t in (0.95, 0.98, 0.99, 0.995)}
    order = np.argsort(-best)[:10]
    tops = []
    for k in order:
        i, j = int(ho_idx[k]), int(best_j[k])
        tops.append({
            "cos": float(best[k]),
            "holdout": {"row_id": int(df.row_id.iloc[i]),
                        "qid": str(df.question_id.iloc[i]),
                        "label": int(df.label.iloc[i]),
                        "split": df.split.iloc[i],
                        "body": str(df.body.iloc[i])[:200]},
            "train": {"row_id": int(df.row_id.iloc[j]),
                      "qid": str(df.question_id.iloc[j]),
                      "label": int(df.label.iloc[j]),
                      "body": str(df.body.iloc[j])[:200]},
        })
    res["near_dup"] = {"n_holdout_sampled": int(len(ho_idx)),
                       "n_train": int(len(tr_idx)),
                       "max_cos_quantiles": qs, "counts_ge": counts,
                       "top_pairs": tops}


def main():
    rng = np.random.default_rng(7)
    out = {}
    for slice_name, cfg in SLICES.items():
        print(f"=== {slice_name} ===", flush=True)
        res = {}
        bal = pd.read_csv(cfg["balanced"], dtype={"question_id": str,
                                                  "answer_id": str})
        res["n"] = int(len(bal))
        # A. split fidelity
        rec = bal.question_id.map(split_of)
        res["split_mismatch_rate"] = float((rec != bal.split).mean())
        res["split_counts"] = bal.split.value_counts().to_dict()
        # qid dtype sanity: any float-looking ids?
        res["qid_floatlike_frac"] = float(bal.question_id.str.contains(r"\.").mean())

        # B. answerer leakage
        try:
            om = owner_map(cfg)
            bal["owner"] = bal.answer_id.map(
                lambda a: om.get(str(int(float(a))), np.nan))
            cov = bal.owner.notna().mean()
            res["owner_coverage"] = float(cov)
            sub = bal[bal.owner.notna()].copy()
            g = sub.groupby("owner").judgement.agg(["mean", "count"])
            both = g[(g["mean"] > 0) & (g["mean"] < 1)]
            res["owners_total"] = int(len(g))
            res["owners_with_both_classes"] = int(len(both))
            res["rows_owner_in_both_classes_frac"] = float(
                sub.owner.isin(both.index).mean())
            # identity-leak AUC
            tr = sub[sub.split == "train"]
            te = sub[sub.split == "test"]
            hist = tr.groupby("owner").judgement.agg(["mean", "count"])
            te = te[te.owner.isin(hist.index)].copy()
            res["test_owner_seen_in_train_frac"] = float(
                len(te) / max(1, (sub.split == "test").sum()))
            if len(te) > 50 and te.judgement.nunique() == 2:
                p = te.owner.map(hist["mean"]).values
                res["identity_leak_auc_on_covered_test"] = float(
                    roc_auc_score(te.judgement.values, p))
                res["n_covered_test"] = int(len(te))
        except Exception as e:  # noqa: BLE001
            res["owner_error"] = repr(e)

        # C. near-dup
        near_dup(slice_name, res, rng)
        out[slice_name] = res
        print(json.dumps({k: v for k, v in res.items() if k != "near_dup"},
                         indent=2), flush=True)
        if isinstance(res.get("near_dup"), dict):
            nd = dict(res["near_dup"])
            nd.pop("top_pairs", None)
            print("near_dup:", json.dumps(nd, indent=2), flush=True)

    (OUT_DIR / "dd_task2_checks.json").write_text(json.dumps(out, indent=2))
    print("wrote dd_task2_checks.json", flush=True)


if __name__ == "__main__":
    main()
