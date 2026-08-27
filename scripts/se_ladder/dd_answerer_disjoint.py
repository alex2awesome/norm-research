"""Deep dive follow-up: answerer-disjoint robustness re-run (2026-06-11).

Audit section 2 of se_deep_dive_2026_06_12.md found answerer identity alone
predicts the label at 0.65-0.67 AUC on covered test rows. This script
re-evaluates the ladder's EXISTING feature matrices under splits that are
BOTH answerer-disjoint and question-disjoint (union-find components over
{owner, question} keys; each component hashed to train/test 80/20).

Per slice (crse, so_python, so_js, so_sql):
  - join OwnerUserId (SO: answers parquet; CR.SE: stream raw Posts.xml);
    missing owners -> "anon_{answer_id}" (own singleton group)
  - build disjoint split; sanity: zero owner/question overlap across
    splits, identity-only coverage on test = 0 (AUC 0.5 by construction)
  - re-fit on the SAME cached features: bank ENS (LR + RF(500) rank-avg,
    same feature-selection rules as se_ladder_eval.py), TF-IDF answer-body
    LR (train->test), bge probe LR (cached embeddings). so_python also gets
    the expanded ENS (bank + h1-h19).
  - same models also run on the ORIGINAL split (train->test) so the
    before/after comparison is protocol-identical (the ladder's TF-IDF
    margin was 5-fold CV, not train->test).

CPU only. No rescoring, no GPU.
Writes outputs/v2_analysis/se_ladder/dd_answerer_disjoint.json
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT_DIR = REPO / "outputs/v2_analysis/se_ladder"
DS = REPO / "datasets"

SLICES = {
    "crse": {
        "balanced": DS / "code-review/crse_balanced_v2/crse_v2_propensity_balanced.csv.gz",
        "owner_src": "crse_xml",
    },
    "so_python": {
        "balanced": DS / "stackoverflow_python/balanced/so_python_v2_propensity_balanced.csv.gz",
        "owner_src": DS / "stackoverflow_python/so_python_answers.parquet",
    },
    "so_js": {
        "balanced": DS / "stackoverflow_js/balanced/so_js_v2_propensity_balanced.csv.gz",
        "owner_src": DS / "stackoverflow_js/so_js_answers.parquet",
    },
    "so_sql": {
        "balanced": DS / "stackoverflow_sql/balanced/so_sql_v2_propensity_balanced.csv.gz",
        "owner_src": DS / "stackoverflow_sql/so_sql_answers.parquet",
    },
}

OWNER_RE = re.compile(rb'OwnerUserId="(\d+)"')
ID_RE = re.compile(rb'Id="(\d+)"')
PTYPE_RE = re.compile(rb'PostTypeId="(\d+)"')


def norm_id(x) -> str:
    return str(int(float(x)))


def crse_owner_map() -> dict:
    om = {}
    with open(DS / "codereview_se/raw_dump/Posts.xml", "rb") as f:
        for line in f:
            if b'PostTypeId="2"' not in line:
                continue
            m_id = ID_RE.search(line)
            m_ow = OWNER_RE.search(line)
            if m_id and m_ow:
                om[m_id.group(1).decode()] = m_ow.group(1).decode()
    return om


def so_owner_map(path: Path) -> dict:
    t = pq.read_table(path, columns=["Id", "OwnerUserId"]).to_pandas()
    t = t[t.OwnerUserId.notna()]
    return dict(zip(t.Id.astype(np.int64).astype(str),
                    t.OwnerUserId.astype(np.int64).astype(str)))


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        p = self.parent.setdefault(x, x)
        if p != x:
            self.parent[x] = p = self.find(p)
        return p

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def disjoint_split(bal: pd.DataFrame) -> pd.Series:
    """Train/test split where no owner and no question spans splits."""
    uf = UnionFind()
    for aid, qid, ow in zip(bal.answer_id, bal.question_id, bal.owner):
        uf.union(("q", qid), ("o", ow))
    # component id per row (via question node)
    comp = bal.question_id.map(lambda q: uf.find(("q", q)))
    # deterministic representative per component: min answer_id (as int)
    rep = (pd.DataFrame({"comp": comp,
                         "aid": bal.answer_id.astype(np.int64)})
           .groupby("comp").aid.min())
    comp_rep = comp.map(rep)

    def assign(r):
        h = int(hashlib.md5(f"adj1::{r}".encode()).hexdigest()[:8], 16)
        return "train" if h / 0xFFFFFFFF < 0.80 else "test"

    rep_split = {r: assign(r) for r in rep.unique()}
    split = comp_rep.map(rep_split)
    sizes = comp.value_counts()
    print(f"  components: {len(sizes)}, max size {sizes.iloc[0]} "
          f"({sizes.iloc[0] / len(bal):.3f} of rows)", flush=True)
    return split


def bank_features(df: pd.DataFrame, tr_mask: np.ndarray) -> list:
    cand = [c for c in df.columns
            if c.endswith("_score") or c.endswith("_applied")]
    tr = df[tr_mask]
    feats = []
    for c in cand:
        v = tr[c]
        if v.notna().mean() < 0.05:
            continue
        nz = v[v.notna()]
        if nz.nunique() < 3 and not c.endswith("_applied"):
            continue
        if c.endswith("_applied") and nz.nunique() < 2:
            continue
        feats.append(c)
    return feats


def bank_ens_auc(df, feats, tr_mask, te_mask):
    Xtr = df.loc[tr_mask, feats].values.astype(float)
    Xte = df.loc[te_mask, feats].values.astype(float)
    ytr = df.loc[tr_mask, "label"].values
    yte = df.loc[te_mask, "label"].values
    lr = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                              solver="liblinear"))])
    rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("clf", RandomForestClassifier(
                       n_estimators=500, min_samples_leaf=3,
                       n_jobs=16, random_state=0))])
    lr.fit(Xtr, ytr)
    rf.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]
    p_rf = rf.predict_proba(Xte)[:, 1]
    p_en = (rankdata(p_lr) + rankdata(p_rf)) / (2 * len(p_lr))
    return {"LR": float(roc_auc_score(yte, p_lr)),
            "RF": float(roc_auc_score(yte, p_rf)),
            "ENS": float(roc_auc_score(yte, p_en)),
            "n_features": len(feats)}


def tfidf_auc(body, label, tr_mask, te_mask):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200000)
    Xtr = vec.fit_transform(body[tr_mask])
    Xte = vec.transform(body[te_mask])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, label[tr_mask])
    return float(roc_auc_score(label[te_mask],
                               clf.predict_proba(Xte)[:, 1]))


def bge_auc(E, label, tr_mask, te_mask):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(E[tr_mask], label[tr_mask])
    return float(roc_auc_score(label[te_mask],
                               clf.predict_proba(E[te_mask])[:, 1]))


def identity_auc(bal, split_col):
    tr = bal[bal[split_col] == "train"]
    te = bal[bal[split_col] == "test"]
    hist = tr.groupby("owner").judgement.mean()
    cov = te.owner.isin(hist.index)
    res = {"test_coverage": float(cov.mean()), "n_covered": int(cov.sum())}
    te2 = te[cov]
    if len(te2) > 50 and te2.judgement.nunique() == 2:
        res["auc_on_covered"] = float(
            roc_auc_score(te2.judgement, te2.owner.map(hist)))
    return res


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = OUT_DIR / "dd_answerer_disjoint.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for slice_name, cfg in SLICES.items():
        if only and slice_name != only:
            continue
        print(f"=== {slice_name} ===", flush=True)
        bal = pd.read_csv(cfg["balanced"],
                          dtype={"question_id": str, "answer_id": str})
        bal["answer_id"] = bal.answer_id.map(norm_id)
        bal["question_id"] = bal.question_id.map(norm_id)
        om = (crse_owner_map() if cfg["owner_src"] == "crse_xml"
              else so_owner_map(cfg["owner_src"]))
        bal["owner"] = bal.answer_id.map(om)
        miss = bal.owner.isna()
        print(f"  owner coverage {1 - miss.mean():.4f}; "
              f"{miss.sum()} hash-assigned singletons", flush=True)
        bal.loc[miss, "owner"] = "anon_" + bal.loc[miss, "answer_id"]

        bal["dsplit"] = disjoint_split(bal)
        res = {"n": int(len(bal)),
               "owner_coverage": float(1 - miss.mean()),
               "n_train_disjoint": int((bal.dsplit == "train").sum()),
               "n_test_disjoint": int((bal.dsplit == "test").sum()),
               "test_label_mean_disjoint": float(
                   bal[bal.dsplit == "test"].judgement.mean())}

        # sanity: disjointness + identity-only AUC
        tr_o = set(bal.loc[bal.dsplit == "train", "owner"])
        te_o = set(bal.loc[bal.dsplit == "test", "owner"])
        tr_q = set(bal.loc[bal.dsplit == "train", "question_id"])
        te_q = set(bal.loc[bal.dsplit == "test", "question_id"])
        res["owner_overlap"] = len(tr_o & te_o)
        res["question_overlap"] = len(tr_q & te_q)
        assert res["owner_overlap"] == 0 and res["question_overlap"] == 0
        res["identity_only_disjoint"] = identity_auc(bal, "dsplit")
        res["identity_only_original"] = identity_auc(bal, "split")

        # attach splits to the ladder input parquet via answer_id
        inp = pd.read_parquet(OUT_DIR / f"{slice_name}_input.parquet")
        inp["answer_id_n"] = inp.answer_id.map(norm_id)
        amap = dict(zip(bal.answer_id, bal.dsplit))
        inp["dsplit"] = inp.answer_id_n.map(amap)
        assert inp.dsplit.notna().all()

        # cached features
        shard_dir = OUT_DIR / "shards" / slice_name
        scored = pd.concat([pd.read_parquet(p) for p in
                            sorted(shard_dir.glob("shard_*.parquet"))],
                           ignore_index=True)
        df = inp[["row_id", "label", "split", "dsplit"]].merge(
            scored, on="row_id", how="inner")
        assert len(df) == len(inp)
        E = np.load(OUT_DIR / f"{slice_name}_bge.npy").astype(np.float32)
        assert len(E) == len(inp)
        body = inp.body.fillna(" ").values
        label = inp.label.values

        for tag, col in (("original", "split"), ("disjoint", "dsplit")):
            tr_m = (inp[col] == "train").values
            te_m = (inp[col] == "test").values
            feats = bank_features(df, (df[col] == "train").values)
            r = {"bank": bank_ens_auc(df, feats,
                                      (df[col] == "train").values,
                                      (df[col] == "test").values),
                 "tfidf_auc": tfidf_auc(body, label, tr_m, te_m),
                 "bge_auc": bge_auc(E, label, tr_m, te_m)}
            if slice_name == "so_python":
                h = pd.read_parquet(OUT_DIR / "so_python_hmetrics.parquet")
                dfh = df.merge(h, on="row_id", how="left")
                feats_h = bank_features(dfh, (dfh[col] == "train").values)
                r["bank_expanded"] = bank_ens_auc(
                    dfh, feats_h, (dfh[col] == "train").values,
                    (dfh[col] == "test").values)
            res[tag] = r
            print(f"  [{tag}] {json.dumps(r)}", flush=True)

        out[slice_name] = res
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
