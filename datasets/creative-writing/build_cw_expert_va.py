#!/usr/bin/env python3
"""Materialise the V/A/T populations + dense-standard splits for the two CW
EXPERT cells that never got the mature instrument treatment:

  cw_royalroad_verdict  market VERDICT  -- RoyalRoad opening chapter -> KU/Amazon
                        pickup, n=1,274 balanced (637/637), already confound-audited
                        (notes/2026-06-12__taste-taxonomy.md S17m)
  cw_wigleaf_curation   editorial CURATION -- flash-fiction piece -> Wigleaf Top-50
                        editor's cut, n=1,568 (404 pos / 1,164 neg), presentation
                        leak already fixed (fetch_source AUC .90 -> .500)

REUSE, NOT REBUILD (feedback_reuse_before_rebuild): both populations are read
verbatim off the existing audited builds and both splits are the EXISTING stable
md5 hash splits, grouped by fiction/story id -- never a seeded shuffle
(feedback_stable_hash_splits):

  royalroad: build_topic_stratified.py splitof() = md5("split::"+fiction_id)%1000,
             <800 train / <900 eval / test  (fiction-grouped; this is the split that
             ships with royalroad_v2_fiction_topicstrat.csv.gz -- NOT deconfound_v2.py's
             md5(fiction_id)%10, which belongs to the smaller n=564 deconf_v2 build)
  wigleaf:   build_dataset.py split_of()        = md5(title|author|year)%10, 0-7/8/9

Nothing here re-cleans, re-scrapes or re-labels; it only reshapes the audited
rows into the layout the frozen dense-standard + va_gemma_banks machinery reads.

  python datasets/creative-writing/build_cw_expert_va.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
CW = REPO / "datasets/creative-writing"

RR_SRC = CW / "royalroad_stubs/built/royalroad_v2_fiction_topicstrat.csv.gz"
WG_DIR = CW / "wigleaf/built"

DENSE_RECIPE = ("Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, 2 epochs, "
                "gradient-checkpointing, select-on-eval (dense-standard, no deviation)")


def sha1_20(s: str) -> str:
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:20]


def write_cell(slug, base: Path, df, manifest_extra):
    """df must carry: row_id, group, text, judgement, split (+ any extra cols)."""
    va = base / "va"
    dense = base / "dense_standard"
    (dense / "split").mkdir(parents=True, exist_ok=True)
    va.mkdir(parents=True, exist_ok=True)

    df = df.reset_index(drop=True)
    assert df["row_id"].is_unique, f"{slug}: row_id not unique"
    assert df["judgement"].isin([0, 1]).all(), f"{slug}: non-binary judgement"
    assert df["split"].isin(["train", "eval", "test"]).all(), f"{slug}: bad split"
    assert df["text"].astype(str).str.strip().str.len().gt(0).all(), f"{slug}: empty text"

    df.to_csv(va / "population.csv.gz", index=False, compression="gzip")

    cols = ["text", "judgement", "group", "row_id"]
    df[cols].to_csv(dense / "data.csv", index=False)
    for sp in ("train", "eval", "test"):
        df.loc[df["split"] == sp, cols].to_csv(dense / f"split/{sp}.csv", index=False)

    counts = df["split"].value_counts().to_dict()
    pos = df.groupby("split")["judgement"].sum().astype(int).to_dict()
    man = {
        "cell": slug,
        "n": int(len(df)),
        "pos_rate": float(df["judgement"].mean()),
        "n_pos_absolute": int(df["judgement"].sum()),
        "n_neg_absolute": int((1 - df["judgement"]).sum()),
        "n_groups": int(df["group"].nunique()),
        "split_row_counts": {k: int(v) for k, v in counts.items()},
        "split_pos_counts_ABSOLUTE": {k: int(v) for k, v in pos.items()},
        "split_neg_counts_ABSOLUTE": {k: int(counts[k] - pos[k]) for k in counts},
        "split_pos_rates": {k: float(v / counts[k]) for k, v in pos.items()},
        "recipe": DENSE_RECIPE,
        **manifest_extra,
    }
    (dense / "manifest.json").write_text(json.dumps(man, indent=2))
    (va / "population_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"[{slug}] n={man['n']} pos={man['n_pos_absolute']} "
          f"({man['pos_rate']:.4f}) groups={man['n_groups']}")
    print(f"[{slug}] split rows {man['split_row_counts']} "
          f"| ABS pos {man['split_pos_counts_ABSOLUTE']} "
          f"| ABS neg {man['split_neg_counts_ABSOLUTE']}")
    print(f"[{slug}] -> {va/'population.csv.gz'} , {dense}")
    return man


# ------------------------------------------------------------- royalroad -----
def build_royalroad():
    df = pd.read_csv(RR_SRC)
    out = pd.DataFrame({
        "row_id": df["fiction_id"].astype(str),
        "group": df["fiction_id"].astype(str),           # one opening chapter per fiction
        "topic_cluster": df["topic_cluster"].astype(str),
        "wayback_era": df["wayback_era"].astype(str),
        "text": df["text"].astype(str),
        "judgement": df["judgement"].astype(int),
        "split": df["split"].astype(str),
    })
    # The canonical split is scripts/datasets/build_topic_stratified.py splitof():
    #   bucket = md5("split::" + fiction_id) % 1000 -> <800 train / <900 eval / test
    # (fiction-grouped, stable hash, never a seeded shuffle). Verify, don't assume.
    def splitof(f):
        b = int(hashlib.md5(("split::" + str(f)).encode()).hexdigest(), 16) % 1000
        return "train" if b < 800 else ("eval" if b < 900 else "test")
    agree = float((out["row_id"].map(splitof) == out["split"]).mean())
    print(f"[royalroad] stable-hash split verification: {agree:.4f} agreement with "
          f'md5("split::"+fiction_id)%1000 (build_topic_stratified.splitof)')
    assert agree == 1.0, "royalroad split is NOT the documented stable hash"
    return write_cell(
        "cw_royalroad_verdict", CW / "royalroad_stubs", out,
        {"title": "RoyalRoad market VERDICT (opening chapter -> KU/Amazon pickup)",
         "source": str(RR_SRC.relative_to(REPO)),
         "group_column": "fiction_id (one opening chapter per fiction)",
         "secondary_group_column": "topic_cluster",
         "n_topic_clusters": int(out["topic_cluster"].nunique()),
         "split_rule": "EXISTING fiction-grouped stable hash "
                       'md5("split::"+fiction_id)%1000 -> <800 train / <900 eval / test '
                       "(scripts/datasets/build_topic_stratified.py splitof); reused "
                       "verbatim, never reshuffled; verified 1.0 agreement at build time",
         "split_hash_agreement": agree,
         "confound_audit": "notes/2026-06-12__taste-taxonomy.md S17m (2026-06-16): both "
                           "pools wayback-sourced, chapter_rank=1, era->y corr -0.079, "
                           "LEXICAL .588 / REGISTER .521 -> '<0.6 CLEAN'",
         "class_weighting": "NOT required (balanced 637/637)"})


# --------------------------------------------------------------- wigleaf -----
def build_wigleaf():
    frames = []
    for sp in ("train", "eval", "test"):
        d = pd.read_csv(WG_DIR / f"{sp}.csv.gz")
        d["split"] = sp
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str)
    rid = df["text"].map(sha1_20)
    if not rid.is_unique:                       # exact-duplicate stories, if any
        dup = int(len(rid) - rid.nunique())
        print(f"[wigleaf] WARNING {dup} exact-duplicate texts; disambiguating by index")
        rid = [f"{r}_{i}" if c > 1 else r
               for i, (r, c) in enumerate(zip(rid, rid.map(rid.value_counts())))]
    out = pd.DataFrame({
        "row_id": list(rid),
        "group": list(rid),                     # one story per row
        "magazine": df["magazine"].astype(str),
        "year": df["year"].astype(int),
        "fetch_source": df["fetch_source"].astype(str),
        "text": df["text"],
        "judgement": df["judgement"].astype(int),
        "split": df["split"],
    })
    return write_cell(
        "cw_wigleaf_curation", CW / "wigleaf", out,
        {"title": "Wigleaf editorial CURATION (flash fiction -> Top-50 editor's cut)",
         "source": "datasets/creative-writing/wigleaf/built/{train,eval,test}.csv.gz",
         "group_column": "story id (sha1(text)[:20]; one story per row)",
         "secondary_group_column": "magazine",
         "n_magazines": int(out["magazine"].nunique()),
         "split_rule": "EXISTING stable md5(title|author|year)%10 -> 0-7 train / 8 eval / "
                       "9 test (scripts/build_dataset.py split_of); reused verbatim by "
                       "reading the three built files, never reshuffled",
         "leak_audit": "presentation leak fixed upstream: identical extract/bio-strip/"
                       "CMS-strip/normalise pipeline for both classes "
                       "(scripts/wig_textproc.py); fetch_source AUC .90 -> .500 "
                       "(notes/2026-06-12__taste-taxonomy.md:933-944)",
         "class_weighting": "REQUIRED: --class_weight_auto on the dense arm "
                            "(404 absolute positives / 1,164 negatives)",
         "power_caveat": "404 absolute positives is the same order of magnitude as the "
                         "mathlib false-null case (~360 minority train rows) that "
                         "motivated the pre-kill checklist; every readout from this cell "
                         "carries a small-minority power caveat."})


if __name__ == "__main__":
    build_royalroad()
    print()
    build_wigleaf()
    print("\nBUILD_CW_EXPERT_VA_DONE")
