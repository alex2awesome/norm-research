#!/usr/bin/env python3
"""Build per-agency train/eval/test splits for autometrics.

Sources:
  - Claims: data/to_upload/<agency>/<agency>_YYYY_YYYY/public_submission_all_text__claims.csv.gz
  - Labels: data/bulk_downloads/match_results_summary/all_cluster_labels.csv.gz

Output schema (matches the pooled notice_and_comment.csv.gz):
  text, judgement    (judgement: 1 = matched yes, 0 = matched no)

Writes to: datasets/notice-and-comment/agencies/<agency>/{train,eval,test}.csv.gz
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_agency_splits")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "datasets" / "notice-and-comment" / "agencies"

REG_DEMO = Path("/Users/spangher/Projects/stanford-research/rfi-research/regulations-demo")
TO_UPLOAD = REG_DEMO / "data" / "to_upload"
LABELS_PATH = REG_DEMO / "data" / "bulk_downloads" / "match_results_summary" / "all_cluster_labels.csv.gz"

TIER1 = ["cms", "epa", "fws", "fda", "faa", "aphis", "noaa", "ed", "ams", "nhtsa", "uscis", "dot"]
TIER2 = ["cdc", "blm", "fs", "osha", "fsis", "irs"]


def _parse_claims(raw: str) -> list[str]:
    if not isinstance(raw, str) or not raw:
        return []
    try:
        v = json.loads(raw)
    except Exception:
        return []
    if isinstance(v, list):
        return [str(c).strip() for c in v if isinstance(c, (str, int, float)) and str(c).strip()]
    return []


def load_agency_labels(agency: str) -> pd.DataFrame:
    lab = pd.read_csv(LABELS_PATH, usecols=["cluster_uid", "matched", "agency"])
    lab = lab[lab["agency"] == agency].copy()
    lab["document_id"] = lab["cluster_uid"].str.split("__").str[-1]
    lab = lab.drop_duplicates(subset="document_id", keep="first")
    lab["judgement"] = (lab["matched"] == "yes").astype(int)
    return lab[["document_id", "judgement"]]


def build_agency(agency: str) -> pd.DataFrame:
    claim_files = sorted(glob.glob(str(TO_UPLOAD / agency / f"{agency}_*" / "public_submission_all_text__claims.csv.gz")))
    if not claim_files:
        log.warning("[%s] no claim files found", agency)
        return pd.DataFrame(columns=["text", "judgement"])

    labels = load_agency_labels(agency)
    log.info("[%s] %d labeled documents, %d claim files", agency, len(labels), len(claim_files))

    rows: list[pd.DataFrame] = []
    for f in claim_files:
        df = pd.read_csv(f, usecols=["document_id", "claims_parsed_json"])
        df["claims"] = df["claims_parsed_json"].apply(_parse_claims)
        df = df[df["claims"].map(len) > 0][["document_id", "claims"]]
        df = df.merge(labels, on="document_id", how="inner")
        if df.empty:
            continue
        ex = df.explode("claims").rename(columns={"claims": "text"})
        ex = ex[["text", "judgement"]].dropna()
        ex["text"] = ex["text"].astype(str).str.strip()
        ex = ex[ex["text"].str.len() > 0]
        rows.append(ex)

    if not rows:
        return pd.DataFrame(columns=["text", "judgement"])

    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset="text").reset_index(drop=True)
    log.info("[%s] %d labeled claim rows (pos=%d neg=%d)", agency, len(out), int(out.judgement.sum()), int((out.judgement == 0).sum()))
    return out


def split_and_write(agency: str, df: pd.DataFrame, seed: int = 42) -> None:
    if df.empty:
        log.warning("[%s] empty dataframe, skipping write", agency)
        return
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(0.7 * n)
    n_eval = int(0.15 * n)
    train_idx = idx[:n_train]
    eval_idx = idx[n_train:n_train + n_eval]
    test_idx = idx[n_train + n_eval:]

    agency_dir = OUT_ROOT / agency
    agency_dir.mkdir(parents=True, exist_ok=True)

    def _pos(a):
        return int(df.iloc[a]["judgement"].sum())

    df.iloc[train_idx].to_csv(agency_dir / "train.csv.gz", index=False, compression="gzip")
    df.iloc[eval_idx].to_csv(agency_dir / "eval.csv.gz", index=False, compression="gzip")
    df.iloc[test_idx].to_csv(agency_dir / "test.csv.gz", index=False, compression="gzip")

    log.info(
        "[%s] wrote train=%d(pos=%d) eval=%d(pos=%d) test=%d(pos=%d) -> %s",
        agency, len(train_idx), _pos(train_idx),
        len(eval_idx), _pos(eval_idx),
        len(test_idx), _pos(test_idx),
        agency_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agencies", nargs="+", default=TIER1 + TIER2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = []
    for ag in args.agencies:
        df = build_agency(ag)
        split_and_write(ag, df, seed=args.seed)
        summary.append({
            "agency": ag,
            "n_total": len(df),
            "n_pos": int(df["judgement"].sum()) if len(df) else 0,
            "n_neg": int((df["judgement"] == 0).sum()) if len(df) else 0,
        })

    summary_df = pd.DataFrame(summary)
    summary_path = OUT_ROOT / "build_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Summary written to %s:\n%s", summary_path, summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
