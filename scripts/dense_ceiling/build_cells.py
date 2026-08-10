#!/usr/bin/env python3
"""Assemble per-cell tables for dense-ceiling experiments.

Produces one parquet per platform with columns:
  pair_id, platform, canonical_pid, candidate_code, label, source

Cells:
  CC  -- comp_unified_claude_bank_scores_v2.parquet (source=='cc') joined to
         comp_unified_editorial_labels.parquet on pair_id
  CC_POOLED -- the same scores file with source=='cc' or 'luogu' kept SEPARATE
               (Luogu rows tagged platform='luogu'); user wants Luogu separate
  CF  -- comp_fourplatform_label_shards/shards/shard_cf_*.jsonl candidate_text
         + results/shard_cf_*.jsonl claude_label (status=='ok')
  AC  -- shard_ac_*.jsonl + shard_ac2_*.jsonl (original labels)
  AC_L1 -- ac_l1_relabel/results/shard_*_results.jsonl pair_id->l1 join to
           the AC candidate_text pool. Written only if >=80% of AC pair_ids
           have L1 labels.
  LC  -- lc_python_2k_sample.parquet + lc_python_2k_claude_labels.parquet
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
V2 = REPO / "outputs/v2_analysis"
OUT = V2 / "dense_ceiling"
OUT.mkdir(parents=True, exist_ok=True)


def write_cell(name: str, df: pd.DataFrame):
    df = df.dropna(subset=["candidate_code", "label"]).copy()
    df = df[df.candidate_code.astype(str).str.len() >= 30].copy()
    df = df.drop_duplicates("pair_id")
    df["canonical_pid"] = df["canonical_pid"].fillna("pid_" + df["pair_id"].astype(str))
    df["label"] = df["label"].astype(int)
    path = OUT / f"cell_{name}.parquet"
    df.to_parquet(path)
    pos = float(df.label.mean())
    print(f"{name:12s}  n={len(df):5d}  pos={pos:.3f}  groups={df.canonical_pid.nunique():5d}  -> {path.name}")
    return df


def build_cc():
    """Assemble CC/Luogu rows from three candidate-text sources:
      1. comp_unified_editorial_labels.parquet (covers ~30%)
      2. comp_unified_claude5k_shards/shards/*.jsonl (covers most clauded5k_*)
      3. comp_unified_claude3900_shards/_input_3900.parquet (covers ~150)
    """
    scores = pd.read_parquet(V2 / "comp_unified_claude_bank_scores_v2.parquet",
                             columns=["pair_id", "claude_label", "source"])

    parts = []
    el = pd.read_parquet(V2 / "comp_unified_editorial_labels.parquet")
    el_cols = ["pair_id", "candidate_id", "editorial_id", "candidate_code"]
    parts.append(el[el_cols].rename(columns={"editorial_id": "canonical_pid"}))

    rows = []
    for f in sorted((V2 / "comp_unified_claude5k_shards/shards").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"pair_id": r["pair_id"],
                         "candidate_id": r.get("candidate_id"),
                         "canonical_pid": r.get("problem_id"),
                         "candidate_code": r.get("candidate_text")})
    parts.append(pd.DataFrame(rows))

    df39 = pd.read_parquet(V2 / "comp_unified_claude3900_shards/_input_3900.parquet")
    parts.append(df39[["pair_id", "candidate_id", "editorial_id", "candidate_code"]]
                 .rename(columns={"editorial_id": "canonical_pid"}))

    pool = pd.concat(parts, ignore_index=True).dropna(subset=["candidate_code"])
    pool = pool.drop_duplicates("pair_id")

    df = scores.merge(pool, on="pair_id", how="left")
    df = df.rename(columns={"claude_label": "label"})
    df["source"] = "claude_1k_" + df["source"]
    return df


def build_fourplatform(plat: str, shard_prefixes):
    shard_dir = V2 / "comp_fourplatform_label_shards"
    inputs = {}
    for prefix in shard_prefixes:
        for f in sorted((shard_dir / "shards").glob(f"{prefix}*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                inputs[r["pair_id"]] = {
                    "pair_id": r["pair_id"],
                    "platform": plat,
                    "canonical_pid": r.get("problem_id"),
                    "candidate_code": r.get("candidate_text"),
                }
    labels = {}
    for prefix in shard_prefixes:
        for f in sorted((shard_dir / "results").glob(f"{prefix}*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") == "ok" and r.get("claude_label") in (0, 1):
                    labels[r["pair_id"]] = int(r["claude_label"])
    rows = []
    for pid, lab in labels.items():
        if pid not in inputs:
            continue
        rec = dict(inputs[pid])
        rec["label"] = lab
        rec["source"] = "claude_fourplatform"
        rows.append(rec)
    return pd.DataFrame(rows)


def build_ac_l1():
    """Return AC rows with l1 label only if coverage >=80% of AC pool."""
    ac = build_fourplatform("ac", ["shard_ac_", "shard_ac2_"])
    if len(ac) == 0:
        return None, 0.0
    l1_dir = V2 / "ac_l1_relabel/results"
    l1 = {}
    for f in sorted(l1_dir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("l1") in (0, 1):
                l1[r["pair_id"]] = int(r["l1"])
    cov = sum(1 for p in ac.pair_id if p in l1) / max(len(ac), 1)
    df = ac.copy()
    df["label_l1"] = df.pair_id.map(l1)
    return df, cov


def build_lc():
    labels = pd.read_parquet(V2 / "lc_python_2k_claude_labels.parquet",
                             columns=["pair_id", "label"])
    sample = pd.read_parquet(V2 / "lc_python_2k_sample.parquet",
                             columns=["pair_id", "question_slug", "code"])
    df = labels.merge(sample, on="pair_id", how="inner")
    df = df.rename(columns={"code": "candidate_code",
                            "question_slug": "canonical_pid"})
    df["platform"] = "lc"
    df["source"] = "claude_lc2k"
    return df


def main():
    print("=== Building cells ===")

    cc_full = build_cc()
    cc_only = cc_full[cc_full.source.str.contains("_cc")].copy()
    cc_only["platform"] = "cc"
    write_cell("cc_v2_cc_only", cc_only)

    cc_luogu = cc_full[cc_full.source.str.contains("_luogu")].copy()
    cc_luogu["platform"] = "luogu"
    write_cell("cc_v2_luogu", cc_luogu)

    # CC pooled (CC + Luogu): user wants both, with Luogu kept separate above.
    # The pooled cell uses platform as group-stratification field; we still split by canonical_pid.
    cc_pooled = cc_full.copy()
    cc_pooled["platform"] = cc_full.source.str.replace("claude_1k_", "")
    write_cell("cc_v2_pooled", cc_pooled)

    cf = build_fourplatform("cf", ["shard_cf_"])
    write_cell("cf", cf)

    ac_l1, cov = build_ac_l1()
    print(f"AC L1 coverage: {cov:.1%} of AC pool")
    if ac_l1 is not None:
        ac_orig = ac_l1.drop(columns=["label_l1"]).copy()
        write_cell("ac", ac_orig)
        if cov >= 0.80:
            ac_l1_clean = ac_l1.dropna(subset=["label_l1"]).copy()
            ac_l1_clean["label"] = ac_l1_clean.label_l1.astype(int)
            ac_l1_clean = ac_l1_clean.drop(columns=["label_l1"])
            write_cell("ac_l1", ac_l1_clean)
        else:
            print(f"  AC L1 below 80% threshold — skipping ac_l1 cell. "
                  "Rerun this script after relabel completes; ac_l1 cell will appear.")
    lc = build_lc()
    write_cell("lc", lc)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
