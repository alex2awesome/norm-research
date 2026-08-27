"""Aggregate Claude labels with the sample; compute distributions and correlations."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

SAMPLE = "/Users/spangher/Projects/stanford-research/norm-research/outputs/v2_analysis/crse_quality_sample_500.parquet"
LABELS_DIR = Path("/Users/spangher/Projects/stanford-research/norm-research/outputs/v2_analysis/crse_quality_labels")
OUT_LABELS = "/Users/spangher/Projects/stanford-research/norm-research/outputs/v2_analysis/crse_quality_labels.parquet"
OUT_REPORT = "/Users/spangher/Projects/stanford-research/norm-research/outputs/v2_analysis/crse_quality_report.txt"

# Ordinal mapping: higher = "more broken"
QUALITY_ORDINAL = {
    "CORRECT": 0,
    "PARTIAL": 1,
    "SLOW": 2,
    "WRONG_LOGIC": 3,
    "SYNTAX_ERR": 4,
    "NOT_CODE": -1,    # exclude
    "CANT_TELL": -1,   # exclude
}

BUCKET_ORDER = ["score_le_neg4", "score_neg1_neg3", "score_0_1", "pad_score_0_1", "score_2_9", "score_ge_10"]


def main():
    # Download sample parquet
    sample = pd.read_parquet(SAMPLE)
    print("sample:", sample.shape)

    rows = []
    for shard in sorted(LABELS_DIR.glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    labels = pd.DataFrame(rows)
    print("labels:", labels.shape)
    print("unique labels:", labels.code_quality.value_counts().to_dict())

    # Merge
    merged = sample.merge(labels, on="answer_id", how="left")
    print("merged:", merged.shape, "missing label:", merged.code_quality.isna().sum())

    # Save labeled parquet (drop big columns to keep it small)
    out_cols = ["answer_id", "question_id", "a_score", "bucket", "question_lang",
                "lang_detected", "Title", "tags_str", "code_len", "code_trunc",
                "prose_trunc", "code_quality", "evidence"]
    merged[out_cols].to_parquet(OUT_LABELS, index=False)
    print("Saved labels parquet ->", OUT_LABELS)

    lines = []
    def p(s=""):
        print(s)
        lines.append(str(s))

    p("=" * 80)
    p("CR.SE Code Quality Analysis (n=500)")
    p("=" * 80)
    p()

    # 1. Sample distribution
    p("## Sample distribution (by score bucket)")
    bucket_counts = merged.bucket.value_counts().reindex(BUCKET_ORDER).fillna(0).astype(int)
    for b, n in bucket_counts.items():
        p(f"  {b:20s}: n={n}")
    p()
    p("NOTE: only 24 answers exist in CR.SE corpus with score<=-4 (extremely rare); the")
    p("'pad_score_0_1' bucket of 76 was used to bring the n=100 target for that bucket up.")
    p("Effective bucket for correlation analysis collapses pad_score_0_1 into score_0_1.")
    p()

    # 2. Overall distribution
    p("## Overall failure-mode distribution (all 500)")
    overall = merged.code_quality.value_counts(dropna=False)
    total = len(merged)
    for lbl, n in overall.items():
        p(f"  {lbl:12s}: {n:4d} ({100*n/total:5.1f}%)")
    p()

    # 3. Per-bucket
    p("## Per-bucket quality distribution (row %)")
    # Collapse padding into score_0_1 for display
    merged["bucket_eff"] = merged.bucket.replace({"pad_score_0_1": "score_0_1"})
    pivot = pd.crosstab(merged.bucket_eff, merged.code_quality, normalize="index") * 100
    pivot = pivot.reindex(["score_le_neg4", "score_neg1_neg3", "score_0_1", "score_2_9", "score_ge_10"])
    # Order columns: CORRECT first
    col_order = ["CORRECT", "PARTIAL", "SLOW", "WRONG_LOGIC", "SYNTAX_ERR", "NOT_CODE", "CANT_TELL"]
    cols_present = [c for c in col_order if c in pivot.columns]
    pivot = pivot[cols_present].round(1)
    p(pivot.to_string())
    p()

    # Per-bucket raw counts
    p("## Per-bucket quality counts")
    pivot_n = pd.crosstab(merged.bucket_eff, merged.code_quality)
    pivot_n = pivot_n.reindex(["score_le_neg4", "score_neg1_neg3", "score_0_1", "score_2_9", "score_ge_10"])
    pivot_n = pivot_n[cols_present]
    p(pivot_n.to_string())
    p()

    # 4. Correlation
    p("## Voting vs Quality (Spearman correlation)")
    merged["q_ord"] = merged.code_quality.map(QUALITY_ORDINAL)
    rho_df = merged.dropna(subset=["q_ord"])
    rho_df = rho_df[rho_df.q_ord >= 0]  # drop CANT_TELL / NOT_CODE
    rho, pval = spearmanr(rho_df.a_score, rho_df.q_ord)
    p(f"  n={len(rho_df)} (after dropping NOT_CODE/CANT_TELL)")
    p(f"  Spearman rho(a_score, q_ord)  = {rho:+.3f}  (p={pval:.3e})")
    p(f"  q_ord: 0=CORRECT, 1=PARTIAL, 2=SLOW, 3=WRONG_LOGIC, 4=SYNTAX_ERR")
    p(f"  Negative rho => higher score correlates with cleaner code.")
    p()

    # Binary: bug-flagged vs not
    bug_labels = {"SYNTAX_ERR", "WRONG_LOGIC"}
    merged["has_bug"] = merged.code_quality.isin(bug_labels).astype(int)
    p("## Bug rate by bucket (SYNTAX_ERR or WRONG_LOGIC / total in bucket)")
    bug_rate = merged.groupby("bucket_eff").has_bug.agg(["sum", "count"])
    bug_rate["pct"] = (100 * bug_rate["sum"] / bug_rate["count"]).round(1)
    bug_rate = bug_rate.reindex(["score_le_neg4", "score_neg1_neg3", "score_0_1", "score_2_9", "score_ge_10"])
    p(bug_rate.to_string())
    p()

    # 5. Verbatim examples per failure mode (up to 3 per label)
    p("## Verbatim examples per failure mode")
    p()
    for lbl in ["SYNTAX_ERR", "WRONG_LOGIC", "SLOW", "PARTIAL", "CANT_TELL", "NOT_CODE"]:
        rows_l = merged[merged.code_quality == lbl].head(3)
        if len(rows_l) == 0:
            continue
        p(f"--- {lbl} ---")
        for _, r in rows_l.iterrows():
            p(f"  answer_id={r.answer_id}  score={r.a_score}  bucket={r.bucket}  lang={r.question_lang}")
            p(f"  title: {str(r.Title)[:120]}")
            p(f"  evidence: {r.evidence}")
            code_snippet = str(r.code_trunc)[:400].replace("\n", " ¶ ")
            p(f"  code[:400]: {code_snippet}")
            p()

    # Save text report
    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines))
    print()
    print("Report saved ->", OUT_REPORT)


if __name__ == "__main__":
    main()
