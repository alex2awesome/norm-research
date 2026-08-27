"""Aggregate Claude shard responses → labels parquet, then correlate with the bank.

Outputs (laptop side):
  lc_multidim_claude_labels.parquet      — per-pair Claude 7-axis labels
  lc_multidim_correlations.parquet        — long-form (metric, dimension, pearson, spearman, n)

Also computes:
  - per-metric "best dimension" summary
  - top-15 RF-importance metrics x 7 dim summary
"""
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
WORK_DIR = ROOT / "outputs/v2_analysis/claude_multidim_work"
SAMPLE_PARQUET_LOCAL = ROOT / "outputs/v2_analysis/lc_multidim_sample_2000.parquet"

OUT_LABELS = ROOT / "outputs/v2_analysis/lc_multidim_claude_labels.parquet"
OUT_CORRS = ROOT / "outputs/v2_analysis/lc_multidim_correlations.parquet"
OUT_SUMMARY = ROOT / "outputs/v2_analysis/lc_multidim_summary.json"

DIMS = ["A_approach", "A_lexical", "A_structural", "A_naming", "A_comments", "A_length", "A_idiom"]


def strip_fences(s):
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*\n(.*)\n```\s*$", s, re.DOTALL)
    if m:
        return m.group(1)
    return s


def extract_json_array(s):
    s = strip_fences(s)
    start = s.find("[")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return None


def parse_shard(path: Path):
    raw = path.read_text() if path.exists() else ""
    if not raw.strip():
        return [], "empty_output"
    arr_text = extract_json_array(raw)
    if arr_text is None:
        items = []
        for line in raw.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
        return items, ("jsonl_fallback" if items else "no_json")
    try:
        return json.loads(arr_text), "ok"
    except Exception as e:
        # Greedy salvage: try to capture each {...} block
        items = []
        for m in re.finditer(r"\{[^{}]*\"pair_id\"[^{}]*\}", arr_text):
            try:
                items.append(json.loads(m.group(0)))
            except Exception:
                pass
        return items, f"salvage:{len(items)}:{e}"


def aggregate_labels():
    shard_files = sorted(WORK_DIR.glob("shard_*_response.txt"))
    print(f"shard files: {len(shard_files)}")

    rows = []
    parse_status = {}
    for sf in shard_files:
        items, status = parse_shard(sf)
        parse_status[sf.name] = {"n": len(items), "status": status}
        print(f"  {sf.name}: {len(items)} items ({status})")
        for it in items:
            if not isinstance(it, dict):
                continue
            pid = it.get("pair_id")
            try:
                pid = int(pid)
            except Exception:
                continue
            row = {"pair_id": pid, "note": it.get("note", "")}
            ok = True
            for d in DIMS:
                v = it.get(d)
                try:
                    iv = int(v)
                    if iv not in (0, 1, 2, 3):
                        ok = False
                        break
                    row[d] = iv
                except Exception:
                    ok = False
                    break
            if ok:
                rows.append(row)

    # Dedup by pair_id (last wins)
    by_pid = {r["pair_id"]: r for r in rows}
    labels_df = pd.DataFrame(list(by_pid.values())).sort_values("pair_id").reset_index(drop=True)
    labels_df.to_parquet(OUT_LABELS, index=False)
    print(f"wrote {OUT_LABELS}  ({len(labels_df)} valid pairs)")
    return labels_df, parse_status


def compute_correlations(labels_df, bank_df, sample_df):
    # Join sample.candidate_id → labels.pair_id, then attach bank by row_id == candidate_id
    merged = sample_df.merge(labels_df, on="pair_id", how="inner")
    merged = merged.merge(bank_df, left_on="candidate_id", right_on="row_id", how="inner")
    print(f"merged rows (with labels AND bank): {len(merged)}")

    score_cols = [c for c in bank_df.columns if c.endswith("_score") and c != "y"]
    print(f"score columns: {len(score_cols)}")

    out_rows = []
    for sc in score_cols:
        # treat -1 / sentinels as NaN
        vals = merged[sc].astype(float)
        vals = vals.where((vals >= 0) & (vals <= 1), np.nan)
        for d in DIMS:
            mask = vals.notna() & merged[d].notna()
            n = int(mask.sum())
            if n < 30:
                out_rows.append({"metric": sc, "dimension": d, "n": n,
                                  "pearson_r": np.nan, "pearson_p": np.nan,
                                  "spearman_r": np.nan, "spearman_p": np.nan})
                continue
            x = vals[mask].values
            y = merged[d][mask].astype(float).values
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                out_rows.append({"metric": sc, "dimension": d, "n": n,
                                  "pearson_r": np.nan, "pearson_p": np.nan,
                                  "spearman_r": np.nan, "spearman_p": np.nan})
                continue
            pr, pp_ = pearsonr(x, y)
            sr, sp = spearmanr(x, y)
            out_rows.append({"metric": sc, "dimension": d, "n": n,
                              "pearson_r": float(pr), "pearson_p": float(pp_),
                              "spearman_r": float(sr), "spearman_p": float(sp)})

    corr_df = pd.DataFrame(out_rows)
    corr_df.to_parquet(OUT_CORRS, index=False)
    print(f"wrote {OUT_CORRS}  ({len(corr_df)} rows)")
    return corr_df, merged


def rf_importances(merged, bank_df, label_col="max_sim"):
    """Compute RF importances against max_sim target — used purely to pick top 15 metrics."""
    from sklearn.ensemble import RandomForestRegressor
    score_cols = [c for c in bank_df.columns if c.endswith("_score") and c != "y"]
    X = merged[score_cols].astype(float).fillna(0.5).values
    y = merged[label_col].astype(float).values
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, n_jobs=-1, random_state=0)
    rf.fit(X, y)
    imps = pd.DataFrame({"metric": score_cols, "importance": rf.feature_importances_})
    return imps.sort_values("importance", ascending=False).reset_index(drop=True)


def summarize(corr_df, merged, top_n_metrics=15):
    summary = {}

    # Label distributions per dimension
    summary["label_distribution_per_dim"] = {}
    for d in DIMS:
        if d in merged.columns:
            counts = merged[d].dropna().astype(int).value_counts().sort_index().to_dict()
            summary["label_distribution_per_dim"][d] = {str(k): int(v) for k, v in counts.items()}

    # Average correlation per dimension (across all metrics)
    summary["avg_abs_pearson_per_dim"] = {}
    summary["avg_pearson_per_dim"] = {}
    for d in DIMS:
        sub = corr_df[corr_df["dimension"] == d]["pearson_r"].dropna()
        summary["avg_abs_pearson_per_dim"][d] = float(np.abs(sub).mean()) if len(sub) else None
        summary["avg_pearson_per_dim"][d] = float(sub.mean()) if len(sub) else None

    # Headline heatmap: top 25 metrics by best |r| across any dim, with row per metric
    metric_best = corr_df.copy()
    metric_best["abs_r"] = metric_best["pearson_r"].abs()
    best_per_metric = metric_best.sort_values("abs_r", ascending=False).groupby("metric").first().reset_index()
    summary["top25_metrics_by_best_abs_pearson"] = best_per_metric.sort_values("abs_r", ascending=False).head(25)[
        ["metric", "dimension", "pearson_r", "spearman_r", "n"]
    ].to_dict(orient="records")

    # Best dimension per metric (overall vote)
    summary["best_dim_count_across_metrics"] = best_per_metric["dimension"].value_counts().to_dict()

    # RF top metrics x dimension matrix
    imps = rf_importances(merged, merged)  # second arg is just for column list
    top_metrics = imps.head(top_n_metrics)["metric"].tolist()
    summary["rf_top15_metrics"] = imps.head(top_n_metrics).to_dict(orient="records")
    rf_breakdown = []
    for m in top_metrics:
        row = {"metric": m}
        for d in DIMS:
            sub = corr_df[(corr_df["metric"] == m) & (corr_df["dimension"] == d)]
            if len(sub):
                row[d] = float(sub.iloc[0]["pearson_r"]) if pd.notna(sub.iloc[0]["pearson_r"]) else None
            else:
                row[d] = None
        rf_breakdown.append(row)
    summary["rf_top15_metric_x_dim"] = rf_breakdown

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-parquet", default=None,
                    help="If given, use this local bank file (else expect rsync of remote bank)")
    args = ap.parse_args()

    labels_df, parse_status = aggregate_labels()

    # We need the bank scores + the sample table to join. Both live on sk3 by default.
    bank_path = args.bank_parquet
    if bank_path is None:
        bank_path = ROOT / "outputs/v2_analysis/leetcode_cpp_metric_scores_fixed.parquet"
    bank_df = pd.read_parquet(bank_path)
    sample_df = pd.read_parquet(SAMPLE_PARQUET_LOCAL)
    print(f"sample rows: {len(sample_df)}, bank rows: {len(bank_df)}")

    corr_df, merged = compute_correlations(labels_df, bank_df, sample_df)

    summary = summarize(corr_df, merged)
    summary["n_labels_parsed"] = int(len(labels_df))
    summary["n_merged"] = int(len(merged))
    summary["parse_status_by_shard"] = parse_status
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {OUT_SUMMARY}")

    print("\n=== AVERAGE |pearson| PER DIMENSION (across all metrics) ===")
    for d, v in summary["avg_abs_pearson_per_dim"].items():
        print(f"  {d:14s}  mean|r| = {v:.4f}" if v is not None else f"  {d:14s}  n/a")

    print("\n=== BEST-DIM VOTE COUNT ACROSS METRICS ===")
    for d, c in sorted(summary["best_dim_count_across_metrics"].items(), key=lambda kv: -kv[1]):
        print(f"  {d:14s}  {c} metrics")

    print("\n=== TOP-15 RF METRICS x DIMENSION (pearson r) ===")
    rfb = pd.DataFrame(summary["rf_top15_metric_x_dim"]).set_index("metric")
    print(rfb.round(3).to_string())


if __name__ == "__main__":
    main()
