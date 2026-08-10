#!/usr/bin/env python3
"""Prepare (text, judgement) CSV for dense model training from PR diffs + verdicts.
Runs on sk3. Builds from batch_runs diffs + consolidated verdicts.

Usage:
  cd /lfs/skampere3/0/alexspan/norm-research/datasets/code-review/pr_test_execution
  python3 scripts/pr_vat/prepare_dense_csv.py
"""
import sys, os, glob, json
import pandas as pd

BATCH = "batch_runs"
VERDICTS_CSV = sys.argv[1] if len(sys.argv) > 1 else None  # optional: path to consolidated verdicts CSV

def main():
    # Load verdicts (merge-status + metadata)
    # Try the local consolidated CSV first, then build from batch_runs
    if VERDICTS_CSV and os.path.exists(VERDICTS_CSV):
        df = pd.read_csv(VERDICTS_CSV)
        print(f"Loaded {len(df)} verdict rows from {VERDICTS_CSV}")
    else:
        # Build from batch_runs verdicts.jsonl
        rows = []
        for f in sorted(glob.glob(f"{BATCH}/*/verdicts.jsonl")):
            repo = f.split("/")[1]
            for l in open(f):
                if not l.strip(): continue
                try: d = json.loads(l)
                except: continue
                rows.append({"repo": repo, "pr_number": d.get("pr_number") or d.get("paper_id"),
                             "verdict": d.get("verdict"), "judgement": d.get("judgement")})
        df = pd.DataFrame(rows)
        print(f"Built {len(df)} rows from batch_runs")

    # Filter to clean merge-status
    clean = df[df["judgement"].isin(["accepted", "rejected"])].copy()
    clean["y"] = (clean["judgement"] == "rejected").astype(int)  # 1=rejected
    print(f"Clean merge-status: {len(clean)} (accepted={len(clean[clean['y']==0])}, rejected={len(clean[clean['y']==1])})")

    # Join with diff text
    out_rows = []
    for _, row in clean.iterrows():
        repo = row["repo"]
        pr = str(row["pr_number"])
        if not pr or pr == "None" or pr == "nan": continue
        diff_path = f"{BATCH}/{repo}/diffs/pr_{pr}.diff"
        if not os.path.exists(diff_path): continue
        try:
            text = open(diff_path, errors="replace").read().strip()
        except: continue
        if not text or len(text) < 50: continue
        # truncate to ~6000 chars (~2048 tokens)
        if len(text) > 6000:
            text = text[:6000]
        out_rows.append({"text": text, "judgement": int(row["y"]), "repo": repo, "pr_number": pr})

    out_df = pd.DataFrame(out_rows)
    outpath = "outputs/pr_dense_training.csv"
    os.makedirs("outputs", exist_ok=True)
    out_df.to_csv(outpath, index=False)
    print(f"\nDense training CSV: {len(out_df)} rows → {outpath}")
    print(f"  rejected (y=1): {out_df['judgement'].sum()} ({out_df['judgement'].mean()*100:.1f}%)")
    print(f"  repos: {out_df['repo'].nunique()}")
    print(f"  avg text length: {out_df['text'].str.len().mean():.0f} chars")

if __name__ == "__main__":
    main()
