"""
Build JSONL inputs for the dataset/benchmark judge run.

Modes:
  --test           Hand-picked 30-paper smoke test (mix of obvious YES + NO).
  --full           All ML-venue papers (ICLR, NEURIPS, ICML, TMLR, COLM)
                   with decision_unified in {accept, reject}.
  --venues_only    Like --full but skip no_decision; this is the default.

Outputs:
  outputs/dataset_judge/inputs/test_papers.jsonl     (~30 rows)
  outputs/dataset_judge/inputs/full_papers.jsonl     (~61K rows)

Each row: {"paper_id", "venue", "year", "decision_unified", "title", "abstract"}.

Run locally:
  python scripts/build_dataset_judge_input.py --test
  python scripts/build_dataset_judge_input.py --full
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
SRC  = ROOT / "datasets/peer-review/unified_papers.csv.gz"
OUT  = ROOT / "outputs/dataset_judge/inputs"
OUT.mkdir(parents=True, exist_ok=True)

ML_VENUE_PREFIXES = ("ICLR", "NEURIPS", "TMLR", "ICML", "COLM")

# Hand-picked smoke test: titles known to be {dataset, benchmark, both, neither}
# Picked to give the judge a fair challenge.
TEST_TITLE_KEYWORDS = {
    # likely DATASET
    "dataset": "DATASET",
    "corpus": "DATASET",
    "we release": "DATASET",
    "we present a new": "DATASET",
    # likely BENCHMARK
    "benchmark": "BENCHMARK",
    "evaluation suite": "BENCHMARK",
    "leaderboard": "BENCHMARK",
    "challenge": "BENCHMARK",
    # likely NEITHER (method-shaped phrases)
    "we propose a novel": "NEITHER",
    "we propose a new method": "NEITHER",
    "we introduce a novel architecture": "NEITHER",
    "convergence": "NEITHER",
    "theoretical analysis": "NEITHER",
}


def load_ml():
    df = pd.read_csv(SRC)
    df = df[df["venue"].str.startswith(ML_VENUE_PREFIXES, na=False)].copy()
    df = df[df["title"].notna() & df["abstract"].notna()]
    return df


def write_jsonl(df: pd.DataFrame, out_path: Path):
    cols = ["paper_id", "venue", "year", "decision_unified", "title", "abstract"]
    with open(out_path, "w") as f:
        for _, r in df[cols].iterrows():
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    print(f"wrote {len(df):,} rows -> {out_path}")


def build_test(df: pd.DataFrame, n_per_bucket: int = 5):
    """Pick a small balanced test set across our 4 expected labels.
    We don't have ground-truth labels yet — we use lexical priors over titles
    just to ensure we sample a mix of paper types for the judge to see.
    """
    rng = 11
    picks = []
    title_low = df["title"].str.lower()
    seen = set()
    # First: balanced buckets from title heuristics
    for kw, expected in TEST_TITLE_KEYWORDS.items():
        sub = df[title_low.str.contains(kw, na=False, regex=False)]
        sub = sub[~sub["paper_id"].isin(seen)]
        if len(sub) == 0:
            continue
        s = sub.sample(min(n_per_bucket, len(sub)), random_state=rng)
        for _, r in s.iterrows():
            picks.append({**r.to_dict(), "expected_hint": expected, "match_kw": kw})
            seen.add(r["paper_id"])
        rng += 1
    out = pd.DataFrame(picks)
    print(f"hand-picked test set: {len(out)} papers across {out['expected_hint'].nunique()} expected-buckets")
    print(out["expected_hint"].value_counts())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="Build smoke test set (~30 papers)")
    ap.add_argument("--full", action="store_true", help="Build full ML-venue accept+reject set")
    args = ap.parse_args()

    if not (args.test or args.full):
        ap.error("specify --test and/or --full")

    df = load_ml()
    print(f"loaded {len(df):,} ML-venue papers (ICLR/NEURIPS/ICML/TMLR/COLM, "
          f"with non-null title+abstract)")

    if args.test:
        test_df = build_test(df, n_per_bucket=5)
        # write extended jsonl with the expected_hint field for grading
        out = OUT / "test_papers.jsonl"
        with open(out, "w") as f:
            for _, r in test_df.iterrows():
                f.write(json.dumps({
                    "paper_id": r["paper_id"],
                    "venue": r["venue"],
                    "year": int(r["year"]) if pd.notna(r["year"]) else None,
                    "decision_unified": r["decision_unified"],
                    "title": r["title"],
                    "abstract": r["abstract"],
                    "expected_hint": r.get("expected_hint"),
                    "match_kw": r.get("match_kw"),
                }, ensure_ascii=False) + "\n")
        print(f"wrote {len(test_df)} test rows -> {out}")

    if args.full:
        full = df[df["decision_unified"].isin(["accept", "reject"])].copy()
        out = OUT / "full_papers.jsonl"
        write_jsonl(full, out)


if __name__ == "__main__":
    main()
