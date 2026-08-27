"""Analyze bank vs Claude agreement on approach-matched LeetCode triples."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/Users/spangher/Projects/stanford-research/norm-research")
INPUT = REPO / "outputs/v2_analysis/lc_approach_matched_triples.parquet"


def main():
    df = pd.read_parquet(INPUT)
    print("triples:", len(df))
    print()
    print("=== Pre-screen ===")
    sa = df.same_approach.fillna("missing").value_counts(dropna=False)
    print(sa.to_dict())

    # Restrict to confirmed same-approach
    used = df[df.same_approach == "yes"].copy()
    print(f"\nconfirmed same-approach triples: {len(used)} of {len(df)}")
    print("style_winner dist:", used.style_winner.fillna("?").value_counts().to_dict())

    # Drop ties for primary agreement (both bank and Claude)
    used = used[used.style_winner.isin(["A", "B"])].copy()
    used["bank_pred_resolved"] = used.bank_pred.where(used.bank_pred.isin(["A", "B"]))
    used = used[used.bank_pred_resolved.notna()].copy()
    print(f"after dropping ties: {len(used)}")

    used["agree"] = (used.bank_pred_resolved == used.style_winner).astype(int)
    overall = used.agree.mean() if len(used) else float("nan")
    print(f"\n=== HEADLINE ===")
    print(f"Bank vs Claude overall agreement: {overall*100:.1f}% (n={len(used)})")

    # Binomial 95% CI
    n = len(used)
    if n > 0:
        p = used.agree.mean()
        se = np.sqrt(p * (1 - p) / n)
        print(f"95% CI: [{(p-1.96*se)*100:.1f}%, {(p+1.96*se)*100:.1f}%]")

    print("\n=== Per language ===")
    used["lang_pair"] = used.apply(
        lambda r: f"{r.lang_a}/{r.lang_b}" if r.lang_a != r.lang_b else r.lang_a, axis=1
    )
    per_lang = used.groupby("lang_pair").agree.agg(["count", "mean"]).sort_values("count", ascending=False)
    per_lang["mean"] = per_lang["mean"] * 100
    print(per_lang.head(10).round(1).to_string())

    print("\n=== Per difficulty ===")
    per_diff = used.groupby("difficulty").agree.agg(["count", "mean"])
    per_diff["mean"] *= 100
    print(per_diff.round(1).to_string())

    print("\n=== Per bank-gap quartile ===")
    used["gap"] = (used.bank_cos_a - used.bank_cos_b).abs()
    used["gap_q"] = pd.qcut(used.gap, q=4, duplicates="drop", labels=False)
    per_gap = used.groupby("gap_q").agree.agg(["count", "mean"])
    per_gap["mean"] *= 100
    print(per_gap.round(1).to_string())

    print("\n=== Per reason length (Claude confidence proxy) ===")
    used["rlen"] = used.reason.fillna("").str.len()
    used["rlen_q"] = pd.qcut(used.rlen, q=3, duplicates="drop", labels=["short", "med", "long"])
    per_rlen = used.groupby("rlen_q", observed=True).agree.agg(["count", "mean"])
    per_rlen["mean"] *= 100
    print(per_rlen.round(1).to_string())

    print("\n=== Disagreement examples (5) ===")
    disagree = used[used.agree == 0].copy()
    # Prefer largest-gap disagreements (clear bank pick, opposite Claude pick)
    disagree = disagree.sort_values("gap", ascending=False)
    for _, r in disagree.head(5).iterrows():
        print("---")
        print(f"triple_id={r.triple_id}  Q={r.question_slug}  lang_a={r.lang_a} lang_b={r.lang_b}")
        print(f"bank_cos_a={r.bank_cos_a:.4f} bank_cos_b={r.bank_cos_b:.4f}  bank_pred={r.bank_pred}")
        print(f"Claude winner: {r.style_winner}")
        print(f"Claude reason: {r.reason}")
        print("EDITORIAL (first 600 chars):")
        print(r.editorial_code[:600])
        print("CAND A (first 400 chars):")
        print(r.code_a[:400])
        print("CAND B (first 400 chars):")
        print(r.code_b[:400])

    # Save summary
    summary = {
        "n_total": int(len(df)),
        "n_same_approach": int(len(df[df.same_approach == "yes"])),
        "n_analyzed": int(n),
        "overall_agreement": float(overall) if n else None,
        "ci_lo": float(p - 1.96 * se) if n else None,
        "ci_hi": float(p + 1.96 * se) if n else None,
        "per_lang": per_lang.reset_index().to_dict("records"),
        "per_difficulty": per_diff.reset_index().to_dict("records"),
        "per_gap_q": per_gap.reset_index().to_dict("records"),
    }
    (REPO / "outputs/v2_analysis/lc_approach_matched_triples.summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print("\nWrote summary.")


if __name__ == "__main__":
    main()
