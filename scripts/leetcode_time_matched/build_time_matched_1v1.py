"""
Build a maximally restrictive time-matched 1v1 label for LeetCode.

For each problem:
  - winner = single most-upvoted solution (globally, full corpus)
  - loser  = closest-in-time same-language solution with strictly fewer upvotes
  - hard window: |t_winner - t_loser| <= window_days
  - upvote gap: winner_up >= 5 AND winner_up >= 2 * loser_up

Three label variants: window in {1, 7, 30} days.

Outputs:
  time_matched_1v1.parquet      (2 rows per pair, full feature columns)
  time_matched_1v1_label.parquet (solution_id, label, pair_id, window_days, ...)

Inputs (sk3, absolute):
  /lfs/skampere3/0/alexspan/norm-research/datasets/leetcode_codecontests/leetcode_solutions.parquet
  /lfs/skampere3/0/alexspan/norm-research/datasets/leetcode_balanced/_with_ts_tmp.parquet
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path

OUTDIR = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/leetcode_balanced")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- language detector ----------
def detect_lang(code):
    if not isinstance(code, str) or len(code) < 5:
        return "unknown"
    c = code
    if re.search(r"^\s*class\s+Solution\s*:", c, re.M) or re.search(r"def\s+\w+\s*\(self", c):
        return "python"
    if re.search(r"^\s*from\s+typing\s+import", c, re.M):
        return "python"
    if re.search(r"^\s*import\s+(collections|heapq|functools|bisect|itertools)", c, re.M):
        return "python"
    if re.search(r"#include", c) or re.search(r"using\s+namespace\s+std", c) or re.search(r"std::", c):
        return "cpp"
    if re.search(r"\bvector\s*<", c) or re.search(r"\bunordered_(map|set)\s*<", c):
        return "cpp"
    if re.search(r"->next", c) or re.search(r"::", c) or re.search(r"\bnullptr\b", c) or re.search(r"\bcout\b", c):
        return "cpp"
    if re.search(r"class\s+Solution\s*\{[^}]*?public\s*:", c, re.S):
        return "cpp"
    if re.search(r"public\s+class\s+Solution", c):
        return "java"
    if re.search(r"\bArrayList\s*<", c) or re.search(r"\bHashMap\s*<", c) or re.search(r"\bSystem\.out\b", c):
        return "java"
    if re.search(r"\bpublic\s+(static\s+)?(int|long|double|String|boolean|void|List|int\[\])\s+\w+", c):
        return "java"
    if re.search(r"class\s+Solution\s*\{", c) and re.search(r"\bpublic\b", c):
        return "java"
    if re.search(r"var\s+\w+\s*=\s*function", c) or re.search(r"=>\s*\{", c):
        return "javascript"
    if re.search(r"const\s+\w+\s*=\s*function", c) or re.search(r"console\.log", c):
        return "javascript"
    if re.search(r"impl\s+Solution", c) or re.search(r"let\s+mut\s+", c):
        return "rust"
    if re.search(r"fn\s+\w+.*->\s*\w", c):
        return "rust"
    if re.search(r"func\s+\w+\s*\(.*\)\s*\w*\s*\{", c):
        return "go"
    if re.search(r"IList<", c) or re.search(r"using\s+System", c):
        return "csharp"
    if re.search(r"func\s+\w+\s*\(.*\)\s*->", c):
        return "swift"
    if re.search(r"^\s*def\s+\w+\s*\(", c, re.M) and not re.search(r"\{", c):
        return "python"
    return "unknown"


def main():
    print("Loading full solutions...")
    full = pd.read_parquet(
        "/lfs/skampere3/0/alexspan/norm-research/datasets/leetcode_codecontests/"
        "leetcode_solutions.parquet"
    )
    full["ts"] = pd.to_datetime(full["created_at"], errors="coerce", utc=True)
    full = full.dropna(subset=["ts"]).reset_index(drop=True)
    print(f"Full corpus rows: {len(full)}")

    print("Loading balanced (for lang_detected)...")
    bal = pd.read_parquet(
        "/lfs/skampere3/0/alexspan/norm-research/datasets/leetcode_balanced/_with_ts_tmp.parquet"
    )
    bal = bal.reset_index(drop=True)
    bal["balanced_row_id"] = bal.index.astype(int)

    # join key for language transfer
    bal["join_key"] = bal["question_slug"].astype(str) + "|" + bal["code"].astype(str)
    full["join_key"] = full["question_slug"].astype(str) + "|" + full["code"].astype(str)

    print("Transferring balanced lang_detected to full corpus where possible...")
    lang_map = (
        bal.drop_duplicates("join_key")
        .set_index("join_key")[["lang_detected", "balanced_row_id"]]
    )
    full = full.merge(lang_map, left_on="join_key", right_index=True, how="left")

    # detect missing
    mask = full["lang_detected"].isna() | (full["lang_detected"] == "unknown")
    print(f"Need to detect language for {mask.sum()} rows...")
    detected = full.loc[mask, "code"].apply(detect_lang)
    full.loc[mask, "lang_detected"] = detected.values
    print("Final language counts (top 8):")
    print(full["lang_detected"].value_counts().head(8))

    # Restrict to top 4 langs
    LANGS = ["cpp", "python", "java", "javascript"]
    df = full[full["lang_detected"].isin(LANGS)].copy()
    df = df.rename(columns={"upvotes": "n_upvotes"})
    print(f"After language filter: {len(df)} rows")

    # Sort: per problem, highest upvotes first
    df = df.sort_values(
        ["question_slug", "n_upvotes", "ts"], ascending=[True, False, True]
    ).reset_index(drop=True)
    df["solution_id"] = df.index.astype(int)

    # Build pairs for each window
    windows = [1, 7, 30]
    all_pair_rows = []
    all_label_rows = []
    pair_counter = 0

    # group by problem
    print("Building pairs per problem...")
    grp = df.groupby("question_slug", sort=False)
    n_groups = len(grp)
    for gi, (slug, g) in enumerate(grp):
        if gi % 500 == 0:
            print(f"  problem {gi}/{n_groups}")
        # winner: top upvotes overall in this problem
        winner = g.iloc[0]
        wu = int(winner["n_upvotes"])
        if wu < 5:
            continue
        wt = winner["ts"]
        wlang = winner["lang_detected"]

        # losers: same problem, fewer upvotes, same language
        # require wu >= 2 * loser_up
        cands = g.iloc[1:]
        cands = cands[cands["lang_detected"] == wlang]
        if len(cands) == 0:
            continue
        cands = cands[cands["n_upvotes"] < wu]
        cands = cands[wu >= 2 * cands["n_upvotes"]]
        if len(cands) == 0:
            continue
        # compute |t - wt|
        delta = (cands["ts"] - wt).abs().dt.total_seconds() / 86400.0
        cands = cands.assign(days_apart=delta.values)

        for w in windows:
            in_win = cands[cands["days_apart"] <= w]
            if len(in_win) == 0:
                continue
            # closest in time
            loser = in_win.sort_values("days_apart").iloc[0]
            pair_id = pair_counter
            pair_counter += 1
            base = dict(
                pair_id=pair_id,
                window_days=w,
                problem_slug=slug,
                language=wlang,
                days_apart=float(loser["days_apart"]),
                winner_upvotes=wu,
                loser_upvotes=int(loser["n_upvotes"]),
                winner_solution_id=int(winner["solution_id"]),
                loser_solution_id=int(loser["solution_id"]),
            )
            row_w = dict(
                base,
                role="winner",
                label=1,
                solution_id=int(winner["solution_id"]),
                code=winner["code"],
                upvotes=wu,
                ts=wt,
                balanced_row_id=winner.get("balanced_row_id", np.nan),
            )
            row_l = dict(
                base,
                role="loser",
                label=0,
                solution_id=int(loser["solution_id"]),
                code=loser["code"],
                upvotes=int(loser["n_upvotes"]),
                ts=loser["ts"],
                balanced_row_id=loser.get("balanced_row_id", np.nan),
            )
            all_pair_rows.append(row_w)
            all_pair_rows.append(row_l)
            all_label_rows.append(
                dict(
                    pair_id=pair_id,
                    window_days=w,
                    solution_id=int(winner["solution_id"]),
                    label=1,
                    language=wlang,
                    problem_slug=slug,
                    balanced_row_id=winner.get("balanced_row_id", np.nan),
                )
            )
            all_label_rows.append(
                dict(
                    pair_id=pair_id,
                    window_days=w,
                    solution_id=int(loser["solution_id"]),
                    label=0,
                    language=wlang,
                    problem_slug=slug,
                    balanced_row_id=loser.get("balanced_row_id", np.nan),
                )
            )

    pairs_df = pd.DataFrame(all_pair_rows)
    label_df = pd.DataFrame(all_label_rows)
    print(f"\nTotal pair rows: {len(pairs_df)} (={len(pairs_df)//2} pairs across all windows)")
    print(f"Per-window pair counts:")
    print(label_df.groupby("window_days")["pair_id"].nunique())

    pairs_out = OUTDIR / "time_matched_1v1.parquet"
    label_out = OUTDIR / "time_matched_1v1_label.parquet"
    pairs_df.to_parquet(pairs_out, index=False)
    label_df.to_parquet(label_out, index=False)
    print(f"\nWrote {pairs_out}")
    print(f"Wrote {label_out}")


if __name__ == "__main__":
    main()
