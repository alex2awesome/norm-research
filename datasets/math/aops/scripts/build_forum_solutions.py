"""
Build the forum-solutions table: solution-bearing replies in matched
contest-problem threads, keyed to wiki problems via forum_wiki_join.parquet.

Filter (per task spec): post_number > 1, has math markup, >= 200 chars of
non-quote content, not a bump/thanks/sticky post.

Output: forum_solutions.parquet (laptop copy; scp'd to sk3
/lfs/skampere3/0/alexspan/aops/forum_solutions.parquet by the caller).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
KEY = ["contest", "year", "variant", "problem_n"]

QUOTE_PAT = re.compile(r"\[quote[^\]]*\].*?\[/quote\]", re.S | re.I)
MATH_PAT = re.compile(r"\$|\\\[|\\\(|\\(?:frac|cdot|le|ge|angle|sqrt|sum|prod|"
                      r"binom|times|boxed|text|pi|alpha|beta|triangle|pmod)\b")
BUMP_PAT = re.compile(r"^\s*(?:bump|bumping|\.|\?|\^+|same|me too|thanks?|"
                      r"ty|wow|nice|cool|lol|orz|sniped)\W*$", re.I)


def strip_quotes(s: str) -> str:
    return QUOTE_PAT.sub(" ", s or "")


def main() -> int:
    posts = pd.read_json(REPO / "forum_priority_posts.jsonl.gz", lines=True)
    join = pd.read_parquet(REPO / "wiki" / "forum_wiki_join.parquet")

    matched = posts.merge(
        join[["topic_id"] + KEY + ["match_sim", "match_kind"]],
        on="topic_id", how="inner")
    print(f"posts in matched threads: {len(matched)} "
          f"({matched.topic_id.nunique()} topics, "
          f"{len(matched.drop_duplicates(KEY))} problems incl. dup-keys)")

    rep = matched[matched.post_number > 1].copy()
    rep["body_noquote"] = rep.post_canonical.map(strip_quotes)
    n_replies = len(rep)

    has_math = rep.body_noquote.str.contains(MATH_PAT, na=False)
    long_enough = rep.body_noquote.str.strip().str.len() >= 200
    not_bump = ~rep.post_canonical.fillna("").str.contains(BUMP_PAT)
    keep = has_math & long_enough & not_bump
    sol = rep[keep].copy()
    print(f"replies: {n_replies}  kept as solutions: {len(sol)} "
          f"(math={has_math.mean():.2f} len200={long_enough.mean():.2f})")

    sol = sol[KEY + [
        "topic_id", "post_number", "post_id", "username", "poster_id",
        "post_canonical", "body_noquote", "thanks_received",
        "nothanks_received", "post_time", "num_edits", "topic_num_views",
        "match_sim", "match_kind"]].reset_index(drop=True)
    out = REPO / "forum_solutions.parquet"
    sol.to_parquet(out, index=False)

    per_prob = sol.groupby(KEY).size()
    print(f"\nproblems with >=1 forum solution: {len(per_prob)}")
    print("solutions per problem:", per_prob.describe().round(2).to_dict())
    print("\nby contest:")
    summary = (sol.groupby("contest")
               .agg(n_solutions=("post_id", "count"),
                    n_problems=("variant", "nunique")))
    summary["n_problems"] = (sol[KEY].drop_duplicates()
                             .groupby("contest").size())
    print(summary.to_string())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
