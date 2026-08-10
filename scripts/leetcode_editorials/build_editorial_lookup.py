"""Build a per-(slug, language) canonical-code lookup parquet.

Input:  datasets/leetcode_editorials/editorials.parquet
Output: datasets/leetcode_editorials/editorial_by_slug.parquet

For each (question_slug, normalized language) we keep the canonical_code
of the highest-upvoted editorial post (ties broken by row order).
Columns: question_slug, language, canonical_code, n_lines, n_chars.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
SRC = REPO / "datasets/leetcode_editorials/editorials.parquet"
DST = REPO / "datasets/leetcode_editorials/editorial_by_slug.parquet"

# Map noisy code_lang values onto the normalised language tags used in
# balanced_v2 (python, java, cpp, javascript, typescript, go, rust, csharp,
# swift, kotlin, ruby, c, sql).
LANG_NORM = {
    "python": "python", "py": "python", "python3": "python",
    "java": "java",
    "cpp": "cpp", "c++": "cpp",
    "c": "c",
    "javascript": "javascript", "js": "javascript", "jsx": "javascript",
    "typescript": "typescript", "ts": "typescript", "tsx": "typescript",
    "go": "go",
    "rust": "rust", "rs": "rust",
    "ruby": "ruby", "rb": "ruby",
    "csharp": "csharp", "cs": "csharp",
    "swift": "swift",
    "kotlin": "kotlin", "kt": "kotlin",
    "sql": "sql", "mysql": "sql",
}


def detect_lang_from_code(code: str) -> str:
    """Cheap structural hints for editorial rows tagged as `unknown`."""
    if not code:
        return "unknown"
    head = code.lstrip()[:400]
    # Python is the dominant editorial language and easy to spot.
    if ("def " in code and ":\n" in code) or "class Solution:" in code:
        return "python"
    if "public class" in code or "public int" in code or "public static" in code:
        return "java"
    if "#include" in code or "vector<" in code or "using namespace std" in code:
        return "cpp"
    if "func " in head and "{" in head:
        return "go"
    if "fn " in head and "->" in head:
        return "rust"
    if "function " in head or "const " in head and "=>" in head:
        return "javascript"
    return "unknown"


def main() -> None:
    print(f"reading {SRC}")
    e = pd.read_parquet(SRC)
    print(f"  rows={len(e)}, slugs={e['question_slug'].nunique()}")

    e = e.copy()
    e["canonical_code"] = e["canonical_code"].fillna("")
    e = e[e["canonical_code"].str.len() > 0].copy()
    print(f"  after non-empty canonical_code: rows={len(e)}, "
          f"slugs={e['question_slug'].nunique()}")

    # Normalise language; fall back to structural detection for unknown.
    code_lang = e["code_lang"].fillna("unknown").str.lower()
    norm = code_lang.map(LANG_NORM)
    # Where we still don't know, sniff the code.
    mask_unknown = norm.isna()
    norm.loc[mask_unknown] = e.loc[mask_unknown, "canonical_code"].apply(
        detect_lang_from_code)
    norm = norm.fillna("unknown")
    e["language"] = norm
    print("  language histogram:",
          e["language"].value_counts().to_dict())

    # Drop entries whose language is still unknown — we cannot match.
    e = e[e["language"] != "unknown"].copy()
    print(f"  after dropping unknown: rows={len(e)}")

    # Highest-upvoted per (slug, language). Fill NaN upvotes with -1 so
    # they sort last.
    e["_uv"] = e["upvotes"].fillna(-1)
    e = e.sort_values(["question_slug", "language", "_uv"],
                      ascending=[True, True, False])
    keep = e.drop_duplicates(subset=["question_slug", "language"], keep="first")

    out = pd.DataFrame({
        "question_slug": keep["question_slug"].values,
        "language": keep["language"].values,
        "canonical_code": keep["canonical_code"].values,
    })
    out["n_lines"] = out["canonical_code"].str.count("\n") + 1
    out["n_chars"] = out["canonical_code"].str.len()

    print(f"  unique (slug,lang) rows: {len(out)}")
    print(f"  unique slugs: {out['question_slug'].nunique()}")
    print(f"  per-lang rows: {out['language'].value_counts().to_dict()}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DST, index=False)
    print(f"wrote {DST}, shape={out.shape}")


if __name__ == "__main__":
    main()
