"""Reconstruct a single-file new-side view for every PR in
code_review_dense_4096tok.

For each PR row (text field = templated "## PR Title / ## PR Description /
## Code Diff (k/N files)" plus a unified diff truncated to fit 4096 tokens),
we:

  1. Strip the PR Title / Description preamble.
  2. Walk every `diff --git a/X b/Y` block, count additions/deletions per
     file (lines beginning with "+" or "-", excluding the file headers).
  3. Pick the largest file by (additions + deletions).
  4. Reconstruct that file's "new-side" partial view: concatenate the
     " " (context) and "+" lines from every hunk in source order, dropping
     "-" lines and hunk markers.
  5. Infer language from the file extension.

Writes outputs/v2_analysis/dense_4096tok_single_file_reconstructed.parquet
with columns:

  paper_id, owner, repo, pr_number, split, judgement, language,
  file_path, file_language, file_changes, file_additions, file_deletions,
  file_text, n_files_in_diff

Run as:
  /lfs/skampere3/0/alexspan/miniconda3/bin/python3.11 \\
    scripts/dense_4096tok_reconstruct_single_file.py
"""
from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
SRC_DIR = REPO / "datasets/code-review/code_review_dense_4096tok"
OUT_PATH = REPO / "outputs/v2_analysis/dense_4096tok_single_file_reconstructed.parquet"

EXT_TO_LANG = {
    ".py": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".c": "C", ".h": "C",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".hpp": "C++",
    ".cs": "C#",
    ".kt": "Kotlin", ".kts": "Kotlin",
    ".scala": "Scala",
    ".php": "PHP",
    ".swift": "Swift",
    ".m": "Objective-C", ".mm": "Objective-C++",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
    ".sql": "SQL",
    ".html": "HTML", ".css": "CSS", ".scss": "CSS",
    ".yml": "YAML", ".yaml": "YAML",
    ".json": "JSON",
    ".md": "Markdown", ".rst": "RST", ".txt": "Text",
    ".xml": "XML",
    ".toml": "TOML", ".ini": "INI",
    ".dockerfile": "Dockerfile",
    ".tf": "Terraform",
    ".proto": "Protobuf",
    ".lua": "Lua",
    ".dart": "Dart",
    ".pl": "Perl", ".pm": "Perl",
    ".r": "R",
    ".jl": "Julia",
}

# Header lines we drop when walking hunks.
_HUNK_HEADER = re.compile(r"^@@ .*@@")


def _infer_lang(path: str) -> str:
    p = path.lower()
    # Dockerfile and Makefile by basename.
    base = p.rsplit("/", 1)[-1]
    if base == "dockerfile" or base.startswith("dockerfile."):
        return "Dockerfile"
    if base in {"makefile", "gnumakefile"} or base.endswith(".mk"):
        return "Makefile"
    # Extension-based lookup.
    if "." in base:
        ext = "." + base.rsplit(".", 1)[-1]
        if ext in EXT_TO_LANG:
            return EXT_TO_LANG[ext]
    return "Other"


def _parse_paper_id(paper_id: str) -> Tuple[str, str, Optional[int]]:
    """`owner/repo#PR` -> (owner, repo, pr_number)."""
    try:
        left, num = paper_id.rsplit("#", 1)
        owner, repo = left.split("/", 1)
        return owner, repo, int(num)
    except Exception:
        return "", "", None


def _strip_preamble(text: str) -> str:
    """Return diff portion starting at the first `diff --git`. If absent, ""."""
    idx = text.find("diff --git")
    return text[idx:] if idx != -1 else ""


def _split_into_file_blocks(diff_text: str) -> List[Tuple[str, List[str]]]:
    """Split unified-diff text into per-file (path, hunk_lines) tuples.

    The path is taken from the `+++ b/...` line when available, falling back
    to `--- a/...`. Lines belonging to a file are everything between the
    `diff --git` line for that file and the next `diff --git` (or EOF).

    The returned hunk_lines list excludes:
      - `diff --git ...` and `index ...` lines
      - `--- a/...` / `+++ b/...` headers
      - `\\ No newline at end of file` markers
    But INCLUDES hunk-header lines (`@@ -X +Y @@`) so we can split into hunks
    later if we want; the reconstruction step drops them.
    """
    if not diff_text:
        return []
    lines = diff_text.splitlines()
    blocks: List[Tuple[str, List[str]]] = []
    cur_path: Optional[str] = None
    cur_lines: List[str] = []
    i = 0
    N = len(lines)
    while i < N:
        ln = lines[i]
        if ln.startswith("diff --git"):
            # flush previous
            if cur_path is not None:
                blocks.append((cur_path, cur_lines))
            cur_path = None
            cur_lines = []
            # Try to grab path from the diff --git header itself as fallback.
            # Format: `diff --git a/<old> b/<new>` — but paths may contain
            # spaces, so we be defensive and only use as fallback.
            i += 1
            # Consume index / mode / similarity lines and ---/+++ headers.
            while i < N and not lines[i].startswith("@@"):
                hdr = lines[i]
                if hdr.startswith("+++ "):
                    p = hdr[4:].strip()
                    if p.startswith("b/"):
                        p = p[2:]
                    if p and p != "/dev/null":
                        cur_path = p
                elif hdr.startswith("--- ") and cur_path is None:
                    p = hdr[4:].strip()
                    if p.startswith("a/"):
                        p = p[2:]
                    if p and p != "/dev/null":
                        cur_path = p
                if lines[i].startswith("diff --git"):
                    break
                i += 1
            continue
        # Otherwise, body line — only keep if we're inside a file block.
        if cur_path is not None:
            if ln.startswith("\\ No newline at end of file"):
                pass
            else:
                cur_lines.append(ln)
        i += 1
    if cur_path is not None:
        blocks.append((cur_path, cur_lines))
    # Some PRs in the corpus get truncated mid-hunk; we keep partial blocks.
    return blocks


def _count_changes(body_lines: List[str]) -> Tuple[int, int]:
    """Count `+` / `-` lines inside hunks. Excludes hunk header lines."""
    add = dele = 0
    in_hunk = False
    for ln in body_lines:
        if _HUNK_HEADER.match(ln):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if not ln:
            continue
        c = ln[0]
        if c == "+":
            add += 1
        elif c == "-":
            dele += 1
        # ' ' (context) lines are not counted as changes.
    return add, dele


def _reconstruct_new_side(body_lines: List[str]) -> str:
    """Concatenate context + added lines (drop deletions and hunk headers)."""
    out: List[str] = []
    in_hunk = False
    for ln in body_lines:
        if _HUNK_HEADER.match(ln):
            in_hunk = True
            # Optional: emit a marker comment so adjacent hunks don't merge
            # mid-statement. Use a blank line — language-agnostic.
            if out and out[-1] != "":
                out.append("")
            continue
        if not in_hunk:
            continue
        if not ln:
            # blank line inside a hunk is context with empty text
            out.append("")
            continue
        c = ln[0]
        if c == "+":
            out.append(ln[1:])
        elif c == " ":
            out.append(ln[1:])
        # skip '-'
    return "\n".join(out)


def reconstruct_one(text: str) -> Dict:
    """Return per-PR reconstruction record (without paper_id/y/etc.)."""
    diff = _strip_preamble(text or "")
    blocks = _split_into_file_blocks(diff)
    if not blocks:
        return {
            "file_path": None, "file_language": None,
            "file_changes": 0, "file_additions": 0, "file_deletions": 0,
            "file_text": None, "n_files_in_diff": 0,
        }
    # Per-file change counts.
    per_file = []
    for path, body in blocks:
        a, d = _count_changes(body)
        per_file.append((path, body, a, d))
    # Pick largest by additions + deletions; ties broken by additions.
    per_file.sort(key=lambda t: (t[2] + t[3], t[2]), reverse=True)
    path, body, add, dele = per_file[0]
    file_text = _reconstruct_new_side(body)
    return {
        "file_path": path,
        "file_language": _infer_lang(path),
        "file_changes": add + dele,
        "file_additions": add,
        "file_deletions": dele,
        "file_text": file_text or None,
        "n_files_in_diff": len(blocks),
    }


def _process_chunk(payload):
    rows = payload
    out = []
    for rec in rows:
        r = reconstruct_one(rec["text"])
        r["paper_id"] = rec["paper_id"]
        r["split"] = rec["split"]
        r["judgement"] = rec["judgement"]
        r["language"] = rec["language"]   # repo-language tag (PR-level)
        r["num_files"] = rec["num_files"]
        owner, repo, prn = _parse_paper_id(rec["paper_id"])
        r["owner"] = owner
        r["repo"] = repo
        r["pr_number"] = prn
        out.append(r)
    return out


def main():
    print("Loading dense PR splits...")
    parts = []
    for split in ("train", "eval", "test"):
        p = SRC_DIR / f"{split}.csv"
        df = pd.read_csv(p, usecols=["paper_id", "text", "judgement",
                                     "language", "num_files"])
        df["split"] = split
        parts.append(df)
        print(f"  {split:>5}: {len(df):,} rows")
    df = pd.concat(parts, ignore_index=True)
    print(f"total rows: {len(df):,}")

    # Records to ship to workers.
    records = df.to_dict(orient="records")

    n_workers = 8
    chunk_size = max(1, len(records) // (n_workers * 8))
    chunks = [records[i:i + chunk_size]
              for i in range(0, len(records), chunk_size)]
    print(f"workers={n_workers}, chunks={len(chunks)}, "
          f"chunk_size={chunk_size}")

    results: List[Dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_process_chunk, c) for c in chunks]
        for k, fut in enumerate(as_completed(futs)):
            results.extend(fut.result())
            if (k + 1) % 8 == 0 or (k + 1) == len(chunks):
                print(f"  chunks done {k+1}/{len(chunks)} — "
                      f"rows so far {len(results):,}")

    out = pd.DataFrame(results)
    # Order columns nicely.
    col_order = [
        "paper_id", "owner", "repo", "pr_number", "split", "judgement",
        "language", "num_files", "n_files_in_diff",
        "file_path", "file_language",
        "file_additions", "file_deletions", "file_changes",
        "file_text",
    ]
    out = out[[c for c in col_order if c in out.columns]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH)
    print(f"\nwrote {OUT_PATH}, shape={out.shape}")

    # ===== Quick summary stats =====
    n = len(out)
    n_with_file = int(out["file_text"].notna().sum())
    n_empty = int(out["file_text"].isna().sum())
    print(f"\nReconstruction stats:")
    print(f"  PRs total                  : {n:,}")
    print(f"  PRs with reconstructed file: {n_with_file:,} ({n_with_file/n:.1%})")
    print(f"  PRs with no usable diff    : {n_empty:,} ({n_empty/n:.1%})")
    print(f"\nfile_changes distribution (added+deleted lines):")
    print(out["file_changes"].describe(percentiles=[.1, .25, .5, .75, .9, .99]))
    print(f"\nLanguage mix (file_language, top 20):")
    print(out["file_language"].value_counts().head(20))
    if "file_text" in out:
        text_len = out["file_text"].fillna("").str.len()
        print(f"\nfile_text length (chars):")
        print(text_len.describe(percentiles=[.1, .25, .5, .75, .9, .99]))


if __name__ == "__main__":
    main()
