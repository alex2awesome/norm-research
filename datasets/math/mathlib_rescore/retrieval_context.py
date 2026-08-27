#!/usr/bin/env python3
"""Attach nearest existing mathlib declarations to PR title+diff records.

This is an optional, label-blind preprocessing step. It consumes the
``library_decl_index.jsonl`` format emitted by ``build_library_index.py``:
``{"file": ..., "kind": ..., "decl": ...}``. The output can be passed to
``score_mathlib_gemma.py``; retrieval evidence is shown only for the two
library-fit and two existing-declaration-reuse rubrics.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np


BASE = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib")
HERE = Path(__file__).resolve().parent


def stable_doc_id(title, diff):
    payload = f"{title}\0{diff}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:24]


def load_prs(path):
    """Project the source onto title and diff only."""
    import pandas as pd

    frame = pd.read_parquet(path, columns=["title", "diff"])
    rows = []
    for record in frame.to_dict(orient="records"):
        title = str(record.get("title") or "")
        diff = str(record.get("diff") or "")
        rows.append({"title": title, "diff": diff, "doc_id": stable_doc_id(title, diff)})
    if not rows:
        raise ValueError(f"no PRs found in {path}")
    return rows


def load_declarations(path):
    records = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"file", "kind", "decl"} - set(record)
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {sorted(missing)}")
            if str(record["decl"]).strip():
                records.append(
                    {
                        "file": str(record["file"]),
                        "kind": str(record["kind"]),
                        "decl": str(record["decl"]),
                    }
                )
    if not records:
        raise ValueError(f"no declarations found in {path}")
    return records


def added_code(diff):
    """Prefer added Lean source and file paths over diff-control noise."""
    paths = []
    added = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            paths.append(line[6:])
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return "\n".join(paths + added)


def query_text(row, char_limit):
    code = added_code(row["diff"])
    if not code.strip():
        code = row["diff"]
    # Preserve identifier punctuation while collapsing large whitespace runs.
    text = re.sub(r"\s+", " ", f"{row['title']}\n{code}").strip()
    return text[:char_limit]


def declaration_text(record):
    return f"{record['file']} {record['kind']} {record['decl']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(BASE / "accept_reject_clean.parquet"))
    parser.add_argument("--index", default=str(BASE / "library_decl_index.jsonl"))
    parser.add_argument("--output", default=str(HERE / "toscore_with_context.jsonl"))
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--query-char-limit", type=int, default=12000)
    parser.add_argument("--max-decl-chars", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-features", type=int, default=250000)
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    if not 1 <= args.neighbors <= 3:
        parser.error("--neighbors must be between 1 and 3")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    rows = load_prs(args.input)
    declarations = load_declarations(args.index)
    corpus = [declaration_text(record) for record in declarations]
    print(
        f"[retrieve] fitting lexical index over {len(declarations)} declarations",
        flush=True,
    )
    # Character n-grams handle qualified Lean identifiers, underscores, Unicode
    # names, and small statement variations without requiring a model download.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    declaration_matrix = vectorizer.fit_transform(corpus)
    k = min(args.neighbors, len(declarations))
    search = NearestNeighbors(
        n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=args.jobs
    )
    search.fit(declaration_matrix)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as fh:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            queries = [query_text(row, args.query_char_limit) for row in batch]
            query_matrix = vectorizer.transform(queries)
            distances, indices = search.kneighbors(query_matrix, return_distance=True)
            for row, row_distances, row_indices in zip(batch, distances, indices):
                context = []
                for rank, (distance, index) in enumerate(
                    zip(row_distances, row_indices), 1
                ):
                    declaration = declarations[int(index)]
                    context.append(
                        {
                            "rank": rank,
                            "similarity": round(float(1.0 - distance), 6),
                            "file": declaration["file"],
                            "kind": declaration["kind"],
                            "decl": declaration["decl"][: args.max_decl_chars],
                        }
                    )
                fh.write(
                    json.dumps(
                        {
                            "doc_id": row["doc_id"],
                            "title": row["title"],
                            "diff": row["diff"],
                            "retrieval_context": context,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(
                f"[retrieve] processed {min(start + args.batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )
    os.replace(temporary, output)
    print(f"[retrieve] wrote {len(rows)} rows -> {output}", flush=True)


if __name__ == "__main__":
    main()
