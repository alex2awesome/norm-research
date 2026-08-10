#!/usr/bin/env python3
"""Freeze label-blind TF-IDF K candidates for CE train/dev identities."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import metric_card, norm_query, read_jsonl, sha256_file, write_jsonl
from .retrieve import build_vectorizers, top_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--identities", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    manifest_path = Path(args.manifest).resolve()
    identity_paths = [Path(value).resolve() for value in args.identities]
    output = Path(args.output).resolve()
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or meta_path.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_meta = manifest["banks"][args.task]
    bank_path = Path(bank_meta["path"]).resolve()
    bank = json.loads(bank_path.read_text(encoding="utf-8"))["metrics"]
    if args.top_k > len(bank):
        raise ValueError("top-k exceeds frozen bank size")

    wanted: dict[str, str] = {}
    for path in identity_paths:
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if not uid:
                raise ValueError(f"identity row lacks norm_uid: {path}")
            if uid in wanted:
                raise ValueError(f"duplicate UID across CE identity inputs: {uid}")
            wanted[uid] = str(path)
    norms: dict[str, dict] = {}
    for corpus, meta in manifest["corpora"].items():
        if meta["task"] != args.task:
            continue
        for row in read_jsonl(Path(meta["path"])):
            uid = str(row["norm_uid"])
            if uid in wanted:
                norms[uid] = row
    missing = sorted(set(wanted) - set(norms))
    if missing:
        raise ValueError(f"identity UIDs absent from task manifest: {missing[:5]}")

    ordered_uids = sorted(norms)
    cards = [metric_card(row) for row in bank]
    queries = [norm_query(norms[uid]) for uid in ordered_uids]
    word, char, card_word, card_char = build_vectorizers(cards)
    scores = (word.transform(queries) @ card_word.T).toarray()
    scores += (char.transform(queries) @ card_char.T).toarray()
    rankings = top_indices(scores, args.top_k)
    rows = []
    for uid, ranking in zip(ordered_uids, rankings, strict=True):
        rows.append(
            {
                "schema_version": "silver-match-v3-ce-lexical-candidates-v1",
                "task": args.task,
                "corpus": norms[uid]["corpus"],
                "norm_uid": uid,
                "bank_source_sha256": bank_meta["source_sha256"],
                "label_fields_consumed": False,
                "retriever": "word_char_tfidf_sum_stable",
                "candidates": [
                    {"metric_id": str(bank[int(index)]["metric_id"]), "rank": rank}
                    for rank, index in enumerate(ranking, 1)
                ],
            }
        )
    write_jsonl(output, rows)
    meta = {
        "schema_version": "silver-match-v3-ce-lexical-candidates-freeze-v1",
        "status": "FROZEN_LABEL_BLIND",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "top_k": args.top_k,
        "rows": len(rows),
        "unique_uids": len(rows),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "bank": {
            "path": str(bank_path),
            "sha256": sha256_file(bank_path),
            "source_sha256": bank_meta["source_sha256"],
            "count": len(bank),
        },
        "identity_inputs": {str(path): sha256_file(path) for path in identity_paths},
        "label_fields_consumed": False,
        "ranking": "descending word+character TF-IDF cosine; stable bank-index tie break",
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**meta, "meta_sha256": sha256_file(meta_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
