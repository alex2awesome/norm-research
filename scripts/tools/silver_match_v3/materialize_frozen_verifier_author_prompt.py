#!/usr/bin/env python3
"""Materialize the sole context-isolated PR verifier prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file


def materialize(author_path: Path, freeze_path: Path, output_path: Path) -> dict:
    author_path = author_path.resolve()
    freeze_path = freeze_path.resolve()
    output_path = output_path.resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError("refusing to overwrite frozen verifier prompt")
    author = json.loads(author_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version")
        != "silver-match-v3-pr-verifier-author-output-freeze-v1"
        or freeze.get("status")
        != "FROZEN_CONTEXT_ISOLATED_BEFORE_VERIFIER_DEV_TRUTH_JOIN"
        or freeze.get("variant_count") != 1
        or freeze.get("verifier_dev_truth_joined_to_author") is not False
        or (freeze.get("transcript_audit") or {}).get("status")
        != "PASS_TOOL_FREE_INLINE_AUTHOR"
        or (freeze.get("author_output") or {}).get("sha256")
        != sha256_file(author_path)
        or author.get("schema_version")
        != "silver-match-v3-pr-verifier-fresh-author-v1"
        or (author.get("selection_rule") or {}).get("variant_count") != 1
        or (author.get("selection_rule") or {}).get(
            "choose_without_verifier_dev_truth"
        )
        is not True
    ):
        raise ValueError("author output is not the sole isolated frozen verifier variant")
    prompt = str(author.get("prompt_text") or "").rstrip() + "\n"
    if len(prompt) < 900:
        raise ValueError("frozen authored verifier prompt is unexpectedly short")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")
    meta = {
        "schema_version": "silver-match-v3-materialized-frozen-verifier-author-prompt-v1",
        "status": "MATERIALIZED_WITHOUT_PROMPT_MUTATION",
        "task": "press-releases",
        "variant_count": 1,
        "verifier_dev_truth_joined_at_authoring": False,
        "author_output": {"path": str(author_path), "sha256": sha256_file(author_path)},
        "author_output_freeze": {
            "path": str(freeze_path),
            "sha256": sha256_file(freeze_path),
        },
        "prompt": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "materialization_rule": "prompt_text.rstrip() + newline",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**meta, "meta_sha256": sha256_file(meta_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-output", required=True)
    parser.add_argument("--output-freeze", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                Path(args.author_output), Path(args.output_freeze), Path(args.output)
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

