#!/usr/bin/env python3
"""Materialize the sole prompt from a pre-select frozen author output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-output", required=True)
    parser.add_argument("--output-freeze", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    author_path = Path(args.author_output).resolve()
    freeze_path = Path(args.output_freeze).resolve()
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError("refusing to overwrite frozen authored prompt")

    author = json.loads(author_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    provenance = author.get("provenance") or {}
    selection = author.get("selection_rule") or {}
    if (
        freeze.get("schema_version") != "silver-match-v3-pr-r4-author-output-freeze-v1"
        or freeze.get("status") != "FROZEN_BEFORE_ANY_SELECT_JOIN"
        or freeze.get("select_material_joined") is not False
        or freeze.get("variant_count") != 1
        or (freeze.get("author_output") or {}).get("sha256") != sha256_file(author_path)
        or author.get("schema_version") != "silver-match-v3-pr-r4-fresh-author-v1"
        or provenance.get("used_only_frozen_handoff") is not True
        or provenance.get("select_material_read") is not False
        or provenance.get("select_scores_read") is not False
        or provenance.get("mi_or_outcomes_read") is not False
        or selection.get("variant_count") != 1
        or selection.get("choose_without_select_scores") is not True
    ):
        raise ValueError("author output is not the sole pre-select frozen variant")
    prompt = str(author.get("prompt_text") or "").rstrip() + "\n"
    if len(prompt) < 500:
        raise ValueError("frozen authored prompt is unexpectedly short")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")
    meta = {
        "schema_version": "silver-match-v3-materialized-frozen-author-prompt-v1",
        "status": "MATERIALIZED_WITHOUT_PROMPT_MUTATION",
        "variant_count": 1,
        "select_material_joined_at_authoring": False,
        "author_output": {"path": str(author_path), "sha256": sha256_file(author_path)},
        "author_output_freeze": {
            "path": str(freeze_path),
            "sha256": sha256_file(freeze_path),
        },
        "prompt": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "materialization_rule": "prompt_text.rstrip() + newline",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**meta, "meta_sha256": sha256_file(meta_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
