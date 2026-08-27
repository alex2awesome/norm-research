#!/usr/bin/env python3
"""Score two-order adjudication on a train-only GEPA prompt panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file
from .score_two_order_adjudicator import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--hashed", required=True)
    parser.add_argument("--panel-role", choices=("prompt_train", "prompt_dev"), required=True)
    parser.add_argument(
        "--explicit-role",
        choices=("optimize", "select"),
        help="Validate a pre-label frozen explicit role instead of legacy hash-split fields.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        key: Path(getattr(args, key)).resolve()
        for key in ("truth", "original", "hashed")
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    truth = list(read_jsonl(paths["truth"]))
    original = {str(row["norm_uid"]): row for row in read_jsonl(paths["original"])}
    hashed = {str(row["norm_uid"]): row for row in read_jsonl(paths["hashed"])}
    uids = {str(row["norm_uid"]) for row in truth}
    if len(uids) != len(truth) or set(original) != uids or set(hashed) != uids:
        raise ValueError("two-order GEPA inputs do not have exact paired coverage")
    if args.explicit_role:
        expected_panel = "prompt_train" if args.explicit_role == "optimize" else "prompt_dev"
        if args.panel_role != expected_panel:
            raise ValueError("explicit GEPA role does not match panel role")
        if any(
            row.get("gepa_role") != args.explicit_role or row.get("split") != "train"
            for row in truth
        ):
            raise ValueError("truth does not preserve its frozen explicit train-only role")
    else:
        if any(str(row.get("predeclared_split")) != "train" for row in truth):
            raise ValueError("GEPA score panel is not wholly predeclared train-only")
        expected_local = "train" if args.panel_role == "prompt_train" else "dev"
        if any(str(row.get("split")) != expected_local for row in truth):
            raise ValueError("truth local split does not match panel role")
    prompt_hashes = {
        str(row.get("prompt_sha256") or "") for row in [*original.values(), *hashed.values()]
    }
    if len(prompt_hashes) != 1 or "" in prompt_hashes:
        raise ValueError("order outputs do not share one prompt hash")
    report = {
        "schema_version": "silver-match-v3-two-order-gepa-score-v1",
        "selection_universe": "predeclared_train_only",
        "panel_role": args.panel_role,
        "explicit_role": args.explicit_role,
        "prompt_sha256": next(iter(prompt_hashes)),
        "metrics": summarize(truth, original, hashed),
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
