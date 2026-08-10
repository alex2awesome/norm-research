from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.freeze_small_checker_benchmark import freeze


CANONICAL_SHA = "b614e345a07123f9fe79d9521351886107476d34cf2b09daa50efce71dc1356f"


def _args(tmp_path: Path, canonical: Path) -> argparse.Namespace:
    truth = tmp_path / "truth.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    primary = tmp_path / "primary.jsonl"
    targets = tmp_path / "targets.jsonl"
    rows = []
    for index, target in enumerate(("CONFIRM_MATCH", "REJECT")):
        uid = f"u{index}"
        rows.append(uid)
        write = {
            "norm_uid": uid,
            "task": "press-releases",
            "gepa_role": "optimize",
            "split": "train",
            "prompt_gradient_eligible": True,
            "source_group": f"g{index}",
        }
        with truth.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(write) + "\n")
        with candidates.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"norm_uid": uid, "candidates": [{"metric_id": "a1"}]}) + "\n")
        with primary.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"norm_uid": uid, "decision": "MATCH", "metric_id": "a1"}) + "\n")
        with targets.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"norm_uid": uid, "target": target}) + "\n")
    report = tmp_path / "REPORT.json"
    report.write_text(json.dumps({
        "schema_version": "silver-match-v3-balanced-verifier-gepa-train-v1",
        "count": 2,
        "output_hashes": {name: sha256_file(path) for name, path in {
            "truth": truth, "candidates": candidates, "primary": primary, "targets": targets
        }.items()},
    }))
    inference = tmp_path / "inference.json"
    inference.write_text(json.dumps({"banks": {"press-releases": {"source_sha256": "bank"}}}))
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("verify\n")
    return argparse.Namespace(
        task="press-releases", canonical_manifest=str(canonical), inference_manifest=str(inference),
        balanced_report=str(report), truth=str(truth), candidates=str(candidates), primary=str(primary),
        targets=str(targets), prompt=str(prompt), model="openai/gpt-5-mini",
        orders=["original", "hashed"], max_api_requests=8, max_alternatives=15,
        max_tokens=180, reasoning_effort="minimal", force_json_object=True,
        seed=1, output_root=str(tmp_path / "run"),
    )


def test_freeze_requires_real_canonical_manifest(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.json"
    canonical.write_text(json.dumps({"banks": {"press-releases": {"count": 1, "source_sha256": "bank"}}}))
    args = _args(tmp_path, canonical)
    with pytest.raises(ValueError, match="canonical manifest"):
        freeze(args)


def test_freeze_rejects_non_mini_model(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.json"
    canonical.write_text("{}")
    args = _args(tmp_path, canonical)
    args.model = "openai/gpt-5"
    with pytest.raises(ValueError, match="gpt-5-mini"):
        freeze(args)
