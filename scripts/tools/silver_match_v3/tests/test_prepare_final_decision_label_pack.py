import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.prepare_final_decision_label_pack import prepare


def _fixture(tmp_path: Path, *, leaked: bool = False) -> Path:
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3.0",
                "task": "demo",
                "source_sha256": "bank-sha",
                "metrics": [
                    {"metric_id": "a0", "name": "Alpha", "description": "A"},
                    {"metric_id": "a1", "name": "Beta", "description": "B"},
                ],
            }
        )
        + "\n"
    )
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(
        corpus,
        [
            {
                "schema_version": "silver-match-v3.0",
                "norm_uid": "u1",
                "corpus": "demo_corpus",
                "task": "demo",
                "row": 0,
                "norm": "The response should define the term.",
                "context": "A reviewer requests a definition.",
                "aspect": "definition",
                "kind": "request",
                "polarity": "negative",
            }
        ],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3.0",
                "banks": {
                    "demo": {
                        "path": str(bank),
                        "source_sha256": "bank-sha",
                        "count": 2,
                    }
                },
                "corpora": {
                    "demo_corpus": {
                        "path": str(corpus),
                        "task": "demo",
                        "count": 1,
                    }
                },
            }
        )
        + "\n"
    )
    blind = tmp_path / "task__demo.blind.jsonl"
    write_jsonl(
        blind,
        [
            {
                "norm_uid": "u1",
                "corpus": "demo_corpus",
                "task": "demo",
                "row": 0,
                "norm": "The response should define the term.",
                "context": "A reviewer requests a definition.",
                "aspect": "definition",
                "kind": "request",
                "polarity": "negative",
                "bank_file": str(bank),
                "bank_source_sha256": "bank-sha",
                "decision": "MATCH" if leaked else None,
                "metric_id": None,
                "confidence": None,
                "reason": None,
            }
        ],
    )
    # The key is intentionally absent: pack construction must never require or
    # open the hidden system-decision sidecar.
    report = tmp_path / "sample_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-final-decision-sample-v1",
                "sample_kind": "abstention",
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "bank_outputs": {
                    "demo": {
                        "path": str(bank),
                        "sha256": sha256_file(bank),
                        "source_sha256": "bank-sha",
                    }
                },
                "outputs": {
                    "task:demo": {
                        "sample_n": 1,
                        "blind": {"path": str(blind), "sha256": sha256_file(blind)},
                        "key": {"path": str(tmp_path / "DO_NOT_READ.key.jsonl")},
                    }
                },
            }
        )
        + "\n"
    )
    return report


def test_prepares_truth_hidden_pack_without_reading_key(tmp_path: Path) -> None:
    report = _fixture(tmp_path)
    output = tmp_path / "pack"
    validation = prepare(
        sample_report_path=report,
        scope="task:demo",
        output_root=output,
        chunk_size=1,
        seed=17,
    )
    assert validation["truth_hidden"] is True
    assert validation["system_key_not_read"] is True
    assert validation["count"] == 1
    item = json.loads((output / "items.jsonl").read_text().strip())
    assert item["permanently_excluded_from_gradients"] is True
    assert item["source_group"] == item["split_group"]
    assert (output / "chunks" / "part-000.jsonl").exists()


def test_rejects_leaked_system_decision(tmp_path: Path) -> None:
    report = _fixture(tmp_path, leaked=True)
    with pytest.raises(ValueError, match="leaks a label field"):
        prepare(
            sample_report_path=report,
            scope="task:demo",
            output_root=tmp_path / "pack",
            chunk_size=1,
            seed=17,
        )
