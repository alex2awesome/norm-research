import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.audit_candidate_outputs import audit_candidates
from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.freeze_retrieval_production_plan import freeze


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path):
    bank = tmp_path / "bank.json"
    _write_json(
        bank,
        {
            "metrics": [
                {"metric_id": "m1", "name": "one", "description": "one"},
                {"metric_id": "m2", "name": "two", "description": "two"},
            ]
        },
    )
    corpora = {}
    candidate_paths = []
    for index, corpus in enumerate(("a", "b")):
        uid = str(index) * 64
        norms = tmp_path / f"{corpus}.jsonl"
        write_jsonl(norms, [{"norm_uid": uid, "row": 0}])
        corpora[corpus] = {
            "task": "t",
            "count": 1,
            "path": str(norms),
        }
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "corpora": corpora,
            "banks": {
                "t": {
                    "count": 2,
                    "path": str(bank),
                    "source_sha256": "bank-sha",
                }
            },
        },
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("config", encoding="utf-8")
    adapter_hashes = {
        path.name: sha256_file(path) for path in adapter.iterdir() if path.is_file()
    }
    dev = tmp_path / "external_dev.jsonl"
    write_jsonl(
        dev,
        [
            {
                "norm_uid": "d" * 64,
                "task": "t",
                "split": "dev",
                "verdict": "MATCH",
            }
        ],
    )
    dev_candidates = tmp_path / "dev_candidates.jsonl"
    write_jsonl(
        dev_candidates,
        [{"norm_uid": "d" * 64, "task": "t", "candidates": []}],
    )
    retrieval_signature = {
        "encoder": "/models/frozen-encoder",
        "query_format": "nemotron",
        "query_views": "evidence+statement",
        "dense_query_instruction": True,
    }
    _write_json(
        dev_candidates.with_suffix(".jsonl.meta.json"),
        {
            "output_sha256": sha256_file(dev_candidates),
            "items": str(dev),
            "items_sha256": sha256_file(dev),
            "manifest_sha256": sha256_file(manifest),
            "adapter": str(adapter),
            "adapter_hashes": adapter_hashes,
            **retrieval_signature,
        },
    )
    fusion = tmp_path / "fusion.json"
    _write_json(
        fusion,
        {
            "task": "t",
            "selection_split": "dev",
            "candidate_inputs": {
                str(dev_candidates): sha256_file(dev_candidates)
            },
            "label_inputs": {str(dev): sha256_file(dev)},
        },
    )
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "task": "t",
            "selection_split": "external_dev_only",
            "frozen_test_consumed": False,
            "label_inputs": {str(dev): sha256_file(dev)},
            "chosen": {
                "name": "adapter",
                "kind": "adapter",
                "fusion_report": str(fusion),
                "fusion_report_sha256": sha256_file(fusion),
                "candidate_inputs": {
                    str(dev_candidates): sha256_file(dev_candidates)
                },
            },
        },
    )
    audits = []
    all_rows = []
    for index, corpus in enumerate(("a", "b")):
        uid = str(index) * 64
        candidates = tmp_path / f"{corpus}.candidates.jsonl"
        rows = [
            {
                "norm_uid": uid,
                "corpus": corpus,
                "task": "t",
                "row": 0,
                "bank_source_sha256": "bank-sha",
                "candidates": [
                    {"metric_id": "m1", "rank": 1},
                    {"metric_id": "m2", "rank": 2},
                ],
            }
        ]
        write_jsonl(candidates, rows)
        _write_json(
            candidates.with_suffix(".jsonl.meta.json"),
            {
                "output_sha256": sha256_file(candidates),
                "manifest_sha256": sha256_file(manifest),
                "corpus": corpus,
                "task": "t",
                "bank_source_sha256": "bank-sha",
                "output_k": 2,
                "fusion_weights": str(fusion),
                "fusion_weights_sha256": sha256_file(fusion),
                "adapter": str(adapter),
                "adapter_hashes": adapter_hashes,
                **retrieval_signature,
            },
        )
        audit = tmp_path / f"{corpus}.audit.json"
        _write_json(
            audit,
            audit_candidates(
                manifest_path=manifest,
                corpus=corpus,
                candidate_paths=[candidates],
                expected_k=2,
            ),
        )
        audits.append(audit)
        candidate_paths.append(candidates)
        all_rows.extend(rows)
    union = tmp_path / "union.jsonl"
    write_jsonl(union, all_rows)
    _write_json(
        union.with_suffix(".jsonl.meta.json"),
        {
            "count": 2,
            "sha256": sha256_file(union),
            "inputs": {
                str(path.resolve()): {"count": 1, "sha256": sha256_file(path)}
                for path in candidate_paths
            },
        },
    )
    return manifest, union, audits, selection


def test_freeze_pins_selected_dev_retriever_to_exact_production(tmp_path):
    manifest, union, audits, selection = _fixture(tmp_path)
    plan = freeze(
        manifest_path=manifest,
        task="t",
        candidate_union_path=union,
        audit_paths=audits,
        selection_path=selection,
        expected_k=2,
    )
    assert plan["status"] == "FROZEN_RETRIEVAL_READY_FOR_K50_POLICY"
    assert plan["expected_count"] == 2
    assert plan["authorization"]["adjudication_authorized"] is False


def test_freeze_rejects_production_signature_not_used_on_dev(tmp_path):
    manifest, union, audits, selection = _fixture(tmp_path)
    payload = json.loads(audits[0].read_text(encoding="utf-8"))
    payload["retrieval_signatures"][0]["encoder"] = "/models/wrong"
    _write_json(audits[0], payload)
    with pytest.raises(ValueError, match="production retrieval signature"):
        freeze(
            manifest_path=manifest,
            task="t",
            candidate_union_path=union,
            audit_paths=audits,
            selection_path=selection,
            expected_k=2,
        )
