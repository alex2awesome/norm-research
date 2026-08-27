import json
from argparse import Namespace
from pathlib import Path

from scripts.tools.silver_match_v3.common import stable_uid
from scripts.tools.silver_match_v3.export_sonnet import export


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _result(path: Path, results):
    _jsonl(
        path,
        [
            {
                "type": "result",
                "key": "fixture-event",
                "agentId": "fixture-sonnet",
                "result": {"results": results},
            }
        ],
    )


def _choice_anchors(corpus="humor"):
    return [
        {
            "idx": "anchor-good-0-0",
            "norm": "criterion alpha is weak",
            "top10": ["Metric Alpha", "Metric Beta"],
        },
        {
            "idx": "anchor-good-0-1",
            "norm": "criterion beta is weak",
            "top10": ["Metric Beta", "Metric Alpha"],
        },
        {
            "idx": "anchor-noise-0",
            "norm": "posted from my phone",
            "top10": ["Metric Alpha", "Metric Beta"],
        },
    ]


def _choice_anchor_results():
    return [
        {"idx": "anchor-good-0-0", "choice": "Metric Alpha", "confidence": "high"},
        {"idx": "anchor-good-0-1", "choice": "Metric Beta", "confidence": "high"},
        {"idx": "anchor-noise-0", "choice": "NOISE", "confidence": "high"},
    ]


def _fixture(tmp_path: Path):
    manifest = tmp_path / "manifest"
    v1 = tmp_path / "v1"
    scratch = tmp_path / "scratch"
    journals = tmp_path / "journals"
    output = tmp_path / "out"

    norms = [
        "the alpha criterion is excellent",
        "there is no matching criterion",
        "this leaked choice must be rejected",
        "duplicate-name bridge is unsafe",
    ]
    canonical = [
        {
            "schema_version": "silver-match-v3.0",
            "norm_uid": stable_uid("humor", row, norm),
            "corpus": "humor",
            "task": "humor",
            "row": row,
            "source_id": str(row),
            "norm": norm,
        }
        for zero_row, norm in enumerate(norms)
        for row in [zero_row + 1]
    ]
    _jsonl(manifest / "norms/humor.jsonl", canonical)
    bank = {
        "schema_version": "silver-match-v3.0",
        "task": "humor",
        "source_sha256": "bank-sha",
        "metrics": [
            {
                "metric_id": "a0",
                "name": "Metric Alpha",
                "name_key": "metric alpha",
                "name_ambiguous": False,
            },
            {
                "metric_id": "a1",
                "name": "Metric Beta",
                "name_key": "metric beta",
                "name_ambiguous": False,
            },
            {
                "metric_id": "a2",
                "name": "Ambiguous Name",
                "name_key": "ambiguous name",
                "name_ambiguous": True,
            },
            {
                "metric_id": "a3",
                "name": "Ambiguous-Name",
                "name_key": "ambiguous name",
                "name_ambiguous": True,
            },
        ],
    }
    (manifest / "banks").mkdir(parents=True)
    (manifest / "banks/humor.json").write_text(json.dumps(bank), encoding="utf-8")
    (manifest / "manifest.json").write_text(
        json.dumps(
            {
                "routing": {"humor": "humor"},
                "corpora": {"humor": {"path": "ignored"}},
                "banks": {"humor": {"path": "ignored", "source_sha256": "bank-sha"}},
            }
        ),
        encoding="utf-8",
    )

    _jsonl(
        v1 / "matches_joined_humor.jsonl",
        [
            {
                "row": row + 1,
                "norm": norm,
                "top10_names": [
                    {"id": "old0", "name": first},
                    {"id": "old1", "name": second},
                ],
            }
            for row, (norm, first, second) in enumerate(
                [
                    (norms[0], "Metric Alpha", "Metric Beta"),
                    (norms[1], "Metric Alpha", "Metric Beta"),
                    (norms[2], "Metric Alpha", "Metric Beta"),
                    (norms[3], "Ambiguous Name", "Metric Alpha"),
                ]
            )
        ],
    )

    for directory in (scratch / "rematch_full", scratch / "rematch_pilot"):
        directory.mkdir(parents=True)
        (directory / "humor_b0000.json").write_text(
            json.dumps({"task": "humor", "items": _choice_anchors()}), encoding="utf-8"
        )
    (scratch / "abstain_rescue").mkdir(parents=True)
    (scratch / "abstain_rescue/b000.json").write_text(
        json.dumps(
            {
                "_anchor_found_target": "Metric Alpha",
                "bank": ["Metric Alpha", "Metric Beta"],
                "items": [
                    {
                        "idx": "anchor-found-0",
                        "norm": "what really sank it for me was the metric alpha — just not up to par",
                    },
                    {"idx": "anchor-gap-0", "norm": "bluebook citations"},
                ],
            }
        ),
        encoding="utf-8",
    )

    return Namespace(
        manifest_root=str(manifest),
        output_root=str(output),
        v1_root=str(v1),
        scratch_root=str(scratch),
        full_journal=str(journals / "full.jsonl"),
        pilot_journal=str(journals / "pilot.jsonl"),
        audit_journal=str(journals / "audit.jsonl"),
        rescue_journal=str(journals / "rescue.jsonl"),
    ), journals, output


def test_export_precedence_typed_rescue_and_leakage_rejection(tmp_path):
    args, journals, output = _fixture(tmp_path)
    anchors = _choice_anchor_results()
    _result(
        journals / "full.jsonl",
        anchors
        + [
            {"idx": "humor-1", "choice": "Metric Alpha", "confidence": "high"},
            {"idx": "humor-2", "choice": "ABSTAIN", "confidence": "low"},
            # It is a current metric, but not this item's candidate.  This is
            # exactly the shared-prompt leakage the exporter must discard.
            {"idx": "humor-3", "choice": "Ambiguous Name", "confidence": "high"},
            {"idx": "humor-4", "choice": "Ambiguous Name", "confidence": "high"},
        ],
    )
    _result(
        journals / "pilot.jsonl",
        anchors + [{"idx": "humor-1", "choice": "Metric Alpha", "confidence": "low"}],
    )
    _result(
        journals / "rescue.jsonl",
        [
            {"idx": "anchor-found-0", "verdict": "found", "metric": "Metric Alpha", "proposed_name": None},
            {"idx": "anchor-gap-0", "verdict": "not_in_bank", "metric": None, "proposed_name": "Citation style"},
            {"idx": "humor-1", "verdict": "found", "metric": "Metric Alpha", "proposed_name": None},
            {"idx": "humor-2", "verdict": "not_in_bank", "metric": None, "proposed_name": "Missing criterion"},
        ],
    )
    _result(
        journals / "audit.jsonl",
        [
            {"idx": "anchor-good-0-0", "top1_fit": "exact", "best_rank": 1, "better_bank_metric": None, "is_preference": True},
            {"idx": "anchor-good-0-1", "top1_fit": "exact", "best_rank": 1, "better_bank_metric": None, "is_preference": True},
            {"idx": "anchor-wrong-0-0", "top1_fit": "wrong", "best_rank": 2, "better_bank_metric": None, "is_preference": True},
            {"idx": "anchor-wrong-0-1", "top1_fit": "wrong", "best_rank": 2, "better_bank_metric": None, "is_preference": True},
            {"idx": "humor-1", "top1_fit": "wrong", "best_rank": 2, "better_bank_metric": None, "is_preference": True},
        ],
    )

    summary = export(args)
    teachers = [json.loads(line) for line in (output / "teachers/sonnet.jsonl").read_text().splitlines()]
    rejected = [json.loads(line) for line in (output / "teachers/sonnet_rejections.jsonl").read_text().splitlines()]

    assert summary["teachers"] == 2
    assert {(row["row"], row["decision"]) for row in teachers} == {(1, "MATCH"), (2, "BANK_GAP")}
    first = next(row for row in teachers if row["row"] == 1)
    assert first["label_source"] == "sonnet_audit"
    assert first["metric_id"] == "a1"
    assert first["norm_uid"] == stable_uid("humor", 1, "the alpha criterion is excellent")
    assert first["current_bank_source_sha256"] == "bank-sha"
    gap = next(row for row in teachers if row["row"] == 2)
    assert gap["label_source"] == "sonnet_rescue"
    assert gap["metric_id"] is None
    assert {row["reason"] for row in rejected} >= {
        "cross_item_candidate_leakage",
        "ambiguous_current_bank_name",
    }


def test_failed_exact_anchor_gate_discards_entire_batch(tmp_path):
    args, journals, output = _fixture(tmp_path)
    bad_anchors = _choice_anchor_results()
    bad_anchors[0]["choice"] = "Metric Beta"
    _result(
        journals / "full.jsonl",
        bad_anchors + [{"idx": "humor-1", "choice": "Metric Alpha", "confidence": "high"}],
    )
    # Empty/missing other journals are supported.
    summary = export(args)
    assert summary["teachers"] == 0
    rejected = [json.loads(line) for line in (output / "teachers/sonnet_rejections.jsonl").read_text().splitlines()]
    assert [row["reason"] for row in rejected] == ["anchor_gate_failed"]
