import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import normalize_name, stable_uid
from scripts.tools.silver_match_v3.export_sonnet_forced import (
    SOURCE_RUN_ID,
    TASK_CONFIG,
    TASK_ORDER,
    IntegrityError,
    export,
)


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _agent_transcript(path: Path, agent_id: str, task_key: str, batch_lines: str, matches):
    batch_path = f"/tmp/match_{task_key}/batches/batch_000.txt"
    rows = [
        {
            "agentId": agent_id,
            "type": "user",
            "message": {
                "role": "user",
                "content": f"Read {batch_path} and return matches",
            },
        },
        {
            "agentId": agent_id,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "read-batch",
                        "name": "Read",
                        "input": {"file_path": batch_path},
                    }
                ],
            },
        },
        {
            "agentId": agent_id,
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "read-batch",
                        "content": "".join(
                            f"{line_no}\t{line}\n"
                            for line_no, line in enumerate(batch_lines.splitlines(), 1)
                        )
                        + f"{len(batch_lines.splitlines()) + 1}\t",
                    }
                ],
            },
        },
        {
            "agentId": agent_id,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "structured",
                        "name": "StructuredOutput",
                        "input": {"matches": matches},
                    }
                ],
            },
        },
    ]
    _jsonl(path, rows)


def _fixture(tmp_path: Path):
    manifest = tmp_path / "manifest"
    workflow = tmp_path / "workflow"
    inputs = tmp_path / "inputs"
    output = tmp_path / "output"
    aspect_files = []
    anchor_files = []
    journal = []

    for task_index, task_key in enumerate(TASK_ORDER):
        config = TASK_CONFIG[task_key]
        current_name = f"Current metric {task_key}"
        obsolete_name = f"Obsolete metric {task_key}"
        ambiguous_name = f"Ambiguous metric {task_key}"
        aspect_path = inputs / "aspects" / f"{task_key}.json"
        aspect_path.parent.mkdir(parents=True, exist_ok=True)
        aspect_path.write_text(
            json.dumps(
                [
                    {"aspect_id": "a0", "name": current_name},
                    {"aspect_id": "a1", "name": obsolete_name},
                    {"aspect_id": "a2", "name": ambiguous_name},
                ]
            ),
            encoding="utf-8",
        )
        aspect_files.append(f"{task_key}={aspect_path}")

        source_key = config["source_key"]
        anchor = {
            source_key: f"source-{task_index}",
            "signal_text": f"human norm for {task_key}",
            "passage_text": f"context for {task_key}",
            "reason": "valid evaluative statement",
            "faithful": 1,
            "valid": 1,
        }
        anchor_path = inputs / "anchors" / f"{task_key}.jsonl"
        _jsonl(anchor_path, [anchor])
        anchor_files.append(f"{task_key}={anchor_path}")

        corpus = config["corpus"]
        task = config["task"]
        norm_uid = stable_uid(corpus, 0, 0, anchor[source_key], 0, anchor["signal_text"])
        _jsonl(
            manifest / "norms" / f"{corpus}.jsonl",
            [
                {
                    "schema_version": "silver-match-v3.0",
                    "norm_uid": norm_uid,
                    "corpus": corpus,
                    "task": task,
                    "row": 0,
                    "source_id": anchor[source_key],
                    "norm": anchor["signal_text"],
                }
            ],
        )
        bank = {
            "schema_version": "silver-match-v3.0",
            "task": task,
            "source_sha256": f"sha-{task_key}",
            "metrics": [
                {
                    "metric_id": "m0",
                    "name": current_name,
                    "name_key": normalize_name(current_name),
                    "name_ambiguous": False,
                },
                {
                    "metric_id": "m1",
                    "name": ambiguous_name,
                    "name_key": normalize_name(ambiguous_name),
                    "name_ambiguous": True,
                },
            ],
        }
        (manifest / "banks").mkdir(parents=True, exist_ok=True)
        (manifest / "banks" / f"{task}.json").write_text(json.dumps(bank), encoding="utf-8")

        matches = [{"id": 0, "aspects": ["a0", "a1", "a2"]}]
        agent_id = f"agent-{task_index}"
        _agent_transcript(
            workflow / f"agent-{task_index}.jsonl",
            agent_id,
            task_key,
            f"0: {anchor['signal_text']}",
            matches,
        )
        key = f"fixture-{task_index}"
        journal.extend(
            [
                {"type": "started", "key": key, "agentId": agent_id},
                {
                    "type": "result",
                    "key": key,
                    "agentId": agent_id,
                    "result": {"matches": matches},
                },
            ]
        )

    _jsonl(workflow / "journal.jsonl", journal)
    args = Namespace(
        manifest_root=str(manifest),
        output_root=str(output),
        workflow_root=str(workflow),
        journal=str(workflow / "journal.jsonl"),
        aspect_file=aspect_files,
        anchor_file=anchor_files,
    )
    return args, output, workflow


def test_forced_labels_are_strictly_bridged_and_train_only(tmp_path):
    args, output, _ = _fixture(tmp_path)
    summary = export(args)
    teachers = [
        json.loads(line)
        for line in (output / "teachers/sonnet_forced_top3.production.jsonl").read_text().splitlines()
    ]
    rejected = [
        json.loads(line)
        for line in (output / "teachers/sonnet_forced_top3.rejections.jsonl").read_text().splitlines()
    ]
    assert summary["sampled_norms"] == 4
    assert summary["batches"] == 4
    assert len(teachers) == 4
    assert {row["metric_id"] for row in teachers} == {"m0"}
    assert all(row["data_role"] == "train_only" for row in teachers)
    assert all(row["eligible_for_evaluation"] is False for row in teachers)
    assert all(row["eligible_for_threshold_calibration"] is False for row in teachers)
    assert all(row["abstention_was_available"] is False for row in teachers)
    assert all(row["source_run_id"] == SOURCE_RUN_ID for row in teachers)
    assert {row["reason"] for row in rejected} == {
        "obsolete_aspect_name_not_in_current_bank",
        "ambiguous_current_bank_name",
    }


def test_cross_batch_output_id_is_rejected_without_poisoning_batch(tmp_path):
    args, output, workflow = _fixture(tmp_path)
    journal = [json.loads(line) for line in (workflow / "journal.jsonl").read_text().splitlines()]
    first = next(row for row in journal if row["type"] == "result")
    leaked = dict(first["result"]["matches"][0])
    leaked["id"] = 99
    first["result"]["matches"].append(leaked)
    _jsonl(workflow / "journal.jsonl", journal)
    # The journal is a faithful copy of the transcript output; the leakage is
    # therefore a model-label error rather than a journal corruption.
    transcript_path = workflow / "agent-0.jsonl"
    transcript = [json.loads(line) for line in transcript_path.read_text().splitlines()]
    for row in transcript:
        for block in row.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("name") == "StructuredOutput":
                block["input"]["matches"].append(leaked)
    _jsonl(transcript_path, transcript)
    summary = export(args)
    assert summary["teachers"] == 4
    rejected = [
        json.loads(line)
        for line in (output / "teachers/sonnet_forced_top3.rejections.jsonl").read_text().splitlines()
    ]
    assert "cross_batch_output_id_leakage" in {row["reason"] for row in rejected}


def test_canonical_source_identity_mismatch_is_rejected(tmp_path):
    args, output, _ = _fixture(tmp_path)
    corpus = TASK_CONFIG["humor"]["corpus"]
    path = Path(args.manifest_root) / "norms" / f"{corpus}.jsonl"
    row = json.loads(path.read_text())
    row["source_id"] = "wrong-source"
    _jsonl(path, [row])
    summary = export(args)
    assert summary["teachers"] == 3
    rejected = [
        json.loads(line)
        for line in (output / "teachers/sonnet_forced_top3.rejections.jsonl").read_text().splitlines()
    ]
    assert "canonical_anchor_identity_mismatch" in {row["reason"] for row in rejected}
