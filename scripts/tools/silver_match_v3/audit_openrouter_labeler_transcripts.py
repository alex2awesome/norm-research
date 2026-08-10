#!/usr/bin/env python3
"""Reconstruct and audit every request in an OpenRouter full-bank label pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .run_openrouter_pack_labels import (
    TRANSCRIPT_SCHEMA,
    build_messages,
    canonical_json,
    parse_json_content,
    provider_schema,
    sha256_text,
    validate_payload,
)


FORBIDDEN_ITEM_FIELDS = {
    "decision",
    "metric_id",
    "acceptable_metric_ids",
    "label",
    "prediction",
    "raw_response",
    "candidate_ids",
}


def _artifact_matches(entry: dict[str, Any], path: Path) -> bool:
    return path.is_file() and entry.get("sha256") == sha256_file(path)


def audit(
    *,
    pack_root: Path,
    guides: list[Path],
    schema_path: Path,
    runner_path: Path,
) -> dict[str, Any]:
    root = pack_root.resolve()
    guides = [path.resolve() for path in guides]
    schema_path, runner_path = schema_path.resolve(), runner_path.resolve()
    validation_path = root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    task = str(validation.get("task") or "")
    bank_path, items_path = root / "bank.json", root / "items.jsonl"
    bank_text = bank_path.read_text(encoding="utf-8")
    bank = json.loads(bank_text)
    bank_ids = {str(row["metric_id"]) for row in bank.get("metrics") or []}
    guide_values = [(path.name, path.read_text(encoding="utf-8")) for path in guides]
    chunks = sorted((root / "chunks").glob("part-*.jsonl"))
    recorded_chunks = {
        Path(path).name: digest
        for path, digest in ((validation.get("outputs") or {}).get("chunks") or {}).items()
    }
    violations: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    observed_models: set[str] = set()
    observed_pass_names: set[str] = set()
    observed_base_urls: set[str] = set()
    items = list(read_jsonl(items_path))
    leaked = [
        str(row.get("norm_uid") or "")
        for row in items
        if FORBIDDEN_ITEM_FIELDS & set(row)
    ]
    if validation.get("truth_hidden") is not True:
        violations.append({"chunk": "*", "kind": "PACK_NOT_TRUTH_HIDDEN", "detail": "validation flag"})
    if leaked:
        violations.append({"chunk": "*", "kind": "ITEM_LABEL_LEAKAGE", "detail": str(leaked[:3])})
    if list(root.glob("*.key.jsonl")) or list(root.glob("*candidate*")):
        violations.append({"chunk": "*", "kind": "KEY_OR_CANDIDATE_PRESENT", "detail": str(root)})
    if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
        violations.append({"chunk": "*", "kind": "BANK_HASH_DRIFT", "detail": str(bank_path)})
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        violations.append({"chunk": "*", "kind": "ITEM_HASH_DRIFT", "detail": str(items_path)})
    if {path.name for path in chunks} != set(recorded_chunks):
        violations.append({"chunk": "*", "kind": "CHUNK_INVENTORY_DRIFT", "detail": str(root)})

    for chunk_path in chunks:
        chunk_id = chunk_path.stem
        raw_path = root / "raw_labels" / f"{chunk_id}.json"
        transcript_path = root / "api_transcripts" / f"{chunk_id}.json"
        try:
            if sha256_file(chunk_path) != recorded_chunks[chunk_path.name]:
                raise ValueError("chunk hash drift")
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            artifacts = transcript.get("input_artifacts") or {}
            if (
                transcript.get("schema_version") != TRANSCRIPT_SCHEMA
                or transcript.get("status") != "COMPLETE"
                or transcript.get("truth_hidden") is not True
                or transcript.get("task") != task
                or transcript.get("chunk_id") != chunk_id
                or transcript.get("api_key_logged") is not False
                or transcript.get("api_base_url") != "https://openrouter.ai/api/v1"
                or not _artifact_matches(artifacts.get("runner") or {}, runner_path)
                or not _artifact_matches(artifacts.get("pack_validation") or {}, validation_path)
                or not _artifact_matches(artifacts.get("bank") or {}, bank_path)
                or not _artifact_matches(artifacts.get("items") or {}, items_path)
                or not _artifact_matches(artifacts.get("chunk") or {}, chunk_path)
                or not _artifact_matches(artifacts.get("schema") or {}, schema_path)
                or [row.get("sha256") for row in artifacts.get("guides") or []]
                != [sha256_file(path) for path in guides]
                or not _artifact_matches(transcript.get("raw_label") or {}, raw_path)
            ):
                raise ValueError("transcript/artifact binding mismatch")
            serialized_transcript = json.dumps(transcript, ensure_ascii=False)
            # Ordinary regulatory text frequently contains the word
            # "authorization" (for example, state-authorization rules).  The
            # request transcript never stores HTTP headers, so scan only for a
            # bearer-token shape rather than that domain word.
            if "Bearer sk-" in serialized_transcript:
                raise ValueError("credential-bearing token appears in transcript")
            observed_models.add(str(transcript.get("model") or ""))
            observed_pass_names.add(str(transcript.get("pass_name") or ""))
            observed_base_urls.add(str(transcript.get("api_base_url") or ""))
            expected_uids = [str(row["norm_uid"]) for row in read_jsonl(chunk_path)]
            validate_payload(
                raw,
                task=task,
                chunk_id=chunk_id,
                expected_uids=expected_uids,
                bank_ids=bank_ids,
            )
            attempts = transcript.get("attempts") or []
            if not attempts:
                raise ValueError("missing API attempts")
            base_messages = build_messages(
                task=task,
                chunk_id=chunk_id,
                guides=guide_values,
                bank_text=bank_text,
                chunk_text=chunk_path.read_text(encoding="utf-8"),
                pass_name=str(transcript.get("pass_name") or ""),
            )
            previous_content = None
            previous_error = None
            for ordinal, attempt in enumerate(attempts, 1):
                request = attempt.get("request") or {}
                messages = request.get("messages")
                if ordinal == 1:
                    expected_messages = base_messages
                else:
                    expected_messages = [
                        *base_messages,
                        {"role": "assistant", "content": previous_content},
                        {
                            "role": "user",
                            "content": (
                                "The prior response violated the frozen JSON/UID/metric contract: "
                                f"{previous_error}. Return the complete corrected JSON object only."
                            ),
                        },
                    ]
                if (
                    int(attempt.get("ordinal", -1)) != ordinal
                    or messages != expected_messages
                    or request.get("model") != transcript.get("model")
                    or (((request.get("response_format") or {}).get("json_schema") or {}).get("schema"))
                    != provider_schema(json.loads(schema_path.read_text(encoding="utf-8")))
                    or attempt.get("request_sha256") != sha256_text(canonical_json(request))
                    or attempt.get("response_content_sha256")
                    != sha256_text(str(attempt.get("response_content") or ""))
                ):
                    raise ValueError(f"request/response reconstruction mismatch at attempt {ordinal}")
                previous_content = str(attempt.get("response_content") or "")
                previous_error = attempt.get("validation_error")
            parsed = parse_json_content(str(attempts[-1].get("response_content") or ""))
            if parsed != raw or attempts[-1].get("validation_error") is not None:
                raise ValueError("final response differs from validated raw label")
            rows.append(
                {
                    "chunk": chunk_id,
                    "chunk_sha256": sha256_file(chunk_path),
                    "raw_label_sha256": sha256_file(raw_path),
                    "transcript_sha256": sha256_file(transcript_path),
                    "request_count": len(attempts),
                    "reported_cost_usd": sum(
                        float((row.get("usage") or {}).get("cost") or 0.0) for row in attempts
                    ),
                }
            )
        except Exception as exc:
            violations.append(
                {"chunk": chunk_id, "kind": type(exc).__name__, "detail": str(exc)}
            )
    if (
        len(observed_models) != 1
        or "" in observed_models
        or len(observed_pass_names) != 1
        or "" in observed_pass_names
        or observed_base_urls != {"https://openrouter.ai/api/v1"}
    ):
        violations.append(
            {
                "chunk": "*",
                "kind": "MIXED_OR_MISSING_BACKEND_IDENTITY",
                "detail": json.dumps(
                    {
                        "models": sorted(observed_models),
                        "pass_names": sorted(observed_pass_names),
                        "base_urls": sorted(observed_base_urls),
                    },
                    sort_keys=True,
                ),
            }
        )
    return {
        "schema_version": "silver-match-v3-openrouter-labeler-transcript-audit-v1",
        "status": "PASS" if not violations and len(rows) == len(chunks) else "FAIL",
        "complete": not violations and len(rows) == len(chunks),
        "truth_hidden": True,
        "pack_root": str(root),
        "task": task,
        "model": next(iter(observed_models)) if len(observed_models) == 1 else None,
        "pass_name": (
            next(iter(observed_pass_names)) if len(observed_pass_names) == 1 else None
        ),
        "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
        "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
        "pack_validation": {"path": str(validation_path), "sha256": sha256_file(validation_path)},
        "runner": {"path": str(runner_path), "sha256": sha256_file(runner_path)},
        "schema": {"path": str(schema_path), "sha256": sha256_file(schema_path)},
        "guides": [{"path": str(path), "sha256": sha256_file(path)} for path in guides],
        "expected_chunks": len(chunks),
        "audited_chunks": len(rows),
        "request_count": sum(row["request_count"] for row in rows),
        "reported_cost_usd": sum(row["reported_cost_usd"] for row in rows),
        "chunks": rows,
        "violations": violations,
        "contract": {
            "exact_request_body_reconstructed_from_only_frozen_bank_chunk_and_guides": True,
            "sample_keys_predictions_proposals_mi_and_outcomes_absent": True,
            "api_credentials_not_logged": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--guide", action="append", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--runner", default=__file__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit(
        pack_root=Path(args.pack_root),
        guides=[Path(value) for value in args.guide],
        schema_path=Path(args.schema),
        runner_path=Path(args.runner),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output": str(output), "output_sha256": sha256_file(output)}, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
