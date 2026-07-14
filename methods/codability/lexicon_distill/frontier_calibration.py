"""Stage and assemble a fresh frontier-LLM development panel for one similarity protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset import LABELS
from .hierarchy_contracts import sha256_file, validate_pair_files


VERSION = "frontier-similarity-calibration-panel-v1"
TIEBREAK_VERSION = "frontier-similarity-calibration-disagreements-v1"
POSTFREEZE_VERSION = "postfreeze-hierarchy-audit-v1"
_BLIND_KEYS = {"pair_id", "task", "level", "concept_a", "concept_b"}


def _rows(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object JSONL row in {path}")
                result.append(value)
    return result


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _validate_frozen_panel(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authenticate a base panel or disagreement-only derivative before any API request."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema_version")
    if schema not in {VERSION, TIEBREAK_VERSION, POSTFREEZE_VERSION}:
        raise ValueError("unsupported calibration manifest")
    if schema == VERSION:
        fields = ("pair_inputs", "pair_outputs", "protocol", "audit", "key")
    elif schema == POSTFREEZE_VERSION:
        fields = ("candidate", "reference", "nodes", "gemma_scores", "protocol", "audit", "key")
    else:
        fields = ("source_panel", "protocol", "audit")
    for field in fields:
        reference = manifest[field]
        if reference is None:
            continue
        if sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ValueError(f"frozen calibration {field} changed")
    if schema == TIEBREAK_VERSION:
        for reference in manifest["source_votes"].values():
            if sha256_file(Path(reference["path"])) != reference["sha256"]:
                raise ValueError("frozen source vote file changed")
    payload_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reference in manifest["payloads"]:
        path = Path(reference["path"])
        if sha256_file(path) != reference["sha256"]:
            raise ValueError("frozen calibration payload changed")
        for row in _rows(path):
            pair_id = row.get("pair_id")
            if (set(row) != _BLIND_KEYS or not isinstance(pair_id, str) or not pair_id
                    or pair_id in seen):
                raise ValueError(f"invalid or non-blind frontier payload in {path}")
            seen.add(pair_id)
            payload_rows.append(row)
    audit = _rows(Path(manifest["audit"]["path"]))
    if ({row.get("pair_id") for row in audit} != seen
            or len(audit) != len(seen) or len(seen) != manifest.get("n_pairs")):
        raise ValueError("frozen payload coverage differs from audit")
    return manifest, payload_rows


def _parse_judge_response(raw: str, expected: list[str]) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("frontier response is not JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"decisions"}:
        raise ValueError("frontier response must contain only decisions")
    decisions = payload["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(expected):
        raise ValueError("frontier response decision count mismatch")
    by_id = {}
    for row in decisions:
        if (not isinstance(row, dict) or set(row) != {"pair_id", "score"}
                or not isinstance(row.get("pair_id"), str)
                or row["pair_id"] in by_id or type(row.get("score")) is not int
                or row["score"] not in (0, 1, 2)):
            raise ValueError("invalid strict frontier decision")
        by_id[row["pair_id"]] = row
    if set(by_id) != set(expected):
        raise ValueError("frontier response pair coverage mismatch")
    return [by_id[pair_id] for pair_id in expected]


def _judge_prompt(protocol: str) -> str:
    return (
        protocol.rstrip() + "\n\n"
        "Independently score every supplied pair. The input contains no model prediction. "
        "Return only one JSON object of the form "
        "{\"decisions\":[{\"pair_id\":\"...\",\"score\":0}]}. "
        "Each input pair_id must occur exactly once; score must be the integer 0, 1, or 2. "
        "Do not add reasoning, prose, markdown, or other fields."
    )


def _direct_completion(
    *, provider: str, model: str, messages: list[dict[str, str]], api_key: str,
    max_tokens: int, timeout: float, reasoning_effort: str | None,
) -> str:
    """Call an owned provider API while keeping the frozen judge contract unchanged."""
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_completion_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            response_format={"type": "json_object"},
        )
        return str(response.choices[0].message.content or "")
    if provider == "anthropic":
        from anthropic import Anthropic

        system = "\n\n".join(row["content"] for row in messages if row["role"] == "system")
        conversation = [row for row in messages if row["role"] != "system"]
        client = Anthropic(api_key=api_key, timeout=timeout, max_retries=0)
        response = client.messages.create(
            model=model, system=system, messages=conversation,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            output_config={
                "effort": reasoning_effort or "low",
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "decisions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "pair_id": {"type": "string"},
                                        "score": {"type": "integer", "enum": [0, 1, 2]},
                                    },
                                    "required": ["pair_id", "score"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["decisions"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        return "".join(
            str(block.text) for block in response.content if getattr(block, "type", None) == "text"
        )
    raise ValueError(f"unsupported direct provider: {provider}")


def run_judge(
    *, manifest_path: str | Path, output_dir: str | Path, model: str,
    api_key_file: str | Path, request_cap: int, max_tokens: int = 4000,
    timeout: float = 180.0, reasoning_effort: str | None = None,
    provider: str = "openrouter",
) -> dict[str, Any]:
    """Resume a capped provider pass over authenticated blind payload shards."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("an explicit judge model is required")
    if provider not in {"openrouter", "openai", "anthropic"}:
        raise ValueError("provider must be openrouter, openai, or anthropic")
    if request_cap < 1 or max_tokens < 1 or timeout <= 0:
        raise ValueError("request cap, max tokens, and timeout must be positive")
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest, _ = _validate_frozen_panel(manifest_file)
    protocol_ref = manifest["protocol"]
    protocol = Path(protocol_ref["path"]).read_text(encoding="utf-8")
    destination = Path(output_dir).expanduser().resolve()
    run_manifest_path = destination / "judge_manifest.json"
    frozen_run = {
        "schema_version": "frontier-openrouter-judge-run-v1",
        "source_manifest": {"path": str(manifest_file), "sha256": sha256_file(manifest_file)},
        "source_schema_version": manifest["schema_version"],
        "cell": manifest.get("cell") or {
            "task": manifest["task"], "level": manifest["level"],
            "protocol_sha256": manifest["protocol"]["sha256"],
        },
        "protocol": dict(protocol_ref),
        "provider": provider,
        "model": model,
    }
    if destination.exists():
        if not run_manifest_path.is_file():
            raise FileExistsError(f"resume directory lacks judge_manifest.json: {destination}")
        if json.loads(run_manifest_path.read_text(encoding="utf-8")) != frozen_run:
            raise ValueError("judge resume arguments differ from frozen run manifest")
    else:
        destination.mkdir(parents=True)
        run_manifest_path.write_text(json.dumps(frozen_run, indent=2, sort_keys=True) + "\n")
    raw_dir = destination / "raw"
    raw_dir.mkdir(exist_ok=True)
    execution_path = destination / "execution.json"
    if execution_path.is_file():
        previous = json.loads(execution_path.read_text(encoding="utf-8"))
        for filename, digest in previous.get("raw_transcript_sha256", {}).items():
            raw_path = raw_dir / filename
            if not raw_path.is_file() or sha256_file(raw_path) != digest:
                raise ValueError(f"persisted raw transcript changed: {raw_path}")
    key_path = Path(api_key_file).expanduser().resolve()
    api_key = key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"empty API key file: {key_path}")

    from scripts.tools.silver_match_v3.adjudicate_gemma_api import chat_completion

    votes: list[dict[str, Any]] = []
    requests_made = 0
    transcript_hashes = {}
    for index, reference in enumerate(manifest["payloads"]):
        batch = _rows(Path(reference["path"]))
        expected = [str(row["pair_id"]) for row in batch]
        raw_path = raw_dir / f"batch_{index:03d}.json"
        if raw_path.exists():
            raw = raw_path.read_text(encoding="utf-8").rstrip("\n")
        elif requests_made >= request_cap:
            break
        else:
            messages = [
                {"role": "system", "content": _judge_prompt(protocol)},
                {"role": "user", "content": json.dumps(batch, ensure_ascii=False,
                                                           separators=(",", ":"))},
            ]
            if provider == "openrouter":
                raw = chat_completion(
                    base_url="https://openrouter.ai/api/v1", model=model,
                    messages=messages,
                    max_tokens=max_tokens, seed=20260714 + index, timeout=timeout,
                    transport_retries=0, api_key=api_key,
                    reasoning_effort=reasoning_effort, reasoning_exclude=True,
                    force_json_object=True,
                )
            else:
                raw = _direct_completion(
                    provider=provider, model=model, messages=messages, api_key=api_key,
                    max_tokens=max_tokens, timeout=timeout, reasoning_effort=reasoning_effort,
                )
            # Validate before persisting: a malformed transcript must never count as a completed
            # shard or silently consume a semantic vote artifact.
            _parse_judge_response(raw, expected)
            raw_path.write_text(raw + ("" if raw.endswith("\n") else "\n"), encoding="utf-8")
            requests_made += 1
        votes.extend(_parse_judge_response(raw, expected))
        transcript_hashes[raw_path.name] = sha256_file(raw_path)

    votes_path = destination / "votes.jsonl"
    _write_jsonl(votes_path, votes)
    complete = len(votes) == manifest["n_pairs"]
    execution = {
        "schema_version": "frontier-openrouter-judge-execution-v1",
        "judge_manifest": {"path": str(run_manifest_path), "sha256": sha256_file(run_manifest_path)},
        "provider": provider,
        "model": model,
        "request_cap_this_run": request_cap,
        "requests_made_this_run": requests_made,
        "n_votes": len(votes),
        "n_expected": manifest["n_pairs"],
        "complete": complete,
        "votes": {"path": str(votes_path), "sha256": sha256_file(votes_path)},
        "raw_transcript_sha256": transcript_hashes,
        "protocol": dict(protocol_ref),
        "blindness": "only frozen blind payload shards were sent; no Gemma labels or probabilities",
    }
    execution_path.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n")
    return execution


def prepare(
    *,
    pair_inputs_path: str | Path,
    pair_outputs_path: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
    per_predicted_class: int = 150,
    per_shard: int = 50,
    seed: int = 20260714,
) -> dict[str, Any]:
    if per_predicted_class < 1 or per_shard < 1:
        raise ValueError("panel and shard sizes must be positive")
    inputs_path = Path(pair_inputs_path).expanduser().resolve()
    outputs_path = Path(pair_outputs_path).expanduser().resolve()
    protocol_file = Path(protocol_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    validation = validate_pair_files(inputs_path, outputs_path)
    protocol_file_sha = sha256_file(protocol_file)
    if protocol_file_sha != validation["protocol_sha256"]:
        raise ValueError(
            "frontier calibration protocol file differs from the protocol used for Gemma scoring")
    inputs = {row["pair_id"]: row for row in _rows(inputs_path)}
    outputs = {row["pair_id"]: row for row in _rows(outputs_path)}
    if set(inputs) != set(outputs):
        raise ValueError("pair inputs and outputs do not align")
    by_prediction = {label: [] for label in LABELS}
    for pair_id, output in outputs.items():
        by_prediction[output["prediction"]].append(pair_id)
    selected = []
    counts = {}
    for label, pair_ids in by_prediction.items():
        ranked = sorted(pair_ids, key=lambda pair_id: hashlib.sha256(
            f"{seed}|{validation['outputs_sha256']}|{label}|{pair_id}".encode()).hexdigest())
        chosen = ranked[: min(per_predicted_class, len(ranked))]
        counts[label] = {"population": len(pair_ids), "sample": len(chosen)}
        selected.extend(chosen)
    selected.sort(key=lambda pair_id: hashlib.sha256(f"blind|{seed}|{pair_id}".encode()).hexdigest())
    if not selected:
        raise ValueError("empty frontier calibration panel")

    destination.mkdir(parents=True)
    payload_dir = destination / "payloads"
    payload_dir.mkdir()
    blind = [{
        "pair_id": pair_id,
        "task": inputs[pair_id]["task"],
        "level": inputs[pair_id]["level"],
        "concept_a": inputs[pair_id]["text_a"],
        "concept_b": inputs[pair_id]["text_b"],
    } for pair_id in selected]
    audit = destination / "audit.jsonl"
    key = destination / "key.jsonl"
    _write_jsonl(audit, blind)
    _write_jsonl(key, [{"pair_id": pair_id, "node_a": inputs[pair_id]["node_a"],
                        "node_b": inputs[pair_id]["node_b"]} for pair_id in selected])
    payloads = []
    for start in range(0, len(blind), per_shard):
        path = payload_dir / f"calibration_{start // per_shard:03d}.jsonl"
        _write_jsonl(path, blind[start:start + per_shard])
        payloads.append({"path": str(path), "sha256": sha256_file(path)})
    manifest = {
        "schema_version": VERSION,
        "cell": {key: validation[key] for key in ("task", "level", "protocol_id")},
        "pair_inputs": {"path": str(inputs_path), "sha256": sha256_file(inputs_path)},
        "pair_outputs": {"path": str(outputs_path), "sha256": sha256_file(outputs_path)},
        "protocol": {"path": str(protocol_file), "sha256": protocol_file_sha},
        "audit": {"path": str(audit), "sha256": sha256_file(audit)},
        "key": {"path": str(key), "sha256": sha256_file(key)},
        "payloads": payloads,
        "predicted_class_strata": counts,
        "n_pairs": len(selected),
        "judge_policy": "independent Sonnet and GPT-5; third frontier judge only on disagreements",
        "blindness": "payload omits Gemma label, probabilities, confidence, nodes, and sampling stratum",
        "use": "development calibration only; exclude these node pairs from final post-freeze audit",
        "vote_schema": {"pair_id": "string", "score": "strict integer 0|1|2"},
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _votes(path: Path, expected: set[str], *, subset: bool = False) -> dict[str, int]:
    result = {}
    for row in _rows(path):
        pair_id, score = row.get("pair_id"), row.get("score")
        if (set(row) != {"pair_id", "score"} or pair_id not in expected or pair_id in result
                or type(score) is not int or score not in (0, 1, 2)):
            raise ValueError(f"invalid frontier vote in {path}")
        result[pair_id] = score
    if not subset and set(result) != expected:
        raise ValueError(f"incomplete frontier votes in {path}")
    return result


def prepare_disagreements(
    *, manifest_path: str | Path, votes_a_path: str | Path, votes_b_path: str | Path,
    output_dir: str | Path, per_shard: int = 50,
) -> dict[str, Any]:
    """Freeze blind payloads for exactly the disagreements between two complete judges."""
    if per_shard < 1:
        raise ValueError("per_shard must be positive")
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest, _ = _validate_frozen_panel(manifest_file)
    if manifest["schema_version"] not in {VERSION, POSTFREEZE_VERSION}:
        raise ValueError("disagreements must derive from a complete two-family panel")
    path_a = Path(votes_a_path).expanduser().resolve()
    path_b = Path(votes_b_path).expanduser().resolve()
    if path_a == path_b:
        raise ValueError("primary vote files must be distinct artifacts")
    audit_rows = _rows(Path(manifest["audit"]["path"]))
    expected = {str(row["pair_id"]) for row in audit_rows}
    a, b = _votes(path_a, expected), _votes(path_b, expected)
    disagreement_ids = {pair_id for pair_id in expected if a[pair_id] != b[pair_id]}
    if not disagreement_ids:
        raise ValueError("primary judges have no disagreements; no third pass is needed")
    selected = [row for row in audit_rows if row["pair_id"] in disagreement_ids]
    if len(selected) != len(disagreement_ids):
        raise ValueError("disagreement IDs do not exactly resolve to the frozen blind audit")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    payload_dir = destination / "payloads"
    payload_dir.mkdir()
    audit_path = destination / "audit.jsonl"
    _write_jsonl(audit_path, selected)
    payloads = []
    for start in range(0, len(selected), per_shard):
        path = payload_dir / f"tiebreak_{start // per_shard:03d}.jsonl"
        _write_jsonl(path, selected[start:start + per_shard])
        payloads.append({"path": str(path), "sha256": sha256_file(path)})
    result = {
        "schema_version": TIEBREAK_VERSION,
        "cell": manifest.get("cell") or {
            "task": manifest["task"], "level": manifest["level"],
            "protocol_sha256": manifest["protocol"]["sha256"],
        },
        "source_panel": {"path": str(manifest_file), "sha256": sha256_file(manifest_file)},
        "source_votes": {
            "judge_a": {"path": str(path_a), "sha256": sha256_file(path_a)},
            "judge_b": {"path": str(path_b), "sha256": sha256_file(path_b)},
        },
        "protocol": dict(manifest["protocol"]),
        "audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "payloads": payloads,
        "n_pairs": len(selected),
        "judge_policy": "third frontier judge only on exact primary-judge disagreements",
        "blindness": "payload copied from frozen blind audit; no Gemma labels or probabilities",
        "vote_schema": {"pair_id": "string", "score": "strict integer 0|1|2"},
    }
    result_path = destination / "manifest.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def assemble(
    *, manifest_path: str | Path, votes_a_path: str | Path, votes_b_path: str | Path,
    tiebreak_votes_path: str | Path | None, predictions_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("schema_version") != VERSION:
        raise ValueError("unsupported calibration manifest")
    for field in ("pair_inputs", "pair_outputs", "protocol", "audit", "key"):
        reference = manifest[field]
        if sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ValueError(f"frozen calibration {field} changed")
    for reference in manifest["payloads"]:
        if sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("frozen calibration payload changed")
    path_a = Path(votes_a_path).expanduser().resolve()
    path_b = Path(votes_b_path).expanduser().resolve()
    if path_a == path_b:
        raise ValueError("Sonnet and GPT-5 votes must be distinct artifacts")
    expected = {row["pair_id"] for row in _rows(Path(manifest["key"]["path"]))}
    a, b = _votes(path_a, expected), _votes(path_b, expected)
    disagreements = {pair_id for pair_id in expected if a[pair_id] != b[pair_id]}
    if disagreements:
        if not tiebreak_votes_path:
            raise ValueError(f"tiebreak required for {len(disagreements)} disagreements")
        tie_path = Path(tiebreak_votes_path).expanduser().resolve()
        if tie_path in {path_a, path_b}:
            raise ValueError("tiebreak votes must be a third artifact")
        tie = _votes(tie_path, expected, subset=True)
        if set(tie) != disagreements:
            raise ValueError("tiebreak votes must cover exactly the disagreements")
    else:
        tie = {}
        tie_path = None
        if tiebreak_votes_path and _rows(Path(tiebreak_votes_path)):
            raise ValueError("tiebreak votes supplied despite full agreement")

    outputs = {row["pair_id"]: row for row in _rows(Path(manifest["pair_outputs"]["path"]))}
    rows = []
    for pair_id in sorted(expected):
        output = outputs[pair_id]
        truth = a[pair_id] if a[pair_id] == b[pair_id] else tie[pair_id]
        rows.append({
            "example_id": pair_id,
            "task": output["task"], "level": output["level"],
            "protocol_id": output["protocol_id"], "split": "frontier_dev",
            "truth": truth, "prediction": LABELS.index(output["prediction"]),
            "target_probs": [float(index == truth) for index in range(3)],
            "probabilities": [float(output["probabilities"][label]) for label in LABELS],
            "view_probabilities": [
                [float(output["order_views"][view]["probabilities"][label]) for label in LABELS]
                for view in ("ab", "ba")
            ],
            "order_consistent": output["order_consistent"],
            "teacher_families": ["gpt5", "sonnet"],
        })
    destination = Path(predictions_path).expanduser().resolve()
    report_destination = Path(report_path).expanduser().resolve()
    if destination.exists() or report_destination.exists():
        raise FileExistsError(destination if destination.exists() else report_destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination, rows)
    report = {
        "schema_version": "frontier-similarity-calibration-assembly-v1",
        "manifest": {"path": str(manifest_file), "sha256": sha256_file(manifest_file)},
        "votes": {
            "sonnet": {"path": str(path_a), "sha256": sha256_file(path_a)},
            "gpt5": {"path": str(path_b), "sha256": sha256_file(path_b)},
            "tiebreak": ({"path": str(tie_path), "sha256": sha256_file(tie_path)} if tie_path else None),
        },
        "n_pairs": len(rows), "n_disagreements": len(disagreements),
        "predictions": {"path": str(destination), "sha256": sha256_file(destination)},
        "semantic_truth": "Sonnet/GPT-5 independent labels with third-pass disagreement resolution",
    }
    report_destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--pair-inputs", required=True); prep.add_argument("--pair-outputs", required=True)
    prep.add_argument("--protocol", required=True); prep.add_argument("--output-dir", required=True)
    prep.add_argument("--per-predicted-class", type=int, default=150)
    prep.add_argument("--per-shard", type=int, default=50); prep.add_argument("--seed", type=int, default=20260714)
    judge = sub.add_parser("judge")
    judge.add_argument("--manifest", required=True); judge.add_argument("--output-dir", required=True)
    judge.add_argument("--model", required=True); judge.add_argument("--api-key-file", required=True)
    judge.add_argument("--provider", choices=("openrouter", "openai", "anthropic"),
                       default="openrouter")
    judge.add_argument("--request-cap", required=True, type=int)
    judge.add_argument("--max-tokens", type=int, default=4000)
    judge.add_argument("--timeout", type=float, default=180.0)
    judge.add_argument("--reasoning-effort", choices=("minimal", "low", "medium"))
    disagree = sub.add_parser("prepare-disagreements")
    disagree.add_argument("--manifest", required=True); disagree.add_argument("--votes-a", required=True)
    disagree.add_argument("--votes-b", required=True); disagree.add_argument("--output-dir", required=True)
    disagree.add_argument("--per-shard", type=int, default=50)
    asm = sub.add_parser("assemble")
    asm.add_argument("--manifest", required=True); asm.add_argument("--votes-a", required=True)
    asm.add_argument("--votes-b", required=True); asm.add_argument("--tiebreak-votes")
    asm.add_argument("--predictions", required=True); asm.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(pair_inputs_path=args.pair_inputs, pair_outputs_path=args.pair_outputs,
                         protocol_path=args.protocol, output_dir=args.output_dir,
                         per_predicted_class=args.per_predicted_class, per_shard=args.per_shard,
                         seed=args.seed)
    elif args.command == "judge":
        result = run_judge(
            manifest_path=args.manifest, output_dir=args.output_dir, model=args.model,
            api_key_file=args.api_key_file, request_cap=args.request_cap,
            max_tokens=args.max_tokens, timeout=args.timeout,
            reasoning_effort=args.reasoning_effort, provider=args.provider,
        )
    elif args.command == "prepare-disagreements":
        result = prepare_disagreements(
            manifest_path=args.manifest, votes_a_path=args.votes_a,
            votes_b_path=args.votes_b, output_dir=args.output_dir,
            per_shard=args.per_shard,
        )
    else:
        result = assemble(manifest_path=args.manifest, votes_a_path=args.votes_a,
                          votes_b_path=args.votes_b, tiebreak_votes_path=args.tiebreak_votes,
                          predictions_path=args.predictions, report_path=args.report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
