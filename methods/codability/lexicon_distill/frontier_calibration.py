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
    else:
        result = assemble(manifest_path=args.manifest, votes_a_path=args.votes_a,
                          votes_b_path=args.votes_b, tiebreak_votes_path=args.tiebreak_votes,
                          predictions_path=args.predictions, report_path=args.report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
