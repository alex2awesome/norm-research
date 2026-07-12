#!/usr/bin/env python3
"""Prepare and freeze real L-channel extractions for contract probes.

This is an unsupervised adapter, not a label producer. It reads only the contract's
synthetic probe texts and the candidate's declared ``LLM_FIELDS`` instructions. Each
positive and negative text is extracted independently; the prompt contains no polarity,
judge score, target label, or paired comparison. The resulting artifact is bound to the
canonical contract, exact candidate bytes, extractor configuration, and probe text hashes.

Typical use::

  python build_probe_extractions_v2.py prepare --contract ... --candidate ... \
      --backend zai_anthropic --model glm-4.7 --out-dir /tmp/probe_l
  python api_field_runner.py --backend zai_anthropic --model glm-4.7 \
      --prompts /tmp/probe_l/prompts.jsonl --out /tmp/probe_l/results.jsonl
  python build_probe_extractions_v2.py finalize --contract ... --candidate ... \
      --backend zai_anthropic --model glm-4.7 --out-dir /tmp/probe_l
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .contract_check_isomorphic import (
        SCHEMA_VERSION,
        canonical_json_sha256,
        text_sha256,
        validate_contract,
    )
except ImportError:  # direct-file execution
    from contract_check_isomorphic import (  # type: ignore[no-redef]
        SCHEMA_VERSION,
        canonical_json_sha256,
        text_sha256,
        validate_contract,
    )


PROMPT_TEMPLATE_VERSION = "metric-seam-independent-field-extraction-v1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_llm_fields(candidate: Path) -> dict[str, str]:
    """Read a literal LLM_FIELDS mapping without importing candidate code."""

    tree = ast.parse(candidate.read_text(), filename=str(candidate))
    value: Any = None
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == "LLM_FIELDS" for target in targets):
            value = ast.literal_eval(node.value)
            break
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{candidate} needs a non-empty literal LLM_FIELDS dict")
    fields: dict[str, str] = {}
    for key, instruction in value.items():
        if not isinstance(key, str) or not key or not isinstance(instruction, str) or not instruction:
            raise ValueError("LLM_FIELDS must map non-empty string keys to instructions")
        fields[key] = instruction
    return fields


def extractor_manifest(
    *,
    contract: Mapping[str, Any],
    candidate: Path,
    backend: str,
    model: str,
    fields: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "metric-seam-probe-extractor-manifest-v1",
        "external_supervision": "none",
        "objective": "prompt_articulability_field_extraction",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "pairing_visible_to_extractor": False,
        "polarity_visible_to_extractor": False,
        "labels_or_reference_visible_to_extractor": False,
        "backend": backend,
        "model": model,
        "temperature": 0.0,
        "candidate_sha256": file_sha256(candidate),
        "contract_sha256": canonical_json_sha256(contract),
        "llm_fields": dict(fields),
    }


def render_prompt(text: str, fields: Mapping[str, str]) -> str:
    requested = "\n".join(f'- "{key}": {instruction}' for key, instruction in fields.items())
    example = json.dumps({key: "<answer>" for key in fields}, ensure_ascii=False)
    return (
        "Extract the requested fields from the document. Treat each field independently and "
        "follow its answer constraints. Return exactly one JSON object, with exactly the keys "
        "listed below and string values. Do not add markdown or explanation.\n\n"
        f"Fields:\n{requested}\n\nRequired shape: {example}\n\n"
        f"<document>\n{text}\n</document>"
    )


def prepare(
    contract_path: Path,
    candidate: Path,
    *,
    backend: str,
    model: str,
    out_dir: Path,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    validate_contract(contract)
    fields = load_llm_fields(candidate)
    manifest = extractor_manifest(
        contract=contract,
        candidate=candidate,
        backend=backend,
        model=model,
        fields=fields,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extractor_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    rows = []
    for index, probe in enumerate(contract["cf_probes"]):
        if probe["channel"] != "L":
            continue
        for side in ("pos", "neg"):
            text = probe[f"text_{side}"]
            rows.append(
                {
                    "channel": "contract_l_field",
                    "aspect_id": f"contract_probe_{index}",
                    "datapoint_id": side,
                    "prompt": render_prompt(text, fields),
                }
            )
    with (out_dir / "prompts.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "n_prompts": len(rows),
        "n_l_probes": len(rows) // 2,
        "extractor_manifest_sha256": canonical_json_sha256(manifest),
        "contract_sha256": canonical_json_sha256(contract),
        "prompts_file_sha256": file_sha256(out_dir / "prompts.jsonl"),
    }
    (out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def parse_response(raw: str, fields: Mapping[str, str]) -> dict[str, str]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()
    left, right = cleaned.find("{"), cleaned.rfind("}")
    if left < 0 or right < left:
        raise ValueError("response contains no JSON object")
    value = json.loads(cleaned[left : right + 1])
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("response keys do not exactly match LLM_FIELDS")
    if not all(isinstance(item, str) for item in value.values()):
        raise ValueError("all extraction values must be strings")
    return {key: value[key] for key in fields}


def finalize(
    contract_path: Path,
    candidate: Path,
    *,
    backend: str,
    model: str,
    out_dir: Path,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    validate_contract(contract)
    fields = load_llm_fields(candidate)
    expected_manifest = extractor_manifest(
        contract=contract,
        candidate=candidate,
        backend=backend,
        model=model,
        fields=fields,
    )
    manifest = json.loads((out_dir / "extractor_manifest.json").read_text())
    if manifest != expected_manifest:
        raise ValueError("extractor manifest no longer matches inputs/configuration")

    result_rows: dict[tuple[int, str], Mapping[str, Any]] = {}
    for line in (out_dir / "results.jsonl").read_text().splitlines():
        row = json.loads(line)
        prefix = "contract_probe_"
        aspect = row.get("aspect_id", "")
        side = row.get("datapoint_id")
        if not isinstance(aspect, str) or not aspect.startswith(prefix) or side not in {"pos", "neg"}:
            raise ValueError("unexpected result identity")
        key = (int(aspect[len(prefix) :]), side)
        if key in result_rows:
            raise ValueError(f"duplicate result {key}")
        result_rows[key] = row

    frozen = []
    for index, probe in enumerate(contract["cf_probes"]):
        if probe["channel"] != "L":
            continue
        base = {
            "index": index,
            "text_pos_sha256": text_sha256(probe["text_pos"]),
            "text_neg_sha256": text_sha256(probe["text_neg"]),
        }
        try:
            pos = parse_response(str(result_rows[(index, "pos")]["raw"]), fields)
            neg = parse_response(str(result_rows[(index, "neg")]["raw"]), fields)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            frozen.append(
                {
                    **base,
                    "available": False,
                    "unavailable_reason": f"structured extraction unavailable: {type(exc).__name__}",
                }
            )
        else:
            frozen.append({**base, "available": True, "pos": pos, "neg": neg})

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": canonical_json_sha256(contract),
        "extractor_manifest_sha256": canonical_json_sha256(manifest),
        "probes": frozen,
    }
    digest = canonical_json_sha256(payload)
    (out_dir / "probe_extractions.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    (out_dir / "probe_extractions.sha256").write_text(digest + "\n")
    summary = {
        "probe_extractions_sha256": digest,
        "n_l_probes": len(frozen),
        "n_available": sum(row["available"] for row in frozen),
        "n_abstained": sum(not row["available"] for row in frozen),
    }
    (out_dir / "finalize_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "finalize"):
        current = sub.add_parser(command)
        current.add_argument("--contract", type=Path, required=True)
        current.add_argument("--candidate", type=Path, required=True)
        current.add_argument("--backend", required=True)
        current.add_argument("--model", required=True)
        current.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    kwargs = {
        "backend": args.backend,
        "model": args.model,
        "out_dir": args.out_dir,
    }
    result = (
        prepare(args.contract, args.candidate, **kwargs)
        if args.command == "prepare"
        else finalize(args.contract, args.candidate, **kwargs)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
