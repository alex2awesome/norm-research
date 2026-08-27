#!/usr/bin/env python3
"""Prepare and ingest the additive source-addressed science prompt transport.

The v6 prompt asked a model to copy full claim and evidence strings.  That made
serialization fidelity part of the measured prompt capability.  This additive v7
instrument instead gives every abstract/body sentence a deterministic source address.
The model selects addresses and typed relation data; code validates the addresses and
hydrates exact source spans after transport.

This module has no model client and preparation makes no API or GPU call.  The only
source fields admitted are ``paper_id``, ``abstract``, and ``body``.  In particular,
dataset labels and code-verifier outputs are neither read nor accepted as inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "datasets/peer-review/peer_review_cv_evidence.jsonl"
DEFAULT_SPEC = Path(__file__).with_name("articulability_addressed_prompt.json")
DEFAULT_MODEL = Path(__file__).with_name(
    "articulability_model_glm47_openrouter_addressed_reasoning_off.json"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/metric_seam_pilot/science_articulability_v7_source_addressed_prepared"
)

REQUEST_SCHEMA = "science-articulability-addressed-request-v1"
RESULT_SCHEMA = "science-articulability-addressed-bound-result-v1"
NORMALIZED_SCHEMA = "science-articulability-addressed-normalized-result-v1"
MANIFEST_SCHEMA = "science-articulability-addressed-bundle-v1"
SOURCE_MAP_SCHEMA = "science-articulability-addressed-source-map-v1"
SEGMENTATION_SCHEMA = "science-articulability-segmentation-contract-v1"

ALLOWED_DECISIONS = {"supported", "contradicted", "evidence_link", "insufficient"}
ALLOWED_RELATIONS = {
    "comparative", "numeric", "theoretical", "empirical", "qualitative"
}
ALLOWED_QUANTITY_STATES = {"aligned", "mismatch", "missing", "not_required"}
ALLOWED_COMPARISON_STATES = {
    "aligned", "reversed_roles", "direction_mismatch", "baseline_mismatch",
    "missing", "not_required",
}
ALLOWED_EVIDENCE_KINDS = {
    "numeric_relation", "comparative_relation", "theory_marker",
    "empirical_artifact", "qualitative_link", "none",
}
_RESPONSE_KEYS = {"paper_id", "selections"}
_SELECTION_KEYS = {
    "claim_sentence_id", "evidence_sentence_id", "decision", "relation",
    "quantity_state", "comparison_state", "evidence_kind", "quantity_count",
    "comparison_present",
}

_ABBREVIATIONS = {
    "e.g.", "i.e.", "et al.", "fig.", "figs.", "eq.", "eqs.", "sec.",
    "secs.", "ref.", "refs.", "vs.", "dr.", "prof.", "no.", "approx.",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _allowed_projection(raw: dict[str, Any], line_number: int) -> dict[str, str]:
    """Create a fresh allowlisted mapping without ever requesting a label field."""
    return {
        "paper_id": str(raw.get("paper_id") or f"line_{line_number}"),
        "abstract": str(raw.get("abstract") or ""),
        "body": str(raw.get("body") or ""),
    }


def load_inputs(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                records.append(_allowed_projection(json.loads(line), line_number))
    ids = [row["paper_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("paper_id values must be unique")
    return records


def _sentence_boundary(text: str, start: int, index: int) -> tuple[int, int] | None:
    """Return ``(sentence_end, next_start)`` after punctuation, or ``None``.

    This is an explicitly frozen, deliberately small prose segmenter.  It protects
    decimal points and a fixed abbreviation list and only splits before a plausible
    sentence-initial character.  It does not call a learned tokenizer.
    """
    char = text[index]
    if char not in ".!?":
        return None
    if (
        char == "."
        and index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    ):
        return None
    prefix = text[max(start, index - 10): index + 1].lower().strip()
    if any(prefix.endswith(abbreviation) for abbreviation in _ABBREVIATIONS):
        return None
    next_index = index + 1
    while next_index < len(text) and text[next_index] in "\"')]}":
        next_index += 1
    sentence_end = next_index
    if next_index < len(text) and not text[next_index].isspace():
        return None
    while next_index < len(text) and text[next_index].isspace():
        next_index += 1
    if (
        next_index < len(text)
        and not (
            text[next_index].isupper()
            or text[next_index].isdigit()
            or text[next_index] in "([\"'“‘"
        )
    ):
        return None
    return sentence_end, next_index


def segment_source(text: str, *, section: str) -> list[dict[str, Any]]:
    """Return stable addresses retaining exact source spans and offsets.

    The returned spans cover every non-whitespace source character exactly once.
    Whitespace between addressed sentences is not included in either neighboring span;
    the original raw source is retained separately in each bound request.
    """
    if section not in {"abstract", "body"}:
        raise ValueError("section must be abstract or body")
    text = str(text or "")
    prefix = "A" if section == "abstract" else "B"
    raw_spans: list[tuple[int, int]] = []
    start = 0
    for index in range(len(text)):
        boundary = _sentence_boundary(text, start, index)
        if boundary is not None:
            sentence_end, next_start = boundary
            raw_spans.append((start, sentence_end))
            start = next_start
    if start < len(text):
        raw_spans.append((start, len(text)))

    spans: list[tuple[int, int]] = []
    for raw_start, raw_end in raw_spans:
        while raw_start < raw_end and text[raw_start].isspace():
            raw_start += 1
        while raw_end > raw_start and text[raw_end - 1].isspace():
            raw_end -= 1
        if raw_start < raw_end:
            spans.append((raw_start, raw_end))

    cursor = 0
    for span_start, span_end in spans:
        if text[cursor:span_start].strip():
            raise AssertionError("segmenter omitted non-whitespace source content")
        cursor = span_end
    if text[cursor:].strip():
        raise AssertionError("segmenter omitted trailing non-whitespace source content")

    addressed: list[dict[str, Any]] = []
    for index, (span_start, span_end) in enumerate(spans):
        exact_text = text[span_start:span_end]
        addressed.append({
            "sentence_id": f"{prefix}{index + 1:04d}",
            "section": section,
            "sentence_index": index,
            "start": span_start,
            "end": span_end,
            "text": exact_text,
            "text_sha256": hash_value(exact_text),
        })
    return addressed


def segmentation_contract() -> dict[str, Any]:
    algorithm_source = "\n\n".join((
        repr(sorted(_ABBREVIATIONS)),
        inspect.getsource(_sentence_boundary),
        inspect.getsource(segment_source),
    ))
    return {
        "schema_version": SEGMENTATION_SCHEMA,
        "algorithm_id": "punctuation_decimal_abbreviation_exact_span_v1",
        "implementation_path": display_path(Path(__file__)),
        "implementation_file_sha256": hash_file(Path(__file__)),
        "algorithm_source_sha256": hashlib.sha256(
            algorithm_source.encode("utf-8")
        ).hexdigest(),
        "address_namespaces": {"abstract": "A", "body": "B"},
        "index_origin": 0,
        "display_id_origin": 1,
        "coverage_invariant": "every_nonwhitespace_source_character_exactly_once",
        "span_text_policy": "exact_source_substring_no_normalization",
        "prompt_layout_policy": (
            "source-order JSONL; exact span text JSON-escaped; inter-sentence whitespace omitted"
        ),
    }


def build_source_map(paper: dict[str, str]) -> dict[str, Any]:
    if set(paper) != {"paper_id", "abstract", "body"}:
        raise ValueError("paper input must contain exactly the allowlisted fields")
    return {
        "schema_version": SOURCE_MAP_SCHEMA,
        "paper_id": paper["paper_id"],
        "abstract": segment_source(paper["abstract"], section="abstract"),
        "body": segment_source(paper["body"], section="body"),
    }


def _validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("input_allowlist") != ["paper_id", "abstract", "body"]:
        raise ValueError("input allowlist must be exactly paper_id+abstract+body")
    if spec.get("external_knowledge") != "forbidden":
        raise ValueError("external knowledge must remain forbidden")
    if spec.get("max_claims") != 5:
        raise ValueError("max_claims must remain five")
    for key in (
        "system_prompt", "decision_semantics", "typed_relation_semantics", "output_schema"
    ):
        if not spec.get(key):
            raise ValueError(f"prompt specification missing {key}")


def _validate_model(model: dict[str, Any]) -> None:
    required = (
        "schema_version", "backend", "protocol", "model", "temperature",
        "max_output_tokens", "system_prompt_transport", "response_transport",
        "execution_status",
    )
    missing = [key for key in required if key not in model]
    if missing:
        raise ValueError(f"model manifest missing {missing}")
    if not model["model"] or model["temperature"] != 0.0:
        raise ValueError("model identity must be concrete and temperature must be zero")
    if model["execution_status"] != "prepared_not_run":
        raise ValueError("addressed model manifest must remain prepared_not_run")


def render_system_prompt(spec: dict[str, Any]) -> str:
    return (
        spec["system_prompt"]
        + "\n\nFROZEN_DECISION_SEMANTICS:\n"
        + json.dumps(spec["decision_semantics"], ensure_ascii=False, sort_keys=True)
        + "\n\nFROZEN_TYPED_RELATION_SEMANTICS:\n"
        + json.dumps(spec["typed_relation_semantics"], ensure_ascii=False, sort_keys=True)
        + "\n\nFROZEN_OUTPUT_SCHEMA:\n"
        + json.dumps(spec["output_schema"], ensure_ascii=False, sort_keys=True)
        + "\n\nTRANSPORT_GUARD: Return addresses and enumerated relation data only. "
        "Do not add source text, quotations, summaries, offsets, counts, status, or graph fields."
    )


def _address_lines(spans: list[dict[str, Any]]) -> str:
    return "\n".join(
        json.dumps(
            {"sentence_id": span["sentence_id"], "text": span["text"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for span in spans
    ) or "(none)"


def render_user_prompt(paper_id: str, source_map: dict[str, Any]) -> str:
    return (
        "PAPER_ID_JSON: " + json.dumps(paper_id, ensure_ascii=False)
        + "\n\nADDRESSED_ABSTRACT_SENTENCES_JSONL:\n"
        + _address_lines(source_map["abstract"])
        + "\n\nADDRESSED_BODY_SENTENCES_JSONL:\n"
        + _address_lines(source_map["body"])
    )


_REQUEST_MATERIAL_KEYS = (
    "paper_input", "paper_input_sha256", "segmentation_contract_sha256",
    "source_map", "source_map_sha256", "prompt_spec_sha256",
    "model_manifest_sha256", "system_prompt", "user_prompt",
)


def build_requests(
    records: Iterable[dict[str, str]], spec: dict[str, Any], model: dict[str, Any]
) -> list[dict[str, Any]]:
    _validate_spec(spec)
    _validate_model(model)
    spec_sha = hash_value(spec)
    model_sha = hash_value(model)
    segmenter_sha = hash_value(segmentation_contract())
    system_prompt = render_system_prompt(spec)
    requests: list[dict[str, Any]] = []
    for sequence_index, paper in enumerate(records):
        source_map = build_source_map(paper)
        material = {
            "paper_input": paper,
            "paper_input_sha256": hash_value(paper),
            "segmentation_contract_sha256": segmenter_sha,
            "source_map": source_map,
            "source_map_sha256": hash_value(source_map),
            "prompt_spec_sha256": spec_sha,
            "model_manifest_sha256": model_sha,
            "system_prompt": system_prompt,
            "user_prompt": render_user_prompt(paper["paper_id"], source_map),
        }
        request_sha = hash_value(material)
        requests.append({
            "schema_version": REQUEST_SCHEMA,
            "request_id": f"science_addressed_{sequence_index:04d}_{request_sha[:16]}",
            "sequence_index": sequence_index,
            "paper_id": paper["paper_id"],
            **material,
            "request_sha256": request_sha,
        })
    return requests


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str = "w") -> None:
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _numbered_lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            yield line_number, line


def prepare(
    input_path: Path, spec_path: Path, model_path: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty bundle: {output_dir}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    records = load_inputs(input_path)
    requests = build_requests(records, spec, model)

    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "requests.jsonl"
    _write_jsonl(requests_path, requests)
    segmenter = segmentation_contract()
    prompt_lengths = [len(row["system_prompt"]) + len(row["user_prompt"]) for row in requests]
    abstract_counts = [len(row["source_map"]["abstract"]) for row in requests]
    body_counts = [len(row["source_map"]["body"]) for row in requests]
    corpus_sha = hash_value([hash_value(record) for record in records])
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_run_no_api_calls",
        "objective": "unsupervised_prompt_articulability_same_input_source_addressed",
        "external_supervision": "none",
        "external_knowledge": "forbidden",
        "gpu_used": False,
        "api_calls_made_by_prepare": 0,
        "label_policy": {
            "loader_allowlist": ["paper_id", "abstract", "body"],
            "note": (
                "The loader projects to a fresh allowlisted mapping. Labels, external "
                "anchors, code-verifier results, and acceptance outcomes are never requested."
            ),
        },
        "input": {
            "path": display_path(input_path),
            "source_file_sha256": hash_file(input_path),
            "allowlisted_corpus_sha256": corpus_sha,
            "record_count": len(records),
        },
        "segmentation_contract": {
            "canonical_sha256": hash_value(segmenter),
            "identity": segmenter,
        },
        "prompt_spec": {
            "path": display_path(spec_path),
            "file_sha256": hash_file(spec_path),
            "canonical_sha256": hash_value(spec),
            "identity": spec,
        },
        "model_manifest": {
            "path": display_path(model_path),
            "file_sha256": hash_file(model_path),
            "canonical_sha256": hash_value(model),
            "identity": model,
        },
        "implementation": {"sha256": hash_file(Path(__file__))},
        "requests": {
            "path": "requests.jsonl",
            "sha256": hash_file(requests_path),
            "count": len(requests),
            "future_api_call_count": len(requests),
            "min_prompt_characters": min(prompt_lengths, default=0),
            "max_prompt_characters": max(prompt_lengths, default=0),
        },
        "address_statistics": {
            "abstract_sentences_total": sum(abstract_counts),
            "body_sentences_total": sum(body_counts),
            "min_abstract_sentences": min(abstract_counts, default=0),
            "max_abstract_sentences": max(abstract_counts, default=0),
            "min_body_sentences": min(body_counts, default=0),
            "max_body_sentences": max(body_counts, default=0),
        },
        "result_contract": {
            "schema_version": RESULT_SCHEMA,
            "required_binding_fields": [
                "schema_version", "request_id", "request_sha256",
                "model_manifest_sha256", "bundle_manifest_sha256", "response",
            ],
            "response": "exactly paper_id+selections; no source text fields",
            "resume_key": "request_id",
        },
        "seam_placement": {
            "code_side": [
                "source segmentation", "address namespace/range validation",
                "duplicate-address rejection", "binding validation", "exact span hydration",
                "derived counts/status/graph bookkeeping", "typed-schema coherence",
            ],
            "prompt_side": [
                "result-bearing claim selection", "evidence selection",
                "relation type", "support/contradiction/evidence-link judgment",
                "typed quantity/comparison/evidence-state judgment",
            ],
        },
        "representation_isomorphism": {
            "preserved": (
                "The only evidence content is the same allowlisted paper_id+abstract+body. "
                "Every non-whitespace source character occurs in exactly one addressed span, "
                "and hydration restores the exact bound source substring and offsets."
            ),
            "caveat": (
                "The prompt sees sentence-addressed JSONL rather than the v6 continuous-text "
                "layout: inter-sentence whitespace is omitted and JSON escaping/IDs are added. "
                "This is information-preserving for non-whitespace text but not presentation-"
                "identical, so v6-v7 differences mix transport and representation effects."
            ),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# Science prompt articulability v7: source-addressed preparation

Status: **prepared, not run**. Preparation made **0 API calls** and used **no GPU**.
The bundle contains {len(requests):,} bound requests. A complete future execution would
require {len(requests):,} independent remote calls.

The model is shown the same allowlisted `paper_id + abstract + body` evidence content as
the full-paper verifier, segmented into deterministic `A####` and `B####` addresses. It
returns addresses and typed semantic judgments only. Ingestion validates bindings,
namespaces, ranges, uniqueness, and schema coherence, then hydrates exact source text and
offsets in code. No acceptance label, supervised anchor, external source, or code-verifier
output is present.

The seam deliberately moves address verification and text reproduction code-side while
leaving claim selection, evidence selection, and relation judgment prompt-side. This
removes copied-text/count serialization from the prompt capability being measured.

Representation caveat: the evidence content is preserved, but the model sees sentence-
addressed JSONL rather than continuous paper layout. Inter-sentence whitespace is omitted
and IDs/JSON escaping are added, so a future v6-v7 comparison is a transport intervention,
not a presentation-identical replicate.

No execution command is included in this prepared-only checkpoint. A future executor must
obey the bound result contract in `manifest.json`; ingestion is available with:

```bash
python -m methods.metric_seam.science_claims_v2.addressed_pipeline ingest \\
  --bundle {display_path(output_dir)} --raw-results /path/to/bound_results.jsonl
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def verify_bundle(bundle: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported addressed bundle manifest")
    if manifest.get("status") != "prepared_not_run_no_api_calls":
        raise ValueError("bundle is not the prepared-only v7 contract")

    current_segmenter = segmentation_contract()
    frozen_segmenter = manifest.get("segmentation_contract") or {}
    if frozen_segmenter.get("canonical_sha256") != hash_value(current_segmenter):
        raise ValueError("segmentation implementation/contract hash mismatch")
    if frozen_segmenter.get("identity") != current_segmenter:
        raise ValueError("segmentation contract identity mismatch")
    if manifest.get("implementation", {}).get("sha256") != hash_file(Path(__file__)):
        raise ValueError("addressed pipeline implementation hash mismatch")

    spec = manifest.get("prompt_spec", {}).get("identity")
    model = manifest.get("model_manifest", {}).get("identity")
    if not isinstance(spec, dict) or hash_value(spec) != manifest["prompt_spec"]["canonical_sha256"]:
        raise ValueError("prompt specification binding mismatch")
    if not isinstance(model, dict) or hash_value(model) != manifest["model_manifest"]["canonical_sha256"]:
        raise ValueError("model manifest binding mismatch")
    _validate_spec(spec)
    _validate_model(model)

    requests_path = bundle / manifest["requests"]["path"]
    if hash_file(requests_path) != manifest["requests"]["sha256"]:
        raise ValueError("requests file hash mismatch")
    rows = _read_jsonl(requests_path)
    if len(rows) != manifest["requests"]["count"]:
        raise ValueError("request count mismatch")

    by_id: dict[str, dict[str, Any]] = {}
    seen_sequences: set[int] = set()
    for row in rows:
        if row.get("schema_version") != REQUEST_SCHEMA:
            raise ValueError("unsupported request schema")
        paper = row.get("paper_input")
        if not isinstance(paper, dict) or set(paper) != {"paper_id", "abstract", "body"}:
            raise ValueError("request paper input violates allowlist")
        if paper["paper_id"] != row.get("paper_id"):
            raise ValueError("request paper_id does not match bound paper input")
        if hash_value(paper) != row.get("paper_input_sha256"):
            raise ValueError("request paper input hash mismatch")
        regenerated_map = build_source_map(paper)
        if regenerated_map != row.get("source_map"):
            raise ValueError("request source map is not deterministic from bound source")
        if hash_value(regenerated_map) != row.get("source_map_sha256"):
            raise ValueError("request source map hash mismatch")
        if row.get("segmentation_contract_sha256") != frozen_segmenter["canonical_sha256"]:
            raise ValueError("request segmentation binding mismatch")
        if row.get("prompt_spec_sha256") != manifest["prompt_spec"]["canonical_sha256"]:
            raise ValueError("request prompt specification binding mismatch")
        if row.get("model_manifest_sha256") != manifest["model_manifest"]["canonical_sha256"]:
            raise ValueError("request model manifest binding mismatch")
        if row.get("system_prompt") != render_system_prompt(spec):
            raise ValueError("request system prompt rendering mismatch")
        if row.get("user_prompt") != render_user_prompt(paper["paper_id"], regenerated_map):
            raise ValueError("request user prompt rendering mismatch")
        material = {key: row.get(key) for key in _REQUEST_MATERIAL_KEYS}
        request_sha = hash_value(material)
        if request_sha != row.get("request_sha256"):
            raise ValueError("request material hash mismatch")
        sequence = row.get("sequence_index")
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence < 0
        ):
            raise ValueError("request sequence_index must be a nonnegative integer")
        expected_id = f"science_addressed_{sequence:04d}_{request_sha[:16]}"
        if row.get("request_id") != expected_id:
            raise ValueError("request id is not derived from sequence and material")
        if sequence in seen_sequences:
            raise ValueError("duplicate request sequence")
        seen_sequences.add(sequence)
        if expected_id in by_id:
            raise ValueError("duplicate request id")
        by_id[expected_id] = row
    if seen_sequences != set(range(len(rows))):
        raise ValueError("request sequence_index values must be contiguous from zero")
    return manifest, by_id


def _extract_json(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if not isinstance(response, str):
        raise ValueError("response must be an object or exact JSON string")
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not exactly one JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def _address_index(request_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_map = request_row["source_map"]
    abstract = {span["sentence_id"]: span for span in source_map["abstract"]}
    body = {span["sentence_id"]: span for span in source_map["body"]}
    return abstract, body


def _validate_selection_coherence(selection: dict[str, Any]) -> None:
    decision = selection["decision"]
    relation = selection["relation"]
    quantity_state = selection["quantity_state"]
    comparison_state = selection["comparison_state"]
    evidence_kind = selection["evidence_kind"]
    quantity_count = selection["quantity_count"]
    comparison_present = selection["comparison_present"]
    evidence_id = selection["evidence_sentence_id"]

    if decision in {"supported", "contradicted", "evidence_link"} and evidence_id is None:
        raise ValueError(f"{decision} requires a body evidence address")
    if decision in {"supported", "contradicted"} and relation not in {"numeric", "comparative"}:
        raise ValueError("only numeric/comparative relations may be certificates")
    if decision == "supported" and relation == "numeric":
        if (
            quantity_state != "aligned"
            or evidence_kind != "numeric_relation"
            or quantity_count < 1
        ):
            raise ValueError(
                "supported numeric relation requires at least one aligned quantity"
            )
    if decision == "supported" and relation == "comparative":
        if (
            comparison_state != "aligned"
            or evidence_kind != "comparative_relation"
            or not comparison_present
        ):
            raise ValueError("supported comparison requires aligned comparative evidence")
        if quantity_state not in {"aligned", "not_required"}:
            raise ValueError("supported comparison has incoherent quantity state")
    if decision == "contradicted":
        if relation != "comparative":
            raise ValueError("contradiction requires a comparative relation")
        if comparison_state not in {"reversed_roles", "direction_mismatch"}:
            raise ValueError("contradiction requires reversed roles or direction")
        if evidence_kind != "comparative_relation":
            raise ValueError("contradiction requires comparative evidence")
        if not comparison_present:
            raise ValueError("contradiction requires an explicit directed comparison")
    if decision == "evidence_link" and evidence_kind not in {
        "theory_marker", "empirical_artifact", "qualitative_link"
    }:
        raise ValueError("evidence_link requires a noncertificate evidence kind")
    if decision == "insufficient" and evidence_kind not in {
        "none", "numeric_relation", "comparative_relation", "theory_marker",
        "empirical_artifact", "qualitative_link",
    }:
        raise ValueError("invalid insufficient evidence kind")


def validate_response(response: dict[str, Any], request_row: dict[str, Any]) -> dict[str, Any]:
    if set(response) != _RESPONSE_KEYS:
        extras = sorted(set(response) - _RESPONSE_KEYS)
        missing = sorted(_RESPONSE_KEYS - set(response))
        raise ValueError(f"response keys must be exact; extras={extras}, missing={missing}")
    if str(response["paper_id"]) != request_row["paper_id"]:
        raise ValueError("response paper_id does not match bound request")
    selections = response["selections"]
    if not isinstance(selections, list) or len(selections) > 5:
        raise ValueError("selections must be a list of at most five entries")
    abstract, body = _address_index(request_row)
    seen_claims: set[str] = set()
    seen_evidence: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, selection in enumerate(selections):
        if not isinstance(selection, dict) or set(selection) != _SELECTION_KEYS:
            raise ValueError(f"selection {index} must contain exactly the frozen typed keys")
        claim_id = selection["claim_sentence_id"]
        evidence_id = selection["evidence_sentence_id"]
        if not isinstance(claim_id, str) or claim_id not in abstract:
            raise ValueError(f"selection {index} claim address is out of range or not abstract")
        if claim_id in seen_claims:
            raise ValueError(f"duplicate claim address: {claim_id}")
        seen_claims.add(claim_id)
        if evidence_id is not None:
            if not isinstance(evidence_id, str) or evidence_id not in body:
                raise ValueError(
                    f"selection {index} evidence address is out of range or not body"
                )
            if evidence_id in seen_evidence:
                raise ValueError(f"duplicate evidence address: {evidence_id}")
            seen_evidence.add(evidence_id)
        if selection["decision"] not in ALLOWED_DECISIONS:
            raise ValueError("invalid decision")
        if selection["relation"] not in ALLOWED_RELATIONS:
            raise ValueError("invalid relation")
        if selection["quantity_state"] not in ALLOWED_QUANTITY_STATES:
            raise ValueError("invalid quantity_state")
        if selection["comparison_state"] not in ALLOWED_COMPARISON_STATES:
            raise ValueError("invalid comparison_state")
        if selection["evidence_kind"] not in ALLOWED_EVIDENCE_KINDS:
            raise ValueError("invalid evidence_kind")
        if (
            not isinstance(selection["quantity_count"], int)
            or isinstance(selection["quantity_count"], bool)
            or selection["quantity_count"] < 0
        ):
            raise ValueError("quantity_count must be a nonnegative integer")
        if not isinstance(selection["comparison_present"], bool):
            raise ValueError("comparison_present must be boolean")
        _validate_selection_coherence(selection)
        validated.append(dict(selection))
    return {"paper_id": request_row["paper_id"], "selections": validated}


def _hydrated_span(span: dict[str, Any], *, relation: str | None = None) -> dict[str, Any]:
    value = {
        "sentence_id": span["sentence_id"],
        "sentence_index": span["sentence_index"],
        "start": span["start"],
        "end": span["end"],
        "text": span["text"],
        "text_sha256": span["text_sha256"],
    }
    if relation is not None:
        value["relation"] = relation
    return value


def _derived_status(decisions: list[str]) -> str:
    if not decisions:
        return "abstain"
    if "supported" in decisions and "contradicted" in decisions:
        return "mixed"
    if "supported" in decisions:
        return "supported"
    if "contradicted" in decisions:
        return "contradicted"
    if "evidence_link" in decisions:
        return "evidence_link"
    return "insufficient"


def hydrate_response(response: dict[str, Any], request_row: dict[str, Any]) -> dict[str, Any]:
    """Deterministically resolve valid addresses to exact bound source spans."""
    response = validate_response(response, request_row)
    abstract, body = _address_index(request_row)
    matches: list[dict[str, Any]] = []
    for selection in response["selections"]:
        decision = selection["decision"]
        relation = selection["relation"]
        evidence_id = selection["evidence_sentence_id"]
        witness_kind = (
            "relation_certificate"
            if decision in {"supported", "contradicted"}
            else "evidence_link" if decision == "evidence_link" else "none"
        )
        match = {
            "decision": decision,
            "witness_kind": witness_kind,
            "reason": "prompt_source_addressed_relation_judgment",
            "claim": _hydrated_span(
                abstract[selection["claim_sentence_id"]], relation=relation
            ),
            "evidence": (
                _hydrated_span(body[evidence_id]) if evidence_id is not None else None
            ),
            "checks": {
                "bm25": None,
                "claim_term_coverage": None,
                "quantity_matches": None,
                "quantity_required": None,
                "relation_state": selection["comparison_state"],
                "prompt_quantity_state": selection["quantity_state"],
                "prompt_evidence_kind": selection["evidence_kind"],
                "prompt_quantity_count": selection["quantity_count"],
                "prompt_comparison_present": selection["comparison_present"],
            },
            "source_addresses": {
                "claim": selection["claim_sentence_id"],
                "evidence": evidence_id,
            },
        }
        matches.append(match)
    decisions = [match["decision"] for match in matches]
    certificates = [
        match for match in matches if match["witness_kind"] == "relation_certificate"
    ]
    evidence_links = [
        match for match in matches if match["witness_kind"] == "evidence_link"
    ]
    return {
        "paper_id": request_row["paper_id"],
        "status": _derived_status(decisions),
        "reason": "derived_from_validated_source_addressed_selections",
        "claim_count": len(matches),
        "certificate_count": len(certificates),
        "evidence_link_count": len(evidence_links),
        "certificates": certificates,
        "evidence_links": evidence_links,
        "matches": matches,
        "graph": {
            "claim_nodes": len(matches),
            "evidence_nodes": sum(match["evidence"] is not None for match in matches),
            "edges": None,
            "matched_edges": sum(match["evidence"] is not None for match in matches),
            "matching": "prompt_source_addressed_one_to_one",
        },
        "transport": {
            "model_returned_source_text": False,
            "source_hydration": "deterministic_exact_bound_spans",
            "status_counts_graph": "derived_in_code",
        },
    }


def _verify_normalized_row(
    row: dict[str, Any],
    *,
    request: dict[str, Any],
    model_sha: str,
    bundle_manifest_sha: str,
) -> None:
    """Refuse to resume from a normalized row whose provenance or replay changed."""
    expected = {
        "schema_version": NORMALIZED_SCHEMA,
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "paper_input_sha256": request["paper_input_sha256"],
        "source_map_sha256": request["source_map_sha256"],
        "segmentation_contract_sha256": request["segmentation_contract_sha256"],
        "prompt_spec_sha256": request["prompt_spec_sha256"],
        "model_manifest_sha256": model_sha,
        "bundle_manifest_sha256": bundle_manifest_sha,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"normalized resume binding mismatch: {key}")
    response = row.get("validated_response")
    if not isinstance(response, dict):
        raise ValueError("normalized resume row lacks validated_response")
    if row.get("response_sha256") != hash_value(response):
        raise ValueError("normalized resume response hash mismatch")
    replayed = hydrate_response(response, request)
    if row.get("result") != replayed:
        raise ValueError("normalized resume deterministic hydration mismatch")


def ingest(
    bundle: Path,
    raw_results_path: Path,
    normalized_path: Path,
    rejects_path: Path,
) -> dict[str, int]:
    manifest, requests = verify_bundle(bundle)
    bundle_manifest_sha = hash_file(bundle / "manifest.json")
    model_sha = manifest["model_manifest"]["canonical_sha256"]
    existing_rows = _read_jsonl(normalized_path) if normalized_path.exists() else []
    existing: set[str] = set()
    for row in existing_rows:
        rid = row.get("request_id")
        if rid in existing:
            raise ValueError(f"normalized output has duplicate request_id: {rid}")
        if rid not in requests:
            raise ValueError(f"normalized output has request outside bundle: {rid}")
        _verify_normalized_row(
            row,
            request=requests[rid],
            model_sha=model_sha,
            bundle_manifest_sha=bundle_manifest_sha,
        )
        existing.add(rid)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_raw: set[str] = set()
    for line_number, raw_line in _numbered_lines(raw_results_path):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            rejected.append({
                "line_number": line_number,
                "request_id": None,
                "reason": f"malformed JSONL result: {exc}",
                "raw_line_sha256": hashlib.sha256(raw_line.encode("utf-8")).hexdigest(),
            })
            continue
        if not isinstance(raw, dict):
            rejected.append({
                "line_number": line_number,
                "request_id": None,
                "reason": "raw result JSON must be an object",
                "raw_result_sha256": hash_value(raw),
            })
            continue
        rid = raw.get("request_id")
        try:
            if rid in seen_raw:
                raise ValueError("duplicate request_id in raw results")
            seen_raw.add(rid)
            if rid not in requests:
                raise ValueError("request_id is not in bundle")
            request = requests[rid]
            expected_bindings = {
                "schema_version": RESULT_SCHEMA,
                "request_sha256": request["request_sha256"],
                "model_manifest_sha256": model_sha,
                "bundle_manifest_sha256": bundle_manifest_sha,
            }
            for key, value in expected_bindings.items():
                if raw.get(key) != value:
                    raise ValueError(f"bound result binding mismatch: {key}")
            response = _extract_json(raw.get("response"))
            hydrated = hydrate_response(response, request)
            if rid not in existing:
                accepted.append({
                    "schema_version": NORMALIZED_SCHEMA,
                    "request_id": rid,
                    "request_sha256": request["request_sha256"],
                    "paper_input_sha256": request["paper_input_sha256"],
                    "source_map_sha256": request["source_map_sha256"],
                    "segmentation_contract_sha256": request[
                        "segmentation_contract_sha256"
                    ],
                    "prompt_spec_sha256": request["prompt_spec_sha256"],
                    "model_manifest_sha256": model_sha,
                    "bundle_manifest_sha256": bundle_manifest_sha,
                    "raw_result_sha256": hash_value(raw),
                    "response_sha256": hash_value(response),
                    "validated_response": response,
                    "result": hydrated,
                })
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({
                "line_number": line_number,
                "request_id": rid,
                "reason": str(exc),
                "raw_result_sha256": hash_value(raw),
            })
    if accepted:
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(normalized_path, accepted, mode="a")
    if rejected:
        rejects_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(rejects_path, rejected, mode="a")
    return {
        "accepted_new": len(accepted),
        "already_present": len(existing),
        "rejected": len(rejected),
        "remaining": len(requests) - len(existing) - len(accepted),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    prepare_parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    prepare_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    prepare_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--bundle", type=Path, default=DEFAULT_OUT)
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--bundle", type=Path, default=DEFAULT_OUT)
    ingest_parser.add_argument("--raw-results", type=Path, required=True)
    ingest_parser.add_argument("--normalized", type=Path)
    ingest_parser.add_argument("--rejects", type=Path)
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare(
            args.input.resolve(), args.spec.resolve(), args.model.resolve(), args.out.resolve()
        )
        print(json.dumps({
            "status": result["status"],
            "requests": result["requests"]["count"],
            "api_calls_made": 0,
            "gpu_used": False,
            "output": str(args.out.resolve()),
        }, sort_keys=True))
    elif args.command == "verify":
        manifest, requests = verify_bundle(args.bundle.resolve())
        print(json.dumps({
            "status": "verified",
            "manifest_schema": manifest["schema_version"],
            "requests": len(requests),
        }, sort_keys=True))
    else:
        normalized = args.normalized or args.bundle / "normalized_addressed_results.jsonl"
        rejects = args.rejects or args.bundle / "rejected_addressed_results.jsonl"
        print(json.dumps(ingest(
            args.bundle.resolve(), args.raw_results.resolve(), normalized.resolve(),
            rejects.resolve(),
        ), sort_keys=True))


if __name__ == "__main__":
    main()
