#!/usr/bin/env python3
"""Prepare and ingest the sealed same-input science prompt counterpart.

This module deliberately contains no model client. ``prepare`` emits an auditable,
provider-neutral request bundle; an independent API process may execute it and return
bound responses. ``ingest`` refuses unbound, mismatched, duplicate, or ungrounded
responses. ``evaluate`` compares prompt and code witnesses without treating either as
ground truth. The trusted loader deserializes each source row, then immediately projects
it to ``paper_id + abstract + body``; label keys/values are never indexed, retained,
rendered, or used by this instrument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "datasets/peer-review/peer_review_cv_evidence.jsonl"
DEFAULT_SPEC = Path(__file__).with_name("articulability_prompt.json")
DEFAULT_MODEL = Path(__file__).with_name("articulability_model_glm47.json")
DEFAULT_OUT = ROOT / "outputs/metric_seam_pilot/science_articulability_v1_prepared"
DEFAULT_CODE = ROOT / "outputs/metric_seam_pilot/science_claims_v2_corrected_v2/results.json"

REQUEST_SCHEMA = "science-articulability-request-v1"
RESULT_SCHEMA = "science-articulability-bound-result-v1"
NORMALIZED_SCHEMA = "science-articulability-normalized-result-v1"
MANIFEST_SCHEMA = "science-articulability-bundle-v1"
TRANSPORT_GROUNDING_GUARD = (
    "SERIALIZATION_GROUNDING_GUARD: Copy every claim.text exactly and verbatim from the "
    "supplied ABSTRACT. Copy every evidence.text exactly and verbatim from the supplied BODY. "
    "Do not paraphrase, shorten, reconstruct, normalize, or add punctuation to either text. "
    "If no exact source sentence supports a certificate, use evidence_link, insufficient, or "
    "abstain as appropriate instead of inventing certificate text."
)
ALLOWED_STATUSES = {
    "supported", "contradicted", "mixed", "evidence_link", "insufficient", "abstain"
}
CERTIFICATE_DECISIONS = {"supported", "contradicted"}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Prefer repository-relative provenance paths, retaining absolute test paths."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _allowed_projection(raw: dict[str, Any], line_number: int) -> dict[str, str]:
    """Return a fresh mapping containing only the three preregistered input fields."""
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
    ids = [r["paper_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("paper_id values must be unique")
    return records


def _validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("input_allowlist") != ["paper_id", "abstract", "body"]:
        raise ValueError("prompt input_allowlist must be exactly paper_id+abstract+body")
    if spec.get("external_knowledge") != "forbidden":
        raise ValueError("external knowledge must remain forbidden")
    for key in ("system_prompt", "user_template", "decision_semantics", "output_schema"):
        if not spec.get(key):
            raise ValueError(f"prompt specification missing {key}")


def _validate_model(model: dict[str, Any]) -> None:
    required = ("schema_version", "backend", "protocol", "model", "temperature",
                "max_output_tokens", "system_prompt_transport", "response_transport")
    missing = [key for key in required if key not in model]
    if missing:
        raise ValueError(f"model manifest missing {missing}")
    if not model["model"] or model["temperature"] != 0.0:
        raise ValueError("model identity must be concrete and temperature must be 0.0")
    if "request_timeout_seconds" in model and (
        not isinstance(model["request_timeout_seconds"], (int, float))
        or isinstance(model["request_timeout_seconds"], bool)
        or model["request_timeout_seconds"] <= 0
    ):
        raise ValueError("request_timeout_seconds must be positive")
    if "max_attempts" in model and (
        not isinstance(model["max_attempts"], int)
        or isinstance(model["max_attempts"], bool)
        or model["max_attempts"] < 1
    ):
        raise ValueError("max_attempts must be a positive integer")


def _request_material(*, paper: dict[str, str], spec_sha: str, model_sha: str,
                      system_prompt: str, user_prompt: str) -> dict[str, Any]:
    return {
        "paper_input_sha256": hash_value(paper),
        "prompt_spec_sha256": spec_sha,
        "model_manifest_sha256": model_sha,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def render_system_prompt(spec: dict[str, Any]) -> str:
    """Render every frozen instruction component into the transported system message."""

    return (
        spec["system_prompt"]
        + "\n\nFROZEN_DECISION_SEMANTICS:\n"
        + json.dumps(spec["decision_semantics"], ensure_ascii=False, sort_keys=True)
        + "\n\nFROZEN_OUTPUT_SCHEMA:\n"
        + json.dumps(spec["output_schema"], ensure_ascii=False, sort_keys=True)
        + "\n\n"
        + TRANSPORT_GROUNDING_GUARD
    )


def build_requests(records: Iterable[dict[str, str]], spec: dict[str, Any],
                   model: dict[str, Any]) -> list[dict[str, Any]]:
    _validate_spec(spec)
    _validate_model(model)
    spec_sha = hash_value(spec)
    model_sha = hash_value(model)
    transported_system_prompt = render_system_prompt(spec)
    requests: list[dict[str, Any]] = []
    for index, paper in enumerate(records):
        user_prompt = spec["user_template"].format(**paper)
        material = _request_material(
            paper=paper, spec_sha=spec_sha, model_sha=model_sha,
            system_prompt=transported_system_prompt, user_prompt=user_prompt,
        )
        request_sha = hash_value(material)
        requests.append({
            "schema_version": REQUEST_SCHEMA,
            "request_id": f"science_articulability_{index:04d}_{request_sha[:16]}",
            "sequence_index": index,
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


def prepare(input_path: Path, spec_path: Path, model_path: Path,
            output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty bundle: {output_dir}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    records = load_inputs(input_path)
    requests = build_requests(records, spec, model)
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "requests.jsonl"
    _write_jsonl(requests_path, requests)
    corpus_sha = hash_value([hash_value(record) for record in records])
    prompt_lengths = [len(r["system_prompt"]) + len(r["user_prompt"]) for r in requests]
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_run",
        "objective": "prompt_articulability_same_input",
        "external_supervision": "none",
        "external_knowledge": "forbidden",
        "label_policy": {
            "loader_allowlist": ["paper_id", "abstract", "body"],
            "note": "The loader projects each source row to a new allowlisted mapping; no label value is accessed or emitted.",
        },
        "input": {
            "path": display_path(input_path),
            "source_file_sha256": hash_file(input_path),
            "allowlisted_corpus_sha256": corpus_sha,
            "record_count": len(records),
        },
        "prompt_spec": {
            "path": display_path(spec_path),
            "file_sha256": hash_file(spec_path),
            "canonical_sha256": hash_value(spec),
        },
        "model_manifest": {
            "path": display_path(model_path),
            "file_sha256": hash_file(model_path),
            "canonical_sha256": hash_value(model),
            "identity": model,
        },
        "artifacts": {
            display_path(Path(__file__)): hash_file(Path(__file__)),
            display_path(Path(__file__).with_name("articulability_api_runner.py")):
                hash_file(Path(__file__).with_name("articulability_api_runner.py")),
        },
        "requests": {
            "path": "requests.jsonl",
            "sha256": hash_file(requests_path),
            "count": len(requests),
            "api_call_count": len(requests),
            "min_prompt_characters": min(prompt_lengths, default=0),
            "max_prompt_characters": max(prompt_lengths, default=0),
        },
        "result_contract": {
            "schema_version": RESULT_SCHEMA,
            "required_fields": [
                "schema_version", "request_id", "request_sha256",
                "model_manifest_sha256", "bundle_manifest_sha256", "response"
            ],
            "response": "JSON object or raw text containing exactly one JSON object",
            "resume_key": "request_id",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                                        sort_keys=True) + "\n", encoding="utf-8")
    readme = f"""# Prepared science prompt-articulability counterpart

Status: **prepared, not run**. This bundle contains {len(requests):,} independent requests, so a
complete execution requires exactly **{len(requests):,} API calls**. The request and manifest files
bind the allowlisted input, prompt specification, model identity, decoding contract, and every
rendered prompt by SHA-256. No acceptance label or external scientific source is used.

An independent runner must return one JSONL object per successful call using the result contract
in `manifest.json`. It must copy `request_id`, `request_sha256`, and
`model_manifest_sha256` from the request, bind the exact bundle manifest SHA, and put the model
text (or parsed JSON object) in `response`. Partial files are expected and can be ingested
repeatedly.

Deterministic five-request CPU/API smoke (temperature 0, serial calls):

```bash
python -m methods.metric_seam.science_claims_v2.articulability_api_runner \\
  --bundle {output_dir} --out {output_dir}/raw_api_results.jsonl \\
  --limit 5 --concurrency 1
```

```bash
python -m methods.metric_seam.science_claims_v2.articulability_pipeline ingest \\
  --bundle {output_dir} --raw-results /path/to/api_results.jsonl
python -m methods.metric_seam.science_claims_v2.articulability_pipeline evaluate \\
  --bundle {output_dir}
```

The optional runner command invokes the declared remote API but no local GPU. The ingest and
evaluate commands invoke neither an API nor a GPU.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def verify_bundle(bundle: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported bundle manifest")
    requests_path = bundle / manifest["requests"]["path"]
    if hash_file(requests_path) != manifest["requests"]["sha256"]:
        raise ValueError("requests file hash mismatch")
    rows = _read_jsonl(requests_path)
    if len(rows) != manifest["requests"]["count"]:
        raise ValueError("request count mismatch")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        material = {key: row[key] for key in (
            "paper_input_sha256", "prompt_spec_sha256", "model_manifest_sha256",
            "system_prompt", "user_prompt"
        )}
        if hash_value(material) != row["request_sha256"]:
            raise ValueError(f"request material mismatch: {row['request_id']}")
        if row["request_id"] in by_id:
            raise ValueError(f"duplicate request id: {row['request_id']}")
        by_id[row["request_id"]] = row
    return manifest, by_id


def _extract_json(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if not isinstance(response, str):
        raise ValueError("response must be an object or string")
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not exactly one JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("response JSON must be an object")
    return parsed


def _canonical_whitespace_span(value: Any) -> str:
    """Fold source-layout whitespace while preserving every lexical character.

    Full-paper text contains PDF line wrapping, so a sentence copied as one line must
    remain admissible. This canonicalization changes only runs of whitespace. It does
    not lowercase, strip punctuation, dehyphenate line breaks, or normalize Unicode.
    """
    return re.sub(r"\s+", " ", str(value)).strip()


def _witness_shape(witness: dict[str, Any]) -> dict[str, Any]:
    """Return the claim/evidence payload from either supported transport shape.

    The frozen schema describes certificates with claim/evidence fields directly and
    evidence links/matches with those fields under ``shape``. Some JSON-mode providers
    serialize the latter fields directly despite preserving every semantic field. We
    accept both representations, but apply the same source-grounding guard to both.
    """
    shape = witness.get("shape")
    if shape is None:
        shape = witness
    if not isinstance(shape, dict):
        raise ValueError("witness shape must be an object")
    return shape


def _validate_grounded_witness(
    witness: dict[str, Any], *, canonical_abstract: str, canonical_body: str, context: str
) -> None:
    shape = _witness_shape(witness)
    claim = shape.get("claim") or {}
    evidence = shape.get("evidence") or {}
    if not isinstance(claim, dict) or not isinstance(evidence, dict):
        raise ValueError(f"{context} claim/evidence must be objects")
    claim_text = _canonical_whitespace_span(claim.get("text"))
    evidence_text = _canonical_whitespace_span(evidence.get("text"))
    if not claim_text or claim_text not in canonical_abstract:
        raise ValueError(
            f"{context} claim text is not a verbatim whitespace-canonical span "
            "of the bound abstract"
        )
    if not evidence_text or evidence_text not in canonical_body:
        raise ValueError(
            f"{context} evidence text is not a verbatim whitespace-canonical span "
            "of the bound body"
        )
    if evidence_text in canonical_abstract:
        raise ValueError(f"{context} uses an abstract sentence as body evidence")


def _validate_strong_relation_certificate(
    witness: dict[str, Any], *, context: str
) -> None:
    """Enforce the frozen prompt's *strong certificate* semantics.

    Verbatim source grounding is necessary but not sufficient.  The prompt of
    record says that only exact numeric or comparative relations are strong
    certificates.  In particular, a qualitative relation with no quantities and
    no comparison object cannot become a certificate merely because its text is a
    literal source span.

    This guard deliberately validates the model's typed witness rather than
    re-judging the science.  It requires an auditable numeric/comparison payload
    and coherent bookkeeping; it does not treat either the prompt or code output
    as external ground truth.
    """
    shape = _witness_shape(witness)
    claim = shape.get("claim")
    evidence = shape.get("evidence")
    checks = shape.get("checks")
    if not isinstance(claim, dict) or not isinstance(evidence, dict):
        raise ValueError(f"{context} claim/evidence must be objects")
    if not isinstance(checks, dict):
        raise ValueError(f"{context} checks must be an object")

    relation = claim.get("relation")
    if relation not in {"numeric", "comparative"}:
        raise ValueError(f"{context} relation must be numeric or comparative")

    claim_quantities = claim.get("quantities")
    evidence_quantities = evidence.get("quantities")
    if not isinstance(claim_quantities, list) or not isinstance(evidence_quantities, list):
        raise ValueError(f"{context} quantities must be lists")
    quantity_matches = checks.get("quantity_matches")
    quantity_required = checks.get("quantity_required")
    for name, value in (
        ("quantity_matches", quantity_matches),
        ("quantity_required", quantity_required),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{context} {name} must be a nonnegative integer")
    if quantity_matches > quantity_required:
        raise ValueError(f"{context} quantity_matches exceeds quantity_required")

    claim_comparison = claim.get("comparison")
    evidence_comparison = evidence.get("comparison")
    comparison_present = isinstance(claim_comparison, dict) and isinstance(
        evidence_comparison, dict
    )
    if (claim_comparison is None) != (evidence_comparison is None):
        raise ValueError(f"{context} comparison payload is present on only one side")
    if claim_comparison is not None and not comparison_present:
        raise ValueError(f"{context} comparison payloads must be objects or null")

    numeric_present = bool(claim_quantities) and bool(evidence_quantities)
    if numeric_present:
        if quantity_required <= 0 or quantity_matches != quantity_required:
            raise ValueError(f"{context} numeric quantities are not exactly matched")
    elif quantity_required != 0 or quantity_matches != 0:
        raise ValueError(f"{context} quantity bookkeeping has no quantity payload")

    if relation == "numeric" and not numeric_present:
        raise ValueError(f"{context} numeric certificate has no quantity relation")
    if relation == "comparative" and not comparison_present:
        raise ValueError(f"{context} comparative certificate has no comparison relation")

    decision = witness.get("decision")
    relation_state = checks.get("relation_state")
    if decision == "supported":
        allowed_states = {"aligned", "not_required"} if numeric_present else {"aligned"}
        if relation_state not in allowed_states:
            raise ValueError(f"{context} supported relation_state is not aligned")
    elif decision == "contradicted":
        if not comparison_present:
            raise ValueError(f"{context} contradicted certificate needs a comparison relation")
        if relation_state not in {
            "aligned_reversed", "reversed_roles", "direction_mismatch"
        }:
            raise ValueError(f"{context} contradicted relation_state is not a reversal")
    else:
        raise ValueError(f"{context} strong certificate has invalid decision")


def _validate_response(response: dict[str, Any], request_row: dict[str, Any]) -> dict[str, Any]:
    required = ("paper_id", "status", "reason", "claim_count", "certificate_count",
                "evidence_link_count", "certificates", "evidence_links", "matches", "graph")
    missing = [key for key in required if key not in response]
    if missing:
        raise ValueError(f"response missing fields: {missing}")
    if str(response["paper_id"]) != request_row["paper_id"]:
        raise ValueError("paper_id does not match bound request")
    if response["status"] not in ALLOWED_STATUSES:
        raise ValueError("invalid status")
    for key in ("claim_count", "certificate_count", "evidence_link_count"):
        if not isinstance(response[key], int) or isinstance(response[key], bool) or response[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if response["claim_count"] > 5:
        raise ValueError("claim_count exceeds the frozen at-most-five contract")
    for key in ("certificates", "evidence_links", "matches"):
        if not isinstance(response[key], list):
            raise ValueError(f"{key} must be a list")
    if response["certificate_count"] != len(response["certificates"]):
        raise ValueError("certificate_count does not match certificates")
    if response["evidence_link_count"] != len(response["evidence_links"]):
        raise ValueError("evidence_link_count does not match evidence_links")
    if response["certificate_count"] + response["evidence_link_count"] > response["claim_count"]:
        raise ValueError("witness counts exceed claim_count")

    # Reconstruct the exact allowlisted source from the bound rendered prompt. This
    # validates textual witness grounding without reopening the labelled source file.
    user_prompt = request_row["user_prompt"]
    marker_a, marker_b = "\n\nABSTRACT:\n", "\n\nBODY:\n"
    if marker_a not in user_prompt or marker_b not in user_prompt:
        raise ValueError("bound user prompt has unexpected shape")
    abstract, body = user_prompt.split(marker_a, 1)[1].split(marker_b, 1)
    canonical_abstract = _canonical_whitespace_span(abstract)
    canonical_body = _canonical_whitespace_span(body)
    for cert in response["certificates"]:
        if not isinstance(cert, dict) or cert.get("decision") not in CERTIFICATE_DECISIONS:
            raise ValueError("invalid certificate decision")
        if cert.get("witness_kind") != "relation_certificate":
            raise ValueError("certificate witness_kind must be relation_certificate")
        _validate_grounded_witness(
            cert, canonical_abstract=canonical_abstract, canonical_body=canonical_body,
            context="certificate"
        )
        _validate_strong_relation_certificate(cert, context="certificate")
    for link in response["evidence_links"]:
        if not isinstance(link, dict) or link.get("decision") != "evidence_link":
            raise ValueError("invalid evidence-link decision")
        if link.get("witness_kind") != "evidence_link":
            raise ValueError("evidence-link witness_kind must be evidence_link")
        _validate_grounded_witness(
            link, canonical_abstract=canonical_abstract, canonical_body=canonical_body,
            context="evidence-link"
        )
    for match in response["matches"]:
        if not isinstance(match, dict):
            raise ValueError("match must be an object")
        decision = match.get("decision")
        witness_kind = match.get("witness_kind")
        if decision not in {"supported", "contradicted", "evidence_link", "insufficient"}:
            raise ValueError("invalid match decision")
        if witness_kind not in {"relation_certificate", "evidence_link", "none"}:
            raise ValueError("invalid match witness_kind")
        if witness_kind != "none":
            _validate_grounded_witness(
                match, canonical_abstract=canonical_abstract, canonical_body=canonical_body,
                context="match"
            )
        if witness_kind == "relation_certificate":
            _validate_strong_relation_certificate(match, context="match")
    return response


def _load_existing(path: Path, requests: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    for row in _read_jsonl(path):
        rid = row.get("request_id")
        if rid not in requests:
            raise ValueError(f"normalized result has unknown request_id: {rid}")
        expected = requests[rid]
        for key in ("request_sha256", "model_manifest_sha256"):
            if row.get(key) != expected[key]:
                raise ValueError(f"normalized result binding mismatch: {rid}/{key}")
        if row.get("paper_id") != expected["paper_id"]:
            raise ValueError(f"normalized result binding mismatch: {rid}/paper_id")
        if not isinstance(row.get("response"), dict):
            raise ValueError(f"normalized result response is not an object: {rid}")
        _validate_response(row["response"], expected)
        if rid in existing and hash_value(row) != hash_value(existing[rid]):
            raise ValueError(f"conflicting normalized duplicate: {rid}")
        existing[rid] = row
    return existing


def ingest(bundle: Path, raw_results: Path, normalized_path: Path,
           rejects_path: Path) -> dict[str, int]:
    manifest, requests = verify_bundle(bundle)
    existing = _load_existing(normalized_path, requests)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_batch: set[str] = set()
    model_sha = manifest["model_manifest"]["canonical_sha256"]
    bundle_manifest_sha = hash_file(bundle / "manifest.json")
    with raw_results.open(encoding="utf-8") as handle:
        raw_lines = list(enumerate(handle, 1))
    for line_number, line in raw_lines:
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("raw result row must be an object")
        except (json.JSONDecodeError, ValueError) as exc:
            rejected.append({"line_number": line_number, "request_id": None,
                             "reason": str(exc),
                             "raw_result_sha256": hashlib.sha256(line.encode()).hexdigest()})
            continue
        rid = raw.get("request_id")
        try:
            if raw.get("schema_version") != RESULT_SCHEMA:
                raise ValueError("unsupported raw result schema")
            if rid not in requests:
                raise ValueError("unknown request_id")
            if rid in seen_batch:
                raise ValueError("duplicate request_id in raw batch")
            seen_batch.add(rid)
            expected = requests[rid]
            if raw.get("request_sha256") != expected["request_sha256"]:
                raise ValueError("request_sha256 mismatch")
            if raw.get("model_manifest_sha256") != model_sha:
                raise ValueError("model_manifest_sha256 mismatch")
            if raw.get("bundle_manifest_sha256") != bundle_manifest_sha:
                raise ValueError("bundle_manifest_sha256 mismatch")
            parsed = _validate_response(_extract_json(raw.get("response")), expected)
            row = {
                "schema_version": NORMALIZED_SCHEMA,
                "request_id": rid,
                "request_sha256": expected["request_sha256"],
                "paper_id": expected["paper_id"],
                "paper_input_sha256": expected["paper_input_sha256"],
                "prompt_spec_sha256": expected["prompt_spec_sha256"],
                "model_manifest_sha256": expected["model_manifest_sha256"],
                "bundle_manifest_sha256": bundle_manifest_sha,
                "response_sha256": hash_value(raw.get("response")),
                "response": parsed,
            }
            if rid in existing:
                if hash_value(row) != hash_value(existing[rid]):
                    raise ValueError("result conflicts with already ingested response")
                continue
            accepted.append(row)
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"line_number": line_number, "request_id": rid,
                             "reason": str(exc), "raw_result_sha256": hash_value(raw)})
    if accepted:
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(normalized_path, accepted, mode="a")
    if rejected:
        rejects_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(rejects_path, rejected, mode="a")
    return {
        "accepted_new": len(accepted), "already_present": len(existing),
        "rejected": len(rejected), "remaining": len(requests) - len(existing) - len(accepted),
    }


def _tokens(text: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _jaccard(left: Any, right: Any) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def _witness_match(prompt: dict[str, Any], code: dict[str, Any]) -> bool:
    return (
        prompt.get("decision") == code.get("decision")
        and (prompt.get("claim") or {}).get("relation") == (code.get("claim") or {}).get("relation")
        and _jaccard((prompt.get("claim") or {}).get("text"),
                     (code.get("claim") or {}).get("text")) >= 0.60
        and _jaccard((prompt.get("evidence") or {}).get("text"),
                     (code.get("evidence") or {}).get("text")) >= 0.60
    )


def _maximum_witness_matches(prompt: list[dict[str, Any]],
                             code: list[dict[str, Any]]) -> int:
    # At most five prompt claims: exhaustive augmenting paths are small and deterministic.
    adjacency = [[j for j, c in enumerate(code) if _witness_match(p, c)] for p in prompt]
    assigned: dict[int, int] = {}

    def visit(i: int, used: set[int]) -> bool:
        for j in adjacency[i]:
            if j in used:
                continue
            used.add(j)
            if j not in assigned or visit(assigned[j], used):
                assigned[j] = i
                return True
        return False

    return sum(visit(i, set()) for i in range(len(prompt)))


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _smoke_execution_summary(
    *, bundle: Path, manifest: dict[str, Any], requests: dict[str, dict[str, Any]],
    normalized: dict[str, dict[str, Any]], raw_results_path: Path | None,
    rejects_path: Path | None,
) -> dict[str, Any]:
    """Summarize a bounded partial execution without retaining hidden reasoning text."""
    if raw_results_path is None:
        return {"status": "not_supplied"}
    rows = _read_jsonl(raw_results_path)
    seen: set[str] = set()
    observed_models: Counter[str] = Counter()
    stop_reasons: Counter[str] = Counter()
    reasoning_values: list[tuple[str, int]] = []
    bundle_sha = hash_file(bundle / "manifest.json")
    model_sha = manifest["model_manifest"]["canonical_sha256"]
    for row in rows:
        rid = row.get("request_id")
        if rid not in requests:
            raise ValueError(f"smoke raw result has unknown request_id: {rid}")
        if rid in seen:
            raise ValueError(f"smoke raw result has duplicate request_id: {rid}")
        seen.add(rid)
        expected = requests[rid]
        bindings = {
            "schema_version": RESULT_SCHEMA,
            "request_sha256": expected["request_sha256"],
            "model_manifest_sha256": model_sha,
            "bundle_manifest_sha256": bundle_sha,
        }
        for key, value in bindings.items():
            if row.get(key) != value:
                raise ValueError(f"smoke raw result binding mismatch: {rid}/{key}")
        metadata = row.get("provider_metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"smoke provider_metadata is not an object: {rid}")
        if metadata.get("model") is not None:
            observed_models[str(metadata["model"])] += 1
        if metadata.get("stop_reason") is not None:
            stop_reasons[str(metadata["stop_reason"])] += 1
        usage = metadata.get("usage") or {}
        details = usage.get("completion_tokens_details") or {} if isinstance(usage, dict) else {}
        reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
            reasoning_values.append((rid, reasoning_tokens))

    rejected_rows = _read_jsonl(rejects_path) if rejects_path is not None else []
    rejected_ids = [row.get("request_id") for row in rejected_rows]
    if len(rejected_ids) != len(set(rejected_ids)):
        raise ValueError("smoke rejection file contains duplicate request_id")
    if any(rid not in seen for rid in rejected_ids):
        raise ValueError("smoke rejection file contains request absent from raw results")
    accepted_ids = seen & set(normalized)
    rejected_set = set(rejected_ids)
    partition_complete = accepted_ids.isdisjoint(rejected_set) and (
        accepted_ids | rejected_set == seen
    )
    requested_reasoning = manifest["model_manifest"]["identity"].get("reasoning")
    nonzero_reasoning = [rid for rid, value in reasoning_values if value > 0]
    return {
        "status": "bounded_partial_execution_receipt",
        "raw_results": {
            "path": display_path(raw_results_path), "sha256": hash_file(raw_results_path),
            "attempted_unique_requests": len(seen),
        },
        "validation": {
            "valid_normalized": len(accepted_ids),
            "rejected": len(rejected_set),
            "valid_rate_among_attempted": _safe_ratio(len(accepted_ids), len(seen)),
            "accepted_rejected_partition_complete": partition_complete,
            "rejection_reasons": dict(sorted(Counter(
                str(row.get("reason")) for row in rejected_rows
            ).items())),
        },
        "provider_observation": {
            "requested_model": manifest["model_manifest"]["identity"].get("model"),
            "observed_models": dict(sorted(observed_models.items())),
            "stop_reasons": dict(sorted(stop_reasons.items())),
            "requested_reasoning": requested_reasoning,
            "reasoning_tokens_reported_for": len(reasoning_values),
            "reported_reasoning_tokens_total": sum(value for _, value in reasoning_values),
            "responses_with_nonzero_reported_reasoning_tokens": len(nonzero_reasoning),
            "reasoning_request_observed_honored_on_all_responses": (
                not nonzero_reasoning if len(reasoning_values) == len(seen) else None
            ),
            "hidden_reasoning_text_retained": False,
            "interpretation": (
                "Provider telemetry describes transport, not response validity. The runner "
                "retains token counts but never hidden reasoning text."
            ),
        },
    }


def evaluate(bundle: Path, normalized_path: Path, code_path: Path | None,
             output_path: Path, report_path: Path, require_complete: bool,
             raw_results_path: Path | None = None,
             rejects_path: Path | None = None) -> dict[str, Any]:
    manifest, requests = verify_bundle(bundle)
    normalized = _load_existing(normalized_path, requests)
    if require_complete and len(normalized) != len(requests):
        raise ValueError(f"incomplete results: {len(normalized)}/{len(requests)}")
    prompt_by_paper = {row["paper_id"]: row["response"] for row in normalized.values()}
    statuses = Counter(row["status"] for row in prompt_by_paper.values())
    prompt_certificate_count = sum(row["certificate_count"] for row in prompt_by_paper.values())
    prompt_certificate_docs = sum(row["certificate_count"] > 0 for row in prompt_by_paper.values())
    comparison: dict[str, Any] = {"status": "not_run"}
    comparator_input: dict[str, Any] | None = None
    if code_path is not None and prompt_by_paper:
        code_payload = json.loads(code_path.read_text(encoding="utf-8"))
        code_by_paper = {row["paper_id"]: row for row in code_payload["records"]}
        shared = sorted(set(prompt_by_paper) & set(code_by_paper))
        status_exact = 0
        presence_exact = 0
        both, prompt_only, code_only, neither = 0, 0, 0, 0
        matched_witnesses = prompt_witnesses = code_witnesses = 0
        matched_evidence_links = prompt_evidence_links = code_evidence_links = 0
        for paper_id in shared:
            p, c = prompt_by_paper[paper_id], code_by_paper[paper_id]
            status_exact += p["status"] == c["status"]
            p_has, c_has = p["certificate_count"] > 0, c["certificate_count"] > 0
            presence_exact += p_has == c_has
            if p_has and c_has:
                both += 1
            elif p_has:
                prompt_only += 1
            elif c_has:
                code_only += 1
            else:
                neither += 1
            p_w, c_w = p["certificates"], c.get("certificates", [])
            prompt_witnesses += len(p_w)
            code_witnesses += len(c_w)
            matched_witnesses += _maximum_witness_matches(p_w, c_w)
            p_l, c_l = p["evidence_links"], c.get("evidence_links", [])
            prompt_evidence_links += len(p_l)
            code_evidence_links += len(c_l)
            matched_evidence_links += _maximum_witness_matches(p_l, c_l)
        informative_presence_papers = both + prompt_only + code_only
        non_estimating_reasons: list[str] = []
        if len(shared) < 2:
            non_estimating_reasons.append("fewer_than_two_shared_papers")
        if informative_presence_papers == 0:
            non_estimating_reasons.append("no_shared_paper_has_a_strong_certificate")
        comparison = {
            "status": (
                "non_estimating_descriptive_comparison"
                if non_estimating_reasons
                else "descriptive_unsupervised_reconstruction_comparison"
            ),
            "estimating": not non_estimating_reasons,
            "non_estimating_reasons": non_estimating_reasons,
            "interpretation": (
                "The prompt and code channels are comparators, not ground truth; unmatched "
                "witnesses are divergence, not error or overperformance. Agreement fractions "
                "from a non-estimating support are bookkeeping only and must not be promoted."
            ),
            "shared_papers": len(shared),
            "status_exact_agreement": _safe_ratio(status_exact, len(shared)),
            "certificate_presence_agreement": _safe_ratio(presence_exact, len(shared)),
            "certificate_presence_cells": {
                "both": both, "prompt_only": prompt_only,
                "code_only": code_only, "neither": neither,
            },
            "witness_match_rule": {
                "decision": "exact", "relation": "exact",
                "claim_token_jaccard_min": 0.60, "evidence_token_jaccard_min": 0.60,
                "assignment": "maximum_cardinality_one_to_one",
            },
            "prompt_witnesses": prompt_witnesses,
            "code_witnesses": code_witnesses,
            "matched_witnesses": matched_witnesses,
            "prompt_witness_match_rate": _safe_ratio(matched_witnesses, prompt_witnesses),
            "code_witness_match_rate": _safe_ratio(matched_witnesses, code_witnesses),
            "prompt_evidence_links": prompt_evidence_links,
            "code_evidence_links": code_evidence_links,
            "matched_evidence_links": matched_evidence_links,
            "prompt_evidence_link_match_rate": _safe_ratio(
                matched_evidence_links, prompt_evidence_links
            ),
            "code_evidence_link_match_rate": _safe_ratio(
                matched_evidence_links, code_evidence_links
            ),
        }
        comparator_input = {"path": display_path(code_path), "sha256": hash_file(code_path)}
    elif code_path is not None:
        comparison = {"status": "not_run_no_valid_prompt_results"}
        comparator_input = {"path": display_path(code_path), "sha256": hash_file(code_path)}
    payload = {
        "schema_version": "science-articulability-evaluation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "same_input_unsupervised_reconstruction",
        "external_supervision": "none",
        "normalization_evaluation_instrument": {
            "path": display_path(Path(__file__)),
            "sha256": hash_file(Path(__file__)),
            "prepared_manifest_recorded_sha256": manifest.get("artifacts", {}).get(
                display_path(Path(__file__))
            ),
            "provenance": (
                "Additive post-execution ingest/evaluation replay. Requests and raw results "
                "remain bound to the immutable prepared manifest; this hash identifies the "
                "current validator and evaluator and does not rewrite preparation history."
            ),
        },
        "bundle": {"path": display_path(bundle),
                   "manifest_sha256": hash_file(bundle / "manifest.json"),
                   "requests_sha256": manifest["requests"]["sha256"]},
        "normalized_results": {"path": display_path(normalized_path),
                               "sha256": hash_file(normalized_path) if normalized_path.exists() else None},
        "code_comparator": comparator_input,
        "summary": {
            "expected": len(requests), "valid": len(normalized),
            "remaining": len(requests) - len(normalized),
            "completion_rate": round(len(normalized) / len(requests), 6) if requests else 0.0,
            "status_counts": dict(sorted(statuses.items())),
            "prompt_certificate_documents": prompt_certificate_docs,
            "prompt_certificate_count": prompt_certificate_count,
        },
        "execution_smoke": _smoke_execution_summary(
            bundle=bundle, manifest=manifest, requests=requests, normalized=normalized,
            raw_results_path=raw_results_path, rejects_path=rejects_path,
        ),
        "isomorphism": comparison,
        "records": [normalized[rid] for rid in sorted(normalized)],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                      sort_keys=True) + "\n", encoding="utf-8")
    s = payload["summary"]
    smoke = payload["execution_smoke"]
    instrument = payload["normalization_evaluation_instrument"]
    smoke_line = "- Execution smoke: `not supplied`"
    if smoke["status"] == "bounded_partial_execution_receipt":
        v = smoke["validation"]
        p = smoke["provider_observation"]
        smoke_line = (
            f"- Execution smoke: {v['valid_normalized']} valid / "
            f"{smoke['raw_results']['attempted_unique_requests']} attempted; "
            f"{v['rejected']} rejected\n"
            f"- Reasoning transport: requested `{json.dumps(p['requested_reasoning'], sort_keys=True)}`; "
            f"{p['responses_with_nonzero_reported_reasoning_tokens']} responses reported nonzero "
            "reasoning tokens (hidden reasoning text was not retained)"
        )
    overlap_line = "- Witness overlap: `not run`"
    if comparison["status"] in {
        "descriptive_unsupervised_reconstruction_comparison",
        "non_estimating_descriptive_comparison",
    }:
        estimating_note = ""
        if not comparison["estimating"]:
            estimating_note = (
                "\n- Estimating support: `no` ("
                + ", ".join(comparison["non_estimating_reasons"])
                + "); agreement fractions are bookkeeping only"
            )
        overlap_line = (
            f"- Strong-certificate overlap: {comparison['matched_witnesses']} matched / "
            f"{comparison['prompt_witnesses']} prompt / {comparison['code_witnesses']} code\n"
            f"- Weaker evidence-link overlap: {comparison['matched_evidence_links']} matched / "
            f"{comparison['prompt_evidence_links']} prompt / "
            f"{comparison['code_evidence_links']} code"
            f"{estimating_note}"
        )
    report_path.write_text(f"""# Science prompt-articulability same-input evaluation

- Valid responses: {s['valid']} / {s['expected']} ({s['completion_rate']:.1%})
- Remaining API calls: {s['remaining']}
- Ingest/evaluation instrument SHA-256: `{instrument['sha256']}`
{smoke_line}
- Statuses: `{json.dumps(s['status_counts'], sort_keys=True)}`
- Prompt relation certificates: {s['prompt_certificate_count']} across {s['prompt_certificate_documents']} papers
- Comparator status: `{comparison['status']}`
{overlap_line}

This is an unsupervised reconstruction comparison over the identical
`paper_id + abstract + body` evidence surface. The code channel is a comparator, not ground
truth. Prompt-only and code-only witnesses are reported as divergence and do not by themselves
establish correctness, superiority, or tacitness. Invalid or unbound API responses are excluded.
""", encoding="utf-8")
    return payload


def pending(bundle: Path, normalized_path: Path, output_path: Path) -> dict[str, int]:
    _, requests = verify_bundle(bundle)
    existing = _load_existing(normalized_path, requests)
    rows = [row for rid, row in requests.items() if rid not in existing]
    _write_jsonl(output_path, rows)
    return {"completed": len(existing), "pending": len(rows), "total": len(requests)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p_prepare.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    p_prepare.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL)
    p_prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--bundle", type=Path, default=DEFAULT_OUT)
    p_ingest.add_argument("--raw-results", type=Path, required=True)
    p_ingest.add_argument("--normalized", type=Path)
    p_ingest.add_argument("--rejects", type=Path)
    p_pending = sub.add_parser("pending")
    p_pending.add_argument("--bundle", type=Path, default=DEFAULT_OUT)
    p_pending.add_argument("--normalized", type=Path)
    p_pending.add_argument("--output", type=Path)
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--bundle", type=Path, default=DEFAULT_OUT)
    p_eval.add_argument("--normalized", type=Path)
    p_eval.add_argument("--code-comparator", type=Path, default=DEFAULT_CODE)
    p_eval.add_argument("--output", type=Path)
    p_eval.add_argument("--report", type=Path)
    p_eval.add_argument("--raw-results", type=Path)
    p_eval.add_argument("--rejects", type=Path)
    p_eval.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.input.resolve(), args.spec.resolve(),
                         args.model_manifest.resolve(), args.output_dir.resolve())
    elif args.command == "ingest":
        normalized = (args.normalized or args.bundle / "normalized_results.jsonl").resolve()
        rejects = (args.rejects or args.bundle / "rejected_results.jsonl").resolve()
        result = ingest(args.bundle.resolve(), args.raw_results.resolve(), normalized, rejects)
    elif args.command == "pending":
        normalized = (args.normalized or args.bundle / "normalized_results.jsonl").resolve()
        output = (args.output or args.bundle / "pending_requests.jsonl").resolve()
        result = pending(args.bundle.resolve(), normalized, output)
    else:
        normalized = (args.normalized or args.bundle / "normalized_results.jsonl").resolve()
        output = (args.output or args.bundle / "evaluation.json").resolve()
        report = (args.report or args.bundle / "REPORT.md").resolve()
        result = evaluate(args.bundle.resolve(), normalized,
                          args.code_comparator.resolve() if args.code_comparator else None,
                          output, report, args.require_complete,
                          args.raw_results.resolve() if args.raw_results else None,
                          args.rejects.resolve() if args.rejects else None)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
