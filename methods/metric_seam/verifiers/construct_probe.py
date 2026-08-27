"""Frozen full-context prompt contract for pre-authoring corpus probes."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Mapping, Sequence

from .lifecycle import UnitProposal, stable_train_sample
from .schema import SchemaError, Verdict


REQUEST_SCHEMA = "metric-seam.construct-base-rate-request.v2"
PARSER_VERSION = "metric-seam.construct-base-rate-parser.v1"
DOCUMENT_PATH = "document.txt"
SAMPLE_SIZE = 32
SAMPLE_SALT = "metric-seam.construct-base-rate.patent-antecedent.v1"
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = """You are the prompt-based articulability probe for one frozen relation.
Judge the relation from the complete supplied document and its logical context. Do not infer or imitate a code detector.
Return exactly one JSON object:
{"applies": boolean, "violated": boolean, "witnesses": [{"path": "document.txt", "start_line": integer, "end_line": integer}]}
Use applies=false only when the document has no occasion described by the proposal. For an applicable item, violated=false means the relation is satisfied and violated=true means it is violated. Cite the exact one-based line numbers printed at the left of the load-bearing document lines; applicable verdicts may not use an empty witness list. Do not emit confidence, scores, rationale, Markdown, or extra keys."""


PATENT_ANTECEDENT_PROPOSAL = UnitProposal(
    unit_id="patents.antecedent_basis.bounded_claim_graph",
    task="patents",
    criterion_id="antecedent_basis",
    construct_text="A claim term has proper antecedent basis within the presented claim set.",
    relation=(
        "Within the presented numbered claims, each definite claim-term reference has an explicit "
        "introduction in the same claim or an inherited introduction through a valid explicit dependency path."
    ),
    occasion="The presented claims contain at least one definite claim-term reference whose antecedent can be assessed.",
    satisfied_when="Every assessed definite claim-term reference has an explicit or inherited antecedent in the presented claim graph.",
    violated_when="At least one assessed definite claim-term reference lacks explicit or inherited antecedent basis in the presented claim graph.",
    required_context="The complete presented patent text, including the full numbered claims section.",
    non_goals=(
        "whole-patent definiteness", "legal validity", "antecedents available only outside the presented claims",
    ),
    proxy_risks=(
        "exact noun-head matching", "article matching without referent identity", "ignoring inherited or semantic coreference",
    ),
)


def compile_request(
    proposal: UnitProposal, *, item_key: str, ctext: str, model: str
) -> dict[str, object]:
    if not item_key or not ctext or not model:
        raise ValueError("request requires item_key, full ctext, and model")
    proposal_value = proposal.to_json_value()
    numbered_ctext = "\n".join(
        f"{line_number:06d}|{line}"
        for line_number, line in enumerate(ctext.splitlines(), 1)
    )
    user_prompt = (
        "FROZEN UNIT PROPOSAL:\n"
        + json.dumps(proposal_value, sort_keys=True, ensure_ascii=False)
        + f"\n\nFULL DOCUMENT ({DOCUMENT_PATH}) WITH ONE-BASED LINE NUMBERS:\n{numbered_ctext}"
    )
    identity = {
        "schema": REQUEST_SCHEMA,
        "unit_id": proposal.unit_id,
        "item_key": item_key,
        "split": "compiler_train",
        "pass_index": 1,
        "model": model,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "proposal": proposal_value,
        "ctext": ctext,
    }
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {
        **identity,
        "request_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "response_contract": {
            "parser_version": PARSER_VERSION,
            "transport_version": "line-numbered-full-context.v2",
            "floats_allowed": False,
        },
    }


def compile_sample_requests(
    proposal: UnitProposal,
    rows: Sequence[Mapping[str, object]],
    *,
    model: str,
) -> list[dict[str, object]]:
    sample = stable_train_sample(rows, salt=SAMPLE_SALT, sample_size=SAMPLE_SIZE)
    requests = []
    for row in sample:
        item_key, ctext = row.get("item_key"), row.get("ctext")
        if not isinstance(item_key, str) or not isinstance(ctext, str):
            raise ValueError("TRAIN items require string item_key and ctext")
        requests.append(compile_request(proposal, item_key=item_key, ctext=ctext, model=model))
    return requests


def parse_response(raw: str, *, ctext: str) -> tuple[Verdict, str]:
    if not isinstance(raw, str) or not raw.strip():
        raise SchemaError("empty construct-probe response")
    candidates: list[tuple[str, str]] = [("strict_json", raw)]
    fences = _FENCE.findall(raw)
    if len(fences) == 1:
        candidates.append(("single_json_fence", fences[0]))
    last_error: Exception | None = None
    for mode, candidate in candidates:
        try:
            verdict = Verdict.from_json(candidate.strip())
            line_count = max(1, len(ctext.splitlines()))
            for witness in verdict.witnesses:
                if witness.path != DOCUMENT_PATH or witness.end_line > line_count:
                    raise SchemaError("witness does not address the supplied full document")
            return verdict, mode
        except (SchemaError, ValueError) as exc:
            last_error = exc
    raise SchemaError(f"construct-probe response failed {PARSER_VERSION}") from last_error


def validate_response_envelope(envelope: Mapping[str, object], request: Mapping[str, object]) -> dict[str, object]:
    if envelope.get("request_sha256") != request.get("request_sha256"):
        raise SchemaError("response/request digest mismatch")
    raw, ctext = envelope.get("raw_response"), request.get("ctext")
    if not isinstance(raw, str) or not isinstance(ctext, str):
        raise SchemaError("response or request text missing")
    verdict, mode = parse_response(raw, ctext=ctext)
    return {
        "request_sha256": request["request_sha256"],
        "unit_id": request["unit_id"],
        "item_key": request["item_key"],
        "split": request["split"],
        "pass_index": request["pass_index"],
        "verdict": verdict.to_json_value(),
        "parser_version": PARSER_VERSION,
        "parse_mode": mode,
    }
