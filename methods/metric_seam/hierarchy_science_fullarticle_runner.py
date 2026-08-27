"""Run the frozen science relation verifier on an additive full-paper representation.

The canonical hierarchy peer-review items contain abstracts only.  Adding body evidence
to the code arm would therefore violate the shared-representation contract.  This module
first measures that blocker, then builds a separate outcome-blind split from the existing
2,400-paper evidence corpus.  Each additive item contains one byte-identical ``ctext``
envelope for any future prompt arm and the current code arm::

    [ABSTRACT]\n...\n\n[EXTRACTED FULL-PAPER BODY]\n...

The body is the historical methods/results/evaluation extraction (capped upstream at
20,000 characters), not necessarily the complete PDF.  Selection does not condition on
body availability: missing bodies remain explicit verifier abstentions.

Only the six construct-fidelity-approved relation-local mappings are eligible.  The
existing strict verifier is source-bound before import and run without prompt outputs,
reference judgements, accept/reject outcomes, external supervision, models, APIs, or
accelerators.  Its certificates establish document-internal numeric/comparative
consistency, never external scientific truth or a whole peer-review judgement.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASK = "peer-review"
CAPABILITY_ID = "science_claims_v2_relation_strict_full_article"

BLOCKER_SCHEMA = "metric-seam.science-canonical-representation-blocker.v1"
ITEM_SCHEMA = "metric-seam.science-fullarticle-shared-items.v1"
EXECUTION_SCHEMA = "metric-seam.science-fullarticle-execution.v1"
GATE_SCHEMA = "metric-seam.science-fullarticle-train-gate.v1"

SELECTION_SALT = "metric-seam-science-fullarticle-shared-items-v1"
N_SELECTED = 300
N_PER_SPLIT = 150
MIN_TRAIN_MEASURED = 30
MIN_TRAIN_COVERAGE = 0.20
MIN_DISTINCT_MEASURED_STATUSES = 2
MAX_TRAIN_FAILED = 0

ABSTRACT_HEADER = "[ABSTRACT]\n"
BODY_HEADER = "\n\n[EXTRACTED FULL-PAPER BODY: METHODS/RESULTS/EVALUATION]\n"
REPRESENTATION_DESCRIPTION = (
    "abstract plus the historical extracted methods/results/evaluation body in one "
    "ctext envelope; body capped upstream at 20,000 characters"
)

DEFAULT_SOURCE = ROOT / "datasets/peer-review/peer_review_cv_evidence.jsonl"
DEFAULT_CANONICAL_ITEMS = (
    ROOT / "outputs/metric_seam_pilot/hierarchy_r123/items_v2/peer-review"
)
DEFAULT_BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
DEFAULT_ITEMS = DEFAULT_BASE / "items_v3/peer-review-fullarticle"
DEFAULT_SEED = DEFAULT_BASE / "peer_review_science_claim_seed_map_v1.json"
DEFAULT_FIDELITY = (
    DEFAULT_BASE / "peer_review_science_claim_construct_fidelity_v1.json"
)
DEFAULT_BLOCKER = (
    DEFAULT_BASE / "peer_review_science_canonical_representation_blocker_v1.json"
)
DEFAULT_TRAIN = DEFAULT_BASE / "peer_review_science_fullarticle_compiler_train_v1.json"
DEFAULT_GATE = DEFAULT_BASE / "peer_review_science_fullarticle_train_gate_v1.json"
DEFAULT_HELDOUT = (
    DEFAULT_BASE / "peer_review_science_fullarticle_heldout_pre_reference_v1.json"
)

_SOURCE_LINE_RE = re.compile(
    r'^(?P<prefix>\s*\{\s*"paper_id"\s*:\s*'
    r'(?:(?:"(?:\\.|[^"\\])*")|(?:-?\d+))\s*,\s*"y"\s*:\s*)'
    r'(?P<outcome>(?:true|false|null|-?\d+(?:\.\d+)?|"(?:\\.|[^"\\])*"))'
    r'(?P<suffix>\s*,)',
)

_THREE_STATES = ("measured", "abstained", "failed")
_MEASURED_VERIFIER_STATUSES = {
    "supported",
    "contradicted",
    "mixed",
    "evidence_link",
    "insufficient",
}


class ScienceExecutionError(ValueError):
    """Raised when an input crosses the additive execution contract."""


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping | Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_outcome_blind_source(path: Path) -> list[dict[str, str]]:
    """Read only paper id, abstract, and body after masking ``y`` before JSON decode.

    The historical file co-locates an accept/reject field.  Its value is replaced with
    JSON null in the raw line before decoding, then the key is removed.  This makes
    selection and execution invariant to the source outcome value rather than merely
    promising not to branch on a decoded label.
    """

    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            match = _SOURCE_LINE_RE.match(line)
            if match is None:
                raise ScienceExecutionError(
                    f"source line {line_number} does not match the outcome-mask contract"
                )
            masked = (
                line[: match.start("outcome")]
                + "null"
                + line[match.end("outcome") :]
            )
            decoded = json.loads(masked)
            if decoded.pop("y", "missing") is not None:
                raise ScienceExecutionError(
                    f"source line {line_number} outcome was not masked before decode"
                )
            if set(decoded) != {"paper_id", "abstract", "body"}:
                raise ScienceExecutionError(
                    f"source line {line_number} has unexpected fields: {sorted(decoded)}"
                )
            paper_id = decoded["paper_id"]
            abstract = decoded["abstract"]
            body = decoded["body"]
            if not isinstance(paper_id, (str, int)):
                raise ScienceExecutionError(
                    f"source line {line_number} has an invalid paper identifier"
                )
            if not isinstance(abstract, str) or not isinstance(body, str):
                raise ScienceExecutionError(
                    f"source line {line_number} has non-string article text"
                )
            rows.append(
                {"paper_id": str(paper_id), "abstract": abstract, "body": body}
            )
    if not rows:
        raise ScienceExecutionError("full-paper evidence source is empty")
    if len({row["paper_id"] for row in rows}) != len(rows):
        raise ScienceExecutionError("full-paper evidence has duplicate paper identifiers")
    return rows


def article_ctext(abstract: str, body: str) -> str:
    """Create the sole representation consumed by code and any future prompt arm."""

    return ABSTRACT_HEADER + abstract.strip() + BODY_HEADER + body.strip()


def parse_article_ctext(ctext: str) -> tuple[str, str]:
    """Recover abstract/body fields from the frozen shared byte representation."""

    if not isinstance(ctext, str) or not ctext.startswith(ABSTRACT_HEADER):
        raise ScienceExecutionError("article ctext is missing the abstract header")
    if ctext.count(BODY_HEADER) != 1:
        raise ScienceExecutionError("article ctext must contain exactly one body header")
    abstract, body = ctext[len(ABSTRACT_HEADER) :].split(BODY_HEADER, 1)
    if not abstract.strip():
        raise ScienceExecutionError("article ctext has an empty abstract")
    return abstract, body


def validate_items(payload: object, *, expected_prefix: str | None = None) -> list[dict]:
    """Require label-free items containing exactly ``item_key`` and ``ctext``."""

    if not isinstance(payload, list) or not payload:
        raise ScienceExecutionError("items must be a nonempty list")
    rows = []
    seen = set()
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping) or set(row) != {"item_key", "ctext"}:
            raise ScienceExecutionError(
                f"item {index} must contain exactly item_key and ctext"
            )
        item_key, ctext = row["item_key"], row["ctext"]
        if (
            not isinstance(item_key, str)
            or not item_key
            or item_key in seen
            or (expected_prefix is not None and not item_key.startswith(expected_prefix))
        ):
            raise ScienceExecutionError(f"item {index} has an invalid item key")
        if not isinstance(ctext, str):
            raise ScienceExecutionError(f"item {index} has non-string ctext")
        parse_article_ctext(ctext)
        seen.add(item_key)
        rows.append({"item_key": item_key, "ctext": ctext})
    return rows


def build_additive_items(source_rows: Sequence[Mapping[str, str]]) -> tuple[dict, list, list]:
    """Freeze 300 outcome-blind items without conditioning on body availability."""

    projected = []
    for row in source_rows:
        if set(row) != {"paper_id", "abstract", "body"}:
            raise ScienceExecutionError("source projection exposed unexpected fields")
        ctext = article_ctext(row["abstract"], row["body"])
        selection_key = hashlib.sha256(
            f"{SELECTION_SALT}\0{ctext}".encode("utf-8")
        ).hexdigest()
        projected.append(
            {
                "selection_key": selection_key,
                "ctext": ctext,
                "body_nonempty": bool(row["body"].strip()),
            }
        )
    by_ctext: dict[str, dict] = {}
    for row in sorted(projected, key=lambda value: (value["selection_key"], value["ctext"])):
        by_ctext.setdefault(row["ctext"], row)
    unique = sorted(
        by_ctext.values(), key=lambda value: (value["selection_key"], value["ctext"])
    )
    if len(unique) < N_SELECTED:
        raise ScienceExecutionError(
            f"only {len(unique)} unique full-paper representations; need {N_SELECTED}"
        )
    selected = unique[:N_SELECTED]
    train_selected = selected[:N_PER_SPLIT]
    heldout_selected = selected[N_PER_SPLIT:]
    train = [
        {"item_key": f"science_train_{index:04d}", "ctext": row["ctext"]}
        for index, row in enumerate(train_selected, start=1)
    ]
    heldout = [
        {"item_key": f"science_heldout_{index:04d}", "ctext": row["ctext"]}
        for index, row in enumerate(heldout_selected, start=1)
    ]
    validate_items(train, expected_prefix="science_train_")
    validate_items(heldout, expected_prefix="science_heldout_")
    manifest = {
        "schema": ITEM_SCHEMA,
        "status": "additive_noncanonical_fullarticle_section_split_frozen",
        "task": TASK,
        "representation": {
            "field": "ctext",
            "description": REPRESENTATION_DESCRIPTION,
            "abstract_header": ABSTRACT_HEADER.rstrip("\n"),
            "body_header": BODY_HEADER.strip("\n"),
            "complete_pdf_claimed": False,
            "body_cap_applied_upstream_chars": 20_000,
            "same_ctext_bytes_required_for_future_prompt_and_code": True,
        },
        "selection": {
            "salt": SELECTION_SALT,
            "rule": (
                "stable SHA-256 order of the permitted abstract+body projection; first 300; "
                "first 150 compiler-train and next 150 sealed heldout"
            ),
            "source_rows_scanned": len(source_rows),
            "unique_projected_rows": len(unique),
            "selected_n": len(selected),
            "compiler_train_n": len(train),
            "sealed_heldout_n": len(heldout),
            "selected_body_nonempty_n": sum(
                row["body_nonempty"] for row in selected
            ),
            "compiler_train_body_nonempty_n": sum(
                row["body_nonempty"] for row in train_selected
            ),
            "sealed_heldout_body_nonempty_n": sum(
                row["body_nonempty"] for row in heldout_selected
            ),
            "conditioned_on_body_availability": False,
            "outcome_or_reference_values_used": False,
            "current_stage_outcome_blind": True,
            "upstream_2400_paper_corpus_historically_outcome_stratified": True,
        },
        "source": {
            "path": "datasets/peer-review/peer_review_cv_evidence.jsonl",
            "historical_builder": "datasets/peer-review/build_fullpaper_evidence.py",
            "permitted_fields": ["paper_id", "abstract", "body"],
            "outcome_field_present_in_historical_source": "y",
            "outcome_value_masked_before_json_decode": True,
            "source_identifiers_emitted": False,
            "upstream_sampling_provenance": (
                "the historical 2,400-paper evidence corpus was sampled in balanced "
                "accept/reject strata before this additive split was designed"
            ),
        },
        "policy": {
            "outcome_fields_emitted": False,
            "reference_fields_emitted": False,
            "compiler_receives_heldout_text": False,
            "external_supervision_used_by_this_split_builder": False,
            "models_or_apis_called": False,
            "accelerators_used": False,
        },
        "comparability": {
            "canonical_hierarchy_items": False,
            "canonical_abstract_only_sample_replaced": False,
            "direct_rate_comparison_to_canonical_hierarchy_execution_permitted": False,
            "reason": (
                "new representation and outcome-blind sample are additive; they repair the "
                "evidence contract but do not retroactively convert abstract-only items"
            ),
        },
        "projection_fingerprints": {
            "compiler_train": _fingerprint(train),
            "sealed_heldout": _fingerprint(heldout),
        },
    }
    return manifest, train, heldout


def build_canonical_representation_blocker(
    canonical_train: Sequence[Mapping],
    canonical_heldout: Sequence[Mapping],
    source_rows: Sequence[Mapping[str, str]],
) -> dict:
    """Quantify why the abstract-only canonical sample cannot run this verifier."""

    source_by_abstract: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in source_rows:
        source_by_abstract[row["abstract"]].append(row)

    def split_summary(rows: Sequence[Mapping]) -> dict:
        if not isinstance(rows, list) or len(rows) != N_PER_SPLIT:
            raise ScienceExecutionError("canonical split must contain exactly 150 items")
        counts = Counter()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != {"item_key", "ctext"}:
                raise ScienceExecutionError(
                    f"canonical item {index} is not a label-free ctext record"
                )
            matches = source_by_abstract.get(row["ctext"], [])
            if not matches:
                counts["no_exact_abstract_join"] += 1
            elif len(matches) > 1:
                counts["ambiguous_exact_abstract_join"] += 1
            elif matches[0]["body"].strip():
                counts["exact_join_with_body"] += 1
            else:
                counts["exact_join_missing_body"] += 1
        for key in (
            "no_exact_abstract_join",
            "ambiguous_exact_abstract_join",
            "exact_join_missing_body",
            "exact_join_with_body",
        ):
            counts.setdefault(key, 0)
        return {
            "n_items": len(rows),
            "join_state_counts": dict(sorted(counts.items())),
            "n_exact_abstract_joins": (
                counts["ambiguous_exact_abstract_join"]
                + counts["exact_join_missing_body"]
                + counts["exact_join_with_body"]
            ),
            "n_exact_joins_with_nonempty_body": counts["exact_join_with_body"],
        }

    train = split_summary(canonical_train)
    heldout = split_summary(canonical_heldout)
    return {
        "schema": BLOCKER_SCHEMA,
        "status": "canonical_execution_blocked_by_representation_mismatch",
        "task": TASK,
        "capability_id": CAPABILITY_ID,
        "canonical_representation": {
            "field": "ctext",
            "content": "abstract only",
            "same_bytes_required_for_prompt_and_code": True,
            "source_identifiers_emitted": False,
        },
        "capability_requires": {
            "abstract": True,
            "distinct_fullpaper_body": True,
            "missing_body_behavior": "abstain",
        },
        "coverage_audit": {
            "compiler_train": train,
            "sealed_heldout": heldout,
            "pooled": {
                "n_items": train["n_items"] + heldout["n_items"],
                "n_exact_abstract_joins": (
                    train["n_exact_abstract_joins"]
                    + heldout["n_exact_abstract_joins"]
                ),
                "n_exact_joins_with_nonempty_body": (
                    train["n_exact_joins_with_nonempty_body"]
                    + heldout["n_exact_joins_with_nonempty_body"]
                ),
            },
        },
        "execution": {
            "performed": False,
            "three_state_outputs": {
                "measured": 0,
                "abstained": 0,
                "failed": 0,
            },
            "why_not": (
                "supplementing canonical ctext with a separately joined body would give the "
                "code arm evidence not present in the shared prompt/code bytes"
            ),
        },
        "forbidden_inputs": {
            "reference_values_loaded": False,
            "outcome_values_loaded": False,
            "prompt_or_reconstruction_outputs_loaded": False,
            "external_supervision_used": False,
        },
        "disposition": {
            "canonical_six_mappings_remain_static_only": True,
            "forced_join_permitted": False,
            "additive_repair": (
                "freeze a new non-comparable abstract+body ctext split before execution"
            ),
        },
    }


def _validate_science_contract(seed: Mapping, fidelity: Mapping) -> list[dict]:
    if seed.get("schema") != "metric-seam.hierarchy-science-claim-seed-map.v1":
        raise ScienceExecutionError("unexpected science seed-map schema")
    if fidelity.get("schema") != "metric-seam.hierarchy-science-claim-construct-fidelity.v1":
        raise ScienceExecutionError("unexpected science construct-fidelity schema")
    if seed.get("task") != TASK or fidelity.get("task") != TASK:
        raise ScienceExecutionError("science execution inputs have the wrong task")
    if fidelity.get("status") != "static-relation-local-adjudication-complete-pre-execution":
        raise ScienceExecutionError("science construct-fidelity audit is not frozen")
    for field in (
        "execution_performed",
        "reference_values_loaded",
        "outcome_labels_loaded",
        "prompt_or_reconstruction_outputs_loaded",
        "external_supervision_loaded_for_this_audit",
    ):
        if fidelity.get(field) is not False:
            raise ScienceExecutionError(
                f"science construct-fidelity input crossed forbidden boundary: {field}"
            )
    inventory = seed.get("capability_inventory")
    if not isinstance(inventory, Mapping) or inventory.get("capability_id") != CAPABILITY_ID:
        raise ScienceExecutionError("science capability identity drifted")
    if inventory.get("channel") != "pure_code" or inventory.get("automatic_discovery") is not False:
        raise ScienceExecutionError("science capability provenance drifted")
    modules = inventory.get("source_modules")
    if not isinstance(modules, list) or len(modules) != 3:
        raise ScienceExecutionError("science capability source inventory is incomplete")
    for module in modules:
        if not isinstance(module, Mapping):
            raise ScienceExecutionError("invalid science capability source record")
        path = ROOT / str(module.get("path"))
        expected = module.get("source_sha256")
        if not path.is_file() or not isinstance(expected, str):
            raise ScienceExecutionError("science capability source is missing")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ScienceExecutionError(
                f"science capability source changed after static audit: {path}"
            )
    rows = fidelity.get("rows")
    if not isinstance(rows, list) or len(rows) != 90:
        raise ScienceExecutionError("science construct-fidelity rows are incomplete")
    eligible = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ScienceExecutionError("invalid science construct-fidelity row")
        relation_local = row.get("verdict") == "partial_relation_local"
        if relation_local != bool(
            row.get("eligible_for_later_relation_local_execution", False)
        ):
            raise ScienceExecutionError("science execution eligibility drifted")
        if relation_local:
            if (
                row.get("candidate_capability_id") != CAPABILITY_ID
                or row.get("eligible_relation_local_depths") != [3]
                or row.get("exact_whole_construct_fidelity") is not False
            ):
                raise ScienceExecutionError("eligible science relation contract drifted")
            eligible.append(
                {
                    "cell_id": row["cell_id"],
                    "level": row["level"],
                    "metric_name": row["metric_name"],
                    "relation_scope": "document_internal_numeric_comparative_consistency",
                    "effective_code_depth": 3,
                }
            )
    if len(eligible) != 6:
        raise ScienceExecutionError(
            f"expected six eligible science relations, found {len(eligible)}"
        )
    return eligible


def _concise_verifier_result(result: Mapping) -> dict:
    certificates = result.get("certificates")
    if not isinstance(certificates, list):
        raise ScienceExecutionError("verifier returned invalid certificates")
    concise_certificates = []
    for certificate in certificates:
        if not isinstance(certificate, Mapping):
            raise ScienceExecutionError("verifier returned an invalid certificate")
        concise_certificates.append(
            {
                "decision": certificate.get("decision"),
                "witness_kind": certificate.get("witness_kind"),
                "reason": certificate.get("reason"),
                "claim": certificate.get("claim"),
                "evidence": certificate.get("evidence"),
                "checks": certificate.get("checks"),
            }
        )
    return {
        "verifier_status": result.get("status"),
        "reason": result.get("reason"),
        "claim_count": result.get("claim_count"),
        "certificate_count": result.get("certificate_count"),
        "evidence_link_count": result.get("evidence_link_count"),
        "decision_counts": result.get("decision_counts", {}),
        "graph": result.get("graph"),
        "relation_certificates": concise_certificates,
    }


def _summarize_execution(rows: Sequence[Mapping], *, n_relations: int) -> dict:
    states = Counter(str(row["measurement_state"]) for row in rows)
    state_counts = {state: states[state] for state in _THREE_STATES}
    statuses = Counter(
        str(row["verifier_status"])
        for row in rows
        if row["measurement_state"] == "measured"
    )
    n_certificates = sum(
        int(row.get("certificate_count") or 0)
        for row in rows
        if row["measurement_state"] == "measured"
    )
    return {
        "n_unique_item_executions": len(rows),
        "n_relation_mappings": n_relations,
        "n_mapping_item_applications": len(rows) * n_relations,
        "three_state_totals_unique_items": state_counts,
        "three_state_totals_mapping_applications": {
            state: count * n_relations for state, count in state_counts.items()
        },
        "measured_coverage": round(state_counts["measured"] / len(rows), 6),
        "measured_verifier_status_counts": dict(sorted(statuses.items())),
        "n_distinct_measured_verifier_statuses": len(statuses),
        "n_relation_certificates": n_certificates,
        "n_items_with_relation_certificate": sum(
            bool(row.get("certificate_count"))
            for row in rows
            if row["measurement_state"] == "measured"
        ),
    }


def _validate_execution_rows(rows: object) -> list[Mapping]:
    """Validate the retained three-state rows before any gate consumes them."""

    if not isinstance(rows, list) or len(rows) != N_PER_SPLIT:
        raise ScienceExecutionError("science execution must retain exactly 150 rows")
    item_keys = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ScienceExecutionError(f"science execution row {index} is not an object")
        item_key = row.get("item_key")
        state = row.get("measurement_state")
        status = row.get("verifier_status")
        if not isinstance(item_key, str) or not item_key:
            raise ScienceExecutionError(f"science execution row {index} has no item key")
        if state not in _THREE_STATES:
            raise ScienceExecutionError(
                f"science execution row {index} has an invalid three-state value"
            )
        if state == "measured" and status not in _MEASURED_VERIFIER_STATUSES:
            raise ScienceExecutionError(
                f"science measured row {index} has an invalid verifier status"
            )
        if state == "abstained" and status != "abstain":
            raise ScienceExecutionError(
                f"science abstained row {index} has an invalid verifier status"
            )
        if state == "failed" and status != "execution_error":
            raise ScienceExecutionError(
                f"science failed row {index} has an invalid verifier status"
            )
        for count_field in (
            "claim_count",
            "certificate_count",
            "evidence_link_count",
        ):
            value = row.get(count_field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ScienceExecutionError(
                    f"science execution row {index} has invalid {count_field}"
                )
        certificates = row.get("relation_certificates")
        if (
            not isinstance(certificates, list)
            or len(certificates) != row.get("certificate_count")
        ):
            raise ScienceExecutionError(
                f"science execution row {index} certificate accounting drifted"
            )
        item_keys.append(item_key)
    if len(set(item_keys)) != len(item_keys):
        raise ScienceExecutionError("science execution has duplicate item keys")
    return rows


def execute_items(
    items: Sequence[Mapping],
    seed: Mapping,
    fidelity: Mapping,
    *,
    phase: str,
    items_path: str,
    verifier: Callable[[str, str, str], Mapping] | None = None,
    self_check: Callable[[], Mapping[str, bool]] | None = None,
) -> dict:
    """Execute one frozen split and retain measured/abstained/failed states."""

    if phase not in {"compiler_train", "heldout_pre_reference"}:
        raise ScienceExecutionError("invalid science execution phase")
    prefix = "science_train_" if phase == "compiler_train" else "science_heldout_"
    normalized = validate_items(list(items), expected_prefix=prefix)
    if len(normalized) != N_PER_SPLIT:
        raise ScienceExecutionError("science execution split must contain 150 items")
    eligible = _validate_science_contract(seed, fidelity)
    rows = []
    checks: dict[str, bool] = {}
    old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        if verifier is None or self_check is None:
            strict = importlib.import_module(
                "methods.metric_seam.science_claims_v2.core_relation_strict"
            )
            verifier = strict.verify_document
            self_check = strict.metamorphic_self_check
        checks = dict(self_check())
        if not checks or not all(value is True for value in checks.values()):
            raise ScienceExecutionError("strict science verifier self-check failed")
        for item in normalized:
            abstract, body = parse_article_ctext(item["ctext"])
            try:
                raw = verifier(item["item_key"], abstract, body)
                if not isinstance(raw, Mapping):
                    raise ScienceExecutionError("verifier returned a non-object")
                status = raw.get("status")
                if status == "abstain":
                    rows.append(
                        {
                            "item_key": item["item_key"],
                            "measurement_state": "abstained",
                            "verifier_status": "abstain",
                            "reason": raw.get("reason"),
                            "claim_count": raw.get("claim_count", 0),
                            "certificate_count": 0,
                            "evidence_link_count": 0,
                            "decision_counts": {},
                            "graph": raw.get("graph"),
                            "relation_certificates": [],
                        }
                    )
                elif status in _MEASURED_VERIFIER_STATUSES:
                    rows.append(
                        {
                            "item_key": item["item_key"],
                            "measurement_state": "measured",
                            **_concise_verifier_result(raw),
                        }
                    )
                else:
                    raise ScienceExecutionError(
                        f"verifier returned unsupported status: {status!r}"
                    )
            except Exception as error:  # Three-state execution must retain failures.
                rows.append(
                    {
                        "item_key": item["item_key"],
                        "measurement_state": "failed",
                        "verifier_status": "execution_error",
                        "reason": "trusted_verifier_exception",
                        "error_type": type(error).__name__,
                        "claim_count": 0,
                        "certificate_count": 0,
                        "evidence_link_count": 0,
                        "decision_counts": {},
                        "graph": None,
                        "relation_certificates": [],
                    }
                )
    finally:
        if old_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda

    _validate_execution_rows(rows)
    result = {
        "schema": EXECUTION_SCHEMA,
        "status": "execution_complete_pre_prompt_pre_reference",
        "phase": phase,
        "task": TASK,
        "items_path": items_path,
        "representation": {
            "description": REPRESENTATION_DESCRIPTION,
            "same_ctext_bytes_for_future_prompt_and_code": True,
            "complete_pdf_claimed": False,
            "canonical_hierarchy_items": False,
            "upstream_corpus_historically_outcome_stratified": True,
            "current_split_and_execution_outcome_blind": True,
        },
        "capability": {
            "capability_id": CAPABILITY_ID,
            "historical_construction": "retrospective_manually_designed_seed",
            "channel": "pure_code",
            "relation_scope": "document_internal_numeric_comparative_consistency",
            "maximum_effective_code_depth": 3,
            "automatic_discovery_claimed": False,
        },
        "relation_mappings": eligible,
        "metamorphic_self_check": checks,
        "execution_policy": {
            "reference_values_loaded": False,
            "outcome_values_loaded": False,
            "prompt_or_reconstruction_outputs_loaded": False,
            "external_supervision_used": False,
            "models_or_apis_called": False,
            "accelerators_used": False,
            "cuda_devices_masked_during_execution": True,
            "worker_process_isolated": False,
            "trusted_source_bound_in_process_verifier": True,
        },
        "claim_limits": [
            "The result is not directly comparable to the canonical abstract-only hierarchy sample.",
            "A measured output is executable document-internal evidence, not external scientific truth.",
            "An abstention or failed run is bounded non-measurement, not evidence of tacitness.",
            "No whole peer-review construct, prompt articulability, reconstruction, or isomorphism is measured.",
            "The represented body is an upstream-capped section extraction, not necessarily the complete PDF.",
            "The upstream 2,400-paper corpus was historically outcome-stratified; this run masks outcomes and uses none as a target or gate.",
        ],
        "summary": _summarize_execution(rows, n_relations=len(eligible)),
        "rows": rows,
    }
    return result


def build_train_gate(train_execution: Mapping) -> dict:
    """Select all six mappings only from target-free train measurability."""

    if train_execution.get("schema") != EXECUTION_SCHEMA:
        raise ScienceExecutionError("unexpected science train execution schema")
    if train_execution.get("phase") != "compiler_train":
        raise ScienceExecutionError("science train gate received a non-train execution")
    policy = train_execution.get("execution_policy")
    if not isinstance(policy, Mapping) or any(
        policy.get(field) is not False
        for field in (
            "reference_values_loaded",
            "outcome_values_loaded",
            "prompt_or_reconstruction_outputs_loaded",
            "external_supervision_used",
            "models_or_apis_called",
            "accelerators_used",
        )
    ):
        raise ScienceExecutionError("science train execution crossed a forbidden input")
    summary = train_execution.get("summary")
    relations = train_execution.get("relation_mappings")
    if not isinstance(summary, Mapping) or not isinstance(relations, list) or len(relations) != 6:
        raise ScienceExecutionError("science train execution is incomplete")
    rows = _validate_execution_rows(train_execution.get("rows"))
    expected_summary = _summarize_execution(rows, n_relations=len(relations))
    if summary != expected_summary:
        raise ScienceExecutionError("science train execution summary drifted from its rows")
    states = summary.get("three_state_totals_unique_items")
    if not isinstance(states, Mapping) or set(states) != set(_THREE_STATES):
        raise ScienceExecutionError("science train three-state accounting is invalid")
    n_measured = states["measured"]
    n_failed = states["failed"]
    coverage = summary.get("measured_coverage")
    n_distinct = summary.get("n_distinct_measured_verifier_statuses")
    criteria = {
        "minimum_measured_items": {
            "threshold": MIN_TRAIN_MEASURED,
            "observed": n_measured,
            "passes": n_measured >= MIN_TRAIN_MEASURED,
        },
        "minimum_measured_coverage": {
            "threshold": MIN_TRAIN_COVERAGE,
            "observed": coverage,
            "passes": isinstance(coverage, (int, float))
            and coverage >= MIN_TRAIN_COVERAGE,
        },
        "minimum_distinct_measured_statuses": {
            "threshold": MIN_DISTINCT_MEASURED_STATUSES,
            "observed": n_distinct,
            "passes": isinstance(n_distinct, int)
            and n_distinct >= MIN_DISTINCT_MEASURED_STATUSES,
        },
        "maximum_failed_items": {
            "threshold": MAX_TRAIN_FAILED,
            "observed": n_failed,
            "passes": n_failed <= MAX_TRAIN_FAILED,
        },
    }
    selected = all(criterion["passes"] for criterion in criteria.values())
    selected_relations = relations if selected else []
    return {
        "schema": GATE_SCHEMA,
        "status": "train_only_gate_frozen",
        "task": TASK,
        "selection_basis": (
            "compiler-train measured coverage, execution failure count, and output-state "
            "nondegeneracy only"
        ),
        "forbidden_selection_inputs": {
            "heldout_items_or_outputs_loaded": False,
            "reference_values_loaded": False,
            "outcome_values_loaded": False,
            "prompt_or_reconstruction_outputs_loaded": False,
            "certificate_polarity_or_target_agreement_used": False,
        },
        "criteria": criteria,
        "selected": selected,
        "selected_relation_mappings": selected_relations,
        "summary": {
            "n_candidate_relation_mappings": len(relations),
            "n_selected_relation_mappings": len(selected_relations),
            "n_train_measured_items": n_measured,
            "n_train_abstained_items": states["abstained"],
            "n_train_failed_items": n_failed,
        },
        "claim_limit": (
            "Passing this gate establishes only train-operational measurability for a "
            "relation-local code program on the additive representation."
        ),
    }


def _gate_matches_relations(gate: Mapping, eligible: Sequence[Mapping]) -> None:
    if gate.get("schema") != GATE_SCHEMA or gate.get("selected") is not True:
        raise ScienceExecutionError("science heldout execution requires a passed train gate")
    selected = gate.get("selected_relation_mappings")
    if not isinstance(selected, list) or selected != list(eligible):
        raise ScienceExecutionError("science heldout gate relation mappings drifted")
    forbidden = gate.get("forbidden_selection_inputs")
    if not isinstance(forbidden, Mapping) or set(forbidden.values()) != {False}:
        raise ScienceExecutionError("science train gate used forbidden selection inputs")


def build_heldout_execution(
    items: Sequence[Mapping],
    seed: Mapping,
    fidelity: Mapping,
    gate: Mapping,
    *,
    items_path: str,
) -> dict:
    """Validate the train-only gate, then execute sealed heldout pre-reference."""

    eligible = _validate_science_contract(seed, fidelity)
    _gate_matches_relations(gate, eligible)
    result = execute_items(
        items,
        seed,
        fidelity,
        phase="heldout_pre_reference",
        items_path=items_path,
    )
    result["train_gate"] = {
        "schema": gate["schema"],
        "selected": True,
        "n_selected_relation_mappings": len(gate["selected_relation_mappings"]),
        "selection_used_heldout": False,
    }
    return result


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def build_and_write_items(
    source_path: Path,
    canonical_items_root: Path,
    items_root: Path,
    blocker_path: Path,
) -> None:
    source_rows = load_outcome_blind_source(source_path)
    canonical_train = _load_json(canonical_items_root / "compiler_train.json")
    canonical_heldout = _load_json(canonical_items_root / "sealed_heldout.json")
    blocker = build_canonical_representation_blocker(
        canonical_train, canonical_heldout, source_rows
    )
    manifest, train, heldout = build_additive_items(source_rows)
    manifest["source"]["path"] = _relative(source_path)
    manifest["comparability"]["canonical_items_path"] = _relative(
        canonical_items_root
    )
    _write_json(blocker_path, blocker)
    _write_json(items_root / "manifest.json", manifest)
    _write_json(items_root / "compiler_train.json", train)
    _write_json(items_root / "sealed_heldout.json", heldout)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-items")
    build.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    build.add_argument(
        "--canonical-items-root", type=Path, default=DEFAULT_CANONICAL_ITEMS
    )
    build.add_argument("--items-root", type=Path, default=DEFAULT_ITEMS)
    build.add_argument("--blocker-out", type=Path, default=DEFAULT_BLOCKER)

    train = subparsers.add_parser("run-train")
    train.add_argument("--items-root", type=Path, default=DEFAULT_ITEMS)
    train.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    train.add_argument("--fidelity", type=Path, default=DEFAULT_FIDELITY)
    train.add_argument("--out", type=Path, default=DEFAULT_TRAIN)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    gate.add_argument("--out", type=Path, default=DEFAULT_GATE)

    heldout = subparsers.add_parser("run-heldout")
    heldout.add_argument("--items-root", type=Path, default=DEFAULT_ITEMS)
    heldout.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    heldout.add_argument("--fidelity", type=Path, default=DEFAULT_FIDELITY)
    heldout.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    heldout.add_argument("--out", type=Path, default=DEFAULT_HELDOUT)

    args = parser.parse_args()
    if args.command == "build-items":
        build_and_write_items(
            args.source.resolve(),
            args.canonical_items_root.resolve(),
            args.items_root.resolve(),
            args.blocker_out.resolve(),
        )
        print(f"wrote {args.blocker_out} and {args.items_root}")
    elif args.command == "run-train":
        items = _load_json(args.items_root / "compiler_train.json")
        result = execute_items(
            items,
            _load_json(args.seed),
            _load_json(args.fidelity),
            phase="compiler_train",
            items_path=_relative(args.items_root / "compiler_train.json"),
        )
        _write_json(args.out, result)
        print(f"wrote {args.out}")
    elif args.command == "gate":
        result = build_train_gate(_load_json(args.train))
        _write_json(args.out, result)
        print(f"wrote {args.out}")
    elif args.command == "run-heldout":
        result = build_heldout_execution(
            _load_json(args.items_root / "sealed_heldout.json"),
            _load_json(args.seed),
            _load_json(args.fidelity),
            _load_json(args.gate),
            items_path=_relative(args.items_root / "sealed_heldout.json"),
        )
        _write_json(args.out, result)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
