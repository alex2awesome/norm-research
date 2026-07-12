"""Channel-faithful construct-contract checker (additive instrument, schema v1).

This module intentionally does *not* replace ``contract_check.py``.  The historical
checker is part of the audit trail.  This checker has a different estimand:

* ``CODE`` probes test code-based verifiability without prompt-produced fields.
* ``L`` probes test prompt-based articulability only when a frozen extraction artifact
  is supplied and cryptographically bound to this exact contract and probe text.
* The CODE and HYBRID gates are reported separately.  A missing L extraction is an
  explicit abstention, never a failed CODE probe.

Both artifacts are pinned by canonical SHA-256 (JSON key order and whitespace do not
matter).  A full hybrid PASS additionally requires the configured L-extraction coverage;
the default is 100%.  Probe scores must be finite, lie in [0, 1], and clear an explicit
positive margin.  Adding an irrelevant field must not change a score, which retains the
historical probe-mode fingerprint guard without conflating a real declared L field with
probe mode.

Frozen extraction artifact schema::

    {
      "schema_version": "metric-seam-probe-extractions-v1",
      "contract_sha256": "<64 hex chars>",
      "extractor_manifest_sha256": "<64 hex chars>",
      "probes": [
        {
          "index": 3,
          "available": true,
          "text_pos_sha256": "...",
          "text_neg_sha256": "...",
          "pos": {"field_name": 0.9},
          "neg": {"field_name": 0.1}
        }
      ]
    }

The extractor-manifest digest binds model/prompt/backend configuration without putting
those mutable details in this checker.  The extraction artifact itself must also be
supplied with its expected canonical digest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Optional


SCHEMA_VERSION = "metric-seam-probe-extractions-v1"
CHANNELS = frozenset({"CODE", "L"})
PROBE_CAPABILITIES = frozenset({"base", "math", "capability"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SENTINEL_KEY = "__metric_seam_irrelevant_field_6f44f1f6__"
_LABEL_ASSIGNMENT_RE = re.compile(
    r"\b(?:judg(?:e|ement|ment)|score|gold|label|target|ground[ _-]?truth)\s*"
    r"(?:=|:)\s*(?:[-+]?\.?\d|true\b|false\b|yes\b|no\b|\[)",
    re.IGNORECASE,
)


class ContractIntegrityError(ValueError):
    """The contract or extraction artifact is not the frozen artifact requested."""


class ContractSchemaError(ValueError):
    """The contract or extraction artifact cannot be interpreted unambiguously."""


def build_probe_ops(capabilities: Iterable[str]) -> tuple[Any, tuple[str, ...]]:
    """Build the blind lane's computation-only operation view for synthetic probes.

    ``SplitScopedOps`` is the single v2 capability allowlist.  Synthetic contract probes
    have no legitimate retrieval population, so this bridge intentionally permits only
    ``base``, ``math``, and conservative ``capability`` operations.  Retrieval must run
    through a prepared blind compiler bundle, whose TRAIN-only corpus and current opaque
    item key make exclusion structural.
    """

    allowed = tuple(sorted(set(capabilities)))
    unknown = set(allowed) - PROBE_CAPABILITIES
    if unknown:
        raise ValueError(
            f"probe capabilities {sorted(unknown)} are forbidden; allowed="
            f"{sorted(PROBE_CAPABILITIES)}. Retrieval requires a TRAIN-scoped bundle."
        )
    here = pathlib.Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from split_ops_v2 import SplitScopedOps  # type: ignore

    # The dummy row exists only because SplitScopedOps binds every operation view to an
    # opaque current item. Retrieval is absent, so no corpus-derived state is constructed.
    owner = SplitScopedOps({"contract_probe": "[synthetic contract probe]"}, allowed)
    return owner.for_item("contract_probe"), allowed


@dataclass(frozen=True)
class CheckConfig:
    """Frozen thresholds for one contract-check run."""

    separation_fraction: float = 0.75
    min_margin: float = 0.01
    required_l_coverage: float = 1.0
    train_min_completeness: float = 0.90
    train_min_items: int = 40
    mode_round_decimals: int = 6

    def validate(self) -> None:
        for name, value in (
            ("separation_fraction", self.separation_fraction),
            ("required_l_coverage", self.required_l_coverage),
            ("train_min_completeness", self.train_min_completeness),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not math.isfinite(self.min_margin) or self.min_margin <= 0.0:
            raise ValueError("min_margin must be finite and > 0")
        if self.train_min_items < 0:
            raise ValueError("train_min_items must be >= 0")


@dataclass(frozen=True)
class FrozenProbeExtraction:
    index: int
    available: bool
    text_pos_sha256: str
    text_neg_sha256: str
    pos: Optional[Mapping[str, Any]] = None
    neg: Optional[Mapping[str, Any]] = None
    unavailable_reason: Optional[str] = None


@dataclass(frozen=True)
class ScoreObservation:
    value: Optional[float]
    error: Optional[str] = None


@dataclass(frozen=True)
class ProbeResult:
    index: int
    channel: str
    outcome: str
    score_pos: Optional[float]
    score_neg: Optional[float]
    delta: Optional[float]
    available: bool
    mode_detected: bool = False
    detail: Optional[str] = None


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    passed: Optional[bool]
    conditional_passed: Optional[bool]
    n_declared: int
    n_eligible: int
    n_separated: int
    n_inverted: int
    n_invalid: int
    n_abstained: int
    separation_fraction: Optional[float]
    l_coverage: Optional[float]
    reason: str


@dataclass(frozen=True)
class DiscriminationResult:
    status: str
    passed: Optional[bool]
    n_items: int
    n_scored: int
    completeness: Optional[float]
    std: Optional[float]
    frac_at_mode: Optional[float]
    n_invalid: int
    n_mode_detected: int
    reason: str


@dataclass(frozen=True)
class ContractCheckResult:
    contract_sha256: str
    extraction_sha256: Optional[str]
    config: CheckConfig
    probes: tuple[ProbeResult, ...]
    code_gate: GateResult
    hybrid_gate: GateResult
    discrimination_gate: DiscriminationResult

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON used by every integrity check in this module."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractIntegrityError(f"artifact is not canonicalizable JSON: {exc}") from exc
    return encoded.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ContractIntegrityError(f"{name} must be a lowercase, full SHA-256 digest")


def validate_contract(contract: Mapping[str, Any], *, strict_blind: bool = True) -> None:
    """Validate a relation contract for the blind reconstruction lane.

    In strict mode, probe rationales may describe corpus phenomena but may not carry a
    numeric judge/score/label assignment.  This is deliberately narrower than a vague
    keyword ban: phrases such as "the score should increase" remain legal, while
    ``judge=.4`` and ``judgement=0`` are rejected.
    """

    probes = contract.get("cf_probes")
    if not isinstance(probes, list) or not probes:
        raise ContractSchemaError("contract.cf_probes must be a non-empty list")
    for index, probe in enumerate(probes):
        if not isinstance(probe, Mapping):
            raise ContractSchemaError(f"probe {index} must be an object")
        for key in ("text_pos", "text_neg", "channel"):
            if key not in probe:
                raise ContractSchemaError(f"probe {index} is missing {key}")
        if not isinstance(probe["text_pos"], str) or not probe["text_pos"]:
            raise ContractSchemaError(f"probe {index}.text_pos must be non-empty text")
        if not isinstance(probe["text_neg"], str) or not probe["text_neg"]:
            raise ContractSchemaError(f"probe {index}.text_neg must be non-empty text")
        if probe["text_pos"] == probe["text_neg"]:
            raise ContractSchemaError(f"probe {index} has identical positive and negative text")
        if probe["channel"] not in CHANNELS:
            raise ContractSchemaError(
                f"probe {index}.channel must be exactly CODE or L, got {probe['channel']!r}"
            )
        if strict_blind:
            for key in ("why", "corpus_phenomenon"):
                value = probe.get(key)
                if isinstance(value, str) and _LABEL_ASSIGNMENT_RE.search(value):
                    raise ContractSchemaError(
                        f"probe {index}.{key} contains a label-bearing numeric assignment; "
                        "blind-v2 contracts must be reconstructed without judge labels"
                    )


def verify_contract_hash(contract: Mapping[str, Any], expected_sha256: str) -> str:
    _require_sha256("expected contract hash", expected_sha256)
    actual = canonical_json_sha256(contract)
    if actual != expected_sha256:
        raise ContractIntegrityError(
            f"contract hash mismatch: expected {expected_sha256}, actual {actual}"
        )
    return actual


def load_frozen_extractions(
    payload: Mapping[str, Any],
    *,
    expected_sha256: str,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> tuple[dict[int, FrozenProbeExtraction], str]:
    """Verify and parse an extraction artifact bound to ``contract``."""

    _require_sha256("expected extraction hash", expected_sha256)
    actual_sha = canonical_json_sha256(payload)
    if actual_sha != expected_sha256:
        raise ContractIntegrityError(
            f"extraction hash mismatch: expected {expected_sha256}, actual {actual_sha}"
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ContractSchemaError(
            f"extraction schema must be {SCHEMA_VERSION!r}"
        )
    if payload.get("contract_sha256") != contract_sha256:
        raise ContractIntegrityError("extractions are bound to a different contract")
    manifest_sha = payload.get("extractor_manifest_sha256")
    _require_sha256("extractor_manifest_sha256", manifest_sha)
    rows = payload.get("probes")
    if not isinstance(rows, list):
        raise ContractSchemaError("extraction.probes must be a list")

    parsed: dict[int, FrozenProbeExtraction] = {}
    probes = contract["cf_probes"]
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("index"), int):
            raise ContractSchemaError("every extraction row needs an integer index")
        index = row["index"]
        if index in parsed:
            raise ContractSchemaError(f"duplicate extraction row for probe {index}")
        if index < 0 or index >= len(probes):
            raise ContractSchemaError(f"extraction index {index} is outside this contract")
        if probes[index]["channel"] != "L":
            raise ContractSchemaError(f"probe {index} is CODE and must not have L extractions")
        available = row.get("available")
        if not isinstance(available, bool):
            raise ContractSchemaError(f"probe {index}.available must be boolean")
        pos_hash = row.get("text_pos_sha256")
        neg_hash = row.get("text_neg_sha256")
        if pos_hash != text_sha256(probes[index]["text_pos"]):
            raise ContractIntegrityError(f"probe {index} positive text hash mismatch")
        if neg_hash != text_sha256(probes[index]["text_neg"]):
            raise ContractIntegrityError(f"probe {index} negative text hash mismatch")
        pos, neg = row.get("pos"), row.get("neg")
        if available and (not isinstance(pos, Mapping) or not isinstance(neg, Mapping)):
            raise ContractSchemaError(
                f"available probe {index} requires pos and neg extraction objects"
            )
        if not available and (pos is not None or neg is not None):
            raise ContractSchemaError(
                f"unavailable probe {index} must not carry pos/neg extractions"
            )
        parsed[index] = FrozenProbeExtraction(
            index=index,
            available=available,
            text_pos_sha256=pos_hash,
            text_neg_sha256=neg_hash,
            pos=dict(pos) if isinstance(pos, Mapping) else None,
            neg=dict(neg) if isinstance(neg, Mapping) else None,
            unavailable_reason=row.get("unavailable_reason"),
        )
    return parsed, actual_sha


def _call_score(
    score: Callable[[str, Mapping[str, Any], Any], Any],
    text: str,
    extracted: Mapping[str, Any],
    ops: Any,
) -> ScoreObservation:
    try:
        raw = score(text, dict(extracted), ops)
    except Exception as exc:  # a probe error is data, not a checker crash
        return ScoreObservation(None, f"{type(exc).__name__}: {exc}")
    if raw is None:
        return ScoreObservation(None, "returned None")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return ScoreObservation(None, f"non-numeric score {type(raw).__name__}")
    value = float(raw)
    if not math.isfinite(value):
        return ScoreObservation(None, "score is not finite")
    if not 0.0 <= value <= 1.0:
        return ScoreObservation(None, f"score {value!r} is outside [0, 1]")
    return ScoreObservation(value)


def _stable_call(
    score: Callable[[str, Mapping[str, Any], Any], Any],
    text: str,
    extracted: Mapping[str, Any],
    ops: Any,
) -> tuple[ScoreObservation, bool]:
    if _SENTINEL_KEY in extracted:
        raise ContractSchemaError(f"reserved extraction key {_SENTINEL_KEY!r} is present")
    first = _call_score(score, text, extracted, ops)
    augmented = dict(extracted)
    augmented[_SENTINEL_KEY] = None
    second = _call_score(score, text, augmented, ops)
    mode_detected = first != second
    return first, mode_detected


def _evaluate_pair(
    *,
    index: int,
    channel: str,
    text_pos: str,
    text_neg: str,
    extracted_pos: Optional[Mapping[str, Any]],
    extracted_neg: Optional[Mapping[str, Any]],
    available: bool,
    unavailable_reason: Optional[str],
    score: Callable[[str, Mapping[str, Any], Any], Any],
    ops: Any,
    min_margin: float,
) -> ProbeResult:
    if not available:
        return ProbeResult(
            index=index,
            channel=channel,
            outcome="ABSTAIN",
            score_pos=None,
            score_neg=None,
            delta=None,
            available=False,
            detail=unavailable_reason or "frozen L extraction unavailable",
        )
    pos, pos_mode = _stable_call(score, text_pos, extracted_pos or {}, ops)
    neg, neg_mode = _stable_call(score, text_neg, extracted_neg or {}, ops)
    mode_detected = pos_mode or neg_mode
    if mode_detected:
        return ProbeResult(
            index, channel, "INVALID", pos.value, neg.value, None, True, True,
            "score changed when an irrelevant extraction field was added",
        )
    if pos.error or neg.error:
        return ProbeResult(
            index, channel, "INVALID", pos.value, neg.value, None, True, False,
            "; ".join(x for x in (pos.error, neg.error) if x),
        )
    assert pos.value is not None and neg.value is not None
    delta = pos.value - neg.value
    if delta < 0.0:
        outcome = "INVERTED"
    elif delta >= min_margin:
        outcome = "SEPARATED"
    else:
        outcome = "BELOW_MARGIN"
    return ProbeResult(
        index, channel, outcome, pos.value, neg.value, delta, True, False, None
    )


def _build_gate(
    name: str,
    results: Iterable[ProbeResult],
    *,
    separation_fraction_required: float,
    l_declared: int,
    l_available: int,
    required_l_coverage: Optional[float],
) -> GateResult:
    rows = tuple(results)
    if not rows:
        return GateResult(
            name, "NOT_APPLICABLE", None, None, 0, 0, 0, 0, 0, 0,
            None, None, "no probes declared for this gate",
        )
    eligible = [row for row in rows if row.available]
    n_sep = sum(row.outcome == "SEPARATED" for row in eligible)
    n_inv = sum(row.outcome == "INVERTED" for row in eligible)
    n_invalid = sum(row.outcome == "INVALID" for row in eligible)
    n_abstained = len(rows) - len(eligible)
    sep_fraction = n_sep / len(eligible) if eligible else None
    conditional = bool(
        eligible
        and not n_inv
        and not n_invalid
        and sep_fraction is not None
        and sep_fraction >= separation_fraction_required
    )
    l_coverage = (l_available / l_declared) if l_declared else None

    if n_invalid:
        status, passed = "FAIL", False
        reason = f"{n_invalid} eligible probe(s) returned invalid or mode-dependent scores"
    elif n_inv:
        status, passed = "FAIL", False
        reason = f"{n_inv} eligible probe(s) were inverted"
    elif required_l_coverage is not None and l_declared and l_coverage < required_l_coverage:
        status, passed = "ABSTAIN", None
        reason = (
            f"L extraction coverage {l_coverage:.3f} is below required "
            f"{required_l_coverage:.3f}; conditional result={conditional}"
        )
    elif not eligible:
        status, passed = "ABSTAIN", None
        reason = "no eligible frozen probes"
    elif conditional:
        status, passed = "PASS", True
        reason = "finite in-range scores cleared inversion and separation gates"
    else:
        status, passed = "FAIL", False
        reason = (
            f"separation fraction {sep_fraction:.3f} is below required "
            f"{separation_fraction_required:.3f}"
        )
    return GateResult(
        name=name,
        status=status,
        passed=passed,
        conditional_passed=conditional if eligible else None,
        n_declared=len(rows),
        n_eligible=len(eligible),
        n_separated=n_sep,
        n_inverted=n_inv,
        n_invalid=n_invalid,
        n_abstained=n_abstained,
        separation_fraction=sep_fraction,
        l_coverage=l_coverage,
        reason=reason,
    )


def _not_run_discrimination() -> DiscriminationResult:
    return DiscriminationResult(
        "NOT_RUN", None, 0, 0, None, None, None, 0, 0,
        "no unlabeled corpus cases were supplied",
    )


def check_discrimination(
    score: Callable[[str, Mapping[str, Any], Any], Any],
    cases: Iterable[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    default_ops: Any,
    config: CheckConfig,
) -> DiscriminationResult:
    """Nondegeneracy check over unlabeled reconstruction inputs only."""

    rows = list(cases)
    values: list[float] = []
    n_invalid = 0
    n_mode = 0
    for index, case in enumerate(rows):
        unexpected = set(case) - {"text", "extracted", "ops"}
        if unexpected:
            raise ContractSchemaError(
                f"discrimination case {index} has forbidden metadata keys {sorted(unexpected)}"
            )
        if "text" not in case:
            raise ContractSchemaError(f"discrimination case {index} has no text")
        extracted = case.get("extracted", {})
        if not isinstance(extracted, Mapping):
            raise ContractSchemaError(f"discrimination case {index}.extracted is not a mapping")
        observation, mode = _stable_call(
            score, case["text"], extracted, case.get("ops", default_ops)
        )
        n_mode += int(mode)
        if mode or observation.error:
            n_invalid += int(observation.error is not None)
            continue
        assert observation.value is not None
        values.append(observation.value)

    n_items = len(rows)
    completeness = len(values) / n_items if n_items else 0.0
    rounded = [round(x, config.mode_round_decimals) for x in values]
    if values:
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        frac_mode = Counter(rounded).most_common(1)[0][1] / len(values)
    else:
        std, frac_mode = None, None
    checks = contract.get("discrimination_checks")
    if not isinstance(checks, Mapping):
        raise ContractSchemaError("contract.discrimination_checks must be an object")
    try:
        min_std = float(checks["min_std"])
        max_frac_mode = float(checks["max_frac_at_mode"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractSchemaError(
            "discrimination_checks needs numeric min_std and max_frac_at_mode"
        ) from exc

    reasons = []
    if n_mode:
        reasons.append(f"{n_mode} mode-dependent score(s)")
    if n_invalid:
        reasons.append(f"{n_invalid} invalid score(s)")
    if n_items < config.train_min_items:
        reasons.append(f"only {n_items} cases (minimum {config.train_min_items})")
    if completeness < config.train_min_completeness:
        reasons.append(
            f"completeness {completeness:.3f} < {config.train_min_completeness:.3f}"
        )
    if std is None or std < min_std:
        reasons.append(f"std {std!r} < {min_std}")
    if frac_mode is None or frac_mode > max_frac_mode:
        reasons.append(f"frac_at_mode {frac_mode!r} > {max_frac_mode}")
    passed = not reasons
    return DiscriminationResult(
        "PASS" if passed else "FAIL",
        passed,
        n_items,
        len(values),
        completeness,
        std,
        frac_mode,
        n_invalid,
        n_mode,
        "all unlabeled nondegeneracy gates passed" if passed else "; ".join(reasons),
    )


def check_contract(
    contract: Mapping[str, Any],
    *,
    expected_contract_sha256: str,
    score: Callable[[str, Mapping[str, Any], Any], Any],
    config: CheckConfig,
    ops: Any = None,
    extraction_payload: Optional[Mapping[str, Any]] = None,
    expected_extraction_sha256: Optional[str] = None,
    discrimination_cases: Optional[Iterable[Mapping[str, Any]]] = None,
    strict_blind: bool = True,
) -> ContractCheckResult:
    """Run channel-faithful CODE and HYBRID gates against one frozen contract."""

    config.validate()
    validate_contract(contract, strict_blind=strict_blind)
    contract_sha = verify_contract_hash(contract, expected_contract_sha256)

    frozen: dict[int, FrozenProbeExtraction] = {}
    extraction_sha = None
    if extraction_payload is not None:
        if expected_extraction_sha256 is None:
            raise ContractIntegrityError(
                "expected_extraction_sha256 is required with an extraction artifact"
            )
        frozen, extraction_sha = load_frozen_extractions(
            extraction_payload,
            expected_sha256=expected_extraction_sha256,
            contract=contract,
            contract_sha256=contract_sha,
        )
    elif expected_extraction_sha256 is not None:
        raise ContractIntegrityError("an extraction hash was supplied without its artifact")

    results: list[ProbeResult] = []
    for index, probe in enumerate(contract["cf_probes"]):
        channel = probe["channel"]
        if channel == "CODE":
            pos_ex, neg_ex, available, reason = {}, {}, True, None
        else:
            row = frozen.get(index)
            available = bool(row and row.available)
            pos_ex = row.pos if row and row.available else None
            neg_ex = row.neg if row and row.available else None
            reason = row.unavailable_reason if row else "no frozen extraction row"
        results.append(
            _evaluate_pair(
                index=index,
                channel=channel,
                text_pos=probe["text_pos"],
                text_neg=probe["text_neg"],
                extracted_pos=pos_ex,
                extracted_neg=neg_ex,
                available=available,
                unavailable_reason=reason,
                score=score,
                ops=ops,
                min_margin=config.min_margin,
            )
        )

    code_rows = [row for row in results if row.channel == "CODE"]
    l_declared = sum(row.channel == "L" for row in results)
    l_available = sum(row.channel == "L" and row.available for row in results)
    code_gate = _build_gate(
        "CODE",
        code_rows,
        separation_fraction_required=config.separation_fraction,
        l_declared=0,
        l_available=0,
        required_l_coverage=None,
    )
    hybrid_gate = _build_gate(
        "HYBRID",
        results,
        separation_fraction_required=config.separation_fraction,
        l_declared=l_declared,
        l_available=l_available,
        required_l_coverage=config.required_l_coverage,
    )
    discrimination = (
        check_discrimination(
            score,
            discrimination_cases,
            contract=contract,
            default_ops=ops,
            config=config,
        )
        if discrimination_cases is not None
        else _not_run_discrimination()
    )
    return ContractCheckResult(
        contract_sha,
        extraction_sha,
        config,
        tuple(results),
        code_gate,
        hybrid_gate,
        discrimination,
    )


def _load_candidate(path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location("metric_seam_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "score", None)):
        raise RuntimeError(f"candidate {path} has no callable score(text, extracted, ops)")
    return module


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=pathlib.Path)
    parser.add_argument("--expected-contract-sha256")
    parser.add_argument("--print-contract-sha256", action="store_true")
    parser.add_argument("--candidate", type=pathlib.Path)
    parser.add_argument("--probe-extractions", type=pathlib.Path)
    parser.add_argument("--expected-extractions-sha256")
    parser.add_argument(
        "--capability",
        action="append",
        choices=sorted(PROBE_CAPABILITIES),
        default=[],
        help="blind-safe computation capability (repeatable; default: base)",
    )
    parser.add_argument("--min-margin", type=float, default=0.01)
    parser.add_argument("--separation-fraction", type=float, default=0.75)
    parser.add_argument("--required-l-coverage", type=float, default=1.0)
    parser.add_argument("--json-out", type=pathlib.Path)
    args = parser.parse_args(argv)

    contract = json.loads(args.contract.read_text())
    if args.print_contract_sha256:
        print(canonical_json_sha256(contract))
        return 0
    if not args.expected_contract_sha256 or not args.candidate:
        parser.error("a check requires --expected-contract-sha256 and --candidate")
    if bool(args.probe_extractions) != bool(args.expected_extractions_sha256):
        parser.error(
            "--probe-extractions and --expected-extractions-sha256 must be supplied together"
        )
    payload = (
        json.loads(args.probe_extractions.read_text()) if args.probe_extractions else None
    )
    module = _load_candidate(args.candidate)
    ops, capabilities = build_probe_ops(args.capability or ["base"])
    result = check_contract(
        contract,
        expected_contract_sha256=args.expected_contract_sha256,
        score=module.score,
        config=CheckConfig(
            separation_fraction=args.separation_fraction,
            min_margin=args.min_margin,
            required_l_coverage=args.required_l_coverage,
        ),
        ops=ops,
        extraction_payload=payload,
        expected_extraction_sha256=args.expected_extractions_sha256,
        discrimination_cases=None,
    )
    output = result.to_dict()
    output["probe_capabilities"] = list(capabilities)
    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n")
    statuses = (result.code_gate.status, result.hybrid_gate.status)
    if "FAIL" in statuses or result.discrimination_gate.status == "FAIL":
        return 1
    if "ABSTAIN" in statuses:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
