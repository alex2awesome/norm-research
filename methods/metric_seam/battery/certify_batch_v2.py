"""Immutable, multiplicity-aware certification for reconstruction-v2 batches.

This is an additive lane.  It does not read a mutable promotion queue and it never
modifies historical certificates.  Its target is explicitly a ``frozen LLM reference``:
agreement with that reference is *reconstruction*, not external ground truth and not a
code-native verifiability certificate.  Contract eligibility and an adversary ACCEPT are
construct-fidelity prerequisites for entering either inferential family.

Manifest schema (all artifact paths are repository-relative and SHA-256 pinned)::

    {
      "schema_version": "metric_seam.certification_batch.v2",
      "batch_id": "math_confirmatory_001",
      "frozen": true,
      "analysis": {
        "alpha": 0.05,
        "minimum_effect": 0.02,
        "g1_minimum_effect": 0.02,
        "coverage_min": 0.90,
        "min_pairs": 20,
        "permutation_samples": 10000,
        "bootstrap_samples": 5000,
        "bootstrap_confidence": 0.95,
        "seed": 20260712
      },
      "entries": [{
        "entry_id": "math__a144::explicit_witness",
        "criterion_id": "math__a144",
        "relation_id": "explicit_witness",
        "heldout_count": 100,
        "candidate_scores": {"path": "...json", "sha256": "..."},
        "h0_scores": {"path": "...json", "sha256": "..."},
        "frozen_llm_reference": {"path": "...json", "sha256": "..."},
        "contract_result": {"path": "...json", "sha256": "..."},
        "adversary_verdict": {"path": "...json", "sha256": "..."},
        "g1_baseline": {"path": "...json", "sha256": "..."}
      }]
    }

Score artifacts are either ``{"scores": {item_id: score}}`` or the score mapping itself.
Contract artifacts must contain ``{"eligible": true, "verdict": "PASS"}``; adversary
artifacts must contain ``{"verdict": "ACCEPT"}``.  The optional G1 comparisons form their
own BH-FDR family and never change the candidate-vs-h0 decision.

The primary p-value is a one-sided paired randomization test of the Spearman-correlation
difference.  It is exact for small samples and uses the valid plus-one Monte Carlo estimate
otherwise.  The paired bootstrap interval is descriptive uncertainty; practical
significance is a separately predeclared minimum-effect gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "metric_seam.certification_batch.v2"
REPORT_SCHEMA_VERSION = "metric_seam.certification_report.v2"
DEFAULT_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SHA256_HEX = frozenset("0123456789abcdef")
_EXACT_PERMUTATION_MAX_N = 16


class CertificationError(RuntimeError):
    """Base class for a fail-closed certification error."""


class IntegrityError(CertificationError):
    """A frozen input is missing, outside the repository, or hash-mismatched."""


class ManifestError(CertificationError):
    """The frozen manifest or one of its artifacts violates the v2 schema."""


class OutputExistsError(CertificationError):
    """A report/snapshot already exists and must not be overwritten."""


@dataclass(frozen=True)
class FrozenArtifact:
    role: str
    relpath: str
    expected_sha256: str
    path: pathlib.Path
    content: bytes

    @property
    def actual_sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


def _no_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ManifestError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def _json_bytes(content: bytes, label: str) -> Any:
    try:
        return json.loads(content.decode("utf-8"), object_pairs_hook=_no_duplicate_object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{label} is not valid UTF-8 JSON: {exc}") from exc


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_repo(repo_root: pathlib.Path, relpath: Any) -> tuple[str, pathlib.Path]:
    if not isinstance(relpath, str) or not relpath.strip():
        raise ManifestError("artifact path must be a non-empty repository-relative string")
    raw = pathlib.PurePosixPath(relpath)
    if raw.is_absolute() or ".." in raw.parts:
        raise IntegrityError(f"artifact path must stay inside the repository: {relpath!r}")
    root = repo_root.resolve()
    path = (root / pathlib.Path(*raw.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise IntegrityError(f"artifact path escapes repository: {relpath!r}") from exc
    return raw.as_posix(), path


def _parse_ref(
    ref: Any, role: str, repo_root: pathlib.Path, cache: dict[str, FrozenArtifact]
) -> FrozenArtifact:
    if not isinstance(ref, Mapping):
        raise ManifestError(f"{role} must be an artifact reference")
    relpath, path = _inside_repo(repo_root, ref.get("path"))
    expected = ref.get("sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(c not in _SHA256_HEX for c in expected.lower())
    ):
        raise ManifestError(f"{role}.sha256 must be a 64-character SHA-256 hex digest")
    expected = expected.lower()
    cached = cache.get(relpath)
    if cached is not None:
        if cached.expected_sha256 != expected:
            raise IntegrityError(
                f"the same artifact has conflicting frozen hashes: {relpath}"
            )
        return FrozenArtifact(role, relpath, expected, cached.path, cached.content)
    if not path.is_file():
        raise IntegrityError(f"frozen artifact does not exist: {relpath}")
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise IntegrityError(
            f"SHA-256 mismatch for {relpath}: expected {expected}, observed {actual}"
        )
    artifact = FrozenArtifact(role, relpath, expected, path, content)
    cache[relpath] = artifact
    return artifact


def _require_number(value: Any, label: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not lo <= number <= hi:
        raise ManifestError(f"{label} must be finite and in [{lo}, {hi}]")
    return number


def _require_int(value: Any, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{label} must be an integer >= {minimum}")
    return value


def _analysis_settings(manifest: Mapping[str, Any], has_g1: bool) -> dict[str, Any]:
    raw = manifest.get("analysis")
    if not isinstance(raw, Mapping):
        raise ManifestError("manifest.analysis must be an object with frozen settings")
    required = {
        "alpha",
        "minimum_effect",
        "coverage_min",
        "min_pairs",
        "permutation_samples",
        "bootstrap_samples",
        "bootstrap_confidence",
        "seed",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ManifestError(f"analysis is missing predeclared settings: {missing}")
    if has_g1 and "g1_minimum_effect" not in raw:
        raise ManifestError("analysis.g1_minimum_effect is required when G1 is present")
    settings = {
        "alpha": _require_number(raw["alpha"], "analysis.alpha", 1e-12, 1 - 1e-12),
        "minimum_effect": _require_number(
            raw["minimum_effect"], "analysis.minimum_effect", 0.0, 2.0
        ),
        "coverage_min": _require_number(
            raw["coverage_min"], "analysis.coverage_min", 0.90, 1.0
        ),
        "min_pairs": _require_int(raw["min_pairs"], "analysis.min_pairs", 3),
        "permutation_samples": _require_int(
            raw["permutation_samples"], "analysis.permutation_samples", 99
        ),
        "bootstrap_samples": _require_int(
            raw["bootstrap_samples"], "analysis.bootstrap_samples", 100
        ),
        "bootstrap_confidence": _require_number(
            raw["bootstrap_confidence"], "analysis.bootstrap_confidence", 0.50, 0.999
        ),
        "seed": _require_int(raw["seed"], "analysis.seed", 0),
    }
    settings["g1_minimum_effect"] = (
        _require_number(raw["g1_minimum_effect"], "analysis.g1_minimum_effect", 0.0, 2.0)
        if has_g1
        else None
    )
    return settings


def _load_score_map(artifact: FrozenArtifact) -> dict[str, float]:
    raw = _json_bytes(artifact.content, artifact.relpath)
    if isinstance(raw, Mapping) and "scores" in raw:
        raw = raw["scores"]
    if not isinstance(raw, Mapping) or not raw:
        raise ManifestError(f"{artifact.role} must contain a non-empty score mapping")
    scores: dict[str, float] = {}
    for item_id, value in raw.items():
        if not isinstance(item_id, str) or not item_id:
            raise ManifestError(f"{artifact.role} has a non-string/empty item identifier")
        scores[item_id] = _require_number(
            value, f"{artifact.role}[{item_id!r}]", 0.0, 1.0
        )
    return scores


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank = (start + end) / 2.0 + 1.0
        for offset in range(start, end + 1):
            result[order[offset]] = rank
        start = end + 1
    return result


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    mean_l = statistics.fmean(left)
    mean_r = statistics.fmean(right)
    dl = [value - mean_l for value in left]
    dr = [value - mean_r for value in right]
    denom = math.sqrt(sum(x * x for x in dl) * sum(y * y for y in dr))
    if denom == 0.0:
        return float("nan")
    return sum(x * y for x, y in zip(dl, dr)) / denom


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    """Tie-aware Spearman correlation used by both observed and resampled statistics."""
    left_values = list(left)
    right_values = list(right)
    return _pearson(_ranks(left_values), _ranks(right_values))


def _correlation_delta(
    candidate: list[float], comparator: list[float], reference: list[float]
) -> tuple[float, float, float]:
    rho_candidate = spearman(candidate, reference)
    rho_comparator = spearman(comparator, reference)
    return rho_candidate, rho_comparator, rho_candidate - rho_comparator


def _derived_seed(base_seed: int, entry_id: str, family: str) -> int:
    payload = f"{base_seed}\0{entry_id}\0{family}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def paired_randomization_test(
    candidate: list[float],
    comparator: list[float],
    reference: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """One-sided paired swap test for a positive Spearman-correlation difference."""
    if not (len(candidate) == len(comparator) == len(reference)):
        raise ValueError("paired vectors must have identical lengths")
    n = len(reference)
    _, _, observed = _correlation_delta(candidate, comparator, reference)
    if not math.isfinite(observed):
        raise ValueError("observed Spearman difference is undefined")
    tolerance = 1e-12

    def permuted_delta(mask: int) -> float:
        left = candidate.copy()
        right = comparator.copy()
        for idx in range(n):
            if mask & (1 << idx):
                left[idx], right[idx] = right[idx], left[idx]
        return _correlation_delta(left, right, reference)[2]

    if n <= _EXACT_PERMUTATION_MAX_N:
        total = 1 << n
        extreme = 0
        undefined = 0
        for mask in range(total):
            statistic = permuted_delta(mask)
            if not math.isfinite(statistic):
                undefined += 1
            elif statistic >= observed - tolerance:
                extreme += 1
        if undefined:
            raise ValueError("paired randomization distribution contains undefined statistics")
        p_value = extreme / total
        return {
            "alternative": "candidate_correlation_greater",
            "method": "exact_paired_swap",
            "p_value": p_value,
            "assignments": total,
            "extreme_assignments": extreme,
            "seed": None,
        }

    rng = random.Random(seed)
    extreme = 0
    undefined = 0
    for _ in range(samples):
        mask = rng.getrandbits(n)
        statistic = permuted_delta(mask)
        if not math.isfinite(statistic):
            undefined += 1
        elif statistic >= observed - tolerance:
            extreme += 1
    if undefined:
        raise ValueError("paired randomization distribution contains undefined statistics")
    return {
        "alternative": "candidate_correlation_greater",
        "method": "monte_carlo_paired_swap_plus_one",
        "p_value": (extreme + 1) / (samples + 1),
        "assignments": samples,
        "extreme_assignments": extreme,
        "seed": seed,
    }


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_ci(
    candidate: list[float],
    comparator: list[float],
    reference: list[float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Percentile CI for the paired Spearman-correlation difference."""
    if not (len(candidate) == len(comparator) == len(reference)):
        raise ValueError("paired vectors must have identical lengths")
    n = len(reference)
    rng = random.Random(seed)
    deltas: list[float] = []
    undefined = 0
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        statistic = _correlation_delta(
            [candidate[i] for i in indices],
            [comparator[i] for i in indices],
            [reference[i] for i in indices],
        )[2]
        if math.isfinite(statistic):
            deltas.append(statistic)
        else:
            undefined += 1
    if len(deltas) < max(50, math.ceil(0.80 * samples)):
        raise ValueError("too many undefined paired bootstrap resamples")
    deltas.sort()
    tail = (1.0 - confidence) / 2.0
    return {
        "method": "paired_item_percentile_bootstrap",
        "confidence": confidence,
        "interval": [_quantile(deltas, tail), _quantile(deltas, 1.0 - tail)],
        "valid_resamples": len(deltas),
        "undefined_resamples": undefined,
        "seed": seed,
    }


def benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return monotone BH adjusted p-values, keyed like ``p_values``."""
    if not p_values:
        return {}
    checked: list[tuple[str, float]] = []
    for key, value in p_values.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid p-value for {key}: {value}")
        checked.append((key, float(value)))
    checked.sort(key=lambda pair: (pair[1], pair[0]))
    m = len(checked)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank_index in range(m - 1, -1, -1):
        key, p_value = checked[rank_index]
        rank = rank_index + 1
        running = min(running, p_value * m / rank)
        adjusted[key] = min(1.0, running)
    return adjusted


def _validate_manifest_header(manifest: Any) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if not isinstance(manifest, Mapping):
        raise ManifestError("batch manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {SCHEMA_VERSION!r}")
    if manifest.get("frozen") is not True:
        raise ManifestError("batch manifest must declare frozen=true")
    batch_id = manifest.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ManifestError("batch_id must be a non-empty string")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("entries must be a non-empty list")
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ManifestError(f"entries[{index}] must be an object")
    return manifest, entries


def _entry_identifier(entry: Mapping[str, Any], index: int) -> str:
    entry_id = entry.get("entry_id")
    if not isinstance(entry_id, str) or not entry_id.strip():
        raise ManifestError(f"entries[{index}].entry_id must be a non-empty string")
    for field in ("criterion_id", "relation_id"):
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            raise ManifestError(f"{entry_id}.{field} must be a non-empty string")
    return entry_id


def _artifact_json(artifact: FrozenArtifact) -> Mapping[str, Any]:
    value = _json_bytes(artifact.content, artifact.relpath)
    if not isinstance(value, Mapping):
        raise ManifestError(f"{artifact.role} must contain a JSON object")
    return value


def _construct_eligibility(
    contract: FrozenArtifact, adversary: FrozenArtifact
) -> tuple[bool, bool, list[str]]:
    contract_value = _artifact_json(contract)
    adversary_value = _artifact_json(adversary)
    contract_pass = (
        contract_value.get("eligible") is True
        and str(contract_value.get("verdict", "")).upper() == "PASS"
    )
    adversary_accept = str(adversary_value.get("verdict", "")).upper() == "ACCEPT"
    reasons: list[str] = []
    if not contract_pass:
        reasons.append("contract_not_eligible_pass")
    if not adversary_accept:
        reasons.append("adversary_not_accept")
    return contract_pass, adversary_accept, reasons


def _paired_analysis(
    entry_id: str,
    family: str,
    candidate_scores: Mapping[str, float],
    comparator_scores: Mapping[str, float],
    reference_scores: Mapping[str, float],
    settings: Mapping[str, Any],
    minimum_effect: float,
) -> dict[str, Any]:
    ids = sorted(candidate_scores)
    candidate = [candidate_scores[item_id] for item_id in ids]
    comparator = [comparator_scores[item_id] for item_id in ids]
    reference = [reference_scores[item_id] for item_id in ids]
    rho_candidate, rho_comparator, delta = _correlation_delta(
        candidate, comparator, reference
    )
    if not all(math.isfinite(x) for x in (rho_candidate, rho_comparator, delta)):
        raise ValueError("observed Spearman correlation is undefined")
    perm_seed = _derived_seed(settings["seed"], entry_id, f"{family}:permutation")
    boot_seed = _derived_seed(settings["seed"], entry_id, f"{family}:bootstrap")
    permutation = paired_randomization_test(
        candidate,
        comparator,
        reference,
        samples=settings["permutation_samples"],
        seed=perm_seed,
    )
    bootstrap = paired_bootstrap_ci(
        candidate,
        comparator,
        reference,
        samples=settings["bootstrap_samples"],
        confidence=settings["bootstrap_confidence"],
        seed=boot_seed,
    )
    return {
        "n_paired": len(ids),
        "rho_candidate": rho_candidate,
        "rho_comparator": rho_comparator,
        "delta_spearman": delta,
        "minimum_effect": minimum_effect,
        "minimum_effect_met": delta >= minimum_effect,
        "bootstrap": bootstrap,
        "permutation": permutation,
        "bh_q_value": None,
        "fdr_reject": None,
        "improvement_supported": None,
    }


def _output_paths(output_path: pathlib.Path, snapshot_path: pathlib.Path | None) -> tuple[pathlib.Path, pathlib.Path]:
    output = output_path.resolve()
    snapshot = (
        snapshot_path.resolve()
        if snapshot_path is not None
        else output.with_name(f"{output.stem}.snapshot.json")
    )
    if output == snapshot:
        raise ManifestError("report and snapshot paths must differ")
    for path in (output, snapshot):
        if path.exists():
            raise OutputExistsError(f"refusing to overwrite immutable output: {path}")
    return output, snapshot


def _write_exclusive_read_only(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
    except FileExistsError as exc:
        raise OutputExistsError(f"refusing to overwrite immutable output: {path}") from exc


def certify_batch(
    manifest_path: pathlib.Path | str,
    output_path: pathlib.Path | str,
    *,
    snapshot_path: pathlib.Path | str | None = None,
    repo_root: pathlib.Path | str | None = None,
) -> tuple[pathlib.Path, pathlib.Path, dict[str, Any]]:
    """Certify one frozen batch and write non-overwriting report and input snapshot."""
    root = pathlib.Path(repo_root or DEFAULT_REPO_ROOT).resolve()
    manifest_file = pathlib.Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise IntegrityError(f"manifest does not exist: {manifest_file}")
    try:
        manifest_relpath = manifest_file.relative_to(root).as_posix()
    except ValueError as exc:
        raise IntegrityError("the frozen batch manifest must be inside the repository") from exc
    output, snapshot = _output_paths(
        pathlib.Path(output_path), pathlib.Path(snapshot_path) if snapshot_path else None
    )

    manifest_bytes = manifest_file.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = _json_bytes(manifest_bytes, manifest_relpath)
    manifest, entries = _validate_manifest_header(manifest)
    has_g1 = any("g1_baseline" in entry for entry in entries)
    settings = _analysis_settings(manifest, has_g1)

    artifact_cache: dict[str, FrozenArtifact] = {}
    artifact_roles: dict[str, set[str]] = {}
    report_entries: list[dict[str, Any]] = []
    primary_p: dict[str, float] = {}
    g1_p: dict[str, float] = {}
    seen_ids: set[str] = set()

    for index, entry in enumerate(entries):
        entry_id = _entry_identifier(entry, index)
        if entry_id in seen_ids:
            raise ManifestError(f"duplicate entry_id: {entry_id}")
        seen_ids.add(entry_id)
        artifacts: dict[str, FrozenArtifact] = {}
        for role in (
            "candidate_scores",
            "h0_scores",
            "frozen_llm_reference",
            "contract_result",
            "adversary_verdict",
        ):
            if role not in entry:
                raise ManifestError(f"{entry_id} is missing artifact reference {role}")
            artifacts[role] = _parse_ref(entry[role], role, root, artifact_cache)
            artifact_roles.setdefault(artifacts[role].relpath, set()).add(
                f"{entry_id}:{role}"
            )
        if "g1_baseline" in entry:
            artifacts["g1_baseline"] = _parse_ref(
                entry["g1_baseline"], "g1_baseline", root, artifact_cache
            )
            artifact_roles.setdefault(artifacts["g1_baseline"].relpath, set()).add(
                f"{entry_id}:g1_baseline"
            )

        candidate_scores = _load_score_map(artifacts["candidate_scores"])
        h0_scores = _load_score_map(artifacts["h0_scores"])
        reference_scores = _load_score_map(artifacts["frozen_llm_reference"])
        if set(candidate_scores) != set(h0_scores):
            raise ManifestError(
                f"{entry_id}: candidate and h0 must have identical item IDs"
            )
        if not set(candidate_scores).issubset(reference_scores):
            extras = sorted(set(candidate_scores) - set(reference_scores))[:5]
            raise ManifestError(
                f"{entry_id}: candidate/h0 IDs absent from frozen LLM reference: {extras}"
            )
        g1_scores: dict[str, float] | None = None
        if "g1_baseline" in artifacts:
            g1_scores = _load_score_map(artifacts["g1_baseline"])
            if set(g1_scores) != set(candidate_scores):
                raise ManifestError(
                    f"{entry_id}: G1 and candidate/h0 must have identical item IDs"
                )

        contract_pass, adversary_accept, reasons = _construct_eligibility(
            artifacts["contract_result"], artifacts["adversary_verdict"]
        )
        paired_coverage = len(candidate_scores) / len(reference_scores)
        heldout_count_raw = entry.get("heldout_count")
        heldout_count: int | None = None
        if heldout_count_raw is not None:
            heldout_count = _require_int(
                heldout_count_raw, f"{entry_id}.heldout_count", 1
            )
            if heldout_count < len(reference_scores):
                raise ManifestError(
                    f"{entry_id}.heldout_count is smaller than reference support"
                )
        reference_availability = (
            len(reference_scores) / heldout_count if heldout_count is not None else None
        )
        candidate_corpus_coverage = (
            len(candidate_scores) / heldout_count if heldout_count is not None else None
        )
        if paired_coverage < settings["coverage_min"]:
            reasons.append("coverage_below_predeclared_minimum")
        if len(candidate_scores) < settings["min_pairs"]:
            reasons.append("paired_support_below_predeclared_minimum")

        result: dict[str, Any] = {
            "entry_id": entry_id,
            "criterion_id": entry["criterion_id"],
            "relation_id": entry["relation_id"],
            "reference_target": "frozen_llm_reference",
            "eligibility": {
                "eligible": not reasons,
                "reasons": reasons,
                "contract_eligible_pass": contract_pass,
                "adversary_accept": adversary_accept,
                "n_reference": len(reference_scores),
                "n_paired": len(candidate_scores),
                "paired_coverage": paired_coverage,
                "paired_coverage_denominator": "frozen_reference_available_items",
                "heldout_count": heldout_count,
                "reference_availability_over_heldout": reference_availability,
                "candidate_coverage_over_heldout": candidate_corpus_coverage,
                "coverage_minimum": settings["coverage_min"],
                "candidate_h0_identical_ids": True,
            },
            "reference_reconstruction_vs_h0": None,
            "g1_reference_reconstruction": {
                "status": "unavailable",
                "reason": "no_g1_baseline_declared",
            },
        }
        if not reasons:
            try:
                primary = _paired_analysis(
                    entry_id,
                    "h0",
                    candidate_scores,
                    h0_scores,
                    reference_scores,
                    settings,
                    settings["minimum_effect"],
                )
            except ValueError as exc:
                result["eligibility"]["eligible"] = False
                result["eligibility"]["reasons"].append(
                    f"invalid_inferential_statistic:{exc}"
                )
            else:
                result["reference_reconstruction_vs_h0"] = primary
                primary_p[entry_id] = primary["permutation"]["p_value"]
                if g1_scores is not None:
                    try:
                        g1 = _paired_analysis(
                            entry_id,
                            "g1",
                            candidate_scores,
                            g1_scores,
                            reference_scores,
                            settings,
                            settings["g1_minimum_effect"],
                        )
                    except ValueError as exc:
                        result["g1_reference_reconstruction"] = {
                            "status": "invalid",
                            "reason": str(exc),
                        }
                    else:
                        result["g1_reference_reconstruction"] = {
                            "status": "evaluated_separate_fdr_family",
                            **g1,
                        }
                        g1_p[entry_id] = g1["permutation"]["p_value"]
        elif g1_scores is not None:
            result["g1_reference_reconstruction"] = {
                "status": "ineligible",
                "reason": "primary construct/support eligibility failed",
            }
        report_entries.append(result)

    primary_q = benjamini_hochberg(primary_p)
    g1_q = benjamini_hochberg(g1_p)
    for result in report_entries:
        entry_id = result["entry_id"]
        primary = result["reference_reconstruction_vs_h0"]
        if primary is not None and entry_id in primary_q:
            primary["bh_q_value"] = primary_q[entry_id]
            primary["fdr_reject"] = primary_q[entry_id] <= settings["alpha"]
            primary["improvement_supported"] = bool(
                primary["fdr_reject"] and primary["minimum_effect_met"]
            )
        g1 = result["g1_reference_reconstruction"]
        if isinstance(g1, dict) and entry_id in g1_q:
            g1["bh_q_value"] = g1_q[entry_id]
            g1["fdr_reject"] = g1_q[entry_id] <= settings["alpha"]
            g1["improvement_supported"] = bool(
                g1["fdr_reject"] and g1["minimum_effect_met"]
            )

    input_snapshot = [
        {
            "path": relpath,
            "sha256": artifact.actual_sha256,
            "bytes": len(artifact.content),
            "roles": sorted(artifact_roles.get(relpath, set())),
        }
        for relpath, artifact in sorted(artifact_cache.items())
    ]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "batch_id": manifest["batch_id"],
        "reference_target": "frozen_llm_reference",
        "terminology": {
            "reported_quantity": "isomorphic_reconstruction_agreement",
            "articulability": "prompt_based",
            "verifiability": "code_certificate_based_and_not_inferred_here",
        },
        "manifest": {"path": manifest_relpath, "sha256": manifest_sha256},
        "analysis": settings,
        "multiplicity": {
            "method": "Benjamini-Hochberg FDR",
            "alpha": settings["alpha"],
            "primary_family": {
                "contrast": "candidate_vs_h0",
                "n_valid_p_values": len(primary_p),
            },
            "g1_family": {
                "contrast": "candidate_vs_optional_g1",
                "n_valid_p_values": len(g1_p),
                "separate_from_primary": True,
            },
        },
        "input_snapshot": input_snapshot,
        "entries": report_entries,
    }

    # Refuse time-of-check/time-of-use mutations before emitting immutable results.
    if _sha256_file(manifest_file) != manifest_sha256:
        raise IntegrityError("frozen batch manifest changed during certification")
    for relpath, artifact in artifact_cache.items():
        if _sha256_file(artifact.path) != artifact.expected_sha256:
            raise IntegrityError(f"frozen artifact changed during certification: {relpath}")

    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    snapshot_payload = {
        "schema_version": "metric_seam.certification_input_snapshot.v2",
        "batch_id": manifest["batch_id"],
        "manifest": {
            "path": manifest_relpath,
            "sha256": manifest_sha256,
            "content": manifest,
        },
        "artifacts": input_snapshot,
        "report": {
            "path": str(output),
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
        },
    }
    snapshot_bytes = (
        json.dumps(snapshot_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_exclusive_read_only(output, report_bytes)
    try:
        _write_exclusive_read_only(snapshot, snapshot_bytes)
    except Exception:
        # Roll back only the report created by this invocation; never touch prior output.
        output.chmod(0o644)
        output.unlink()
        raise
    return output, snapshot, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--snapshot", type=pathlib.Path)
    parser.add_argument("--repo-root", type=pathlib.Path, default=DEFAULT_REPO_ROOT)
    args = parser.parse_args(argv)
    output, snapshot, report = certify_batch(
        args.manifest,
        args.output,
        snapshot_path=args.snapshot,
        repo_root=args.repo_root,
    )
    n_supported = sum(
        bool((entry.get("reference_reconstruction_vs_h0") or {}).get("improvement_supported"))
        for entry in report["entries"]
    )
    print(
        f"certified frozen-LLM reconstruction batch {report['batch_id']}: "
        f"{n_supported}/{len(report['entries'])} primary improvements supported"
    )
    print(f"report: {output}")
    print(f"input snapshot: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
