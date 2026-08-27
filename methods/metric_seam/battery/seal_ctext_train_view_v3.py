#!/usr/bin/env python3
"""Create an immutable, TRAIN-only compiler view from a label-bearing item file.

This is an additive specialization of :mod:`seal_ctext_items_v2`.  The trusted
preparation process may deserialize a historical item file, but the artifact it
emits contains only an articulated relation contract and opaque ``item_key`` / 
``ctext`` TRAIN rows.  It never materializes held-out rows, source identifiers,
or arbitrary source fields in the compiler view or its manifest.

The module is a projection boundary, not an adversarial security boundary.  A
compiler must receive only the emitted bundle, never the source item file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
from typing import Any, Iterable, Mapping

try:
    from .seal_ctext_items_v2 import canonical_bytes, sha256
except ImportError:  # pragma: no cover - direct-script compatibility
    from seal_ctext_items_v2 import canonical_bytes, sha256  # type: ignore[no-redef]


SCHEMA_BUNDLE = "metric-seam.sanitized-ctext-train-compiler-view.v2"
SCHEMA_MANIFEST = "metric-seam.sanitized-ctext-train-preparation-manifest.v2"
SANITIZER_SCHEMA = "metric-seam.credential-value-sanitizer.v1"
REDACTION_TOKEN = "[REDACTED_CREDENTIAL]"

_FORBIDDEN_CONTRACT_KEYS = {
    "answer",
    "correlation",
    "gold",
    "ground_truth",
    "judge",
    "judgement",
    "label",
    "reference_value",
    "residual",
    "result",
    "rho",
    "score",
    "target_value",
}


@dataclass(frozen=True)
class CredentialPattern:
    """A credential-like surface and the span that must be replaced."""

    category: str
    regex: re.Pattern[str]
    value_group: str | None = None


# Order is frozen and therefore part of the representation.  Specific token
# forms run before the generic assignment form.  No matching value is ever
# surfaced in a receipt or exception.
CREDENTIAL_PATTERNS = (
    CredentialPattern(
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
            r"[\s\S]*?"
            r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
        ),
    ),
    CredentialPattern(
        "aws_access_key",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    ),
    CredentialPattern(
        "github_token",
        re.compile(r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9_]{36,255}"),
    ),
    CredentialPattern(
        "google_api_key",
        re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])"),
    ),
    CredentialPattern(
        "openai_style_key",
        re.compile(
            r"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"
        ),
    ),
    CredentialPattern(
        "slack_token",
        re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}(?![A-Za-z0-9-])"),
    ),
    CredentialPattern(
        "jwt_compact",
        re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,}\."
            r"[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}(?![A-Za-z0-9_-])"
        ),
    ),
    CredentialPattern(
        "credential_assignment_long_literal",
        re.compile(
            r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
            r"password|private[_-]?key|secret[_-]?key)\b\s*[:=]\s*"
            r"(?P<quote>[\"'])(?P<value>[A-Za-z0-9_./+=:@-]{16,})(?P=quote)",
            re.IGNORECASE,
        ),
        value_group="value",
    ),
)


def credential_pattern_counts(text: str) -> dict[str, int]:
    """Return category counts only; matching values are intentionally inaccessible."""
    return {
        pattern.category: sum(1 for _ in pattern.regex.finditer(text))
        for pattern in CREDENTIAL_PATTERNS
    }


def sanitize_ctext(text: str) -> tuple[str, dict[str, int]]:
    """Deterministically replace credential VALUE spans with a fixed token."""
    sanitized = text
    counts: Counter[str] = Counter()
    for pattern in CREDENTIAL_PATTERNS:
        def replace(match: re.Match[str]) -> str:
            counts[pattern.category] += 1
            if pattern.value_group is None:
                return REDACTION_TOKEN
            start, end = match.span(pattern.value_group)
            relative_start = start - match.start()
            relative_end = end - match.start()
            whole = match.group(0)
            return whole[:relative_start] + REDACTION_TOKEN + whole[relative_end:]

        sanitized = pattern.regex.sub(replace, sanitized)
    return sanitized, {
        pattern.category: counts.get(pattern.category, 0)
        for pattern in CREDENTIAL_PATTERNS
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in sorted(set(names)):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _audit_contract(value: Any, path: str = "$") -> None:
    """Reject obvious outcome channels from the projected relation contract."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in _FORBIDDEN_CONTRACT_KEYS:
                raise ValueError(f"projected contract has forbidden key {path}.{key}")
            _audit_contract(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _audit_contract(child, f"{path}[{index}]")


def _project_rows(
    raw: Any,
    *,
    train_count: int,
    split_seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("source must be a non-empty JSON list")
    if train_count <= 0 or train_count >= len(raw):
        raise ValueError("train_count must be positive and smaller than the corpus")

    by_id: dict[str, str] = {}
    redactions_by_id: dict[str, dict[str, int]] = {}
    source_keys: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"source row {index} is not an object")
        source_keys.update(str(key) for key in row)
        datapoint_id, ctext = row.get("datapoint_id"), row.get("ctext")
        if not isinstance(datapoint_id, str) or not datapoint_id:
            raise ValueError(f"source row {index} has invalid datapoint_id")
        if datapoint_id in by_id:
            raise ValueError(f"duplicate datapoint_id {datapoint_id!r}")
        if not isinstance(ctext, str):
            raise ValueError(f"source row {index} has no string ctext")
        # Sanitize every row before selecting a split.  Heldout text is transformed
        # in memory only and is never materialized in the compiler artifact.
        sanitized, counts = sanitize_ctext(ctext)
        by_id[datapoint_id] = sanitized
        redactions_by_id[datapoint_id] = counts

    shuffled = sorted(by_id)
    random.Random(split_seed).shuffle(shuffled)
    selected = set(shuffled[:train_count])

    # Sorting after selection matches the existing blind-reconstruction lane.
    # Original identifiers decide order inside this trusted function but are not
    # emitted.  The compiler receives only a canonical opaque enumeration.
    train_items = [
        {"item_key": f"train_{index:04d}", "ctext": by_id[datapoint_id]}
        for index, datapoint_id in enumerate(sorted(selected), 1)
    ]
    projection = {
        "source_keys_observed": sorted(source_keys),
        "source_values_copied_for": ["sanitized_ctext"],
        "source_identifiers_used_only_for_partition_and_order": True,
        "source_identifiers_emitted": False,
        "all_other_source_values_discarded": True,
        "outcome_values_recorded_in_manifest": False,
    }

    categories = [pattern.category for pattern in CREDENTIAL_PATTERNS]

    def summarize(ids: Iterable[str]) -> dict[str, Any]:
        ids = list(ids)
        totals = {
            category: sum(redactions_by_id[datapoint_id][category] for datapoint_id in ids)
            for category in categories
        }
        return {
            "row_count": len(ids),
            "changed_row_count": sum(
                any(redactions_by_id[datapoint_id].values()) for datapoint_id in ids
            ),
            "category_counts": totals,
            "total_matches": sum(totals.values()),
        }

    all_ids = set(by_id)
    redaction_summary = {
        "schema": SANITIZER_SCHEMA,
        "replacement_token": REDACTION_TOKEN,
        "operation_order": categories,
        "applied_before_partition": True,
        "full": summarize(all_ids),
        "train": summarize(selected),
        "heldout": summarize(all_ids - selected),
        "identifiers_or_values_recorded": False,
    }
    return train_items, projection, redaction_summary


def prepare_train_view(
    *,
    source: Path,
    contract_path: Path,
    out_dir: Path,
    task: str,
    criterion_id: str,
    train_count: int = 150,
    split_seed: int = 7,
    dependency_files: Mapping[str, Path] | None = None,
    dependency_packages: Iterable[str] = (),
) -> tuple[Path, Path]:
    """Freeze a compiler-visible bundle and a non-sensitive preparation manifest."""
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable preparation {out_dir}")
    if not task or not criterion_id:
        raise ValueError("task and criterion_id are required")

    raw = json.loads(source.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _audit_contract(contract)
    train_items, projection, redaction_summary = _project_rows(
        raw, train_count=train_count, split_seed=split_seed
    )
    bundle = {
        "schema": SCHEMA_BUNDLE,
        "objective": {
            "name": "unsupervised reconstruction of an articulated metric",
            "articulability_axis": "prompt/LLM implementation",
            "verifiability_axis": "executable/code implementation",
            "isomorphism_axis": "same construct and sanitized_ctext representation",
            "external_supervised_anchor": False,
        },
        "task": task,
        "criterion_id": criterion_id,
        "construct": contract,
        "interface": {
            "representation": "sanitized_ctext only",
            "sanitizer_schema": SANITIZER_SCHEMA,
            "item_keys": "opaque TRAIN-only aliases",
            "compiler_item_allowed_keys": ["ctext", "item_key"],
            "reference_values_available": False,
            "residuals_available": False,
            "heldout_items_available": False,
            "heldout_identifiers_available": False,
        },
        "train_items": train_items,
    }
    for index, item in enumerate(bundle["train_items"]):
        if set(item) != {"ctext", "item_key"}:
            raise AssertionError(f"compiler row {index} exceeds the item allowlist")
    expected_keys = [f"train_{index:04d}" for index in range(1, train_count + 1)]
    if [item["item_key"] for item in bundle["train_items"]] != expected_keys:
        raise AssertionError("compiler aliases are not canonical")

    bundle_bytes = canonical_bytes(bundle)
    dependency_records = {
        name: {"path": str(path.resolve()), "sha256": sha256(path)}
        for name, path in sorted((dependency_files or {}).items())
    }
    manifest = {
        "schema": SCHEMA_MANIFEST,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": _sha256_bytes(bundle_bytes)[:20],
        "task": task,
        "criterion_id": criterion_id,
        "inputs": {
            "source": {"path": str(source.resolve()), "sha256": sha256(source)},
            "projected_contract": {
                "path": str(contract_path.resolve()),
                "sha256": sha256(contract_path),
            },
        },
        "partition": {
            "algorithm": "sorted datapoint_id; random.Random(seed).shuffle; first train_count",
            "seed": split_seed,
            "corpus_count": len(raw),
            "train_count": train_count,
            "heldout_count": len(raw) - train_count,
            "train_source_identifiers_emitted": False,
            "heldout_identifiers_materialized": False,
            "heldout_rows_materialized": False,
        },
        "projection": projection,
        "credential_redaction": redaction_summary,
        "policy": {
            "objective": "unsupervised reconstruction",
            "trusted_preparer_deserialized_source": True,
            "compiler_receives_source_file": False,
            "compiler_receives_train_sanitized_ctext_only": True,
            "compiler_receives_heldout": False,
            "compiler_receives_reference_values": False,
            "model_calls": False,
            "gpu_used": False,
            "preparation_is_security_boundary": False,
            "artifacts_created_once_read_only": True,
            "future_prompt_code_and_heldout_must_apply_same_sanitizer": True,
            "historical_raw_reference_reuse_for_changed_rows_allowed": False,
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": _package_versions(dependency_packages),
        },
        "implementation": dependency_records,
        "artifacts": {
            "compiler_bundle.json": {
                "sha256": _sha256_bytes(bundle_bytes),
                "n_train_items": train_count,
                "allowed_item_keys": ["ctext", "item_key"],
                "readonly": True,
            }
        },
    }

    out_dir.mkdir(parents=True)
    try:
        bundle_path = out_dir / "compiler_bundle.json"
        manifest_path = out_dir / "prepare_manifest.json"
        _write_exclusive(bundle_path, bundle_bytes)
        _write_exclusive(manifest_path, canonical_bytes(manifest))
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return bundle_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--criterion-id", required=True)
    parser.add_argument("--train-count", type=int, default=150)
    parser.add_argument("--split-seed", type=int, default=7)
    args = parser.parse_args()
    bundle_path, manifest_path = prepare_train_view(
        source=args.source,
        contract_path=args.contract,
        out_dir=args.out_dir,
        task=args.task,
        criterion_id=args.criterion_id,
        train_count=args.train_count,
        split_seed=args.split_seed,
        dependency_files={"sealer_v3": Path(__file__)},
    )
    # Never print ctext or source identifiers.
    print(json.dumps({
        "bundle": str(bundle_path),
        "bundle_sha256": sha256(bundle_path),
        "manifest": str(manifest_path),
        "train_count": args.train_count,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
