#!/usr/bin/env python3
"""Bridge trusted Sonnet teachers from calibration IDs to production IDs.

The first Sonnet export was deliberately built against the frozen legacy-
signal calibration manifest.  Production reconstruction assigns different
UIDs because it restores the source segment/record/signal identity.  This
module performs a fail-closed bridge and preserves the old UID as provenance.

Resolution order is:

1. corpus alias + exact normalized norm + exact nonempty source identity;
2. corpus alias + a *unique* exact normalized norm in the whole production
   corpus, recording whether its physical row also agrees.

No fuzzy text matching is allowed.  Missing and ambiguous joins, stale bank
IDs, bank-name drift, and post-bridge label conflicts are ledgered rather than
silently accepted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import SCHEMA_VERSION
from .common import normalize_space, read_jsonl, stable_uid, write_jsonl
from .config import CORPUS_ALIASES, DEFAULT_OUTPUT_ROOT


DEFAULT_CALIBRATION_ROOT = Path("/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_calibration")
SOURCE_PRIORITY = {
    "sonnet_full": 1,
    "sonnet_pilot": 2,
    "sonnet_rescue": 3,
    "sonnet_audit": 4,
}


class BridgeIntegrityError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BridgeIntegrityError(f"expected object in {path}")
    return value


def _alias(corpus: str) -> str:
    return CORPUS_ALIASES.get(corpus, corpus)


def load_legacy_norms(
    calibration_root: Path, corpora: Sequence[str]
) -> dict[str, dict[str, Any]]:
    by_uid: dict[str, dict[str, Any]] = {}
    for corpus in sorted(set(corpora)):
        path = calibration_root / "norms" / f"{corpus}.jsonl"
        for physical_row, row in enumerate(read_jsonl(path)):
            if row.get("corpus") != corpus or int(row.get("row", -1)) != physical_row:
                raise BridgeIntegrityError(f"legacy manifest routing/row drift at {path}:{physical_row + 1}")
            norm = normalize_space(row.get("norm"))
            expected = stable_uid(corpus, physical_row, norm)
            if row.get("norm_uid") != expected:
                raise BridgeIntegrityError(f"legacy calibration UID drift at {path}:{physical_row + 1}")
            if expected in by_uid:
                raise BridgeIntegrityError(f"duplicate calibration UID: {expected}")
            by_uid[expected] = row
    return by_uid


def load_production_norms(
    production_root: Path, corpora: Sequence[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, list[dict[str, Any]]]]]:
    rows_by_corpus = {}
    text_index = {}
    for legacy_corpus in sorted(set(corpora)):
        corpus = _alias(legacy_corpus)
        if corpus in rows_by_corpus:
            continue
        path = production_root / "norms" / f"{corpus}.jsonl"
        rows = list(read_jsonl(path))
        by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for physical_row, row in enumerate(rows):
            if row.get("corpus") != corpus or int(row.get("row", -1)) != physical_row:
                raise BridgeIntegrityError(f"production routing/row drift at {path}:{physical_row + 1}")
            by_text[normalize_space(row.get("norm"))].append(row)
        rows_by_corpus[corpus] = rows
        text_index[corpus] = dict(by_text)
    return rows_by_corpus, text_index


def load_banks(production_root: Path, tasks: Sequence[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for task in sorted(set(tasks)):
        path = production_root / "banks" / f"{task}.json"
        bank = _load_json(path)
        metrics = {str(row["metric_id"]): row for row in bank.get("metrics") or []}
        result[task] = {
            "source_sha256": str(bank.get("source_sha256") or ""),
            "metrics": metrics,
        }
    return result


def resolve_norm(
    legacy: Mapping[str, Any],
    production_rows: Mapping[str, list[dict[str, Any]]],
    text_index: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> tuple[dict[str, Any] | None, str, Any]:
    legacy_corpus = str(legacy["corpus"])
    corpus = _alias(legacy_corpus)
    norm = normalize_space(legacy.get("norm"))
    source_id = normalize_space(legacy.get("source_id"))
    # The capped bge_pertask signal files used ``i=<physical row>``.  The
    # calibration freezer correctly preserved it, but it is an array index,
    # not a document/source identity.  Treat only non-positional values as
    # source evidence.  (Row zero became an empty string through the legacy
    # normalizer; later rows are the decimal row number.)
    if source_id == str(legacy.get("row")):
        source_id = ""
    rows = production_rows[corpus]

    if source_id:
        candidates = [
            row
            for row in text_index[corpus].get(norm, ())
            if normalize_space(row.get("source_id")) == source_id
        ]
        if len(candidates) == 1:
            return candidates[0], "alias_source_id_and_norm_exact", None
        return None, (
            "production_norm_missing" if not candidates else "ambiguous_production_norm"
        ), {"candidate_count": len(candidates), "source_id": source_id}

    candidates = text_index[corpus].get(norm, ())
    if len(candidates) == 1:
        row_index = int(legacy["row"])
        if (
            0 <= row_index < len(rows)
            and rows[row_index]["norm_uid"] == candidates[0]["norm_uid"]
        ):
            return candidates[0], "alias_row_and_unique_norm_exact", None
        return candidates[0], "alias_unique_norm_exact", None
    return None, (
        "production_norm_missing" if not candidates else "ambiguous_production_norm"
    ), {"candidate_count": len(candidates)}


def _rejection(
    reason: str,
    teacher: Mapping[str, Any],
    legacy: Mapping[str, Any] | None,
    *,
    details: Any = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "reason": reason,
        "label_source": teacher.get("label_source"),
        "legacy_calibration_norm_uid": teacher.get("norm_uid"),
        "legacy_calibration_corpus": teacher.get("corpus"),
        "legacy_calibration_row": (
            legacy.get("legacy_calibration_row", legacy.get("row"))
            if legacy
            else teacher.get("legacy_calibration_row", teacher.get("row"))
        ),
        "task": teacher.get("task"),
        "decision": teacher.get("decision"),
        "metric_id": teacher.get("metric_id"),
        "details": details,
    }


def _signature(row: Mapping[str, Any]) -> tuple[str, str | None]:
    return str(row.get("decision")), row.get("metric_id")


def bridge(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.teachers)
    calibration_root = Path(args.calibration_root)
    production_root = Path(args.production_root)
    output_root = Path(args.output_root)
    teachers = list(read_jsonl(source_path))
    if not teachers:
        raise BridgeIntegrityError(f"no teachers in {source_path}")

    corpora = [str(row["corpus"]) for row in teachers]
    tasks = [str(row["task"]) for row in teachers]
    legacy_by_uid = load_legacy_norms(calibration_root, corpora)
    production_rows, text_index = load_production_norms(production_root, corpora)
    banks = load_banks(production_root, tasks)

    bridged = []
    rejected = []
    for teacher in teachers:
        old_uid = str(teacher.get("norm_uid") or "")
        legacy = legacy_by_uid.get(old_uid)
        if legacy is None:
            rejected.append(_rejection("calibration_norm_uid_missing", teacher, None))
            continue
        if teacher.get("task") != legacy.get("task") or teacher.get("corpus") != legacy.get("corpus"):
            rejected.append(_rejection("teacher_calibration_routing_mismatch", teacher, legacy))
            continue

        task = str(teacher["task"])
        bank = banks[task]
        if teacher.get("current_bank_source_sha256") != bank["source_sha256"]:
            rejected.append(
                _rejection(
                    "current_bank_hash_mismatch",
                    teacher,
                    legacy,
                    details={
                        "teacher": teacher.get("current_bank_source_sha256"),
                        "production": bank["source_sha256"],
                    },
                )
            )
            continue
        if teacher.get("decision") == "MATCH":
            metric = bank["metrics"].get(str(teacher.get("metric_id")))
            if metric is None:
                rejected.append(_rejection("metric_id_missing_from_current_bank", teacher, legacy))
                continue
            expected_name = normalize_space((teacher.get("notes") or {}).get("current_metric_name"))
            if expected_name and normalize_space(metric.get("name")) != expected_name:
                rejected.append(
                    _rejection(
                        "metric_id_name_drift",
                        teacher,
                        legacy,
                        details={"teacher": expected_name, "production": metric.get("name")},
                    )
                )
                continue
        elif teacher.get("metric_id") is not None:
            rejected.append(_rejection("non_match_has_metric_id", teacher, legacy))
            continue

        production, method, details = resolve_norm(legacy, production_rows, text_index)
        if production is None:
            rejected.append(_rejection(method, teacher, legacy, details=details))
            continue
        if production.get("task") != task:
            rejected.append(
                _rejection(
                    "production_task_mismatch",
                    teacher,
                    legacy,
                    details={"production_task": production.get("task")},
                )
            )
            continue

        row = dict(teacher)
        row.update(
            {
                "norm_uid": production["norm_uid"],
                "corpus": production["corpus"],
                "row": production["row"],
                "source_id": production.get("source_id"),
                "uid_universe": "production_canonical_v3",
                "production_uid_bridge_method": method,
                "legacy_calibration_norm_uid": old_uid,
                "legacy_calibration_corpus": legacy["corpus"],
                "legacy_calibration_row": legacy["row"],
            }
        )
        bridged.append(row)

    # Joining by text can expose duplicate legacy rows.  Agreements count once;
    # conflicts are removed wholesale so no arbitrary label wins.
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bridged:
        by_uid[str(row["norm_uid"])].append(row)
    selected = []
    for uid, group in by_uid.items():
        signatures = {_signature(row) for row in group}
        if len(signatures) > 1:
            for row in group:
                rejected.append(
                    _rejection(
                        "conflicting_labels_after_production_bridge",
                        row,
                        row,
                        details={"production_norm_uid": uid, "signatures": sorted(map(str, signatures))},
                    )
                )
            continue
        group.sort(
            key=lambda row: (
                SOURCE_PRIORITY.get(str(row.get("label_source")), 0),
                {"high": 2, "low": 1, None: 0}.get(row.get("confidence"), 0),
                str(row.get("legacy_calibration_norm_uid")),
            ),
            reverse=True,
        )
        winner = group[0]
        if len(group) > 1:
            winner["agreeing_production_bridge_duplicates"] = [
                row["legacy_calibration_norm_uid"] for row in group[1:]
            ]
            for row in group[1:]:
                rejected.append(
                    _rejection(
                        "agreeing_duplicate_after_production_bridge",
                        row,
                        row,
                        details={"production_norm_uid": uid, "retained": winner["legacy_calibration_norm_uid"]},
                    )
                )
        selected.append(winner)

    selected.sort(key=lambda row: (str(row["corpus"]), int(row["row"])))
    rejected.sort(
        key=lambda row: (
            str(row.get("legacy_calibration_corpus") or ""),
            int(row.get("legacy_calibration_row") or -1),
            str(row.get("reason") or ""),
        )
    )
    teacher_path = output_root / "teachers/sonnet.production.jsonl"
    rejection_path = output_root / "teachers/sonnet.production_bridge_rejections.jsonl"
    write_jsonl(teacher_path, selected)
    write_jsonl(rejection_path, rejected)
    return {
        "source_teachers": len(teachers),
        "production_teachers": len(selected),
        "rejections": len(rejected),
        "by_decision": dict(sorted(Counter(row["decision"] for row in selected).items())),
        "by_corpus": dict(sorted(Counter(row["corpus"] for row in selected).items())),
        "by_bridge_method": dict(
            sorted(Counter(row["production_uid_bridge_method"] for row in selected).items())
        ),
        "rejections_by_reason": dict(sorted(Counter(row["reason"] for row in rejected).items())),
        "teacher_path": str(teacher_path),
        "rejection_path": str(rejection_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teachers", default=str(DEFAULT_OUTPUT_ROOT / "teachers/sonnet.jsonl"))
    parser.add_argument("--calibration-root", default=str(DEFAULT_CALIBRATION_ROOT))
    parser.add_argument("--production-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> None:
    print(json.dumps(bridge(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
