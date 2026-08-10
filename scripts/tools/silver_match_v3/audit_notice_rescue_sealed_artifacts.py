#!/usr/bin/env python3
"""Audit the complete sealed N&C rescue inference universe without rerunning it.

This is deliberately task-local.  The historical N&C rescue GPU artifacts are
complete, but one no-op ``--resume`` invocation overwrote trial-000's lifecycle
counts in its metadata.  Completeness therefore comes from byte hashes plus
row-level input/output identity and candidate binding, never from ``new_count``
alone.  No model implementation is imported or hash-gated here.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .aggregate_abstention_rescue import aggregate_rescue
from .combine_two_order_abstention_verifications import combine as combine_abstentions
from .combine_two_order_verifications import combine as combine_verifications
from .common import read_jsonl, sha256_file


TASK = "notice-and-comment"
EXPECTED_CORPORA = {"notice_and_comment", "nc_public_comments"}
EXPECTED_BANK_COUNT = 88
EXPECTED_TRIALS = tuple(f"trial-{index:03d}" for index in range(4))
EXPECTED_ADJUDICATOR_PROMPT = (
    "c839a28c4c452de8faa937e064d2ad4824d06dfe65730ce29263815d355cd111"
)
EXPECTED_VERIFIER_PROMPT = (
    "7f8fb51b43bf367ed96dd3cf5e1b871e87a4dced87c2198035992aeb751c696d"
)
EXPECTED_ABSTENTION_PROMPT = (
    "bb7ce2f492d6e933242b9650eae98eb588fd66b7403ccbb6eac19c3181b32b5d"
)
EXPECTED_ADJUDICATOR_COMPONENTS = {
    "scripts/tools/silver_match_v3/prompts/gepa_notice_k50_shepherded_v1.txt": (
        "65beee77376ff707ef594bae70088e780a3d11e0e21583becaffa48312ed7913"
    ),
    "scripts/tools/silver_match_v3/prompts/gepa_round2_candidate.txt": (
        "068ae4f55fe74375bd062108d978b124066df006834b8b78b6d1dafaa4d74056"
    ),
}
EXPECTED_VERIFIER_COMPONENTS = {
    "scripts/tools/silver_match_v3/prompts/verify_match_v1.txt": (
        "4a2b61d758dfb57a5659d2e0eaef256ed1296f5cc412dc68e805e77a04596dbc"
    ),
    "scripts/tools/silver_match_v3/prompts/verify_notice_shepherded_v2.txt": (
        "f14f58533f8c7f1e27cca4f2044b74fce883a25cfdab9221e03489058a7a6a96"
    ),
}
CPU_IMPLEMENTATIONS = (
    "merge_rescue_decisions.py",
    "filter_labels.py",
    "audit_final_outputs.py",
    "prepare_final_decision_audit.py",
    "prepare_false_abstention_audit.py",
)
AUDIT_RECOMPUTE_IMPLEMENTATIONS = (
    "aggregate_abstention_rescue.py",
    "combine_two_order_verifications.py",
    "combine_two_order_abstention_verifications.py",
)
VERIFIER_DECISIONS = {
    "CONFIRM_MATCH",
    "AMBIGUOUS_MATCH",
    "BETTER_CANDIDATE",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
}
TYPED_DECISIONS = {
    "MATCH_FAMILY_ONLY",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
}
CONFIDENCES = {"high", "medium", "low"}


def _resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else anchor.parent / value


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _exact_glob(root: Path, pattern: str, expected: Iterable[Path]) -> None:
    observed = {path.resolve() for path in root.glob(pattern)}
    wanted = {path.resolve() for path in expected}
    if observed != wanted:
        raise ValueError(
            f"artifact universe mismatch under {root}/{pattern}: "
            f"missing={sorted(map(str, wanted-observed))} "
            f"extra={sorted(map(str, observed-wanted))}"
        )


def _assert_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}: {path}")
    return actual


def _validate_bank_binding(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    task: str = TASK,
    expected_count: int = EXPECTED_BANK_COUNT,
) -> tuple[Path, Path, dict[str, Any], set[str], str]:
    bank_meta = (manifest.get("banks") or {}).get(task) or {}
    raw_bank_path = str(bank_meta.get("path") or "")
    raw_source_path = str(bank_meta.get("source_path") or "")
    bank_path = _resolve(raw_bank_path, manifest_path)
    source_path = _resolve(raw_source_path, manifest_path)
    bank_payload = _json(bank_path)
    bank_ids = {
        str(row.get("metric_id") or "") for row in bank_payload.get("metrics") or []
    }
    bank_sha = str(bank_meta.get("source_sha256") or "")
    expected_bank_path = (
        Path(raw_bank_path).resolve()
        if Path(raw_bank_path).is_absolute()
        else (manifest_path.parent / raw_bank_path).resolve()
    )
    expected_source_path = (
        Path(raw_source_path).resolve()
        if Path(raw_source_path).is_absolute()
        else (manifest_path.parent / raw_source_path).resolve()
    )
    if (
        bank_path != expected_bank_path
        or source_path != expected_source_path
        or int(bank_meta.get("count", -1)) != expected_count
        or len(bank_ids) != expected_count
        or "" in bank_ids
        or str(bank_payload.get("source_sha256") or "") != bank_sha
        or sha256_file(source_path) != bank_sha
    ):
        raise ValueError(
            f"authoritative {task} bank path/source artifact is not exactly the "
            f"manifest-bound {expected_count}-leaf bank"
        )
    return bank_path, source_path, bank_payload, bank_ids, bank_sha


def _validate_artifact_lock(
    *,
    artifact_lock_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    bank_path: Path,
    task: str = TASK,
    expected_bank_count: int = EXPECTED_BANK_COUNT,
    corpora: set[str] = EXPECTED_CORPORA,
) -> dict[str, Any]:
    lock = _json(artifact_lock_path)
    manifest_ref = lock.get("manifest") or {}
    bank_ref = (lock.get("banks") or {}).get(task) or {}
    if (
        lock.get("schema_version") != "silver-match-v3.0"
        or Path(str(manifest_ref.get("path") or "")).resolve()
        != manifest_path.resolve()
        or manifest_ref.get("sha256") != sha256_file(manifest_path)
        or Path(str(bank_ref.get("path") or "")).resolve() != bank_path.resolve()
        or int(bank_ref.get("count", -1)) != expected_bank_count
        or bank_ref.get("sha256") != sha256_file(bank_path)
    ):
        raise ValueError("artifact lock does not bind manifest/canonical task bank")
    norm_refs: dict[str, Any] = {}
    for corpus in sorted(corpora):
        manifest_corpus = (manifest.get("corpora") or {}).get(corpus) or {}
        lock_corpus = (lock.get("norms") or {}).get(corpus) or {}
        manifest_norm = Path(str(manifest_corpus.get("path") or ""))
        if not manifest_norm.is_absolute():
            manifest_norm = manifest_path.parent / manifest_norm
        locked_norm = Path(str(lock_corpus.get("path") or ""))
        if (
            manifest_corpus.get("task") != task
            or Path(locked_norm).resolve() != manifest_norm.resolve()
            or int(lock_corpus.get("count", -1))
            != int(manifest_corpus.get("count", -2))
            or lock_corpus.get("sha256") != sha256_file(manifest_norm)
        ):
            raise ValueError(f"artifact lock does not bind canonical norm file: {corpus}")
        norm_refs[corpus] = {
            "path": str(manifest_norm.resolve()),
            "count": int(lock_corpus["count"]),
            "sha256": sha256_file(manifest_norm),
        }
    return {
        "path": str(artifact_lock_path.resolve()),
        "sha256": sha256_file(artifact_lock_path),
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "bank": {
            "path": str(bank_path.resolve()),
            "count": expected_bank_count,
            "sha256": sha256_file(bank_path),
        },
        "norms": norm_refs,
    }


def _candidate_ids(row: dict[str, Any]) -> list[str]:
    values = [str(value.get("metric_id") or "") for value in row.get("candidates") or []]
    if not values or "" in values or len(values) != len(set(values)):
        raise ValueError(f"missing/duplicate candidate IDs for {row.get('norm_uid')}")
    return values


def _identity(row: dict[str, Any], bank_sha: str) -> tuple[str, str, int]:
    uid = str(row.get("norm_uid") or "")
    corpus = str(row.get("corpus") or "")
    task = str(row.get("task") or "")
    if (
        not uid
        or corpus not in EXPECTED_CORPORA
        or task != TASK
        or int(row.get("rescue_bank_count", EXPECTED_BANK_COUNT)) != EXPECTED_BANK_COUNT
        or str(
            row.get("bank_source_sha256")
            or row.get("candidate_bank_source_sha256")
            or ""
        )
        != bank_sha
    ):
        raise ValueError(f"task/corpus/bank identity mismatch: {uid!r}")
    return uid, corpus, int(row.get("row"))


def _compact_capture(
    path: Path, *, trial: int, bank_sha: str, bank_ids: set[str]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        uid, corpus, canonical_row = _identity(row, bank_sha)
        if uid in output:
            raise ValueError(f"duplicate capture UID: {path}/{uid}")
        ids = _candidate_ids(row)
        primary_ids = [str(value) for value in row.get("primary_candidate_ids") or []]
        if not set(ids) <= bank_ids:
            raise ValueError(f"capture candidate outside bank: {path}/{uid}")
        if (
            not primary_ids
            or len(primary_ids) != len(set(primary_ids))
            or not set(primary_ids) <= bank_ids
        ):
            raise ValueError(f"capture primary-slate binding invalid: {path}/{uid}")
        if (
            int(row.get("rescue_trial", -1)) != trial
            or int(row.get("rescue_bank_count", -1)) != EXPECTED_BANK_COUNT
            or not row.get("rescue_system")
            or not row.get("rescue_lane")
        ):
            raise ValueError(f"capture provenance mismatch: {path}/{uid}")
        output[uid] = {
            "corpus": corpus,
            "row": canonical_row,
            "candidate_ids": ids,
            "primary_candidate_ids": primary_ids,
            "rescue_system": row.get("rescue_system"),
            "rescue_lane": row.get("rescue_lane"),
            "rescue_capture": int(row.get("rescue_capture", -1)),
        }
    return output


def _compact_primary(
    paths: list[Path], *, bank_sha: str, bank_ids: set[str]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            corpus = str(row.get("corpus") or "")
            task = str(row.get("task") or "")
            candidate_ids = [str(value) for value in row.get("candidate_ids") or []]
            if (
                not uid
                or uid in output
                or corpus not in EXPECTED_CORPORA
                or task != TASK
                or str(
                    row.get("bank_source_sha256")
                    or row.get("candidate_bank_source_sha256")
                    or ""
                )
                != bank_sha
                or len(candidate_ids) != len(set(candidate_ids))
                or not set(candidate_ids) <= bank_ids
            ):
                raise ValueError(f"invalid/duplicate primary row: {path}/{uid!r}")
            output[uid] = {
                "corpus": corpus,
                "row": int(row.get("row")),
                "candidate_ids": candidate_ids,
            }
    return output


def _validate_capture_universe(
    captures: dict[str, dict[str, dict[str, Any]]],
    *,
    primary: dict[str, dict[str, Any]],
    bank_ids: set[str],
    coverage_repeats: int,
) -> dict[str, Any]:
    if set(captures) != set(EXPECTED_TRIALS):
        raise ValueError("capture trial-name universe mismatch")
    uid_sets = {name: set(rows) for name, rows in captures.items()}
    first_uids = uid_sets[EXPECTED_TRIALS[0]]
    if any(values != first_uids for values in uid_sets.values()):
        raise ValueError("cross-trial UID universe mismatch")
    if not first_uids <= set(primary):
        raise ValueError("rescue captures contain UIDs absent from primary inputs")
    system_counts: Counter[str] = Counter()
    capture_counts: Counter[int] = Counter()
    for uid in first_uids:
        reference = primary[uid]
        exposures: Counter[str] = Counter()
        systems: set[str] = set()
        captures_seen: set[int] = set()
        for name in EXPECTED_TRIALS:
            row = captures[name][uid]
            if (row["corpus"], row["row"]) != (
                reference["corpus"],
                reference["row"],
            ):
                raise ValueError(f"cross-trial/primary routing mismatch: {uid}")
            if row["primary_candidate_ids"] != reference["candidate_ids"]:
                raise ValueError(f"capture/primary candidate-set binding mismatch: {uid}")
            exposures.update(row["candidate_ids"])
            systems.add(str(row["rescue_system"]))
            captures_seen.add(int(row["rescue_capture"]))
        mismatched = {
            metric_id: exposures.get(metric_id, 0)
            for metric_id in bank_ids
            if exposures.get(metric_id, 0) != coverage_repeats
        }
        if set(exposures) != bank_ids or mismatched:
            raise ValueError(
                f"per-UID full-bank repeated coverage mismatch: {uid}: "
                f"missing={sorted(bank_ids-set(exposures))[:3]} "
                f"counts={list(sorted(mismatched.items()))[:3]}"
            )
        primary_ids = set(reference["candidate_ids"])
        if not primary_ids or not primary_ids <= set(exposures):
            raise ValueError(f"primary slate was not re-included: {uid}")
        if any(exposures[metric_id] != coverage_repeats for metric_id in primary_ids):
            raise ValueError(f"primary slate lacks repeated exposure: {uid}")
        if len(systems) < 2:
            raise ValueError(f"fewer than two distinct rescue systems: {uid}")
        if captures_seen != set(range(coverage_repeats)):
            raise ValueError(f"capture-repeat identities mismatch: {uid}")
        system_counts.update(systems)
        capture_counts.update(captures_seen)
    return {
        "uids": len(first_uids),
        "bank_metrics_per_uid": len(bank_ids),
        "exact_exposures_per_metric_per_uid": coverage_repeats,
        "minimum_distinct_systems_per_uid": 2,
        "primary_slate_reincluded_for_every_uid": True,
        "system_uid_counts": dict(sorted(system_counts.items())),
        "capture_uid_counts": {
            str(key): value for key, value in sorted(capture_counts.items())
        },
    }


def _validate_trial_output(
    path: Path,
    *,
    capture: dict[str, dict[str, Any]],
    trial: int,
    bank_sha: str,
) -> dict[str, Any]:
    observed: set[str] = set()
    decisions: Counter[str] = Counter()
    for row in read_jsonl(path):
        uid, corpus, canonical_row = _identity(row, bank_sha)
        if uid in observed or uid not in capture:
            raise ValueError(f"duplicate/extraneous trial output UID: {path}/{uid}")
        source = capture[uid]
        candidate_ids = [str(value) for value in row.get("candidate_ids") or []]
        if (
            corpus != source["corpus"]
            or canonical_row != source["row"]
            or candidate_ids != source["candidate_ids"]
            or row.get("rescue_system") != source["rescue_system"]
            or row.get("rescue_lane") != source["rescue_lane"]
            or int(row.get("rescue_trial", -1)) != trial
            or row.get("order_mode") != "original"
            or row.get("prompt_sha256") != EXPECTED_ADJUDICATOR_PROMPT
        ):
            raise ValueError(f"trial input/output binding mismatch: {path}/{uid}")
        decision = str(row.get("decision") or "")
        metric_id = row.get("metric_id")
        if decision == "MATCH" and str(metric_id or "") not in candidate_ids:
            raise ValueError(f"trial MATCH outside slate: {path}/{uid}")
        if decision != "MATCH" and metric_id is not None:
            raise ValueError(f"trial abstention carries metric: {path}/{uid}")
        observed.add(uid)
        decisions[decision] += 1
    if observed != set(capture):
        raise ValueError(f"trial output UID coverage mismatch: {path}")
    return {"count": len(observed), "decision_counts": dict(sorted(decisions.items()))}


def _validate_trial_meta(
    meta: dict[str, Any],
    *,
    name: str,
    input_path: Path,
    output_path: Path,
    expected_count: int,
) -> dict[str, Any] | None:
    if (
        meta.get("schema_version") != "silver-match-v3.0"
        or meta.get("order_mode") != "original"
        or int(meta.get("max_candidates", -1)) != 50
        or meta.get("prompt_sha256") != EXPECTED_ADJUDICATOR_PROMPT
        or meta.get("prompt_component_sha256") != EXPECTED_ADJUDICATOR_COMPONENTS
        or meta.get("input_candidates_sha256") != sha256_file(input_path)
        or meta.get("output_sha256") != sha256_file(output_path)
        or "google--gemma-4-31b-it" not in str(meta.get("model") or "")
    ):
        raise ValueError(f"trial metadata provenance mismatch: {name}")
    eligible = int(meta.get("eligible_count", -1))
    new = int(meta.get("new_count", -1))
    if eligible == expected_count and new == expected_count:
        return None
    if name == "trial-000" and eligible == 0 and new == 0:
        return {
            "artifact": name,
            "kind": "RESUME_METADATA_LIFECYCLE_COUNTS_OVERWRITTEN",
            "metadata_eligible_count": eligible,
            "metadata_new_count": new,
            "content_revalidated_count": expected_count,
            "output_sha256": sha256_file(output_path),
            "disposition": "LIFECYCLE_DEGRADED_CONTENT_REVALIDATED",
        }
    raise ValueError(
        f"unexpected trial metadata counts: {name}: eligible={eligible} new={new}"
    )


def _compact_finalists(
    path: Path, *, bank_sha: str, bank_ids: set[str]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        uid, corpus, canonical_row = _identity(row, bank_sha)
        if uid in output:
            raise ValueError(f"duplicate finalist UID: {uid}")
        ids = _candidate_ids(row)
        proposed = {str(value) for value in row.get("rescue_proposed_metric_ids") or []}
        if len(ids) > 16 or not set(ids) <= bank_ids or not proposed <= set(ids):
            raise ValueError(f"invalid finalist slate: {uid}")
        output[uid] = {"corpus": corpus, "row": canonical_row, "candidate_ids": ids}
    return output


def _compact_no_match(path: Path, *, bank_sha: str) -> dict[str, tuple[str, int]]:
    output: dict[str, tuple[str, int]] = {}
    for row in read_jsonl(path):
        uid, corpus, canonical_row = _identity(row, bank_sha)
        if uid in output:
            raise ValueError(f"duplicate no-match UID: {uid}")
        if (
            row.get("rescue_exhaustive") is not True
            or int(row.get("rescue_coverage_repeats", -1)) != 2
            or row.get("rescue_reincludes_primary") is not True
        ):
            raise ValueError(f"invalid no-match rescue provenance: {uid}")
        output[uid] = (corpus, canonical_row)
    return output


def _validate_adjudication_pair(
    path: Path,
    *,
    order: str,
    finalists: dict[str, dict[str, Any]],
    bank_sha: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        uid, corpus, canonical_row = _identity(row, bank_sha)
        if uid in output or uid not in finalists:
            raise ValueError(f"duplicate/extraneous finalist adjudication UID: {uid}")
        source = finalists[uid]
        ids = [str(value) for value in row.get("candidate_ids") or []]
        ids_ok = ids == source["candidate_ids"] if order == "original" else set(ids) == set(source["candidate_ids"])
        if (
            not ids_ok
            or corpus != source["corpus"]
            or canonical_row != source["row"]
            or row.get("order_mode") != order
            or row.get("prompt_sha256") != EXPECTED_ADJUDICATOR_PROMPT
        ):
            raise ValueError(f"finalist adjudication binding mismatch: {order}/{uid}")
        decision = str(row.get("decision") or "")
        metric_id = row.get("metric_id")
        if decision == "MATCH" and str(metric_id or "") not in ids:
            raise ValueError(f"finalist MATCH outside slate: {order}/{uid}")
        if decision != "MATCH" and metric_id is not None:
            raise ValueError(f"finalist abstention carries metric: {order}/{uid}")
        output[uid] = {
            "corpus": corpus,
            "row": canonical_row,
            "candidate_ids": ids,
            "decision": decision,
            "metric_id": metric_id,
            "model": row.get("model"),
        }
    if set(output) != set(finalists):
        raise ValueError(f"finalist adjudication coverage mismatch: {order}")
    return output


def _validate_inference_meta(
    meta: dict[str, Any],
    *,
    input_path: Path,
    output_path: Path,
    order: str,
    count: int,
    prompt_sha: str,
    components: dict[str, str] | None = None,
) -> None:
    component_value = meta.get("prompt_component_sha256")
    if component_value is None:
        component_value = meta.get("prompt_components")
    if (
        meta.get("order_mode") != order
        or meta.get("input_candidates_sha256", meta.get("audits_sha256"))
        != sha256_file(input_path)
        or meta.get("output_sha256") != sha256_file(output_path)
        or int(meta.get("eligible_count", meta.get("new_count", -1))) != count
        or meta.get("prompt_sha256") != prompt_sha
        or (components is not None and component_value != components)
        or "google--gemma-4-31b-it" not in str(meta.get("model") or "")
    ):
        raise ValueError(f"sealed inference metadata mismatch: {output_path}")


def _validate_verifier_output(
    path: Path,
    *,
    order: str,
    expected_uids: set[str],
    original: dict[str, dict[str, Any]],
    bank_sha: str,
) -> set[str]:
    observed: set[str] = set()
    for row in read_jsonl(path):
        uid, _, _ = _identity(row, bank_sha)
        if uid in observed or uid not in expected_uids:
            raise ValueError(f"duplicate/extraneous verifier UID: {order}/{uid}")
        primary = original[uid]
        alternatives = [str(value) for value in row.get("alternative_ids") or []]
        expected_alternatives = set(primary["candidate_ids"]) - {str(primary["metric_id"])}
        if (
            row.get("order_mode") != order
            or row.get("prompt_sha256") != EXPECTED_VERIFIER_PROMPT
            or row.get("primary_prompt_sha256") != EXPECTED_ADJUDICATOR_PROMPT
            or str(row.get("primary_metric_id") or "") != str(primary["metric_id"])
            or set(alternatives) != expected_alternatives
        ):
            raise ValueError(f"contrastive verifier binding mismatch: {order}/{uid}")
        decision = str(row.get("decision") or "")
        metric_id = row.get("metric_id")
        confidence = str(row.get("confidence") or "")
        parse_error = row.get("parse_error")
        if decision == "INVALID_OUTPUT":
            if not parse_error or metric_id is not None or confidence != "low":
                raise ValueError(f"invalid-output verifier schema mismatch: {order}/{uid}")
        elif (
            decision not in VERIFIER_DECISIONS
            or confidence not in CONFIDENCES
            or parse_error
            or (
                decision == "CONFIRM_MATCH"
                and str(metric_id or "") != str(primary["metric_id"])
            )
            or (decision == "BETTER_CANDIDATE" and str(metric_id or "") not in alternatives)
            or (
                decision not in {"CONFIRM_MATCH", "BETTER_CANDIDATE"}
                and metric_id is not None
            )
        ):
            raise ValueError(f"verifier decision/metric schema mismatch: {order}/{uid}")
        observed.add(uid)
    if observed != expected_uids:
        raise ValueError(f"contrastive verifier coverage mismatch: {order}")
    return observed


def _validate_typed_output(
    path: Path,
    *,
    order: str,
    expected: dict[str, tuple[str, int]],
    bank_sha: str,
) -> set[str]:
    observed: set[str] = set()
    for row in read_jsonl(path):
        uid, corpus, canonical_row = _identity(row, bank_sha)
        if uid in observed or uid not in expected:
            raise ValueError(f"duplicate/extraneous typed abstention UID: {order}/{uid}")
        if (
            (corpus, canonical_row) != expected[uid]
            or row.get("order_mode") != order
            or row.get("prompt_sha256") != EXPECTED_ABSTENTION_PROMPT
            or int(row.get("rescue_coverage_repeats", -1)) != 2
            or row.get("rescue_reincludes_primary") is not True
        ):
            raise ValueError(f"typed abstention binding mismatch: {order}/{uid}")
        decision = str(row.get("decision") or "")
        confidence = str(row.get("confidence") or "")
        parse_error = row.get("parse_error")
        if row.get("metric_id") is not None:
            raise ValueError(f"typed abstention carries metric: {order}/{uid}")
        if decision == "INVALID_OUTPUT":
            if not parse_error or confidence != "low":
                raise ValueError(f"invalid typed-output schema mismatch: {order}/{uid}")
        elif (
            decision not in TYPED_DECISIONS | {"POSSIBLE_EXACT_BANK_MATCH"}
            or confidence not in CONFIDENCES
            or parse_error
        ):
            raise ValueError(f"typed-output decision schema mismatch: {order}/{uid}")
        if decision in TYPED_DECISIONS and row.get("confirmed_decision") != decision:
            raise ValueError(f"typed-output confirmation mismatch: {order}/{uid}")
        if decision == "POSSIBLE_EXACT_BANK_MATCH" and row.get("possible_exact_bank_match") is not True:
            raise ValueError(f"typed possible-match flag mismatch: {order}/{uid}")
        observed.add(uid)
    if observed != set(expected):
        raise ValueError(f"typed abstention coverage mismatch: {order}")
    return observed


def _validate_combined(
    path: Path,
    *,
    expected_uids: set[str],
    bank_sha: str,
    kind: str,
) -> None:
    observed: set[str] = set()
    for row in read_jsonl(path):
        uid, _, _ = _identity(row, bank_sha)
        if uid in observed or uid not in expected_uids:
            raise ValueError(f"duplicate/extraneous {kind} combined UID: {uid}")
        observed.add(uid)
    if observed != expected_uids:
        raise ValueError(f"{kind} combined coverage mismatch")


def _assert_same_jsonl(left: Path, right: Path, label: str) -> None:
    left_rows = list(read_jsonl(left))
    right_rows = list(read_jsonl(right))
    if left_rows != right_rows:
        raise ValueError(f"independent recomputation differs: {label}")


def audit(
    *,
    manifest_path: Path,
    artifact_lock_path: Path,
    rescue_root: Path,
    primary_paths: list[Path],
    adjudicator_selection: Path,
    verifier_selection: Path,
    verifier_policy: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    artifact_lock_path = artifact_lock_path.resolve()
    rescue_root = rescue_root.resolve()
    primary_paths = [path.resolve() for path in primary_paths]
    manifest = _json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    bank_path, source_path, bank_payload, bank_ids, bank_sha = _validate_bank_binding(
        manifest_path, manifest
    )
    artifact_lock = _validate_artifact_lock(
        artifact_lock_path=artifact_lock_path,
        manifest_path=manifest_path,
        manifest=manifest,
        bank_path=bank_path,
    )
    manifest_corpora = {
        str(name)
        for name, meta in (manifest.get("corpora") or {}).items()
        if meta.get("task") == TASK
    }
    if manifest_corpora != EXPECTED_CORPORA:
        raise ValueError(f"unexpected N&C corpus universe: {sorted(manifest_corpora)}")

    captures_root = rescue_root / "captures"
    trials_root = rescue_root / "trial_adjudications"
    aggregate_root = rescue_root / "aggregate"
    finalists_root = rescue_root / "finalists"
    typed_root = rescue_root / "typed_abstentions"
    capture_paths = {name: captures_root / f"{name}.jsonl" for name in EXPECTED_TRIALS}
    trial_paths = {
        name: trials_root / f"{name}.original.jsonl" for name in EXPECTED_TRIALS
    }
    _exact_glob(captures_root, "trial-*.jsonl", capture_paths.values())
    _exact_glob(trials_root, "trial-*.original.jsonl", trial_paths.values())
    _exact_glob(
        trials_root,
        "trial-*.original.jsonl.meta.json",
        [path.with_suffix(path.suffix + ".meta.json") for path in trial_paths.values()],
    )

    rescue_manifest_path = captures_root / "rescue_manifest.json"
    rescue_manifest = _json(rescue_manifest_path)
    if (
        rescue_manifest.get("schema_version") != "silver-match-v3-abstention-rescue-v1"
        or rescue_manifest.get("manifest_sha256") != manifest_sha
        or rescue_manifest.get("eligible_by_task") != {TASK: 13263}
        or rescue_manifest.get("rescue_trials_by_task") != {TASK: 53052}
        or int(rescue_manifest.get("coverage_repeats", -1)) != 2
        or rescue_manifest.get("reinclude_primary") is not True
        or rescue_manifest.get("coverage_invariant")
        != "every frozen bank metric appears exactly coverage_repeats times in rescue trials"
        or set((rescue_manifest.get("outputs") or {}))
        != {str(path) for path in capture_paths.values()}
    ):
        raise ValueError("rescue manifest topology/provenance mismatch")
    recorded_primary = rescue_manifest.get("primary_inputs") or {}
    if set(recorded_primary) != {str(path) for path in primary_paths}:
        raise ValueError("primary input universe differs from rescue manifest")
    for path in primary_paths:
        _assert_hash(path, recorded_primary[str(path)], "primary")
    primary = _compact_primary(primary_paths, bank_sha=bank_sha, bank_ids=bank_ids)
    for raw_path, expected_hash in (rescue_manifest.get("candidate_inputs") or {}).items():
        _assert_hash(Path(raw_path), expected_hash, "candidate input")

    lifecycle_defects: list[dict[str, Any]] = []
    trial_summaries: dict[str, Any] = {}
    captures: dict[str, dict[str, dict[str, Any]]] = {}
    common_uids: set[str] | None = None
    common_model: str | None = None
    for trial, name in enumerate(EXPECTED_TRIALS):
        capture_path = capture_paths[name]
        capture_ref = rescue_manifest["outputs"][str(capture_path)]
        _assert_hash(capture_path, capture_ref["sha256"], "capture")
        capture = _compact_capture(
            capture_path, trial=trial, bank_sha=bank_sha, bank_ids=bank_ids
        )
        captures[name] = capture
        if len(capture) != int(capture_ref["count"]) or len(capture) != 13263:
            raise ValueError(f"capture count mismatch: {name}")
        if common_uids is None:
            common_uids = set(capture)
        elif set(capture) != common_uids:
            raise ValueError(f"capture UID universe mismatch: {name}")
        trial_path = trial_paths[name]
        content = _validate_trial_output(
            trial_path, capture=capture, trial=trial, bank_sha=bank_sha
        )
        meta = _json(trial_path.with_suffix(trial_path.suffix + ".meta.json"))
        defect = _validate_trial_meta(
            meta,
            name=name,
            input_path=capture_path,
            output_path=trial_path,
            expected_count=13263,
        )
        if defect:
            lifecycle_defects.append(defect)
        model = str(meta.get("model") or "")
        if common_model is None:
            common_model = model
        elif model != common_model:
            raise ValueError("trial model snapshot mismatch")
        trial_summaries[name] = {
            "capture": {
                "path": str(capture_path),
                "sha256": sha256_file(capture_path),
                "count": len(capture),
            },
            "adjudication": {
                "path": str(trial_path),
                "sha256": sha256_file(trial_path),
                **content,
            },
            "metadata": {
                "path": str(trial_path.with_suffix(trial_path.suffix + ".meta.json")),
                "sha256": sha256_file(
                    trial_path.with_suffix(trial_path.suffix + ".meta.json")
                ),
            },
        }
    assert common_uids is not None
    coverage_proof = _validate_capture_universe(
        captures,
        primary=primary,
        bank_ids=bank_ids,
        coverage_repeats=2,
    )

    aggregate_report_path = aggregate_root / "aggregate_report.json"
    aggregate_report = _json(aggregate_report_path)
    finalist_candidates = aggregate_root / "match_finalists.jsonl"
    no_match_path = aggregate_root / "no_match_provisional.jsonl"
    aggregate_outputs = aggregate_report.get("outputs") or {}
    aggregate_inputs = aggregate_report.get("adjudication_inputs") or {}
    if (
        aggregate_report.get("schema_version")
        != "silver-match-v3-abstention-rescue-aggregate-v1"
        or aggregate_report.get("manifest_sha256") != manifest_sha
        or aggregate_report.get("rescue_manifest_sha256")
        != sha256_file(rescue_manifest_path)
        or int(aggregate_report.get("expected_trial_rows", -1)) != 53052
        or int(aggregate_report.get("observed_trial_rows", -1)) != 53052
        or aggregate_report.get("primary_inputs") != recorded_primary
        or aggregate_inputs
        != {str(path): sha256_file(path) for path in trial_paths.values()}
        or set(aggregate_outputs) != {str(finalist_candidates), str(no_match_path)}
    ):
        raise ValueError("aggregate report topology/provenance mismatch")
    for path in (finalist_candidates, no_match_path):
        ref = aggregate_outputs[str(path)]
        _assert_hash(path, ref["sha256"], "aggregate output")
    finalists = _compact_finalists(
        finalist_candidates, bank_sha=bank_sha, bank_ids=bank_ids
    )
    no_match = _compact_no_match(no_match_path, bank_sha=bank_sha)
    if (
        len(finalists) != int(aggregate_outputs[str(finalist_candidates)]["count"])
        or len(no_match) != int(aggregate_outputs[str(no_match_path)]["count"])
        or set(finalists) & set(no_match)
        or (set(finalists) | set(no_match)) != common_uids
    ):
        raise ValueError("aggregate output partition/count mismatch")

    # Rebuild the aggregate from the sealed trials into a fresh temporary root.
    # This independently rechecks coverage, trial outcomes, finalist content,
    # and typed no-match consensus rather than trusting the historical report.
    with tempfile.TemporaryDirectory(prefix="notice-rescue-aggregate-recompute-") as raw:
        recompute_root = Path(raw)
        recomputed_report = aggregate_rescue(
            manifest_path=manifest_path,
            rescue_manifest_path=rescue_manifest_path,
            primary_paths=primary_paths,
            adjudication_paths=list(trial_paths.values()),
            output_root=recompute_root,
            max_finalists=16,
        )
        _assert_same_jsonl(
            recompute_root / "match_finalists.jsonl",
            finalist_candidates,
            "aggregate finalists",
        )
        _assert_same_jsonl(
            recompute_root / "no_match_provisional.jsonl",
            no_match_path,
            "aggregate no-match",
        )
        aggregate_recompute = {
            "implementation": {
                "path": str(Path(__file__).with_name("aggregate_abstention_rescue.py")),
                "sha256": sha256_file(
                    Path(__file__).with_name("aggregate_abstention_rescue.py")
                ),
            },
            "finalist_count": int(
                recomputed_report["outputs"][str(recompute_root / "match_finalists.jsonl")][
                    "count"
                ]
            ),
            "no_match_count": int(
                recomputed_report["outputs"][str(recompute_root / "no_match_provisional.jsonl")][
                    "count"
                ]
            ),
            "content_equal": True,
        }

    adj_paths = {
        order: finalists_root / f"adjudicate.{order}.jsonl"
        for order in ("original", "hashed")
    }
    verify_paths = {
        order: finalists_root / f"verify.{order}.jsonl"
        for order in ("original", "hashed")
    }
    _exact_glob(finalists_root, "adjudicate.*.jsonl", adj_paths.values())
    _exact_glob(
        finalists_root,
        "adjudicate.*.jsonl.meta.json",
        [path.with_suffix(path.suffix + ".meta.json") for path in adj_paths.values()],
    )
    _exact_glob(finalists_root, "verify.*.jsonl", [*verify_paths.values(), finalists_root / "verify.strict-combined.jsonl"])
    _exact_glob(
        finalists_root,
        "verify.*.jsonl.meta.json",
        [path.with_suffix(path.suffix + ".meta.json") for path in verify_paths.values()],
    )
    adjudications: dict[str, dict[str, dict[str, Any]]] = {}
    for order, path in adj_paths.items():
        adjudications[order] = _validate_adjudication_pair(
            path, order=order, finalists=finalists, bank_sha=bank_sha
        )
        meta = _json(path.with_suffix(path.suffix + ".meta.json"))
        _validate_inference_meta(
            meta,
            input_path=finalist_candidates,
            output_path=path,
            order=order,
            count=len(finalists),
            prompt_sha=EXPECTED_ADJUDICATOR_PROMPT,
            components=EXPECTED_ADJUDICATOR_COMPONENTS,
        )
        if str(meta.get("model") or "") != common_model:
            raise ValueError("finalist/trial model snapshot mismatch")
    for uid in finalists:
        left, right = adjudications["original"][uid], adjudications["hashed"][uid]
        if (
            left["corpus"] != right["corpus"]
            or left["row"] != right["row"]
            or set(left["candidate_ids"]) != set(right["candidate_ids"])
            or left["model"] != right["model"]
        ):
            raise ValueError(f"finalist order-pair mismatch: {uid}")
    match_uids = {
        uid for uid, row in adjudications["original"].items() if row["decision"] == "MATCH"
    }
    for order, path in verify_paths.items():
        _validate_verifier_output(
            path,
            order=order,
            expected_uids=match_uids,
            original=adjudications["original"],
            bank_sha=bank_sha,
        )
        meta = _json(path.with_suffix(path.suffix + ".meta.json"))
        _validate_inference_meta(
            meta,
            input_path=finalist_candidates,
            output_path=path,
            order=order,
            count=len(match_uids),
            prompt_sha=EXPECTED_VERIFIER_PROMPT,
            components=EXPECTED_VERIFIER_COMPONENTS,
        )
        if (
            meta.get("primary_sha256") != sha256_file(adj_paths["original"])
            or meta.get("manifest_sha256") != manifest_sha
            or str(meta.get("model") or "") != common_model
        ):
            raise ValueError(f"contrastive verifier meta linkage mismatch: {order}")
    combined_verify = finalists_root / "verify.strict-combined.jsonl"
    combined_verify_report_path = combined_verify.with_suffix(
        combined_verify.suffix + ".report.json"
    )
    combined_verify_report = _json(combined_verify_report_path)
    _validate_combined(
        combined_verify,
        expected_uids=match_uids,
        bank_sha=bank_sha,
        kind="finalist verification",
    )
    if (
        combined_verify_report.get("complete") is not True
        or int(combined_verify_report.get("count", -1)) != len(match_uids)
        or combined_verify_report.get("selected_prompt_sha256")
        != EXPECTED_VERIFIER_PROMPT
        or combined_verify_report.get("output_sha256") != sha256_file(combined_verify)
        or (combined_verify_report.get("inputs") or {}).get("original", {}).get("sha256")
        != sha256_file(verify_paths["original"])
        or (combined_verify_report.get("inputs") or {}).get("hashed", {}).get("sha256")
        != sha256_file(verify_paths["hashed"])
        or (combined_verify_report.get("inputs") or {}).get("primary", {}).get("sha256")
        != sha256_file(adj_paths["original"])
        or (combined_verify_report.get("inputs") or {}).get("selection", {}).get("sha256")
        != sha256_file(verifier_selection)
        or (combined_verify_report.get("inputs") or {}).get("policy", {}).get("sha256")
        != sha256_file(verifier_policy)
    ):
        raise ValueError("strict finalist verification report linkage mismatch")
    with tempfile.TemporaryDirectory(prefix="notice-rescue-verifier-recompute-") as raw:
        recomputed_verify = Path(raw) / "verify.strict-combined.jsonl"
        recomputed_verify_report = combine_verifications(
            primary_path=adj_paths["original"],
            original_path=verify_paths["original"],
            hashed_path=verify_paths["hashed"],
            selection_path=verifier_selection,
            policy_path=verifier_policy,
            output_path=recomputed_verify,
        )
        _assert_same_jsonl(
            recomputed_verify, combined_verify, "strict finalist verification"
        )
        if recomputed_verify_report.get("counts") != combined_verify_report.get("counts"):
            raise ValueError("strict finalist verification recomputed counts differ")

    typed_paths = {
        order: typed_root / f"verify.{order}.jsonl"
        for order in ("original", "hashed")
    }
    combined_typed = typed_root / "verify.strict-combined.jsonl"
    _exact_glob(typed_root, "verify.*.jsonl", [*typed_paths.values(), combined_typed])
    _exact_glob(
        typed_root,
        "verify.*.jsonl.meta.json",
        [path.with_suffix(path.suffix + ".meta.json") for path in typed_paths.values()],
    )
    for order, path in typed_paths.items():
        _validate_typed_output(
            path, order=order, expected=no_match, bank_sha=bank_sha
        )
        meta = _json(path.with_suffix(path.suffix + ".meta.json"))
        _validate_inference_meta(
            meta,
            input_path=no_match_path,
            output_path=path,
            order=order,
            count=len(no_match),
            prompt_sha=EXPECTED_ABSTENTION_PROMPT,
            components={
                "scripts/tools/silver_match_v3/prompts/verify_abstention_v1.txt": EXPECTED_ABSTENTION_PROMPT
            },
        )
        if meta.get("manifest_sha256") != manifest_sha or str(meta.get("model") or "") != common_model:
            raise ValueError(f"typed abstention meta linkage mismatch: {order}")
    combined_typed_report_path = combined_typed.with_suffix(
        combined_typed.suffix + ".report.json"
    )
    combined_typed_report = _json(combined_typed_report_path)
    _validate_combined(
        combined_typed,
        expected_uids=set(no_match),
        bank_sha=bank_sha,
        kind="typed abstention",
    )
    if (
        combined_typed_report.get("complete") is not True
        or int(combined_typed_report.get("count", -1)) != len(no_match)
        or combined_typed_report.get("output_sha256") != sha256_file(combined_typed)
        or (combined_typed_report.get("inputs") or {}).get("audits", {}).get("sha256")
        != sha256_file(no_match_path)
        or (combined_typed_report.get("inputs") or {}).get("original", {}).get("sha256")
        != sha256_file(typed_paths["original"])
        or (combined_typed_report.get("inputs") or {}).get("hashed", {}).get("sha256")
        != sha256_file(typed_paths["hashed"])
    ):
        raise ValueError("strict typed abstention report linkage mismatch")
    with tempfile.TemporaryDirectory(prefix="notice-rescue-typed-recompute-") as raw:
        recomputed_typed = Path(raw) / "verify.strict-combined.jsonl"
        recomputed_typed_report = combine_abstentions(
            audits_path=no_match_path,
            original_path=typed_paths["original"],
            hashed_path=typed_paths["hashed"],
            output_path=recomputed_typed,
        )
        _assert_same_jsonl(
            recomputed_typed, combined_typed, "strict typed abstention verification"
        )
        if recomputed_typed_report.get("counts") != combined_typed_report.get("counts"):
            raise ValueError("strict typed abstention recomputed counts differ")

    implementations_root = Path(__file__).resolve().parent
    implementations = {
        name: {
            "path": str(implementations_root / name),
            "sha256": sha256_file(implementations_root / name),
        }
        for name in CPU_IMPLEMENTATIONS
    }
    audit_recompute_implementations = {
        name: {
            "path": str(implementations_root / name),
            "sha256": sha256_file(implementations_root / name),
        }
        for name in AUDIT_RECOMPUTE_IMPLEMENTATIONS
    }
    artifacts = {
        str(path): sha256_file(path)
        for path in [
            manifest_path,
            artifact_lock_path,
            bank_path,
            source_path,
            *[Path(ref["path"]) for ref in artifact_lock["norms"].values()],
            rescue_manifest_path,
            aggregate_report_path,
            finalist_candidates,
            no_match_path,
            *capture_paths.values(),
            *trial_paths.values(),
            *[path.with_suffix(path.suffix + ".meta.json") for path in trial_paths.values()],
            *adj_paths.values(),
            *[path.with_suffix(path.suffix + ".meta.json") for path in adj_paths.values()],
            *verify_paths.values(),
            *[path.with_suffix(path.suffix + ".meta.json") for path in verify_paths.values()],
            combined_verify,
            combined_verify_report_path,
            *typed_paths.values(),
            *[path.with_suffix(path.suffix + ".meta.json") for path in typed_paths.values()],
            combined_typed,
            combined_typed_report_path,
            *primary_paths,
            adjudicator_selection,
            verifier_selection,
            verifier_policy,
        ]
    }
    return {
        "schema_version": "silver-match-v3-notice-sealed-rescue-audit-v1",
        "status": "PASS_SEALED_GPU_ARTIFACTS_CONTENT_REVALIDATED",
        "complete": True,
        "task": TASK,
        "corpora": sorted(EXPECTED_CORPORA),
        "authoritative_bank": {
            "path": str(bank_path),
            "file_sha256": sha256_file(bank_path),
            "source_sha256": bank_sha,
            "source_path": str(source_path),
            "source_file_sha256": sha256_file(source_path),
            "metric_count": len(bank_ids),
        },
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "artifact_lock": artifact_lock,
        "model_snapshot": common_model,
        "trial_count": len(EXPECTED_TRIALS),
        "trial_rows": sum(value["adjudication"]["count"] for value in trial_summaries.values()),
        "unique_rescue_rows": len(common_uids),
        "finalist_rows": len(finalists),
        "no_match_rows": len(no_match),
        "contrastive_verifier_rows": len(match_uids),
        "lifecycle_metadata_defects": lifecycle_defects,
        "trial_summaries": trial_summaries,
        "per_uid_full_bank_coverage_proof": coverage_proof,
        "aggregate_independent_recomputation": aggregate_recompute,
        "strict_combines_independently_recomputed": {
            "finalist_verification": True,
            "typed_abstention": True,
        },
        "artifacts": dict(sorted(artifacts.items())),
        "cpu_continuation_implementations": implementations,
        "audit_recompute_implementations": audit_recompute_implementations,
        "gpu_inference_required": False,
        "gpu_implementation_hashes_used_as_acceptance_gate": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-lock", required=True)
    parser.add_argument("--rescue-root", required=True)
    parser.add_argument("--primary", action="append", required=True)
    parser.add_argument("--adjudicator-selection", required=True)
    parser.add_argument("--verifier-selection", required=True)
    parser.add_argument("--verifier-policy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit(
        manifest_path=Path(args.manifest),
        artifact_lock_path=Path(args.artifact_lock),
        rescue_root=Path(args.rescue_root),
        primary_paths=[Path(path) for path in args.primary],
        adjudicator_selection=Path(args.adjudicator_selection).resolve(),
        verifier_selection=Path(args.verifier_selection).resolve(),
        verifier_policy=Path(args.verifier_policy).resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "trial_rows": report["trial_rows"],
                "unique_rescue_rows": report["unique_rescue_rows"],
                "output": str(output),
                "output_sha256": sha256_file(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
