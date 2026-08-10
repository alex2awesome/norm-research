#!/usr/bin/env python3
"""Freeze a source-disjoint, truth-hidden Humor CE label collection.

The collection has three roles with different sampling contracts:

* ``blind`` is a natural, corpus-proportional hash sample from the canonical
  Humor population.  Candidate inventories are not consulted for this role.
* ``dev`` is balanced across explicitly declared *candidate* strata.
* ``train`` follows explicitly declared candidate-stratum quotas.

Candidate strata are weak sampling signals, never asserted labels.  The
labeler-facing items and chunks omit them entirely.  Every supplied prior-role
file is excluded by both canonical UID and document-level source identity.
Colon-delimited and unit-separator source-group encodings are treated as
aliases of the same canonical identity.

The output root is append-only and is directly compatible with
``validate_independent_teacher_labels``: it contains ``items.jsonl``, a full
``bank.json``, deterministic ``chunks/part-*.jsonl``, and ``validation.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import normalize_space, read_jsonl, sha256_file, write_jsonl


TASK = "humor"
EXPECTED_CANONICAL_NORMS = 77_378
EXPECTED_BANK_METRICS = 285
EXPECTED_BANK_SOURCE_SHA256 = (
    "1b4a29d34b4ef4d999e0cb0b2d1125286372349ff6dfa21a6adc5bc8e76f0de9"
)
DEFAULT_TRAIN_COUNT = 5_000
DEFAULT_DEV_COUNT = 600
DEFAULT_BLIND_COUNT = 1_000
DEFAULT_SEED = 2026071307
STRATUM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
ROLE_ORDER = {"train": 0, "dev": 1, "blind": 2}

# Only canonical evidence/provenance fields needed by a full-bank labeler are
# hydrated.  In particular, no candidate flag or weak prior label can enter a
# chunk accidentally through ``{**norm}`` expansion.
SAFE_ITEM_FIELDS = (
    "row",
    "kind",
    "norm",
    "context",
    "aspect",
    "polarity",
    "source_id",
    "paper_id",
    "source_segment",
    "source_record_row",
    "source_signal_index",
)
FORBIDDEN_LABELER_FIELDS = {
    "decision",
    "metric_id",
    "acceptable_metric_ids",
    "confidence",
    "reason",
    "label",
    "prediction",
    "predictions",
    "outcome",
    "candidate_flags",
    "candidate_strata",
    "selection_stratum",
    "inventory_memberships",
}


@dataclass(frozen=True)
class NamedPath:
    name: str | None
    path: Path


@dataclass(frozen=True)
class Quota:
    stratum: str
    count: int


@dataclass(frozen=True)
class CollectionConfig:
    manifest: Path
    candidate_inventories: tuple[NamedPath, ...]
    exclusion_roles: tuple[NamedPath, ...]
    train_quotas: tuple[Quota, ...]
    dev_quotas: tuple[Quota, ...]
    output_root: Path
    train_count: int = DEFAULT_TRAIN_COUNT
    dev_count: int = DEFAULT_DEV_COUNT
    blind_count: int = DEFAULT_BLIND_COUNT
    chunk_size: int = 20
    seed: int = DEFAULT_SEED


def _stable(seed: int, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in (seed, *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve(value: str | Path, anchor: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def _ref(path: Path, *, count: int | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if count is not None:
        value["count"] = count
    return value


def _valid_stratum(value: str) -> str:
    value = value.strip()
    if not STRATUM_RE.fullmatch(value):
        raise ValueError(f"invalid candidate stratum name: {value!r}")
    return value


def _parse_named_path(value: str, *, name_required: bool) -> NamedPath:
    name: str | None = None
    path_value = value
    if "=" in value:
        possible_name, possible_path = value.split("=", 1)
        if possible_name and possible_path:
            name = _valid_stratum(possible_name)
            path_value = possible_path
    if name_required and name is None:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return NamedPath(name=name, path=path)


def _parse_quota(value: str) -> Quota:
    if "=" not in value:
        raise ValueError(f"expected STRATUM=COUNT, got {value!r}")
    name, raw_count = value.split("=", 1)
    stratum = _valid_stratum(name)
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise ValueError(f"invalid quota count: {value!r}") from exc
    if count < 1:
        raise ValueError(f"quota must be positive: {value!r}")
    return Quota(stratum=stratum, count=count)


def _canonical_group(norm: Mapping[str, Any]) -> str:
    task = normalize_space(norm.get("task"))
    corpus = normalize_space(norm.get("corpus"))
    paper_id = normalize_space(norm.get("paper_id"))
    source_id = normalize_space(norm.get("source_id"))
    uid = normalize_space(norm.get("norm_uid"))
    if paper_id:
        kind, identity = "paper", paper_id
    elif source_id:
        kind, identity = "source", source_id
    else:
        kind, identity = "norm", uid
    return "\x1f".join((task, corpus, kind, identity))


def _group_aliases(norm: Mapping[str, Any]) -> set[str]:
    canonical = _canonical_group(norm)
    task, corpus, kind, identity = canonical.split("\x1f", 3)
    return {
        canonical,
        ":".join((corpus, kind, identity)),
        ":".join((task, corpus, kind, identity)),
    }


def _load_universe(
    manifest_path: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[str]],
    dict[str, str],
    dict[str, Any],
    Path,
    dict[str, Any],
]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_meta = (manifest.get("banks") or {}).get(TASK)
    if not isinstance(bank_meta, dict):
        raise ValueError("manifest lacks the Humor bank")
    if str(bank_meta.get("source_sha256") or "") != EXPECTED_BANK_SOURCE_SHA256:
        raise ValueError("manifest does not bind the exact Humor R2 bank source SHA")
    bank_path = _resolve(str(bank_meta.get("path") or ""), manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    metrics = list(bank.get("metrics") or [])
    metric_ids = [str(row.get("metric_id") or "") for row in metrics]
    if (
        bank.get("task") != TASK
        or bank.get("source_sha256") != EXPECTED_BANK_SOURCE_SHA256
        or len(metrics) != EXPECTED_BANK_METRICS
        or "" in metric_ids
        or len(metric_ids) != len(set(metric_ids))
    ):
        raise ValueError("bank is not the exact unique 285-card Humor bank")

    norms: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    alias_to_group: dict[str, str] = {}
    corpus_inputs: dict[str, Any] = {}
    for corpus, meta in sorted((manifest.get("corpora") or {}).items()):
        if meta.get("task") != TASK:
            continue
        norm_path = _resolve(str(meta.get("path") or ""), manifest_path)
        count = 0
        for row in read_jsonl(norm_path):
            uid = normalize_space(row.get("norm_uid"))
            if not uid or uid in norms:
                raise ValueError(f"missing or duplicate canonical Humor UID: {uid!r}")
            if row.get("task") != TASK or str(row.get("corpus") or "") != corpus:
                raise ValueError(f"canonical task/corpus mismatch for {uid}")
            if row.get("row") is None:
                raise ValueError(f"canonical norm lacks row provenance: {uid}")
            norms[uid] = row
            group = _canonical_group(row)
            groups[group].append(uid)
            for alias in _group_aliases(row):
                prior = alias_to_group.setdefault(alias, group)
                if prior != group:
                    raise ValueError(f"ambiguous source-group alias: {alias!r}")
            count += 1
        corpus_inputs[corpus] = _ref(norm_path, count=count)
    if len(norms) != EXPECTED_CANONICAL_NORMS:
        raise ValueError(
            f"expected {EXPECTED_CANONICAL_NORMS} canonical Humor norms, got {len(norms)}"
        )
    return norms, dict(groups), alias_to_group, bank, bank_path, corpus_inputs


def _resolve_group(value: object, alias_to_group: Mapping[str, str]) -> str:
    # Do not call normalize_space: it collapses the unit separator that is a
    # structural delimiter in the canonical remote representation.
    rendered = str(value or "").strip(" \t\r\n")
    if not rendered:
        raise ValueError("empty supplied source group")
    try:
        return alias_to_group[rendered]
    except KeyError as exc:
        raise ValueError(f"source group is absent from canonical Humor: {rendered!r}") from exc


def _supplied_groups(row: Mapping[str, Any]) -> list[str]:
    return [
        str(row[field])
        for field in ("source_group", "split_group", "gepa_split_group")
        if row.get(field) not in (None, "")
    ]


def _validate_row_identity(
    row: Mapping[str, Any],
    *,
    source: str,
    norms: Mapping[str, Mapping[str, Any]],
    alias_to_group: Mapping[str, str],
    allow_group_only: bool,
) -> tuple[str | None, str]:
    row_task = str(row.get("task") or "")
    if row_task and row_task != TASK:
        raise ValueError(f"{source}: non-Humor row reached Humor identity validation")
    uid = normalize_space(row.get("norm_uid"))
    if uid:
        if uid not in norms:
            raise ValueError(f"{source}: UID absent from canonical Humor: {uid}")
        canonical = _canonical_group(norms[uid])
    elif allow_group_only:
        canonical = ""
    else:
        raise ValueError(f"{source}: candidate row lacks norm_uid")
    supplied = _supplied_groups(row)
    resolved = [_resolve_group(value, alias_to_group) for value in supplied]
    if canonical and any(group != canonical for group in resolved):
        raise ValueError(f"{source}: source group disagrees with canonical UID {uid}")
    if not canonical:
        if not resolved or len(set(resolved)) != 1:
            raise ValueError(f"{source}: group-only row lacks one canonical source identity")
        canonical = resolved[0]
    return uid or None, canonical


def _load_exclusions(
    specs: Sequence[NamedPath],
    *,
    norms: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, Sequence[str]],
    alias_to_group: Mapping[str, str],
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    excluded_uids: set[str] = set()
    excluded_groups: set[str] = set()
    reports: list[dict[str, Any]] = []
    for spec in specs:
        lines = [
            line.strip()
            for line in spec.path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        if not lines:
            raise ValueError(f"empty exclusion role file: {spec.path}")
        local_uids: set[str] = set()
        local_groups: set[str] = set()
        skipped_other_task = 0
        if all(line.startswith("{") for line in lines):
            rows: Iterable[Mapping[str, Any]] = read_jsonl(spec.path)
            file_format = "jsonl"
            for row in rows:
                row_task = str(row.get("task") or "")
                if row_task and row_task != TASK:
                    skipped_other_task += 1
                    continue
                uid, group = _validate_row_identity(
                    row,
                    source=f"exclusion {spec.name}",
                    norms=norms,
                    alias_to_group=alias_to_group,
                    allow_group_only=True,
                )
                if uid is not None:
                    local_uids.add(uid)
                local_groups.add(group)
        elif any(line.startswith("{") for line in lines):
            raise ValueError(f"mixed JSONL/plain exclusion file: {spec.path}")
        else:
            file_format = "newline_uid_or_source_group"
            if len(lines) != len(set(lines)):
                raise ValueError(f"duplicate values in exclusion role file: {spec.path}")
            for value in lines:
                if value in norms:
                    local_uids.add(value)
                    local_groups.add(_canonical_group(norms[value]))
                else:
                    local_groups.add(_resolve_group(value, alias_to_group))
        excluded_uids.update(local_uids)
        excluded_groups.update(local_groups)
        reports.append(
            {
                "role": spec.name,
                **_ref(spec.path),
                "format": file_format,
                "direct_uids": len(local_uids),
                "canonical_source_groups": len(local_groups),
                "canonical_uids_covered_by_groups": len(
                    {uid for group in local_groups for uid in groups[group]}
                ),
                "skipped_explicit_other_task_rows": skipped_other_task,
            }
        )
    # A source-group exclusion always dominates an individual UID exclusion.
    excluded_uids.update(uid for group in excluded_groups for uid in groups[group])
    return excluded_uids, excluded_groups, reports


def _row_flags(row: Mapping[str, Any]) -> set[str]:
    flags: set[str] = set()
    # The audited Humor inventory emits ``priority_strata`` plus a single
    # ``primary_priority_stratum``.  Consume those forms directly so freezing
    # never depends on an untracked filtering transform.  These values remain
    # weak selection signals and are stripped from labeler-facing outputs.
    for field in (
        "candidate_flags",
        "candidate_strata",
        "strata",
        "priority_strata",
        "primary_priority_stratum",
    ):
        value = row.get(field)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            values = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, list):
            values = [str(part).strip() for part in value if str(part).strip()]
        else:
            raise ValueError(f"candidate flag field {field} must be string or list")
        flags.update(_valid_stratum(part) for part in values)
    return flags


def _load_candidates(
    specs: Sequence[NamedPath],
    *,
    norms: Mapping[str, Mapping[str, Any]],
    alias_to_group: Mapping[str, str],
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    memberships: dict[str, set[str]] = defaultdict(set)
    reports: list[dict[str, Any]] = []
    for spec in specs:
        seen: set[str] = set()
        task_rows = 0
        skipped_other_task = 0
        inventory_flags: Counter[str] = Counter()
        for row in read_jsonl(spec.path):
            row_task = str(row.get("task") or "")
            if row_task and row_task != TASK:
                skipped_other_task += 1
                continue
            uid, _ = _validate_row_identity(
                row,
                source=f"candidate inventory {spec.path}",
                norms=norms,
                alias_to_group=alias_to_group,
                allow_group_only=False,
            )
            assert uid is not None
            if uid in seen:
                raise ValueError(f"candidate inventory repeats UID: {spec.path}/{uid}")
            seen.add(uid)
            flags = _row_flags(row)
            if spec.name is not None:
                flags.add(spec.name)
            if not flags:
                raise ValueError(
                    f"candidate row has no declared inventory stratum: {spec.path}/{uid}"
                )
            memberships[uid].update(flags)
            inventory_flags.update(flags)
            task_rows += 1
        reports.append(
            {
                "declared_inventory_stratum": spec.name,
                **_ref(spec.path),
                "humor_rows": task_rows,
                "unique_humor_uids": len(seen),
                "candidate_flag_memberships": dict(sorted(inventory_flags.items())),
                "skipped_explicit_other_task_rows": skipped_other_task,
                "payload_fields_used": [
                    "task",
                    "norm_uid",
                    "source_group aliases when supplied",
                    (
                        "candidate_flags/candidate_strata/strata/priority_strata/"
                        "primary_priority_stratum when supplied"
                    ),
                ],
            }
        )
    if not memberships:
        raise ValueError("candidate inventories contain no Humor identities")
    return dict(memberships), reports


def _largest_remainder(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    population = sum(sizes.values())
    if total < 0 or total > population:
        raise ValueError("requested natural blind count exceeds eligible source groups")
    if population == 0:
        if total == 0:
            return {key: 0 for key in sizes}
        raise ValueError("natural blind population is empty")
    exact = {key: total * value / population for key, value in sizes.items()}
    allocated = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(allocated.values())
    order = sorted(sizes, key=lambda key: (-(exact[key] - allocated[key]), key))
    for key in order[:remainder]:
        allocated[key] += 1
    return allocated


def _natural_blind(
    *,
    norms: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, Sequence[str]],
    excluded_groups: set[str],
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_corpus: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for group, uids in groups.items():
        if group in excluded_groups:
            continue
        representative = min(
            uids,
            key=lambda uid: (_stable(seed, "blind-representative", uid), uid),
        )
        by_corpus[str(norms[representative]["corpus"])].append((group, representative))
    population = {corpus: len(values) for corpus, values in sorted(by_corpus.items())}
    quotas = _largest_remainder(population, count)
    selected: list[dict[str, Any]] = []
    for corpus, values in sorted(by_corpus.items()):
        ordered = sorted(
            values,
            key=lambda value: (
                _stable(seed, "blind-natural", corpus, value[0]),
                value[0],
            ),
        )
        for group, uid in ordered[: quotas[corpus]]:
            selected.append(
                {
                    "norm_uid": uid,
                    "source_group": group,
                    "corpus": corpus,
                    "collection_role": "blind",
                    "split": "test",
                }
            )
    selected.sort(key=lambda row: (_stable(seed, "blind-output", row["source_group"]), row["norm_uid"]))
    if len(selected) != count:
        raise AssertionError("natural blind allocation drifted")
    return selected, {
        "sampling": "natural corpus-proportional source-group hash sample",
        "candidate_inventories_consulted": False,
        "eligible_source_group_population_by_corpus": population,
        "quota_by_corpus": quotas,
        "selected_by_corpus": dict(sorted(Counter(row["corpus"] for row in selected).items())),
    }


def _select_candidate_role(
    *,
    role: str,
    quotas: Sequence[Quota],
    memberships: Mapping[str, set[str]],
    norms: Mapping[str, Mapping[str, Any]],
    used_groups: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for quota_ordinal, quota in enumerate(quotas):
        by_group: dict[str, list[str]] = defaultdict(list)
        for uid, flags in memberships.items():
            if quota.stratum not in flags:
                continue
            group = _canonical_group(norms[uid])
            if group not in used_groups:
                by_group[group].append(uid)
        ordered_groups = sorted(
            by_group,
            key=lambda group: (
                _stable(seed, role, quota_ordinal, quota.stratum, group),
                group,
            ),
        )
        if len(ordered_groups) < quota.count:
            raise ValueError(
                f"{role}/{quota.stratum} has {len(ordered_groups)} disjoint source groups; "
                f"quota requires {quota.count}"
            )
        for group in ordered_groups[: quota.count]:
            uid = min(
                by_group[group],
                key=lambda value: (
                    _stable(seed, role, quota.stratum, "uid", value),
                    value,
                ),
            )
            selected.append(
                {
                    "norm_uid": uid,
                    "source_group": group,
                    "corpus": str(norms[uid]["corpus"]),
                    "collection_role": role,
                    "split": role,
                    "selection_stratum": quota.stratum,
                    "inventory_memberships": sorted(memberships[uid]),
                    "quota_target_is_sampling_metadata_not_label": True,
                }
            )
            used_groups.add(group)
    selected.sort(
        key=lambda row: (
            _stable(seed, role, "output", row["source_group"]),
            row["norm_uid"],
        )
    )
    return selected


def _validate_config(config: CollectionConfig) -> None:
    if config.output_root.exists():
        raise FileExistsError(f"append-only output root already exists: {config.output_root}")
    if not config.candidate_inventories:
        raise ValueError("at least one candidate inventory is required")
    if not config.exclusion_roles:
        raise ValueError("at least one prior-role exclusion file is required")
    if min(config.train_count, config.dev_count, config.blind_count, config.chunk_size) < 1:
        raise ValueError("role counts and chunk size must be positive")
    for role, quotas, target in (
        ("train", config.train_quotas, config.train_count),
        ("dev", config.dev_quotas, config.dev_count),
    ):
        names = [quota.stratum for quota in quotas]
        if not quotas or len(names) != len(set(names)):
            raise ValueError(f"{role} quotas are empty or repeat a stratum")
        if sum(quota.count for quota in quotas) != target:
            raise ValueError(f"{role} quotas must sum exactly to {target}")
    dev_values = [quota.count for quota in config.dev_quotas]
    if max(dev_values) - min(dev_values) > 1:
        raise ValueError("dev candidate-stratum quotas must be balanced within one row")


def _hydrate_item(
    norm: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    item = {field: norm[field] for field in SAFE_ITEM_FIELDS if field in norm}
    item.update(
        {
            "schema_version": "silver-match-v3-humor-ce-truth-item-v1",
            "task": TASK,
            "corpus": str(norm["corpus"]),
            "norm_uid": str(norm["norm_uid"]),
            "source_group": str(identity["source_group"]),
            "split_group": str(identity["source_group"]),
            "split": str(identity["split"]),
            "predeclared_split": str(identity["split"]),
            "collection_role": str(identity["collection_role"]),
            "truth_hidden": True,
        }
    )
    leaked = FORBIDDEN_LABELER_FIELDS & set(item)
    if leaked:
        raise AssertionError(f"labeler item leaks fields: {sorted(leaked)}")
    return item


def freeze_collection(config: CollectionConfig) -> dict[str, Any]:
    """Build the immutable collection and return its validation payload."""

    _validate_config(config)
    manifest_path = config.manifest.resolve()
    (
        norms,
        groups,
        alias_to_group,
        bank,
        bank_path,
        corpus_inputs,
    ) = _load_universe(manifest_path)
    excluded_uids, excluded_groups, exclusion_reports = _load_exclusions(
        config.exclusion_roles,
        norms=norms,
        groups=groups,
        alias_to_group=alias_to_group,
    )

    # Freeze the natural blind panel before candidate inventories are loaded.
    blind, blind_report = _natural_blind(
        norms=norms,
        groups=groups,
        excluded_groups=excluded_groups,
        count=config.blind_count,
        seed=config.seed,
    )
    used_groups = {str(row["source_group"]) for row in blind} | set(excluded_groups)

    memberships, candidate_reports = _load_candidates(
        config.candidate_inventories,
        norms=norms,
        alias_to_group=alias_to_group,
    )
    # Evaluation capacity is protected before train sampling.  Both operations
    # are driven only by declared inventory membership and fixed hashes.
    dev = _select_candidate_role(
        role="dev",
        quotas=config.dev_quotas,
        memberships=memberships,
        norms=norms,
        used_groups=used_groups,
        seed=config.seed,
    )
    train = _select_candidate_role(
        role="train",
        quotas=config.train_quotas,
        memberships=memberships,
        norms=norms,
        used_groups=used_groups,
        seed=config.seed,
    )
    if len(dev) != config.dev_count or len(train) != config.train_count:
        raise AssertionError("candidate role counts drifted from declared quotas")

    identities = train + dev + blind
    uid_sets = {
        role: {str(row["norm_uid"]) for row in identities if row["collection_role"] == role}
        for role in ROLE_ORDER
    }
    group_sets = {
        role: {str(row["source_group"]) for row in identities if row["collection_role"] == role}
        for role in ROLE_ORDER
    }
    for left in ROLE_ORDER:
        for right in ROLE_ORDER:
            if ROLE_ORDER[left] >= ROLE_ORDER[right]:
                continue
            if uid_sets[left] & uid_sets[right] or group_sets[left] & group_sets[right]:
                raise AssertionError(f"collection roles overlap: {left}/{right}")
    selected_uids = set().union(*uid_sets.values())
    selected_groups = set().union(*group_sets.values())
    if selected_uids & excluded_uids or selected_groups & excluded_groups:
        raise AssertionError("selected identities overlap a supplied role exclusion")

    identity_rows: list[dict[str, Any]] = []
    for row in identities:
        identity = {
            "schema_version": "silver-match-v3-humor-ce-truth-identity-v1",
            "task": TASK,
            **row,
            "truth_hidden": True,
            "labels_predictions_metric_ids_reasons_mi_and_outcomes_used": False,
        }
        identity_rows.append(identity)
    identity_rows.sort(
        key=lambda row: (
            ROLE_ORDER[str(row["collection_role"])],
            _stable(config.seed, "identity-output", row["source_group"]),
            row["norm_uid"],
        )
    )
    identity_by_uid = {str(row["norm_uid"]): row for row in identity_rows}
    if len(identity_by_uid) != len(identity_rows):
        raise AssertionError("identity output repeats a UID")
    items = [_hydrate_item(norms[uid], identity_by_uid[uid]) for uid in identity_by_uid]
    items.sort(
        key=lambda row: (
            ROLE_ORDER[str(row["collection_role"])],
            _stable(config.seed, "item-output", row["source_group"]),
            row["norm_uid"],
        )
    )

    metrics = list(bank["metrics"])
    metrics.sort(
        key=lambda row: (
            _stable(config.seed, "bank-order", row["metric_id"]),
            str(row["metric_id"]),
        )
    )
    materialized_bank = {**bank, "metrics": metrics}

    root = config.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    identities_path = root / "identities.jsonl"
    items_path = root / "items.jsonl"
    bank_output = root / "bank.json"
    write_jsonl(identities_path, identity_rows)
    role_identity_paths: dict[str, Path] = {}
    for role in ROLE_ORDER:
        path = root / "identities" / f"{role}.jsonl"
        write_jsonl(
            path,
            [row for row in identity_rows if row["collection_role"] == role],
        )
        role_identity_paths[role] = path
    write_jsonl(items_path, items)
    bank_output.write_text(
        json.dumps(materialized_bank, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    chunk_paths: list[Path] = []
    chunk_order = sorted(
        items,
        key=lambda row: (
            _stable(config.seed, "chunk-order", row["norm_uid"]),
            row["norm_uid"],
        ),
    )
    for start in range(0, len(chunk_order), config.chunk_size):
        path = root / "chunks" / f"part-{start // config.chunk_size:03d}.jsonl"
        write_jsonl(path, chunk_order[start : start + config.chunk_size])
        chunk_paths.append(path)

    outputs = {
        "identities": _ref(identities_path, count=len(identity_rows)),
        "identities_by_role": {
            role: _ref(path, count=len(uid_sets[role]))
            for role, path in role_identity_paths.items()
        },
        "items": _ref(items_path, count=len(items)),
        "bank": _ref(bank_output, count=len(metrics)),
        "chunks": {str(path): sha256_file(path) for path in chunk_paths},
    }
    selection_strata = {
        role: dict(
            sorted(
                Counter(
                    str(row["selection_stratum"])
                    for row in identity_rows
                    if row["collection_role"] == role
                ).items()
            )
        )
        for role in ("train", "dev")
    }
    membership_counts = {
        role: dict(
            sorted(
                Counter(
                    flag
                    for row in identity_rows
                    if row["collection_role"] == role
                    for flag in row.get("inventory_memberships", [])
                ).items()
            )
        )
        for role in ("train", "dev")
    }
    freeze = {
        "schema_version": "silver-match-v3-humor-ce-truth-collection-freeze-v1",
        "status": "FROZEN_BEFORE_ANY_COLLECTION_LABELS_PREDICTIONS_OR_OUTCOMES",
        "task": TASK,
        "selection_seed": config.seed,
        "role_counts": {role: len(uid_sets[role]) for role in ROLE_ORDER},
        "total_count": len(identity_rows),
        "canonical_norm_count": len(norms),
        "canonical_source_group_count": len(groups),
        "bank_metric_count": len(metrics),
        "bank_source_sha256": EXPECTED_BANK_SOURCE_SHA256,
        "selection": {
            "order": [
                "natural_blind_before_loading_candidate_inventories",
                "balanced_dev",
                "train",
            ],
            "train": {
                "candidate_quota_targets": {
                    quota.stratum: quota.count for quota in config.train_quotas
                },
                "selected_by_sampling_stratum": selection_strata["train"],
                "selected_inventory_membership_counts": membership_counts["train"],
            },
            "dev": {
                "balanced": True,
                "candidate_quota_targets": {
                    quota.stratum: quota.count for quota in config.dev_quotas
                },
                "selected_by_sampling_stratum": selection_strata["dev"],
                "selected_inventory_membership_counts": membership_counts["dev"],
            },
            "blind": blind_report,
            "candidate_quota_targets_are_sampling_metadata_not_claimed_outcomes": True,
            "candidate_inventory_memberships_are_not_truth": True,
        },
        "selected_by_role_and_corpus": {
            role: dict(
                sorted(
                    Counter(
                        str(row["corpus"])
                        for row in identity_rows
                        if row["collection_role"] == role
                    ).items()
                )
            )
            for role in ROLE_ORDER
        },
        "source_isolation": {
            "canonical_representation": "task\\x1fcorpus\\x1fkind\\x1fidentity",
            "accepted_aliases": [
                "task\\x1fcorpus\\x1fkind\\x1fidentity",
                "corpus:kind:identity",
                "task:corpus:kind:identity",
            ],
            "cross_role_uid_overlap": 0,
            "cross_role_source_group_overlap": 0,
            "selected_exclusion_uid_overlap": 0,
            "selected_exclusion_source_group_overlap": 0,
            "excluded_canonical_uids": len(excluded_uids),
            "excluded_canonical_source_groups": len(excluded_groups),
        },
        "inputs": {
            "manifest": _ref(manifest_path),
            "canonical_norms_by_corpus": corpus_inputs,
            "bank_source": _ref(bank_path, count=len(metrics)),
            "candidate_inventories": candidate_reports,
            "role_exclusions": exclusion_reports,
        },
        "outputs": outputs,
        "content_contract": {
            "current_or_future_collection_labels_read": False,
            "model_predictions_read": False,
            "mi_or_downstream_outcomes_read": False,
            "candidate_payload_outcome_fields_read": False,
            "candidate_identity_and_declared_sampling_flags_only": True,
            "blind_selected_without_candidate_inventory_access": True,
            "labeler_items_and_chunks_omit_candidate_sampling_strata": True,
            "labeler_items_and_chunks_omit_prior_decisions_confidence_and_metric_ids": True,
            "seed_search_or_performance_tuning_used": False,
        },
        "usage_contract": {
            "train_labels_may_train_ce_after_independent_validation": True,
            "dev_labels_may_select_only_predeclared_ce_variants": True,
            "blind_labels_are_evaluation_only": True,
            "blind_labels_may_not_select_prompts_thresholds_epochs_or_seeds": True,
            "may_not_train_retriever_or_gepa_prompt_from_this_identity_freeze": True,
            "validate_with": "scripts/tools/silver_match_v3/validate_independent_teacher_labels.py",
            "strict_transcript_isolation_required": True,
        },
    }
    freeze_path = root / "FREEZE.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = {
        "schema_version": "silver-match-v3-humor-ce-truth-label-pack-v1",
        "status": "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING",
        "task": TASK,
        "count": len(items),
        "source_groups": len(selected_groups),
        "chunk_size": config.chunk_size,
        "chunk_count": len(chunk_paths),
        "bank_metric_count": len(metrics),
        "bank_source_sha256": EXPECTED_BANK_SOURCE_SHA256,
        "truth_hidden": True,
        "prior_decisions_proposals_predictions_and_outcomes_hidden": True,
        "selection_freeze": _ref(freeze_path),
        "outputs": outputs,
        "transcript_isolation_contract": {
            "chunks_contain_only_canonical_evidence_and_predeclared_role": True,
            "candidate_sampling_strata_absent": True,
            "one_raw_label_payload_required_per_exact_chunk": True,
            "transcript_audit_must_bind_items_bank_validation_chunks_and_raw_labels": True,
        },
    }
    validation_path = root / "validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**validation, "validation_sha256": sha256_file(validation_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--candidate-inventory",
        action="append",
        required=True,
        help=(
            "Repeat [STRATUM=]PATH. With STRATUM=, every Humor UID in the JSONL "
            "is a candidate for that sampling stratum. Without it, rows must carry "
            "candidate_flags, candidate_strata, or strata."
        ),
    )
    parser.add_argument(
        "--exclude-role",
        action="append",
        required=True,
        help="Repeat ROLE=PATH for every prior train/dev/test/select/blind identity file.",
    )
    parser.add_argument(
        "--train-quota",
        action="append",
        required=True,
        help="Repeat STRATUM=COUNT; targets must sum to --train-count.",
    )
    parser.add_argument(
        "--dev-quota",
        action="append",
        required=True,
        help="Repeat balanced STRATUM=COUNT; targets must sum to --dev-count.",
    )
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--dev-count", type=int, default=DEFAULT_DEV_COUNT)
    parser.add_argument("--blind-count", type=int, default=DEFAULT_BLIND_COUNT)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    config = CollectionConfig(
        manifest=Path(args.manifest),
        candidate_inventories=tuple(
            _parse_named_path(value, name_required=False)
            for value in args.candidate_inventory
        ),
        exclusion_roles=tuple(
            _parse_named_path(value, name_required=True) for value in args.exclude_role
        ),
        train_quotas=tuple(_parse_quota(value) for value in args.train_quota),
        dev_quotas=tuple(_parse_quota(value) for value in args.dev_quota),
        output_root=Path(args.output_root),
        train_count=args.train_count,
        dev_count=args.dev_count,
        blind_count=args.blind_count,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    print(json.dumps(freeze_collection(config), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
