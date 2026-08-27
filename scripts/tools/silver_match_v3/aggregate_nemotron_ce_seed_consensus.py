#!/usr/bin/env python3
"""Create a conservative norm-level consensus from two Nemotron CE seeds.

Each input must be a hash-bound ``run_nemotron_ce.py`` score artifact backed
by its original checkpoint and completed training report.  The two checkpoints
keep their independently selected development gates.  A norm is automatically
matched only when both seeds pass their own gate and select the same metric.
Every other norm is retained as a provisional routing decision; this module
does not manufacture human adjudication or abstention subtypes.

The implementation uses a temporary SQLite database so production runs do not
need to hold two full all-task score files in memory.  Candidate order may
differ between seeds, but their exact norm/candidate/source/split universes may
not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import normalize_space, read_jsonl, sha256_file
from .run_nemotron_ce import (
    CLASS_NAMES,
    SCORE_META_SCHEMA,
    SCORE_SCHEMA,
    verify_checkpoint_contract,
)


SEED_MANIFEST_SCHEMA = "silver-match-v3-nemotron-ce-two-seed-manifest-v1"
CONSENSUS_SCHEMA = "silver-match-v3-nemotron-ce-two-seed-consensus-v1"
CONSENSUS_REPORT_SCHEMA = "silver-match-v3-nemotron-ce-two-seed-consensus-report-v1"
ROUTING_CATEGORIES = (
    "MATCH",
    "CE_REJECT_BOTH",
    "FAMILY_SIGNAL",
    "SEED_DISAGREEMENT",
    "BELOW_GATE",
    "NO_CANDIDATES",
)
ROUTING_PRECEDENCE = (
    "NO_CANDIDATES",
    "MATCH",
    "SEED_DISAGREEMENT",
    "FAMILY_SIGNAL",
    "CE_REJECT_BOTH",
    "BELOW_GATE",
)
PROBABILITY_TOLERANCE = 1e-4


@dataclass(frozen=True)
class SeedArtifact:
    seed_id: str
    scores: Path
    scores_sha256: str
    scores_meta: Path
    scores_meta_sha256: str
    checkpoint: Path
    training_report: Path
    training_report_sha256: str


@dataclass(frozen=True)
class NormUniverse:
    path: Path
    sha256: str


@dataclass(frozen=True)
class VerifiedSeed:
    artifact: SeedArtifact
    score_meta: Mapping[str, Any]
    checkpoint_contract: Mapping[str, Any]
    score_threshold: float
    margin_threshold: float


def _json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _query_identity_sha256(
    db: sqlite3.Connection, query: str
) -> str:
    """Hash an ordered SQLite identity stream without materializing it."""

    digest = hashlib.sha256()
    for row in db.execute(query):
        encoded = json.dumps(
            list(row), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    digest = normalize_space(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field} must be a SHA256 hex digest")
    return digest


def _resolve_manifest_path(raw: Any, manifest_path: Path) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_seed_manifest(
    path: Path, expected_sha256: str
) -> tuple[tuple[SeedArtifact, SeedArtifact], NormUniverse | None, dict[str, Any]]:
    """Load a content-addressed two-seed manifest.

    Relative artifact paths are resolved against the manifest directory, which
    makes a frozen bundle relocatable without weakening any content checks.
    """

    path = path.resolve()
    expected_sha256 = _require_sha256(expected_sha256, "seed_manifest_sha256")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha256:
        raise ValueError("seed manifest SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SEED_MANIFEST_SCHEMA:
        raise ValueError("unknown two-seed manifest schema")
    raw_seeds = payload.get("seeds")
    if not isinstance(raw_seeds, list) or len(raw_seeds) != 2:
        raise ValueError("two-seed manifest must contain exactly two seeds")
    seeds: list[SeedArtifact] = []
    for index, row in enumerate(raw_seeds):
        if not isinstance(row, Mapping):
            raise ValueError(f"seed manifest row {index} is not an object")
        seed_id = normalize_space(row.get("seed_id"))
        if not seed_id:
            raise ValueError(f"seed manifest row {index} lacks seed_id")
        scores = _resolve_manifest_path(row.get("scores"), path)
        raw_meta = row.get("scores_meta") or str(scores) + ".meta.json"
        seed = SeedArtifact(
            seed_id=seed_id,
            scores=scores,
            scores_sha256=_require_sha256(
                row.get("scores_sha256"), f"seeds[{index}].scores_sha256"
            ),
            scores_meta=_resolve_manifest_path(raw_meta, path),
            scores_meta_sha256=_require_sha256(
                row.get("scores_meta_sha256"),
                f"seeds[{index}].scores_meta_sha256",
            ),
            checkpoint=_resolve_manifest_path(row.get("checkpoint"), path),
            training_report=_resolve_manifest_path(row.get("training_report"), path),
            training_report_sha256=_require_sha256(
                row.get("training_report_sha256"),
                f"seeds[{index}].training_report_sha256",
            ),
        )
        seeds.append(seed)
    if seeds[0].seed_id == seeds[1].seed_id:
        raise ValueError("two seed IDs must be distinct")

    universe = None
    raw_universe = payload.get("norm_universe")
    if raw_universe is not None:
        if not isinstance(raw_universe, Mapping):
            raise ValueError("norm_universe must be an object")
        universe = NormUniverse(
            path=_resolve_manifest_path(raw_universe.get("path"), path),
            sha256=_require_sha256(
                raw_universe.get("sha256"), "norm_universe.sha256"
            ),
        )
    provenance = {
        "path": str(path),
        "sha256": observed_sha,
        "schema_version": SEED_MANIFEST_SCHEMA,
    }
    return (seeds[0], seeds[1]), universe, provenance


def _predicted_relation(probabilities: Mapping[str, float]) -> str:
    # CLASS_NAMES order is the production argmax tie-break.
    return max(CLASS_NAMES, key=lambda name: probabilities[name])


def _validate_probability_row(row: Mapping[str, Any], source: str) -> dict[str, float]:
    raw = row.get("probabilities")
    if not isinstance(raw, Mapping) or set(raw) != set(CLASS_NAMES):
        raise ValueError(f"probability schema mismatch in {source}")
    values: dict[str, float] = {}
    for name in CLASS_NAMES:
        value = float(raw[name])
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"invalid {name} probability in {source}")
        values[name] = value
    if not math.isclose(
        sum(values.values()), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE
    ):
        raise ValueError(f"probabilities do not sum to one in {source}")
    predicted = normalize_space(row.get("predicted_relation")).upper()
    if predicted not in CLASS_NAMES:
        raise ValueError(f"invalid predicted_relation in {source}")
    if predicted != _predicted_relation(values):
        raise ValueError(f"predicted_relation/probability mismatch in {source}")
    return values


def _checkpoint_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Drop relocatable paths while retaining every content/decision field."""

    keys = (
        "training_report_sha256",
        "checkpoint_metadata_sha256",
        "head_sha256",
        "adapter_tree_sha256",
        "artifact_sha256",
        "labels",
        "score_threshold",
        "top_margin_threshold",
        "max_sequence_length",
        "threshold_provenance",
    )
    return {key: contract.get(key) for key in keys}


def verify_seed_artifact(artifact: SeedArtifact) -> VerifiedSeed:
    """Verify score, metadata, selected checkpoint, and training-report bytes."""

    if sha256_file(artifact.scores) != artifact.scores_sha256:
        raise ValueError(f"score SHA256 mismatch for seed {artifact.seed_id}")
    if sha256_file(artifact.scores_meta) != artifact.scores_meta_sha256:
        raise ValueError(f"score metadata SHA256 mismatch for seed {artifact.seed_id}")
    meta = json.loads(artifact.scores_meta.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCORE_META_SCHEMA:
        raise ValueError(f"unknown score metadata schema for seed {artifact.seed_id}")
    if meta.get("output_sha256") != artifact.scores_sha256:
        raise ValueError(f"score/meta output hash mismatch for seed {artifact.seed_id}")
    if tuple(meta.get("labels") or ()) != CLASS_NAMES:
        raise ValueError(f"score probability label order mismatch for seed {artifact.seed_id}")
    if int(meta.get("num_shards", -1)) != 1:
        raise ValueError(f"seed {artifact.seed_id} is not a complete merged score artifact")
    raw_contract = meta.get("checkpoint_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError(f"score metadata lacks checkpoint contract for {artifact.seed_id}")
    if raw_contract.get("threshold_provenance") != "checkpoint.dev":
        raise ValueError(f"seed {artifact.seed_id} does not carry a development-frozen gate")
    if raw_contract.get("training_report_sha256") != artifact.training_report_sha256:
        raise ValueError(f"training report identity drift for seed {artifact.seed_id}")
    verified = verify_checkpoint_contract(
        artifact.checkpoint,
        artifact.training_report,
        artifact.training_report_sha256,
    )
    if _checkpoint_identity(raw_contract) != _checkpoint_identity(verified):
        raise ValueError(f"checkpoint/meta contract drift for seed {artifact.seed_id}")
    return VerifiedSeed(
        artifact=artifact,
        score_meta=meta,
        checkpoint_contract=verified,
        score_threshold=float(verified["score_threshold"]),
        margin_threshold=float(verified["top_margin_threshold"]),
    )


def _source_dimensions(
    source_group: str, task: Any = None, corpus: Any = None
) -> tuple[str, str]:
    task_value = normalize_space(task)
    corpus_value = normalize_space(corpus)
    # ``run_nemotron_ce`` normalizes whitespace, so historical unit-separator
    # source groups may arrive either intact or as ``task corpus source id``.
    pieces = (
        source_group.split("\x1f")
        if "\x1f" in source_group
        else source_group.split()
    )
    if not task_value and pieces:
        task_value = normalize_space(pieces[0])
    if not corpus_value and len(pieces) >= 2:
        corpus_value = normalize_space(pieces[1])
    return task_value or "<unknown>", corpus_value or "<unknown>"


def _create_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute(
        """
        CREATE TABLE norms (
            norm_uid TEXT PRIMARY KEY,
            source_group TEXT NOT NULL,
            split TEXT NOT NULL,
            task TEXT NOT NULL,
            corpus TEXT NOT NULL,
            from_universe INTEGER NOT NULL DEFAULT 0
        ) WITHOUT ROWID
        """
    )
    db.execute(
        """
        CREATE TABLE candidates (
            norm_uid TEXT NOT NULL,
            metric_id TEXT NOT NULL,
            source_group TEXT NOT NULL,
            split TEXT NOT NULL,
            a_exact REAL NOT NULL,
            a_family REAL NOT NULL,
            a_reject REAL NOT NULL,
            a_predicted TEXT NOT NULL,
            a_gold TEXT,
            b_exact REAL,
            b_family REAL,
            b_reject REAL,
            b_predicted TEXT,
            b_gold TEXT,
            PRIMARY KEY (norm_uid, metric_id)
        ) WITHOUT ROWID
        """
    )
    return db


def _norm_contract(row: Mapping[str, Any], source: str) -> tuple[str, str, str, str, str]:
    uid = normalize_space(row.get("norm_uid") or row.get("uid"))
    raw_source_group = row.get("source_group") or row.get("split_group")
    source_group = normalize_space(raw_source_group)
    split = normalize_space(row.get("split") or row.get("predeclared_split"))
    if not uid or not source_group or not split:
        raise ValueError(f"norm/source/split contract is incomplete in {source}")
    task, corpus = _source_dimensions(
        str(raw_source_group or source_group),
        row.get("task"),
        row.get("corpus") or row.get("source_corpus") or row.get("corpus_name"),
    )
    return uid, source_group, split, task, corpus


def _upsert_norm(
    db: sqlite3.Connection,
    contract: tuple[str, str, str, str, str],
    *,
    from_universe: bool,
) -> None:
    uid, source_group, split, task, corpus = contract
    existing = db.execute(
        "SELECT source_group, split, task, corpus, from_universe FROM norms WHERE norm_uid=?",
        (uid,),
    ).fetchone()
    if existing is None:
        db.execute(
            "INSERT INTO norms VALUES (?, ?, ?, ?, ?, ?)",
            (uid, source_group, split, task, corpus, int(from_universe)),
        )
        return
    old_source, old_split, old_task, old_corpus, old_universe = existing
    if (old_source, old_split) != (source_group, split):
        raise ValueError(f"norm crosses source_group/split: {uid}")
    # Explicit universe metadata can replace source-group inference, but two
    # explicit conflicting values may not be silently merged.
    chosen_task = task if from_universe and task != "<unknown>" else old_task
    chosen_corpus = corpus if from_universe and corpus != "<unknown>" else old_corpus
    if from_universe and old_universe:
        if old_task != "<unknown>" and task != "<unknown>" and old_task != task:
            raise ValueError(f"norm crosses task values in universe: {uid}")
        if old_corpus != "<unknown>" and corpus != "<unknown>" and old_corpus != corpus:
            raise ValueError(f"norm crosses corpus values in universe: {uid}")
    db.execute(
        "UPDATE norms SET task=?, corpus=?, from_universe=? WHERE norm_uid=?",
        (chosen_task, chosen_corpus, int(bool(old_universe or from_universe)), uid),
    )


def _load_universe(db: sqlite3.Connection, universe: NormUniverse) -> int:
    if sha256_file(universe.path) != universe.sha256:
        raise ValueError("norm universe SHA256 mismatch")
    count_before = db.execute("SELECT COUNT(*) FROM norms").fetchone()[0]
    with db:
        for line_no, row in enumerate(read_jsonl(universe.path), 1):
            _upsert_norm(
                db,
                _norm_contract(row, f"{universe.path}:{line_no}"),
                from_universe=True,
            )
    count_after = db.execute("SELECT COUNT(*) FROM norms").fetchone()[0]
    if count_after == count_before:
        raise ValueError("norm universe is empty")
    return int(count_after - count_before)


def _load_seed_a(db: sqlite3.Connection, seed: VerifiedSeed, universe_given: bool) -> tuple[int, int]:
    rows = norms = 0
    seen_norms: set[str] = set()
    with db:
        for line_no, row in enumerate(read_jsonl(seed.artifact.scores), 1):
            source = f"{seed.artifact.scores}:{line_no}"
            if row.get("schema_version") != SCORE_SCHEMA:
                raise ValueError(f"score schema mismatch in {source}")
            contract = _norm_contract(row, source)
            uid, source_group, split, _, _ = contract
            if universe_given:
                existing = db.execute(
                    "SELECT source_group, split FROM norms WHERE norm_uid=?", (uid,)
                ).fetchone()
                if existing is None:
                    raise ValueError(f"seed score norm lies outside frozen universe: {uid}")
                if tuple(existing) != (source_group, split):
                    raise ValueError(f"score/universe source or split drift: {uid}")
            else:
                _upsert_norm(db, contract, from_universe=False)
            metric_id = normalize_space(row.get("metric_id") or row.get("candidate_metric_id"))
            if not metric_id:
                raise ValueError(f"score row lacks metric_id in {source}")
            probability = _validate_probability_row(row, source)
            predicted = normalize_space(row.get("predicted_relation")).upper()
            gold = row.get("gold_relation")
            gold_value = normalize_space(gold).upper() if gold is not None else None
            try:
                db.execute(
                    """
                    INSERT INTO candidates (
                        norm_uid, metric_id, source_group, split,
                        a_exact, a_family, a_reject, a_predicted, a_gold
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uid,
                        metric_id,
                        source_group,
                        split,
                        probability["EXACT"],
                        probability["FAMILY"],
                        probability["REJECT"],
                        predicted,
                        gold_value,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"duplicate seed-A candidate pair: {(uid, metric_id)}") from exc
            rows += 1
            if uid not in seen_norms:
                seen_norms.add(uid)
                norms += 1
    if rows != int(seed.score_meta.get("row_count", -1)):
        raise ValueError(f"seed {seed.artifact.seed_id} row_count metadata drift")
    if norms != int(seed.score_meta.get("norm_group_count", -1)):
        raise ValueError(f"seed {seed.artifact.seed_id} norm_group_count metadata drift")
    return rows, norms


def _load_seed_b(db: sqlite3.Connection, seed: VerifiedSeed) -> tuple[int, int]:
    rows = norms = 0
    seen_norms: set[str] = set()
    with db:
        for line_no, row in enumerate(read_jsonl(seed.artifact.scores), 1):
            source = f"{seed.artifact.scores}:{line_no}"
            if row.get("schema_version") != SCORE_SCHEMA:
                raise ValueError(f"score schema mismatch in {source}")
            uid, source_group, split, _, _ = _norm_contract(row, source)
            metric_id = normalize_space(row.get("metric_id") or row.get("candidate_metric_id"))
            if not metric_id:
                raise ValueError(f"score row lacks metric_id in {source}")
            probability = _validate_probability_row(row, source)
            predicted = normalize_space(row.get("predicted_relation")).upper()
            gold = row.get("gold_relation")
            gold_value = normalize_space(gold).upper() if gold is not None else None
            existing = db.execute(
                """
                SELECT source_group, split, a_gold, b_exact
                FROM candidates WHERE norm_uid=? AND metric_id=?
                """,
                (uid, metric_id),
            ).fetchone()
            if existing is None:
                raise ValueError(
                    f"seed candidate universe drift; seed B has extra pair: {(uid, metric_id)}"
                )
            if (existing[0], existing[1]) != (source_group, split):
                raise ValueError(f"seed source/split universe drift: {(uid, metric_id)}")
            if existing[3] is not None:
                raise ValueError(f"duplicate seed-B candidate pair: {(uid, metric_id)}")
            if existing[2] is not None and gold_value is not None and existing[2] != gold_value:
                raise ValueError(f"gold relation drift between seeds: {(uid, metric_id)}")
            db.execute(
                """
                UPDATE candidates SET
                    b_exact=?, b_family=?, b_reject=?, b_predicted=?, b_gold=?
                WHERE norm_uid=? AND metric_id=?
                """,
                (
                    probability["EXACT"],
                    probability["FAMILY"],
                    probability["REJECT"],
                    predicted,
                    gold_value,
                    uid,
                    metric_id,
                ),
            )
            rows += 1
            if uid not in seen_norms:
                seen_norms.add(uid)
                norms += 1
    if rows != int(seed.score_meta.get("row_count", -1)):
        raise ValueError(f"seed {seed.artifact.seed_id} row_count metadata drift")
    if norms != int(seed.score_meta.get("norm_group_count", -1)):
        raise ValueError(f"seed {seed.artifact.seed_id} norm_group_count metadata drift")
    missing = db.execute("SELECT COUNT(*) FROM candidates WHERE b_exact IS NULL").fetchone()[0]
    if missing:
        raise ValueError(f"seed candidate universe drift; seed B misses {missing} pairs")
    return rows, norms


def _candidate_rows(db: sqlite3.Connection, uid: str) -> list[sqlite3.Row]:
    return list(
        db.execute(
            """
            SELECT metric_id,
                   a_exact, a_family, a_reject, a_predicted, a_gold,
                   b_exact, b_family, b_reject, b_predicted, b_gold
            FROM candidates WHERE norm_uid=? ORDER BY metric_id
            """,
            (uid,),
        )
    )


def _seed_state(
    candidates: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    if not candidates:
        return {
            "top_metric_id": None,
            "top_predicted_relation": None,
            "top_exact_probability": None,
            "second_exact_probability": None,
            "top_exact_margin": None,
            "score_threshold": score_threshold,
            "top_margin_threshold": margin_threshold,
            "passes_frozen_gate": False,
            "has_family_argmax_candidate": False,
            "all_candidates_reject": False,
        }
    ranked = sorted(candidates, key=lambda row: (-float(row[f"{prefix}_exact"]), row["metric_id"]))
    top = ranked[0]
    top_score = float(top[f"{prefix}_exact"])
    second = float(ranked[1][f"{prefix}_exact"]) if len(ranked) > 1 else 0.0
    predicted = str(top[f"{prefix}_predicted"])
    passes = (
        predicted == "EXACT"
        and top_score >= score_threshold
        and top_score - second >= margin_threshold
    )
    return {
        "top_metric_id": str(top["metric_id"]),
        "top_predicted_relation": predicted,
        "top_exact_probability": top_score,
        "second_exact_probability": second,
        "top_exact_margin": top_score - second,
        "score_threshold": score_threshold,
        "top_margin_threshold": margin_threshold,
        "passes_frozen_gate": passes,
        "has_family_argmax_candidate": any(
            str(row[f"{prefix}_predicted"]) == "FAMILY" for row in candidates
        ),
        "all_candidates_reject": all(
            str(row[f"{prefix}_predicted"]) == "REJECT" for row in candidates
        ),
    }


def _route(seed_a: Mapping[str, Any], seed_b: Mapping[str, Any]) -> str:
    if seed_a["top_metric_id"] is None:
        return "NO_CANDIDATES"
    passed_a = bool(seed_a["passes_frozen_gate"])
    passed_b = bool(seed_b["passes_frozen_gate"])
    same_top = seed_a["top_metric_id"] == seed_b["top_metric_id"]
    if passed_a and passed_b and same_top:
        return "MATCH"
    if (passed_a or passed_b) and not same_top:
        return "SEED_DISAGREEMENT"
    if (
        seed_a["top_predicted_relation"] == "FAMILY"
        or seed_b["top_predicted_relation"] == "FAMILY"
    ):
        return "FAMILY_SIGNAL"
    if (
        seed_a["top_predicted_relation"] == "REJECT"
        and seed_b["top_predicted_relation"] == "REJECT"
    ):
        return "CE_REJECT_BOTH"
    return "BELOW_GATE"


def _counter_report(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    categories = {name: int(counter.get(name, 0)) for name in ROUTING_CATEGORIES}
    match_count = categories["MATCH"]
    abstain_count = total - match_count
    disagreement = categories["SEED_DISAGREEMENT"]
    noise = categories["CE_REJECT_BOTH"]
    return {
        "norm_count": total,
        "automatic_match_count": match_count,
        "automatic_match_rate": match_count / total if total else 0.0,
        "provisional_abstention_count": abstain_count,
        "provisional_abstention_rate": abstain_count / total if total else 0.0,
        "seed_disagreement_routing_count": disagreement,
        "seed_disagreement_routing_rate": disagreement / total if total else 0.0,
        "ce_reject_both_noise_routing_count": noise,
        "ce_reject_both_noise_routing_rate": noise / total if total else 0.0,
        "routing_category_counts": categories,
        "routing_category_rates": {
            name: count / total if total else 0.0 for name, count in categories.items()
        },
    }


def _probability_payload(row: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    payload = {
        "predicted_relation": str(row[f"{prefix}_predicted"]),
        "probabilities": {
            "EXACT": float(row[f"{prefix}_exact"]),
            "FAMILY": float(row[f"{prefix}_family"]),
            "REJECT": float(row[f"{prefix}_reject"]),
        },
    }
    gold = row[f"{prefix}_gold"]
    if gold is not None:
        payload["gold_relation"] = str(gold)
    return payload


def aggregate_seed_consensus(
    seed_a_artifact: SeedArtifact,
    seed_b_artifact: SeedArtifact,
    *,
    output: Path,
    report_output: Path,
    norm_universe: NormUniverse | None = None,
    manifest_provenance: Mapping[str, Any] | None = None,
    sqlite_path: Path | None = None,
) -> dict[str, Any]:
    """Verify, join, gate, and write a complete two-seed consensus release."""

    output = output.resolve()
    report_output = report_output.resolve()
    if output.exists() or report_output.exists():
        raise FileExistsError("refusing to overwrite consensus output/report")
    if output == report_output:
        raise ValueError("consensus JSONL and report paths must differ")
    seed_a = verify_seed_artifact(seed_a_artifact)
    seed_b = verify_seed_artifact(seed_b_artifact)
    if seed_a.artifact.seed_id == seed_b.artifact.seed_id:
        raise ValueError("two seed IDs must be distinct")
    checkpoint_fingerprint_a = (
        seed_a.checkpoint_contract.get("checkpoint_metadata_sha256"),
        seed_a.checkpoint_contract.get("head_sha256"),
        seed_a.checkpoint_contract.get("adapter_tree_sha256"),
    )
    checkpoint_fingerprint_b = (
        seed_b.checkpoint_contract.get("checkpoint_metadata_sha256"),
        seed_b.checkpoint_contract.get("head_sha256"),
        seed_b.checkpoint_contract.get("adapter_tree_sha256"),
    )
    if checkpoint_fingerprint_a == checkpoint_fingerprint_b:
        raise ValueError("two seed IDs resolve to the same checkpoint content")

    owned_temp = sqlite_path is None
    if sqlite_path is None:
        handle = tempfile.NamedTemporaryFile(prefix="nemotron-ce-consensus-", suffix=".sqlite3", delete=False)
        handle.close()
        sqlite_path = Path(handle.name)
    else:
        sqlite_path = sqlite_path.resolve()
        if sqlite_path.exists():
            raise FileExistsError(f"refusing to overwrite SQLite work file: {sqlite_path}")
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    db = _create_db(sqlite_path)
    db.row_factory = sqlite3.Row
    try:
        universe_count = None
        if norm_universe is not None:
            universe_count = _load_universe(db, norm_universe)
        rows_a, norms_a = _load_seed_a(db, seed_a, norm_universe is not None)
        rows_b, norms_b = _load_seed_b(db, seed_b)
        if (rows_a, norms_a) != (rows_b, norms_b):
            raise ValueError("seed norm/candidate universe counts differ")
        if norm_universe is not None:
            outside = db.execute(
                "SELECT COUNT(*) FROM norms WHERE from_universe=0"
            ).fetchone()[0]
            if outside:
                raise ValueError(f"scores contain {outside} norms outside frozen universe")

        overall: Counter[str] = Counter()
        by_split: dict[str, Counter[str]] = defaultdict(Counter)
        by_corpus: dict[str, Counter[str]] = defaultdict(Counter)
        any_disagreement = gate_disagreement = top_metric_disagreement = 0
        norm_count = int(db.execute("SELECT COUNT(*) FROM norms").fetchone()[0])
        candidate_count = int(db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
        source_group_count = int(
            db.execute("SELECT COUNT(DISTINCT source_group) FROM norms").fetchone()[0]
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            for norm in db.execute(
                "SELECT norm_uid, source_group, split, task, corpus FROM norms ORDER BY norm_uid"
            ):
                uid = str(norm["norm_uid"])
                raw_candidates = _candidate_rows(db, uid)
                candidates = [dict(row) for row in raw_candidates]
                state_a = _seed_state(
                    candidates,
                    prefix="a",
                    score_threshold=seed_a.score_threshold,
                    margin_threshold=seed_a.margin_threshold,
                )
                state_b = _seed_state(
                    candidates,
                    prefix="b",
                    score_threshold=seed_b.score_threshold,
                    margin_threshold=seed_b.margin_threshold,
                )
                category = _route(state_a, state_b)
                automatic = category == "MATCH"
                metric_id = state_a["top_metric_id"] if automatic else None
                top_diff = state_a["top_metric_id"] != state_b["top_metric_id"]
                gate_diff = state_a["passes_frozen_gate"] != state_b["passes_frozen_gate"]
                relation_diff = (
                    state_a["top_predicted_relation"] != state_b["top_predicted_relation"]
                )
                top_metric_disagreement += int(top_diff)
                gate_disagreement += int(gate_diff)
                any_disagreement += int(top_diff or gate_diff or relation_diff)
                overall[category] += 1
                by_split[str(norm["split"])][category] += 1
                by_corpus[str(norm["corpus"])][category] += 1
                record = {
                    "schema_version": CONSENSUS_SCHEMA,
                    "norm_uid": uid,
                    "task": str(norm["task"]),
                    "corpus": str(norm["corpus"]),
                    "source_group": str(norm["source_group"]),
                    "split": str(norm["split"]),
                    "decision": "MATCH" if automatic else "ROUTE_TO_ADJUDICATION",
                    "routing_category": category,
                    "automatic_match": automatic,
                    "metric_id": metric_id,
                    "candidate_count": len(candidates),
                    "seed_decisions": {
                        seed_a.artifact.seed_id: state_a,
                        seed_b.artifact.seed_id: state_b,
                    },
                    "candidates": [
                        {
                            "metric_id": str(row["metric_id"]),
                            seed_a.artifact.seed_id: _probability_payload(row, "a"),
                            seed_b.artifact.seed_id: _probability_payload(row, "b"),
                        }
                        for row in candidates
                    ],
                    "provisional_routing_only": not automatic,
                    "human_abstention_subtype_assigned": False,
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        overall_report = _counter_report(overall)
        overall_report.update(
            {
                "any_seed_disagreement_count": any_disagreement,
                "any_seed_disagreement_rate": (
                    any_disagreement / norm_count if norm_count else 0.0
                ),
                "top_metric_disagreement_count": top_metric_disagreement,
                "top_metric_disagreement_rate": (
                    top_metric_disagreement / norm_count if norm_count else 0.0
                ),
                "gate_pass_disagreement_count": gate_disagreement,
                "gate_pass_disagreement_rate": (
                    gate_disagreement / norm_count if norm_count else 0.0
                ),
            }
        )
        norm_universe_sha = _query_identity_sha256(
            db, "SELECT norm_uid, source_group, split FROM norms ORDER BY norm_uid"
        )
        candidate_universe_sha = _query_identity_sha256(
            db,
            "SELECT norm_uid, metric_id, source_group, split FROM candidates "
            "ORDER BY norm_uid, metric_id",
        )
        report = {
            "schema_version": CONSENSUS_REPORT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "COMPLETE",
            "output": str(output),
            "output_sha256": sha256_file(output),
            "norm_count": norm_count,
            "candidate_pair_count": candidate_count,
            "source_group_count": source_group_count,
            "norm_universe_sha256": norm_universe_sha,
            "candidate_universe_sha256": candidate_universe_sha,
            "external_norm_universe": (
                {
                    "path": str(norm_universe.path),
                    "sha256": norm_universe.sha256,
                    "norm_count": universe_count,
                }
                if norm_universe is not None
                else None
            ),
            "zero_candidate_norms_observable": norm_universe is not None,
            "manifest_provenance": dict(manifest_provenance or {}),
            "seeds": [
                {
                    "seed_id": seed.artifact.seed_id,
                    "scores": str(seed.artifact.scores),
                    "scores_sha256": seed.artifact.scores_sha256,
                    "scores_meta": str(seed.artifact.scores_meta),
                    "scores_meta_sha256": seed.artifact.scores_meta_sha256,
                    "checkpoint": str(seed.artifact.checkpoint),
                    "training_report": str(seed.artifact.training_report),
                    "training_report_sha256": seed.artifact.training_report_sha256,
                    "checkpoint_contract": dict(seed.checkpoint_contract),
                    "frozen_gate": {
                        "score_threshold": seed.score_threshold,
                        "top_margin_threshold": seed.margin_threshold,
                        "provenance": "checkpoint.dev",
                    },
                }
                for seed in (seed_a, seed_b)
            ],
            "consensus_policy": {
                "automatic_match": (
                    "both seeds independently pass checkpoint.dev gates and select "
                    "the same top metric"
                ),
                "routing_precedence": list(ROUTING_PRECEDENCE),
                "nonmatch_decision": "ROUTE_TO_ADJUDICATION",
                "human_abstention_subtypes_created": False,
                "ce_reject_both_is_model_only_noise_routing_not_human_noise_truth": True,
            },
            "metrics": {
                "overall": overall_report,
                "by_split": {
                    key: _counter_report(value) for key, value in sorted(by_split.items())
                },
                "by_corpus": {
                    key: _counter_report(value) for key, value in sorted(by_corpus.items())
                },
            },
            "validation": {
                "score_probability_schema": list(CLASS_NAMES),
                "seed_norm_candidate_source_split_universes_identical": True,
                "all_score_and_metadata_hashes_verified": True,
                "all_checkpoint_artifact_hashes_verified_against_training_reports": True,
                "all_thresholds_from_checkpoint_dev": True,
                "test_threshold_tuning_performed": False,
                "all_norms_preserved": True,
            },
        }
        _json_exclusive(report_output, report)
        return report
    finally:
        db.close()
        if owned_temp:
            sqlite_path.unlink(missing_ok=True)


def _direct_seed(args: argparse.Namespace, prefix: str) -> SeedArtifact:
    def value(name: str) -> Any:
        return getattr(args, f"{prefix}_{name}")

    scores = Path(value("scores")).resolve()
    raw_meta = value("scores_meta") or str(scores) + ".meta.json"
    return SeedArtifact(
        seed_id=normalize_space(value("id")),
        scores=scores,
        scores_sha256=_require_sha256(value("scores_sha256"), f"{prefix}_scores_sha256"),
        scores_meta=Path(raw_meta).resolve(),
        scores_meta_sha256=_require_sha256(
            value("scores_meta_sha256"), f"{prefix}_scores_meta_sha256"
        ),
        checkpoint=Path(value("checkpoint")).resolve(),
        training_report=Path(value("training_report")).resolve(),
        training_report_sha256=_require_sha256(
            value("training_report_sha256"), f"{prefix}_training_report_sha256"
        ),
    )


def _add_seed_args(parser: argparse.ArgumentParser, prefix: str) -> None:
    flag = prefix.replace("_", "-")
    parser.add_argument(f"--{flag}-id")
    parser.add_argument(f"--{flag}-scores")
    parser.add_argument(f"--{flag}-scores-sha256")
    parser.add_argument(f"--{flag}-scores-meta")
    parser.add_argument(f"--{flag}-scores-meta-sha256")
    parser.add_argument(f"--{flag}-checkpoint")
    parser.add_argument(f"--{flag}-training-report")
    parser.add_argument(f"--{flag}-training-report-sha256")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-manifest")
    parser.add_argument("--seed-manifest-sha256")
    _add_seed_args(parser, "seed_a")
    _add_seed_args(parser, "seed_b")
    parser.add_argument("--norm-universe")
    parser.add_argument("--norm-universe-sha256")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--sqlite-path")
    args = parser.parse_args(argv)
    if args.seed_manifest:
        if not args.seed_manifest_sha256:
            parser.error("--seed-manifest-sha256 is required with --seed-manifest")
        direct = [
            getattr(args, f"seed_{letter}_{field}")
            for letter in ("a", "b")
            for field in (
                "id",
                "scores",
                "scores_sha256",
                "scores_meta",
                "scores_meta_sha256",
                "checkpoint",
                "training_report",
                "training_report_sha256",
            )
        ]
        if any(direct):
            parser.error("direct seed arguments cannot be combined with --seed-manifest")
    else:
        required = (
            "id",
            "scores",
            "scores_sha256",
            "scores_meta_sha256",
            "checkpoint",
            "training_report",
            "training_report_sha256",
        )
        missing = [
            f"--seed-{letter}-{field.replace('_', '-')}"
            for letter in ("a", "b")
            for field in required
            if not getattr(args, f"seed_{letter}_{field}")
        ]
        if missing:
            parser.error("missing direct seed arguments: " + ", ".join(missing))
    if bool(args.norm_universe) != bool(args.norm_universe_sha256):
        parser.error("--norm-universe and --norm-universe-sha256 must be provided together")
    return args


def main() -> None:
    args = parse_args()
    manifest_provenance = None
    if args.seed_manifest:
        seeds, universe, manifest_provenance = load_seed_manifest(
            Path(args.seed_manifest), args.seed_manifest_sha256
        )
    else:
        seeds = (_direct_seed(args, "seed_a"), _direct_seed(args, "seed_b"))
        universe = None
    if args.norm_universe:
        if universe is not None:
            raise ValueError("norm universe provided both in manifest and command line")
        universe = NormUniverse(
            Path(args.norm_universe).resolve(),
            _require_sha256(args.norm_universe_sha256, "norm_universe_sha256"),
        )
    result = aggregate_seed_consensus(
        seeds[0],
        seeds[1],
        output=Path(args.output),
        report_output=Path(args.report_output),
        norm_universe=universe,
        manifest_provenance=manifest_provenance,
        sqlite_path=Path(args.sqlite_path) if args.sqlite_path else None,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
