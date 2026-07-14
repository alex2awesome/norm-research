#!/usr/bin/env python3
"""Build an immutable, release-independent CR-3 GPU evidence store.

Historical prompts are reusable as achieved-value candidates even when their original
stream cannot support a new confirmation claim. Content-addressed executor signatures
and MCQ choice rows are copied only after semantic validation. The store never promotes
imported data to a fresh audit role.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Iterable

import numpy as np


SCHEMA_VERSION = "cr3-evidence-store-v1"
CANDIDATE_SCHEMA = "cr3-historical-candidate-v1"
CELL_STORE_SCHEMA = "cr3-evidence-cells-v14"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def evidence_cell_key(kind: str, fields: dict) -> str:
    """Content key for one immutable v14 induction or rule/probe cell."""
    return _payload_sha256({
        "schema": CELL_STORE_SCHEMA,
        "kind": str(kind),
        "fields": dict(fields),
    })


class EvidenceCellStore:
    """Append-only cell cache shared by v14 inductions and executor scoring.

    A held-out vector is intentionally not a cache unit: rule execution is stored one
    probe at a time so expanding H never invalidates already-scored evidence.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=120.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS evidence_cells ("
            "cache_key TEXT PRIMARY KEY, kind TEXT NOT NULL, payload_json TEXT NOT NULL, "
            "payload_sha256 TEXT NOT NULL) WITHOUT ROWID"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EvidenceCellStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @staticmethod
    def induction_key(
        *, template_sha256: str, decoder_revision: str, arm: str,
        panel_sha256: str, state: int,
    ) -> str:
        return evidence_cell_key("induction", {
            "template_sha256": str(template_sha256),
            "decoder_revision": str(decoder_revision),
            "arm": str(arm),
            "panel_sha256": str(panel_sha256),
            "state": int(state),
        })

    @staticmethod
    def rule_probe_key(
        *, rule_sha256: str, probe_sha256: str, executor_revision: str,
        readout_id: str, execution_template_sha256: str,
    ) -> str:
        return evidence_cell_key("rule_probe", {
            "rule_sha256": str(rule_sha256),
            "probe_sha256": str(probe_sha256),
            "executor_revision": str(executor_revision),
            "readout_id": str(readout_id),
            "execution_template_sha256": str(execution_template_sha256),
        })

    def get(self, key: str) -> dict | None:
        row = self.connection.execute(
            "SELECT payload_json, payload_sha256 FROM evidence_cells WHERE cache_key=?",
            (str(key),),
        ).fetchone()
        if row is None:
            return None
        payload_json, observed_sha = map(str, row)
        if hashlib.sha256(payload_json.encode("utf-8")).hexdigest() != observed_sha:
            raise RuntimeError(f"evidence cell checksum mismatch for {key}")
        payload = json.loads(payload_json)
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        if canonical != payload_json:
            raise RuntimeError(f"evidence cell is not canonical for {key}")
        return payload

    def put(self, key: str, kind: str, payload: dict) -> dict:
        payload_dict = dict(payload)
        payload_json = json.dumps(
            payload_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        self.connection.execute(
            "INSERT OR IGNORE INTO evidence_cells"
            "(cache_key,kind,payload_json,payload_sha256) VALUES (?,?,?,?)",
            (str(key), str(kind), payload_json, digest),
        )
        self.connection.commit()
        stored = self.get(key)
        if stored != payload_dict:
            raise RuntimeError(f"repeated evidence key {key} disagrees for {kind}")
        return stored

    def count(self, kind: str | None = None) -> int:
        if kind is None:
            row = self.connection.execute("SELECT COUNT(*) FROM evidence_cells").fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM evidence_cells WHERE kind=?", (str(kind),)
            ).fetchone()
        return int(row[0])


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as sink:
        json.dump(payload, sink, sort_keys=True, indent=2)
        sink.write("\n")
        sink.flush()
        os.fsync(sink.fileno())
    _fsync_directory(path.parent)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        sink.flush()
        os.fsync(sink.fileno())
    _fsync_directory(path.parent)


def _signature_entry(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    required = {"signature", "namespace_sha256", "criterion", "criterion_sha256"}
    if not required.issubset(z.files):
        raise ValueError(f"signature cache entry lacks fields: {path}")
    namespace = str(z["namespace_sha256"])
    criterion = str(z["criterion"])
    criterion_sha = str(z["criterion_sha256"])
    signature = np.asarray(z["signature"], dtype=np.float64)
    expected_criterion_sha = hashlib.sha256(criterion.encode("utf-8")).hexdigest()
    if (namespace != path.parent.name or criterion_sha != path.stem
            or criterion_sha != expected_criterion_sha or signature.ndim != 1
            or np.any(~np.isfinite(signature))):
        raise ValueError(f"invalid content-addressed signature cache entry: {path}")
    semantic_sha = hashlib.sha256(
        namespace.encode("ascii") + b"\0" + criterion.encode("utf-8") + b"\0"
        + signature.astype("<f8", copy=False).tobytes()
    ).hexdigest()
    return {
        "namespace_sha256": namespace,
        "criterion_sha256": criterion_sha,
        "criterion": criterion,
        "signature": signature,
        "semantic_sha256": semantic_sha,
    }


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    shutil.copy2(source, destination)
    return "copy"


def _merge_signature_cache(source_root: Path, destination_root: Path) -> dict:
    source_cache = source_root / "signature_cache"
    counts = {"new": 0, "duplicate": 0, "hardlink": 0, "copy": 0}
    semantic_rows = []
    if not source_cache.is_dir():
        return {**counts, "semantic_rows": semantic_rows}
    for source in sorted(source_cache.glob("*/*.npz")):
        observed = _signature_entry(source)
        destination = destination_root / source.relative_to(source_cache)
        if destination.exists():
            existing = _signature_entry(destination)
            if existing["semantic_sha256"] != observed["semantic_sha256"]:
                raise ValueError(f"signature cache collision disagrees: {source}")
            counts["duplicate"] += 1
        else:
            method = _link_or_copy(source, destination)
            counts["new"] += 1
            counts[method] += 1
        semantic_rows.append((
            str(source.relative_to(source_cache)), observed["semantic_sha256"]))
    return {**counts, "semantic_rows": semantic_rows}


def _valid_choice_payload(payload: str) -> np.ndarray:
    values = np.asarray(json.loads(str(payload)), dtype=float)
    if (values.ndim != 1 or len(values) < 2 or np.any(~np.isfinite(values))
            or np.any(values < 0.0) or values.sum() <= 0.0):
        raise ValueError("invalid cached MCQ probability row")
    return values / values.sum()


def _merge_choice_cache(source_root: Path, destination: sqlite3.Connection) -> dict:
    source_path = source_root / "mcq_query_cache" / "choice_probabilities.sqlite"
    counts = {"new": 0, "duplicate": 0}
    if not source_path.is_file():
        return counts
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    try:
        for key, payload in source.execute(
                "SELECT cache_key, probabilities_json FROM choice_rows ORDER BY cache_key"):
            normalized = _valid_choice_payload(str(payload))
            existing = destination.execute(
                "SELECT probabilities_json FROM choice_rows WHERE cache_key=?", (str(key),)
            ).fetchone()
            if existing is None:
                destination.execute(
                    "INSERT INTO choice_rows(cache_key, probabilities_json) VALUES (?, ?)",
                    (str(key), json.dumps(normalized.tolist(), separators=(",", ":"))),
                )
                counts["new"] += 1
            else:
                previous = _valid_choice_payload(str(existing[0]))
                if not np.array_equal(previous, normalized):
                    raise ValueError(f"MCQ choice-cache collision disagrees for key {key}")
                counts["duplicate"] += 1
    finally:
        source.close()
    destination.commit()
    return counts


def _scored_prompt_index(root: Path) -> dict[str, dict[str, list[dict]]]:
    indexed: dict[str, dict[str, list[dict]]] = {}
    for path in sorted(root.glob("*/**/scored.npz")):
        relative = path.relative_to(root)
        metric_key = relative.parts[0]
        if metric_key in {"mcq_codebook_candidates", "signature_cache"}:
            continue
        try:
            z = np.load(path, allow_pickle=True)
            if "texts" not in z or "sigs" not in z:
                continue
            texts = [str(value) for value in z["texts"]]
            signatures = np.asarray(z["sigs"], float)
            if signatures.ndim != 2 or len(signatures) != len(texts):
                continue
            artifact_sha = file_sha256(path)
            namespace = str(z["cache_namespace_sha256"]) if "cache_namespace_sha256" in z else None
        except Exception:
            continue
        by_text = indexed.setdefault(metric_key, {})
        for text in texts:
            by_text.setdefault(text, []).append({
                "path": str(relative),
                "sha256": artifact_sha,
                "cache_namespace_sha256": namespace,
            })
    return indexed


def _candidate_rows(source_roots: list[Path]) -> tuple[dict[str, list[dict]], list[dict]]:
    candidates: dict[str, dict[str, dict]] = {}
    rejected = []
    scored_by_root = {str(root): _scored_prompt_index(root) for root in source_roots}
    for root in source_roots:
        for path in sorted(root.glob("*/**/proposal_*.jsonl")):
            relative = path.relative_to(root)
            metric_key = relative.parts[0]
            artifact_sha = file_sha256(path)
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    try:
                        original = json.loads(line)
                        text = str(original.get("text", "")).strip()
                        if not text:
                            raise ValueError("empty prompt text")
                    except Exception as exc:
                        rejected.append({
                            "source_root": str(root),
                            "path": str(relative),
                            "line": line_number,
                            "reason": str(exc),
                        })
                        continue
                    provenance = {
                        "source_root": str(root),
                        "path": str(relative),
                        "artifact_sha256": artifact_sha,
                        "line": line_number,
                        "family": original.get("family"),
                        "model": original.get("model"),
                        "model_revision": original.get("model_revision"),
                        "proposal_mode": original.get("proposal_mode", "atomic"),
                        "seed": original.get("seed"),
                        "scored_artifacts": scored_by_root[str(root)].get(
                            metric_key, {}).get(text, []),
                    }
                    entry = candidates.setdefault(metric_key, {}).setdefault(text, {
                        "text": text,
                        "source_provenance": [],
                    })
                    entry["source_provenance"].append(provenance)

    output = {}
    for metric_key, by_text in candidates.items():
        used_seeds = set()
        rows = []
        for index, text in enumerate(sorted(by_text)):
            salt = 0
            while True:
                raw = hashlib.sha256(
                    f"{metric_key}\0{text}\0{salt}".encode("utf-8")).digest()[:8]
                seed = int.from_bytes(raw, "big") & ((1 << 63) - 1)
                if seed not in used_seeds:
                    used_seeds.add(seed)
                    break
                salt += 1
            sources = by_text[text]["source_provenance"]
            first = sources[0]
            rows.append({
                "schema": CANDIDATE_SCHEMA,
                "text": text,
                "family": "historical_candidate",
                "model": str(first.get("model") or "historical"),
                "model_revision": str(first.get("model_revision") or "unknown"),
                "temperature": 0.0,
                "seed": seed,
                "attempt_idx": index,
                "accepted_idx": index,
                "prompt_sha256": hashlib.sha256(
                    b"cr3-evidence-store-candidate-only-v1").hexdigest(),
                "generator_config_sha256": _payload_sha256({
                    "schema": CANDIDATE_SCHEMA,
                    "role": "candidate_only",
                    "metric_key": metric_key,
                }),
                "proposal_mode": str(first.get("proposal_mode") or "historical"),
                "evidence_role": "candidate_only",
                "eligible_as_fresh_audit": False,
                "source_was_scored": any(source["scored_artifacts"] for source in sources),
                "source_provenance": sources,
            })
        output[metric_key] = rows
    return output, rejected


def load_evidence_manifest(root: str | Path) -> dict:
    """Validate an immutable evidence-store manifest and its declared payloads."""
    source = Path(root).resolve()
    path = source / "evidence_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    core = dict(payload)
    observed = str(core.pop("manifest_sha256", ""))
    if core.get("schema") != SCHEMA_VERSION or observed != _payload_sha256(core):
        raise ValueError(f"invalid or mutated evidence manifest: {path}")
    for metric_key, entry in core["metrics"].items():
        candidate = source / entry["candidate_path"]
        if (not candidate.is_file()
                or file_sha256(candidate) != entry["candidate_sha256"]):
            raise ValueError(f"evidence candidates changed for {metric_key}")
    semantic_rows = []
    for signature_path in sorted((source / "signature_cache").glob("*/*.npz")):
        signature = _signature_entry(signature_path)
        semantic_rows.append((
            str(signature_path.relative_to(source / "signature_cache")),
            signature["semantic_sha256"],
        ))
    declared_signatures = core["signature_cache"]
    if (len(semantic_rows) != int(declared_signatures["n_unique_entries"])
            or _payload_sha256(sorted(semantic_rows))
            != declared_signatures["semantic_merkle_sha256"]):
        raise ValueError("evidence signature cache changed")
    choice = source / core["choice_cache"]["path"]
    if not choice.is_file() or file_sha256(choice) != core["choice_cache"]["sha256"]:
        raise ValueError("evidence choice cache changed")
    return payload


def install_evidence_store(source_root: str | Path, destination_root: str | Path) -> dict:
    """Install validated content-addressed caches into one immutable CR-3 run.

    Candidate prompt admission is deliberately handled by the mining orchestrator;
    this function only reuses deterministic executor and MCQ query results.
    """
    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    manifest = load_evidence_manifest(source)
    destination.mkdir(parents=True, exist_ok=True)
    install_path = destination / "evidence_install.json"
    if install_path.exists():
        installed = json.loads(install_path.read_text())
        if (installed.get("schema") != "cr3-evidence-install-v1"
                or installed.get("source_manifest_sha256")
                != manifest["manifest_sha256"]):
            raise ValueError("run contains a different evidence-store installation")
        return installed

    signature = _merge_signature_cache(source, destination / "signature_cache")
    signature.pop("semantic_rows", None)
    choice_path = destination / "mcq_query_cache" / "choice_probabilities.sqlite"
    choice_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(choice_path)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS choice_rows (cache_key TEXT PRIMARY KEY, "
            "probabilities_json TEXT NOT NULL) WITHOUT ROWID")
        choice = _merge_choice_cache(source, connection)
    finally:
        connection.close()
    payload = {
        "schema": "cr3-evidence-install-v1",
        "source_root": str(source),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "signature_cache": signature,
        "choice_cache": choice,
        "role_contract": manifest["role_contract"],
    }
    _write_json(install_path, payload)
    return payload


def build_evidence_store(source_roots: Iterable[str | Path], out_root: str | Path) -> dict:
    roots = [Path(root).resolve() for root in source_roots]
    if not roots or len(set(roots)) != len(roots) or any(not root.is_dir() for root in roots):
        raise ValueError("source roots must be unique existing directories")
    destination = Path(out_root).resolve()
    if destination.exists():
        raise FileExistsError(f"evidence destination already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        source_metadata = []
        signature_counts = {"new": 0, "duplicate": 0, "hardlink": 0, "copy": 0}
        semantic_rows = []
        choice_database = temporary / "mcq_query_cache" / "choice_probabilities.sqlite"
        choice_database.parent.mkdir(parents=True)
        choice = sqlite3.connect(choice_database)
        choice.execute(
            "CREATE TABLE choice_rows (cache_key TEXT PRIMARY KEY, "
            "probabilities_json TEXT NOT NULL) WITHOUT ROWID")
        choice_counts = {"new": 0, "duplicate": 0}
        for root in roots:
            manifest = root / "run_manifest.json"
            source_metadata.append({
                "root": str(root),
                "run_manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
                "run_manifest_schema": (
                    json.loads(manifest.read_text()).get("schema") if manifest.is_file() else None),
            })
            observed = _merge_signature_cache(root, temporary / "signature_cache")
            semantic_rows.extend(observed.pop("semantic_rows"))
            for key, value in observed.items():
                signature_counts[key] += int(value)
            observed_choice = _merge_choice_cache(root, choice)
            for key, value in observed_choice.items():
                choice_counts[key] += int(value)
        choice.execute("PRAGMA journal_mode=DELETE")
        choice.commit()
        choice.close()

        candidates, rejected = _candidate_rows(roots)
        metrics = {}
        for metric_key, rows in sorted(candidates.items()):
            path = temporary / "candidates" / f"{metric_key}.jsonl"
            _write_jsonl(path, rows)
            metrics[metric_key] = {
                "candidate_path": str(path.relative_to(temporary)),
                "candidate_sha256": file_sha256(path),
                "n_unique_candidates": len(rows),
                "n_source_scored": int(sum(row["source_was_scored"] for row in rows)),
                "evidence_role": "candidate_only",
            }
        semantic_rows = sorted(set(semantic_rows))
        core = {
            "schema": SCHEMA_VERSION,
            "sources": source_metadata,
            "metrics": metrics,
            "signature_cache": {
                **signature_counts,
                "n_unique_entries": signature_counts["new"],
                "semantic_merkle_sha256": _payload_sha256(semantic_rows),
            },
            "choice_cache": {
                **choice_counts,
                "path": str(choice_database.relative_to(temporary)),
                "sha256": file_sha256(choice_database),
            },
            "rejected": rejected,
            "role_contract": {
                "imported_prompts": "candidate_only",
                "may_raise_achieved_lower_bound": True,
                "may_enter_frozen_discovery_pool": True,
                "may_serve_as_fresh_confirmation": False,
                "uses_external_labels": False,
            },
        }
        payload = {**core, "manifest_sha256": _payload_sha256(core)}
        _write_json(temporary / "evidence_manifest.json", payload)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return payload
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args(argv)
    payload = build_evidence_store(args.source_root, args.out_root)
    print(json.dumps({
        "out_root": str(Path(args.out_root).resolve()),
        "metrics": len(payload["metrics"]),
        "signature_cache": payload["signature_cache"],
        "choice_cache": payload["choice_cache"],
        "rejected": len(payload["rejected"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
