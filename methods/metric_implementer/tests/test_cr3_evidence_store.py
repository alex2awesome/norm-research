"""Immutable reuse tests for historical CR-3 GPU evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from methods.metric_implementer.experiments.cr3_evidence_store import (
    build_evidence_store,
    install_evidence_store,
    load_evidence_manifest,
)
from scripts.tools.cr3_mining_worker import _content_cached_signature


def _choice_cache(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE choice_rows (cache_key TEXT PRIMARY KEY, "
        "probabilities_json TEXT NOT NULL) WITHOUT ROWID")
    connection.execute(
        "INSERT INTO choice_rows VALUES (?, ?)", ("query-key", "[0.25,0.75]"))
    connection.commit()
    connection.close()


def _source_root(root: Path, *, duplicate: bool = False):
    root.mkdir()
    (root / "run_manifest.json").write_text(json.dumps({"schema": "historical"}))
    proposal = root / "humor_R3_metric0" / "monitor" / "iter_000" / "proposal_phi.jsonl"
    proposal.parent.mkdir(parents=True)
    row = {
        "text": "Does the historical criterion hold?",
        "family": "phi",
        "model": "model",
        "model_revision": "revision",
        "proposal_mode": "atomic",
        "seed": 17,
    }
    proposal.write_text(json.dumps(row) + "\n")
    scored = proposal.parent / "scored.npz"
    np.savez_compressed(
        scored,
        texts=np.asarray([row["text"]], object),
        sigs=np.asarray([[0.1, 0.9]], float),
        cache_namespace_sha256=np.asarray("namespace"),
    )
    _content_cached_signature(
        str(root / "signature_cache"),
        "namespace",
        row["text"],
        2,
        lambda _text: np.asarray([0.1, 0.9]),
    )
    _choice_cache(root / "mcq_query_cache" / "choice_probabilities.sqlite")
    if duplicate:
        row["seed"] = 29
        proposal.write_text(json.dumps(row) + "\n")


def test_evidence_store_deduplicates_candidates_and_validates_gpu_caches(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _source_root(first)
    _source_root(second, duplicate=True)
    output = tmp_path / "evidence"

    payload = build_evidence_store([first, second], output)

    assert payload["schema"] == "cr3-evidence-store-v1"
    assert payload["signature_cache"]["n_unique_entries"] == 1
    assert payload["signature_cache"]["duplicate"] == 1
    assert payload["choice_cache"]["new"] == 1
    assert payload["choice_cache"]["duplicate"] == 1
    metric = payload["metrics"]["humor_R3_metric0"]
    assert metric["n_unique_candidates"] == 1
    assert metric["n_source_scored"] == 1
    candidate_path = output / metric["candidate_path"]
    candidate = json.loads(candidate_path.read_text())
    assert candidate["evidence_role"] == "candidate_only"
    assert candidate["eligible_as_fresh_audit"] is False
    assert len(candidate["source_provenance"]) == 2

    source_cache = next((first / "signature_cache").glob("*/*.npz"))
    stored_cache = next((output / "signature_cache").glob("*/*.npz"))
    assert os.stat(source_cache).st_ino == os.stat(stored_cache).st_ino

    connection = sqlite3.connect(output / payload["choice_cache"]["path"])
    assert connection.execute("SELECT count(*) FROM choice_rows").fetchone()[0] == 1
    connection.close()

    assert load_evidence_manifest(output)["manifest_sha256"] == payload["manifest_sha256"]
    installed_root = tmp_path / "installed"
    installed = install_evidence_store(output, installed_root)
    assert installed["role_contract"]["may_serve_as_fresh_confirmation"] is False
    assert installed["signature_cache"]["new"] == 1
    replayed = install_evidence_store(output, installed_root)
    assert replayed == installed
    assert len(list((installed_root / "signature_cache").glob("*/*.npz"))) == 1
    connection = sqlite3.connect(
        installed_root / "mcq_query_cache" / "choice_probabilities.sqlite")
    assert connection.execute("SELECT count(*) FROM choice_rows").fetchone()[0] == 1
    connection.close()

    next((output / "signature_cache").glob("*/*.npz")).unlink()
    with pytest.raises(ValueError, match="signature cache changed"):
        load_evidence_manifest(output)
