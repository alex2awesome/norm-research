from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3 import freeze_humor_ce_truth_collection as freezer
from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.validate_independent_teacher_labels import (
    main as validate_labels,
)


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_name: str,
) -> tuple[freezer.CollectionConfig, dict[str, Path], list[dict[str, object]]]:
    monkeypatch.setattr(freezer, "EXPECTED_CANONICAL_NORMS", 24)
    monkeypatch.setattr(freezer, "EXPECTED_BANK_METRICS", 3)

    source = tmp_path / "source"
    norms_by_corpus: dict[str, list[dict[str, object]]] = {
        "humor_a": [],
        "humor_b": [],
    }
    all_norms: list[dict[str, object]] = []
    for index in range(24):
        corpus = "humor_a" if index < 12 else "humor_b"
        row = {
            "schema_version": "canonical-v1",
            "task": "humor",
            "corpus": corpus,
            "row": index,
            "norm_uid": f"uid-{index:03d}",
            "source_id": f"thread:{index:03d}",
            "kind": "critique",
            "norm": f"Criterion {index}",
            "context": f"Evidence context {index}",
            "aspect": "unused weak hint",
        }
        norms_by_corpus[corpus].append(row)
        all_norms.append(row)
    norm_paths: dict[str, Path] = {}
    for corpus, rows in norms_by_corpus.items():
        path = source / f"{corpus}.jsonl"
        write_jsonl(path, rows)
        norm_paths[corpus] = path

    bank = {
        "task": "humor",
        "source_sha256": freezer.EXPECTED_BANK_SOURCE_SHA256,
        "metrics": [
            {"metric_id": f"a{index}", "name": f"Metric {index}", "description": "d"}
            for index in range(3)
        ],
    }
    bank_path = source / "bank.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    manifest = {
        "banks": {
            "humor": {
                "path": str(bank_path),
                "source_sha256": freezer.EXPECTED_BANK_SOURCE_SHA256,
            }
        },
        "corpora": {
            corpus: {"task": "humor", "path": str(path), "count": len(norms_by_corpus[corpus])}
            for corpus, path in norm_paths.items()
        },
    }
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Both inventories deliberately carry irrelevant weak outcome fields.  The
    # freezer is allowed to use only their identities and declared strata.
    legacy_path, disagreement_path = source / "legacy.jsonl", source / "disagreement.jsonl"
    legacy_rows = []
    disagreement_rows = []
    for row in all_norms:
        canonical = freezer._canonical_group(row)
        _, corpus, kind, identity = canonical.split("\x1f", 3)
        legacy_rows.append(
            {
                "task": "humor",
                "norm_uid": row["norm_uid"],
                "source_group": f"{corpus}:{kind}:{identity}",
                "decision": "BANK_GAP",
                "confidence": "low",
                "metric_id": "legacy-ignored",
            }
        )
        disagreement_rows.append(
            {
                "task": "humor",
                "norm_uid": row["norm_uid"],
                "source_group": canonical,
                "prediction": "ignored",
                "outcome": 1,
            }
        )
    write_jsonl(legacy_path, legacy_rows)
    write_jsonl(disagreement_path, disagreement_rows)

    # Exercise both colon and unit-separator exclusion forms, including one
    # group-only role row.  Every norm in these groups must be excluded.
    first_group = freezer._canonical_group(all_norms[0])
    _, corpus, kind, identity = first_group.split("\x1f", 3)
    exclusion_path = source / "prior_roles.jsonl"
    write_jsonl(
        exclusion_path,
        [
            {
                "task": "humor",
                "norm_uid": all_norms[0]["norm_uid"],
                "source_group": f"{corpus}:{kind}:{identity}",
            },
            {
                "task": "humor",
                "source_group": freezer._canonical_group(all_norms[1]),
            },
        ],
    )
    paths = {
        "manifest": manifest_path,
        "bank": bank_path,
        "legacy": legacy_path,
        "disagreement": disagreement_path,
        "exclusion": exclusion_path,
    }
    config = freezer.CollectionConfig(
        manifest=manifest_path,
        candidate_inventories=(
            freezer.NamedPath("legacy_nonmatch", legacy_path),
            freezer.NamedPath("disagreement", disagreement_path),
        ),
        exclusion_roles=(freezer.NamedPath("prior_roles", exclusion_path),),
        train_quotas=(
            freezer.Quota("legacy_nonmatch", 2),
            freezer.Quota("disagreement", 2),
        ),
        dev_quotas=(
            freezer.Quota("legacy_nonmatch", 1),
            freezer.Quota("disagreement", 1),
        ),
        output_root=tmp_path / output_name,
        train_count=4,
        dev_count=2,
        blind_count=4,
        chunk_size=3,
        seed=1776,
    )
    return config, paths, all_norms


def _write_abstention_labels(pack: Path, raw_root: Path) -> None:
    raw_root.mkdir()
    for chunk in sorted((pack / "chunks").glob("part-*.jsonl")):
        rows = list(read_jsonl(chunk))
        (raw_root / f"{chunk.stem}.json").write_text(
            json.dumps(
                {
                    "task": "humor",
                    "chunk_id": chunk.stem,
                    "labels": [
                        {
                            "norm_uid": row["norm_uid"],
                            "decision": "NO_EXPLICIT_CRITERION",
                            "metric_id": None,
                            "confidence": "high",
                            "reason": "The evidence contains no explicit criterion.",
                        }
                        for row in rows
                    ],
                }
            ),
            encoding="utf-8",
        )


def test_freezes_isolated_pack_and_is_validator_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, all_norms = _fixture(tmp_path, monkeypatch, output_name="pack_a")
    result = freezer.freeze_collection(config)
    pack = config.output_root
    validation = json.loads((pack / "validation.json").read_text())
    freeze = json.loads((pack / "FREEZE.json").read_text())
    identities = list(read_jsonl(pack / "identities.jsonl"))
    items = list(read_jsonl(pack / "items.jsonl"))

    assert freeze["role_counts"] == {"train": 4, "dev": 2, "blind": 4}
    assert freeze["selection"]["dev"]["balanced"] is True
    assert freeze["selection"]["dev"]["candidate_quota_targets"] == {
        "legacy_nonmatch": 1,
        "disagreement": 1,
    }
    assert freeze["selection"]["blind"]["candidate_inventories_consulted"] is False
    assert freeze["selection"]["blind"]["selected_by_corpus"] == {
        "humor_a": 2,
        "humor_b": 2,
    }
    assert freeze["source_isolation"]["cross_role_source_group_overlap"] == 0
    assert len({row["norm_uid"] for row in identities}) == 10
    assert len({row["source_group"] for row in identities}) == 10
    assert all("\x1f" in row["source_group"] for row in identities)
    assert {all_norms[0]["norm_uid"], all_norms[1]["norm_uid"]}.isdisjoint(
        {row["norm_uid"] for row in identities}
    )
    assert len(items) == 10
    assert all(
        not (freezer.FORBIDDEN_LABELER_FIELDS & set(row)) and row["truth_hidden"] is True
        for row in items
    )
    assert validation["outputs"]["items"]["sha256"] == sha256_file(pack / "items.jsonl")
    assert validation["outputs"]["bank"]["sha256"] == sha256_file(pack / "bank.json")
    assert validation["selection_freeze"]["sha256"] == sha256_file(pack / "FREEZE.json")
    assert result["validation_sha256"] == sha256_file(pack / "validation.json")
    assert len(list((pack / "chunks").glob("part-*.jsonl"))) == 4

    raw_root = tmp_path / "raw_labels"
    _write_abstention_labels(pack, raw_root)
    validated, report = tmp_path / "validated.jsonl", tmp_path / "validated.report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_independent_teacher_labels",
            "--pack-root",
            str(pack),
            "--raw-label-dir",
            str(raw_root),
            "--output",
            str(validated),
            "--report",
            str(report),
        ],
    )
    validate_labels()
    validated_rows = list(read_jsonl(validated))
    assert len(validated_rows) == 10
    assert Counter(row["split"] for row in validated_rows) == {
        "train": 4,
        "dev": 2,
        "test": 4,
    }

    with pytest.raises(FileExistsError, match="append-only"):
        freezer.freeze_collection(config)


def test_selection_is_deterministic_when_ignored_outcomes_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_a, paths, _ = _fixture(tmp_path, monkeypatch, output_name="pack_a")
    freezer.freeze_collection(config_a)

    # Change only fields outside the declared identity/stratum interface.
    for path in (paths["legacy"], paths["disagreement"]):
        rows = list(read_jsonl(path))
        for row in rows:
            row["decision"] = "MATCH"
            row["confidence"] = "high"
            row["metric_id"] = "different-ignored-value"
            row["prediction"] = {"arbitrary": "changed"}
            row["outcome"] = 999
        write_jsonl(path, rows)

    config_b = freezer.CollectionConfig(
        **{
            **config_a.__dict__,
            "output_root": tmp_path / "pack_b",
        }
    )
    freezer.freeze_collection(config_b)
    for relative in ("identities.jsonl", "items.jsonl", "bank.json"):
        assert (config_a.output_root / relative).read_bytes() == (
            config_b.output_root / relative
        ).read_bytes()
    chunks_a = sorted((config_a.output_root / "chunks").glob("part-*.jsonl"))
    chunks_b = sorted((config_b.output_root / "chunks").glob("part-*.jsonl"))
    assert [path.name for path in chunks_a] == [path.name for path in chunks_b]
    assert [path.read_bytes() for path in chunks_a] == [path.read_bytes() for path in chunks_b]
    # Provenance reports still distinguish the changed candidate artifacts.
    freeze_a = json.loads((config_a.output_root / "FREEZE.json").read_text())
    freeze_b = json.loads((config_b.output_root / "FREEZE.json").read_text())
    assert freeze_a["inputs"]["candidate_inventories"] != freeze_b["inputs"]["candidate_inventories"]


def test_rejects_unbalanced_dev_and_mismatched_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, paths, all_norms = _fixture(tmp_path, monkeypatch, output_name="bad_pack")
    unbalanced = freezer.CollectionConfig(
        **{
            **config.__dict__,
            "dev_count": 4,
            "dev_quotas": (
                freezer.Quota("legacy_nonmatch", 3),
                freezer.Quota("disagreement", 1),
            ),
        }
    )
    with pytest.raises(ValueError, match="balanced"):
        freezer.freeze_collection(unbalanced)

    rows = list(read_jsonl(paths["legacy"]))
    rows[2]["source_group"] = freezer._canonical_group(all_norms[3])
    write_jsonl(paths["legacy"], rows)
    with pytest.raises(ValueError, match="disagrees with canonical UID"):
        freezer.freeze_collection(config)


def test_rejects_nonexact_bank_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, paths, _ = _fixture(tmp_path, monkeypatch, output_name="bad_bank_pack")
    manifest = json.loads(paths["manifest"].read_text())
    manifest["banks"]["humor"]["source_sha256"] = "wrong-bank"
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exact Humor R2 bank source SHA"):
        freezer.freeze_collection(config)


def test_reads_audited_inventory_priority_strata() -> None:
    assert freezer._row_flags(
        {
            "priority_strata": ["legacy_nonmatch_re_adjudication", "uncovered_leaf_proxy"],
            "primary_priority_stratum": "legacy_nonmatch_re_adjudication",
        }
    ) == {"legacy_nonmatch_re_adjudication", "uncovered_leaf_proxy"}
