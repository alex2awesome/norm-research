from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_patent_claim_graph_additive_runner_v1 import (
    PatentClaimGraphExecutionError,
    build_execution,
    execute_split,
    main,
    validate_items,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "metric_seam_pilot" / "hierarchy_r123"
MANIFEST = OUT / "items_v2" / "patents" / "manifest.json"
TRAIN = OUT / "items_v2" / "patents" / "compiler_train.json"
HELDOUT = OUT / "items_v2" / "patents" / "sealed_heldout.json"
TRAIN_ARTIFACT = OUT / "patents_claim_graph_additive_compiler_train_v2.json"
HELDOUT_ARTIFACT = (
    OUT / "patents_claim_graph_additive_heldout_pre_reference_v1.json"
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_runner_accepts_only_opaque_key_and_exact_ctext() -> None:
    item = {"item_key": "train_0001", "ctext": "CLAIMS:\n1 . A system."}
    validate_items([item], phase="compiler_train")
    for forbidden in ("reference", "outcome", "label", "source_id"):
        contaminated = [{**item, forbidden: 1}]
        with pytest.raises(PatentClaimGraphExecutionError, match="exactly"):
            validate_items(contaminated, phase="compiler_train")
    with pytest.raises(PatentClaimGraphExecutionError, match="split key"):
        validate_items(
            [{"item_key": "heldout_0001", "ctext": item["ctext"]}],
            phase="compiler_train",
        )


def test_manifest_fails_closed_on_shared_bytes_and_blindness() -> None:
    items = _load(TRAIN)
    manifest = _load(MANIFEST)
    assert validate_manifest(manifest, items, phase="compiler_train") == 4000

    drift = deepcopy(manifest)
    drift["representation"]["same_bytes_required_for_prompt_and_code"] = False
    with pytest.raises(PatentClaimGraphExecutionError, match="shared-ctext"):
        validate_manifest(drift, items, phase="compiler_train")

    contaminated = deepcopy(manifest)
    contaminated["selection"]["outcome_or_reference_values_used"] = True
    with pytest.raises(PatentClaimGraphExecutionError, match="reference blind"):
        validate_manifest(contaminated, items, phase="compiler_train")


def test_execution_is_nonaggregating_and_certificate_scoped() -> None:
    items = [
        {
            "item_key": "train_0001",
            "ctext": (
                "CLAIMS:\n1 . A system comprising a sensor, wherein m is an integer.\n\n"
                "2 . The system of claim 1, wherein the sensor has a length of at least "
                "10 mm and m=2."
            ),
        }
    ]
    artifact = execute_split(
        items, phase="compiler_train", representation_max_chars=4000
    )
    assert artifact["summary"]["n_items"] == 1
    assert artifact["summary"]["failure_types"] == {}
    assert artifact["design"]["exact_frozen_ctext_used"] is True
    assert artifact["design"]["outcome_or_reference_values_loaded"] is False
    assert artifact["design"]["model_or_api_calls_made"] is False
    assert artifact["design"]["accelerators_used"] is False
    assert artifact["design"]["whole_patent_score_emitted"] is False
    assert artifact["rows"][0]["result"]["presented_character_count"] == len(
        items[0]["ctext"]
    )
    assert artifact["summary"]["relation_certificates"][
        "formula_variable_definition_alignment"
    ]["n_items_with_finite_certificates"] == 1


def test_build_execution_binds_exact_source_bytes() -> None:
    items = _load(TRAIN)[:2]
    manifest = {
        **_load(MANIFEST),
        "selection": {**_load(MANIFEST)["selection"], "train_n": 2},
    }
    item_bytes = json.dumps(items).encode()
    manifest_bytes = json.dumps(manifest).encode()
    artifact = build_execution(
        items,
        manifest,
        phase="compiler_train",
        item_source_bytes=item_bytes,
        manifest_source_bytes=manifest_bytes,
        item_source_path="frozen-items.json",
        manifest_source_path="frozen-manifest.json",
    )
    assert artifact["sources"]["items"]["path"] == "frozen-items.json"
    assert len(artifact["sources"]["items"]["sha256"]) == 64
    assert artifact["sources"]["manifest"]["path"] == "frozen-manifest.json"

    with pytest.raises(PatentClaimGraphExecutionError, match="exact bound"):
        build_execution(
            items,
            manifest,
            phase="compiler_train",
            item_source_bytes=b"[]",
            manifest_source_bytes=manifest_bytes,
            item_source_path="wrong.json",
            manifest_source_path="frozen-manifest.json",
        )


def test_cli_refuses_to_overwrite_execution_receipt(tmp_path: Path) -> None:
    items = [{"item_key": "train_0001", "ctext": "CLAIMS:\n1 . A system."}]
    manifest = deepcopy(_load(MANIFEST))
    manifest["selection"]["train_n"] = 1
    item_path = tmp_path / "items.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "receipt.json"
    item_path.write_text(json.dumps(items), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    argv = [
        "--items",
        str(item_path),
        "--manifest",
        str(manifest_path),
        "--phase",
        "compiler_train",
        "--output",
        str(output),
    ]
    assert main(argv) == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)


@pytest.mark.parametrize(
    ("items_path", "phase", "artifact_path"),
    [
        (TRAIN, "compiler_train", TRAIN_ARTIFACT),
        (HELDOUT, "heldout_pre_reference", HELDOUT_ARTIFACT),
    ],
)
def test_checked_in_receipt_equals_fresh_exact_ctext_execution(
    items_path: Path, phase: str, artifact_path: Path
) -> None:
    item_bytes = items_path.read_bytes()
    manifest_bytes = MANIFEST.read_bytes()
    fresh = build_execution(
        json.loads(item_bytes),
        json.loads(manifest_bytes),
        phase=phase,
        item_source_bytes=item_bytes,
        manifest_source_bytes=manifest_bytes,
        item_source_path=str(items_path.relative_to(ROOT)),
        manifest_source_path=str(MANIFEST.relative_to(ROOT)),
    )
    assert _load(artifact_path) == fresh
