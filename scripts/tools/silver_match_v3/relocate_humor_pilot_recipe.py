#!/usr/bin/env python3
"""Create an audited path-only relocation of the frozen Humor CE pilot recipe.

The final handoff may be built on a host other than the pilot-training host.
This utility rewrites only the winner root, base-manifest location, and base
model location while proving every actual recipe field is unchanged.  It does
not copy checkpoints, select a new winner, open held-out outcomes, or train.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .common import sha256_file
from .freeze_humor_final_stack_handoff import validate_pilot_recipe


SCHEMA = "silver-match-v3-humor-pilot-recipe-path-relocation-v1"


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def relocate(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = Path(args.selection).resolve()
    source_model = Path(args.source_model).resolve()
    target_model = Path(args.target_model)
    local_root = Path(args.output_root).resolve()
    published_root = Path(args.published_output_root)
    if local_root.exists():
        raise FileExistsError(local_root)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    recipe, audit = validate_pilot_recipe(selection_path, ce_model=source_model)
    winner = dict(selection["winner_record"])
    source_run = Path(audit["winner_run_config"]["path"]).resolve()
    source_manifest = Path(audit["base_model_manifest"]["path"]).resolve()
    run_config = json.loads(source_run.read_text(encoding="utf-8"))
    base_manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if Path(str(run_config.get("model") or "")).resolve() != source_model:
        raise ValueError("source run config/model differs")
    if Path(str(base_manifest.get("model") or "")).resolve() != source_model:
        raise ValueError("source base manifest/model differs")

    local_run = local_root / "winner" / "run_config.json"
    local_manifest = local_root / "BASE_MODEL_MANIFEST.json"
    local_selection = local_root / "PILOT_SELECTION.json"
    published_run_root = published_root / "winner"
    published_manifest = published_root / "BASE_MODEL_MANIFEST.json"
    run_config["model"] = str(target_model)
    base_manifest["model"] = str(target_model)
    _write(local_run, run_config)
    _write(local_manifest, base_manifest)

    winner["root"] = str(published_run_root)
    winner["run_config_sha256"] = sha256_file(local_run)
    relocated = dict(selection)
    relocated["winner_record"] = winner
    relocated["base_manifest"] = str(published_manifest)
    relocated["base_manifest_sha256"] = sha256_file(local_manifest)
    base_contract = dict(relocated.get("base_contract") or {})
    base_contract["manifest"] = str(published_manifest)
    base_contract["manifest_sha256"] = sha256_file(local_manifest)
    relocated["base_contract"] = base_contract
    relocated["path_relocation_only"] = True
    relocated["source_selection_sha256"] = sha256_file(selection_path)
    _write(local_selection, relocated)

    # Re-validate via a local shadow carrying the exact published-path bytes.
    # The published selection itself is intentionally not modified for this.
    shadow = dict(relocated)
    shadow_winner = dict(winner)
    shadow_winner["root"] = str(local_run.parent)
    shadow["winner_record"] = shadow_winner
    shadow["base_manifest"] = str(local_manifest)
    shadow_path = local_root / ".LOCAL_VALIDATION_SELECTION.json"
    _write(shadow_path, shadow)
    relocated_recipe, _ = validate_pilot_recipe(shadow_path, ce_model=target_model)
    if relocated_recipe != recipe:
        raise ValueError("pilot recipe fields changed during path relocation")
    shadow_path.unlink()

    report = {
        "schema_version": SCHEMA,
        "status": "AUDITED_PATH_ONLY_RELOCATION_NO_RESELECTION",
        "task": "humor",
        "source": {
            "selection": _ref(selection_path),
            "winner_run_config": _ref(source_run),
            "base_manifest": _ref(source_manifest),
            "model": str(source_model),
        },
        "relocated": {
            "selection": _ref(local_selection),
            "winner_run_config": _ref(local_run),
            "base_manifest": _ref(local_manifest),
            "published_root": str(published_root),
            "model": str(target_model),
        },
        "recipe_fields": recipe,
        "winner_unchanged": selection.get("winner"),
        "selection_outcomes_changed": False,
        "test_or_blind_outcomes_opened": False,
        "training_steps_run": 0,
        "gpu_processes_launched": 0,
    }
    _write(local_root / "RELOCATION_REPORT.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--published-output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(relocate(args), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
