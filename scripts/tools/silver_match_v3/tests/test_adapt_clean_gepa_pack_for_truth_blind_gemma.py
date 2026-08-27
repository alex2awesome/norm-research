from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools.silver_match_v3.adapt_clean_gepa_pack_for_truth_blind_gemma import (
    adapt,
)
from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class CleanGepaGemmaAdapterTest(unittest.TestCase):
    def _fixture(self, root: Path) -> argparse.Namespace:
        task, role, bank_hash = "peer-review", "optimize", "a" * 64
        manifest = root / "manifest.json"
        canonical_bank = root / "canonical_bank.json"
        identities = root / "identities.jsonl"
        identity_freeze = root / "identity_freeze.json"
        exclusion_union = root / "exclusions.jsonl"
        exclusion_inventory = root / "exclusion_inventory.json"
        prelabel = root / "prelabel.json"
        clean_root = root / "clean"
        clean_root.mkdir()
        prior = root / "prior_outputs"
        prior.mkdir()

        bank = {
            "task": task,
            "source_sha256": bank_hash,
            "metrics": [
                {"metric_id": "m1", "name": "one"},
                {"metric_id": "m2", "name": "two"},
            ],
        }
        _write_json(canonical_bank, bank)
        _write_json(
            manifest,
            {
                "banks": {
                    task: {
                        "path": str(canonical_bank),
                        "source_sha256": bank_hash,
                    }
                },
                "corpora": {},
            },
        )
        identity_rows = [
            {
                "norm_uid": uid,
                "task": task,
                "corpus": "reviews",
                "source_group": f"g-{uid}",
                "gepa_role": role,
                "upstream_split": "train",
            }
            for uid in ("n1", "n2")
        ]
        item_rows = [
            {
                "norm_uid": row["norm_uid"],
                "task": task,
                "corpus": row["corpus"],
                "source_group": row["source_group"],
                "gepa_role": role,
                "predeclared_split": "train",
                "truth_hidden": True,
                "row": 1,
                "norm": "clear evaluation",
            }
            for row in identity_rows
        ]
        write_jsonl(identities, identity_rows)
        write_jsonl(clean_root / "items.jsonl", item_rows)
        _write_json(clean_root / "bank.json", {**bank, "metrics": list(reversed(bank["metrics"]))})
        write_jsonl(
            exclusion_union,
            [
                {
                    "norm_uid": "excluded",
                    "task": task,
                    "corpus": "reviews",
                    "source_group": "g-excluded",
                }
            ],
        )
        _write_json(
            identity_freeze,
            {
                "schema_version": "silver-match-v3-clean-gepa-panel-freeze-v1",
                "status": "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES",
                "task": task,
                "role": role,
                "required_upstream_split": "train",
                "selected_count": 2,
                "selected_source_groups": 2,
                "outputs": {"identities": {"sha256": sha256_file(identities)}},
                "exclusion_union": {
                    "selected_uid_overlap": 0,
                    "selected_source_group_overlap": 0,
                },
                "content_contract": {
                    "selection_uses_identity_and_source_group_only": True,
                    "downstream_outcomes_read": False,
                    "metric_ids_read": False,
                    "model_prediction_fields_read": False,
                    "truth_fields_read": False,
                },
            },
        )
        _write_json(
            exclusion_inventory,
            {
                "schema_version": "silver-match-v3-gepa-exclusion-union-v1",
                "status": "FROZEN_BEFORE_NEW_PANEL_SELECTION_PREDICTIONS_OR_LABELS",
                "task": task,
                "all_required_categories_present": True,
                "identity_union": {
                    "path": str(exclusion_union),
                    "sha256": sha256_file(exclusion_union),
                    "uids": 1,
                    "source_groups": 1,
                },
                "content_contract": {
                    "model_predictions_metric_ids_reasons_and_outcomes_used": False,
                    "parsed_sources_used_only_identity_fields": True,
                    "sealed_test_or_outcome_structured_content_parsed": False,
                },
            },
        )
        _write_json(
            prelabel,
            {
                "schema_version": "silver-match-v3-independent-pack-view-audit-v1",
                "status": "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING",
                "task": task,
                "count": 2,
                "bank_metric_count": 2,
                "same_uid_set": True,
                "same_bank_leaf_set": True,
                "same_canonical_item_content_by_uid": True,
                "same_frozen_source_pack": True,
                "prior_truth_or_predictions_exposed_to_either_pass": False,
                "candidate_proposals_exposed_to_either_pass": False,
                "pass_predictions_mutually_visible": False,
                "post_label_artifacts_present": False,
            },
        )
        _write_json(
            clean_root / "validation.json",
            {
                "schema_version": "silver-match-v3-clean-gepa-label-pack-v1",
                "status": "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING",
                "task": task,
                "gepa_role": role,
                "count": 2,
                "source_groups": 2,
                "bank_metric_count": 2,
                "bank_source_sha256": bank_hash,
                "truth_hidden": True,
                "prior_decisions_proposals_predictions_and_outcomes_hidden": True,
                "inputs": {
                    "manifest": {"sha256": sha256_file(manifest)},
                    "identities": {"sha256": sha256_file(identities)},
                    "identity_freeze": {"sha256": sha256_file(identity_freeze)},
                    "bank_source": {"sha256": sha256_file(canonical_bank)},
                },
                "outputs": {
                    "items": {"sha256": sha256_file(clean_root / "items.jsonl")},
                    "bank": {"sha256": sha256_file(clean_root / "bank.json")},
                },
                "usage_contract": {
                    "optimize_may_mutate_prompts": True,
                    "may_train_or_select_retriever": False,
                    "may_use_for_mi_or_outcome_estimation": False,
                    "may_use_as_test_or_blind_audit": False,
                },
            },
        )
        return argparse.Namespace(
            manifest=str(manifest),
            task=task,
            role=role,
            expected_count=2,
            clean_pack_root=str(clean_root),
            identities=str(identities),
            identity_freeze=str(identity_freeze),
            exclusion_inventory=str(exclusion_inventory),
            exclusion_union=str(exclusion_union),
            prelabel_independence_audit=str(prelabel),
            canonical_bank=str(canonical_bank),
            prior_model_output_root=[str(prior)],
            output_root=str(root / "out"),
        )

    def test_positive_all_optimize_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self._fixture(Path(raw))
            result = adapt(args)
            self.assertEqual(result["status"], "EXACT_TRUTH_AND_CANDIDATE_HIDDEN_PACK_PASS")
            partition = json.loads(
                Path(result["partition_freeze"]["path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(partition["role_counts"], {"optimize": 2})
            validation = json.loads(
                Path(result["source_pack_validation"]["path"]).read_text(encoding="utf-8")
            )
            self.assertTrue(validation["candidate_proposals_hidden"])
            self.assertEqual(validation["bank_metric_count"], 2)

    def test_prior_model_output_fails_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self._fixture(Path(raw))
            prior_file = Path(args.prior_model_output_root[0]) / "part-000.json"
            prior_file.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prior model output exists"):
                adapt(args)
            self.assertFalse(Path(args.output_root).exists())

    def test_forbidden_item_label_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self._fixture(Path(raw))
            items = Path(args.clean_pack_root) / "items.jsonl"
            rows = list(read_jsonl_for_test(items))
            rows[0]["metric_id"] = "m1"
            write_jsonl(items, rows)
            validation_path = Path(args.clean_pack_root) / "validation.json"
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            validation["outputs"]["items"]["sha256"] = sha256_file(items)
            _write_json(validation_path, validation)
            with self.assertRaisesRegex(ValueError, "contract violation"):
                adapt(args)
            self.assertFalse(Path(args.output_root).exists())

    def test_stale_prelabel_audit_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self._fixture(Path(raw))
            path = Path(args.prelabel_independence_audit)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["post_label_artifacts_present"] = True
            _write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "pre-label attestation"):
                adapt(args)
            self.assertFalse(Path(args.output_root).exists())

    def test_exclusion_overlap_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self._fixture(Path(raw))
            identities = list(read_jsonl_for_test(Path(args.identities)))
            union = Path(args.exclusion_inventory).parent / "exclusions.jsonl"
            write_jsonl(union, [identities[0]])
            inventory_path = Path(args.exclusion_inventory)
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            inventory["identity_union"].update(
                {"sha256": sha256_file(union), "uids": 1, "source_groups": 1}
            )
            _write_json(inventory_path, inventory)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                adapt(args)
            self.assertFalse(Path(args.output_root).exists())


def read_jsonl_for_test(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


if __name__ == "__main__":
    unittest.main()
