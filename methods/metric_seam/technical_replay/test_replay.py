from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from methods.metric_seam.technical_replay.core import (
    ManifestError,
    evaluate_manifest,
    resolve_pointer,
    validate_manifest,
)
from methods.metric_seam.technical_replay.fullpaper_probe import (
    numeric_recurrence,
    specific_numeric_tokens,
)


def _axis(assessment="not_evaluated", artifacts=None, measurements=None):
    return {
        "assessment": assessment,
        "claim": "bounded claim",
        "limitations": [],
        "artifacts_used": artifacts or [],
        "measurements": measurements or [],
    }


def _manifest(mode="manual", reference_access="seen", required=None, observed=None):
    return {
        "schema_version": "technical-replay-v2",
        "experiment_id": "unit-test",
        "external_supervision": "none",
        "objective_definitions": {
            "articulability": "prompt channel",
            "verifiability": "code channel",
            "isomorphic_reconstruction": "reference agreement",
            "constructive_extension": "code-native disagreement certificate",
        },
        "cases": [
            {
                "case_id": "c1",
                "domain": "math",
                "relation": "equation equivalence",
                "discovery_mode": mode,
                "pipeline_status": "selected",
                "selection_mode": "retrospective_seed",
                "relation_depth": {
                    "level": 3,
                    "label": "formal_solver_or_evidence_graph",
                    "mechanism": "symbolic equation checker",
                },
                "reference_access_during_discovery": reference_access,
                "corpus": {
                    "observed_sections": observed or ["answer"],
                    "required_sections": required or ["answer"],
                    "limitations": [],
                },
                "artifacts": [
                    {
                        "artifact_id": "data",
                        "path": "fixture.json",
                        "role": "test fixture",
                        "discovery_mode": mode,
                    }
                ],
                "utility": _axis(
                    "supported",
                    ["data"],
                    [
                        {
                            "measurement_id": "n",
                            "kind": "length",
                            "artifact_id": "data",
                            "pointer": "/rows",
                        }
                    ],
                ),
                "axes": {
                    "articulability": _axis(),
                    "verifiability": _axis(),
                    "isomorphic_reconstruction": _axis(
                        "supported",
                        ["data"],
                        [
                            {
                                "measurement_id": "a",
                                "kind": "scalar",
                                "artifact_id": "data",
                                "pointer": "/a",
                            },
                            {
                                "measurement_id": "b",
                                "kind": "scalar",
                                "artifact_id": "data",
                                "pointer": "/b",
                            },
                            {
                                "measurement_id": "delta",
                                "kind": "difference",
                                "left": "a",
                                "right": "b",
                            },
                        ],
                    ),
                    "constructive_extension": _axis(
                        "supported",
                        ["data"],
                        [
                            {
                                "measurement_id": "positive",
                                "kind": "count_where",
                                "artifact_id": "data",
                                "pointer": "/rows",
                                "field_pointer": "/score",
                                "gt": 0,
                            }
                        ],
                    ),
                },
            }
        ],
    }


class ReplayTests(unittest.TestCase):
    def _evaluate(self, manifest):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fixture.json").write_text(
                json.dumps({"a": 0.75, "b": 0.25, "rows": [{"score": 1}, {"score": 0}]})
            )
            return evaluate_manifest(manifest, root)

    def test_json_pointer(self):
        self.assertEqual(resolve_pointer({"a/b": [{"~x": 3}]}, "/a~1b/0/~0x"), 3)

    def test_invalid_discovery_mode_rejected(self):
        manifest = _manifest(mode="automatic")
        with self.assertRaises(ManifestError):
            validate_manifest(manifest)

    def test_all_four_objectives_are_mandatory(self):
        manifest = _manifest()
        del manifest["cases"][0]["axes"]["constructive_extension"]
        with self.assertRaises(ManifestError):
            validate_manifest(manifest)

    def test_measurements_and_canonical_record_resolve(self):
        result = self._evaluate(_manifest())
        case = result["cases"][0]
        iso = case["axes"]["isomorphic_reconstruction"]
        self.assertEqual(iso["measurements"]["delta"], 0.5)
        self.assertEqual(case["axes"]["constructive_extension"]["measurements"]["positive"], 1)
        self.assertIn("outcome", case["canonical_v2"])

    def test_manual_replay_cannot_claim_automatic_decomposition(self):
        result = self._evaluate(_manifest(mode="manual", reference_access="sealed"))
        case = result["cases"][0]
        permission = case["axes"]["isomorphic_reconstruction"]["claim_permissions"]
        self.assertFalse(permission["may_claim_historical_automatic_selection"])
        self.assertTrue(permission["may_claim_selected_pipeline_result"])
        self.assertFalse(permission["may_claim_confirmatory_isomorphic_reconstruction"])
        self.assertEqual(case["canonical_v2"]["pipeline_status"], "selected")
        self.assertEqual(case["canonical_v2"]["selection_mode"], "retrospective_seed")
        canonical_permissions = case["canonical_v2"]["claim_permissions"]
        self.assertTrue(canonical_permissions["may_claim_selected_pipeline"])
        self.assertFalse(canonical_permissions["may_claim_automatic_decomposition"])
        self.assertFalse(canonical_permissions["may_claim_tacitness"])

    def test_oracle_taint_blocks_unconditioned_extension(self):
        result = self._evaluate(_manifest(mode="oracle", reference_access="sealed"))
        permission = result["cases"][0]["axes"]["constructive_extension"]["claim_permissions"]
        self.assertFalse(permission["may_claim_unconditioned_constructive_extension"])

    def test_missing_corpus_section_forces_ineligible(self):
        result = self._evaluate(
            _manifest(required=["answer", "formal_proof"], observed=["answer"])
        )
        case = result["cases"][0]
        self.assertFalse(case["corpus_eligibility"]["eligible"])
        self.assertEqual(
            case["axes"]["isomorphic_reconstruction"]["assessment"], "ineligible"
        )

    def test_numeric_recurrence_is_conservative_and_cross_sectional(self):
        self.assertEqual(specific_numeric_tokens("12.3 percent on 1,200 cases"), {"12.3%", "1200"})
        self.assertEqual(specific_numeric_tokens("published in 2024 with 5 folds"), set())
        result = numeric_recurrence("Improves 12.3% on 1200 cases", "Results show 12.3 percent")
        self.assertEqual(result["matched_tokens"], ["12.3%"])
        self.assertEqual(result["certificate"], "positive_recurrence")
        self.assertEqual(numeric_recurrence("Improves 12.3%", "")["certificate"], "unresolved")


if __name__ == "__main__":
    unittest.main()
