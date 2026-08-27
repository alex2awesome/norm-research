import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from methods.metric_seam.adjudicate_math_construct_fidelity_r1_r2_cross_audit import (
    A12_CAVEAT,
    A174_CAVEAT,
    A198_CAVEAT,
    CHANGE_SPECS,
    DEFAULT_OUT,
    DEFAULT_SOURCE_AUDIT,
    EXPECTED_SOURCE_AUDIT_SHA256,
    REPO_ROOT,
    build,
    validate,
)


QUALITY_A126 = (
    "TB::math-stackexchange::general::R1::parented_tree::151::"
    "a3d74e378f643f073486"
)
TRANSFER_A126 = (
    "TB::math-stackexchange::general::R1::parented_tree::76::"
    "84bfde44a5e0db465693"
)
POLYA_A126 = (
    "TB::math-stackexchange::general::R2::grandparent::70::"
    "8a29d84dc956b2aa6287"
)
A30_AUDIENCE = (
    "TB::math-stackexchange::general::R1::parented_tree::327::"
    "d0c2268bbef193ffa36d"
)
A168_WORD_SYMBOL = (
    "TB::math-stackexchange::general::R1::merged_tree::87::"
    "01c3ffc70f1d857bb128"
)


class MathConstructFidelityR1R2CrossAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = REPO_ROOT / DEFAULT_SOURCE_AUDIT
        cls.source_bytes = cls.source_path.read_bytes()
        cls.source = json.loads(cls.source_bytes)
        cls.artifact = build()
        cls.source_by_id = {row["cell_id"]: row for row in cls.source["rows"]}
        cls.changes_by_id = {
            change["cell_id"]: change for change in cls.artifact["changes"]
        }

    def test_frozen_source_and_complete_retrieved_coverage(self):
        self.assertEqual(
            EXPECTED_SOURCE_AUDIT_SHA256,
            hashlib.sha256(self.source_bytes).hexdigest(),
        )
        self.assertEqual(60, self.artifact["review_coverage"]["source_rows"])
        self.assertEqual(
            28, self.artifact["review_coverage"]["retrieved_candidates_reviewed"]
        )
        self.assertTrue(
            self.artifact["review_coverage"]["all_retrieved_candidates_reviewed"]
        )
        self.assertEqual(14, self.artifact["review_coverage"]["changed_rows"])
        self.assertEqual(14, self.artifact["review_coverage"]["unchanged_retrieved_rows"])
        self.assertEqual(set(CHANGE_SPECS), set(self.changes_by_id))

    def test_overlay_lists_only_real_field_changes_with_exact_guards(self):
        for cell_id, change in self.changes_by_id.items():
            source_row = self.source_by_id[cell_id]
            self.assertIsNotNone(source_row["candidate"])
            self.assertEqual(set(change["before"]), set(change["after"]))
            self.assertNotEqual(change["before"], change["after"])
            for field, value in change["before"].items():
                self.assertEqual(source_row[field], value)
            self.assertEqual(
                source_row["candidate"]["aspect_id"],
                change["candidate_guard"]["aspect_id"],
            )
            self.assertEqual(
                source_row["candidate"]["program_sha256"],
                change["candidate_guard"]["program_sha256"],
            )

    def test_presence_function_collision_is_the_only_verdict_change(self):
        verdict_changes = [
            change
            for change in self.artifact["changes"]
            if "verdict" in change["after"]
        ]
        self.assertEqual([QUALITY_A126], [change["cell_id"] for change in verdict_changes])
        change = self.changes_by_id[QUALITY_A126]
        self.assertEqual("partial", change["before"]["verdict"])
        self.assertEqual("mismatch", change["after"]["verdict"])
        self.assertEqual("none", change["after"]["scope"])
        self.assertFalse(change["after"]["eligible_for_relation_local_execution"])

        # Method exposure and explicit Pólya operations are requested
        # subrelations in these two distinct constructs, so they remain partial.
        self.assertNotIn(TRANSFER_A126, self.changes_by_id)
        self.assertNotIn(POLYA_A126, self.changes_by_id)
        self.assertEqual("partial", self.source_by_id[TRANSFER_A126]["verdict"])
        self.assertEqual("partial", self.source_by_id[POLYA_A126]["verdict"])

    def test_adversarial_retentions_are_not_silently_reclassified(self):
        # Local grounded-definition/scaffolding relations survive for a30,
        # although audience identification and prerequisite fit remain residual.
        self.assertNotIn(A30_AUDIENCE, self.changes_by_id)
        self.assertEqual("partial", self.source_by_id[A30_AUDIENCE]["verdict"])

        # Formal correctness rows retain only a narrow presentation/elision
        # subrelation; the overlay adds the a12 aggregation clamp, not correctness.
        a12_changes = [
            c for c in self.artifact["changes"]
            if c["candidate_guard"]["aspect_id"] == "a12"
        ]
        self.assertEqual(4, len(a12_changes))
        self.assertTrue(
            all(set(c["after"]) == {"polarity_aggregation_applicability_caveats"}
                for c in a12_changes)
        )

        # Axiomatic form (a78) and bounded economy anti-patterns (a72) remain
        # narrow partials; neither is promoted to whole-construct verification.
        for aspect in ("a72", "a78"):
            rows = [
                row for row in self.source["rows"]
                if row.get("candidate") and row["candidate"]["aspect_id"] == aspect
            ]
            self.assertTrue(rows)
            self.assertTrue(all(row["verdict"] == "partial" for row in rows))
            self.assertTrue(all(row["cell_id"] not in self.changes_by_id for row in rows))

    def test_dead_gated_a42_branch_is_corrected_on_all_five_rows(self):
        a42_changes = [
            c for c in self.artifact["changes"]
            if c["candidate_guard"]["aspect_id"] == "a42"
        ]
        self.assertEqual(5, len(a42_changes))
        for change in a42_changes:
            implemented = " ".join(change["after"]["implemented_relations"]).lower()
            caveats = " ".join(
                change["after"]["polarity_aggregation_applicability_caveats"]
            ).lower()
            self.assertNotIn("example marker", implemented)
            self.assertNotIn("case marker", implemented)
            self.assertIn("fixed 0.05 base", implemented)
            self.assertIn("not decision-contributing", caveats)
            self.assertNotIn("verdict", change["after"])

        source = (
            REPO_ROOT / "methods/metric_seam/hybrids/programs_math/a42_h1.py"
        ).read_text()
        self.assertIn("if has_example or has_visual:", source)
        self.assertIn("elaboration = _elaboration_score(t, ops)", source)
        self.assertIn("base = 0.05", source)

    def test_empty_field_and_interface_effects_are_disclosed(self):
        a12 = [
            c for c in self.artifact["changes"]
            if c["candidate_guard"]["aspect_id"] == "a12"
        ]
        self.assertTrue(all(A12_CAVEAT in c["after"]["polarity_aggregation_applicability_caveats"] for c in a12))

        a198 = [
            c for c in self.artifact["changes"]
            if c["candidate_guard"]["aspect_id"] == "a198"
        ]
        self.assertEqual(2, len(a198))
        self.assertTrue(all(A198_CAVEAT in c["after"]["polarity_aggregation_applicability_caveats"] for c in a198))

        a174 = [
            c for c in self.artifact["changes"]
            if c["candidate_guard"]["aspect_id"] == "a174"
        ]
        self.assertEqual(1, len(a174))
        self.assertIn(
            A174_CAVEAT,
            a174[0]["after"]["polarity_aggregation_applicability_caveats"],
        )

        a12_source = (REPO_ROOT / "methods/metric_seam/hybrids/programs_math/a12_h0.py").read_text()
        a198_source = (REPO_ROOT / "methods/metric_seam/hybrids/programs_math/a198_h1.py").read_text()
        a174_source = (REPO_ROOT / "methods/metric_seam/hybrids/programs_math/a174_h0.py").read_text()
        ops_source = (REPO_ROOT / "methods/metric_seam/hybrids/ops_math.py").read_text()
        self.assertIn("base = min(base, 0.18)", a12_source)
        self.assertIn("if isinstance(eq_stats, dict):", a198_source)
        self.assertIn("return n_display, n_inline, n_numbered, avg", ops_source)
        self.assertIn('llm_none = (hint == "")', a174_source)
        self.assertIn("bad_units = max(0, bad_units - 1)", a174_source)

    def test_depth_correction_is_relation_based_and_unique(self):
        depth_changes = [
            c for c in self.artifact["changes"]
            if "audited_depth" in c["after"]
        ]
        self.assertEqual([A168_WORD_SYMBOL], [c["cell_id"] for c in depth_changes])
        self.assertEqual(1, depth_changes[0]["before"]["audited_depth"])
        self.assertEqual(2, depth_changes[0]["after"]["audited_depth"])
        source = (
            REPO_ROOT / "methods/metric_seam/hybrids/programs_math/a168_h0.py"
        ).read_text()
        self.assertIn(r're.search(r"\\cong", ans)', source)
        self.assertIn(r're.search(r"\bmod\b", ans, re.I)', source)

    def test_counts_recompute_after_overlay(self):
        self.assertEqual(
            {"mismatch": 9, "partial": 19},
            self.artifact["before_counts"]["retrieved_verdicts"],
        )
        self.assertEqual(
            {"mismatch": 10, "partial": 18},
            self.artifact["after_counts_if_overlay_applied"]["retrieved_verdicts"],
        )
        self.assertEqual(
            {"1": 10, "2": 18},
            self.artifact["after_counts_if_overlay_applied"]["retrieved_depths"],
        )
        self.assertEqual(
            18,
            self.artifact["after_counts_if_overlay_applied"][
                "eligible_for_relation_local_execution"
            ],
        )

    def test_validator_rejects_stale_before_and_candidate_guard(self):
        stale = copy.deepcopy(self.artifact)
        stale["changes"][0]["before"][next(iter(stale["changes"][0]["before"]))] = "tampered"
        with self.assertRaises(ValueError):
            validate(stale, self.source)

        wrong_candidate = copy.deepcopy(self.artifact)
        wrong_candidate["changes"][0]["candidate_guard"]["program_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate(wrong_candidate, self.source)

    def test_builder_rejects_rebased_source_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_bytes(self.source_bytes + b"\n")
            with self.assertRaises(ValueError):
                build(path)

    def test_generated_artifact_is_reproducible_and_static(self):
        output = json.loads((REPO_ROOT / DEFAULT_OUT).read_text())
        self.assertEqual(self.artifact, output)
        self.assertFalse(output["forbidden_inputs_used"])
        self.assertFalse(output["candidate_execution_performed"])
        self.assertFalse(output["candidate_import_performed"])
        self.assertFalse(output["model_or_api_calls_performed"])
        validate(output, self.source)


if __name__ == "__main__":
    unittest.main()
