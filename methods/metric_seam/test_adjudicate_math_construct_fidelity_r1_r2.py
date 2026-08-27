import hashlib
import json
import unittest
from collections import Counter

from methods.metric_seam.adjudicate_math_construct_fidelity_r1_r2 import (
    DEFAULT_SEED_MAP,
    REPO_ROOT,
    build,
    validate,
)


class MathConstructFidelityR1R2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed_path = REPO_ROOT / DEFAULT_SEED_MAP
        cls.seed = json.loads(cls.seed_path.read_text())
        cls.artifact = build(DEFAULT_SEED_MAP)

    def test_complete_cell_and_candidate_binding(self):
        rows = self.artifact["rows"]
        self.assertEqual(60, len(rows))
        self.assertEqual(60, len({r["cell_id"] for r in rows}))
        self.assertEqual(Counter({"R1": 30, "R2": 30}), Counter(r["level"] for r in rows))
        source = {
            r["cell_id"]: r
            for r in self.seed["rows"]
            if r["level"] in {"R1", "R2"}
        }
        for row in rows:
            selected = source[row["cell_id"]]["selected_seed"]
            if selected is None:
                self.assertIsNone(row["candidate"])
                self.assertEqual("no_candidate_bounded_non_discovery", row["verdict"])
                continue
            candidate = row["candidate"]
            self.assertEqual(selected["aspect_id"], candidate["aspect_id"])
            self.assertEqual(selected["source_path"], candidate["source_path"])
            source_path = REPO_ROOT / candidate["source_path"]
            self.assertEqual(
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
                candidate["program_sha256"],
            )

    def test_expected_static_outcomes(self):
        rows = self.artifact["rows"]
        by_level = {
            level: Counter(r["verdict"] for r in rows if r["level"] == level)
            for level in ("R1", "R2")
        }
        self.assertEqual(
            Counter(partial=13, mismatch=5, no_candidate_bounded_non_discovery=12),
            by_level["R1"],
        )
        self.assertEqual(
            Counter(partial=6, mismatch=4, no_candidate_bounded_non_discovery=20),
            by_level["R2"],
        )
        self.assertFalse(any(r["verdict"] == "exact" for r in rows))
        self.assertEqual(19, sum(r["eligible_for_relation_local_execution"] for r in rows))

    def test_llm_fields_are_disclosed_but_not_credited_as_code(self):
        for row in self.artifact["rows"]:
            candidate = row["candidate"]
            if candidate is None:
                continue
            implemented = " ".join(row["implemented_relations"]).lower()
            for field_name in candidate["llm_fields_excluded_from_implemented_relations"]:
                self.assertNotIn(field_name.lower(), implemented)

    def test_negative_results_are_bounded(self):
        for row in self.artifact["rows"]:
            if row["verdict"] in {"mismatch", "no_candidate_bounded_non_discovery"}:
                self.assertIn("bounded non-discovery", row["interpretation"])

    def test_public_validator_accepts_built_artifact(self):
        validate(self.artifact, self.seed)


if __name__ == "__main__":
    unittest.main()
