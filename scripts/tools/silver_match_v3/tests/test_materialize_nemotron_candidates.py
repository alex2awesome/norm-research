from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.tools.silver_match_v3.materialize_nemotron_candidates import ranking_rows


class MaterializeNemotronCandidatesTest(unittest.TestCase):
    def test_deterministic_rank_and_truth_blind_contract(self) -> None:
        norms = [
            {
                "norm_uid": "u1",
                "corpus": "c",
                "task": "humor",
                "row": 1,
                "source_group": "g1",
            }
        ]
        bank = [{"metric_id": "a1"}, {"metric_id": "a2"}, {"metric_id": "a3"}]
        rows = ranking_rows(
            np.asarray([[0.5, 0.5, -0.1]], dtype=np.float32),
            norms,
            bank,
            "bank-sha",
            2,
            "model",
            Path("/adapter"),
        )
        self.assertEqual([row["metric_id"] for row in rows[0]["candidates"]], ["a1", "a2"])
        self.assertEqual([row["rank"] for row in rows[0]["candidates"]], [1, 2])
        self.assertFalse(rows[0]["truth_fields_read"])
        self.assertNotIn("decision", rows[0])
        self.assertNotIn("metric_id", rows[0])

    def test_invalid_score_shape_fails(self) -> None:
        with self.assertRaises(ValueError):
            ranking_rows(
                np.zeros((2, 2)),
                [{"norm_uid": "u", "corpus": "c", "task": "humor", "source_group": "g"}],
                [{"metric_id": "a1"}, {"metric_id": "a2"}],
                "sha",
                2,
                "model",
                Path("/adapter"),
            )


if __name__ == "__main__":
    unittest.main()
