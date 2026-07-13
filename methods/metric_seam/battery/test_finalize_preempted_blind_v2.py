"""Tests for construct-fidelity preemption before reference access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    from .finalize_preempted_blind_v2 import finalize_preempted
except ImportError:
    from finalize_preempted_blind_v2 import finalize_preempted  # type: ignore[no-redef]


def _adversary(*, suite_pass: bool = False, freeze_verified: bool = True) -> dict:
    return {
        "suite_pass": suite_pass,
        "freeze_verified": freeze_verified,
        "case_counts": {"pair_cases": 34, "range_cases": 10},
        "metrics": {
            "pair_pass_rate": 30 / 34,
            "range_pass_rate": 7 / 10,
            "minimum_pair_category_pass_rate": 0.0,
        },
        "conditions": {"pair_pass_rate": True, "pair_category_floor": False},
    }


def _bind_candidate(adversary: dict, candidate: Path) -> dict:
    adversary["candidate"] = {"sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}
    return adversary


class FinalizePreemptedBlindTests(unittest.TestCase):
    def test_failed_frozen_adversary_preempts_reference(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.py"
            prepare = root / "prepare.json"
            result = root / "adversary.json"
            candidate.write_text("def score(*args): return 0.5\n")
            prepare.write_text("{}\n")
            result.write_text(json.dumps(_bind_candidate(_adversary(), candidate)))

            payload = finalize_preempted(
                criterion_id="math__a216",
                relation_id="equation_reference_graph",
                candidate=candidate,
                prepare_manifest=prepare,
                adversary_result=result,
            )

            self.assertEqual(payload["outcome"], "proxy_mismatch")
            self.assertFalse(payload["heldout_reference_opened"])
            self.assertEqual(
                payload["preemption_reason"],
                "construct_fidelity_failed_before_reference_access",
            )
            self.assertFalse(payload["claim_permissions"]["may_claim_tacitness"])
            self.assertFalse(payload["claim_permissions"]["may_claim_code_verifiability"])
            self.assertEqual(
                payload["artifact_sha256"]["candidate"],
                hashlib.sha256(candidate.read_bytes()).hexdigest(),
            )

    def test_requires_failed_suite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("candidate.py", "prepare.json", "result.json")]
            paths[0].write_text("pass\n")
            paths[1].write_text("{}\n")
            paths[2].write_text(json.dumps(_bind_candidate(
                _adversary(suite_pass=True), paths[0]
            )))
            with self.assertRaisesRegex(ValueError, "requires a failed adversary suite"):
                finalize_preempted(
                    criterion_id="math__a216",
                    relation_id="equation_reference_graph",
                    candidate=paths[0],
                    prepare_manifest=paths[1],
                    adversary_result=paths[2],
                )

    def test_requires_verified_freeze(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("candidate.py", "prepare.json", "result.json")]
            paths[0].write_text("pass\n")
            paths[1].write_text("{}\n")
            paths[2].write_text(json.dumps(_bind_candidate(
                _adversary(freeze_verified=False), paths[0]
            )))
            with self.assertRaisesRegex(ValueError, "freeze was not verified"):
                finalize_preempted(
                    criterion_id="math__a216",
                    relation_id="equation_reference_graph",
                    candidate=paths[0],
                    prepare_manifest=paths[1],
                    adversary_result=paths[2],
                )

    def test_requires_construct_category_failure(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("candidate.py", "prepare.json", "result.json")]
            paths[0].write_text("pass\n")
            paths[1].write_text("{}\n")
            adversary = _bind_candidate(_adversary(), paths[0])
            adversary["conditions"]["pair_category_floor"] = True
            adversary["conditions"]["range_pass_rate"] = False
            paths[2].write_text(json.dumps(adversary))
            with self.assertRaisesRegex(ValueError, "construct-category floor"):
                finalize_preempted(
                    criterion_id="math__a216",
                    relation_id="equation_reference_graph",
                    candidate=paths[0],
                    prepare_manifest=paths[1],
                    adversary_result=paths[2],
                )

    def test_requires_exact_candidate_executed_by_adversary(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("candidate.py", "prepare.json", "result.json")]
            paths[0].write_text("pass\n")
            paths[1].write_text("{}\n")
            adversary = _bind_candidate(_adversary(), paths[0])
            paths[2].write_text(json.dumps(adversary))
            paths[0].write_text("raise RuntimeError\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                finalize_preempted(
                    criterion_id="math__a216",
                    relation_id="equation_reference_graph",
                    candidate=paths[0],
                    prepare_manifest=paths[1],
                    adversary_result=paths[2],
                )


if __name__ == "__main__":
    unittest.main()
