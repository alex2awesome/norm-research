"""Focused integrity and denominator tests for sealed blind evaluation v2."""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
import stat
import sys
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from blind_reconstruction_v2 import IntegrityError, prepare_run, run_candidate  # noqa: E402
from evaluate_blind_v2 import _comparison, evaluate_sealed  # noqa: E402


class SealedEvaluationTests(unittest.TestCase):
    def setUp(self):
        # Keeping fixtures under the repository makes repo-relative manifest paths a
        # testable invariant rather than weakening production path policy for tests.
        self.tmp = tempfile.TemporaryDirectory(prefix="sealed_eval_test_", dir=ROOT)
        self.work = pathlib.Path(self.tmp.name)
        self.items = self.work / "items.json"
        rows = [
            {
                "datapoint_id": f"d{i:05d}",
                "ctext": "document " + ("x" * (i + 1)),
            }
            for i in range(10)
        ]
        self.items.write_text(json.dumps(rows), encoding="utf-8")
        self.contract = self.work / "contract.json"
        self.contract.write_text(json.dumps({
            "construct_definition": "Rewards an explicit repeated marker.",
            "boundary_notes": "The prompt/LLM and code channels target the same text.",
            "cf_probes": [
                {
                    "text_pos": "marker xxxxx",
                    "text_neg": "marker x",
                    "why": "The positive contains more executed evidence.",
                    "probe_type": "genuine_contrast",
                    "channel": "L",
                }
            ],
            "discrimination_checks": {"min_std": 0.01},
        }), encoding="utf-8")
        self.run_dir = self.work / "run"
        self.bundle, _ = prepare_run(
            out_dir=self.run_dir,
            task="math",
            aspect_id="a_test",
            items_path=self.items,
            contract_path=self.contract,
            train_count=6,
            split_seed=7,
            capabilities={"base", "retrieval"},
        )
        self.candidate = self.run_dir / "candidate.py"
        self.candidate.write_text(
            "def score(text, extracted, ops):\n"
            "    hits = ops.retrieve_similar(text, k=2)\n"
            "    assert all(key.startswith('train_') for _, key in hits)\n"
            "    return min(1.0, text.count('x') / 12.0)\n",
            encoding="utf-8",
        )
        self.execution_result, self.execution_manifest = run_candidate(
            bundle_path=self.bundle,
            candidate_path=self.candidate,
            execution_name="agentic_r1",
            timeout_per_item=2,
            process_timeout=30,
        )
        self.reference = self.work / "results.jsonl"
        with self.reference.open("w", encoding="utf-8") as handle:
            for i in range(10):
                for channel, score in (("pass1", i), ("pass2", min(10, i + 1))):
                    handle.write(json.dumps({
                        "channel": channel,
                        "aspect_id": "a_test",
                        "datapoint_id": f"d{i:05d}",
                        "raw": f"SCORE: {score}",
                        "score": score,
                    }) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _evaluate(self):
        return evaluate_sealed(
            bundle_path=self.bundle,
            execution_manifest_path=self.execution_manifest,
            out_dir=self.work / "sealed_eval",
            repo_root=ROOT,
            reference_path=self.reference,
            include_historical=False,
            timeout_per_item=2,
            process_timeout=30,
        )

    def test_exact_complement_is_executed_before_reference_and_frozen(self):
        metrics_path, manifest_path = self._evaluate()
        metrics = json.loads(metrics_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        candidate_scores = json.loads(
            (metrics_path.parent / "candidate_scores.json").read_text()
        )["score_map"]

        ids = [f"d{i:05d}" for i in range(10)]
        random.Random(7).shuffle(ids)
        expected_heldout = set(ids[6:])
        self.assertEqual(set(candidate_scores), expected_heldout)
        self.assertEqual(metrics["candidate"]["heldout_count"], 4)
        self.assertEqual(metrics["candidate"]["reference_available_count"], 4)
        self.assertEqual(metrics["candidate"]["common_count"], 4)
        self.assertEqual(
            metrics["candidate"]["candidate_coverage_conditional_on_reference"], 1.0
        )
        self.assertLessEqual(
            manifest["evaluation_order"]["candidate_execution_completed_at"],
            manifest["evaluation_order"]["llm_reference_load_started_at"],
        )
        self.assertTrue(manifest["policy"]["reference_values_sent_to_candidate"] is False)
        for path in metrics_path.parent.iterdir():
            mode = stat.S_IMODE(path.stat().st_mode)
            self.assertEqual(mode & stat.S_IWUSR, 0, path.name)

        receipt_hash = (metrics_path.parent / "sealed_manifest.sha256").read_text().split()[0]
        self.assertEqual(
            receipt_hash,
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )
        challenge = json.loads(
            (metrics_path.parent / "channel_challenge.json").read_text()
        )
        self.assertFalse(challenge["is_contract_pass"])
        self.assertFalse(challenge["frozen_channel_labels_changed"])
        self.assertEqual(challenge["all_frozen_channels"], ["L"])

    def test_candidate_change_after_label_free_run_fails_closed(self):
        self.candidate.write_text(self.candidate.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(IntegrityError, "candidate changed"):
            self._evaluate()

    def test_prior_execution_result_change_fails_closed(self):
        self.execution_result.chmod(0o644)
        self.execution_result.write_text(self.execution_result.read_text() + " ")
        with self.assertRaisesRegex(IntegrityError, "execution result changed"):
            self._evaluate()

    def test_coverage_denominators_do_not_conflate_reference_support(self):
        report = _comparison(
            {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4},
            {"a": 0.0, "b": 0.5},
            heldout_count=10,
        )
        self.assertEqual(report["heldout_count"], 10)
        self.assertEqual(report["reference_available_count"], 2)
        self.assertEqual(report["common_count"], 2)
        self.assertEqual(report["candidate_coverage_all_heldout"], 0.4)
        self.assertEqual(report["candidate_coverage_conditional_on_reference"], 1.0)
        self.assertEqual(report["reference_availability_all_heldout"], 0.2)


if __name__ == "__main__":
    unittest.main()
