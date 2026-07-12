"""Focused invariants for the additive blind reconstruction lane."""

from __future__ import annotations

import json
import pathlib
import stat
import sys
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from blind_reconstruction_v2 import (  # noqa: E402
    CandidatePolicyError,
    IntegrityError,
    audit_candidate_source,
    prepare_run,
    run_candidate,
    verify_prepared,
)
from split_ops_v2 import SplitScopedOps  # noqa: E402


class SplitScopedOpsTests(unittest.TestCase):
    def test_retrieval_is_train_only_and_self_excluding(self):
        corpus = {
            "train_0001": "alpha beta shared",
            "train_0002": "alpha beta other",
            "train_0003": "zeta only",
        }
        ops = SplitScopedOps(corpus, {"base", "retrieval"}).for_item("train_0001")
        hits = ops.retrieve_similar(corpus["train_0001"], k=10)
        self.assertNotIn("train_0001", [key for _, key in hits])
        self.assertEqual({key for _, key in hits}, {"train_0002", "train_0003"})
        # Passing a different exclude_id cannot cancel mandatory self exclusion.
        hits = ops.retrieve_similar(corpus["train_0001"], k=10,
                                   exclude_id="train_0002")
        self.assertEqual([key for _, key in hits], ["train_0003"])
        with self.assertRaises(KeyError):
            SplitScopedOps(corpus, {"retrieval"}).for_item("heldout_0001")

    def test_unallowed_operations_are_absent(self):
        ops = SplitScopedOps({"train_0001": "text"}, {"base"}).for_item("train_0001")
        self.assertEqual(ops.normalize("a   b"), "a b")
        with self.assertRaises(AttributeError):
            ops.retrieve_similar("text")
        with self.assertRaises(AttributeError):
            ops.parse_math("x=1")

    def test_capability_view_uses_v2_direction_preserving_number_check(self):
        ops = SplitScopedOps({"train_0001": "text"}, {"capability"}).for_item("train_0001")
        rows = ops.number_consistency("The value rose from 100 to 50, a 50% increase.")
        delta = [row for row in rows if row.get("kind") == "delta_pct"]
        self.assertEqual(len(delta), 1)
        self.assertFalse(delta[0]["direction_consistent"])
        self.assertFalse(delta[0]["consistent"])


class BlindHarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.items = self.root / "items.json"
        rows = [
            {"datapoint_id": f"d{i:05d}", "ctext": f"unlabeled document {i}"}
            for i in range(8)
        ]
        self.items.write_text(json.dumps(rows))
        self.contract = self.root / "contract.json"
        self.contract.write_text(json.dumps({
            "construct_definition": "Checks whether the calculation is executable.",
            "boundary_notes": "Not stylistic quality.",
            "cf_probes": [{
                "text_pos": "2 + 2 = 4",
                "text_neg": "2 + 2 = 5",
                "why": "d00007 illustrates that the contrast is arithmetic.",
                "corpus_phenomenon": "Verified against d00007 and d00006.",
                "probe_type": "genuine_contrast",
                "channel": "CODE",
            }],
            "discrimination_checks": {"min_std": 0.05},
        }))
        self.fields = self.root / "fields.jsonl"
        with self.fields.open("w") as handle:
            for i in range(8):
                handle.write(json.dumps({
                    "channel": "field", "aspect_id": "a1__claim",
                    "datapoint_id": f"d{i:05d}", "raw": f"claim {i}",
                }) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _prepare(self) -> pathlib.Path:
        out = self.root / "run"
        bundle, _ = prepare_run(
            out_dir=out, task="math", aspect_id="a1", items_path=self.items,
            contract_path=self.contract, train_count=5, split_seed=7,
            capabilities={"base", "retrieval"}, allowed_fields={"claim"},
            fields_path=self.fields,
            field_prompts={"claim": "extract the main claim"},
        )
        return bundle

    def test_bundle_exposes_only_opaque_unlabeled_train_interface(self):
        bundle_path = self._prepare()
        bundle = json.loads(bundle_path.read_text())
        encoded = bundle_path.read_text().lower()
        self.assertEqual(len(bundle["train_items"]), 5)
        self.assertEqual([x["item_key"] for x in bundle["train_items"]],
                         [f"train_{i:04d}" for i in range(1, 6)])
        self.assertNotIn("d00007", encoded)
        self.assertNotIn("corpus_phenomenon", encoded)
        self.assertFalse(bundle["interface"]["judge_values_available"])
        self.assertFalse(bundle["interface"]["heldout_identifiers_available"])
        self.assertEqual(set(bundle["allowed"]["fields"]), {"claim"})
        self.assertEqual(bundle["allowed"]["fields"]["claim"]["prompt"],
                         "extract the main claim")
        mode = stat.S_IMODE(bundle_path.stat().st_mode)
        self.assertEqual(mode & stat.S_IWUSR, 0)

    def test_label_bearing_contract_is_rejected_not_silently_projected(self):
        contaminated = json.loads(self.contract.read_text())
        contaminated["cf_probes"][0]["corpus_phenomenon"] = (
            "d00007 had judgement=1 and d00006 had judge: 0"
        )
        self.contract.write_text(json.dumps(contaminated))
        with self.assertRaisesRegex(ValueError, "forbidden outcome labels"):
            self._prepare()

    def test_label_bearing_cached_field_is_rejected(self):
        rows = []
        for i in range(8):
            rows.append(json.dumps({
                "channel": "field", "aspect_id": "a1__claim",
                "datapoint_id": f"d{i:05d}", "raw": "judge=1",
            }))
        self.fields.write_text("\n".join(rows) + "\n")
        with self.assertRaisesRegex(ValueError, "bundle violates blind policy"):
            self._prepare()

    def test_candidate_runs_in_fresh_process_with_label_free_feedback(self):
        bundle = self._prepare()
        candidate = self.root / "candidate.py"
        candidate.write_text(
            "LLM_FIELDS = {'claim': 'extract the main claim'}\n"
            "def score(text, extracted, ops):\n"
            "    neighbors = ops.retrieve_similar(text, k=2)\n"
            "    return min(1.0, len(extracted.get('claim', '')) / 20 + 0.1 * len(neighbors))\n"
        )
        result_path, manifest_path = run_candidate(
            bundle_path=bundle, candidate_path=candidate, timeout_per_item=2,
            process_timeout=20,
        )
        result = json.loads(result_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(result["n_items"], 5)
        self.assertEqual(result["n_scoreable"], 5)
        self.assertNotIn("rho", result)
        self.assertNotIn("judge", result)
        self.assertTrue(manifest["execution"]["fresh_process"])
        self.assertFalse(manifest["execution"]["os_security_boundary"])

    def test_prepared_hash_detects_tampering(self):
        bundle = self._prepare()
        verify_prepared(bundle)
        bundle.chmod(0o644)
        bundle.write_text(bundle.read_text() + " ")
        with self.assertRaises(IntegrityError):
            verify_prepared(bundle)

    def test_candidate_policy_blocks_direct_data_access(self):
        candidate = self.root / "escape.py"
        candidate.write_text(
            "import os\n"
            "def score(text, extracted, ops):\n"
            "    return float(bool(open('/tmp/labels').read()))\n"
        )
        with self.assertRaises(CandidatePolicyError):
            audit_candidate_source(candidate)

    def test_undeclared_prompt_field_fails_closed(self):
        bundle = self._prepare()
        candidate = self.root / "candidate.py"
        candidate.write_text(
            "LLM_FIELDS = {'not_allowed': 'extract a hidden field'}\n"
            "def score(text, extracted, ops):\n"
            "    return 0.5\n"
        )
        with self.assertRaises(RuntimeError):
            run_candidate(bundle_path=bundle, candidate_path=candidate,
                          timeout_per_item=2, process_timeout=20)

    def test_changed_prompt_cannot_reuse_cached_field(self):
        bundle = self._prepare()
        candidate = self.root / "candidate.py"
        candidate.write_text(
            "LLM_FIELDS = {'claim': 'a different extraction prompt'}\n"
            "def score(text, extracted, ops):\n"
            "    return 0.5\n"
        )
        with self.assertRaisesRegex(RuntimeError, "does not match frozen"):
            run_candidate(bundle_path=bundle, candidate_path=candidate,
                          timeout_per_item=2, process_timeout=20)


if __name__ == "__main__":
    unittest.main()
