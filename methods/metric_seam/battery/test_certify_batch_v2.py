"""Regression tests for immutable reconstruction-v2 batch certification."""

from __future__ import annotations

import hashlib
import json
import pathlib
import stat
import sys
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from certify_batch_v2 import (  # noqa: E402
    IntegrityError,
    ManifestError,
    OutputExistsError,
    benjamini_hochberg,
    certify_batch,
)


class FrozenBatchFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.artifact_index = 0

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def artifact(self, payload, prefix: str = "artifact") -> dict[str, str]:
        self.artifact_index += 1
        relpath = f"frozen/{prefix}_{self.artifact_index:03d}.json"
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        path.write_bytes(content)
        return {"path": relpath, "sha256": hashlib.sha256(content).hexdigest()}

    @staticmethod
    def score_vectors(n: int = 10):
        reference = {f"item_{i:02d}": i / (n - 1) for i in range(n)}
        candidate = dict(reference)
        h0 = {f"item_{i:02d}": (n - 1 - i) / (n - 1) for i in range(n)}
        return candidate, h0, reference

    def entry(
        self,
        entry_id: str,
        *,
        n: int = 10,
        paired_n: int | None = None,
        adversary: str = "ACCEPT",
        contract_eligible: bool = True,
        include_g1: bool = False,
    ) -> dict:
        candidate, h0, reference = self.score_vectors(n)
        if paired_n is not None:
            keep = set(sorted(candidate)[:paired_n])
            candidate = {key: value for key, value in candidate.items() if key in keep}
            h0 = {key: value for key, value in h0.items() if key in keep}
        row = {
            "entry_id": entry_id,
            "criterion_id": entry_id.split("::", 1)[0],
            "relation_id": entry_id.split("::", 1)[-1],
            "candidate_scores": self.artifact({"scores": candidate}, "candidate"),
            "h0_scores": self.artifact({"scores": h0}, "h0"),
            "frozen_llm_reference": self.artifact(
                {"scores": reference}, "frozen_llm_reference"
            ),
            "contract_result": self.artifact(
                {
                    "eligible": contract_eligible,
                    "verdict": "PASS" if contract_eligible else "FAIL",
                },
                "contract",
            ),
            "adversary_verdict": self.artifact(
                {"verdict": adversary}, "adversary"
            ),
        }
        if include_g1:
            # G1 is deliberately its own comparison/FDR family.
            row["g1_baseline"] = self.artifact({"scores": h0}, "g1")
        return row

    def manifest(self, entries: list[dict], batch_id: str = "frozen_batch_001") -> pathlib.Path:
        raw = {
            "schema_version": "metric_seam.certification_batch.v2",
            "batch_id": batch_id,
            "frozen": True,
            "analysis": {
                "alpha": 0.05,
                "minimum_effect": 0.01,
                "g1_minimum_effect": 0.01,
                "coverage_min": 0.90,
                "min_pairs": 3,
                "permutation_samples": 199,
                "bootstrap_samples": 100,
                "bootstrap_confidence": 0.95,
                "seed": 20260712,
            },
            "entries": entries,
        }
        path = self.root / "batches" / f"{batch_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
        return path

    def run_batch(self, manifest: pathlib.Path, stem: str = "report"):
        output = self.root / "reports" / f"{stem}.json"
        return certify_batch(manifest, output, repo_root=self.root)


class MultiplicityTests(FrozenBatchFixture):
    def test_bh_adjustment_is_monotone_and_not_raw_bootstrap_support(self):
        adjusted = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.20})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.20)

    def test_primary_and_g1_are_separate_fdr_families(self):
        manifest = self.manifest(
            [
                self.entry("math__a144::witness", include_g1=True),
                self.entry("code__a104::test_linkage"),
            ]
        )
        _, _, report = self.run_batch(manifest)
        multiplicity = report["multiplicity"]
        self.assertEqual(multiplicity["primary_family"]["n_valid_p_values"], 2)
        self.assertEqual(multiplicity["g1_family"]["n_valid_p_values"], 1)
        first = report["entries"][0]
        primary = first["reference_reconstruction_vs_h0"]
        g1 = first["g1_reference_reconstruction"]
        self.assertGreaterEqual(primary["bh_q_value"], primary["permutation"]["p_value"])
        self.assertGreaterEqual(g1["bh_q_value"], g1["permutation"]["p_value"])
        self.assertEqual(g1["status"], "evaluated_separate_fdr_family")


class EligibilityTests(FrozenBatchFixture):
    def test_reference_availability_and_conditional_coverage_are_distinct(self):
        row = self.entry("math__a144::witness", n=5)
        row["heldout_count"] = 10
        manifest = self.manifest([row])
        _, _, report = self.run_batch(manifest)
        eligibility = report["entries"][0]["eligibility"]
        self.assertEqual(eligibility["paired_coverage"], 1.0)
        self.assertEqual(eligibility["reference_availability_over_heldout"], 0.5)
        self.assertEqual(eligibility["candidate_coverage_over_heldout"], 0.5)
        self.assertEqual(
            eligibility["paired_coverage_denominator"],
            "frozen_reference_available_items",
        )

    def test_below_ninety_percent_paired_coverage_is_not_inferentially_eligible(self):
        manifest = self.manifest(
            [self.entry("science__a1::number_consistency", n=10, paired_n=8)]
        )
        _, _, report = self.run_batch(manifest)
        row = report["entries"][0]
        self.assertFalse(row["eligibility"]["eligible"])
        self.assertEqual(row["eligibility"]["paired_coverage"], 0.8)
        self.assertIn(
            "coverage_below_predeclared_minimum", row["eligibility"]["reasons"]
        )
        self.assertIsNone(row["reference_reconstruction_vs_h0"])
        self.assertEqual(report["multiplicity"]["primary_family"]["n_valid_p_values"], 0)

    def test_a31_style_adversary_reject_cannot_be_promoted(self):
        manifest = self.manifest(
            [self.entry("patents__a31::claim_use", adversary="REJECT")]
        )
        _, _, report = self.run_batch(manifest)
        row = report["entries"][0]
        self.assertFalse(row["eligibility"]["eligible"])
        self.assertFalse(row["eligibility"]["adversary_accept"])
        self.assertIn("adversary_not_accept", row["eligibility"]["reasons"])
        self.assertIsNone(row["reference_reconstruction_vs_h0"])

    def test_candidate_and_h0_must_have_identical_item_ids(self):
        row = self.entry("math__a1::scope")
        h0_path = self.root / row["h0_scores"]["path"]
        h0 = json.loads(h0_path.read_text())
        h0["scores"].pop(sorted(h0["scores"])[0])
        content = (json.dumps(h0, sort_keys=True) + "\n").encode("utf-8")
        h0_path.write_bytes(content)
        row["h0_scores"]["sha256"] = hashlib.sha256(content).hexdigest()
        manifest = self.manifest([row])
        with self.assertRaisesRegex(ManifestError, "identical item IDs"):
            self.run_batch(manifest)


class IntegrityAndImmutabilityTests(FrozenBatchFixture):
    def test_invalid_artifact_hash_fails_before_writing(self):
        row = self.entry("code__a30::exception_path")
        row["candidate_scores"]["sha256"] = "0" * 64
        manifest = self.manifest([row])
        output = self.root / "reports" / "bad_hash.json"
        with self.assertRaisesRegex(IntegrityError, "SHA-256 mismatch"):
            certify_batch(manifest, output, repo_root=self.root)
        self.assertFalse(output.exists())
        self.assertFalse((output.parent / "bad_hash.snapshot.json").exists())

    def test_frozen_batch_queue_is_unchanged_and_outputs_never_overwritten(self):
        manifest = self.manifest([self.entry("math__a156::substitution")])
        manifest_before = manifest.read_bytes()
        output, snapshot, _ = self.run_batch(manifest, "immutable")
        report_before = output.read_bytes()
        snapshot_before = snapshot.read_bytes()
        self.assertEqual(manifest.read_bytes(), manifest_before)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
        self.assertEqual(stat.S_IMODE(snapshot.stat().st_mode), 0o444)
        with self.assertRaises(OutputExistsError):
            certify_batch(manifest, output, snapshot_path=snapshot, repo_root=self.root)
        self.assertEqual(output.read_bytes(), report_before)
        self.assertEqual(snapshot.read_bytes(), snapshot_before)
        self.assertEqual(manifest.read_bytes(), manifest_before)


if __name__ == "__main__":
    unittest.main()
