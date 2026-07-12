"""Regression tests for the channel-faithful contract checker.

Run directly or with pytest:
  python3 methods/metric_seam/battery/test_contract_check_isomorphic.py
"""

from __future__ import annotations

import copy
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from contract_check_isomorphic import (  # noqa: E402
    CheckConfig,
    ContractIntegrityError,
    ContractSchemaError,
    SCHEMA_VERSION,
    build_probe_ops,
    canonical_json_sha256,
    check_contract,
    main,
    text_sha256,
    validate_contract,
)


def contract() -> dict:
    return {
        "construct_definition": "Separate the operative relation from a mention.",
        "cf_probes": [
            {
                "text_pos": "C_POS_0 operative",
                "text_neg": "C_NEG_0 mention",
                "why": "The positive executes the relation; the negative only names it.",
                "channel": "CODE",
            },
            {
                "text_pos": "C_POS_1 operative",
                "text_neg": "C_NEG_1 mention",
                "why": "A parser can distinguish these forms.",
                "channel": "CODE",
            },
            {
                "text_pos": "L_POS_2 semantic",
                "text_neg": "L_NEG_2 semantic",
                "why": "Prompt reconstruction supplies the semantic-use assessment.",
                "channel": "L",
            },
            {
                "text_pos": "L_POS_3 semantic",
                "text_neg": "L_NEG_3 semantic",
                "why": "Prompt reconstruction supplies the semantic-use assessment.",
                "channel": "L",
            },
        ],
        "discrimination_checks": {"min_std": 0.10, "max_frac_at_mode": 0.50},
        "boundary_notes": "CODE is verifiability; L is prompt-based articulability.",
    }


def extraction_payload(c: dict, *, available: bool = True) -> dict:
    csha = canonical_json_sha256(c)
    rows = []
    for index in (2, 3):
        probe = c["cf_probes"][index]
        row = {
            "index": index,
            "available": available,
            "text_pos_sha256": text_sha256(probe["text_pos"]),
            "text_neg_sha256": text_sha256(probe["text_neg"]),
        }
        if available:
            row.update({"pos": {"semantic_use": 0.9}, "neg": {"semantic_use": 0.1}})
        else:
            row["unavailable_reason"] = "extractor did not return a usable field"
        rows.append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": csha,
        "extractor_manifest_sha256": "a" * 64,
        "probes": rows,
    }


def faithful_score(text, extracted, _ops):
    if "semantic_use" in extracted:
        return extracted["semantic_use"]
    return 0.8 if "C_POS" in text else 0.2


class ContractCheckerTests(unittest.TestCase):
    def setUp(self):
        self.contract = contract()
        self.contract_sha = canonical_json_sha256(self.contract)
        self.config = CheckConfig(min_margin=0.05, train_min_items=4)

    def check(self, score=faithful_score, **kwargs):
        return check_contract(
            self.contract,
            expected_contract_sha256=self.contract_sha,
            score=score,
            config=self.config,
            **kwargs,
        )

    def test_canonical_contract_hash_ignores_json_whitespace_and_order(self):
        rendered = json.dumps(self.contract, indent=7, sort_keys=False)
        reordered = json.loads(rendered)
        self.assertEqual(self.contract_sha, canonical_json_sha256(reordered))
        with self.assertRaises(ContractIntegrityError):
            check_contract(
                self.contract,
                expected_contract_sha256="0" * 64,
                score=faithful_score,
                config=self.config,
            )

    def test_missing_l_is_abstention_and_code_still_passes(self):
        result = self.check()
        self.assertEqual("PASS", result.code_gate.status)
        self.assertTrue(result.code_gate.passed)
        self.assertEqual("ABSTAIN", result.hybrid_gate.status)
        self.assertIsNone(result.hybrid_gate.passed)
        self.assertEqual(2, result.hybrid_gate.n_abstained)
        # The conditional code result is visible, but is not mislabeled a full hybrid PASS.
        self.assertTrue(result.hybrid_gate.conditional_passed)

    def test_unavailable_l_rows_are_not_invoked_or_charged_to_code(self):
        payload = extraction_payload(self.contract, available=False)
        result = self.check(
            extraction_payload=payload,
            expected_extraction_sha256=canonical_json_sha256(payload),
        )
        self.assertEqual("PASS", result.code_gate.status)
        self.assertEqual("ABSTAIN", result.hybrid_gate.status)
        self.assertEqual(2, result.hybrid_gate.n_eligible)

    def test_frozen_l_fields_enable_a_distinct_hybrid_gate(self):
        payload = extraction_payload(self.contract)
        result = self.check(
            extraction_payload=payload,
            expected_extraction_sha256=canonical_json_sha256(payload),
        )
        self.assertEqual("PASS", result.code_gate.status)
        self.assertEqual("PASS", result.hybrid_gate.status)
        self.assertEqual(4, result.hybrid_gate.n_separated)
        self.assertEqual(1.0, result.hybrid_gate.l_coverage)

    def test_extractions_are_bound_to_artifact_contract_and_exact_text(self):
        payload = extraction_payload(self.contract)
        with self.assertRaises(ContractIntegrityError):
            self.check(
                extraction_payload=payload,
                expected_extraction_sha256="b" * 64,
            )

        changed = copy.deepcopy(payload)
        changed["probes"][0]["text_pos_sha256"] = "0" * 64
        with self.assertRaisesRegex(ContractIntegrityError, "positive text hash mismatch"):
            self.check(
                extraction_payload=changed,
                expected_extraction_sha256=canonical_json_sha256(changed),
            )

        wrong_contract = copy.deepcopy(payload)
        wrong_contract["contract_sha256"] = "f" * 64
        with self.assertRaisesRegex(ContractIntegrityError, "different contract"):
            self.check(
                extraction_payload=wrong_contract,
                expected_extraction_sha256=canonical_json_sha256(wrong_contract),
            )

    def test_nonfinite_range_and_margin_are_hard_gates(self):
        for score in (
            lambda _t, _e, _o: float("nan"),
            lambda _t, _e, _o: 1.2,
        ):
            result = self.check(score=score)
            self.assertEqual("FAIL", result.code_gate.status)
            self.assertGreater(result.code_gate.n_invalid, 0)

        def below_margin(text, _extracted, _ops):
            return 0.51 if "POS" in text else 0.50

        result = self.check(score=below_margin)
        self.assertEqual("FAIL", result.code_gate.status)
        self.assertEqual(0, result.code_gate.n_invalid)
        self.assertEqual(0, result.code_gate.n_separated)

    def test_empty_or_nonempty_probe_mode_bypass_is_rejected(self):
        def fingerprint(text, extracted, _ops):
            if extracted:
                return 0.9 if "POS" in text else 0.1
            return 0.5

        result = self.check(score=fingerprint)
        self.assertEqual("FAIL", result.code_gate.status)
        self.assertGreater(result.code_gate.n_invalid, 0)
        self.assertTrue(any(row.mode_detected for row in result.probes if row.channel == "CODE"))

    def test_blind_contract_rejects_embedded_judge_assignments(self):
        for field, text in (
            ("why", "A leaked example has judge=.4."),
            ("corpus_phenomenon", "The source row says judgement=0."),
            ("why", "Observed score: 0.75 in the target set."),
        ):
            contaminated = copy.deepcopy(self.contract)
            contaminated["cf_probes"][0][field] = text
            with self.assertRaisesRegex(ContractSchemaError, "label-bearing"):
                validate_contract(contaminated)
        # Ordinary discussion of score behavior is not a false positive.
        clean = copy.deepcopy(self.contract)
        clean["cf_probes"][0]["why"] = "The score should increase for the operative relation."
        validate_contract(clean)

    def test_unlabeled_discrimination_gate_uses_no_reference_values(self):
        cases = [
            {"text": "C_POS one", "extracted": {}},
            {"text": "C_POS two", "extracted": {}},
            {"text": "C_NEG one", "extracted": {}},
            {"text": "C_NEG two", "extracted": {}},
        ]
        result = self.check(discrimination_cases=cases)
        self.assertEqual("PASS", result.discrimination_gate.status)
        self.assertEqual(4, result.discrimination_gate.n_items)
        self.assertEqual(4, result.discrimination_gate.n_scored)

    def test_blind_safe_probe_capability_bridge_supports_code_without_retrieval(self):
        ops, allowed = build_probe_ops({"base"})
        self.assertEqual(("base",), allowed)

        def uses_deep_interface(text, _extracted, bound_ops):
            normalized = bound_ops.normalize(text)
            return 0.8 if "C_POS" in normalized else 0.2

        result = self.check(score=uses_deep_interface, ops=ops)
        self.assertEqual("PASS", result.code_gate.status)
        with self.assertRaisesRegex(ValueError, "Retrieval requires a TRAIN-scoped bundle"):
            build_probe_ops({"retrieval"})

    def test_cli_passes_explicit_capability_ops_to_deep_candidate(self):
        all_code = copy.deepcopy(self.contract)
        for probe in all_code["cf_probes"]:
            probe["channel"] = "CODE"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            contract_path = tmp / "contract.json"
            candidate_path = tmp / "candidate.py"
            contract_path.write_text(json.dumps(all_code))
            candidate_path.write_text(
                "def score(text, extracted, ops):\n"
                "    normalized = ops.normalize(text)\n"
                "    return 0.8 if 'POS' in normalized else 0.2\n"
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(
                    [
                        "--contract", str(contract_path),
                        "--expected-contract-sha256", canonical_json_sha256(all_code),
                        "--candidate", str(candidate_path),
                        "--capability", "base",
                    ]
                )
            self.assertEqual(0, exit_code)
            self.assertEqual(["base"], json.loads(output.getvalue())["probe_capabilities"])


if __name__ == "__main__":
    unittest.main()
