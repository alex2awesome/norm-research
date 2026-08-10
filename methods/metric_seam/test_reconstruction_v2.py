"""Dependency-free unit tests for the reconstruction-v2 outcome vocabulary."""

import unittest

try:
    from .reconstruction_v2 import (
        AxisEvidence,
        DiscoveryMode,
        Outcome,
        PipelineStatus,
        ReconstructionEvidence,
        RelationMatchVerdict,
        SelectionMode,
        Status,
        SubrelationEvidence,
        build_decomposition,
        claim_permissions,
        classify,
        decomposition_readout,
        validate_record,
    )
except ImportError:  # direct-file execution
    from reconstruction_v2 import (  # type: ignore[no-redef]
        AxisEvidence,
        DiscoveryMode,
        Outcome,
        PipelineStatus,
        ReconstructionEvidence,
        RelationMatchVerdict,
        SelectionMode,
        Status,
        SubrelationEvidence,
        build_decomposition,
        claim_permissions,
        classify,
        decomposition_readout,
        validate_record,
    )


def ev(**overrides):
    base = dict(
        criterion_id="math__a144",
        relation_id="explicit_witness",
        discovery_mode=DiscoveryMode.REPLAY,
        articulability=AxisEvidence(Status.FAIL),
        verifiability=AxisEvidence(Status.FAIL),
        hybrid=AxisEvidence(Status.FAIL),
        reference_isomorphism=AxisEvidence(Status.PASS),
        construct_fidelity=AxisEvidence(Status.PASS),
    )
    base.update(overrides)
    return ReconstructionEvidence(**base)


class ClassificationTests(unittest.TestCase):
    def test_axes_stay_separate(self):
        self.assertEqual(
            classify(
                ev(
                    articulability=AxisEvidence(Status.PASS),
                    verifiability=AxisEvidence(Status.FAIL),
                )
            ),
            Outcome.ARTICULABLE_ONLY,
        )
        self.assertEqual(
            classify(
                ev(
                    articulability=AxisEvidence(Status.FAIL),
                    verifiability=AxisEvidence(Status.PASS),
                )
            ),
            Outcome.VERIFIABLE_ONLY,
        )

    def test_constructive_extension_requires_certificate(self):
        with self.assertRaises(ValueError):
            ev(
                verifiability=AxisEvidence(Status.PASS),
                reference_isomorphism=AxisEvidence(Status.FAIL),
                verified_reference_disagreement=True,
            )
        evidence = ev(
            verifiability=AxisEvidence(Status.PASS),
            reference_isomorphism=AxisEvidence(Status.FAIL),
            input_fidelity=AxisEvidence(Status.PASS),
            program_fidelity=AxisEvidence(Status.PASS),
            reference_instrument_fidelity=AxisEvidence(Status.PASS),
            verified_reference_disagreement=True,
            verifier_certificate="reports/sympy_replay.json",
        )
        self.assertEqual(classify(evidence), Outcome.CONSTRUCTIVE_EXTENSION)

    def test_constructive_extension_requires_real_same_instrument_disagreement(self):
        required = {
            "input_fidelity": AxisEvidence(Status.PASS),
            "program_fidelity": AxisEvidence(Status.PASS),
            "reference_instrument_fidelity": AxisEvidence(Status.PASS),
        }
        with self.assertRaisesRegex(ValueError, "reference reconstruction FAIL"):
            ev(
                verifiability=AxisEvidence(Status.PASS),
                verified_reference_disagreement=True,
                verifier_certificate="reports/sympy_replay.json",
                **required,
            )
        for missing in required:
            axes = dict(required)
            axes[missing] = AxisEvidence(Status.UNAVAILABLE)
            with self.assertRaisesRegex(ValueError, missing.replace("_", "-")):
                ev(
                    verifiability=AxisEvidence(Status.PASS),
                    reference_isomorphism=AxisEvidence(Status.FAIL),
                    verified_reference_disagreement=True,
                    verifier_certificate="reports/sympy_replay.json",
                    **axes,
                )

    def test_reference_agreement_cannot_rescue_bad_proxy(self):
        evidence = ev(
            articulability=AxisEvidence(Status.PASS),
            verifiability=AxisEvidence(Status.PASS),
            construct_fidelity=AxisEvidence(Status.FAIL),
        )
        self.assertEqual(classify(evidence), Outcome.PROXY_MISMATCH)

    def test_hybrid_complement(self):
        evidence = ev(hybrid=AxisEvidence(Status.PASS))
        self.assertEqual(classify(evidence), Outcome.HYBRID_COMPLEMENT)

    def test_mock_provenance_does_not_disqualify_selected_seed(self):
        evidence = ev(
            discovery_mode=DiscoveryMode.MOCK,
            pipeline_status=PipelineStatus.SELECTED,
            selection_mode=SelectionMode.RETROSPECTIVE_SEED,
            verifiability=AxisEvidence(Status.PASS),
        )
        self.assertEqual(classify(evidence), Outcome.VERIFIABLE_ONLY)
        permissions = claim_permissions(evidence)
        self.assertTrue(permissions["may_claim_selected_pipeline"])
        self.assertTrue(permissions["may_claim_code_verifiability"])
        self.assertFalse(permissions["may_claim_automatic_decomposition"])

    def test_failure_never_licenses_tacitness(self):
        permissions = claim_permissions(
            ev(
                reference_isomorphism=AxisEvidence(Status.FAIL),
                construct_fidelity=AxisEvidence(Status.PASS),
            )
        )
        self.assertFalse(permissions["may_claim_tacitness"])
        self.assertIn("bounded_non_discovery", permissions["failure_interpretation"])

    def test_positive_channel_witness_survives_unavailable_isomorphism(self):
        evidence = ev(
            verifiability=AxisEvidence(Status.PASS),
            reference_isomorphism=AxisEvidence(Status.UNAVAILABLE),
        )
        self.assertEqual(classify(evidence), Outcome.VERIFIABLE_ONLY)
        self.assertTrue(claim_permissions(evidence)["may_claim_code_verifiability"])

        both = ev(
            articulability=AxisEvidence(Status.PASS),
            verifiability=AxisEvidence(Status.PASS),
            reference_isomorphism=AxisEvidence(Status.UNAVAILABLE),
        )
        self.assertEqual(classify(both), Outcome.DUAL_IMPLEMENTATION)

    def test_reconstruction_agreement_does_not_alone_license_isomorphism(self):
        evidence = ev(
            articulability=AxisEvidence(Status.PASS),
            reference_isomorphism=AxisEvidence(Status.PASS),
        )
        permissions = claim_permissions(evidence)
        self.assertTrue(permissions["may_claim_reconstruction_agreement"])
        self.assertFalse(permissions["may_claim_isomorphism"])
        self.assertFalse(permissions["may_claim_isomorphic_reconstruction"])
        self.assertFalse(evidence.isomorphism_established)

        complete = ev(
            articulability=AxisEvidence(Status.PASS),
            reference_isomorphism=AxisEvidence(Status.PASS),
            input_fidelity=AxisEvidence(Status.PASS),
            program_fidelity=AxisEvidence(Status.PASS),
            reference_instrument_fidelity=AxisEvidence(Status.PASS),
        )
        complete_permissions = claim_permissions(complete)
        self.assertTrue(complete_permissions["may_claim_reconstruction_agreement"])
        self.assertTrue(complete_permissions["may_claim_isomorphism"])
        self.assertTrue(complete.isomorphism_established)

    def test_record_parser_accepts_canonical_reconstruction_name(self):
        record = ev().as_dict()
        self.assertIn("reference_reconstruction", record)
        self.assertNotIn("reference_isomorphism", record)
        parsed = validate_record(record)
        self.assertEqual(parsed.reference_reconstruction.status, Status.PASS)
        self.assertFalse(parsed.isomorphism_established)

    def test_nested_serialization_uses_canonical_reconstruction_name(self):
        row = SubrelationEvidence(
            evidence=ev(relation_id="presence"),
            construct_relation="whether a witness is present",
            program_relation="typed AST witness enumeration",
            relation_match=RelationMatchVerdict.CODE_NATIVE,
        )
        row_record = row.as_dict()
        self.assertIn("reference_reconstruction", row_record["evidence"])
        self.assertNotIn("reference_isomorphism", row_record["evidence"])

        decomposition = build_decomposition("math__a144", [row])
        decomposition_record = decomposition.as_dict()
        nested = decomposition_record["subrelations"][0]["evidence"]
        self.assertIn("reference_reconstruction", nested)
        self.assertNotIn("reference_isomorphism", nested)

    def test_subrelations_do_not_silently_collapse_to_parent(self):
        presence = SubrelationEvidence(
            evidence=ev(
                relation_id="presence",
                verifiability=AxisEvidence(Status.PASS),
            ),
            construct_relation="whether a witness is present",
            program_relation="typed AST witness enumeration",
            relation_match=RelationMatchVerdict.CODE_NATIVE,
        )
        function = SubrelationEvidence(
            evidence=ev(
                relation_id="function",
                articulability=AxisEvidence(Status.PASS),
                reference_isomorphism=AxisEvidence(Status.PASS),
            ),
            construct_relation="whether the witness performs the requested function",
            program_relation="prompt judgement over witness and surrounding proof",
            relation_match=RelationMatchVerdict.PROMPT_NATIVE,
        )
        readout = decomposition_readout(
            build_decomposition("math__a144", [presence, function])
        )
        self.assertIsNone(readout["parent_outcome"])
        self.assertEqual(
            [row["outcome"] for row in readout["subrelations"]],
            [Outcome.VERIFIABLE_ONLY.value, Outcome.ARTICULABLE_ONLY.value],
        )

    def test_decomposition_rejects_duplicate_or_cross_criterion_relations(self):
        row = SubrelationEvidence(
            evidence=ev(relation_id="presence"),
            construct_relation="presence",
            program_relation="token lookup",
            relation_match=RelationMatchVerdict.CAPABILITY_MISMATCH,
        )
        with self.assertRaises(ValueError):
            build_decomposition("math__a144", [row, row])
        with self.assertRaises(ValueError):
            build_decomposition(
                "creative_writing__a333",
                [row],
            )


if __name__ == "__main__":
    unittest.main()
