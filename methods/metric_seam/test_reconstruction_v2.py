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
            verified_reference_disagreement=True,
            verifier_certificate="reports/sympy_replay.json",
        )
        self.assertEqual(classify(evidence), Outcome.CONSTRUCTIVE_EXTENSION)

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
