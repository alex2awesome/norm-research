"""Known-answer and prior-bypass tests for enforced metric DAGs."""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dag_schema_enforced import (  # noqa: E402
    DagTypeError,
    UndeclaredInputError,
    execute,
    validate,
)


def node(node_id, ntype, level, inputs, output_type, fn, **extra):
    return {
        "id": node_id,
        "ntype": ntype,
        "level": level,
        "inputs": inputs,
        "output_type": output_type,
        "fn": fn,
        **extra,
    }


class EnforcedDagTests(unittest.TestCase):
    def test_code_node_cannot_read_undeclared_llm_or_raw_text(self):
        for hidden_key in ("llm", "text", "extracted"):
            program = {
                "nodes": [
                    node(
                        "out",
                        "A",
                        3,
                        {},
                        "score",
                        lambda inputs, key=hidden_key: inputs[key],
                    )
                ],
                "out": "out",
            }
            # Default-argument closures are intentionally rejected as smuggling, so use a
            # clean one-argument function for the actual runtime regression below.
            self.assertTrue(any("exactly one" in err for err in validate(program)))

        def reads_old_full_context(inputs):
            return inputs["extracted"]

        program = {
            "nodes": [node("out", "A", 3, {}, "score", reads_old_full_context)],
            "out": "out",
        }
        self.assertEqual([], validate(program))
        with self.assertRaises(UndeclaredInputError):
            execute(program, text="document", llm_fields={"score": 0.8})

    def test_get_with_default_does_not_hide_an_undeclared_read(self):
        def bypass(inputs):
            return inputs.get("text", "secret")

        program = {"nodes": [node("out", "A", 3, {}, "score", bypass)], "out": "out"}
        with self.assertRaises(UndeclaredInputError):
            execute(program, text="document")

    def test_disconnected_llm_junk_is_rejected(self):
        program = {
            "nodes": [
                node("out", "A", 3, {}, "score", lambda _inputs: 0.5),
                node(
                    "junk",
                    "T",
                    2,
                    {"field": {"source": "llm", "key": "quality", "type": "score"}},
                    "score",
                    lambda inputs: inputs["field"],
                ),
            ],
            "out": "out",
        }
        errors = validate(program)
        self.assertTrue(any("do not reach out" in err and "junk" in err for err in errors))

    def test_declared_but_unread_llm_source_does_not_create_a_seam(self):
        program = {
            "nodes": [
                node(
                    "out",
                    "A",
                    3,
                    {
                        "document": {"source": "text", "type": "text"},
                        "quality": {"source": "llm", "key": "quality", "type": "score"},
                    },
                    "score",
                    lambda inputs: 0.6 if inputs["document"] else 0.4,
                )
            ],
            "out": "out",
        }
        result = execute(program, text="nonempty", llm_fields={"quality": 0.99})
        self.assertEqual(0.6, result.output)
        self.assertEqual((), result.seam.articulability_frontier)
        self.assertEqual("C", result.trace["out"].implementation)
        self.assertNotIn("llm", result.trace["out"].taints)

    def test_actual_llm_access_derives_articulability_frontier(self):
        program = {
            "nodes": [
                node(
                    "llm_use",
                    "T",
                    2,
                    {"quality": {"source": "llm", "key": "quality", "type": "score"}},
                    "score",
                    lambda inputs: inputs["quality"],
                ),
                node(
                    "out",
                    "A",
                    3,
                    {"quality": {"node": "llm_use", "type": "score"}},
                    "score",
                    lambda inputs: inputs["quality"],
                ),
            ],
            "out": "out",
        }
        result = execute(program, text="doc", llm_fields={"quality": 0.81})
        self.assertEqual(0.81, result.output)
        self.assertEqual(("llm_use",), result.seam.articulability_frontier)
        self.assertEqual((2,), result.seam.frontier_levels)
        self.assertEqual("C+LLM", result.trace["llm_use"].implementation)
        self.assertEqual("C+LLM", result.trace["out"].implementation)
        self.assertEqual(("verifiability", "articulability"), result.trace["out"].channels)

    def test_connected_but_dynamically_unused_llm_branch_does_not_change_seam(self):
        program = {
            "nodes": [
                node(
                    "llm_junk",
                    "T",
                    2,
                    {"quality": {"source": "llm", "key": "quality", "type": "score"}},
                    "score",
                    lambda inputs: inputs["quality"],
                ),
                node(
                    "out",
                    "A",
                    3,
                    {
                        "unused": {"node": "llm_junk", "type": "score"},
                        "document": {"source": "text", "type": "text"},
                    },
                    "score",
                    lambda inputs: 0.7 if inputs["document"] else 0.3,
                ),
            ],
            "out": "out",
        }
        result = execute(program, text="doc", llm_fields={"quality": 0.9})
        self.assertEqual(("out",), result.contributing_nodes)
        self.assertEqual((), result.seam.articulability_frontier)
        self.assertEqual("C", result.trace["out"].implementation)

    def test_evidence_taint_is_orthogonal_and_propagates(self):
        program = {
            "nodes": [
                node(
                    "claim_check",
                    "T",
                    2,
                    {
                        "claims": {
                            "source": "evidence",
                            "key": "claim_chart",
                            "type": "sequence",
                        }
                    },
                    "number",
                    lambda inputs: len(inputs["claims"]),
                ),
                node(
                    "out",
                    "A",
                    3,
                    {"count": {"node": "claim_check", "type": "number"}},
                    "score",
                    lambda inputs: min(1.0, inputs["count"] / 2.0),
                ),
            ],
            "out": "out",
        }
        result = execute(
            program,
            text="claim",
            evidence={"claim_chart": ["element 1", "element 2"]},
        )
        self.assertEqual(1.0, result.output)
        self.assertEqual(("claim_check",), result.seam.evidence_frontier)
        self.assertEqual(0, result.seam.n_articulability_tainted)
        self.assertTrue(result.trace["out"].evidence_tainted)
        self.assertEqual("C", result.trace["out"].implementation)

    def test_static_types_levels_and_author_declared_impl_are_rejected(self):
        bad_type = {
            "nodes": [
                node("feature", "T", 2, {}, "mapping", lambda _inputs: {}),
                node(
                    "out",
                    "A",
                    3,
                    {"x": {"node": "feature", "type": "number"}},
                    "score",
                    lambda _inputs: 0.5,
                ),
            ],
            "out": "out",
        }
        self.assertTrue(any("outputs mapping" in err for err in validate(bad_type)))

        inversion = {
            "nodes": [
                node("verdict", "A", 3, {}, "score", lambda _inputs: 0.5),
                node(
                    "span",
                    "T",
                    1,
                    {"v": {"node": "verdict", "type": "score"}},
                    "score",
                    lambda inputs: inputs["v"],
                ),
                node(
                    "out",
                    "A",
                    3,
                    {"x": {"node": "span", "type": "score"}},
                    "score",
                    lambda inputs: inputs["x"],
                ),
            ],
            "out": "out",
        }
        self.assertTrue(any("LEVEL INVERSION" in err for err in validate(inversion)))

        declared = {
            "nodes": [
                node("out", "A", 3, {}, "score", lambda _inputs: 0.5, impl="L")
            ],
            "out": "out",
        }
        self.assertTrue(any("derive it" in err for err in validate(declared)))

    def test_runtime_output_must_be_finite_and_in_range(self):
        for bad in (float("nan"), 1.2, -0.1, "0.5"):
            def returns_bad(_inputs, value=bad):
                return value

            # Default arguments are rejected; define through a callable object with exactly
            # one visible argument for runtime type tests.
            class ReturnBad:
                def __init__(self, value):
                    self.value = value

                def __call__(self, _inputs):
                    return self.value

            program = {
                "nodes": [node("out", "A", 3, {}, "score", ReturnBad(bad))],
                "out": "out",
            }
            self.assertEqual([], validate(program))
            with self.assertRaises(DagTypeError):
                execute(program, text="doc")


if __name__ == "__main__":
    unittest.main()
