"""Adversarial regressions for callable-level DAG provenance policy v3."""

from __future__ import annotations

import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dag_schema_hardened import (  # noqa: E402
    CallableProvenanceError,
    HARDENED_POLICY_VERSION,
    NodeExecutionError,
    audit_callable,
    execute,
    validate,
)


def node(node_id, ntype, level, inputs, output_type, fn):
    return {
        "id": node_id,
        "ntype": ntype,
        "level": level,
        "inputs": inputs,
        "output_type": output_type,
        "fn": fn,
    }


_SMUGGLED = {}
_MODULE_SCORE = 0.97


def stash_llm_in_global(inputs):
    _SMUGGLED["score"] = inputs["quality"]
    return 0.1


def read_smuggled_global(inputs):
    # Read the declared dependency so both nodes are on the dynamic slice, then ignore it.
    inputs["dependency"]
    return _SMUGGLED["score"]


def read_module_state(_inputs):
    return _MODULE_SCORE


class HardenedDagTests(unittest.TestCase):
    def test_auditor_global_smuggling_reproducer_fails_closed(self):
        program = {
            "provenance_policy": HARDENED_POLICY_VERSION,
            "nodes": [
                node(
                    "stash",
                    "T",
                    2,
                    {"quality": {"source": "llm", "key": "quality", "type": "score"}},
                    "score",
                    stash_llm_in_global,
                ),
                node(
                    "out",
                    "A",
                    3,
                    {"dependency": {"node": "stash", "type": "score"}},
                    "score",
                    read_smuggled_global,
                ),
            ],
            "out": "out",
        }
        errors = validate(program)
        self.assertTrue(any("ambient global read '_SMUGGLED'" in error for error in errors))
        with self.assertRaises(CallableProvenanceError):
            execute(program, text="doc", llm_fields={"quality": 0.83})
        self.assertEqual({}, _SMUGGLED, "validation must fail before upstream side effects")

    def test_closure_and_module_state_are_rejected(self):
        hidden = 0.91

        def closure(_inputs):
            return hidden

        closure_errors = audit_callable(closure).errors
        self.assertTrue(any("nonlocal state" in error for error in closure_errors))
        self.assertTrue(
            any("ambient global read '_MODULE_SCORE'" in error for error in audit_callable(read_module_state).errors)
        )

    def test_stateful_callable_object_is_rejected(self):
        class Stateful:
            def __init__(self):
                self.hidden = 0.8

            def __call__(self, _inputs):
                return self.hidden

        audit = audit_callable(Stateful())
        self.assertFalse(audit.accepted)
        self.assertTrue(any("plain Python function" in error for error in audit.errors))

    def test_private_introspection_is_rejected(self):
        def reads_globals(_inputs):
            return reads_globals.__globals__["_MODULE_SCORE"]

        errors = audit_callable(reads_globals).errors
        self.assertTrue(any("nonlocal state" in error for error in errors))
        self.assertTrue(any("forbidden attribute access '__globals__'" in error for error in errors))

    def test_allowed_builtin_is_rebound_not_inherited_from_function_globals(self):
        def template(inputs):
            return min(1.0, len(inputs["document"]) / 10.0)

        poisoned = types.FunctionType(
            template.__code__,
            {
                "len": lambda _value: 1000,
                "min": lambda *_values: 0.99,
            },
            "poisoned",
        )
        self.assertTrue(audit_callable(poisoned).accepted)
        program = {
            "nodes": [
                node(
                    "out",
                    "A",
                    3,
                    {"document": {"source": "text", "type": "text"}},
                    "score",
                    poisoned,
                )
            ],
            "out": "out",
        }
        result = execute(program, text="abc")
        self.assertAlmostEqual(0.3, result.output)
        self.assertEqual(frozenset({"code", "document"}), result.trace["out"].taints)

    def test_mutable_declared_value_cannot_be_a_cross_node_state_channel(self):
        payload = {"values": [0.2]}

        def tries_to_mutate(inputs):
            inputs["payload"]["values"].append(inputs["quality"])
            return 0.1

        def output(inputs):
            return inputs["dependency"]

        program = {
            "nodes": [
                node(
                    "mutator",
                    "T",
                    2,
                    {
                        "payload": {
                            "source": "evidence",
                            "key": "payload",
                            "type": "mapping",
                        },
                        "quality": {
                            "source": "llm",
                            "key": "quality",
                            "type": "score",
                        },
                    },
                    "score",
                    tries_to_mutate,
                ),
                node(
                    "out",
                    "A",
                    3,
                    {"dependency": {"node": "mutator", "type": "score"}},
                    "score",
                    output,
                ),
            ],
            "out": "out",
        }
        with self.assertRaises(NodeExecutionError):
            execute(
                program,
                text="doc",
                llm_fields={"quality": 0.9},
                evidence={"payload": payload},
            )
        self.assertEqual({"values": [0.2]}, payload)

    def test_ordinary_typed_hybrid_dag_still_executes_and_traces(self):
        def use_quality(inputs):
            return inputs["quality"]

        def combine(inputs):
            return min(1.0, inputs["quality"] + len(inputs["document"]) / 100.0)

        program = {
            "provenance_policy": HARDENED_POLICY_VERSION,
            "nodes": [
                node(
                    "prompt_field",
                    "T",
                    2,
                    {"quality": {"source": "llm", "key": "quality", "type": "score"}},
                    "score",
                    use_quality,
                ),
                node(
                    "out",
                    "A",
                    3,
                    {
                        "quality": {"node": "prompt_field", "type": "score"},
                        "document": {"source": "text", "type": "text"},
                    },
                    "score",
                    combine,
                ),
            ],
            "out": "out",
        }
        self.assertEqual([], validate(program))
        result = execute(program, text="abcd", llm_fields={"quality": 0.7})
        self.assertAlmostEqual(0.74, result.output)
        self.assertEqual(("prompt_field",), result.seam.articulability_frontier)
        self.assertEqual("C+LLM", result.trace["out"].implementation)


if __name__ == "__main__":
    unittest.main()
