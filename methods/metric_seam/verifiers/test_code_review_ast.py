from __future__ import annotations

import json

from methods.metric_seam.run_code_review_ast_train import (
    DEFAULT_SELECTION,
    DEFAULT_TRAIN,
    validate_train_binding,
    validate_selection_binding,
)
from methods.metric_seam.verifiers.code_review_ast import (
    ALL_AST_UNIT_SPECS,
    verify_conditional_nesting,
    verify_control_nesting,
    verify_maintainability_smells,
    verify_python_visibility_boundary,
    verify_swallowed_go_error,
    verify_test_layer_balance,
)
from methods.metric_seam.verifiers.code_review_ast_mutations import (
    mutation_templates_by_id,
)
from methods.metric_seam.verifiers.code_review_mutations import (
    build_train_violation_pair,
)
from methods.metric_seam.verifiers.diff_lines import validate_verdict_addresses


def _new_file(path: str, source: str) -> str:
    lines = source.rstrip("\n").splitlines()
    body = "".join(f"+{line}\n" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}"
    )


def test_real_units_bind_exactly_to_frozen_selection() -> None:
    validate_selection_binding(json.loads(DEFAULT_SELECTION.read_text()))


def test_train_runner_binds_frozen_train_bytes_not_filename(tmp_path) -> None:
    validate_train_binding(DEFAULT_TRAIN)
    renamed = tmp_path / "compiler_train.json"
    renamed.write_text("[]\n", encoding="utf-8")
    try:
        validate_train_binding(renamed)
    except ValueError as exc:
        assert "frozen compiler_train bytes" in str(exc)
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("unfrozen TRAIN bytes were accepted")


def test_every_frozen_mutation_triggers_its_structured_verifier() -> None:
    base = _new_file("README.txt", "ordinary source placeholder")
    templates = mutation_templates_by_id()
    for spec in ALL_AST_UNIT_SPECS:
        template = templates[spec.unit_id]
        pair = build_train_violation_pair(
            base,
            item_key="train-fixture",
            unit_id=spec.unit_id,
            mutation_kind=template.mutation_kind,
            source_lines=template.source_lines,
            extension=template.extension,
        )
        verdict = spec.verifier(pair.planted_violated)
        assert verdict.applies is True, spec.unit_id
        assert verdict.violated is True, spec.unit_id
        assert verdict.witnesses, spec.unit_id
        validate_verdict_addresses(
            pair.planted_violated, verdict, require_added=True
        )


def test_control_flow_satisfied_verdict_has_scope_witness() -> None:
    diff = _new_file(
        "src/route.py",
        "def route(enabled):\n"
        "    if enabled:\n"
        "        return 1\n"
        "    return 0\n",
    )
    for verifier in (verify_control_nesting, verify_conditional_nesting):
        verdict = verifier(diff)
        assert verdict.applies and not verdict.violated
        assert verdict.witnesses


def test_long_parameter_relation_has_binary_boundary_and_scope_witness() -> None:
    satisfied = _new_file(
        "src/format.py",
        "def format_record(record):\n"
        "    value = record.get('value')\n"
        "    if value is None:\n"
        "        return ''\n"
        "    return str(value)\n",
    )
    verdict = verify_maintainability_smells(satisfied)
    assert verdict.applies and not verdict.violated and verdict.witnesses

    violated = _new_file(
        "src/report.py",
        "def report(a, b, c, d, e, f, g, h):\n    return a\n",
    )
    verdict = verify_maintainability_smells(violated)
    assert verdict.applies and verdict.violated and verdict.witnesses


def test_visibility_boundary_accepts_explicit_all_and_witnesses_scope() -> None:
    diff = _new_file(
        "src/api.py",
        "__all__ = ['create', 'read', 'update', 'delete']\n"
        "def create(): return {}\n"
        "def read(value): return value\n"
        "def update(value): return value\n"
        "def delete(value): return None\n",
    )
    verdict = verify_python_visibility_boundary(diff)
    assert verdict.applies and not verdict.violated and verdict.witnesses


def test_test_layer_balance_distinguishes_unit_plus_e2e_from_e2e_only() -> None:
    e2e = _new_file(
        "tests/test_checkout.py",
        "from playwright.sync_api import sync_playwright\n"
        "def test_checkout():\n    assert sync_playwright\n",
    )
    assert verify_test_layer_balance(e2e).violated

    unit = _new_file(
        "tests/unit/test_price.py",
        "def test_price():\n    assert 2 + 2 == 4\n",
    )
    combined = e2e + unit
    verdict = verify_test_layer_balance(combined)
    assert verdict.applies and not verdict.violated and verdict.witnesses


def test_go_error_return_is_satisfied_and_witnessed_not_swallowed() -> None:
    diff = _new_file(
        "internal/load.go",
        "package internal\n"
        "func load() error {\n"
        " err := read()\n"
        " if err != nil {\n"
        "  return err\n"
        " }\n"
        " return nil\n"
        "}\n",
    )
    verdict = verify_swallowed_go_error(diff)
    assert verdict.applies and not verdict.violated and verdict.witnesses
