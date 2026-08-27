"""Unit-specific TRAIN mutation templates for the deterministic pilot."""

from __future__ import annotations

from dataclasses import dataclass

from .code_review_controls import controls_by_id


@dataclass(frozen=True)
class AstMutationTemplate:
    unit_id: str
    mutation_kind: str
    extension: str
    source_lines: tuple[str, ...]


def _long_parameters_python() -> tuple[str, ...]:
    return (
        "def assemble_report(account, region, period, currency, locale, owner, format, destination):",
        "    values = (account, region, period, currency)",
        "    options = (locale, owner, format, destination)",
        "    return values, options",
    )


def _e2e_only_python() -> tuple[str, ...]:
    return (
        "from playwright.sync_api import sync_playwright",
        "",
        "def test_checkout_flow():",
        "    with sync_playwright() as playwright:",
        "        browser = playwright.chromium.launch()",
        "        page = browser.new_page()",
        "        page.goto('http://localhost:8080/checkout')",
        "        assert page.title()",
        "        browser.close()",
    )


def _broad_public_python() -> tuple[str, ...]:
    return (
        "def create_record():",
        "    return {}",
        "",
        "def read_record(record):",
        "    return record",
        "",
        "def update_record(record, values):",
        "    record.update(values)",
        "    return record",
        "",
        "def delete_record(record):",
        "    record.clear()",
        "",
        "def list_records(records):",
        "    return list(records)",
    )


_controls = controls_by_id()

AST_MUTATION_TEMPLATES = (
    AstMutationTemplate(
        unit_id="code-review:llama8b:k0:n0",
        mutation_kind="three_deep_conditional",
        extension="go",
        source_lines=_controls["pcr901"].planted_source,
    ),
    AstMutationTemplate(
        unit_id="code-review:llama8b:k18:n0",
        mutation_kind="eight_parameter_function",
        extension="py",
        source_lines=_long_parameters_python(),
    ),
    AstMutationTemplate(
        unit_id="code-review:llama8b:k38:n0",
        mutation_kind="e2e_without_lower_layer",
        extension="py",
        source_lines=_e2e_only_python(),
    ),
    AstMutationTemplate(
        unit_id="code-review:llama8b:k92:n0",
        mutation_kind="broad_public_python_surface",
        extension="py",
        source_lines=_broad_public_python(),
    ),
    AstMutationTemplate(
        unit_id="pcr901",
        mutation_kind="three_deep_conditional",
        extension="go",
        source_lines=_controls["pcr901"].planted_source,
    ),
    AstMutationTemplate(
        unit_id="pcr902",
        mutation_kind="swallowed_error",
        extension="go",
        source_lines=_controls["pcr902"].planted_source,
    ),
)


def mutation_templates_by_id() -> dict[str, AstMutationTemplate]:
    return {template.unit_id: template for template in AST_MUTATION_TEMPLATES}

