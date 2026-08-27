"""Tests for the additive, fact-only scope graph capability."""

from methods.metric_seam.hybrids.ops_code_scope_v2 import (
    CodeScopeOpsV2,
    declaration_use_graph,
    split_identifier_morphemes,
)


def _diff(path: str, added: str) -> str:
    lines = added.splitlines()
    body = "\n".join(f"+{line}" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}\n"
    )


def _file(graph: dict, path: str) -> dict:
    return next(row for row in graph["files"] if row["path"] == path)


def test_identifier_morphemes_preserve_acronyms_words_and_digits():
    assert split_identifier_morphemes("HTTPResponse2xx_retry_count") == [
        "HTTP",
        "Response",
        "2",
        "xx",
        "retry",
        "count",
    ]


def test_qualifiers_include_enclosing_class_and_function_scopes():
    text = _diff(
        "src/accounts.py",
        """class Savings:
    def apply_interest(self, annual_rate):
        adjusted_balance = annual_rate
        return adjusted_balance

class Checking:
    def apply_interest(self, annual_rate):
        adjusted_balance = annual_rate
        return adjusted_balance""",
    )
    graph = declaration_use_graph(text)
    declarations = _file(graph, "src/accounts.py")["declarations"]
    qualified = {row["qualified_name"] for row in declarations}
    assert (
        "src/accounts.py::class:Savings::function:apply_interest::adjusted_balance"
        in qualified
    )
    assert (
        "src/accounts.py::class:Checking::function:apply_interest::adjusted_balance"
        in qualified
    )
    # Same method-local surfaces in different class scopes are not collisions.
    assert _file(graph, "src/accounts.py")["same_scope_collisions"] == []


def test_resolution_edges_and_ancestor_shadowing_are_scope_local():
    text = _diff(
        "src/retry.py",
        """retry_count = 3
def schedule(retry_count):
    remaining_attempts = retry_count
    return remaining_attempts""",
    )
    file_graph = _file(declaration_use_graph(text), "src/retry.py")
    declarations = {row["qualified_name"]: row for row in file_graph["declarations"]}
    parameter = "src/retry.py::function:schedule::retry_count"
    local = "src/retry.py::function:schedule::remaining_attempts"
    assert declarations[parameter]["resolved_use_count"] == 1
    assert declarations[local]["resolved_use_count"] == 1
    shadow = next(row for row in file_graph["ancestor_shadowing"] if row["name"] == "retry_count")
    assert shadow["declaration_id"] == parameter
    assert shadow["ancestor_declaration_ids"] == ["src/retry.py::retry_count"]


def test_javascript_same_scope_duplicate_declarations_are_exposed_not_judged():
    text = _diff(
        "src/config.js",
        """function configure() {
  let requestTimeout = 10;
  let requestTimeout = 20;
  return requestTimeout;
}""",
    )
    file_graph = _file(declaration_use_graph(text), "src/config.js")
    collision = next(
        row for row in file_graph["same_scope_collisions"] if row["name"] == "requestTimeout"
    )
    assert collision["relation"] == "same_scope_multiple_declarations"
    assert collision["harmfulness_unresolved"] is True
    assert len(collision["declaration_ids"]) == 2


def test_member_references_are_not_falsely_resolved_as_lexical_names():
    text = _diff(
        "src/client.py",
        """def submit(client, payload):
    response = client.send(payload)
    return response.status""",
    )
    uses = _file(declaration_use_graph(text), "src/client.py")["uses"]
    send = next(row for row in uses if row["name"] == "send")
    status = next(row for row in uses if row["name"] == "status")
    assert send["resolution"] == "member_unresolved"
    assert status["resolution"] == "member_unresolved"


def test_profile_records_truncation_and_unsupported_files():
    text = _diff("README.md", "new prose") + "\n[...]\n+orphaned_tail()"
    graph = CodeScopeOpsV2.declaration_use_graph(text)
    assert graph["truncated_input"] is True
    assert graph["orphan_fragment_count"] == 1
    assert graph["unsupported_files"] == ["README.md"]
    assert not hasattr(CodeScopeOpsV2, "score")


def test_supported_typed_language_grammars_emit_scopes_declarations_and_edges():
    cases = {
        "src/Service.java": (
            "class Service { void configure(int timeout) { "
            "int remaining = timeout; } }"
        ),
        "src/service.go": (
            "package service\n"
            "func configure(timeout int) int { remaining := timeout; return remaining }"
        ),
        "src/service.ts": (
            "function configure(timeout: number) { "
            "const remaining = timeout; return remaining; }"
        ),
    }
    for path, added in cases.items():
        file_graph = _file(declaration_use_graph(_diff(path, added)), path)
        assert file_graph["parse_has_error"] is False
        assert len(file_graph["scopes"]) >= 2
        assert len(file_graph["declarations"]) >= 3
        assert file_graph["declaration_use_edges"]
