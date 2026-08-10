"""Counterexample tests for the additive language-aware scope graph v3."""

from methods.metric_seam.hybrids.ops_code_scope_v3 import (
    CodeScopeOpsV3,
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


def _file(path: str, added: str) -> dict:
    graph = declaration_use_graph(_diff(path, added))
    return next(row for row in graph["files"] if row["path"] == path)


def test_fact_only_surface_and_morphemes_remain_generic():
    assert split_identifier_morphemes("HTTPResponse2xx_retry_count") == [
        "HTTP",
        "Response",
        "2",
        "xx",
        "retry",
        "count",
    ]
    assert not hasattr(CodeScopeOpsV3, "score")
    graph = CodeScopeOpsV3.declaration_use_graph(_diff("x.py", "clear_name = 1"))
    assert graph["schema"].endswith(".v3")
    assert graph["semantic_rules"][
        "python_member_assignment_targets_are_not_bindings"
    ] is True


def test_python_attribute_and_subscript_targets_are_uses_not_bindings():
    file_graph = _file(
        "src/client.py",
        """def assign(self, timeout, items):
    self.timeout = timeout
    items[0] = timeout
    return self.timeout""",
    )
    declarations = {row["name"]: row for row in file_graph["declarations"]}
    assert set(declarations) == {"assign", "items", "self", "timeout"}
    assert len(declarations["self"]["binding_sites"]) == 1
    assert len(declarations["timeout"]["binding_sites"]) == 1
    timeout_members = [
        row
        for row in file_graph["uses"]
        if row["name"] == "timeout" and row["role"] == "member_reference"
    ]
    assert len(timeout_members) == 2
    assert all(row["resolution"] == "member_unresolved" for row in timeout_members)


def test_python_named_redefinitions_collide_but_lexical_rebinding_coalesces():
    file_graph = _file(
        "src/rebind.py",
        """item = 1
item = 2
def work():
    return 1
def work():
    return 2
class Box:
    pass
class Box:
    pass
def consume(value):
    value = 3
    return value""",
    )
    by_name: dict[str, list[dict]] = {}
    for row in file_graph["declarations"]:
        by_name.setdefault(row["name"], []).append(row)
    assert len(by_name["item"]) == 1
    assert len(by_name["item"][0]["binding_sites"]) == 2
    assert len(by_name["work"]) == 2
    assert len(by_name["Box"]) == 2
    assert len(by_name["value"]) == 1
    assert len(by_name["value"][0]["binding_sites"]) == 2
    collision_names = {row["name"] for row in file_graph["same_scope_collisions"]}
    assert collision_names == {"Box", "work"}


def test_python_function_lookup_skips_class_namespace():
    file_graph = _file(
        "src/names.py",
        """status = "module"
class Service:
    status = "class"
    def read(self):
        return status""",
    )
    use = next(
        row
        for row in file_graph["uses"]
        if row["name"] == "status" and row["fragment_line"] == 5
    )
    assert use["resolved_declaration_ids"] == ["src/names.py::status"]
    method_scope = next(row for row in file_graph["scopes"] if row["name"] == "read")
    assert use["scope_id"] == method_scope["scope_id"]


def test_javascript_let_uses_block_scope_while_var_uses_function_scope():
    file_graph = _file(
        "src/scope.js",
        """function read() {
  let value = 1;
  {
    let value = 2;
    var hoisted = value;
  }
  return value + hoisted;
}""",
    )
    values = [row for row in file_graph["declarations"] if row["name"] == "value"]
    assert {row["scope_kind"] for row in values} == {"block", "function"}
    hoisted = next(row for row in file_graph["declarations"] if row["name"] == "hoisted")
    assert hoisted["scope_kind"] == "function"
    assert not any(
        row["name"] == "value" for row in file_graph["same_scope_collisions"]
    )
    value_uses = [row for row in file_graph["uses"] if row["name"] == "value"]
    assert len(value_uses) == 2
    assert all(row["resolution"] == "resolved" for row in value_uses)
    assert value_uses[0]["resolved_declaration_ids"] != value_uses[1][
        "resolved_declaration_ids"
    ]
    assert any(row["name"] == "value" for row in file_graph["ancestor_shadowing"])


def test_typescript_const_block_scope_and_var_function_scope():
    file_graph = _file(
        "src/scope.ts",
        """function read(input: number) {
  const count = input;
  if (count > 0) {
    const count = 2;
    var visibleAfter = count;
  }
  return count + visibleAfter;
}""",
    )
    counts = [row for row in file_graph["declarations"] if row["name"] == "count"]
    assert {row["scope_kind"] for row in counts} == {"block", "function"}
    visible = next(
        row for row in file_graph["declarations"] if row["name"] == "visibleAfter"
    )
    assert visible["scope_kind"] == "function"
    assert all(row["name"] != "number" for row in file_graph["uses"])
    assert not any(
        row["resolution"] == "ambiguous_same_scope"
        for row in file_graph["uses"]
        if row["name"] == "count"
    )


def test_go_var_short_var_parameters_and_type_filtering():
    file_graph = _file(
        "src/scope.go",
        """package scope
var globalCount int
func read(first, second int) int {
    var localCount int = first
    shortCount := localCount
    return shortCount + second
}""",
    )
    declarations = {row["name"]: row for row in file_graph["declarations"]}
    assert {
        "first",
        "globalCount",
        "localCount",
        "read",
        "second",
        "shortCount",
    }.issubset(declarations)
    assert declarations["first"]["kind"] == "parameter"
    assert declarations["globalCount"]["binding_forms"] == ["var_declaration"]
    assert declarations["shortCount"]["binding_forms"] == [
        "short_var_declaration"
    ]
    assert all(row["name"] != "int" for row in file_graph["uses"])
    assert all(
        row["resolution"] == "resolved"
        for row in file_graph["uses"]
        if row["name"] in {"first", "localCount", "second", "shortCount"}
    )


def test_go_short_declaration_reuses_existing_parameter_binding():
    file_graph = _file(
        "src/reuse.go",
        """package reuse
func read(value int) int {
    value, extra := value + 1, 2
    return value + extra
}""",
    )
    values = [row for row in file_graph["declarations"] if row["name"] == "value"]
    assert len(values) == 1
    assert len(values[0]["binding_sites"]) == 2
    assert set(values[0]["binding_forms"]) == {
        "parameter_declaration",
        "short_var_declaration",
    }
    assert not any(
        row["name"] == "value" for row in file_graph["same_scope_collisions"]
    )


def test_java_nested_block_resolution_remains_language_local():
    file_graph = _file(
        "src/Service.java",
        """class Service {
  int read(int input) {
    int outerValue = input;
    {
      int innerValue = outerValue;
    }
    return outerValue;
  }
}""",
    )
    declarations = {row["name"]: row for row in file_graph["declarations"]}
    assert declarations["outerValue"]["scope_kind"] == "method"
    assert declarations["innerValue"]["scope_kind"] == "block"
    outer_uses = [row for row in file_graph["uses"] if row["name"] == "outerValue"]
    assert len(outer_uses) == 2
    assert all(row["resolution"] == "resolved" for row in outer_uses)


def test_java_enhanced_for_variable_binds_in_loop_scope():
    file_graph = _file(
        "src/Service.java",
        """class Service {
  void read(java.util.List<String> values) {
    for (String value : values) {
      consume(value);
    }
  }
}""",
    )
    value = next(row for row in file_graph["declarations"] if row["name"] == "value")
    assert value["scope_kind"] == "block"
    assert value["binding_forms"] == ["enhanced_for_binding"]
    use = next(row for row in file_graph["uses"] if row["name"] == "value")
    assert use["resolution"] == "resolved"
    assert use["resolved_declaration_ids"] == [value["declaration_id"]]


def test_profile_preserves_fragment_and_parse_limitations():
    text = _diff("README.md", "new prose") + "\n[...]\n+orphaned_tail()"
    graph = declaration_use_graph(text)
    assert graph["truncated_input"] is True
    assert graph["orphan_fragment_count"] == 1
    assert graph["unsupported_files"] == ["README.md"]
    assert "added diff lines only" in graph["analysis_unit"]
    assert any("fragment-local" in limitation for limitation in graph["limitations"])
