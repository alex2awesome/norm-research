"""Language-aware scope/declaration/use facts for added code in unified diffs.

Version 3 is additive: the frozen v2 operation remains byte-for-byte unchanged.
This implementation corrects several language-semantics counterexamples while
retaining the same research boundary.  It emits generic relation facts, never
an a407 score, and analyzes only added diff fragments rather than a repository
checkout.

The implementation is deliberately conservative about assignment targets,
type syntax, member references, and Python class namespaces.  Unsupported
binding forms, parse recovery, concatenated hunks, imports, and dynamic/runtime
bindings remain explicit limitations in every returned profile.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

try:
    from .ops_code_scope_v2 import (
        _Binding,
        _NAMED_DECLARATION_NODES,
        _PARAMETER_LIST_NODES,
        _PARAMETER_NODES,
        _Scope,
        _added_source,
        _diff_line,
        _language,
        _name_node,
        _node_key,
        _node_text,
        _parser,
        _scope_label,
        _scope_name,
        _walk,
        parse_unified_diff,
        split_identifier_morphemes,
    )
except ImportError:  # pragma: no cover - direct-module compatibility
    from ops_code_scope_v2 import (  # type: ignore[no-redef]
        _Binding,
        _NAMED_DECLARATION_NODES,
        _PARAMETER_LIST_NODES,
        _PARAMETER_NODES,
        _Scope,
        _added_source,
        _diff_line,
        _language,
        _name_node,
        _node_key,
        _node_text,
        _parser,
        _scope_label,
        _scope_name,
        _walk,
        parse_unified_diff,
        split_identifier_morphemes,
    )


SCHEMA = "metric-seam.code-scope-declaration-use-graph.v3"

_NAMED_SCOPE_NODES = {
    "python": {
        "class_definition": "class",
        "function_definition": "function",
        "lambda": "lambda",
        "list_comprehension": "comprehension",
        "set_comprehension": "comprehension",
        "dictionary_comprehension": "comprehension",
        "generator_expression": "comprehension",
    },
    "javascript": {
        "class_declaration": "class",
        "function_declaration": "function",
        "function_expression": "function",
        "generator_function_declaration": "function",
        "method_definition": "method",
        "arrow_function": "function",
    },
    "typescript": {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "function_declaration": "function",
        "function_expression": "function",
        "generator_function_declaration": "function",
        "method_definition": "method",
        "arrow_function": "function",
    },
    "java": {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "record_declaration": "record",
        "method_declaration": "method",
        "constructor_declaration": "constructor",
        "lambda_expression": "lambda",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "func_literal": "function",
    },
}

_FUNCTION_NODE_TYPES = {
    "python": {"function_definition", "lambda"},
    "javascript": {
        "function_declaration",
        "function_expression",
        "generator_function_declaration",
        "method_definition",
        "arrow_function",
    },
    "typescript": {
        "function_declaration",
        "function_expression",
        "generator_function_declaration",
        "method_definition",
        "arrow_function",
    },
    "java": {
        "method_declaration",
        "constructor_declaration",
        "lambda_expression",
    },
    "go": {"function_declaration", "method_declaration", "func_literal"},
}

_JS_CONTROL_SCOPES = {
    "catch_clause",
    "for_in_statement",
    "for_statement",
    "switch_statement",
}
_JAVA_CONTROL_SCOPES = {
    "catch_clause",
    "enhanced_for_statement",
    "for_statement",
    "switch_expression",
}
_GO_CONTROL_SCOPES = {
    "expression_switch_statement",
    "for_statement",
    "if_statement",
    "select_statement",
    "type_switch_statement",
}

_BINDING_LEAVES = {
    "identifier",
    "shorthand_property_identifier_pattern",
}
_NAMED_BINDING_LEAVES = _BINDING_LEAVES | {
    "property_identifier",
    "type_identifier",
}
_NONLEXICAL_TARGETS = {
    "attribute",
    "field_access",
    "field_expression",
    "member_expression",
    "selector_expression",
    "subscript",
    "subscript_expression",
}
_TYPE_NODES = {
    "generic_type",
    "integral_type",
    "predefined_type",
    "primitive_type",
    "scoped_type_identifier",
    "type",
    "type_annotation",
    "type_arguments",
    "type_identifier",
}
_USE_IDENTIFIER_NODES = {
    "identifier",
    "property_identifier",
    "shorthand_property_identifier",
}
_PYTHON_FUNCTIONAL_SCOPES = {"function", "lambda", "comprehension"}
_FUNCTION_SCOPES = {"function", "method", "constructor", "lambda"}


def _field_nodes(node, field: str) -> list[Any]:
    children_by_field_name = getattr(node, "children_by_field_name", None)
    if children_by_field_name is not None:
        return list(children_by_field_name(field))
    child = node.child_by_field_name(field)
    return [child] if child is not None else []


def _binding_target(node):
    for field in ("pattern", "name", "left"):
        target = node.child_by_field_name(field)
        if target is not None:
            return target
    return None


def _binding_identifiers(node, *, named_declaration: bool = False) -> list[Any]:
    """Return true lexical binding leaves, never member/subscript operands."""
    if node is None:
        return []
    leaves = _NAMED_BINDING_LEAVES if named_declaration else _BINDING_LEAVES
    if node.type in leaves:
        return [node]
    if node.type in _NONLEXICAL_TARGETS or node.type in _TYPE_NODES:
        return []
    out: list[Any] = []
    for child in node.named_children:
        out.extend(_binding_identifiers(child, named_declaration=named_declaration))
    return out


def _scope_kind(node, language: str) -> str | None:
    named = _NAMED_SCOPE_NODES.get(language, {}).get(node.type)
    if named is not None:
        return named
    parent_type = node.parent.type if node.parent is not None else None
    if language in {"javascript", "typescript"}:
        if node.type == "statement_block":
            if parent_type in _FUNCTION_NODE_TYPES[language]:
                return None
            return "block"
        if node.type in _JS_CONTROL_SCOPES:
            return "block"
    elif language == "java":
        if node.type == "block":
            if parent_type in _FUNCTION_NODE_TYPES[language]:
                return None
            return "block"
        if node.type in _JAVA_CONTROL_SCOPES:
            return "block"
    elif language == "go":
        if node.type == "block":
            if parent_type in _FUNCTION_NODE_TYPES[language]:
                return None
            return "block"
        if node.type in _GO_CONTROL_SCOPES:
            return "block"
    return None


def _nearest_function_or_file_scope(scopes: dict[str, _Scope], scope_id: str) -> str:
    current: str | None = scope_id
    while current is not None:
        scope = scopes[current]
        if scope.kind in _FUNCTION_SCOPES or scope.kind == "file":
            return current
        current = scope.parent_id
    raise AssertionError("scope graph has no file ancestor")


def _lookup_scope_ids(
    scopes: dict[str, _Scope], scope_id: str, language: str
) -> Iterable[str]:
    # Python class bodies are executable namespaces, not closure scopes for a
    # method/function defined inside them.  Continue past the class to an outer
    # function or module instead of resolving an unqualified name to class x.
    skip_python_classes = (
        language == "python" and scopes[scope_id].kind in _PYTHON_FUNCTIONAL_SCOPES
    )
    current: str | None = scope_id
    while current is not None:
        scope = scopes[current]
        if not (skip_python_classes and scope.kind == "class"):
            yield current
        current = scope.parent_id


def _groups_for_events(language: str, events: list[_Binding]) -> list[list[_Binding]]:
    """Coalesce rebinding sites without hiding true declaration collisions."""
    if language == "python":
        groups: list[list[_Binding]] = []
        lexical_group: list[_Binding] | None = None
        for event in events:
            if event.form == "named_declaration":
                groups.append([event])
            else:
                if lexical_group is None:
                    lexical_group = []
                    groups.append(lexical_group)
                lexical_group.append(event)
        return groups
    if language == "go":
        groups = []
        lexical_group: list[_Binding] | None = None
        for event in events:
            if event.kind == "parameter":
                if lexical_group is None:
                    lexical_group = []
                    groups.append(lexical_group)
                lexical_group.append(event)
            elif event.form == "short_var_declaration":
                if lexical_group is None:
                    lexical_group = []
                    groups.append(lexical_group)
                lexical_group.append(event)
            else:
                group = [event]
                groups.append(group)
                if event.kind == "variable" and lexical_group is None:
                    lexical_group = group
        return groups
    return [[event] for event in events]


def _use_role(node) -> str:
    parent = node.parent
    if node.type == "property_identifier":
        return "member_reference"
    if parent is None:
        return "lexical_reference"
    if parent.type == "keyword_argument" and parent.child_by_field_name("name") == node:
        return "label_reference"
    member_fields = ("attribute", "field", "name", "property")
    if parent.type in {
        "attribute",
        "field_access",
        "field_expression",
        "member_expression",
        "method_invocation",
        "selector_expression",
    } and any(parent.child_by_field_name(field) == node for field in member_fields):
        return "member_reference"
    return "lexical_reference"


def _collect_file_graph(changed_file) -> dict[str, Any] | None:
    language = _language(changed_file.path)
    parser = _parser(language) if language else None
    if language is None or parser is None:
        return None
    source, new_linenos = _added_source(changed_file)
    if not source:
        return None
    tree = parser.parse(source)

    root_scope = _Scope(
        scope_id=changed_file.path,
        path=changed_file.path,
        parent_id=None,
        kind="file",
        name=changed_file.path,
        depth=0,
        start_byte=0,
        end_byte=len(source),
        start_line=1,
        end_line=max(1, len(new_linenos)),
    )
    scopes: dict[str, _Scope] = {root_scope.scope_id: root_scope}
    binding_events: list[_Binding] = []
    binding_node_keys: set[tuple[int, int, str]] = set()
    event_counter = 0

    def add_binding(
        target,
        *,
        scope_id: str,
        kind: str,
        form: str,
        named_declaration: bool = False,
    ) -> None:
        nonlocal event_counter
        for identifier in _binding_identifiers(
            target, named_declaration=named_declaration
        ):
            name = _node_text(identifier, source)
            if not name:
                continue
            if language == "go" and name == "_":
                # The Go blank identifier discards a value and never binds.
                continue
            event_counter += 1
            binding_node_keys.add(_node_key(identifier))
            binding_events.append(
                _Binding(
                    event_id=f"{changed_file.path}#binding-{event_counter:04d}",
                    path=changed_file.path,
                    scope_id=scope_id,
                    name=name,
                    kind=kind,
                    form=form,
                    language=language,
                    start_byte=identifier.start_byte,
                    end_byte=identifier.end_byte,
                    fragment_line=identifier.start_point.row + 1,
                    diff_new_lineno=_diff_line(identifier, new_linenos),
                )
            )

    def new_scope(node, active_scope: _Scope, kind: str) -> _Scope:
        name = _scope_name(node, source)
        base_id = f"{active_scope.scope_id}::{_scope_label(kind, name)}"
        scope_id = base_id
        suffix = 2
        while scope_id in scopes:
            scope_id = f"{base_id}@{suffix}"
            suffix += 1
        scope = _Scope(
            scope_id=scope_id,
            path=changed_file.path,
            parent_id=active_scope.scope_id,
            kind=kind,
            name=name,
            depth=active_scope.depth + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
        )
        scopes[scope_id] = scope
        return scope

    def visit(node, active_scope: _Scope) -> None:
        kind = _scope_kind(node, language)
        next_scope = active_scope
        if kind is not None:
            named = _name_node(node)
            if named is not None and node.type in _NAMED_DECLARATION_NODES:
                add_binding(
                    named,
                    scope_id=active_scope.scope_id,
                    kind=_NAMED_DECLARATION_NODES[node.type],
                    form="named_declaration",
                    named_declaration=True,
                )
            next_scope = new_scope(node, active_scope, kind)

        if node.type in _PARAMETER_NODES:
            targets = _field_nodes(node, "name") if language == "go" else []
            if not targets:
                target = _binding_target(node)
                targets = [target if target is not None else node]
            for target in targets:
                add_binding(
                    target,
                    scope_id=next_scope.scope_id,
                    kind="parameter",
                    form=node.type,
                )
        elif node.type in _PARAMETER_LIST_NODES:
            for child in node.named_children:
                if child.type in _BINDING_LEAVES:
                    add_binding(
                        child,
                        scope_id=next_scope.scope_id,
                        kind="parameter",
                        form="bare_parameter",
                    )

        if language == "python" and node.type in {
            "assignment",
            "for_in_clause",
            "for_statement",
            "named_expression",
        }:
            target = _binding_target(node)
            if target is not None:
                add_binding(
                    target,
                    scope_id=next_scope.scope_id,
                    kind="variable",
                    form=(
                        "loop_binding"
                        if node.type in {"for_in_clause", "for_statement"}
                        else "assignment_binding"
                    ),
                )
        elif language in {"javascript", "typescript"} and node.type == "variable_declarator":
            target = node.child_by_field_name("name")
            parent = node.parent
            is_var = parent is not None and parent.type == "variable_declaration"
            target_scope_id = (
                _nearest_function_or_file_scope(scopes, next_scope.scope_id)
                if is_var
                else next_scope.scope_id
            )
            add_binding(
                target,
                scope_id=target_scope_id,
                kind="variable",
                form="var_declaration" if is_var else "lexical_declaration",
            )
        elif language == "java" and node.type == "variable_declarator":
            add_binding(
                node.child_by_field_name("name"),
                scope_id=next_scope.scope_id,
                kind="variable",
                form="variable_declaration",
            )
        elif language == "java" and node.type == "enhanced_for_statement":
            add_binding(
                node.child_by_field_name("name"),
                scope_id=next_scope.scope_id,
                kind="variable",
                form="enhanced_for_binding",
            )
        elif language == "go" and node.type == "var_spec":
            for target in _field_nodes(node, "name"):
                add_binding(
                    target,
                    scope_id=next_scope.scope_id,
                    kind="variable",
                    form="var_declaration",
                )
        elif language == "go" and node.type in {"short_var_declaration", "range_clause"}:
            target = node.child_by_field_name("left")
            if target is not None:
                add_binding(
                    target,
                    scope_id=next_scope.scope_id,
                    kind="variable",
                    form=(
                        "short_var_declaration"
                        if node.type == "short_var_declaration"
                        else "range_binding"
                    ),
                )

        if node.type == "catch_clause":
            for target in _field_nodes(node, "parameter"):
                add_binding(
                    target,
                    scope_id=next_scope.scope_id,
                    kind="parameter",
                    form="catch_parameter",
                )

        for child in node.named_children:
            visit(child, next_scope)

    visit(tree.root_node, root_scope)

    events_by_scope_name: dict[tuple[str, str], list[_Binding]] = defaultdict(list)
    for event in binding_events:
        events_by_scope_name[(event.scope_id, event.name)].append(event)

    declaration_rows: list[dict[str, Any]] = []
    declaration_ids_by_scope_name: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (scope_id, name), events in sorted(events_by_scope_name.items()):
        groups = _groups_for_events(language, events)
        for group_index, group in enumerate(groups, 1):
            first = group[0]
            qualified_name = f"{scope_id}::{name}"
            declaration_id = (
                qualified_name if len(groups) == 1 else f"{qualified_name}#{group_index}"
            )
            declaration_ids_by_scope_name[(scope_id, name)].append(declaration_id)
            scope = scopes[scope_id]
            declaration_rows.append(
                {
                    "declaration_id": declaration_id,
                    "qualified_name": qualified_name,
                    "path": changed_file.path,
                    "scope_id": scope_id,
                    "scope_kind": scope.kind,
                    "scope_depth": scope.depth,
                    "name": name,
                    "morphemes": split_identifier_morphemes(name),
                    "kind": first.kind,
                    "binding_forms": sorted({event.form for event in group}),
                    "binding_sites": [
                        {
                            "event_id": event.event_id,
                            "fragment_line": event.fragment_line,
                            "diff_new_lineno": event.diff_new_lineno,
                        }
                        for event in group
                    ],
                    "scope_line_span": max(1, scope.end_line - scope.start_line + 1),
                    "resolved_use_count": 0,
                }
            )

    declaration_by_id = {row["declaration_id"]: row for row in declaration_rows}

    def resolve(scope_id: str, name: str) -> tuple[str | None, list[str]]:
        for current in _lookup_scope_ids(scopes, scope_id, language):
            candidates = declaration_ids_by_scope_name.get((current, name), [])
            if candidates:
                return current, candidates
        return None, []

    uses: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for node in _walk(tree.root_node):
        if node.type not in _USE_IDENTIFIER_NODES or _node_key(node) in binding_node_keys:
            continue
        name = _node_text(node, source)
        if language == "go" and name == "_":
            continue
        containing = [
            scope
            for scope in scopes.values()
            if scope.start_byte <= node.start_byte < scope.end_byte
        ]
        scope = max(
            containing,
            key=lambda candidate: (
                candidate.depth,
                -(candidate.end_byte - candidate.start_byte),
            ),
        )
        role = _use_role(node)
        resolved_scope_id, resolved_ids = (
            resolve(scope.scope_id, name)
            if role == "lexical_reference"
            else (None, [])
        )
        use_id = f"{changed_file.path}#use-{len(uses) + 1:04d}"
        resolution = (
            "member_unresolved"
            if role == "member_reference"
            else "label_unresolved"
            if role == "label_reference"
            else "unresolved"
            if not resolved_ids
            else "ambiguous_same_scope"
            if len(resolved_ids) > 1
            else "resolved"
        )
        uses.append(
            {
                "use_id": use_id,
                "path": changed_file.path,
                "scope_id": scope.scope_id,
                "name": name,
                "morphemes": split_identifier_morphemes(name),
                "role": role,
                "resolution": resolution,
                "resolved_scope_id": resolved_scope_id,
                "resolved_declaration_ids": resolved_ids,
                "fragment_line": node.start_point.row + 1,
                "diff_new_lineno": _diff_line(node, new_linenos),
            }
        )
        for declaration_id in resolved_ids:
            edges.append({"declaration_id": declaration_id, "use_id": use_id})
            declaration_by_id[declaration_id]["resolved_use_count"] += 1

    collisions = []
    for (scope_id, name), declaration_ids in sorted(declaration_ids_by_scope_name.items()):
        if len(declaration_ids) > 1:
            collisions.append(
                {
                    "scope_id": scope_id,
                    "name": name,
                    "declaration_ids": declaration_ids,
                    "relation": "same_scope_multiple_declarations",
                    "harmfulness_unresolved": True,
                }
            )

    shadowing = []
    for row in declaration_rows:
        ancestor_ids = list(
            _lookup_scope_ids(scopes, row["scope_id"], language)
        )[1:]
        ancestors: list[str] = []
        for ancestor_id in ancestor_ids:
            ancestors.extend(
                declaration_ids_by_scope_name.get((ancestor_id, row["name"]), [])
            )
        if ancestors:
            shadowing.append(
                {
                    "declaration_id": row["declaration_id"],
                    "name": row["name"],
                    "ancestor_declaration_ids": ancestors,
                    "relation": "ancestor_name_shadowing",
                    "harmfulness_unresolved": True,
                }
            )

    parse_errors = sum(1 for node in _walk(tree.root_node) if node.type == "ERROR")
    missing_nodes = sum(1 for node in _walk(tree.root_node) if node.is_missing)
    return {
        "path": changed_file.path,
        "language": language,
        "added_line_count": len(new_linenos),
        "parse_has_error": bool(tree.root_node.has_error),
        "parse_error_nodes": parse_errors,
        "parse_missing_nodes": missing_nodes,
        "scopes": [scope.__dict__ for scope in scopes.values()],
        "declarations": declaration_rows,
        "uses": uses,
        "declaration_use_edges": edges,
        "same_scope_collisions": collisions,
        "ancestor_shadowing": shadowing,
    }


def declaration_use_graph(text: str) -> dict[str, Any]:
    """Return generic scope/declaration/use facts from added diff fragments."""
    document = parse_unified_diff(text)
    files = []
    unsupported = []
    empty_supported = []
    for changed_file in document.files:
        language = _language(changed_file.path)
        if language is None:
            if changed_file.added_lines:
                unsupported.append(changed_file.path)
            continue
        graph = _collect_file_graph(changed_file)
        if graph is None:
            empty_supported.append(changed_file.path)
        else:
            files.append(graph)

    return {
        "schema": SCHEMA,
        "analysis_unit": "added diff lines only; not a repository checkout",
        "truncated_input": document.truncated,
        "orphan_fragment_count": len(document.orphan_fragments),
        "supported_files_analyzed": len(files),
        "unsupported_files": unsupported,
        "supported_files_without_added_code": empty_supported,
        "semantic_rules": {
            "python_member_assignment_targets_are_not_bindings": True,
            "python_named_redefinitions_remain_distinct": True,
            "python_function_lookup_skips_class_namespaces": True,
            "javascript_typescript_block_scopes": True,
            "javascript_typescript_var_uses_function_or_file_scope": True,
            "go_var_short_var_and_parameters": True,
            "type_identifiers_are_not_lexical_uses": True,
        },
        "files": files,
        "limitations": [
            "lexical resolution is fragment-local and may miss unchanged declarations",
            "only enumerated binding forms and supported grammars are represented",
            "global, nonlocal, import, and language-specific hoisting details remain partial",
            "dynamic dispatch and runtime bindings are not resolved",
            "member and label references are recorded but not lexically resolved",
            "parse recovery and concatenated hunk fragments can alter apparent scopes",
            "collision and shadowing relations do not establish harmfulness",
            "identifier morphemes do not establish semantic context-fit",
        ],
    }


class CodeScopeOpsV3:
    """Fact-only operation surface; deliberately contains no criterion scorer."""

    declaration_use_graph = staticmethod(declaration_use_graph)
    split_identifier_morphemes = staticmethod(split_identifier_morphemes)
