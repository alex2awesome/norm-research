"""Generic scope-aware declaration/use facts for added code in unified diffs.

This additive module leaves the frozen :mod:`ops_code` API unchanged.  It emits
relation facts rather than a criterion-specific scalar: lexical scopes,
declarations, identifier uses, resolution edges, same-scope multiple-binding
events, ancestor shadowing, and identifier morphemes.  A later metric program
may decide how (or whether) those facts bear on an articulated construct.

Only added diff lines are available.  Missing repository context, omitted hunk
context, unsupported grammars, parse recovery, imports, and dynamic binding can
all make resolution incomplete.  The returned profile records these limits and
must not be presented as whole-program static analysis.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Any, Iterable

try:
    from .ops_code import ChangedFile, parse_unified_diff
except ImportError:  # pragma: no cover - direct-module compatibility
    from ops_code import ChangedFile, parse_unified_diff  # type: ignore[no-redef]


SCHEMA = "metric-seam.code-scope-declaration-use-graph.v2"

_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
}

_SCOPE_NODES = {
    "python": {
        "class_definition": "class",
        "function_definition": "function",
        "lambda": "lambda",
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

_NAMED_DECLARATION_NODES = {
    "class_definition": "class",
    "function_definition": "function",
    "class_declaration": "class",
    "interface_declaration": "interface",
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "method_definition": "method",
    "enum_declaration": "enum",
    "record_declaration": "record",
    "method_declaration": "method",
    "constructor_declaration": "constructor",
    "type_declaration": "type",
}

_PARAMETER_NODES = {
    "default_parameter",
    "formal_parameter",
    "identifier_parameter",
    "optional_parameter",
    "parameter_declaration",
    "required_parameter",
    "rest_pattern",
    "spread_parameter",
    "typed_default_parameter",
    "typed_parameter",
    "variadic_parameter_declaration",
}

_PARAMETER_LIST_NODES = {
    "formal_parameters",
    "parameters",
    "parameter_list",
}

_BINDING_NODES = {
    "assignment": "assignment_binding",
    "for_statement": "loop_binding",
    "named_expression": "assignment_binding",
    "variable_declarator": "variable_declaration",
    "short_var_declaration": "variable_declaration",
    "range_clause": "loop_binding",
}

_IDENTIFIER_NODES = {"identifier", "property_identifier", "type_identifier"}
_BOUND_IDENTIFIER_NODES = _IDENTIFIER_NODES | {
    "field_identifier",
    "shorthand_property_identifier_pattern",
}

_MORPHEME = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)


@dataclass(frozen=True)
class _Scope:
    scope_id: str
    path: str
    parent_id: str | None
    kind: str
    name: str
    depth: int
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int


@dataclass(frozen=True)
class _Binding:
    event_id: str
    path: str
    scope_id: str
    name: str
    kind: str
    form: str
    language: str
    start_byte: int
    end_byte: int
    fragment_line: int
    diff_new_lineno: int | None


_PARSERS: dict[str, Any] = {}


def split_identifier_morphemes(name: str) -> list[str]:
    """Split snake/camel/acronym/digit surfaces without judging their meaning."""
    pieces: list[str] = []
    for segment in re.split(r"[^A-Za-z0-9]+", name or ""):
        if segment:
            pieces.extend(_MORPHEME.findall(segment))
    return pieces


def _language(path: str) -> str | None:
    return _LANGUAGE.get(PurePosixPath(path.lower()).suffix)


def _parser(language: str):
    if language in _PARSERS:
        return _PARSERS[language]
    try:
        from tree_sitter import Language, Parser

        if language == "python":
            import tree_sitter_python as grammar

            capsule = grammar.language()
        elif language == "javascript":
            import tree_sitter_javascript as grammar

            capsule = grammar.language()
        elif language == "typescript":
            import tree_sitter_typescript as grammar

            capsule = grammar.language_typescript()
        elif language == "java":
            import tree_sitter_java as grammar

            capsule = grammar.language()
        elif language == "go":
            import tree_sitter_go as grammar

            capsule = grammar.language()
        else:  # pragma: no cover - guarded by the extension map
            return None
        parser = Parser(Language(capsule))
    except (ImportError, TypeError):
        return None
    _PARSERS[language] = parser
    return parser


def _walk(node) -> Iterable[Any]:
    yield node
    for child in node.children:
        yield from _walk(child)


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _node_key(node) -> tuple[int, int, str]:
    return node.start_byte, node.end_byte, node.type


def _name_node(node):
    named = node.child_by_field_name("name")
    if named is not None:
        return named
    for child in node.children:
        if child.type in _BOUND_IDENTIFIER_NODES:
            return child
    return None


def _scope_name(node, source: bytes) -> str:
    named = _name_node(node)
    if named is not None:
        return _node_text(named, source)
    return f"anonymous@L{node.start_point.row + 1}"


def _scope_label(kind: str, name: str) -> str:
    # Source identifiers can contain punctuation (for example JS property names).
    # Escape separators so the qualified path remains unambiguous.
    escaped = name.replace("%", "%25").replace(":", "%3A")
    return f"{kind}:{escaped}"


def _added_source(changed_file: ChangedFile) -> tuple[bytes, list[int | None]]:
    lines: list[str] = []
    new_linenos: list[int | None] = []
    for hunk in changed_file.hunks:
        for line in hunk.lines:
            if line.kind == "add":
                lines.append(line.text)
                new_linenos.append(line.new_lineno)
    return "\n".join(lines).encode("utf-8", errors="replace"), new_linenos


def _diff_line(node, new_linenos: list[int | None]) -> int | None:
    row = node.start_point.row
    return new_linenos[row] if 0 <= row < len(new_linenos) else None


def _binding_identifiers(node) -> list[Any]:
    """Return syntactically bound leaves beneath a declaration target."""
    if node is None:
        return []
    if node.type in _BOUND_IDENTIFIER_NODES:
        return [node]
    out = []
    for child in node.named_children:
        # Type nodes describe a binding but do not introduce its lexical name.
        if child.type in {
            "generic_type",
            "predefined_type",
            "primitive_type",
            "type",
            "type_annotation",
            "type_identifier",
        }:
            continue
        out.extend(_binding_identifiers(child))
    return out


def _binding_target(node):
    for field in ("name", "left", "pattern"):
        target = node.child_by_field_name(field)
        if target is not None:
            return target
    return None


def _nearest_scope(scopes: dict[str, _Scope], byte: int) -> _Scope:
    containing = [
        scope
        for scope in scopes.values()
        if scope.start_byte <= byte <= scope.end_byte
    ]
    return max(containing, key=lambda scope: (scope.depth, -scope.end_byte + scope.start_byte))


def _collect_file_graph(changed_file: ChangedFile) -> dict[str, Any] | None:
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

    def add_binding(target, *, scope_id: str, kind: str, form: str) -> None:
        nonlocal event_counter
        for identifier in _binding_identifiers(target):
            name = _node_text(identifier, source)
            if not name:
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

    def visit(node, active_scope: _Scope) -> None:
        kind = _SCOPE_NODES.get(language, {}).get(node.type)
        next_scope = active_scope
        if kind:
            named = _name_node(node)
            if named is not None and node.type in _NAMED_DECLARATION_NODES:
                add_binding(
                    named,
                    scope_id=active_scope.scope_id,
                    kind=_NAMED_DECLARATION_NODES[node.type],
                    form="named_declaration",
                )
            name = _scope_name(node, source)
            base_id = f"{active_scope.scope_id}::{_scope_label(kind, name)}"
            scope_id = base_id
            suffix = 2
            while scope_id in scopes:
                scope_id = f"{base_id}@{suffix}"
                suffix += 1
            next_scope = _Scope(
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
            scopes[scope_id] = next_scope

        if node.type in _PARAMETER_NODES:
            target = _binding_target(node) or node
            add_binding(target, scope_id=next_scope.scope_id, kind="parameter", form=node.type)
        elif node.type in _PARAMETER_LIST_NODES:
            # Python and JavaScript allow bare identifier parameters directly in
            # the list node, without a wrapper parameter node.
            for child in node.named_children:
                if child.type in _BOUND_IDENTIFIER_NODES:
                    add_binding(
                        child,
                        scope_id=next_scope.scope_id,
                        kind="parameter",
                        form="bare_parameter",
                    )
        elif node.type in _BINDING_NODES:
            target = _binding_target(node)
            if target is not None:
                add_binding(
                    target,
                    scope_id=next_scope.scope_id,
                    kind="variable",
                    form=_BINDING_NODES[node.type],
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
        # Python assignment sites are repeated bindings to one lexical name.
        # Typed variable declaration events and named declarations remain
        # separate facts so downstream analysis can see duplicate declarations.
        groups: list[list[_Binding]] = []
        for event in events:
            if event.language == "python":
                matching = next(
                    (
                        group
                        for group in groups
                        if group[0].language == "python"
                    ),
                    None,
                )
                if matching is not None:
                    matching.append(event)
                    continue
            groups.append([event])

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
        current: str | None = scope_id
        while current is not None:
            candidates = declaration_ids_by_scope_name.get((current, name), [])
            if candidates:
                return current, candidates
            current = scopes[current].parent_id
        return None, []

    uses: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for node in _walk(tree.root_node):
        if node.type not in _IDENTIFIER_NODES or _node_key(node) in binding_node_keys:
            continue
        name = _node_text(node, source)
        scope = _nearest_scope(scopes, node.start_byte)
        role = "lexical_reference"
        parent = node.parent
        if node.type == "property_identifier" or (
            parent is not None
            and parent.type in {"attribute", "field_expression", "member_expression"}
            and (
                parent.child_by_field_name("property") == node
                or parent.child_by_field_name("attribute") == node
                or parent.child_by_field_name("field") == node
            )
        ):
            role = "member_reference"
        resolved_scope_id, resolved_ids = (
            (None, []) if role == "member_reference" else resolve(scope.scope_id, name)
        )
        use_id = f"{changed_file.path}#use-{len(uses) + 1:04d}"
        resolution = (
            "member_unresolved"
            if role == "member_reference"
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
        ancestor_id = scopes[row["scope_id"]].parent_id
        ancestors: list[str] = []
        while ancestor_id is not None:
            ancestors.extend(
                declaration_ids_by_scope_name.get((ancestor_id, row["name"]), [])
            )
            ancestor_id = scopes[ancestor_id].parent_id
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
    return {
        "path": changed_file.path,
        "language": language,
        "added_line_count": len(new_linenos),
        "parse_has_error": bool(tree.root_node.has_error),
        "parse_error_nodes": parse_errors,
        "scopes": [scope.__dict__ for scope in scopes.values()],
        "declarations": declaration_rows,
        "uses": uses,
        "declaration_use_edges": edges,
        "same_scope_collisions": collisions,
        "ancestor_shadowing": shadowing,
    }


def declaration_use_graph(text: str) -> dict[str, Any]:
    """Return scope/declaration/use facts from the added portions of a diff."""
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
        "files": files,
        "limitations": [
            "lexical resolution is fragment-local and may miss unchanged declarations",
            "dynamic dispatch and runtime bindings are not resolved",
            "member references are recorded but not lexically resolved",
            "parse recovery and concatenated hunk fragments can alter apparent scopes",
            "collision and shadowing relations do not establish harmfulness",
            "identifier morphemes do not establish semantic context-fit",
        ],
    }


class CodeScopeOpsV2:
    """Fact-only operation surface; deliberately contains no a407 scorer."""

    declaration_use_graph = staticmethod(declaration_use_graph)
    split_identifier_morphemes = staticmethod(split_identifier_morphemes)
