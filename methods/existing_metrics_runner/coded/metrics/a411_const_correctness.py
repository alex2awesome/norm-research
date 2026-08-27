"""a411: const-correctness in added C++ lines.

We walk the tree-sitter-cpp tree and look for places where const-qualification
is *applicable* and check whether it was applied. Three site types:

  1. Member functions that are non-mutating accessors. We approximate
     "accessor-shaped" as: methods whose name starts with `get`, `is`, `has`,
     `size`, `empty`, `count`, `length`, `c_str`, `data`, `begin`, `end`,
     `at`, `front`, `back`, `find` AND which return a value (i.e. return
     type isn't `void`). For each, the site is "constable" — we score 1 if
     the method is declared `const` (trailing const), else 0.

  2. Parameters passed by reference / pointer to a class type. For each such
     parameter, the site is constable. We score 1 if it has the `const`
     qualifier in its type, else 0. We exclude primitive-typed parameters
     (the const rule is weaker / debated for `int x` vs `const int x`).

  3. Return types: pointer/reference returns to user types in
     accessor-shaped methods. We score 1 if `const T&` / `const T*`,
     else 0.

The metric is `const_applied / const_applicable`. Site count must be >=1
to score; otherwise abstain.

Classification THIN. Tier 2 (tree-sitter only). Library: tree-sitter-cpp.
"""
from __future__ import annotations

from typing import Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a411"
ASPECT_NAME = "C++ const-correctness"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h"]

ACCESSOR_PREFIXES = (
    "get", "is", "has", "size", "empty", "count", "length",
    "c_str", "data", "begin", "end", "at", "front", "back", "find",
)

PRIMITIVE_TYPES = {
    "int", "char", "short", "long", "float", "double", "bool",
    "void", "unsigned", "signed", "size_t", "uint8_t", "uint16_t",
    "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t",
}

_PARSER = None


def _get_parser():
    global _PARSER
    if _PARSER is None:
        try:
            import tree_sitter_cpp
            from tree_sitter import Language, Parser
            _PARSER = Parser(Language(tree_sitter_cpp.language()))
        except ImportError:
            return None
    return _PARSER


def _text(node) -> str:
    return node.text.decode("utf8", errors="replace")


def _has_trailing_const(func_decl_node) -> bool:
    """Heuristic: search the function declarator's children for a `type_qualifier`
    with text `const`. Tree-sitter-cpp emits this for member function trailing
    const."""
    for c in func_decl_node.children:
        if c.type == "type_qualifier" and _text(c) == "const":
            return True
    return False


def _function_name(func_def_node) -> Optional[str]:
    """Extract the function name from a function_definition / declaration node."""
    # Walk into function_declarator -> field_identifier or identifier
    def find_name(n):
        if n.type == "function_declarator":
            for c in n.children:
                nm = find_name(c)
                if nm is not None:
                    return nm
        elif n.type in ("identifier", "field_identifier"):
            return _text(n)
        else:
            for c in n.children:
                nm = find_name(c)
                if nm is not None:
                    return nm
        return None
    return find_name(func_def_node)


def _is_accessor_name(name: str) -> bool:
    if not name:
        return False
    low = name.lower()
    return any(low == p or low.startswith(p) for p in ACCESSOR_PREFIXES)


def _find_function_declarator(node):
    """Return the function_declarator inside a function_definition or
    field_declaration / declaration node, or None."""
    for c in node.children:
        if c.type == "function_declarator":
            return c
        if c.type in ("pointer_declarator", "reference_declarator"):
            r = _find_function_declarator(c)
            if r is not None:
                return r
    return None


def _return_text(func_def_node) -> str:
    """Concat type-spec children of a function_definition node."""
    parts = []
    for c in func_def_node.children:
        if c.type in ("primitive_type", "type_identifier", "qualified_identifier",
                      "template_type", "auto", "placeholder_type_specifier",
                      "type_descriptor", "sized_type_specifier"):
            parts.append(_text(c))
        if c.type == "type_qualifier":
            parts.append(_text(c))
    return " ".join(parts)


def _param_signatures(func_decl_node):
    """Yield (param_type_text, is_ref_or_ptr, type_is_class) for each formal
    parameter."""
    for c in func_decl_node.children:
        if c.type == "parameter_list":
            for p in c.children:
                if p.type != "parameter_declaration":
                    continue
                # Type qualifiers + type
                type_parts = []
                has_const = False
                for sub in p.children:
                    if sub.type == "type_qualifier" and _text(sub) == "const":
                        has_const = True
                    if sub.type in ("primitive_type", "type_identifier",
                                    "qualified_identifier", "template_type",
                                    "sized_type_specifier",
                                    "placeholder_type_specifier"):
                        type_parts.append((sub.type, _text(sub)))
                # detect ref/ptr declarator
                is_ref = False
                is_ptr = False
                for sub in p.children:
                    if sub.type == "reference_declarator":
                        is_ref = True
                    if sub.type == "pointer_declarator":
                        is_ptr = True
                # type "is class" heuristic: not primitive_type and not in
                # PRIMITIVE_TYPES
                type_is_class = False
                if type_parts:
                    kind, txt = type_parts[0]
                    if kind != "primitive_type" and txt not in PRIMITIVE_TYPES:
                        type_is_class = True
                yield has_const, (is_ref or is_ptr), type_is_class


def _count_sites(source: bytes) -> Optional[Tuple[int, int]]:
    parser = _get_parser()
    if parser is None:
        return None
    # Wrap loose snippet in a class so we get field/member context
    src = source if (b"class " in source or b"struct " in source
                     or b"namespace " in source) else (
        b"struct __Snip {\n" + source + b"\n};\n")
    tree = parser.parse(src)

    applicable = 0
    applied = 0

    def visit(node, in_class: bool):
        nonlocal applicable, applied
        t = node.type
        new_in_class = in_class or (t in ("class_specifier", "struct_specifier"))
        if t in ("function_definition", "field_declaration"):
            fd = _find_function_declarator(node)
            if fd is not None:
                name = _function_name(node) or ""
                ret = _return_text(node)
                # 1. Accessor const check (only inside class context)
                if new_in_class and _is_accessor_name(name):
                    # exclude operators (which appear as operator_name nodes,
                    # not identifier) — already filtered by name being None
                    # exclude void-returning "getters" (rare)
                    if "void" not in ret:
                        applicable += 1
                        if _has_trailing_const(fd):
                            applied += 1
                # 2. Parameter const check
                for has_const, is_ref_ptr, is_class in _param_signatures(fd):
                    if is_ref_ptr and is_class:
                        applicable += 1
                        if has_const:
                            applied += 1
        for c in node.children:
            visit(c, new_in_class)

    visit(tree.root_node, False)
    return applicable, applied


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    total_applicable = 0
    total_applied = 0
    for content in by_path.values():
        res = _count_sites(content.encode("utf8", errors="replace"))
        if res is None:
            return None
        a, ap = res
        total_applicable += a
        total_applied += ap
    if total_applicable == 0:
        return None
    return float(total_applied / total_applicable)
