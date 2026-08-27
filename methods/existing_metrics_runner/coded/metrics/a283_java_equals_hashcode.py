"""a283: Java equals/hashCode contract correctness.

Contract (per java.lang.Object): if a class overrides equals(), it must also
override hashCode(), and vice versa. We detect violations using a tree-sitter
Java parser, which handles partial diff snippets gracefully (error-recovery).

We only count an override of `equals` when:
  - method name is exactly `equals`
  - exactly one parameter
  - parameter type is `Object` (per Object.equals signature)

Score per class: 1.0 if no contract violation, 0.0 if violation. Metric score
is mean over classes touched in the added code.

Caveats noted by tool research:
  - Anonymous inner classes are object_creation_expression, not
    class_declaration — we extend the query to catch them.
  - Skip violations whose enclosing class is inside an ERROR subtree.
  - Custom `equals(Foo)` overloads (not @Override of Object.equals) are
    NOT counted as the contract's equals — correct per spec.
"""
from __future__ import annotations

from typing import List, Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a283"
ASPECT_NAME = "Java equals/hashCode contract correctness"
TIER = 3
TOOLS = ["tree-sitter-java"]
APPLIES_TO_LANGS = ["Java"]
CLASSIFICATION = "THIN"

JAVA_EXTS = [".java"]

# Lazy: avoid import cost when metric isn't used
_PARSER = None
_JAVA_LANG = None


def _get_parser():
    global _PARSER, _JAVA_LANG
    if _PARSER is None:
        try:
            import tree_sitter_java
            from tree_sitter import Language, Parser
            _JAVA_LANG = Language(tree_sitter_java.language())
            _PARSER = Parser(_JAVA_LANG)
        except ImportError:
            return None
    return _PARSER


def _has_error_ancestor(node) -> bool:
    cur = node.parent
    while cur is not None:
        if cur.type == "ERROR":
            return True
        cur = cur.parent
    return False


def _enclosing_class_id(method_node):
    """Tree-sitter wrappers don't share Python id even when they point at the
    same C node — use (start_byte, end_byte, type) as a stable key."""
    cur = method_node.parent
    while cur is not None:
        if cur.type in ("class_declaration", "interface_declaration",
                        "enum_declaration", "record_declaration",
                        "object_creation_expression"):
            return (cur.start_byte, cur.end_byte, cur.type), cur
        cur = cur.parent
    return None


def _method_info(method_node):
    """Return (name, param_types) or None on parse failure."""
    name = None
    params = []
    for child in method_node.children:
        if child.type == "identifier" and name is None:
            name = child.text.decode("utf8", errors="replace")
        if child.type == "formal_parameters":
            for fp in child.children:
                if fp.type == "formal_parameter":
                    # find type_identifier or generic_type
                    for sub in fp.children:
                        if sub.type in ("type_identifier", "generic_type",
                                        "scoped_type_identifier"):
                            params.append(sub.text.decode("utf8",
                                                         errors="replace"))
                            break
    return name, params


def _file_violations(source: bytes) -> Optional[int]:
    parser = _get_parser()
    if parser is None:
        return None
    # Wrap loose method bodies in a synthetic class so tree-sitter can parse
    src = source if (b"class " in source or b"interface " in source
                     or b"record " in source) else (
        b"class __Snip {\n" + source + b"\n}\n")
    tree = parser.parse(src)
    per_class = {}  # id(class_node) -> {"equals_obj": bool, "hashcode": bool, "ok": bool}

    def walk(node):
        if node.type == "method_declaration":
            info = _method_info(node)
            if info and info[0] is not None and not _has_error_ancestor(node):
                name, params = info
                cls = _enclosing_class_id(node)
                if cls is not None:
                    cid, _ = cls
                    bucket = per_class.setdefault(
                        cid, {"equals_obj": False, "hashcode": False})
                    if name == "equals" and params == ["Object"]:
                        bucket["equals_obj"] = True
                    elif name == "hashCode" and params == []:
                        bucket["hashcode"] = True
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    if not per_class:
        return None  # no classes parsed
    violations = sum(1 for b in per_class.values()
                     if b["equals_obj"] ^ b["hashcode"])
    return violations, len(per_class)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, JAVA_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, JAVA_EXTS)
    if not by_path:
        return None
    total_v = total_c = 0
    any_observed = False
    for content in by_path.values():
        res = _file_violations(content.encode("utf8", errors="replace"))
        if res is None:
            continue
        v, c = res
        total_v += v
        total_c += c
        any_observed = True
    if not any_observed or total_c == 0:
        # No class with either equals or hashCode override observed — norm
        # doesn't fire on this diff. Abstain rather than score 1.0.
        return None
    return float(1.0 - (total_v / total_c))
