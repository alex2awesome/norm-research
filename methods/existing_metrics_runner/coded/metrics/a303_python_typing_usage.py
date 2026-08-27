"""a303: Python typing semantics and usage.

Norm: "Apply typing correctly: Any semantics, None/type(None), casts for
static checking only, aliasing/annotation intent, constraints on defaulted
type parameters, and the static (non-runtime) nature of type checking."

Tool choice rationale
---------------------
The hint floated three options:
  (1) `mypy --strict` (or pyright/pytype) on the added snippets, scoring by
      violation density, OR
  (2) tree-sitter coverage: fraction of param/return positions that carry
      annotations.

I tried (1) on the actual fixture diffs and it FAILED for the obvious
reason: PR diffs deliver syntactically incomplete fragments (no class
header, no surrounding function, mid-block context). On every Python
fixture except one, mypy returned a single SYNTAX error and zero typing
errors — i.e. the type checker never ran. Type checkers need whole-module
parse trees by design; tree-sitter's error-recovery parser does not.

So the metric uses (2) — a tree-sitter-python coverage measure — and adds
two THIN-but-real correctness signals on top of pure coverage:

  A. annotation coverage (the headline number)
     - fraction of function PARAMS (excluding `self`/`cls`/`*args`/`**kwargs`)
       that carry a `: T` annotation
     - fraction of function DEFINITIONS that carry a `-> T` return annotation
     - if `typing` / `typing_extensions` was imported but the file then
       defines zero annotated positions, this is bad-faith and scored 0.

  B. Any-overuse penalty
     - `typing.Any` is legal but signals declined typing; a file where
       >50% of typed positions use bare `Any` is penalized down. This
       captures "Any semantics" in the norm description.

  C. None/type(None) correctness
     - `None` is the correct annotation, `type(None)` is the runtime form
       and incorrect inside an annotation. Detected via the AST shape of
       the annotation node. Each occurrence inside a `type` annotation node
       counts as a violation and lowers the file score.

We score per file then average across files. Abstain when no function
definitions are observed in the added code (router-gated).

Classification: THIN. The norm is about a well-defined, mechanical
property of source code (does this position have a type? is the type
sensible?). It is not about taste. The reason this is not Tier-3 is
purely environmental: the artifact format (PR diff) is not a complete
module, so we can't run a checker. The Tier-2 coverage measure is the
strongest honest signal available.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a303"
ASPECT_NAME = "Python typing semantics and usage"
TIER = 2
TOOLS = ["tree-sitter-python"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

_PARSER = None


def _get_parser():
    global _PARSER
    if _PARSER is None:
        try:
            import tree_sitter_python
            from tree_sitter import Language, Parser
            _PARSER = Parser(Language(tree_sitter_python.language()))
        except ImportError:
            return None
    return _PARSER


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


# Names that should never be counted as a "param needing annotation".
_EXEMPT_PARAM_NAMES = frozenset({"self", "cls"})

# Param-node types in tree-sitter-python's grammar (we examine the *kind* of
# each child of a `parameters` node and bin it as annotated vs un-annotated).
_ANNOTATED_PARAM_TYPES = {
    "typed_parameter",         # x: int
    "typed_default_parameter",  # x: int = 1
}
_UNANNOTATED_PARAM_TYPES = {
    "identifier",               # x
    "default_parameter",        # x = 1
    "list_splat_pattern",       # *args  -> typically un-annotated, fine to skip
    "dictionary_splat_pattern",  # **kw   -> ditto
}

# These are "annotatable" but Python convention exempts them from naming
# penalty (we treat *args/**kwargs as exempt from coverage altogether).
_EXEMPT_PARAM_TYPES = {"list_splat_pattern", "dictionary_splat_pattern"}


def _param_first_identifier(node) -> Optional[str]:
    """Walk to the first identifier inside a parameter node."""
    if node.type == "identifier":
        return None  # caller already knows
    for c in node.children:
        if c.type == "identifier":
            return c
    return None


def _is_any_typed(type_node, src: bytes) -> bool:
    """True if a `type` annotation node is just `Any` or `typing.Any`."""
    if type_node is None:
        return False
    s = _text(type_node, src).strip()
    return s == "Any" or s == "typing.Any" or s == "t.Any"


def _contains_type_of_none(type_node, src: bytes) -> bool:
    """True if annotation uses `type(None)` instead of `None`.

    `type(None)` is the runtime value of NoneType; inside an annotation it
    should be `None` per PEP 484 §"Using None / type(None)". Tree-sitter
    parses `type(None)` as a `call` whose function identifier is `type`.
    """
    if type_node is None:
        return False
    # Walk for any `call` whose function part is identifier `type` and
    # whose argument is `None`.
    stack = [type_node]
    while stack:
        n = stack.pop()
        if n.type == "call":
            func = n.children[0] if n.children else None
            if func is not None and func.type == "identifier" \
                    and _text(func, src) == "type":
                # check args
                for c in n.children:
                    if c.type == "argument_list":
                        for cc in c.children:
                            if cc.type == "none":
                                return True
        stack.extend(n.children)
    return False


def _walk_function_defs(root, src: bytes):
    """Yield every function_definition node anywhere in the tree."""
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "function_definition":
            yield n
        stack.extend(n.children)


def _param_type_node(param_node):
    """For a typed_parameter / typed_default_parameter, return the `type` child."""
    for c in param_node.children:
        if c.type == "type":
            return c
    return None


def _return_type_node(fn_node):
    """Find the return type: child `type` node appearing AFTER the `->` token."""
    saw_arrow = False
    for c in fn_node.children:
        if c.type == "->":
            saw_arrow = True
            continue
        if saw_arrow and c.type == "type":
            return c
    return None


def _file_score(code: bytes) -> Optional[float]:
    parser = _get_parser()
    if parser is None:
        return None
    tree = parser.parse(code)
    root = tree.root_node
    if root.type not in ("module", "program"):
        return None

    fns = list(_walk_function_defs(root, code))
    if not fns:
        return None

    # ---- (A) coverage ----
    n_param_slots = 0
    n_param_annotated = 0
    n_return_slots = 0
    n_return_annotated = 0
    # ---- (B) Any-overuse and (C) type(None) tracking ----
    n_typed_positions = 0       # denominator for Any rate
    n_any_positions = 0
    n_type_none_violations = 0

    for fn in fns:
        # Return slot
        rt = _return_type_node(fn)
        n_return_slots += 1
        if rt is not None:
            n_return_annotated += 1
            n_typed_positions += 1
            if _is_any_typed(rt, code):
                n_any_positions += 1
            if _contains_type_of_none(rt, code):
                n_type_none_violations += 1

        # Param slots
        params_node = None
        for c in fn.children:
            if c.type == "parameters":
                params_node = c
                break
        if params_node is None:
            continue
        for p in params_node.children:
            if p.type in ("(", ")", ","):
                continue
            if p.type in _EXEMPT_PARAM_TYPES:
                continue  # *args / **kwargs — not counted
            if p.type == "identifier":
                name = _text(p, code)
                if name in _EXEMPT_PARAM_NAMES:
                    continue
                n_param_slots += 1
                # un-annotated
                continue
            if p.type == "default_parameter":
                # plain `x=1` — un-annotated; check name for self/cls
                first_id = _param_first_identifier(p)
                if first_id is not None and \
                        _text(first_id, code) in _EXEMPT_PARAM_NAMES:
                    continue
                n_param_slots += 1
                continue
            if p.type in _ANNOTATED_PARAM_TYPES:
                first_id = _param_first_identifier(p)
                if first_id is not None and \
                        _text(first_id, code) in _EXEMPT_PARAM_NAMES:
                    continue
                n_param_slots += 1
                n_param_annotated += 1
                tnode = _param_type_node(p)
                n_typed_positions += 1
                if _is_any_typed(tnode, code):
                    n_any_positions += 1
                if _contains_type_of_none(tnode, code):
                    n_type_none_violations += 1

    if n_param_slots + n_return_slots == 0:
        return None

    # ---- combine ----
    # Coverage: weight params and returns equally.
    param_cov = (n_param_annotated / n_param_slots) if n_param_slots else None
    ret_cov = (n_return_annotated / n_return_slots) if n_return_slots else None
    parts = [c for c in (param_cov, ret_cov) if c is not None]
    coverage = sum(parts) / len(parts) if parts else 0.0

    # Any-overuse penalty: 1.0 when <=20% of typed positions are Any,
    # decays linearly to 0.5 at 100% Any.
    if n_typed_positions == 0:
        any_factor = 1.0
    else:
        any_rate = n_any_positions / n_typed_positions
        if any_rate <= 0.2:
            any_factor = 1.0
        else:
            # 0.2 -> 1.0, 1.0 -> 0.5
            any_factor = 1.0 - 0.5 * (any_rate - 0.2) / 0.8

    # type(None) violation penalty: each violation subtracts 0.1, floor 0.5.
    none_factor = max(0.5, 1.0 - 0.1 * n_type_none_violations)

    # Bad-faith check: file imports `typing` yet annotated nothing.
    imported_typing = b"import typing" in code or b"from typing" in code
    if imported_typing and n_param_annotated + n_return_annotated == 0:
        coverage = 0.0

    return float(coverage * any_factor * none_factor)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    scs: List[float] = []
    for content in by_path.values():
        s = _file_score(content.encode("utf8", errors="replace"))
        if s is not None:
            scs.append(s)
    if not scs:
        return None
    return float(sum(scs) / len(scs))
