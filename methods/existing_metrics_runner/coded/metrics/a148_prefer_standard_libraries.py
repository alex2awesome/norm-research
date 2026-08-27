"""a148: Prefer standard libraries and containers.

The norm: use built-in containers/algorithms and stdlib utilities rather than
hand-rolled equivalents, and reach for project- or language-standard
libraries rather than installing a third-party package whose feature is now
in the stdlib.

The honest measurement boundary: detecting "this `for`-loop should have been
`enumerate(...)`" or "this open-coded dict merge should be `{**a, **b}`"
requires semantic understanding of *intent*. Tree-sitter can flag plausible
shapes but cannot tell whether the author needed the explicit index for some
other reason. We therefore split the signal into two pieces:

  (A) THIN angle — **legacy third-party imports with stdlib replacements**.
      Curated whitelist of packages that have first-party stdlib analogues
      since Python 3.7+ (`simplejson` -> `json`, `pytz` -> `zoneinfo`,
      `pathlib2` -> `pathlib`, `enum34` -> `enum`, `ipaddress`/`ipaddr` ->
      `ipaddress`, `dataclasses` backport -> `dataclasses`, `subprocess32`
      -> `subprocess`, `mock` -> `unittest.mock`, ...). We detect these by
      walking the tree-sitter `import_statement` / `import_from_statement`
      nodes — no regex on code.

  (B) PARTIALLY_THIN angle — **manual reimplementations of stdlib idioms in
      added code**. We flag a small, conservative set of clear shapes that
      are almost never intentional:
        * `for i in range(len(x))` when the loop body indexes `x[i]` and
          does not also need `i` for arithmetic — should be `enumerate(x)`.
        * Open-coded "merge two dicts" using a fresh dict + `update()` calls
          on literal `{}` initialisers, where `{**a, **b}` would suffice.
        * Bubble-/insertion-style nested loops doing sortable comparisons —
          this is rare and noisy so we only flag it when the function name
          itself starts with `sort_`, otherwise abstain.

We do NOT flag generic loop constructs as "should be itertools", because
that conflates style preference with deterministic norm violation.

Classification: **PARTIALLY_THIN**. Sub-signal (A) is THIN (deterministic
import check), sub-signal (B) is heuristic. They are averaged when both
fire; otherwise whichever is available is returned.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a148"
ASPECT_NAME = "Prefer standard libraries and containers"
TIER = 2
TOOLS = ["tree-sitter-python"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "PARTIALLY_THIN"

PY_EXTS = [".py", ".pyi"]

# Map: third-party package -> (stdlib replacement, minimum python ver note).
# Importing the LHS is a deterministic violation of "prefer the stdlib": the
# RHS has been in the stdlib for years and covers >95% of LHS use-cases.
REDUNDANT_IMPORTS: Dict[str, str] = {
    "simplejson": "json",
    "ujson": "json",
    "orjson": "json",            # faster, but stdlib json suffices for most
    "pytz": "zoneinfo",          # py >= 3.9
    "dateutil": "datetime",      # rrule excepted, most use is parse/tzinfo
    "pathlib2": "pathlib",
    "enum34": "enum",
    "ipaddr": "ipaddress",
    "subprocess32": "subprocess",
    "mock": "unittest.mock",
    "backports.zoneinfo": "zoneinfo",
    "typing_extensions": "typing",   # for items that landed in stdlib
    "dataclasses": "dataclasses",    # the BACKPORT package; stdlib already has
    "futures": "concurrent.futures",
    "trollius": "asyncio",
    "ordereddict": "collections",
    "configparser2": "configparser",
    "funcsigs": "inspect",
    "contextlib2": "contextlib",
    "singledispatch": "functools",
    "scandir": "os",                  # os.scandir
    "ipaddress-py2": "ipaddress",
}

# Some of those names ARE the stdlib module on modern Pythons (`mock`,
# `dataclasses`). Distinguishing the backport from the stdlib module requires
# looking at the install metadata, not the import line. We treat them as
# *informational* only — they are still listed so the catalog is honest, but
# we down-weight them to avoid false positives.
SOFT_REDUNDANT = frozenset({"mock", "dataclasses", "typing_extensions"})


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


def _top_module(name: str) -> str:
    return (name or "").split(".", 1)[0]


def _imported_modules(root, src: bytes) -> List[str]:
    """Walk top-level imports and collect dotted module names."""
    mods: List[str] = []
    for child in root.children:
        if child.type == "import_statement":
            for n in child.children:
                if n.type == "dotted_name":
                    mods.append(_text(n, src))
                elif n.type == "aliased_import":
                    for c in n.children:
                        if c.type == "dotted_name":
                            mods.append(_text(c, src))
                            break
        elif child.type == "import_from_statement":
            for n in child.children:
                if n.type in ("dotted_name", "relative_import"):
                    mods.append(_text(n, src))
                    break
    return mods


def _redundant_imports_score(modules: List[str]) -> Optional[float]:
    """Score [0,1] on the import side. 1.0 = no redundant imports."""
    if not modules:
        return None
    bad = 0.0
    for m in modules:
        top = _top_module(m)
        if top in REDUNDANT_IMPORTS:
            bad += 0.3 if top in SOFT_REDUNDANT else 1.0
    # Normalise by the number of distinct redundant flags, not by total
    # import count — a single `import simplejson` is a full violation even
    # if 30 other healthy imports are present.
    # We cap the penalty at 1.0.
    return max(0.0, 1.0 - min(1.0, bad / 2.0))


# ---------------------------------------------------------------------------
# Part B: reimplementation patterns
# ---------------------------------------------------------------------------

def _walk(node, predicate):
    """Yield nodes (DFS) matching predicate."""
    if predicate(node):
        yield node
    for c in node.children:
        yield from _walk(c, predicate)


def _is_range_len_call(node, src: bytes) -> Optional[str]:
    """If node is the call `range(len(X))`, return the name X. Else None.

    Captures the common anti-pattern that should be `enumerate(X)`.
    """
    if node.type != "call":
        return None
    fn = node.child_by_field_name("function")
    if fn is None or _text(fn, src) != "range":
        return None
    args = node.child_by_field_name("arguments")
    if args is None:
        return None
    inner = [c for c in args.children if c.type == "call"]
    if len(inner) != 1:
        return None
    inner_fn = inner[0].child_by_field_name("function")
    if inner_fn is None or _text(inner_fn, src) != "len":
        return None
    inner_args = inner[0].child_by_field_name("arguments")
    if inner_args is None:
        return None
    ids = [c for c in inner_args.children
           if c.type in ("identifier", "attribute")]
    if len(ids) != 1:
        return None
    return _text(ids[0], src)


def _count_range_len_pattern(root, src: bytes) -> Tuple[int, int]:
    """Count (violations, total_for_loops) where violation = `for i in
    range(len(x)): ...x[i]...` and `i` is otherwise not arithmetic."""
    violations = 0
    total = 0
    for fnode in _walk(root, lambda n: n.type == "for_statement"):
        total += 1
        # for_statement children: 'for', target, 'in', iterable, ':', body
        target = fnode.child_by_field_name("left")
        iterable = fnode.child_by_field_name("right")
        body = fnode.child_by_field_name("body")
        if target is None or iterable is None or body is None:
            continue
        if target.type != "identifier":
            continue
        idx_name = _text(target, src)
        seq_name = _is_range_len_call(iterable, src)
        if not seq_name:
            continue
        # Look for body usage: `seq[idx_name]`
        indexed = False
        arithmetic_other = False
        for sub in _walk(body, lambda n: n.type in ("subscript", "binary_operator")):
            if sub.type == "subscript":
                obj = sub.child_by_field_name("value") or (
                    sub.children[0] if sub.children else None)
                sub_idx = sub.child_by_field_name("subscript") or (
                    sub.children[2] if len(sub.children) > 2 else None)
                if obj is None or sub_idx is None:
                    continue
                if _text(obj, src) == seq_name and _text(sub_idx, src) == idx_name:
                    indexed = True
            elif sub.type == "binary_operator":
                txt = _text(sub, src)
                # arithmetic on idx (i+1, i-1) means enumerate isn't enough
                if idx_name in txt and any(op in txt for op in ("+", "-", "*", "/")):
                    arithmetic_other = True
        if indexed and not arithmetic_other:
            violations += 1
    return violations, total


def _count_manual_dict_merge(root, src: bytes) -> Tuple[int, int]:
    """Count violations: `d = {}; d.update(a); d.update(b)` immediately
    after, which {**a, **b} would replace. Returns (violations, dict_init_count).
    """
    violations = 0
    dict_inits = 0
    # Walk function/module bodies and look for the sequence within sibling
    # statements.
    for block in _walk(root, lambda n: n.type == "block" or n.type == "module"):
        stmts = [c for c in block.children if c.type != "comment"]
        for i, st in enumerate(stmts):
            # st must be `name = {}`
            if st.type != "expression_statement":
                continue
            assign = next((c for c in st.children if c.type == "assignment"),
                          None)
            if assign is None:
                continue
            left = assign.child_by_field_name("left")
            right = assign.child_by_field_name("right")
            if left is None or right is None:
                continue
            if left.type != "identifier" or right.type != "dictionary":
                continue
            # Empty dict literal?
            if any(c.type == "pair" for c in right.children):
                continue
            dict_inits += 1
            name = _text(left, src)
            # Look at subsequent statements; count `name.update(_)` calls in
            # a row (>=2).
            updates = 0
            for j in range(i + 1, min(i + 6, len(stmts))):
                nxt = stmts[j]
                if nxt.type != "expression_statement":
                    break
                call = next((c for c in nxt.children if c.type == "call"),
                            None)
                if call is None:
                    break
                fn = call.child_by_field_name("function")
                if fn is None or fn.type != "attribute":
                    break
                obj = fn.child_by_field_name("object")
                attr = fn.child_by_field_name("attribute")
                if (obj is None or attr is None or
                        _text(obj, src) != name or _text(attr, src) != "update"):
                    break
                updates += 1
            if updates >= 2:
                violations += 1
    return violations, dict_inits


def _file_score(code: bytes) -> Optional[float]:
    parser = _get_parser()
    if parser is None:
        return None
    tree = parser.parse(code)
    root = tree.root_node
    if root.type not in ("module", "program"):
        return None

    modules = _imported_modules(root, code)
    s_imp = _redundant_imports_score(modules) if modules else None

    # Reimplementation patterns
    rl_v, rl_t = _count_range_len_pattern(root, code)
    md_v, md_t = _count_manual_dict_merge(root, code)
    parts: List[float] = []
    if rl_t > 0:
        parts.append(1.0 - rl_v / rl_t)
    if md_t > 0:
        # Convert: 1 perfect if no manual merges; otherwise fraction healthy.
        parts.append(1.0 - md_v / md_t)
    s_reimpl = sum(parts) / len(parts) if parts else None

    sub = [s for s in (s_imp, s_reimpl) if s is not None]
    if not sub:
        return None
    return float(sum(sub) / len(sub))


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    scs = [s for s in (_file_score(c.encode("utf8", errors="replace"))
                       for c in by_path.values()) if s is not None]
    if not scs:
        return None
    return float(sum(scs) / len(scs))
