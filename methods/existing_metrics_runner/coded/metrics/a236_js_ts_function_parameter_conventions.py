"""a236: JS/TS function parameter conventions.

Aspect description: "Keep function signatures predictable and type-safe:
prefer rest parameters over `arguments`, place default parameters last, and
keep parameter destructuring simple."

We tally per JS/TS function (`function_declaration`, `function_expression`,
`arrow_function`, `method_definition`, plus generator variants) via
tree-sitter, then combine four sub-scores:

  (1) **Rest over `arguments`** — functions using `...rest` are conforming;
      functions whose body references the `arguments` identifier are
      penalized. Arrow functions don't HAVE `arguments`, so an arrow
      reading `arguments` is closure-leaking and is also penalized.

  (2) **Defaults last** — within one parameter list, every parameter with a
      default value (`assignment_pattern` in JS; `required_parameter` /
      `optional_parameter` carrying a `=` initializer in TS) must occur at or
      after every parameter without one. We count the number of "default
      then non-default" adjacencies as the violation.

  (3) **Simple destructuring** — `object_pattern` / `array_pattern` used as
      a parameter is fine *iff* it stays shallow. Each pattern is scored
      1.0 if it has <= 5 surface bindings and depth <= 1, decaying to 0
      as it grows. Deep nested destructuring (e.g.
      `{a: {b: {c}}}`) is the anti-pattern.

  (4) **Explicit parameter types (TS only)** — for `.ts`/`.tsx` files, each
      `required_parameter` / `optional_parameter` should carry a
      `type_annotation`. JS files don't penalize here (no type system).

Each sub-score is mean over the relevant functions/params in the diff;
final score is the unweighted mean of whichever sub-scores have a denominator.

CLASSIFICATION = PARTIALLY_THIN: the rest/arguments and defaults-last
checks correspond to deterministic ESLint rules (`prefer-rest-params`,
`default-param-last`), but "simple destructuring" is a soft style threshold
that doesn't map to a single universal rule.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a236"
ASPECT_NAME = "JS/TS function parameter conventions"
TIER = 2
TOOLS = ["tree-sitter-javascript", "tree-sitter-typescript"]
APPLIES_TO_LANGS = ["JavaScript", "TypeScript"]
CLASSIFICATION = "PARTIALLY_THIN"

JS_EXTS = (".js", ".jsx", ".mjs", ".cjs")
TS_EXTS = (".ts", ".tsx")

FUNCTION_NODE_TYPES = frozenset({
    "function_declaration",
    "function_expression",
    "arrow_function",
    "method_definition",
    "generator_function_declaration",
    "generator_function",
})

# Surface-binding budget for "simple" destructuring.
SIMPLE_BINDING_LIMIT = 5
SIMPLE_DEPTH_LIMIT = 1  # one level of pattern; nested patterns penalised


_PARSERS: Dict[str, object] = {}


def _get_parser(lang: str):
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "js":
            import tree_sitter_javascript as m
            L = Language(m.language())
        elif lang == "ts":
            import tree_sitter_typescript as m
            L = Language(m.language_typescript())
        elif lang == "tsx":
            import tree_sitter_typescript as m
            L = Language(m.language_tsx())
        else:
            _PARSERS[lang] = None
            return None
        _PARSERS[lang] = Parser(L)
    except Exception:
        _PARSERS[lang] = None
    return _PARSERS[lang]


def _lang_for_ext(ext: str) -> Optional[str]:
    ext = ext.lower()
    if ext in JS_EXTS:
        return "js"
    if ext == ".tsx":
        return "tsx"
    if ext in TS_EXTS:
        return "ts"
    return None


# ---- per-parameter classification ------------------------------------------

def _is_rest(param) -> bool:
    """A parameter is a rest param if it is or contains a `rest_pattern`.

    JS: the parameter node itself is `rest_pattern`.
    TS: `required_parameter` containing a `rest_pattern` child.
    """
    if param.type == "rest_pattern":
        return True
    for c in param.children:
        if c.type == "rest_pattern":
            return True
    return False


def _has_default(param) -> bool:
    """Has an `=` initializer.

    JS: parameter is `assignment_pattern`.
    TS: `required_parameter` with an `=` token child OR `optional_parameter`
    with an initializer.
    """
    if param.type == "assignment_pattern":
        return True
    # optional_parameter has `?` after name — that is *optional*, not
    # defaulted; only count if there's an actual initializer.
    if param.type in ("required_parameter", "optional_parameter"):
        # Look for an `=` token child followed by a value.
        # tree-sitter exposes the literal `=` as its own child node.
        for c in param.children:
            if c.type == "=":
                return True
    return False


def _pattern_root(param):
    """Return the pattern node (object_pattern / array_pattern) if this
    parameter destructures, else None.

    Handles both the JS shape (param IS the pattern) and the TS shape
    (`required_parameter` whose first non-decorator child is the pattern).
    """
    if param.type in ("object_pattern", "array_pattern"):
        return param
    if param.type in ("required_parameter", "optional_parameter"):
        for c in param.children:
            if c.type in ("object_pattern", "array_pattern"):
                return c
    # `assignment_pattern` may wrap a pattern with a default: `{a}={}`.
    if param.type == "assignment_pattern":
        for c in param.children:
            if c.type in ("object_pattern", "array_pattern"):
                return c
    return None


def _pattern_depth_bindings(node, depth: int = 0) -> Tuple[int, int]:
    """Walk a pattern; return (max_depth, n_leaf_bindings)."""
    max_d = depth
    n_bind = 0
    if node.type in ("identifier", "shorthand_property_identifier_pattern"):
        return depth, 1
    for c in node.children:
        if c.type in ("object_pattern", "array_pattern"):
            d, b = _pattern_depth_bindings(c, depth + 1)
            if d > max_d:
                max_d = d
            n_bind += b
        elif c.type in ("pair_pattern", "rest_pattern", "assignment_pattern",
                        "object_assignment_pattern"):
            d, b = _pattern_depth_bindings(c, depth)
            if d > max_d:
                max_d = d
            n_bind += b
        elif c.type in ("identifier",
                        "shorthand_property_identifier_pattern"):
            n_bind += 1
    return max_d, n_bind


def _has_type_annotation(param) -> bool:
    """TS only: `required_parameter` / `optional_parameter` with a
    `type_annotation` child."""
    if param.type not in ("required_parameter", "optional_parameter"):
        return False
    for c in param.children:
        if c.type == "type_annotation":
            return True
    return False


# ---- function-level walk ----------------------------------------------------

def _params_of(fn_node) -> List[object]:
    """Return parameter nodes (excluding punctuation) for a function node."""
    fp = None
    for c in fn_node.children:
        if c.type == "formal_parameters":
            fp = c
            break
    if fp is None:
        return []
    out = []
    for c in fp.children:
        if c.type in ("(", ")", ","):
            continue
        out.append(c)
    return out


def _body_uses_arguments(fn_node, src: bytes) -> bool:
    """Does the function body reference the `arguments` identifier?

    We walk descendants but stop at nested function nodes (those have their
    own `arguments` binding for non-arrow functions; arrow nested counts as
    its enclosing function, but we conservatively scope per function).
    """
    body = None
    for c in fn_node.children:
        if c.type in ("statement_block",):
            body = c
            break
        # arrow_function may have an expression body directly.
        if fn_node.type == "arrow_function" and \
                c.type not in ("formal_parameters", "(", ")",
                               "=>", "type_annotation"):
            body = c
            break
    if body is None:
        return False

    found = [False]

    def walk(n):
        if found[0]:
            return
        # stop descending into nested non-arrow functions
        if n is not body and n.type in (
            "function_declaration", "function_expression",
            "method_definition", "generator_function_declaration",
            "generator_function",
        ):
            return
        if n.type == "identifier":
            if src[n.start_byte:n.end_byte] == b"arguments":
                found[0] = True
                return
        for c in n.children:
            walk(c)

    walk(body)
    return found[0]


def _tally(code: bytes, lang: str) -> Optional[Dict[str, int]]:
    parser = _get_parser(lang)
    if parser is None:
        return None
    root = parser.parse(code).root_node
    is_ts = lang in ("ts", "tsx")

    counts = {
        # (1) rest / arguments
        "fn_total": 0,
        "fn_uses_arguments": 0,
        "fn_uses_rest": 0,
        # (2) defaults last (per function-with-defaults)
        "fn_with_defaults": 0,
        "fn_defaults_last_violations": 0,
        # (3) destructuring complexity (per destructuring param)
        "destruct_params": 0,
        "destruct_simple": 0,
        # (4) TS typing (per TS param in TS files)
        "ts_param_total": 0,
        "ts_param_typed": 0,
    }

    def walk(n):
        if n.type in FUNCTION_NODE_TYPES:
            counts["fn_total"] += 1
            params = _params_of(n)

            # (1) arguments / rest
            uses_args = _body_uses_arguments(n, code)
            uses_rest = any(_is_rest(p) for p in params)
            if uses_args:
                counts["fn_uses_arguments"] += 1
            if uses_rest:
                counts["fn_uses_rest"] += 1

            # (2) defaults-last: count default-then-nondefault adjacencies
            #     among non-rest params. Rest is always last anyway.
            non_rest = [p for p in params if not _is_rest(p)]
            defaults = [_has_default(p) for p in non_rest]
            has_any_default = any(defaults)
            if has_any_default:
                counts["fn_with_defaults"] += 1
                violations = sum(
                    1 for a, b in zip(defaults, defaults[1:])
                    if a and not b
                )
                # binary: pass iff zero violations
                if violations > 0:
                    counts["fn_defaults_last_violations"] += 1

            # (3) destructuring complexity per destructuring param
            for p in params:
                pat = _pattern_root(p)
                if pat is None:
                    continue
                counts["destruct_params"] += 1
                depth, n_bind = _pattern_depth_bindings(pat, 0)
                if (depth <= SIMPLE_DEPTH_LIMIT and
                        n_bind <= SIMPLE_BINDING_LIMIT):
                    counts["destruct_simple"] += 1

            # (4) TS typing — only meaningful in TS files
            if is_ts:
                for p in params:
                    # rest parameters in TS still go through required_parameter
                    if p.type in ("required_parameter", "optional_parameter"):
                        counts["ts_param_total"] += 1
                        if _has_type_annotation(p):
                            counts["ts_param_typed"] += 1

        for c in n.children:
            walk(c)

    walk(root)
    return counts


def _collect(diff_text: str) -> Optional[Dict[str, int]]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    agg: Optional[Dict[str, int]] = None
    saw_lang = False
    for path, body in by_path.items():
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        lang = _lang_for_ext(ext)
        if lang is None:
            continue
        saw_lang = True
        c = _tally(body.encode("utf8", errors="replace"), lang)
        if c is None:
            continue
        if agg is None:
            agg = dict(c)
        else:
            for k, v in c.items():
                agg[k] = agg.get(k, 0) + v
    if not saw_lang:
        return None
    return agg


def applies(diff_text: str) -> bool:
    """Apply iff the diff touches a JS/TS file AND we observe at least one
    function with at least one parameter (otherwise nothing to score)."""
    c = _collect(diff_text)
    if not c:
        return False
    # need at least some function to score
    return c.get("fn_total", 0) > 0 and (
        c.get("fn_with_defaults", 0) > 0
        or c.get("destruct_params", 0) > 0
        or c.get("ts_param_total", 0) > 0
        or c.get("fn_uses_arguments", 0) > 0
        or c.get("fn_uses_rest", 0) > 0
        or c.get("fn_total", 0) > 0
    )


def score(diff_text: str) -> Optional[float]:
    c = _collect(diff_text)
    if not c:
        return None
    fn_total = c.get("fn_total", 0)
    if fn_total == 0:
        return None

    sub_scores: List[float] = []

    # (1) rest over arguments — penalty per function using `arguments`.
    # Functions that don't use `arguments` are conforming (whether or not
    # they happen to use rest).
    s1 = 1.0 - (c["fn_uses_arguments"] / fn_total)
    sub_scores.append(s1)

    # (2) defaults last — only over functions that actually use defaults.
    if c["fn_with_defaults"] > 0:
        s2 = 1.0 - (c["fn_defaults_last_violations"] / c["fn_with_defaults"])
        sub_scores.append(s2)

    # (3) simple destructuring — only over functions with destructured params.
    if c["destruct_params"] > 0:
        s3 = c["destruct_simple"] / c["destruct_params"]
        sub_scores.append(s3)

    # (4) TS explicit param types — only over TS-file parameters.
    if c["ts_param_total"] > 0:
        s4 = c["ts_param_typed"] / c["ts_param_total"]
        sub_scores.append(s4)

    if not sub_scores:
        # We had functions but no measurable axis fired (e.g. JS file with
        # all positional params, no defaults, no destructuring, no
        # `arguments` use). Treat as trivially compliant.
        return 1.0
    return float(sum(sub_scores) / len(sub_scores))
