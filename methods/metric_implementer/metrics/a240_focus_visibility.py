"""a240: Focus visibility and non-obscuration (WCAG SC 2.4.7 / 2.4.11).

Keyboard focus indicators must remain visible. The classic anti-pattern is
``outline: none`` (or ``outline: 0``) on an interactive element WITHOUT a
replacement focus style (a ``:focus`` / ``:focus-visible`` rule that paints
an alternative visible indicator such as box-shadow, border, or another
outline). This is the canonical lint rule ``a11y/no-outline-none``
(stylelint plugin) checks.

Library-first rationale:
  - The canonical CLI tool is `stylelint` with the
    `stylelint-a11y` plugin (`a11y/no-outline-none`), but stylelint
    requires a node_modules + plugin install rooted in a project, which
    the sandbox does not provide. We therefore replicate the rule using
    a CSS parser — `tinycss2` — to walk declarations structurally. No
    regex on code.
  - tinycss2 parses CSS/SCSS-ish source into qualified rules and
    declarations; for each rule containing ``outline: none|0|none-like``
    we check whether the same stylesheet contains a paired ``:focus``
    or ``:focus-visible`` rule on the same selector that defines
    *some* alternative visible focus indicator (outline-color,
    box-shadow, border, background, color change).

What we check per CSS-family file in the added lines:
  Numerator   = number of ``outline: none|0`` rules that EITHER are
                themselves ``:focus``/``:focus-visible`` rules with an
                alternative indicator, OR have a sibling ``:focus`` /
                ``:focus-visible`` rule (matching base selector) that
                supplies one.
  Denominator = total number of ``outline: none|0`` declarations found.

Score:
  - 1.0  : every outline:none has a paired focus replacement
  - 0.0  : no outline:none has a replacement
  - in between: partial coverage
  - None : applies()=False (no CSS-family file with the pattern) OR
           tinycss2 parse failure on every file (tool failure).

Applies():
  True iff the diff adds at least one CSS-family file (.css/.scss/.sass/
  .less/.styl) AND that file contains at least one ``outline`` declaration
  whose value is ``none`` or ``0``. Without that decl the norm has nothing
  to measure on this diff and we abstain.

Classification:
  PARTIALLY_THIN. Replacement detection is heuristic (we treat any
  box-shadow / border / outline-color / background-color declaration in a
  :focus rule as "replacement supplied"). The thin part is the structural
  detection of ``outline: none``; the partially-thin part is whether the
  alternative is *actually visible*, which requires color-contrast
  computation we don't perform.

Reference: a347 (WCAG alt-text, narrow JSX) for the per-file narrow
applies() pattern, and a184 (narrow tool-checked) for the parse-then-walk
shape.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a240"
ASPECT_NAME = "Focus visibility and non-obscuration"
TIER = 2
TOOLS = ["tinycss2"]
APPLIES_TO_LANGS = ["CSS", "SCSS", "SASS", "LESS", "Stylus"]
CLASSIFICATION = "PARTIALLY_THIN"

CSS_EXTS = (".css", ".scss", ".sass", ".less", ".styl")

# Declarations in a :focus rule that we treat as "replacement focus
# indicator supplied". This is the same set the stylelint a11y plugin
# recognizes (modulo color-contrast).
REPLACEMENT_PROPS = {
    "box-shadow",
    "border",
    "border-color",
    "border-width",
    "border-style",
    "outline-color",
    "outline-style",
    "outline-width",
    "outline-offset",
    "background",
    "background-color",
    "color",
    "filter",
    # If they explicitly set outline to a non-none value in :focus.
    "outline",
}

FOCUS_PSEUDOS = (":focus", ":focus-visible", ":focus-within")


# ------------------------------------------------------------------
# tinycss2 helpers
# ------------------------------------------------------------------

def _serialize(tokens) -> str:
    try:
        import tinycss2
        return tinycss2.serialize(tokens)
    except Exception:
        return ""


def _value_is_none_or_zero(value_tokens) -> bool:
    """True if the declaration value is effectively ``none`` or ``0`` —
    ignoring whitespace/comments. We accept multi-token shorthand like
    ``0 none`` as well: any token matching ``none``/``0``/``0px`` with no
    other non-trivial token (ident other than ``none``, length other than
    0) counts.
    """
    saw_killer = False
    for t in value_tokens or []:
        tt = t.type
        if tt in ("whitespace", "comment"):
            continue
        if tt == "ident":
            if t.value.lower() == "none":
                saw_killer = True
                continue
            return False
        if tt == "number":
            # tinycss2 numbers expose int_value (or value)
            iv = getattr(t, "int_value", None)
            v = getattr(t, "value", None)
            if iv == 0 or v == 0:
                saw_killer = True
                continue
            return False
        if tt == "dimension":
            # 0px, 0em — treat as zero
            v = getattr(t, "value", None)
            if v == 0:
                saw_killer = True
                continue
            return False
        # any other token type (function, hash, percentage…) → not none/0
        return False
    return saw_killer


def _base_selector(prelude_tokens) -> str:
    """Return the prelude with focus-family pseudo-class stripped, so we
    can match a ``.btn:focus { box-shadow }`` rule against a sibling
    ``.btn { outline: none }`` rule.

    We do a small structural normalization: drop whitespace, lowercase,
    strip any trailing ``:focus*`` segment from each compound selector.
    Comma-separated lists are split and each piece normalized.
    """
    raw = _serialize(prelude_tokens).strip().lower()
    if not raw:
        return ""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    norm: List[str] = []
    for p in parts:
        # repeatedly strip focus pseudos from the end of each compound
        s = p
        for pseudo in FOCUS_PSEUDOS:
            while s.endswith(pseudo):
                s = s[: -len(pseudo)].rstrip()
        # also strip mid-token focus pseudos inside the compound
        for pseudo in FOCUS_PSEUDOS:
            s = s.replace(pseudo, "")
        s = " ".join(s.split())  # collapse whitespace
        if s:
            norm.append(s)
    return ",".join(sorted(norm))


def _has_focus_pseudo(prelude_tokens) -> bool:
    raw = _serialize(prelude_tokens).lower()
    return any(p in raw for p in FOCUS_PSEUDOS)


def _walk_rules(rules, qualified: List, at_rules: List):
    """Flatten rule list, recursing into nested at-rules (@media, @supports)
    so we catch ``@media (...) { .btn { outline: none } }``.
    """
    for r in rules or []:
        if r.type == "qualified-rule":
            qualified.append(r)
        elif r.type == "at-rule":
            at_rules.append(r)
            # Recurse into the block content of @media / @supports / @layer
            block = getattr(r, "content", None)
            if block is not None:
                try:
                    import tinycss2
                    nested = tinycss2.parse_rule_list(block)
                    _walk_rules(nested, qualified, at_rules)
                except Exception:
                    pass


def _parse_decls(block_tokens):
    """Return list of (prop_lower, value_tokens) declarations from a block."""
    try:
        import tinycss2
        decls = tinycss2.parse_declaration_list(
            block_tokens, skip_comments=True, skip_whitespace=True
        )
    except Exception:
        return []
    out = []
    for d in decls:
        if d.type != "declaration":
            continue
        out.append((d.name.lower(), d.value))
    return out


def _measure_css(code: str) -> Optional[Tuple[int, int]]:
    """Return (num_outline_none_decls, num_with_replacement) or None on
    total parse failure.
    """
    try:
        import tinycss2
        rules = tinycss2.parse_stylesheet(
            code, skip_comments=True, skip_whitespace=True
        )
    except Exception:
        return None
    qualified: List = []
    at_rules: List = []
    _walk_rules(rules, qualified, at_rules)
    if not qualified:
        return (0, 0)

    # Build index: base_selector -> list of (is_focus_rule, decls)
    base_index: Dict[str, List[Tuple[bool, List[Tuple[str, object]]]]] = {}
    rule_records: List[Tuple[str, bool, List[Tuple[str, object]]]] = []
    for r in qualified:
        base = _base_selector(r.prelude)
        is_focus = _has_focus_pseudo(r.prelude)
        decls = _parse_decls(r.content)
        base_index.setdefault(base, []).append((is_focus, decls))
        rule_records.append((base, is_focus, decls))

    total = 0
    ok = 0
    for base, is_focus, decls in rule_records:
        # Find this rule's outline:none / outline:0 declarations.
        outline_killers = [
            (prop, val) for (prop, val) in decls
            if prop in ("outline", "outline-style", "outline-width")
            and _value_is_none_or_zero(val)
        ]
        # outline-style:none / outline-width:0 are the explicit shorthands.
        # We focus on the ``outline`` shorthand and ``outline-style:none``
        # since outline-width:0 is the same effective bug.
        outline_killers = [
            (p, v) for (p, v) in outline_killers
            if p in ("outline", "outline-style") or
               (p == "outline-width" and _value_is_none_or_zero(v))
        ]
        if not outline_killers:
            continue
        total += len(outline_killers)

        # Does THIS rule supply a replacement?
        local_replacement = any(
            prop in REPLACEMENT_PROPS and not (
                prop == "outline" and _value_is_none_or_zero(val)
            )
            for (prop, val) in decls
            if prop != "outline-style" and prop != "outline-width"
        )
        # Does a sibling :focus / :focus-visible rule with the same base
        # selector supply a replacement?
        sibling_replacement = False
        for (sib_is_focus, sib_decls) in base_index.get(base, []):
            if not sib_is_focus:
                continue
            if any(prop in REPLACEMENT_PROPS and not (
                       prop == "outline" and _value_is_none_or_zero(val))
                   for (prop, val) in sib_decls):
                sibling_replacement = True
                break

        if is_focus and local_replacement:
            ok += len(outline_killers)
        elif sibling_replacement:
            ok += len(outline_killers)
        # else: outline-killer with no replacement → counts against score

    return (total, ok)


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

def _collect(diff_text: str) -> Optional[Tuple[int, int]]:
    """Sum (total, ok) over all CSS-family added files. Return None if no
    CSS file parsed (tool failure on everything) or no files at all.
    """
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    total_sum = 0
    ok_sum = 0
    any_parsed = False
    any_killers = False
    for path, body in by_path.items():
        if not path.lower().endswith(CSS_EXTS):
            continue
        res = _measure_css(body)
        if res is None:
            continue  # parse failure on this file
        any_parsed = True
        t, k = res
        total_sum += t
        ok_sum += k
        if t > 0:
            any_killers = True
    if not any_parsed:
        return None
    if not any_killers:
        # CSS present but no outline:none anywhere — norm doesn't apply.
        return (0, 0)
    return (total_sum, ok_sum)


def applies(diff_text: str) -> bool:
    # Cheap pre-filter: at least one CSS-family file added.
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return False
    if not any(p.lower().endswith(CSS_EXTS) for p in by_path):
        return False
    # Expensive confirm: must actually find at least one outline:none/0.
    res = _collect(diff_text)
    if res is None:
        return False
    total, _ok = res
    return total > 0


def score(diff_text: str) -> Optional[float]:
    res = _collect(diff_text)
    if res is None:
        return None
    total, ok = res
    if total == 0:
        return None
    return float(ok / total)
