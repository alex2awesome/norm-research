"""a347: Provide alternative text for non-text content (WCAG 1.1.1).

Detects WCAG SC 1.1.1 violations in JSX and HTML added lines:

  - `<img>` (and `<area>`, `<input type="image">`) without an `alt`
    attribute. `alt=""` is OK (decorative). Dynamic `alt={expr}` in JSX
    is accepted as present (we cannot statically resolve `expr`).
  - `<a>` and `<button>` with NO accessible text content (no inner text,
    no `aria-label`, no `aria-labelledby`, no `title`). An empty anchor
    or button is the canonical icon-button accessibility bug that
    eslint-plugin-jsx-a11y/anchor-has-content & button-has-content catch.

Library-first rationale:
  - The canonical tool is `eslint-plugin-jsx-a11y` (multiple WCAG rules,
    designed for JSX/TSX). The shared sandbox does not have a Node
    project rooted in the diff, so a true `eslint --stdin` invocation
    has flag-encoding/peer-dep ambiguity per added-lines snippets. We
    therefore replicate the two highest-precision a11y rules (alt-text,
    anchor/button-has-content) on a tree-sitter AST. This stays a
    structural measurement, never a regex-on-code.
  - For HTML / Vue / Svelte / Handlebars / ERB templates we use
    tree-sitter-html to walk `element` and `self_closing_tag` nodes.
  - For JSX / TSX (and JS/TS, since modern React uses .js/.ts too)
    we use tree-sitter-javascript / tree-sitter-typescript's
    `jsx_element` and `jsx_self_closing_element` nodes.

Applies():
  True iff the diff adds at least one element that the WCAG rules cover
  (an <img>, <area>, <input type="image">, <a>, or <button>). Diffs
  with no JSX/HTML — the vast majority of PRs — abstain. Low
  applicability is correct here: this norm is genuinely narrow.

Score:
  Numerator = elements that satisfy the alt/label requirement.
  Denominator = elements observed.
  None when no relevant element is found (applies()=False) or when no
  parsable file was found (tool failure).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a347"
ASPECT_NAME = "WCAG 1.1.1 alternative text for non-text content"
TIER = 3
TOOLS = ["tree-sitter-html", "tree-sitter-javascript",
         "tree-sitter-typescript"]
APPLIES_TO_LANGS = ["JavaScript", "TypeScript", "HTML", "Vue", "Svelte"]
CLASSIFICATION = "THIN"

# Extensions where JSX may appear. We deliberately include .js/.ts because
# React + React Native projects routinely put JSX in .js files (and the
# tree-sitter-javascript grammar parses JSX natively).
JSX_EXTS = (".jsx", ".tsx", ".js", ".ts", ".mjs", ".cjs")
TS_EXTS = (".ts", ".tsx")
# Extensions for HTML-family templates. Vue/Svelte single-file components
# expose <template> blocks tree-sitter-html parses; Handlebars/ERB/EJS we
# parse as best-effort HTML (the tag syntax we care about — img/a/button —
# is regular HTML inside these templates).
HTML_EXTS = (".html", ".htm", ".vue", ".svelte", ".hbs", ".handlebars",
             ".ejs", ".erb", ".njk")

# ------------------------------------------------------------------
# Parser cache
# ------------------------------------------------------------------

_PARSERS: Dict[str, object] = {}


def _parser_for(kind: str):
    """kind in {'js', 'ts', 'html'}. Returns Parser or None if unavailable."""
    if kind in _PARSERS:
        return _PARSERS[kind]
    try:
        from tree_sitter import Language, Parser
        if kind == "js":
            import tree_sitter_javascript as mod
            lang = Language(mod.language())
        elif kind == "ts":
            import tree_sitter_typescript as mod
            # tsx grammar handles both .ts and .tsx tags
            lang = Language(mod.language_tsx())
        elif kind == "html":
            import tree_sitter_html as mod
            lang = Language(mod.language())
        else:
            _PARSERS[kind] = None
            return None
        _PARSERS[kind] = Parser(lang)
    except Exception:
        _PARSERS[kind] = None
    return _PARSERS[kind]


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


def _in_error(node) -> bool:
    cur = node.parent
    while cur is not None:
        if cur.type == "ERROR":
            return True
        cur = cur.parent
    return False


# ------------------------------------------------------------------
# JSX walker
# ------------------------------------------------------------------

# Elements requiring an alt attribute (jsx-a11y/alt-text).
JSX_ALT_TAGS = {"img", "area"}
# Elements that must have accessible text content
# (jsx-a11y/anchor-has-content, jsx-a11y/heading-has-content,
# button-name family).
JSX_CONTENT_TAGS = {"a", "button"}

# Attributes that supply an accessible name when inner text is absent.
ARIA_LABEL_ATTRS = {"aria-label", "aria-labelledby", "title"}


def _jsx_attrs(opening_or_self, src: bytes) -> Dict[str, Optional[str]]:
    """Return {attr_name: literal_value_or_None}.

    Value None means the attribute exists but is dynamic ({expr}) or its
    literal value isn't statically resolvable. Absence from the dict
    means the attribute is not on the tag.
    """
    out: Dict[str, Optional[str]] = {}
    for ch in opening_or_self.children:
        if ch.type != "jsx_attribute":
            continue
        name_node = None
        value_node = None
        for sub in ch.children:
            if sub.type == "property_identifier" and name_node is None:
                name_node = sub
            elif sub.type == "jsx_namespace_name" and name_node is None:
                # not standard but appears in some TS configs
                name_node = sub
            elif sub.type == "string":
                value_node = sub
            elif sub.type == "jsx_expression":
                value_node = sub
        if name_node is None:
            continue
        attr_name = _text(name_node, src).lower()
        if value_node is None:
            # bare attr like <img alt> — counts as PRESENT but unknown value
            out[attr_name] = None
            continue
        if value_node.type == "string":
            # strip quotes
            raw = _text(value_node, src)
            if len(raw) >= 2 and raw[0] in "\"'":
                out[attr_name] = raw[1:-1]
            else:
                out[attr_name] = raw
        else:
            # dynamic JSX expression — attribute is present but value unknown
            out[attr_name] = None
    return out


def _jsx_has_inner_text(jsx_element_node, src: bytes) -> bool:
    """True if the jsx_element has nontrivial text/child content.

    "Nontrivial" = any jsx_text with non-whitespace OR any nested
    jsx_element / jsx_expression with content.
    """
    for ch in jsx_element_node.children:
        if ch.type == "jsx_text":
            if _text(ch, src).strip():
                return True
        elif ch.type in ("jsx_element", "jsx_self_closing_element",
                         "jsx_expression", "jsx_fragment"):
            return True
    return False


def _walk_jsx(node, src: bytes, results: List[Tuple[str, bool]]):
    """Walk JSX AST. Append (tag_name, ok) per relevant element.

    `ok` = True iff the element satisfies the accessibility rule.
    """
    t = node.type
    if t == "jsx_self_closing_element" and not _in_error(node):
        tag = None
        for ch in node.children:
            if ch.type == "identifier":
                tag = _text(ch, src)
                break
        if tag is not None:
            tlow = tag.lower()
            if tlow in JSX_ALT_TAGS:
                attrs = _jsx_attrs(node, src)
                results.append((tlow, "alt" in attrs))
            elif tlow == "input":
                attrs = _jsx_attrs(node, src)
                # only flag input type="image"
                if (attrs.get("type") or "").lower() == "image":
                    results.append(("input[image]", "alt" in attrs))
            elif tlow in JSX_CONTENT_TAGS:
                # self-closing <button /> or <a /> with no children: must
                # have aria-label/aria-labelledby/title to be accessible
                attrs = _jsx_attrs(node, src)
                ok = any(a in attrs for a in ARIA_LABEL_ATTRS)
                results.append((tlow, ok))
    elif t == "jsx_element" and not _in_error(node):
        # find jsx_opening_element child
        opening = None
        for ch in node.children:
            if ch.type == "jsx_opening_element":
                opening = ch
                break
        if opening is not None:
            tag = None
            for ch in opening.children:
                if ch.type == "identifier":
                    tag = _text(ch, src)
                    break
            if tag is not None:
                tlow = tag.lower()
                if tlow in JSX_ALT_TAGS:
                    attrs = _jsx_attrs(opening, src)
                    results.append((tlow, "alt" in attrs))
                elif tlow == "input":
                    attrs = _jsx_attrs(opening, src)
                    if (attrs.get("type") or "").lower() == "image":
                        results.append(("input[image]", "alt" in attrs))
                elif tlow in JSX_CONTENT_TAGS:
                    attrs = _jsx_attrs(opening, src)
                    has_label = any(a in attrs for a in ARIA_LABEL_ATTRS)
                    has_text = _jsx_has_inner_text(node, src)
                    results.append((tlow, has_label or has_text))
    for c in node.children:
        _walk_jsx(c, src, results)


def _measure_jsx(code: bytes, ts: bool) -> List[Tuple[str, bool]]:
    parser = _parser_for("ts" if ts else "js")
    if parser is None:
        return []
    tree = parser.parse(code)
    out: List[Tuple[str, bool]] = []
    _walk_jsx(tree.root_node, code, out)
    return out


# ------------------------------------------------------------------
# HTML walker
# ------------------------------------------------------------------

HTML_ALT_TAGS = {"img", "area"}
HTML_CONTENT_TAGS = {"a", "button"}


def _html_attrs(tag_node, src: bytes) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for ch in tag_node.children:
        if ch.type != "attribute":
            continue
        name = None
        val = None
        for sub in ch.children:
            if sub.type == "attribute_name":
                name = _text(sub, src).lower()
            elif sub.type == "quoted_attribute_value":
                # contains an inner attribute_value
                inner = None
                for q in sub.children:
                    if q.type == "attribute_value":
                        inner = _text(q, src)
                        break
                val = inner if inner is not None else ""
            elif sub.type == "attribute_value":
                val = _text(sub, src)
        if name is None:
            continue
        out[name] = val
    return out


def _html_tag_name(start_or_self_node, src: bytes) -> Optional[str]:
    for ch in start_or_self_node.children:
        if ch.type == "tag_name":
            return _text(ch, src).lower()
    return None


def _html_element_inner_text(element_node, src: bytes) -> bool:
    """True iff element has nontrivial text or descendant element children."""
    for ch in element_node.children:
        if ch.type == "text":
            if _text(ch, src).strip():
                return True
        elif ch.type == "element":
            return True
    return False


def _walk_html(node, src: bytes, results: List[Tuple[str, bool]]):
    t = node.type
    if t == "self_closing_tag":
        tag = _html_tag_name(node, src)
        if tag in HTML_ALT_TAGS:
            attrs = _html_attrs(node, src)
            results.append((tag, "alt" in attrs))
        elif tag == "input":
            attrs = _html_attrs(node, src)
            if (attrs.get("type") or "").lower() == "image":
                results.append(("input[image]", "alt" in attrs))
    elif t == "element":
        # locate start_tag
        start = None
        for ch in node.children:
            if ch.type == "start_tag":
                start = ch
                break
            if ch.type == "self_closing_tag":
                # void-style self-closing inside element node
                start = ch
                break
        if start is not None:
            tag = _html_tag_name(start, src)
            if tag in HTML_ALT_TAGS:
                attrs = _html_attrs(start, src)
                results.append((tag, "alt" in attrs))
            elif tag == "input":
                attrs = _html_attrs(start, src)
                if (attrs.get("type") or "").lower() == "image":
                    results.append(("input[image]", "alt" in attrs))
            elif tag in HTML_CONTENT_TAGS:
                attrs = _html_attrs(start, src)
                has_label = any(a in attrs for a in ARIA_LABEL_ATTRS)
                has_text = _html_element_inner_text(node, src)
                results.append((tag, has_label or has_text))
    for c in node.children:
        _walk_html(c, src, results)


def _measure_html(code: bytes) -> List[Tuple[str, bool]]:
    parser = _parser_for("html")
    if parser is None:
        return []
    tree = parser.parse(code)
    out: List[Tuple[str, bool]] = []
    _walk_html(tree.root_node, code, out)
    return out


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

def _collect(diff_text: str) -> List[Tuple[str, bool]]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return []
    out: List[Tuple[str, bool]] = []
    for path, body in by_path.items():
        pl = path.lower()
        code = body.encode("utf8", errors="replace")
        if pl.endswith(HTML_EXTS):
            out.extend(_measure_html(code))
        elif pl.endswith(TS_EXTS):
            out.extend(_measure_jsx(code, ts=True))
        elif pl.endswith(JSX_EXTS):
            out.extend(_measure_jsx(code, ts=False))
    return out


def applies(diff_text: str) -> bool:
    # Cheap pre-filter: at least one file with a relevant extension.
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return False
    if not any(p.lower().endswith(JSX_EXTS + HTML_EXTS)
               for p in by_path):
        return False
    # Now confirm at least one covered element actually exists.
    return len(_collect(diff_text)) > 0


def score(diff_text: str) -> Optional[float]:
    items = _collect(diff_text)
    if not items:
        return None
    ok = sum(1 for _, good in items if good)
    return float(ok / len(items))
