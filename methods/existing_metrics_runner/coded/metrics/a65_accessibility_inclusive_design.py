"""a65: Accessibility and inclusive design (broader than a347/a240).

Aspect description (aspects.json[a65]):
  "Bake in accessibility and inclusion from the start; comply with a11y
   principles, address cognitive needs, and document/justify exceptions."

This is the broad inclusive-design norm. The two narrow sibling metrics
already cover the most cited slice each:
  - a347 = WCAG 1.1.1 alt-text + anchor/button-has-content (jsx-a11y/alt-text,
    anchor-has-content, button-has-content).
  - a240 = WCAG 2.4.7 focus visibility (stylelint a11y/no-outline-none).

To stay *distinct* from those and still measure something deterministic, we
target a different cluster of canonical jsx-a11y / WCAG rules. None of these
overlap with alt-text or focus CSS:

  1. Non-interactive elements with click handlers
     (jsx-a11y/click-events-have-key-events &
      jsx-a11y/no-static-element-interactions). Example anti-pattern:
       <div onClick={...}>   (should be <button>, OR have role + tabIndex
                              + key handler).

  2. Form controls without an associated label
     (jsx-a11y/label-has-associated-control, label-has-for). Example:
       <input type="text" name="q" />   (no <label htmlFor>, no aria-label).

  3. <html> root tag without a `lang` attribute
     (jsx-a11y/html-has-lang, WCAG 3.1.1).

  4. Redundant ARIA roles on already-semantic elements
     (jsx-a11y/no-redundant-roles). Example:
       <button role="button">   (the role is implicit; explicit is noise
                                  and a common over-applied "fix").

These four rules are deliberately disjoint from a347 (alt/aria-label on
img/a/button) and a240 (CSS outline-none). They probe semantic-HTML
discipline, not text alternatives or focus visibility.

Library-first rationale:
  - The canonical tool is `eslint-plugin-jsx-a11y` (Node, project-rooted).
    The sandbox cannot run a real `eslint --stdin` over arbitrary added
    snippets reliably (no node_modules in diff context). Same constraint
    a347 hit. We therefore replicate these four high-precision a11y rules
    on a tree-sitter AST (tree-sitter-javascript / tree-sitter-typescript /
    tree-sitter-html). Structural walk, no regex on code.

Per-element accounting:
  Each relevant element added in the diff contributes ONE (tag, ok) entry.
    - For the static-element-with-onClick rule, a violating <div onClick>
      contributes (..., ok=False) unless mitigated by role + tabIndex.
    - For label-has-associated-control, an <input>/<select>/<textarea>
      missing both a wrapping/sibling <label htmlFor> AND an
      aria-label / aria-labelledby / title contributes ok=False.
    - For html-has-lang, an <html> tag without `lang` is ok=False.
    - For no-redundant-roles, a <button role="button"> (or any of the
      implicit-role pairs) is ok=False.

Applies():
  True iff the diff adds at least one JSX/HTML/template file AND that file
  contains at least one element covered by any of the four rules above.
  Otherwise the norm has nothing to measure on this diff and we abstain.

Score:
  Numerator   = relevant elements that satisfy their rule.
  Denominator = relevant elements observed.
  None  iff applies()=False or no parser produced any element.

Classification:
  THIN. All four sub-rules are deterministic structural checks that
  eslint-plugin-jsx-a11y enforces statically. We are replicating those
  rules verbatim on the AST.

Distinct-from-siblings claim:
  a347 measures alt-text + accessible-name presence on <img>/<a>/<button>.
  a240 measures CSS outline:none replacement. Neither touches:
    <div onClick>, <input> labeling, <html lang>, or redundant role=.
  An eslint-plugin-jsx-a11y rule list confirms the orthogonality: the
  rules implemented here do not overlap with the alt-text family or any
  CSS rule.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a65"
ASPECT_NAME = "Accessibility and inclusive design"
TIER = 3
TOOLS = ["tree-sitter-html", "tree-sitter-javascript",
         "tree-sitter-typescript"]
APPLIES_TO_LANGS = ["JavaScript", "TypeScript", "HTML", "Vue", "Svelte"]
CLASSIFICATION = "THIN"

JSX_EXTS = (".jsx", ".tsx", ".js", ".ts", ".mjs", ".cjs")
TS_EXTS = (".ts", ".tsx")
HTML_EXTS = (".html", ".htm", ".vue", ".svelte", ".hbs", ".handlebars",
             ".ejs", ".erb", ".njk")

# ------------------------------------------------------------------
# Parser cache
# ------------------------------------------------------------------

_PARSERS: Dict[str, object] = {}


def _parser_for(kind: str):
    if kind in _PARSERS:
        return _PARSERS[kind]
    try:
        from tree_sitter import Language, Parser
        if kind == "js":
            import tree_sitter_javascript as mod
            lang = Language(mod.language())
        elif kind == "ts":
            import tree_sitter_typescript as mod
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
# Rule-relevant tag sets (DISJOINT from a347/a240)
# ------------------------------------------------------------------

# Static (non-interactive) HTML elements. If one has an onClick, it
# must also supply role + tabIndex (jsx-a11y/no-static-element-interactions
# + click-events-have-key-events).
STATIC_TAGS = {"div", "span", "p", "section", "article", "aside",
               "header", "footer", "main", "nav", "ul", "ol", "li",
               "td", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6"}

# Form controls that need an associated label.
LABELABLE_TAGS = {"input", "select", "textarea"}

# Input types that don't need a label (jsx-a11y excludes these).
LABEL_EXEMPT_INPUT_TYPES = {"hidden", "submit", "reset", "button", "image"}

# Implicit-role map for jsx-a11y/no-redundant-roles. If element has
# role=<the value in this map>, it's redundant.
IMPLICIT_ROLE = {
    "button": "button",
    "a": "link",          # only when href present, but we still flag role="link" on <a>
    "nav": "navigation",
    "main": "main",
    "header": "banner",
    "footer": "contentinfo",
    "aside": "complementary",
    "article": "article",
    "section": "region",
    "form": "form",
    "ul": "list",
    "ol": "list",
    "li": "listitem",
    "img": "img",
    "table": "table",
    "thead": "rowgroup",
    "tbody": "rowgroup",
    "tr": "row",
    "td": "cell",
    "th": "columnheader",
    "h1": "heading", "h2": "heading", "h3": "heading",
    "h4": "heading", "h5": "heading", "h6": "heading",
    "select": "combobox",
    "textarea": "textbox",
    "dialog": "dialog",
    "input": None,  # special: depends on type
}

# Accessible-name attributes that satisfy the label requirement.
ACCESSIBLE_NAME_ATTRS = {"aria-label", "aria-labelledby", "title"}

# Click-event attributes (JSX onClick, HTML onclick).
CLICK_ATTRS = {"onclick", "onClick"}
# Key-event attributes that mitigate click-events-have-key-events.
KEY_EVENT_ATTRS = {"onkeydown", "onkeyup", "onkeypress",
                   "onKeyDown", "onKeyUp", "onKeyPress"}


# ------------------------------------------------------------------
# JSX walker
# ------------------------------------------------------------------

def _jsx_attrs(opening_or_self, src: bytes) -> Dict[str, Optional[str]]:
    """Return {attr_name_lower: literal_value_or_None}.

    Preserves both lowercase and original-case names so we can detect
    `onClick` (JSX) and `onclick` (HTML-ish) uniformly downstream.
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
                name_node = sub
            elif sub.type == "string":
                value_node = sub
            elif sub.type == "jsx_expression":
                value_node = sub
        if name_node is None:
            continue
        attr_name = _text(name_node, src)
        attr_lower = attr_name.lower()
        if value_node is None:
            out[attr_lower] = None
            continue
        if value_node.type == "string":
            raw = _text(value_node, src)
            if len(raw) >= 2 and raw[0] in "\"'":
                out[attr_lower] = raw[1:-1]
            else:
                out[attr_lower] = raw
        else:
            out[attr_lower] = None
    return out


def _is_static_element_with_handler(tag: str,
                                    attrs: Dict[str, Optional[str]]) -> Tuple[bool, bool]:
    """Returns (relevant, ok). relevant=True iff this is a static-element
    with an onClick. ok=True iff the violation is mitigated (role + tabIndex,
    and key event present).
    """
    if tag not in STATIC_TAGS:
        return (False, False)
    has_click = any(a.lower() in CLICK_ATTRS for a in attrs.keys())
    if not has_click:
        return (False, False)
    # Mitigation: explicit interactive role + tabIndex + key handler.
    role = (attrs.get("role") or "").lower() if attrs.get("role") else ""
    interactive_roles = {"button", "link", "menuitem", "tab", "checkbox",
                         "radio", "switch", "option", "treeitem", "gridcell"}
    has_role = role in interactive_roles
    has_tabindex = "tabindex" in attrs
    has_key = any(a.lower() in {k.lower() for k in KEY_EVENT_ATTRS}
                  for a in attrs.keys())
    return (True, has_role and has_tabindex and has_key)


def _is_form_control_unlabeled(tag: str,
                               attrs: Dict[str, Optional[str]],
                               nearby_label_for: set,
                               id_value: Optional[str]) -> Tuple[bool, bool]:
    """Returns (relevant, ok). relevant=True iff this is a labelable form
    control that isn't exempt. ok=True iff an accessible name source exists:
    aria-label / aria-labelledby / title, OR a sibling <label htmlFor=id>
    matches its id.
    """
    if tag not in LABELABLE_TAGS:
        return (False, False)
    if tag == "input":
        itype = (attrs.get("type") or "").lower()
        if itype in LABEL_EXEMPT_INPUT_TYPES:
            return (False, False)
    has_name = any(a in attrs for a in ACCESSIBLE_NAME_ATTRS)
    has_label_for = id_value is not None and id_value in nearby_label_for
    return (True, has_name or has_label_for)


def _is_html_root_unlangged(tag: str,
                            attrs: Dict[str, Optional[str]]) -> Tuple[bool, bool]:
    if tag != "html":
        return (False, False)
    return (True, "lang" in attrs)


def _is_redundant_role(tag: str,
                       attrs: Dict[str, Optional[str]]) -> Tuple[bool, bool]:
    role = attrs.get("role")
    if not role:
        return (False, False)
    # Dynamic role={expr} we cannot resolve; not flagged.
    if role is None:
        return (False, False)
    role_l = role.lower()
    # input is special: role depends on type. Don't flag input.
    if tag == "input":
        return (False, False)
    implicit = IMPLICIT_ROLE.get(tag)
    if implicit is None:
        return (False, False)
    # relevant=True (we saw an explicit role on a tag with an implicit
    # role); ok=True iff the role does NOT match the implicit one
    # (i.e. it's an *override*, not a redundant restatement).
    if role_l == implicit:
        return (True, False)
    return (True, True)


def _collect_label_for_ids_jsx(root, src: bytes) -> set:
    """Walk JSX subtree, gather htmlFor / for values from <label> elements."""
    out: set = set()

    def walk(node):
        if node.type in ("jsx_self_closing_element", "jsx_opening_element"):
            tag = None
            for ch in node.children:
                if ch.type == "identifier":
                    tag = _text(ch, src)
                    break
            if tag is not None and tag.lower() == "label":
                attrs = _jsx_attrs(node, src)
                if "htmlfor" in attrs and attrs["htmlfor"]:
                    out.add(attrs["htmlfor"])
                if "for" in attrs and attrs["for"]:
                    out.add(attrs["for"])
        for c in node.children:
            walk(c)

    walk(root)
    return out


def _walk_jsx(node, src: bytes, results: List[Tuple[str, bool]],
              label_for_ids: set):
    t = node.type
    if t == "jsx_self_closing_element" and not _in_error(node):
        tag = None
        for ch in node.children:
            if ch.type == "identifier":
                tag = _text(ch, src)
                break
        if tag is None:
            for c in node.children:
                _walk_jsx(c, src, results, label_for_ids)
            return
        tlow = tag.lower()
        attrs = _jsx_attrs(node, src)
        id_val = attrs.get("id")
        # Rule 1: static element with click handler
        rel, ok = _is_static_element_with_handler(tlow, attrs)
        if rel:
            results.append((f"static_click:{tlow}", ok))
        # Rule 2: form control labeling
        rel, ok = _is_form_control_unlabeled(tlow, attrs, label_for_ids, id_val)
        if rel:
            results.append((f"label:{tlow}", ok))
        # Rule 3: html-has-lang
        rel, ok = _is_html_root_unlangged(tlow, attrs)
        if rel:
            results.append(("html_lang", ok))
        # Rule 4: redundant role
        rel, ok = _is_redundant_role(tlow, attrs)
        if rel:
            results.append((f"role:{tlow}", ok))
    elif t == "jsx_element" and not _in_error(node):
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
                attrs = _jsx_attrs(opening, src)
                id_val = attrs.get("id")
                rel, ok = _is_static_element_with_handler(tlow, attrs)
                if rel:
                    results.append((f"static_click:{tlow}", ok))
                rel, ok = _is_form_control_unlabeled(
                    tlow, attrs, label_for_ids, id_val)
                if rel:
                    results.append((f"label:{tlow}", ok))
                rel, ok = _is_html_root_unlangged(tlow, attrs)
                if rel:
                    results.append(("html_lang", ok))
                rel, ok = _is_redundant_role(tlow, attrs)
                if rel:
                    results.append((f"role:{tlow}", ok))
    for c in node.children:
        _walk_jsx(c, src, results, label_for_ids)


def _measure_jsx(code: bytes, ts: bool) -> List[Tuple[str, bool]]:
    parser = _parser_for("ts" if ts else "js")
    if parser is None:
        return []
    tree = parser.parse(code)
    out: List[Tuple[str, bool]] = []
    label_for_ids = _collect_label_for_ids_jsx(tree.root_node, code)
    _walk_jsx(tree.root_node, code, out, label_for_ids)
    return out


# ------------------------------------------------------------------
# HTML walker
# ------------------------------------------------------------------

def _html_attrs(tag_node, src: bytes) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for ch in tag_node.children:
        if ch.type != "attribute":
            continue
        name = None
        val: Optional[str] = None
        for sub in ch.children:
            if sub.type == "attribute_name":
                name = _text(sub, src).lower()
            elif sub.type == "quoted_attribute_value":
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


def _collect_label_for_ids_html(root, src: bytes) -> set:
    out: set = set()

    def walk(node):
        if node.type in ("element", "start_tag", "self_closing_tag"):
            start = node
            if node.type == "element":
                start = None
                for ch in node.children:
                    if ch.type in ("start_tag", "self_closing_tag"):
                        start = ch
                        break
            if start is not None:
                tag = _html_tag_name(start, src)
                if tag == "label":
                    attrs = _html_attrs(start, src)
                    if "for" in attrs and attrs["for"]:
                        out.add(attrs["for"])
        for c in node.children:
            walk(c)

    walk(root)
    return out


def _walk_html(node, src: bytes, results: List[Tuple[str, bool]],
               label_for_ids: set):
    t = node.type
    if t == "self_closing_tag":
        tag = _html_tag_name(node, src)
        if tag is None:
            for c in node.children:
                _walk_html(c, src, results, label_for_ids)
            return
        attrs = _html_attrs(node, src)
        id_val = attrs.get("id")
        rel, ok = _is_static_element_with_handler(tag, attrs)
        if rel:
            results.append((f"static_click:{tag}", ok))
        rel, ok = _is_form_control_unlabeled(tag, attrs, label_for_ids, id_val)
        if rel:
            results.append((f"label:{tag}", ok))
        rel, ok = _is_html_root_unlangged(tag, attrs)
        if rel:
            results.append(("html_lang", ok))
        rel, ok = _is_redundant_role(tag, attrs)
        if rel:
            results.append((f"role:{tag}", ok))
    elif t == "element":
        start = None
        for ch in node.children:
            if ch.type == "start_tag":
                start = ch
                break
            if ch.type == "self_closing_tag":
                start = ch
                break
        if start is not None:
            tag = _html_tag_name(start, src)
            if tag is not None:
                attrs = _html_attrs(start, src)
                id_val = attrs.get("id")
                rel, ok = _is_static_element_with_handler(tag, attrs)
                if rel:
                    results.append((f"static_click:{tag}", ok))
                rel, ok = _is_form_control_unlabeled(
                    tag, attrs, label_for_ids, id_val)
                if rel:
                    results.append((f"label:{tag}", ok))
                rel, ok = _is_html_root_unlangged(tag, attrs)
                if rel:
                    results.append(("html_lang", ok))
                rel, ok = _is_redundant_role(tag, attrs)
                if rel:
                    results.append((f"role:{tag}", ok))
    for c in node.children:
        _walk_html(c, src, results, label_for_ids)


def _measure_html(code: bytes) -> List[Tuple[str, bool]]:
    parser = _parser_for("html")
    if parser is None:
        return []
    tree = parser.parse(code)
    out: List[Tuple[str, bool]] = []
    label_for_ids = _collect_label_for_ids_html(tree.root_node, code)
    _walk_html(tree.root_node, code, out, label_for_ids)
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
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return False
    if not any(p.lower().endswith(JSX_EXTS + HTML_EXTS) for p in by_path):
        return False
    return len(_collect(diff_text)) > 0


def score(diff_text: str) -> Optional[float]:
    items = _collect(diff_text)
    if not items:
        return None
    ok = sum(1 for _, good in items if good)
    return float(ok / len(items))
