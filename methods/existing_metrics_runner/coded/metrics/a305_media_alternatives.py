"""a305: Non-text content alternatives — media (audio/video) angle.

The aspect description is broad ("appropriate alternatives for non-text
content: high-quality alt text, captions/transcripts for media, and place
meaningful images in markup not CSS"). The *image alt-text* portion of
this norm is already measured by a347 (WCAG 1.1.1 alt-text on
<img>/<area>/<input type=image> plus accessible-name on <a>/<button>).

To stay honest and distinct from a347 we restrict a305 to the
**media-alternative** sub-norm: captions/transcripts/text tracks for
<audio> and <video> elements. This is WCAG 1.2.x territory
(SC 1.2.1/1.2.2/1.2.3), distinct from 1.1.1 which a347 covers.

What we check, per <video> or <audio> element added by the diff:

  <video> is "ok" iff it has at least one child
          <track kind="captions"> OR <track kind="subtitles">
          OR has an aria-describedby / aria-label / aria-labelledby
          attribute pointing to an external transcript-like description.

  <audio> is "ok" iff it has at least one child <track> (kind captions/
          subtitles/descriptions) OR has aria-describedby/aria-label/
          aria-labelledby (typical transcript-link pattern) OR an inner
          <a href> child (the "Transcript: ..." link pattern that the
          W3C tutorial recommends).

Library-first rationale:
  - The canonical lint is eslint-plugin-jsx-a11y/media-has-caption,
    which checks exactly this. The shared sandbox does not have a
    Node project rooted in the diff, so we replicate the rule using
    tree-sitter-javascript / tree-sitter-typescript (for JSX/TSX) and
    tree-sitter-html (for HTML/Vue/Svelte/Handlebars/ERB) — the same
    library approach a347 uses. No regex on code.

Score:
  Numerator   = media elements that satisfy the caption/transcript rule
  Denominator = media elements observed
  None        = applies()=False, i.e. no <audio>/<video> in the diff, OR
                parser unavailable on every file (tool failure).

Applies():
  True iff the diff adds at least one <video> or <audio> element in
  a JSX/HTML-family file. Most PRs have no media tag at all; this norm
  is genuinely narrow and we abstain liberally.

Classification:
  THIN. The presence-vs-absence of a <track> child or transcript-link
  attribute is a structural property the tree-sitter AST exposes
  directly. The judgement of whether captions are *high-quality* is
  not measured here — that part is thick, but the "any captions at
  all?" baseline is the deterministic core that lints actually check.

Reference: a347 (image-alt-text sibling — disjoint element set), a240
(narrow CSS pattern), a16 (THICK template).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a305"
ASPECT_NAME = "Non-text content alternatives (media captions/transcripts)"
TIER = 3
TOOLS = ["tree-sitter-html", "tree-sitter-javascript",
         "tree-sitter-typescript"]
APPLIES_TO_LANGS = ["JavaScript", "TypeScript", "HTML", "Vue", "Svelte"]
CLASSIFICATION = "THIN"

# Restrict to JSX/HTML-family extensions. Same set as a347 — but the
# element targets are disjoint (<audio>/<video> here vs <img>/<a>/<button>
# in a347), so the two metrics are independent measurements.
JSX_EXTS = (".jsx", ".tsx", ".js", ".ts", ".mjs", ".cjs")
TS_EXTS = (".ts", ".tsx")
HTML_EXTS = (".html", ".htm", ".vue", ".svelte", ".hbs", ".handlebars",
             ".ejs", ".erb", ".njk")

MEDIA_TAGS = {"video", "audio"}
# Attributes that count as "transcript / description supplied externally".
TRANSCRIPT_ATTRS = {"aria-describedby", "aria-label", "aria-labelledby"}
# Caption/subtitle/description track kinds (per HTML <track kind> values).
CAPTION_KINDS = {"captions", "subtitles", "descriptions"}


# ------------------------------------------------------------------
# Parser cache (mirrors a347's pattern)
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
# JSX walker
# ------------------------------------------------------------------

def _jsx_attrs(opening_or_self, src: bytes) -> Dict[str, Optional[str]]:
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
        attr_name = _text(name_node, src).lower()
        if value_node is None:
            out[attr_name] = None
            continue
        if value_node.type == "string":
            raw = _text(value_node, src)
            if len(raw) >= 2 and raw[0] in "\"'":
                out[attr_name] = raw[1:-1]
            else:
                out[attr_name] = raw
        else:
            out[attr_name] = None
    return out


def _jsx_opening(jsx_element_node):
    for ch in jsx_element_node.children:
        if ch.type == "jsx_opening_element":
            return ch
    return None


def _jsx_tag_name(opening_or_self, src: bytes) -> Optional[str]:
    for ch in opening_or_self.children:
        if ch.type == "identifier":
            return _text(ch, src).lower()
    return None


def _jsx_has_caption_child(jsx_element_node, src: bytes) -> bool:
    """True iff any descendant <track kind="captions|subtitles|descriptions">
    sits inside this media element. Bare <track> with NO kind defaults to
    subtitles per HTML spec, so we accept that too.
    """
    stack = list(jsx_element_node.children)
    while stack:
        node = stack.pop()
        tag = None
        attrs: Dict[str, Optional[str]] = {}
        if node.type == "jsx_self_closing_element":
            tag = _jsx_tag_name(node, src)
            if tag == "track":
                attrs = _jsx_attrs(node, src)
        elif node.type == "jsx_element":
            opening = _jsx_opening(node)
            if opening is not None:
                tag = _jsx_tag_name(opening, src)
                if tag == "track":
                    attrs = _jsx_attrs(opening, src)
        if tag == "track":
            kind = (attrs.get("kind") or "subtitles") or "subtitles"
            kind_l = kind.lower() if isinstance(kind, str) else "subtitles"
            if kind_l in CAPTION_KINDS:
                return True
        # Recurse
        for ch in node.children:
            stack.append(ch)
    return False


def _jsx_has_transcript_link(jsx_element_node, src: bytes) -> bool:
    """True iff this media element contains an <a href> descendant — the
    canonical 'Transcript: ...' link pattern.
    """
    stack = list(jsx_element_node.children)
    while stack:
        node = stack.pop()
        opening = None
        tag = None
        if node.type == "jsx_self_closing_element":
            opening = node
            tag = _jsx_tag_name(node, src)
        elif node.type == "jsx_element":
            opening = _jsx_opening(node)
            if opening is not None:
                tag = _jsx_tag_name(opening, src)
        if tag == "a" and opening is not None:
            attrs = _jsx_attrs(opening, src)
            if "href" in attrs:
                return True
        for ch in node.children:
            stack.append(ch)
    return False


def _walk_jsx(node, src: bytes, results: List[Tuple[str, bool]]):
    t = node.type
    if t == "jsx_self_closing_element" and not _in_error(node):
        tag = _jsx_tag_name(node, src)
        if tag in MEDIA_TAGS:
            # Self-closing <video />: no children, so must rely on
            # aria-describedby/aria-label/aria-labelledby.
            attrs = _jsx_attrs(node, src)
            ok = any(a in attrs for a in TRANSCRIPT_ATTRS)
            results.append((tag, ok))
    elif t == "jsx_element" and not _in_error(node):
        opening = _jsx_opening(node)
        if opening is not None:
            tag = _jsx_tag_name(opening, src)
            if tag in MEDIA_TAGS:
                attrs = _jsx_attrs(opening, src)
                has_aria = any(a in attrs for a in TRANSCRIPT_ATTRS)
                has_track = _jsx_has_caption_child(node, src)
                has_link = (tag == "audio" and
                            _jsx_has_transcript_link(node, src))
                results.append((tag, has_track or has_aria or has_link))
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


def _html_start_of(element_node):
    for ch in element_node.children:
        if ch.type == "start_tag":
            return ch
        if ch.type == "self_closing_tag":
            return ch
    return None


def _html_has_caption_child(element_node, src: bytes) -> bool:
    """Walk descendants looking for a <track> with caption-like kind."""
    stack = list(element_node.children)
    while stack:
        node = stack.pop()
        if node.type == "self_closing_tag":
            tag = _html_tag_name(node, src)
            if tag == "track":
                attrs = _html_attrs(node, src)
                kind = (attrs.get("kind") or "subtitles") or "subtitles"
                kind_l = kind.lower() if isinstance(kind, str) else "subtitles"
                if kind_l in CAPTION_KINDS:
                    return True
        elif node.type == "element":
            start = _html_start_of(node)
            if start is not None:
                tag = _html_tag_name(start, src)
                if tag == "track":
                    attrs = _html_attrs(start, src)
                    kind = (attrs.get("kind") or "subtitles") or "subtitles"
                    kind_l = (kind.lower() if isinstance(kind, str)
                              else "subtitles")
                    if kind_l in CAPTION_KINDS:
                        return True
        for ch in node.children:
            stack.append(ch)
    return False


def _html_has_transcript_link(element_node, src: bytes) -> bool:
    stack = list(element_node.children)
    while stack:
        node = stack.pop()
        if node.type == "element":
            start = _html_start_of(node)
            if start is not None and _html_tag_name(start, src) == "a":
                attrs = _html_attrs(start, src)
                if "href" in attrs:
                    return True
        for ch in node.children:
            stack.append(ch)
    return False


def _walk_html(node, src: bytes, results: List[Tuple[str, bool]]):
    t = node.type
    if t == "self_closing_tag":
        tag = _html_tag_name(node, src)
        if tag in MEDIA_TAGS:
            attrs = _html_attrs(node, src)
            ok = any(a in attrs for a in TRANSCRIPT_ATTRS)
            results.append((tag, ok))
    elif t == "element":
        start = _html_start_of(node)
        if start is not None:
            tag = _html_tag_name(start, src)
            if tag in MEDIA_TAGS:
                attrs = _html_attrs(start, src)
                has_aria = any(a in attrs for a in TRANSCRIPT_ATTRS)
                has_track = _html_has_caption_child(node, src)
                has_link = (tag == "audio" and
                            _html_has_transcript_link(node, src))
                results.append((tag, has_track or has_aria or has_link))
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
