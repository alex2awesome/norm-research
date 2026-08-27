"""a407: Name expressiveness — beyond shape and beyond intention-blacklist.

This metric SHARES SCOPE with a164 (casing) and a70 (intention-revealing
blacklist). We do NOT duplicate either; we add the third axis:

  axis (a164):  shape — does the name follow snake_case / PascalCase?
  axis (a70):   semantics by blacklist — is the name in {data, tmp, util}?
  axis (a407): EXPRESSIVENESS — does the name carry phrase-level information?

Specifically a407 measures:
  (1) Identifier-character entropy. Information-theoretic measure of
      whether the name is just letters jammed together (`xyz`) vs has
      structure (`getUserById`). We compute Shannon entropy of the
      character distribution; longer names with vocabulary diversity score
      higher.
  (2) Token count after splitting snake_case / camelCase: a name that
      decomposes into >= 2 tokens (e.g. `parse_diff`, `getMaxValue`)
      scores higher than a single opaque token (`parsediff`, `getmax`).

We DO NOT blacklist generic words (a70 already does that). We DO NOT check
casing (a164 already does that).

Aggregation: per declared identifier we compute
    expressive = 0.5 * entropy_norm + 0.5 * (1 if tokens>=2 else 0.0)
where entropy_norm = min(H / 3.5, 1.0).
File score = mean(expressive). Diff score = mean across files.

Examples:
  + def f(x): ...                -> very short, low entropy           ~0.1
  + def parse_diff(text): ...    -> 2 tokens, good entropy            ~0.95
  + def XYZ(): ...               -> 1 opaque token, low entropy       ~0.3
  + def getUserByIdSafe(): ...   -> 4 tokens, good entropy            ~1.0

CLASSIFICATION: PARTIALLY_THIN — entropy is a surface proxy for "the name
carries information"; it cannot tell whether `getUserById` actually
describes the function's behavior. We accept this and document it.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a407"
ASPECT_NAME = "Name expressiveness (entropy + token-count)"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}
DECL_NODES = {
    "py": {"function_definition", "class_definition"},
    "js": {"function_declaration", "class_declaration",
           "method_definition", "variable_declarator"},
    "ts": {"function_declaration", "class_declaration",
           "method_definition", "variable_declarator",
           "interface_declaration", "type_alias_declaration"},
    "java": {"class_declaration", "interface_declaration",
             "method_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration", "type_spec"},
}
EXEMPT_NAMES = {"i", "j", "k", "n", "x", "y", "z", "_", "self", "cls",
                "T", "K", "V"}

_PARSERS: Dict[str, object] = {}


def _get_parser(lang: str):
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "py":
            import tree_sitter_python as m; L = m.language()
        elif lang == "js":
            import tree_sitter_javascript as m; L = m.language()
        elif lang == "ts":
            import tree_sitter_typescript as m; L = m.language_typescript()
        elif lang == "java":
            import tree_sitter_java as m; L = m.language()
        elif lang == "go":
            import tree_sitter_go as m; L = m.language()
        else:
            return None
        _PARSERS[lang] = Parser(Language(L))
        return _PARSERS[lang]
    except ImportError:
        return None


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


def _split_tokens(name: str) -> List[str]:
    """Split snake_case AND camelCase: 'getUserById' -> ['get','User','By','Id'];
    'parse_diff' -> ['parse','diff']."""
    # First split on underscores
    parts: List[str] = []
    for chunk in name.split("_"):
        if not chunk:
            continue
        # Split chunk on camel boundaries (lower->upper)
        cur = chunk[0]
        for ch in chunk[1:]:
            if ch.isupper() and cur and cur[-1].islower():
                parts.append(cur)
                cur = ch
            else:
                cur += ch
        if cur:
            parts.append(cur)
    return [p for p in parts if p]


def _entropy(name: str) -> float:
    if not name:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in name:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(name)
    h = 0.0
    for v in counts.values():
        p = v / n
        h -= p * math.log2(p)
    return h


def _expressiveness(name: str) -> Optional[float]:
    if name in EXEMPT_NAMES:
        return None
    if len(name) <= 1:
        return 0.05
    H = _entropy(name)
    H_norm = min(H / 3.5, 1.0)
    tokens = _split_tokens(name)
    token_bonus = 1.0 if len(tokens) >= 2 else 0.0
    return 0.5 * H_norm + 0.5 * token_bonus


def _collect_names(code: bytes, lang: str) -> List[str]:
    parser = _get_parser(lang)
    if parser is None:
        return []
    try:
        tree = parser.parse(code)
    except Exception:
        return []
    targets = DECL_NODES.get(lang, set())
    out: List[str] = []

    def walk(n):
        if n.type in targets:
            # find the first identifier-like child
            for c in n.children:
                if c.type in ("identifier", "type_identifier",
                              "property_identifier", "field_identifier"):
                    out.append(_text(c, code))
                    break
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return out


def _path_lang(path: str) -> Optional[str]:
    p = path.lower()
    for ext, lang in EXT_TO_LANG.items():
        if p.endswith(ext):
            return lang
    return None


def applies(diff_text: str) -> bool:
    by_path = parse_diff_added_by_file(diff_text)
    return any(_path_lang(p) is not None for p in by_path)


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    file_scores: List[float] = []
    for path, content in by_path.items():
        lang = _path_lang(path)
        if lang is None:
            continue
        names = _collect_names(content.encode("utf8", errors="replace"), lang)
        scs = [s for s in (_expressiveness(n) for n in names)
               if s is not None]
        if scs:
            file_scores.append(sum(scs) / len(scs))
    if not file_scores:
        return None
    return float(sum(file_scores) / len(file_scores))
