"""a104: Automated tests — presence and design quality.

Detects test FILES by path convention (a path is a path; regex on paths is
the standard idiom). Detects test FUNCTIONS by parsing source with tree-sitter
per language (Python, JavaScript, TypeScript, Java, Go) — handles partial
diff snippets via error-recovery.

Norm conformance: high if the diff adds tests proportional to source changes.

Caveat: we can't judge test DESIGN quality (assertion strength, mock fidelity)
without running the tests — so this is PARTIALLY_THIN.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Optional

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a104"
ASPECT_NAME = "Automated tests: presence and design quality"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"

# REGEX_OK: file path classification — paths aren't a language to parse,
# and the test-path conventions (tests/, *_test.py, __tests__/) are
# byte-level patterns that regex handles directly.
TEST_PATH_RE = re.compile(
    r"(^|/)(test_|tests?/|spec/|specs?/|__tests__/)|"
    r"(_test|\.test|_spec|\.spec)\.[^/]+$",
    re.IGNORECASE,
)
NON_CODE_EXTS = (".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml",
                 ".lock", ".gitignore", ".cfg", ".ini")
EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}

_PARSERS: Dict[str, object] = {}


def _get_parser(lang_short: str):
    if lang_short in _PARSERS:
        return _PARSERS[lang_short]
    try:
        from tree_sitter import Language, Parser
        if lang_short == "py":
            import tree_sitter_python as m; lang = m.language()
        elif lang_short == "js":
            import tree_sitter_javascript as m; lang = m.language()
        elif lang_short == "ts":
            import tree_sitter_typescript as m
            lang = m.language_typescript()
        elif lang_short == "java":
            import tree_sitter_java as m; lang = m.language()
        elif lang_short == "go":
            import tree_sitter_go as m; lang = m.language()
        else:
            return None
        _PARSERS[lang_short] = Parser(Language(lang))
        return _PARSERS[lang_short]
    except ImportError:
        return None


def _count_test_functions(code: bytes, lang_short: str) -> int:
    """Walk AST, count function/method declarations matching test conventions."""
    parser = _get_parser(lang_short)
    if parser is None:
        return 0
    tree = parser.parse(code)
    count = 0

    def text(n) -> str:
        return code[n.start_byte:n.end_byte].decode("utf8", errors="replace")

    def py_walk(node):
        nonlocal count
        if node.type == "function_definition":
            for c in node.children:
                if c.type == "identifier":
                    if text(c).startswith("test_"):
                        count += 1
                    break
        for c in node.children:
            py_walk(c)

    def js_walk(node):
        nonlocal count
        # function declarations: function testFoo() {}
        if node.type == "function_declaration":
            for c in node.children:
                if c.type == "identifier" and text(c).startswith("test"):
                    count += 1
                    break
        # Jest/Mocha calls: it(...), test(...), describe(...)
        if node.type == "call_expression":
            if node.children and node.children[0].type == "identifier":
                fn_name = text(node.children[0])
                if fn_name in ("it", "test", "describe"):
                    count += 1
        for c in node.children:
            js_walk(c)

    def java_walk(node):
        nonlocal count
        if node.type == "method_declaration":
            has_test_annotation = False
            name = None
            for c in node.children:
                if c.type == "modifiers":
                    for sub in c.children:
                        if sub.type == "marker_annotation":
                            for s2 in sub.children:
                                if s2.type == "identifier" and text(s2) == "Test":
                                    has_test_annotation = True
                elif c.type == "identifier":
                    name = text(c)
            if has_test_annotation or (name and name.startswith("test")):
                count += 1
        for c in node.children:
            java_walk(c)

    def go_walk(node):
        nonlocal count
        if node.type == "function_declaration":
            for c in node.children:
                if c.type == "identifier" and text(c).startswith("Test"):
                    count += 1
                    break
        for c in node.children:
            go_walk(c)

    if lang_short == "py":
        py_walk(tree.root_node)
    elif lang_short in ("js", "ts"):
        js_walk(tree.root_node)
    elif lang_short == "java":
        java_walk(tree.root_node)
    elif lang_short == "go":
        go_walk(tree.root_node)
    return count


def _is_code(path: str) -> bool:
    return not path.lower().endswith(NON_CODE_EXTS)


def applies(diff_text: str) -> bool:
    by_path = parse_diff_added_by_file(diff_text)
    return any(_is_code(p) for p in by_path)


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    test_lines = src_lines = 0
    n_test_fns = 0
    any_code = False
    for path, code in by_path.items():
        if not _is_code(path):
            continue
        any_code = True
        n_lines = 1 + code.count("\n")
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        lang = EXT_TO_LANG.get(ext)
        if TEST_PATH_RE.search(path):
            test_lines += n_lines
            if lang:
                n_test_fns += _count_test_functions(
                    code.encode("utf8", errors="replace"), lang)
        else:
            src_lines += n_lines
            if lang:
                n_test_fns += _count_test_functions(
                    code.encode("utf8", errors="replace"), lang)
    if not any_code:
        return None
    if src_lines == 0:
        return 1.0
    ratio = test_lines / max(src_lines, 1)
    s_ratio = math.tanh(ratio * 2.0)
    s_count = math.tanh(n_test_fns / 5.0)
    return float(0.6 * s_ratio + 0.4 * s_count)
