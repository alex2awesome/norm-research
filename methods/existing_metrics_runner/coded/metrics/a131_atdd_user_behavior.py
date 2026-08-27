"""a131: Acceptance tests specify and verify user-visible behavior (ATDD).

ATDD asks whether tests EXPRESS USER INTENT at the user-observable boundary
rather than asserting on internal implementation details. The "true" version
of this norm requires three pieces of information:

  (1) the PR's intended user-visible behavior change (intent),
  (2) which test assertions correspond to that intent, and
  (3) whether those assertions probe the user/system boundary or internal API.

Pieces (1) and (2) are not deterministically observable from a diff alone —
they require reasoning about correspondence between English intent and test
code, which is exactly the kind of semantic alignment the article-ability
boundary names. THICK in the strict sense.

HOWEVER, there is a defensible *partial* signal: certain test idioms are
strongly associated with ATDD/BDD style, and others with implementation-detail
unit testing. We can count these structurally via tree-sitter:

  PRO-ATDD signals:
    - .feature files (Gherkin / Cucumber)
    - BDD-style `describe`/`it` calls with descriptive *string* literals (JS/TS)
    - imports of user-boundary drivers:
        Python: selenium, playwright, requests, httpx, fastapi.testclient,
                django.test.Client, rest_framework.test.APIClient
        JS/TS:  supertest, @playwright/test, cypress, puppeteer
        Java:   selenium, rest-assured, mockmvc, cucumber.api
        Go:     net/http/httptest

  ANTI-ATDD signals (white-box / implementation-detail):
    - heavy mocking imports:
        Python: unittest.mock, mock, pytest_mock
        JS/TS:  jest.mock(), sinon, vi.mock
        Java:   org.mockito

We ratio (pro signals) / (pro + anti + baseline). It is a *style* proxy, not
a measurement of intent-correspondence. It will misclassify:
  - Selenium tests that drive UI but assert on internal CSS classes (looks
    user-facing, isn't).
  - Pure unit tests that incidentally happen to be at a public boundary.
  - Snapshot tests (no obvious driver import).

That is why we mark PARTIALLY_THIN and not THIN: the structural proxy
correlates with ATDD style but does not measure the norm itself.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a131"
ASPECT_NAME = "ATDD: tests verify user-visible behavior"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go", "Gherkin"]
CLASSIFICATION = "PARTIALLY_THIN"

# REGEX_OK: file_path — test file conventions are path patterns, not code.
TEST_PATH_RE = re.compile(
    r"(^|/)(test_|tests?/|spec/|specs?/|__tests__/|features?/|e2e/|"
    r"integration/|acceptance/)|"
    r"(_test|\.test|_spec|\.spec)\.[^/]+$",
    re.IGNORECASE,
)

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}

# User-boundary driver/client imports — finding any of these in a test file
# is a strong signal the test interacts at a user-observable seam.
PRO_IMPORTS = {
    "py": {
        "selenium", "playwright", "requests", "httpx",
        "fastapi.testclient", "starlette.testclient",
        "django.test", "rest_framework.test",
        "webtest", "splinter", "behave", "pytest_bdd",
    },
    "js": {
        "supertest", "@playwright/test", "playwright", "cypress",
        "puppeteer", "@cucumber/cucumber", "cucumber",
        "@testing-library/react", "@testing-library/dom",
    },
    "ts": {
        "supertest", "@playwright/test", "playwright", "cypress",
        "puppeteer", "@cucumber/cucumber", "cucumber",
        "@testing-library/react", "@testing-library/dom",
    },
    "java": {
        "io.cucumber", "cucumber.api",
        "org.openqa.selenium", "io.restassured",
        "org.springframework.test.web.servlet",
    },
    "go": {
        "net/http/httptest", "github.com/cucumber/godog",
    },
}

# Heavy-mocking imports — anti-signal for ATDD (white-box testing).
ANTI_IMPORTS = {
    "py": {"unittest.mock", "mock", "pytest_mock"},
    "js": {"sinon", "ts-mockito", "jest-mock"},
    "ts": {"sinon", "ts-mockito", "jest-mock"},
    "java": {"org.mockito", "mockito"},
    "go": {"github.com/stretchr/testify/mock", "github.com/golang/mock/gomock"},
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


def _collect_python(code: bytes) -> Tuple[set, int, int]:
    """Return (import_modules, bdd_call_count, mock_call_count)."""
    parser = _get_parser("py")
    if parser is None:
        return set(), 0, 0
    tree = parser.parse(code)
    mods: set = set()
    bdd_calls = 0
    mock_calls = 0

    def text(n) -> str:
        return code[n.start_byte:n.end_byte].decode("utf8", errors="replace")

    def walk(node):
        nonlocal bdd_calls, mock_calls
        if node.type == "import_statement":
            for c in node.children:
                if c.type in ("dotted_name", "aliased_import"):
                    mods.add(text(c).split(" as ")[0].strip())
        elif node.type == "import_from_statement":
            # children: 'from' <dotted_name|relative_import> 'import' ...
            for c in node.children:
                if c.type == "dotted_name":
                    mods.add(text(c))
                    break
                if c.type == "relative_import":
                    # relative import — skip
                    break
        elif node.type == "call":
            # detect mock.patch / Mock() / MagicMock()
            if node.children:
                fn_node = node.children[0]
                fn_text = text(fn_node)
                if fn_text in ("patch", "Mock", "MagicMock") or \
                   fn_text.endswith(".patch") or fn_text.endswith(".Mock") or \
                   fn_text.endswith(".MagicMock"):
                    mock_calls += 1
                # BDD: given/when/then decorators or pytest_bdd scenario
                if fn_text in ("given", "when", "then", "scenario", "scenarios"):
                    bdd_calls += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return mods, bdd_calls, mock_calls


def _collect_js_ts(code: bytes, lang: str) -> Tuple[set, int, int]:
    parser = _get_parser(lang)
    if parser is None:
        return set(), 0, 0
    tree = parser.parse(code)
    mods: set = set()
    bdd_calls = 0
    mock_calls = 0

    def text(n) -> str:
        return code[n.start_byte:n.end_byte].decode("utf8", errors="replace")

    def walk(node):
        nonlocal bdd_calls, mock_calls
        if node.type == "import_statement":
            # find string source
            for c in node.children:
                if c.type == "string":
                    s = text(c).strip("'\"`")
                    mods.add(s)
        elif node.type == "call_expression":
            if node.children:
                fn_node = node.children[0]
                fn_text = text(fn_node)
                # require/import('mod')
                if fn_text == "require" and len(node.children) >= 2:
                    args = node.children[1]
                    for sc in args.children:
                        if sc.type == "string":
                            mods.add(text(sc).strip("'\"`"))
                # describe/it/scenario/feature with string arg = BDD-ish
                if fn_text in ("describe", "it", "scenario", "feature",
                                "given", "when", "then", "Given", "When",
                                "Then", "test"):
                    args_nodes = [c for c in node.children
                                  if c.type == "arguments"]
                    if args_nodes:
                        for sc in args_nodes[0].children:
                            if sc.type == "string":
                                bdd_calls += 1
                                break
                # jest.mock / vi.mock / sinon.stub
                if fn_text in ("jest.mock", "vi.mock", "sinon.stub",
                                "sinon.spy", "sinon.mock"):
                    mock_calls += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return mods, bdd_calls, mock_calls


def _collect_java(code: bytes) -> Tuple[set, int, int]:
    parser = _get_parser("java")
    if parser is None:
        return set(), 0, 0
    tree = parser.parse(code)
    mods: set = set()
    bdd_calls = 0
    mock_calls = 0

    def text(n) -> str:
        return code[n.start_byte:n.end_byte].decode("utf8", errors="replace")

    def walk(node):
        nonlocal bdd_calls, mock_calls
        if node.type == "import_declaration":
            for c in node.children:
                if c.type in ("scoped_identifier", "identifier"):
                    mods.add(text(c))
                    break
        elif node.type == "marker_annotation" or node.type == "annotation":
            for c in node.children:
                if c.type == "identifier":
                    name = text(c)
                    if name in ("Given", "When", "Then", "And", "But"):
                        bdd_calls += 1
                    break
        elif node.type == "method_invocation":
            # mock(), verify(), when()
            ts = text(node)[:64]
            if ts.startswith("mock(") or ts.startswith("Mockito.") or \
               ts.startswith("verify(") or ts.startswith("when("):
                mock_calls += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return mods, bdd_calls, mock_calls


def _collect_go(code: bytes) -> Tuple[set, int, int]:
    parser = _get_parser("go")
    if parser is None:
        return set(), 0, 0
    tree = parser.parse(code)
    mods: set = set()

    def text(n) -> str:
        return code[n.start_byte:n.end_byte].decode("utf8", errors="replace")

    def walk(node):
        if node.type == "import_spec":
            for c in node.children:
                if c.type == "interpreted_string_literal":
                    mods.add(text(c).strip("\""))
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return mods, 0, 0


def _any_pro_match(mods: set, lang: str) -> int:
    pro = PRO_IMPORTS.get(lang, set())
    hits = 0
    for m in mods:
        for p in pro:
            if m == p or m.startswith(p + ".") or m.startswith(p + "/"):
                hits += 1
                break
    return hits


def _any_anti_match(mods: set, lang: str) -> int:
    anti = ANTI_IMPORTS.get(lang, set())
    hits = 0
    for m in mods:
        for p in anti:
            if m == p or m.startswith(p + ".") or m.startswith(p + "/"):
                hits += 1
                break
    return hits


def applies(diff_text: str) -> bool:
    """True iff the diff adds at least one test file we can parse."""
    by_path = parse_diff_added_by_file(diff_text)
    for path in by_path:
        # .feature files are inherently ATDD
        if path.lower().endswith(".feature"):
            return True
        if not TEST_PATH_RE.search(path):
            continue
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in EXT_TO_LANG:
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None

    pro = 0
    anti = 0
    feature_files = 0
    test_files_parsed = 0

    for path, code in by_path.items():
        if path.lower().endswith(".feature"):
            # any non-empty .feature added is a strong ATDD signal
            feature_files += 1
            pro += 2
            continue
        if not TEST_PATH_RE.search(path):
            continue
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        lang = EXT_TO_LANG.get(ext)
        if not lang:
            continue
        code_bytes = code.encode("utf8", errors="replace")
        if lang == "py":
            mods, bdd, mock = _collect_python(code_bytes)
        elif lang in ("js", "ts"):
            mods, bdd, mock = _collect_js_ts(code_bytes, lang)
        elif lang == "java":
            mods, bdd, mock = _collect_java(code_bytes)
        elif lang == "go":
            mods, bdd, mock = _collect_go(code_bytes)
        else:
            continue
        test_files_parsed += 1
        pro += _any_pro_match(mods, lang) + bdd
        anti += _any_anti_match(mods, lang) + mock

    if test_files_parsed == 0 and feature_files == 0:
        return None

    # Smoothed ratio: 0.5 baseline when no signal either way; pulls toward 1
    # for strong pro, toward 0 for strong anti. Constants tuned so 1 pro
    # vs 1 anti yields ~0.5; 3 pro / 0 anti ~ 0.75.
    return float((pro + 1.0) / (pro + anti + 2.0))
