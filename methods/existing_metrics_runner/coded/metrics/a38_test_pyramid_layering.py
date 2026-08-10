"""a38: Testing strategy and layering (test pyramid).

The norm asks for a *balanced* test mix: many fast unit tests, fewer
integration tests, fewest broad E2E/UI tests. A PR that adds tests should
contribute to that pyramid shape; a PR that adds only E2E tests (top-heavy /
"ice-cream cone" anti-pattern) violates it.

Relationship to a131 (ATDD with BDD framework + boundary drivers)
-----------------------------------------------------------------
a131 measures one dimension of test style — whether tests reach for
user-observable boundary drivers (selenium, playwright, supertest, .feature
files) vs. heavy mocking. It rewards "pro-ATDD" imports and penalizes
mock-heavy tests with a smoothed pro/anti ratio.

a38 reuses the *same* family of import signals but routes them to a
different question: instead of pro-vs-anti for a single dimension, we
*classify each test file into a layer* (unit, integration, e2e) and reward
pyramid SHAPE: many unit, some integration, few e2e. Specifically:

  - E2E layer    : path contains e2e/, ui/, system/, browser/, smoke/;
                   OR imports selenium, playwright, cypress, puppeteer,
                   webdriver, splinter.
  - Integration  : path contains integration/, functional/, contract/, api/;
                   OR imports HTTP clients used in tests (requests, httpx,
                   fastapi.testclient, django.test.Client, supertest,
                   rest-assured, mockmvc, net/http/httptest);
                   OR imports docker/testcontainers.
  - Unit         : path contains unit/, __tests__/, or generic tests/ with
                   no integration/e2e signal; OR file uses mock imports
                   (unittest.mock, jest.mock, sinon, mockito) — heavy
                   mocking is a hallmark of unit-layer isolation.

A file can be classified at most once; the rule of precedence is
E2E > Integration > Unit (the broadest scope wins, since a test that
spins up a browser is E2E even if it also mocks something).

Pyramid score:
  Let U, I, E = counts in each layer added by the diff, T = U+I+E.
  Ideal pyramid roughly U:I:E ~ 7:2:1 (Cohn's classic recommendation).
  We score by L1 distance from this target distribution after normalising.
  Bonuses: presence of >=2 distinct layers is rewarded (single-layer test
  suites violate the "balanced mix" clause regardless of which layer).

Honest caveats (PARTIALLY_THIN, not THIN)
-----------------------------------------
The layer classifier is structural: it sees imports and paths, not
runtime behavior. Concrete misclassifications we tolerate:

  - A test that imports requests but only hits a localhost in-process
    server is really a unit test by the pyramid's own criteria.
  - A test under tests/unit/ that spins up a real database is integration
    in spirit but classified as unit.
  - "Component" tests (mount-and-test a UI component without a browser)
    have no clean layer mapping and tend to be miscounted as unit.
  - Single-file PRs that add 1 test of any layer score modestly (we cannot
    evaluate "pyramid shape" on N=1).

This is similar in flavor to a131's pro/anti ratio (same import seam,
similar caveats around false positives) but distinct in its output: a131
answers "are tests user-facing?", a38 answers "are tests pyramid-shaped?".
A PR with a single Playwright test scores HIGH on a131 (pro-ATDD!) but
LOW on a38 (top-heavy ice-cream cone).
"""
from __future__ import annotations

import math
import re
from typing import Dict, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a38"
ASPECT_NAME = "Testing strategy and layering (pyramid)"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"

# REGEX_OK: file_path — test file conventions are path patterns, not code.
TEST_PATH_RE = re.compile(
    r"(^|/)(test_|tests?/|spec/|specs?/|__tests__/|features?/|e2e/|"
    r"integration/|acceptance/|functional/|unit/|ui/|system/|smoke/|"
    r"browser/|contract/)|"
    r"(_test|\.test|_spec|\.spec)\.[^/]+$",
    re.IGNORECASE,
)

# REGEX_OK: file_path — directory-segment classification of test paths.
E2E_PATH_RE = re.compile(
    r"(^|/)(e2e|ui|system|browser|smoke|end[-_]to[-_]end|acceptance)/",
    re.IGNORECASE,
)
# REGEX_OK: file_path — integration-layer path conventions.
INTEGRATION_PATH_RE = re.compile(
    r"(^|/)(integration|functional|contract|api|component)/",
    re.IGNORECASE,
)
# REGEX_OK: file_path — unit-layer path conventions.
UNIT_PATH_RE = re.compile(
    r"(^|/)(unit|__tests__|specs?)/",
    re.IGNORECASE,
)

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}

# Layer-tagging imports.  Match against module string (Python/JS/TS/Go) or
# fully-qualified import declaration (Java).
E2E_IMPORTS = {
    "py": {"selenium", "playwright", "splinter", "webdriver",
           "pytest_playwright"},
    "js": {"@playwright/test", "playwright", "cypress", "puppeteer",
           "webdriverio", "selenium-webdriver", "nightwatch"},
    "ts": {"@playwright/test", "playwright", "cypress", "puppeteer",
           "webdriverio", "selenium-webdriver", "nightwatch"},
    "java": {"org.openqa.selenium", "io.github.bonigarcia",
             "com.microsoft.playwright"},
    "go": {"github.com/tebeka/selenium", "github.com/chromedp/chromedp"},
}
INTEGRATION_IMPORTS = {
    "py": {"requests", "httpx", "fastapi.testclient",
           "starlette.testclient", "django.test", "rest_framework.test",
           "webtest", "testcontainers", "docker",
           "aiohttp"},
    "js": {"supertest", "axios", "node-fetch", "testcontainers",
           "@testcontainers/postgresql", "got"},
    "ts": {"supertest", "axios", "node-fetch", "testcontainers",
           "@testcontainers/postgresql", "got"},
    "java": {"io.restassured", "org.springframework.test.web.servlet",
             "org.testcontainers", "org.apache.http",
             "okhttp3"},
    "go": {"net/http/httptest", "github.com/testcontainers/testcontainers-go"},
}
UNIT_IMPORTS = {
    "py": {"unittest.mock", "mock", "pytest_mock"},
    "js": {"sinon", "ts-mockito", "jest-mock", "@sinonjs/sinon"},
    "ts": {"sinon", "ts-mockito", "jest-mock", "@sinonjs/sinon"},
    "java": {"org.mockito", "mockito"},
    "go": {"github.com/stretchr/testify/mock",
           "github.com/golang/mock/gomock"},
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


def _collect_imports_python(code: bytes) -> set:
    parser = _get_parser("py")
    if parser is None:
        return set()
    tree = parser.parse(code)
    mods: set = set()

    def text(n) -> str:
        return code[n.start_byte:n.end_byte].decode("utf8", errors="replace")

    def walk(node):
        if node.type == "import_statement":
            for c in node.children:
                if c.type in ("dotted_name", "aliased_import"):
                    mods.add(text(c).split(" as ")[0].strip())
        elif node.type == "import_from_statement":
            for c in node.children:
                if c.type == "dotted_name":
                    mods.add(text(c))
                    break
                if c.type == "relative_import":
                    break
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return mods


def _collect_imports_js_ts(code: bytes, lang: str) -> set:
    parser = _get_parser(lang)
    if parser is None:
        return set()
    tree = parser.parse(code)
    mods: set = set()

    def text(n) -> str:
        return code[n.start_byte:n.end_byte].decode("utf8", errors="replace")

    def walk(node):
        if node.type == "import_statement":
            for c in node.children:
                if c.type == "string":
                    mods.add(text(c).strip("'\"`"))
        elif node.type == "call_expression":
            if node.children:
                fn_node = node.children[0]
                fn_text = text(fn_node)
                if fn_text == "require" and len(node.children) >= 2:
                    args = node.children[1]
                    for sc in args.children:
                        if sc.type == "string":
                            mods.add(text(sc).strip("'\"`"))
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return mods


def _collect_imports_java(code: bytes) -> set:
    parser = _get_parser("java")
    if parser is None:
        return set()
    tree = parser.parse(code)
    mods: set = set()

    def text(n) -> str:
        return code[n.start_byte:n.end_byte].decode("utf8", errors="replace")

    def walk(node):
        if node.type == "import_declaration":
            for c in node.children:
                if c.type in ("scoped_identifier", "identifier"):
                    mods.add(text(c))
                    break
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return mods


def _collect_imports_go(code: bytes) -> set:
    parser = _get_parser("go")
    if parser is None:
        return set()
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
    return mods


def _import_hits(mods: set, registry: set) -> bool:
    for m in mods:
        for p in registry:
            if m == p or m.startswith(p + ".") or m.startswith(p + "/"):
                return True
    return False


def _classify_layer(path: str, mods: set, lang: str) -> Optional[str]:
    """Classify a test file as 'e2e', 'integration', 'unit', or None.

    Precedence: E2E > Integration > Unit (broadest scope wins).
    """
    if E2E_PATH_RE.search(path):
        return "e2e"
    if _import_hits(mods, E2E_IMPORTS.get(lang, set())):
        return "e2e"
    if INTEGRATION_PATH_RE.search(path):
        return "integration"
    if _import_hits(mods, INTEGRATION_IMPORTS.get(lang, set())):
        return "integration"
    if UNIT_PATH_RE.search(path):
        return "unit"
    if _import_hits(mods, UNIT_IMPORTS.get(lang, set())):
        return "unit"
    # Generic tests/ path with no specific layer signal: treat as unit
    # (the conventional default), but only if path actually looks like a
    # test file.
    if TEST_PATH_RE.search(path):
        return "unit"
    return None


def _imports_for(code: bytes, lang: str) -> set:
    if lang == "py":
        return _collect_imports_python(code)
    if lang in ("js", "ts"):
        return _collect_imports_js_ts(code, lang)
    if lang == "java":
        return _collect_imports_java(code)
    if lang == "go":
        return _collect_imports_go(code)
    return set()


def applies(diff_text: str) -> bool:
    """True iff the diff adds at least one parseable test file."""
    by_path = parse_diff_added_by_file(diff_text)
    for path in by_path:
        if not TEST_PATH_RE.search(path):
            continue
        if "." not in path:
            continue
        ext = "." + path.rsplit(".", 1)[-1].lower()
        if ext in EXT_TO_LANG:
            return True
        # .feature files count as e2e/acceptance
        if path.lower().endswith(".feature"):
            return True
    # Also applies if any e2e/-style path is added (even without code ext)
    for path in by_path:
        if E2E_PATH_RE.search(path) or INTEGRATION_PATH_RE.search(path):
            return True
    return False


# Target pyramid proportions (Cohn-style 7:2:1 unit:integration:e2e).
TARGET = {"unit": 0.7, "integration": 0.2, "e2e": 0.1}


def _pyramid_balance(counts: Dict[str, int]) -> float:
    """Score [0,1] for how pyramid-shaped the (unit,int,e2e) mix is.

    0.0 = all weight on a single non-unit layer (worst inversion).
    1.0 = proportions match TARGET exactly.
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    obs = {k: counts.get(k, 0) / total for k in TARGET}
    # L1 distance to TARGET, normalised: max possible L1 is 2.0.
    l1 = sum(abs(obs[k] - TARGET[k]) for k in TARGET)
    return max(0.0, 1.0 - l1 / 2.0)


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None

    counts: Dict[str, int] = {"unit": 0, "integration": 0, "e2e": 0}
    test_files_seen = 0

    for path, code in by_path.items():
        # .feature files are inherently acceptance/e2e style
        if path.lower().endswith(".feature"):
            counts["e2e"] += 1
            test_files_seen += 1
            continue
        if not TEST_PATH_RE.search(path) and \
                not E2E_PATH_RE.search(path) and \
                not INTEGRATION_PATH_RE.search(path):
            continue
        if "." not in path:
            continue
        ext = "." + path.rsplit(".", 1)[-1].lower()
        lang = EXT_TO_LANG.get(ext)
        mods: set = set()
        if lang is not None:
            mods = _imports_for(code.encode("utf8", errors="replace"), lang)
        layer = _classify_layer(path, mods, lang or "")
        if layer is None:
            continue
        counts[layer] += 1
        test_files_seen += 1

    if test_files_seen == 0:
        return None

    # Component score 1: pyramid shape fidelity.
    shape = _pyramid_balance(counts)

    # Component score 2: diversity bonus. A single-layer PR cannot meaningfully
    # demonstrate balance; multi-layer PRs are rewarded.
    n_layers = sum(1 for v in counts.values() if v > 0)
    if test_files_seen == 1:
        # Cannot really judge balance from 1 file; collapse toward 0.5 baseline.
        diversity = 0.5
    elif n_layers == 1:
        # Multiple tests but only one layer — still a single-layer suite.
        # Unit-only is more forgivable than e2e-only (cheap, fast),
        # so reward unit-only modestly and penalise e2e-only.
        if counts["unit"] > 0:
            diversity = 0.55
        elif counts["integration"] > 0:
            diversity = 0.35
        else:  # e2e-only — top-heavy "ice-cream cone"
            diversity = 0.15
    elif n_layers == 2:
        diversity = 0.75
    else:
        diversity = 1.0

    # Blend.  Shape dominates when we have enough files to estimate it;
    # diversity dominates when we have only 1-2 files.
    weight_shape = math.tanh(test_files_seen / 4.0)
    blended = weight_shape * shape + (1.0 - weight_shape) * diversity
    return float(max(0.0, min(1.0, blended)))
