"""a88: Testing mix (pyramid) and fast unit tests.

Norm
----
"Favor many small, fast unit tests; fewer integration/service tests; and
minimal E2E/UI tests. Keep tests focused and the suite quick and reliable."

This is the *testing pyramid* norm: among the test files in this PR, what
fraction look like cheap, isolated unit tests vs. integration / E2E? A PR
that adds 10 unit tests and 1 E2E test honors the pyramid; one that adds
0 unit tests and 5 Selenium tests inverts it.

Differentiation from related metrics
-------------------------------------
This metric sits in a cluster with a131, a89, a104, a309. It is NOT a
duplicate of any of them — each measures a different shadow of the
testing-quality articulability boundary:

  a104  presence/density of tests   (test-line-ratio + test-fn-count)
  a309  *which* source files have a matching test file (file-stem
        correspondence)
  a131  pro-ATDD style: user-boundary import drivers ratio at import seam
        — higher polarity = e2e/integration (it REWARDS playwright,
        selenium, supertest, cucumber)
  a89   interaction-vs-state assertion style WITHIN a test body
        (verify(...) / toHaveBeenCalled vs assertEqual / toBe)
  a88   pyramid: per-test-FILE classification into
        {unit, integration, e2e, unknown}, score = unit / classified.
        OPPOSITE POLARITY from a131: selenium/playwright lowers a88,
        unittest.mock-heavy raises a88.

a131 and a88 are *inversely* correlated by construction but they measure
different things: a131 sums import hits across all test files; a88
classifies each file then ratios at the file level. A PR with one
heavy-selenium test file and nine pure-mock test files scores high on a88
(9/10 = 0.9 unit) and ambiguous on a131 (the one selenium file might
dominate the import-hit count). The two are designed to be combined
downstream.

How the classifier works
------------------------
For each test file (path matches the test-path regex), we walk the
tree-sitter AST and extract imported modules and call expressions, then
apply a priority cascade:

  1. PATH HINT (highest priority):
       - `/e2e/`, `/end_to_end/`, `/uitests?/` → e2e
       - `/integration/`, `/it_tests?/`, `/system/`, `/acceptance/`,
         `/features?/` → integration
       - `/unit/`, `/units/` → unit
  2. IMPORT HINT:
       - any e2e driver (selenium, playwright, cypress, puppeteer,
         webdriver, splinter, browser-use, behave_webdriver) → e2e
       - any integration driver (requests, httpx, urllib3,
         sqlalchemy.create_engine, psycopg2, pymongo, redis,
         docker, testcontainers, kafka, aiohttp client, boto3,
         google.cloud, fastapi.testclient, django.test.Client,
         supertest, nock for integration HTTP) → integration
       - any heavy-mock import (unittest.mock, mock, pytest_mock,
         sinon, jest.mock, vi.mock, mockito) → unit
  3. FALLBACK: unknown (file is not counted in the denominator).

PATH wins because conventionally `tests/integration/foo_test.py` IS an
integration test by the project's own admission, even if its body only
calls `mock.patch`. The classifier reports the same answer the project
reports about itself.

Score
-----
    score = #unit_files / max(#classified_files, 1)

where #classified_files = unit + integration + e2e (unknown excluded).

If we found test files but zero were classified, we abstain (return None).
We do NOT abstain if classified_files >= 1 — even a single file gives a
0/1 or 1/1 reading that is informative.

Caveats (why PARTIALLY_THIN)
----------------------------
"Fast" cannot be measured statically. A test with no IO imports may still
spin a slow generator; a selenium test in `unit/` might still be slow.
The pyramid ratio is the *structural shadow* of the norm; "speed" itself
is a runtime property the diff does not expose. We mark PARTIALLY_THIN
to reflect this.

If a38 ("Testing strategy and layering (pyramid)") is implemented
elsewhere with semantic reasoning that supersedes the structural ratio,
a38 should be marked THICK and a88 retained as the structural surrogate.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Set, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a88"
ASPECT_NAME = "Testing pyramid: unit-test fraction"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"

# REGEX_OK: file_path — test path conventions are filesystem patterns
# (tests/, __tests__/, *_test.go, *.spec.ts). Paths aren't a language to
# parse; tree-sitter would not help.
TEST_PATH_RE = re.compile(
    r"(^|/)(test_|tests?/|spec/|specs?/|__tests__/|features?/|e2e/|"
    r"integration/|acceptance/|unit/|end_to_end/)|"
    r"(_test|\.test|_spec|\.spec|Test|Tests|IT)\.[^/]+$",
    re.IGNORECASE,
)

# REGEX_OK: file_path — path-segment classifier for the e2e layer hint.
_E2E_PATH_RE = re.compile(
    r"(^|/)(e2e|end_to_end|ui[_-]?tests?|browser[_-]?tests?|"
    r"selenium[_-]?tests?|smoke)/", re.IGNORECASE)
# REGEX_OK: file_path — path-segment classifier for the integration layer.
_INTEG_PATH_RE = re.compile(
    r"(^|/)(integration|it[_-]?tests?|system|sys[_-]?tests?|"
    r"acceptance|features?|functional|api[_-]?tests?|"
    r"contract|component[_-]?tests?)/", re.IGNORECASE)
# REGEX_OK: file_path — path-segment classifier for the unit layer.
_UNIT_PATH_RE = re.compile(
    r"(^|/)(unit|units|unit[_-]?tests?)/", re.IGNORECASE)

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}

# Imports that indicate a test drives a browser/UI → E2E.
E2E_IMPORTS = {
    "py": {
        "selenium", "playwright", "playwright.sync_api",
        "playwright.async_api", "splinter",
        "behave_webdriver", "robot", "robotframework",
    },
    "js": {
        "@playwright/test", "playwright", "cypress", "puppeteer",
        "webdriverio", "@wdio/cli", "nightwatch", "testcafe",
    },
    "ts": {
        "@playwright/test", "playwright", "cypress", "puppeteer",
        "webdriverio", "@wdio/cli", "nightwatch", "testcafe",
    },
    "java": {
        "org.openqa.selenium", "io.github.bonigarcia.wdm",
        "com.microsoft.playwright",
    },
    "go": {
        "github.com/tebeka/selenium",
        "github.com/playwright-community/playwright-go",
        "github.com/chromedp/chromedp",
    },
}

# Imports that indicate the test crosses a real process / service / DB / HTTP
# boundary → INTEGRATION.
INTEG_IMPORTS = {
    "py": {
        # HTTP clients used to hit a real server (or a test server)
        "requests", "httpx", "urllib3", "aiohttp",
        # In-process HTTP-server test clients live in a grey zone; we count
        # them as integration because they exercise the routing stack.
        "fastapi.testclient", "starlette.testclient",
        "django.test", "rest_framework.test", "webtest",
        # DB
        "sqlalchemy", "psycopg2", "psycopg", "pymongo", "redis",
        "mysql", "mysqlclient", "pymysql", "cx_Oracle",
        # Message queues / cloud
        "kafka", "confluent_kafka", "pika",
        "boto3", "google.cloud", "azure",
        # Container-based integration
        "testcontainers", "docker",
        # Acceptance / BDD frameworks
        "behave", "pytest_bdd",
    },
    "js": {
        "supertest", "axios", "node-fetch", "got", "request",
        "pg", "mysql", "mysql2", "mongodb", "mongoose", "redis",
        "ioredis", "kafkajs",
        "testcontainers", "dockerode",
        "@cucumber/cucumber", "cucumber",
    },
    "ts": {
        "supertest", "axios", "node-fetch", "got", "request",
        "pg", "mysql", "mysql2", "mongodb", "mongoose", "redis",
        "ioredis", "kafkajs",
        "testcontainers", "dockerode",
        "@cucumber/cucumber", "cucumber",
    },
    "java": {
        # HTTP
        "io.restassured", "okhttp3", "org.apache.http",
        "java.net.http",
        # In-process server test clients
        "org.springframework.test.web.servlet",
        "org.springframework.boot.test.web.client",
        # DB / JPA
        "javax.sql", "java.sql",
        "org.hibernate", "org.springframework.jdbc",
        # Containers
        "org.testcontainers",
        # Acceptance
        "io.cucumber", "cucumber.api",
    },
    "go": {
        "net/http", "net/http/httptest",
        "database/sql",
        "github.com/lib/pq", "go.mongodb.org/mongo-driver",
        "github.com/go-redis/redis", "github.com/redis/go-redis",
        "github.com/testcontainers/testcontainers-go",
        "github.com/cucumber/godog",
    },
}

# Imports that indicate isolation via mocking → UNIT.
UNIT_IMPORTS = {
    "py": {
        "unittest.mock", "mock", "pytest_mock", "freezegun",
        "responses", "requests_mock", "httpretty", "moto", "vcr",
    },
    "js": {
        "sinon", "jest-mock", "ts-mockito", "nock", "msw",
        "@testing-library/react", "@testing-library/dom",
        "fetch-mock",
    },
    "ts": {
        "sinon", "jest-mock", "ts-mockito", "nock", "msw",
        "@testing-library/react", "@testing-library/dom",
        "fetch-mock",
    },
    "java": {
        "org.mockito", "mockito",
        "org.powermock",
        "org.easymock", "easymock",
        "com.github.tomakehurst.wiremock",
    },
    "go": {
        "github.com/stretchr/testify/mock",
        "github.com/golang/mock/gomock",
        "go.uber.org/mock/gomock",
    },
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


def _extract_imports_python(code: bytes) -> Set[str]:
    parser = _get_parser("py")
    if parser is None:
        return set()
    tree = parser.parse(code)
    mods: Set[str] = set()

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


def _extract_imports_js_ts(code: bytes, lang: str) -> Set[str]:
    parser = _get_parser(lang)
    if parser is None:
        return set()
    tree = parser.parse(code)
    mods: Set[str] = set()

    def text(n) -> str:
        return code[n.start_byte:n.end_byte].decode("utf8", errors="replace")

    def walk(node):
        if node.type == "import_statement":
            for c in node.children:
                if c.type == "string":
                    s = text(c).strip("'\"`")
                    mods.add(s)
        elif node.type == "call_expression":
            if node.children:
                fn = node.children[0]
                fn_text = text(fn)
                if fn_text == "require" and len(node.children) >= 2:
                    args = node.children[1]
                    for sc in args.children:
                        if sc.type == "string":
                            mods.add(text(sc).strip("'\"`"))
                # jest.mock('mod') / vi.mock('mod') — the argument is the
                # mocked module, but the *call itself* signals mock-isolation
                # so we still mark this file as having unit-imports below by
                # adding a synthetic marker.
                if fn_text in ("jest.mock", "vi.mock"):
                    mods.add("__mock_call__")
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return mods


def _extract_imports_java(code: bytes) -> Set[str]:
    parser = _get_parser("java")
    if parser is None:
        return set()
    tree = parser.parse(code)
    mods: Set[str] = set()

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


def _extract_imports_go(code: bytes) -> Set[str]:
    parser = _get_parser("go")
    if parser is None:
        return set()
    tree = parser.parse(code)
    mods: Set[str] = set()

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


def _match_any(mods: Set[str], catalog: Set[str]) -> bool:
    """A module 'm' matches catalog entry 'p' iff m == p, m starts with
    'p.', or m starts with 'p/'. The latter handles Go import paths."""
    for m in mods:
        for p in catalog:
            if m == p:
                return True
            if m.startswith(p + "."):
                return True
            if m.startswith(p + "/"):
                return True
    return False


def _classify_path(path: str) -> Optional[str]:
    if _E2E_PATH_RE.search(path):
        return "e2e"
    if _INTEG_PATH_RE.search(path):
        return "integration"
    if _UNIT_PATH_RE.search(path):
        return "unit"
    return None


def _classify_imports(mods: Set[str], lang: str) -> Optional[str]:
    """Layer hint from imports. E2E beats integration beats unit because a
    test that imports Selenium AND requests is fundamentally an E2E test."""
    if _match_any(mods, E2E_IMPORTS.get(lang, set())):
        return "e2e"
    if _match_any(mods, INTEG_IMPORTS.get(lang, set())):
        return "integration"
    if _match_any(mods, UNIT_IMPORTS.get(lang, set())):
        return "unit"
    return None


def _is_test_file(path: str) -> Optional[str]:
    if not TEST_PATH_RE.search(path):
        return None
    if "." not in path:
        return None
    ext = "." + path.rsplit(".", 1)[-1].lower()
    return EXT_TO_LANG.get(ext)


def applies(diff_text: str) -> bool:
    """True iff the diff adds at least one supported-language test file we
    could classify."""
    by_path = parse_diff_added_by_file(diff_text)
    for path in by_path:
        if _is_test_file(path) is not None:
            return True
    return False


def _classify_file(path: str, code: str, lang: str) -> Optional[str]:
    # PATH HINT first — the project's own layering is the ground truth it
    # claims about its tests.
    path_hint = _classify_path(path)
    if path_hint is not None:
        return path_hint
    # IMPORT HINT next.
    code_bytes = code.encode("utf8", errors="replace")
    if lang == "py":
        mods = _extract_imports_python(code_bytes)
    elif lang in ("js", "ts"):
        mods = _extract_imports_js_ts(code_bytes, lang)
    elif lang == "java":
        mods = _extract_imports_java(code_bytes)
    elif lang == "go":
        mods = _extract_imports_go(code_bytes)
    else:
        return None
    return _classify_imports(mods, lang)


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None

    counts = {"unit": 0, "integration": 0, "e2e": 0}
    seen_any_test_file = False
    for path, code in by_path.items():
        lang = _is_test_file(path)
        if lang is None:
            continue
        seen_any_test_file = True
        layer = _classify_file(path, code, lang)
        if layer in counts:
            counts[layer] += 1
        # else: unknown — excluded from denominator

    if not seen_any_test_file:
        return None
    classified = counts["unit"] + counts["integration"] + counts["e2e"]
    if classified == 0:
        # We saw test files but could not place any of them on the pyramid.
        # Abstain — a constant fallback would be noise.
        return None
    return float(counts["unit"]) / float(classified)
