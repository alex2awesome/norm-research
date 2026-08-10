"""a89: Behavior-focused and interaction-oriented testing.

Norm: tests should specify *observable* behavior via public interfaces and
mirror real interactions; they should *not* couple to implementation
internals.

Relationship to a131 (ATDD: tests verify user-visible behavior)
---------------------------------------------------------------
a89 is ADJACENT but not identical to a131. The norms sit on the same
spectrum (unit/white-box <-> behavior/black-box) but measure at different
granularities:

  a131  measures *file-level* style: which testing drivers/libraries the
        test file imports (selenium, playwright, supertest, mockito ...)
        and BDD-call presence (describe/it/given/when/then). It is a
        "what does this test reach for?" signal at the import seam.

  a89   measures *statement-level* assertion style WITHIN test bodies:
        is the test asserting on RETURN VALUES / STATE (assertEqual,
        expect(x).toBe(...), assertTrue) — implementation-internal — or
        on INTERACTIONS at a collaborator boundary (mock.assert_called*,
        Mockito.verify(...), expect(spy).toHaveBeenCalled(),
        sinon.assert.calledWith(...)) — closer to observable behavior.

A PR can score high on one and low on the other:
  - File imports `@playwright/test` (a131 +) but every assertion is
    `expect(internalCounter).toBe(3)` (a89 = state-oriented, ~0.0).
  - File mocks heavily (a131 -) but every assertion verifies the
    interaction surface of a public dependency (a89 = interaction-
    oriented, ~1.0).

Honest caveats (why PARTIALLY_THIN, not THIN)
---------------------------------------------
"Interaction-oriented" is a proxy for "behavior-focused" — these correlate
but are not the same thing. The norm's true subject ("does this test
mirror real user interaction with the public interface?") requires
knowing what the public interface is, which a diff does not directly
expose. We measure a structural shadow of the norm:

   interaction_assertions / (interaction_assertions + state_assertions)

with a small smoothing prior. This will misclassify:
  - Tests using mocks purely to inject stubs but asserting on returned
    state (counts as interaction but is state-oriented in spirit).
  - Tests using assertEqual on the result of a service call that DOES
    cross a public boundary (counts as state but is behavior-oriented).
  - Snapshot tests, property tests, and parameterized tests with no
    explicit assert call (silently abstain on the file).

We accept this noise because the resulting score is a different
*structural shadow* of the same articulability boundary that a131 picks
up — and the two signals are designed to be combined downstream.

Implementation
--------------
We use tree-sitter-python / -javascript / -typescript / -java to walk
test-file ASTs and count call expressions whose callee identifier matches:

  interaction_set (Python):
      assert_called, assert_called_once, assert_called_with,
      assert_called_once_with, assert_any_call, assert_has_calls,
      assert_not_called, called_with  (matched against `.attribute` tail)

  state_set (Python unittest):
      assertEqual, assertEquals, assertNotEqual, assertTrue, assertFalse,
      assertIs, assertIsNone, assertIsNotNone, assertIn, assertNotIn,
      assertGreater, assertLess, assertGreaterEqual, assertLessEqual,
      assertAlmostEqual, assertRaises, assertRaisesRegex, assertDictEqual,
      assertListEqual, assertSetEqual, assertTupleEqual

  state_set (Python pytest): bare `assert` STATEMENT (counted as state)

  Java interaction: Mockito.verify, verify(...).method(...) — we count
      identifier `verify` in method invocations; also `verifyZeroInteractions`,
      `verifyNoMoreInteractions`, `verifyNoInteractions`.
  Java state: assertEquals, assertTrue, assertFalse, assertNull,
      assertNotNull, assertSame, assertThat (when the matcher is a value
      matcher — we cannot disambiguate, so it counts as state).

  JS/TS interaction: any call whose callee chain ends in
      toHaveBeenCalled, toHaveBeenCalledWith, toHaveBeenCalledTimes,
      toHaveBeenLastCalledWith, toHaveBeenNthCalledWith, calledWith,
      calledOnce, called, calledOnceWith, calledOnceWithExactly,
      notCalled. Also sinon.assert.calledWith etc.
  JS/TS state: callee tail in
      toBe, toEqual, toStrictEqual, toBeTruthy, toBeFalsy, toBeNull,
      toBeUndefined, toBeDefined, toContain, toMatch, toMatchObject,
      toMatchSnapshot, toBeGreaterThan, toBeLessThan, toBeCloseTo,
      assertEqual, assert.equal, assert.deepEqual, assert.strictEqual.

We only inspect files that look like tests (path heuristic, the same
TEST_PATH_RE used by a131) — the norm doesn't apply to production code.

Score: smoothed ratio interaction / (interaction + state + smoothing).
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a89"
ASPECT_NAME = "Behavior-focused interaction-oriented testing"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java"]
CLASSIFICATION = "PARTIALLY_THIN"

# REGEX_OK: file_path — test paths are path patterns, not code.
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
}

# --- Python ---
PY_INTERACTION_ATTRS = {
    "assert_called", "assert_called_once",
    "assert_called_with", "assert_called_once_with",
    "assert_any_call", "assert_has_calls", "assert_not_called",
}
PY_STATE_ATTRS = {
    "assertEqual", "assertEquals", "assertNotEqual",
    "assertTrue", "assertFalse",
    "assertIs", "assertIsNot", "assertIsNone", "assertIsNotNone",
    "assertIn", "assertNotIn",
    "assertGreater", "assertLess", "assertGreaterEqual", "assertLessEqual",
    "assertAlmostEqual", "assertNotAlmostEqual",
    "assertRaises", "assertRaisesRegex", "assertWarns",
    "assertDictEqual", "assertListEqual", "assertSetEqual",
    "assertTupleEqual", "assertCountEqual",
    "assertIsInstance", "assertNotIsInstance",
    "assertRegex", "assertNotRegex",
    "assertMultiLineEqual", "assertSequenceEqual",
}

# --- JS/TS ---
JS_INTERACTION_TAILS = {
    "toHaveBeenCalled", "toHaveBeenCalledWith", "toHaveBeenCalledTimes",
    "toHaveBeenLastCalledWith", "toHaveBeenNthCalledWith",
    "calledWith", "calledOnce", "calledOnceWith", "calledOnceWithExactly",
    "notCalled", "neverCalledWith", "called",
}
JS_STATE_TAILS = {
    "toBe", "toEqual", "toStrictEqual",
    "toBeTruthy", "toBeFalsy",
    "toBeNull", "toBeUndefined", "toBeDefined", "toBeNaN",
    "toContain", "toContainEqual",
    "toMatch", "toMatchObject", "toMatchSnapshot",
    "toMatchInlineSnapshot",
    "toBeGreaterThan", "toBeGreaterThanOrEqual",
    "toBeLessThan", "toBeLessThanOrEqual",
    "toBeCloseTo", "toBeInstanceOf",
    "toThrow", "toThrowError",
    # node:assert / chai
    "equal", "deepEqual", "strictEqual", "deepStrictEqual",
    "notEqual", "notDeepEqual", "isTrue", "isFalse",
}

# --- Java ---
JAVA_INTERACTION_METHODS = {
    "verify", "verifyZeroInteractions", "verifyNoMoreInteractions",
    "verifyNoInteractions", "inOrder",
}
JAVA_STATE_METHODS = {
    "assertEquals", "assertNotEquals", "assertTrue", "assertFalse",
    "assertNull", "assertNotNull", "assertSame", "assertNotSame",
    "assertThrows", "assertDoesNotThrow", "assertAll",
    "assertArrayEquals", "assertIterableEquals", "assertLinesMatch",
    "assertTimeout", "assertTimeoutPreemptively",
    # Hamcrest / AssertJ value assertions
    "assertThat",
}

_PARSERS: Dict[str, object] = {}


def _get_parser(lang_short: str):
    if lang_short in _PARSERS:
        return _PARSERS[lang_short]
    try:
        from tree_sitter import Language, Parser
        if lang_short == "py":
            import tree_sitter_python as m
            lang = m.language()
        elif lang_short == "js":
            import tree_sitter_javascript as m
            lang = m.language()
        elif lang_short == "ts":
            import tree_sitter_typescript as m
            lang = m.language_typescript()
        elif lang_short == "java":
            import tree_sitter_java as m
            lang = m.language()
        else:
            return None
        _PARSERS[lang_short] = Parser(Language(lang))
        return _PARSERS[lang_short]
    except ImportError:
        return None


def _callee_attribute_tail(code: bytes, call_node) -> Optional[str]:
    """For a Python `call` node whose function is an attribute, return the
    final attribute name (e.g. for `mock_x.foo.assert_called_once` -> 'assert_called_once').
    For a plain identifier call, return that identifier.
    """
    if not call_node.children:
        return None
    fn = call_node.children[0]
    if fn.type == "identifier":
        return code[fn.start_byte:fn.end_byte].decode("utf8", "replace")
    if fn.type == "attribute":
        # attribute children end with the final identifier
        last_ident = None
        for c in fn.children:
            if c.type == "identifier":
                last_ident = c
        if last_ident is not None:
            return code[last_ident.start_byte:last_ident.end_byte].decode(
                "utf8", "replace")
    return None


def _collect_python(code: bytes) -> Tuple[int, int]:
    parser = _get_parser("py")
    if parser is None:
        return 0, 0
    tree = parser.parse(code)
    interaction = 0
    state = 0

    def walk(node):
        nonlocal interaction, state
        if node.type == "call":
            tail = _callee_attribute_tail(code, node)
            if tail is not None:
                if tail in PY_INTERACTION_ATTRS:
                    interaction += 1
                elif tail in PY_STATE_ATTRS:
                    state += 1
        elif node.type == "assert_statement":
            # pytest-style bare `assert <expr>` — state-oriented
            state += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return interaction, state


def _js_ts_callee_tail(code: bytes, call_node) -> Optional[str]:
    """For a JS/TS call_expression, return the trailing identifier of the
    callee (handles foo.bar.baz() and foo['bar'].baz())."""
    if not call_node.children:
        return None
    fn = call_node.children[0]
    if fn.type == "identifier":
        return code[fn.start_byte:fn.end_byte].decode("utf8", "replace")
    if fn.type == "member_expression":
        # last child of member_expression is the property identifier
        last_ident = None
        for c in fn.children:
            if c.type == "property_identifier" or c.type == "identifier":
                last_ident = c
        if last_ident is not None:
            return code[last_ident.start_byte:last_ident.end_byte].decode(
                "utf8", "replace")
    return None


def _collect_js_ts(code: bytes, lang: str) -> Tuple[int, int]:
    parser = _get_parser(lang)
    if parser is None:
        return 0, 0
    tree = parser.parse(code)
    interaction = 0
    state = 0

    def walk(node):
        nonlocal interaction, state
        if node.type == "call_expression":
            tail = _js_ts_callee_tail(code, node)
            if tail is not None:
                if tail in JS_INTERACTION_TAILS:
                    interaction += 1
                elif tail in JS_STATE_TAILS:
                    state += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return interaction, state


def _java_method_name(code: bytes, method_node) -> Optional[str]:
    """For a Java method_invocation, return the method name identifier."""
    # Structure: [object?] '.' name '(' args ')'  OR  name '(' args ')'
    # The method name is the `identifier` child that appears immediately
    # before the `argument_list` child.
    children = list(method_node.children)
    for i, c in enumerate(children):
        if c.type == "argument_list" and i > 0:
            prev = children[i - 1]
            if prev.type == "identifier":
                return code[prev.start_byte:prev.end_byte].decode(
                    "utf8", "replace")
    return None


def _collect_java(code: bytes) -> Tuple[int, int]:
    parser = _get_parser("java")
    if parser is None:
        return 0, 0
    tree = parser.parse(code)
    interaction = 0
    state = 0

    def walk(node):
        nonlocal interaction, state
        if node.type == "method_invocation":
            name = _java_method_name(code, node)
            if name is not None:
                if name in JAVA_INTERACTION_METHODS:
                    interaction += 1
                elif name in JAVA_STATE_METHODS:
                    state += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return interaction, state


def _is_test_file(path: str) -> Optional[str]:
    """Return language short-code if `path` looks like a test file and is in
    a supported language; else None."""
    if not TEST_PATH_RE.search(path):
        return None
    if "." not in path:
        return None
    ext = "." + path.rsplit(".", 1)[-1].lower()
    return EXT_TO_LANG.get(ext)


def applies(diff_text: str) -> bool:
    """True iff the diff adds at least one supported-language test file."""
    by_path = parse_diff_added_by_file(diff_text)
    for path in by_path:
        if _is_test_file(path) is not None:
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None

    interaction = 0
    state = 0
    files_with_assertions = 0

    for path, code in by_path.items():
        lang = _is_test_file(path)
        if lang is None:
            continue
        code_bytes = code.encode("utf8", "replace")
        if lang == "py":
            i, s = _collect_python(code_bytes)
        elif lang in ("js", "ts"):
            i, s = _collect_js_ts(code_bytes, lang)
        elif lang == "java":
            i, s = _collect_java(code_bytes)
        else:
            continue
        interaction += i
        state += s
        if (i + s) > 0:
            files_with_assertions += 1

    if files_with_assertions == 0:
        # We saw test files but no recognized assertions (snapshot-only,
        # property tests, helpers without asserts). Abstain.
        return None

    # Smoothed ratio: 1 prior count on each side so 0/0 (impossible here)
    # would be 0.5; 2 interaction / 0 state -> 0.6; 4 interaction / 0 state
    # -> 0.71; 0 interaction / 4 state -> 0.17.
    return float((interaction + 0.5) / (interaction + state + 1.0))
