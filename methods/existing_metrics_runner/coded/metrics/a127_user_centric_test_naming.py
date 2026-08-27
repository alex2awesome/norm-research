"""a127: Behavior-focused / user-centric testing.

Norm: tests validate externally visible, user-relevant behavior and avoid
brittle assertions tied to implementation details or pixel fidelity.

Relationship to a131, a89, a104 — and why a127 has its own structural shadow
---------------------------------------------------------------------------
The "user-centric testing" cluster has substantial overlap. The honest
question is whether a127 has a measurable angle that is NOT already a131 or
a89:

  a131 — measures *file-level* imports/drivers: does this test file import
         selenium / playwright / supertest / mockito / pytest_bdd? Does it
         call describe/it/given/when/then? It is the "test-style stack" seam.

  a89  — measures *assertion-style* WITHIN test bodies: interaction
         assertions (assert_called_*, verify(), toHaveBeenCalled) vs state
         assertions (assertEqual, toBe, assertThat). Ratio = behavior-vs-
         implementation orientation at the assert call.

  a104 — measures test PRESENCE (count of test functions, test-line ratio
         relative to source). Quantity, not orientation.

a127's distinct angle: **test NAMES as user-readable behavior specs, plus
pixel/snapshot-fidelity anti-signal.**

Two near-orthogonal sub-signals neither a131 nor a89 capture:

  (1) Name readability.
      A test named `it("returns 401 when token is missing")` or
      `def test_user_can_log_in_with_valid_credentials():` SPECIFIES
      user-visible behavior in English. A test named `test_handler_001`,
      `test_method`, `it("works")`, `it("test 1")` does not. The norm
      "validate externally visible behavior" is literally written into the
      test name when the team is doing this well — that's the BDD/Hoare-
      triple discipline. Walking AST + counting "English-sentence-like"
      identifiers/strings is a structural shadow of intent expression that
      a131 (imports) and a89 (assert call) miss entirely.

  (2) Pixel/snapshot fidelity (anti-signal).
      The a127 description explicitly calls out "avoid... pixel fidelity"
      assertions. Snapshot tests (toMatchSnapshot, toMatchInlineSnapshot,
      `expect(...).toMatchImageSnapshot()`, pytest-image-diff, jest-image-
      snapshot) and exact-CSS/pixel assertions are brittle to
      implementation-detail changes. They couple to *rendered representation*
      rather than *user-observable behavior*. We count these as a penalty.

Note overlap with a89 on snapshot: a89 lists toMatchSnapshot as `state`,
not as a specific anti-signal — its ratio just deflates. a127 treats pixel/
image/css-snapshot calls as a STRONGER anti-signal than ordinary state
assertions, because the norm a127 names explicitly singles out "pixel
fidelity" as the failure mode.

Caveats (this is PARTIALLY_THIN, not THIN)
------------------------------------------
- "English-sentence-like" is a surface heuristic. A test named
  `test_foo_bar_baz_qux` (5 underscored words, no verb) is technically
  multi-token English but conveys nothing about user behavior. We mitigate
  by requiring presence of a verb-like token (`returns`, `should`, `when`,
  `does`, `raises`, etc.) for the "user-centric" bucket.
- Snapshot tests CAN be appropriate when the snapshot IS the user-visible
  artifact (e.g. an HTML email rendering). We over-penalize this case.
- a127 should be combined with a89 (assertion style) and a131 (driver
  imports) downstream. The three measure different structural shadows of
  the same articulability boundary.

CLASSIFICATION: PARTIALLY_THIN.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a127"
ASPECT_NAME = "Behavior-focused / user-centric testing (names + pixel-fidelity)"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"

# REGEX_OK: file_path — test path conventions are byte patterns, not code.
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

# Verb-like / spec-language tokens. Presence of any of these in a test name
# (case-insensitive substring) is the signal that the name *describes*
# behavior rather than just labels a slot.
SPEC_VERBS = {
    "should", "returns", "return", "raises", "raise", "throws", "throw",
    "creates", "create", "deletes", "delete", "updates", "update",
    "rejects", "reject", "accepts", "accept", "allows", "allow",
    "denies", "deny", "fails", "fail", "succeeds", "succeed",
    "validates", "validate", "produces", "produce", "renders", "render",
    "handles", "handle", "dispatches", "dispatch", "emits", "emit",
    "is", "are", "has", "have", "does", "do", "can", "cannot", "will",
    "when", "if", "given", "then", "with", "without", "for",
    "shows", "show", "hides", "hide", "navigates", "navigate",
    "loads", "load", "saves", "save", "exists", "exist",
    "matches", "match", "responds", "respond", "redirects", "redirect",
    "logs", "log", "ignores", "ignore", "skips", "skip",
    "passes", "pass", "calls", "call", "sets", "set", "gets", "get",
    "computes", "compute", "parses", "parse", "encodes", "encode",
    "decodes", "decode", "filters", "filter", "sorts", "sort",
    "yields", "yield", "errors", "error", "warns", "warn",
}

# Snapshot / pixel-fidelity callee tails (JS/TS). Anti-signal: these couple
# to rendered representation, not observable behavior.
JS_SNAPSHOT_TAILS = {
    "toMatchSnapshot", "toMatchInlineSnapshot",
    "toMatchImageSnapshot", "toMatchTrimmedInlineSnapshot",
    "toMatchSpecificSnapshot",
}
# Python snapshot/pixel libs — names exposed at call level.
PY_SNAPSHOT_NAMES = {
    "assert_match", "assert_image_equal", "compare_images",
    "snapshot",  # syrupy-style fixture name appears as call target
}
# Java image-diff method names (rare; included for completeness).
JAVA_SNAPSHOT_NAMES = {"assertImageEquals", "compareImages"}

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
        elif lang_short == "go":
            import tree_sitter_go as m
            lang = m.language()
        else:
            return None
        _PARSERS[lang_short] = Parser(Language(lang))
        return _PARSERS[lang_short]
    except ImportError:
        return None


# REGEX_OK: tool_output — splitting test names on word boundaries is text
# tokenization on identifiers, not parsing source code.
_NAME_SPLIT_RE = re.compile(r"[_\s\-]+|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _name_tokens(name: str) -> List[str]:
    """Split a test name (identifier or BDD string) into lowercase tokens.
    Handles snake_case, kebab-case, camelCase, PascalCase, and space-
    separated BDD strings."""
    name = name.strip()
    if name.lower().startswith("test"):
        name = name[4:].lstrip("_-")
    if name.lower().startswith("it"):
        name = name[2:].lstrip("_-")
    parts = [p for p in _NAME_SPLIT_RE.split(name) if p]
    return [p.lower() for p in parts]


def _is_user_centric_name(name: str) -> bool:
    """Heuristic: the test name conveys observable behavior.

    Criteria (must satisfy all):
      - at least 3 meaningful tokens (after stripping test_/it prefix)
      - at least one token is a spec verb / spec-language word
      - not purely numeric tail (test_001, test_2a)
    """
    toks = _name_tokens(name)
    if len(toks) < 3:
        return False
    # numeric-tail tests like test_handler_001 / test_method_2a
    # REGEX_OK: tool_output — matching a tokenized name fragment, not source code.
    if toks and re.fullmatch(r"\d+[a-z]?", toks[-1]):
        return False
    if not any(t in SPEC_VERBS for t in toks):
        return False
    return True


def _collect_python(code: bytes) -> Tuple[int, int, int]:
    """Return (test_funcs_total, user_centric_funcs, pixel_calls)."""
    parser = _get_parser("py")
    if parser is None:
        return 0, 0, 0
    tree = parser.parse(code)
    total = 0
    centric = 0
    pixel = 0

    def text(n) -> str:
        return code[n.start_byte:n.end_byte].decode("utf8", "replace")

    def walk(node):
        nonlocal total, centric, pixel
        if node.type == "function_definition":
            # children: 'def' identifier parameters ':' block
            name_node = None
            for c in node.children:
                if c.type == "identifier":
                    name_node = c
                    break
            if name_node is not None:
                nm = text(name_node)
                if nm.startswith("test"):
                    total += 1
                    if _is_user_centric_name(nm):
                        centric += 1
        elif node.type == "call":
            # detect snapshot-style call: callee name or attribute tail in
            # PY_SNAPSHOT_NAMES, or function literal containing "snapshot" /
            # "image" in image-diff style.
            if node.children:
                fn = node.children[0]
                if fn.type == "identifier":
                    nm = text(fn)
                    if nm in PY_SNAPSHOT_NAMES:
                        pixel += 1
                elif fn.type == "attribute":
                    last = None
                    for c in fn.children:
                        if c.type == "identifier":
                            last = c
                    if last is not None:
                        nm = text(last)
                        if nm in PY_SNAPSHOT_NAMES:
                            pixel += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return total, centric, pixel


def _js_ts_callee_tail(code: bytes, call_node) -> Optional[str]:
    if not call_node.children:
        return None
    fn = call_node.children[0]
    if fn.type == "identifier":
        return code[fn.start_byte:fn.end_byte].decode("utf8", "replace")
    if fn.type == "member_expression":
        last = None
        for c in fn.children:
            if c.type in ("property_identifier", "identifier"):
                last = c
        if last is not None:
            return code[last.start_byte:last.end_byte].decode("utf8", "replace")
    return None


def _collect_js_ts(code: bytes, lang: str) -> Tuple[int, int, int]:
    """Return (it_describe_total, user_centric_strings, snapshot_calls).

    For BDD calls (it/test/describe with a string label), we count the
    *string* as a candidate test "name" and check if it reads as a user-
    centric behavior spec.
    """
    parser = _get_parser(lang)
    if parser is None:
        return 0, 0, 0
    tree = parser.parse(code)
    total = 0
    centric = 0
    snap = 0

    def text(n) -> str:
        return code[n.start_byte:n.end_byte].decode("utf8", "replace")

    def walk(node):
        nonlocal total, centric, snap
        if node.type == "call_expression":
            tail = _js_ts_callee_tail(code, node)
            if tail in ("it", "test", "describe", "specify"):
                # find string label
                args = [c for c in node.children if c.type == "arguments"]
                if args:
                    for sc in args[0].children:
                        if sc.type == "string":
                            # tree-sitter: string node has string_fragment children
                            frag = []
                            for f in sc.children:
                                if f.type == "string_fragment":
                                    frag.append(text(f))
                            s = "".join(frag) if frag else text(sc).strip("'\"`")
                            total += 1
                            if _is_user_centric_name(s):
                                centric += 1
                            break
            # function declarations: identifier callees are not used for tests
            # here; JS uses string-labelled it()/test(). function_declaration
            # is handled below.
            if tail in JS_SNAPSHOT_TAILS:
                snap += 1
        elif node.type == "function_declaration":
            for c in node.children:
                if c.type == "identifier":
                    nm = text(c)
                    if nm.startswith("test"):
                        total += 1
                        if _is_user_centric_name(nm):
                            centric += 1
                    break
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return total, centric, snap


def _collect_java(code: bytes) -> Tuple[int, int, int]:
    """Return (test_methods_total, user_centric_methods, snapshot_calls)."""
    parser = _get_parser("java")
    if parser is None:
        return 0, 0, 0
    tree = parser.parse(code)
    total = 0
    centric = 0
    snap = 0

    def text(n) -> str:
        return code[n.start_byte:n.end_byte].decode("utf8", "replace")

    # Track whether the next method_declaration is preceded by @Test or
    # @DisplayName (modifiers children come first inside method_declaration).
    def method_has_test_annotation(method_node) -> Tuple[bool, Optional[str]]:
        """Return (has_@Test, @DisplayName-string-or-None)."""
        has_test = False
        display = None
        for c in method_node.children:
            if c.type == "modifiers":
                for m in c.children:
                    if m.type in ("marker_annotation", "annotation"):
                        # find identifier child
                        ann_name = None
                        for x in m.children:
                            if x.type == "identifier":
                                ann_name = text(x)
                                break
                        if ann_name == "Test":
                            has_test = True
                        elif ann_name == "DisplayName":
                            # find the string arg
                            for x in m.children:
                                if x.type == "annotation_argument_list":
                                    for y in x.children:
                                        if y.type == "string_literal":
                                            display = text(y).strip('"')
                                            break
                                    break
        return has_test, display

    def walk(node):
        nonlocal total, centric, snap
        if node.type == "method_declaration":
            has_test, display = method_has_test_annotation(node)
            # Find method name
            name = None
            children = list(node.children)
            for i, c in enumerate(children):
                if c.type == "identifier":
                    name = text(c)
                    # last identifier before formal_parameters
                # we want the identifier just before formal_parameters
            # Better: pick identifier whose next sibling is formal_parameters
            for i, c in enumerate(children):
                if c.type == "formal_parameters" and i > 0:
                    prev = children[i - 1]
                    if prev.type == "identifier":
                        name = text(prev)
                    break
            if has_test or (name and name.startswith("test")):
                total += 1
                # Prefer @DisplayName if present (BDD-style English label)
                spec_source = display if display else (name or "")
                if _is_user_centric_name(spec_source):
                    centric += 1
        elif node.type == "method_invocation":
            # Snapshot-like calls (rare)
            children = list(node.children)
            for i, c in enumerate(children):
                if c.type == "argument_list" and i > 0:
                    prev = children[i - 1]
                    if prev.type == "identifier":
                        nm = text(prev)
                        if nm in JAVA_SNAPSHOT_NAMES:
                            snap += 1
                    break
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return total, centric, snap


def _collect_go(code: bytes) -> Tuple[int, int, int]:
    """Go test funcs are top-level func TestXxx(t *testing.T) — name tokens
    come from CamelCase split of Xxx. Snapshot is rarer in Go; skip."""
    parser = _get_parser("go")
    if parser is None:
        return 0, 0, 0
    tree = parser.parse(code)
    total = 0
    centric = 0

    def text(n) -> str:
        return code[n.start_byte:n.end_byte].decode("utf8", "replace")

    def walk(node):
        nonlocal total, centric
        if node.type == "function_declaration":
            for c in node.children:
                if c.type == "identifier":
                    nm = text(c)
                    if nm.startswith("Test") and len(nm) > 4:
                        total += 1
                        if _is_user_centric_name(nm):
                            centric += 1
                    break
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return total, centric, 0


def _is_test_file(path: str) -> Optional[str]:
    if not TEST_PATH_RE.search(path):
        return None
    if "." not in path:
        return None
    ext = "." + path.rsplit(".", 1)[-1].lower()
    return EXT_TO_LANG.get(ext)


def applies(diff_text: str) -> bool:
    by_path = parse_diff_added_by_file(diff_text)
    for path in by_path:
        if _is_test_file(path) is not None:
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None

    total = 0
    centric = 0
    pixel = 0

    for path, code in by_path.items():
        lang = _is_test_file(path)
        if lang is None:
            continue
        b = code.encode("utf8", "replace")
        if lang == "py":
            t, c, p = _collect_python(b)
        elif lang in ("js", "ts"):
            t, c, p = _collect_js_ts(b, lang)
        elif lang == "java":
            t, c, p = _collect_java(b)
        elif lang == "go":
            t, c, p = _collect_go(b)
        else:
            continue
        total += t
        centric += c
        pixel += p

    if total == 0:
        # We saw test files but extracted no test names — partial diffs,
        # unfamiliar style. Abstain.
        return None

    # Smoothed user-centric rate, then pixel-fidelity penalty.
    # Base: (centric + 0.5) / (total + 1.0)  — Laplace prior 0.5
    base = (centric + 0.5) / (total + 1.0)
    # Each pixel/snapshot assertion subtracts a small amount, capped at 0.
    # Constant chosen so 4 snapshot calls in a 10-name file moves score ~0.16.
    penalty = min(0.4, 0.04 * pixel)
    s = base - penalty
    return float(max(0.0, min(1.0, s)))
