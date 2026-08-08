"""a309: Tests included with changes — per-source-file test correspondence.

This metric measures whether the PR includes tests *for the specific source
files it touches*, not merely whether any test exists in the PR.

How this differs from a104 ("test presence and design quality")
---------------------------------------------------------------
a104 computes a composite of (test-line-ratio across the PR) and
(test-function-count). A PR that adds one giant test file unrelated to the
source changed will score high on a104.

a309 instead asks a structural alignment question: for each non-test source
file changed in the diff, is there *also* a plausibly-corresponding test
file changed? The correspondence is name-based at the file level:

  src/foo/bar.py        ↔ tests/test_bar.py, tests/foo/test_bar.py,
                          src/foo/bar_test.py, etc.
  src/foo/bar.go        ↔ src/foo/bar_test.go
  src/Foo.java          ↔ src/test/.../FooTest.java
  src/foo/bar.ts        ↔ tests/bar.test.ts, __tests__/bar.test.ts, etc.

Score = (# source files with a matched touched test file) / (# source files
touched). 1.0 means every source file changed has a corresponding test file
also changed; 0.0 means the PR changed source but no matching tests.

We deliberately keep correspondence at the *stem* level (basename without
extension) because tree-sitter would not help here — we are reasoning about
filesystem-level naming conventions, not code structure. The diff is parsed
with whatthepatch via the shared sandbox helper.

Classification
--------------
PARTIALLY_THIN. The norm "tests demonstrate the new or changed behavior" is
strictly thicker than file-level naming correspondence — a test file with
the right name might still assert nothing about the modified behavior. But
file-level correspondence is a sharper structural proxy than a PR-wide
test/source ratio: a fully-empty test file with the wrong name does not
fool this metric.

applies() requires the diff to touch at least one identifiable source code
file (so it abstains on docs-only / config-only PRs, which a309 cannot
score). A PR with only test files touched is scored 1.0 (vacuously
satisfied — there is no untested source change).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a309"
ASPECT_NAME = "Tests included with changes (file-level)"
TIER = 2
TOOLS = []  # pure path/diff reasoning; whatthepatch via sandbox
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go",
                    "Ruby", "Rust", "C", "C++", "C#", "Kotlin", "Scala",
                    "PHP", "Swift"]
CLASSIFICATION = "PARTIALLY_THIN"

# REGEX_OK: file_path — test path conventions are byte-level filesystem
# patterns (tests/, __tests__/, *_test.go, *.spec.ts). Paths aren't a
# language to parse; tree-sitter would not help.
TEST_PATH_RE = re.compile(
    r"(^|/)(test_|tests?/|spec/|specs?/|__tests__/|"
    r"features?/|e2e/|integration/|acceptance/)|"
    r"(_test|\.test|_spec|\.spec|Test|Tests|Spec|IT)\.[^/]+$",
    re.IGNORECASE,
)

# Java test layout uses a sibling tree: src/test/java/... vs src/main/java/...
# REGEX_OK: file_path — Maven/Gradle src/test vs src/main is a filesystem
# convention, not Java source code.
JAVA_TEST_DIR_RE = re.compile(r"(^|/)src/test/", re.IGNORECASE)
# REGEX_OK: file_path — same convention, opposite side.
JAVA_MAIN_DIR_RE = re.compile(r"(^|/)src/main/", re.IGNORECASE)

# Non-source artifacts we never count as "source needing a test".
NON_SOURCE_EXTS = (
    ".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml",
    ".lock", ".gitignore", ".cfg", ".ini", ".xml", ".properties",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".csv", ".tsv", ".sql", ".dockerfile", ".dockerignore",
    ".env", ".editorconfig", ".prettierrc", ".eslintrc",
)
# Extensions we consider "real source" — restricting to typed code files
# avoids treating e.g. shell scripts and config-like *.sh as untested code.
SOURCE_EXTS = (
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".java",
    ".go",
    ".rb",
    ".rs",
    ".c", ".h",
    ".cc", ".cpp", ".cxx", ".hpp",
    ".cs",
    ".kt", ".kts",
    ".scala",
    ".php",
    ".swift",
)


def _is_test_path(path: str) -> bool:
    if TEST_PATH_RE.search(path):
        return True
    if JAVA_TEST_DIR_RE.search(path):
        return True
    return False


def _is_source_path(path: str) -> bool:
    low = path.lower()
    if low.endswith(NON_SOURCE_EXTS):
        return False
    if not low.endswith(SOURCE_EXTS):
        return False
    return not _is_test_path(path)


def _stem(path: str) -> str:
    """basename without extension, lowercased."""
    base = path.rsplit("/", 1)[-1]
    if "." in base:
        base = base.rsplit(".", 1)[0]
    return base.lower()


def _candidate_test_stems(src_stem: str) -> Set[str]:
    """Generate the set of test-file stems that we count as 'matching' a
    given source stem. Conventions covered:

      Python : test_foo, foo_test
      Go     : foo_test
      JS/TS  : foo.test, foo.spec
      Java   : FooTest, FooTests, FooIT, TestFoo
      Generic: foo (same name under tests/ dir)
    """
    s = src_stem
    cands = {
        s,                # tests/foo.py mirrors src/foo.py
        f"test_{s}",      # python pytest convention
        f"{s}_test",      # python/go convention
        f"{s}.test",      # JS/TS jest/vitest
        f"{s}.spec",      # JS/TS jasmine/mocha/karma
        f"{s}test",       # FooTest.java → stem 'footest' after lower
        f"{s}tests",      # FooTests.java
        f"{s}it",         # FooIT.java (integration tests)
        f"test{s}",       # TestFoo.java
    }
    return cands


def _stems_of(paths: List[str]) -> Set[str]:
    return {_stem(p) for p in paths}


def _java_main_to_test(path: str) -> Optional[str]:
    """If a Java source under src/main/ has a sibling test name in src/test/,
    return the expected sibling stem. Used only for hint; the main matching
    is stem-based above.
    """
    if not JAVA_MAIN_DIR_RE.search(path):
        return None
    base = path.rsplit("/", 1)[-1]
    if "." not in base:
        return None
    stem = base.rsplit(".", 1)[0]
    return stem


def applies(diff_text: str) -> bool:
    """True iff the diff touches at least one source code file (test or
    non-test). PRs that are docs-only / config-only abstain (None)."""
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return False
    for p in by_path:
        low = p.lower()
        if low.endswith(SOURCE_EXTS):
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None

    source_paths: List[str] = []
    test_paths: List[str] = []
    for p in by_path:
        low = p.lower()
        if not low.endswith(SOURCE_EXTS):
            continue
        if _is_test_path(p):
            test_paths.append(p)
        else:
            source_paths.append(p)

    # If the PR touches only test files, the norm is vacuously satisfied:
    # there are no untested source changes.
    if not source_paths and test_paths:
        return 1.0
    if not source_paths:
        # No code at all — applies() should have returned False, but guard.
        return None

    test_stems = _stems_of(test_paths)
    if not test_stems:
        # Source changed, no test files at all → score 0.
        return 0.0

    matched = 0
    for sp in source_paths:
        s = _stem(sp)
        cands = _candidate_test_stems(s)
        # Java extra: src/main/java/com/x/Foo.java → src/test/java/com/x/FooTest.java
        # The stem-based test set already includes 'footest', 'tests', etc.
        if cands & test_stems:
            matched += 1
    return float(matched) / float(len(source_paths))
