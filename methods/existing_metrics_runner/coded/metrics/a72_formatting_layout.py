"""a72: Formatting and layout conventions for readability.

Measures whether code ADDED in a PR diff conforms to the canonical formatter
for its language. For each touched language whose formatter is installed,
we write the added lines to a temp file and run the formatter in check-only
mode. The score is the fraction of touched files that are already formatted,
weighted by added-line count.

Per-language tooling (library-first; NO regex on code):
    Python      ruff format --check         (rc=1 iff file needs reformat)
    Go          gofmt -l                    (prints filename iff needs reformat)
    JS / JSX    prettier --check            (prints "[warn] <file>" iff dirty)
    TS / TSX    prettier --check            (same)
    Java        google-java-format -n       (prints filename iff needs reformat)

Score = (conforming added-lines) / (total added-lines across measurable langs).
Range [0, 1]. 1.0 = every touched, measurable file passes its formatter.

Abstention semantics:
- applies(): True iff the diff touches at least one file whose language has a
  formatter we COULD invoke (this method is cheap — we don't probe PATH here
  because tool availability can vary per run, and `applies` must be cheap).
- score() returns None when no measurable file's formatter is actually
  available on this machine (e.g. a Java-only PR on a JVM-less box). Java
  formatter shells to a JVM; if `java` is missing we just don't run it.

Important: we score added lines reconstructed from the diff, NOT the full
file. The reconstructed text is often syntactically incomplete (snippets of
hunks). Formatters tolerate partial code poorly — Python's ruff/black will
reject syntactically broken input. We treat tool errors (rc<0, parse error)
as "unmeasurable" for that file and exclude it from both numerator and
denominator, so a PR with all-broken snippets abstains rather than scoring 0.
"""
from __future__ import annotations

import shutil
from typing import Dict, List, Optional, Tuple

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a72"
ASPECT_NAME = "Formatting and layout conventions"
TIER = 3
TOOLS = ["ruff", "gofmt", "prettier", "google-java-format"]
APPLIES_TO_LANGS = ["Python", "Go", "JavaScript", "TypeScript", "Java"]
CLASSIFICATION = "THIN"

# Per-language extension groups.
LANG_EXTS: Dict[str, List[str]] = {
    "python": [".py", ".pyi"],
    "go": [".go"],
    "js": [".js", ".jsx", ".mjs", ".cjs"],
    "ts": [".ts", ".tsx"],
    "java": [".java"],
}

ALL_EXTS = [e for es in LANG_EXTS.values() for e in es]


def applies(diff_text: str) -> bool:
    """True iff the diff touches at least one file in a covered language.

    We do not probe PATH here (applies() must be cheap and side-effect-free).
    score() returns None if no covered file's formatter is actually installed.
    """
    return bool(added_files_by_ext(diff_text, ALL_EXTS))


# ---------------------------------------------------------------------------
# Per-language check helpers. Each takes {path: added_text} for ONE language,
# writes to a temp dir, runs the formatter, and returns
# (conforming_added_lines, measurable_added_lines).
# A file is "measurable" if the formatter ran cleanly on it (no parse error).
# ---------------------------------------------------------------------------

def _line_count(s: str) -> int:
    if not s:
        return 0
    return s.count("\n") + (0 if s.endswith("\n") else 1)


def _formatter_input(s: str) -> str:
    """Restore the terminal newline removed by the shared diff projection."""
    return s if not s or s.endswith("\n") else s + "\n"


def _check_python(by_path: Dict[str, str]) -> Optional[Tuple[int, int]]:
    """Run `ruff format --check` per file. rc==0 -> conforming, rc==1 -> needs
    reformat, rc>=2 -> parse error (file unmeasurable)."""
    if not have_tool("ruff"):
        return None
    conforming_lines = 0
    measurable_lines = 0
    for src_path, content in by_path.items():
        rc, _out, _err = run(
            ["ruff", "format", "--check", "--no-cache",
             "--stdin-filename", src_path, "-"],
            stdin=_formatter_input(content),
            timeout=10.0,
        )
        lines = _line_count(content)
        if rc == 0:
            conforming_lines += lines
            measurable_lines += lines
        elif rc == 1:
            measurable_lines += lines
        # rc >= 2 is a parse/tool failure and remains unmeasurable.
    return conforming_lines, measurable_lines


def _check_go(by_path: Dict[str, str]) -> Optional[Tuple[int, int]]:
    """gofmt -l <dir> prints, one per line, paths needing formatting."""
    if not have_tool("gofmt"):
        return None
    normalized = {path: _formatter_input(content) for path, content in by_path.items()}
    td = write_temp_files(normalized)
    try:
        conforming_lines = 0
        measurable_lines = 0
        for file_path, content in zip(sorted(td.iterdir()), by_path.values()):
            rc, out, _err = run(["gofmt", "-l", str(file_path)], timeout=10.0)
            if rc < 0:
                return None
            if rc != 0:
                continue
            lines = _line_count(content)
            measurable_lines += lines
            if not out.strip():
                conforming_lines += lines
        return conforming_lines, measurable_lines
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _check_prettier(by_path: Dict[str, str]) -> Optional[Tuple[int, int]]:
    """prettier --check <file>. Per file: rc=0 -> conforming, rc=1 -> dirty,
    rc=2 -> parse error (treated as unmeasurable, excluded). We invoke
    per-file so a parse error on snippet A doesn't poison snippet B."""
    if not have_tool("prettier"):
        return None
    normalized = {path: _formatter_input(content) for path, content in by_path.items()}
    td = write_temp_files(normalized)
    try:
        files_on_disk = sorted(str(p) for p in td.iterdir())
        conforming_lines = 0
        measurable_lines = 0
        # Iterate paired with by_path so we can attribute line counts.
        contents_in_order = list(by_path.values())
        for f, content in zip(files_on_disk, contents_in_order):
            rc, _out, err = run(
                ["prettier", "--check", "--no-config", "--no-editorconfig", f],
                timeout=15.0,
            )
            if rc < 0:
                return None
            if rc >= 2 or "[error]" in err:
                # parse error -> unmeasurable
                continue
            lines = _line_count(content)
            measurable_lines += lines
            if rc == 0:
                conforming_lines += lines
        if measurable_lines == 0:
            return None
        return conforming_lines, measurable_lines
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _check_java(by_path: Dict[str, str]) -> Optional[Tuple[int, int]]:
    """google-java-format -n <files> prints filenames needing reformat. Needs
    a JVM; we just try and bail out on tool-missing / parse-error."""
    if not have_tool("google-java-format"):
        return None
    normalized = {path: _formatter_input(content) for path, content in by_path.items()}
    td = write_temp_files(normalized)
    try:
        files_on_disk = sorted(str(p) for p in td.iterdir())
        # Parse errors go to stderr with "...:LL: error:" lines. If everything
        # erroed, treat as unmeasurable.
        # On success, stdout lists files needing reformat (one per line).
        # We cannot perfectly distinguish "clean" from "errored" per file
        # without per-file invocations, so do per-file to be safe.
        conforming_lines = 0
        measurable_lines = 0
        for f, content in zip(files_on_disk, by_path.values()):
            rc1, out1, err1 = run(
                ["google-java-format", "-n", f], timeout=30.0,
            )
            if rc1 < 0:
                return None
            had_error = (rc1 != 0) or ("error:" in err1)
            if had_error:
                # parse error -> unmeasurable
                continue
            lines = _line_count(content)
            measurable_lines += lines
            # If stdout contains the file path, it needs reformat.
            if f.strip() in out1:
                pass  # dirty, not ok
            else:
                conforming_lines += lines
        return conforming_lines, measurable_lines
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------

LANG_CHECKERS = {
    "python": _check_python,
    "go": _check_go,
    "js": _check_prettier,
    "ts": _check_prettier,
    "java": _check_java,
}


def score(diff_text: str) -> Optional[float]:
    # Bucket added files by language.
    per_lang: Dict[str, Dict[str, str]] = {}
    for lang, exts in LANG_EXTS.items():
        files = added_files_by_ext(diff_text, exts)
        if files:
            per_lang[lang] = files
    if not per_lang:
        return None

    total_conforming_lines = 0
    total_measurable_lines = 0
    any_tool_ran = False
    for lang, files in per_lang.items():
        checker = LANG_CHECKERS[lang]
        res = checker(files)
        if res is None:
            # tool not installed for this language; just skip it
            continue
        conforming_lines, measurable_lines = res
        if measurable_lines == 0:
            continue
        any_tool_ran = True
        total_measurable_lines += measurable_lines
        total_conforming_lines += conforming_lines

    if not any_tool_ran or total_measurable_lines == 0:
        return None
    return float(total_conforming_lines / total_measurable_lines)
