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
# (n_conforming_files, n_measurable_files, total_added_lines_measurable).
# A file is "measurable" if the formatter ran cleanly on it (no parse error).
# ---------------------------------------------------------------------------

def _line_count(s: str) -> int:
    if not s:
        return 0
    return s.count("\n") + (0 if s.endswith("\n") else 1)


def _check_python(by_path: Dict[str, str]) -> Optional[Tuple[int, int, int]]:
    """Run `ruff format --check` per file. rc==0 -> conforming, rc==1 -> needs
    reformat, rc>=2 -> parse error (file unmeasurable)."""
    if not have_tool("ruff"):
        return None
    td = write_temp_files(by_path)
    try:
        n_ok = 0
        n_meas = 0
        n_lines_meas = 0
        for src_path, content in by_path.items():
            # Find the file we wrote for this src_path (mirrors write_temp_files'
            # ordering). Simpler: re-derive by running one ruff invocation per
            # file using --stdin-filename so we can attribute results.
            rc, _out, _err = run(
                ["ruff", "format", "--check", "--no-cache",
                 "--stdin-filename", src_path, "-"],
                stdin=content,
                timeout=10.0,
            )
            if rc == 0:
                n_ok += 1
                n_meas += 1
                n_lines_meas += _line_count(content)
            elif rc == 1:
                n_meas += 1
                n_lines_meas += _line_count(content)
            else:
                # parse error / tool failure -> exclude
                continue
        return n_ok, n_meas, n_lines_meas
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _check_go(by_path: Dict[str, str]) -> Optional[Tuple[int, int, int]]:
    """gofmt -l <dir> prints, one per line, paths needing formatting."""
    if not have_tool("gofmt"):
        return None
    td = write_temp_files(by_path)
    try:
        rc, out, _err = run(["gofmt", "-l", str(td)], timeout=10.0)
        if rc < 0:
            return None
        # gofmt -l prints empty when all-OK; prints filenames otherwise.
        # If it fails to parse a file it writes to stderr and rc != 0.
        # We treat the WHOLE directory as one language batch: lines with a
        # filename indicate that file needs formatting.
        dirty_files = {ln.strip() for ln in out.splitlines() if ln.strip()}
        n_total = len(by_path)
        n_dirty = len(dirty_files)
        # If gofmt errored on some files (rc!=0 and no stdout listing), we
        # conservatively count only files that we can attribute. Here we
        # assume non-errored files = all when stdout-only signalling is used.
        n_ok = n_total - n_dirty
        n_lines = sum(_line_count(c) for c in by_path.values())
        return n_ok, n_total, n_lines
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _check_prettier(by_path: Dict[str, str]) -> Optional[Tuple[int, int, int]]:
    """prettier --check <file>. Per file: rc=0 -> conforming, rc=1 -> dirty,
    rc=2 -> parse error (treated as unmeasurable, excluded). We invoke
    per-file so a parse error on snippet A doesn't poison snippet B."""
    if not have_tool("prettier"):
        return None
    td = write_temp_files(by_path)
    try:
        files_on_disk = sorted(str(p) for p in td.iterdir())
        n_ok = 0
        n_meas = 0
        n_lines = 0
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
            n_meas += 1
            n_lines += _line_count(content)
            if rc == 0:
                n_ok += 1
            # rc == 1: dirty (don't increment n_ok)
        if n_meas == 0:
            return None
        return n_ok, n_meas, n_lines
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _check_java(by_path: Dict[str, str]) -> Optional[Tuple[int, int, int]]:
    """google-java-format -n <files> prints filenames needing reformat. Needs
    a JVM; we just try and bail out on tool-missing / parse-error."""
    if not have_tool("google-java-format"):
        return None
    td = write_temp_files(by_path)
    try:
        files_on_disk = sorted(str(p) for p in td.iterdir())
        rc, out, err = run(
            ["google-java-format", "-n", *files_on_disk],
            timeout=30.0,  # JVM startup is slow
        )
        if rc < 0:
            return None
        # Parse errors go to stderr with "...:LL: error:" lines. If everything
        # erroed, treat as unmeasurable.
        # On success, stdout lists files needing reformat (one per line).
        # We cannot perfectly distinguish "clean" from "errored" per file
        # without per-file invocations, so do per-file to be safe.
        n_ok = 0
        n_meas = 0
        n_lines = 0
        for f in files_on_disk:
            rc1, out1, err1 = run(
                ["google-java-format", "-n", f], timeout=30.0,
            )
            if rc1 < 0:
                return None
            had_error = (rc1 != 0) or ("error:" in err1)
            if had_error:
                # parse error -> unmeasurable
                continue
            n_meas += 1
            # If stdout contains the file path, it needs reformat.
            if f.strip() in out1:
                pass  # dirty, not ok
            else:
                n_ok += 1
            # Attribute added-line count
            # Find the original src_path matching this f's index
        # We can't easily map back to src_path -> use uniform per-measurable.
        # Simplification: per-file added-lines from by_path values in order.
        contents = list(by_path.values())
        # Take the first n_meas contents' line counts as the measurable lines
        # (best-effort; we don't know which specific files errored).
        n_lines = sum(_line_count(c) for c in contents[:n_meas])
        return n_ok, n_meas, n_lines
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
        n_ok, n_meas, n_lines = res
        if n_meas == 0:
            continue
        any_tool_ran = True
        # Weight: assume conforming files' lines proportional to n_ok/n_meas.
        # This avoids needing per-file line attribution while still tracking
        # density. Equivalent to: each measurable file contributes its line
        # count to denom, and (line_count * (1 if conforming else 0)) to num.
        # We use the per-language ratio as a fair approximation.
        conforming_lines = int(round(n_lines * (n_ok / n_meas)))
        total_measurable_lines += n_lines
        total_conforming_lines += conforming_lines

    if not any_tool_ran or total_measurable_lines == 0:
        return None
    return float(total_conforming_lines / total_measurable_lines)
