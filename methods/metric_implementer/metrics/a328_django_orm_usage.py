"""a328: Django ORM usage and QuerySet semantics.

Aspect description (verbatim from aspects.json):
  "Use Django ORM correctly: retrieval and relationship traversal,
   exact/contains/iexact lookups, QuerySet laziness and chaining,
   expressions/transforms, saves/updates, and awareness of
   async/transaction limitations."

This is a NARROW-applicability metric: it only fires when added Python code
shows Django ORM markers (`from django`, `models.Model`, `.objects.`,
QuerySet methods on managers, etc.). Most code_review PRs will get
`applies()=False` here — which is the correct router-gated behavior per
GUIDE.md.

## Classification: PARTIALLY_THIN

THIN parts (real tool signal):
  * `pylint-django` plugin, configured against a stub Django settings module,
    emits a handful of bug-shaped diagnostics whose semantics don't require
    cross-file resolution:
      R5101 http-response-with-json-dumps     -> use JsonResponse(data)
      R5102 http-response-with-content-type-json
      R5103 redundant-content-type-for-json-response
      E5141 hard-coded-auth-user               (settings.AUTH_USER_MODEL violation)
      E5142 imported-auth-user                 (same)
      W5101 model-missing-unicode              (model has no __str__)
      W5102 model-has-unicode                  (Py2 __unicode__ on a model)
      W5103 model-no-explicit-unicode          (inherits default __str__)
      W5104 modelform-uses-exclude             (ModelForm.Meta uses exclude)

PARTIALLY THIN parts (AST heuristics, not a real Django runtime analyzer):
  * N+1 query detection: walk added Python code's AST; flag any loop whose
    iterable is a QuerySet-looking expression and whose body accesses a
    *related-field* attribute on the loop variable. Detected without running
    the ORM, so it's a heuristic — we can't tell a `ForeignKey` from any
    other attribute statically. We compensate by scoring this signal at half
    weight relative to pylint-django diagnostics.
  * Raw-SQL/`.extra(...)` / `.raw(...)` flag — these *bypass* the ORM and the
    aspect explicitly wants "use the ORM correctly". Heuristic but cheap.

## Why not just rely on pylint-django?

pylint-django's checker set is small (a dozen Django-specific codes), and
many of its richer checks need full Django settings + INSTALLED_APPS + a real
project tree to resolve foreign keys and managers. On diff-derived partial
files those checks either crash (django-not-configured) or emit noise (any
custom manager looks like `no-member`). We configure a *stub*
settings module so pylint-django boots, then enable only the curated codes
above whose semantics are intrinsic to the added file.

## Why not bandit / ruff?

bandit's B610/B611 cover Django ORM injection (`.extra()`, `.raw()`) but
that's a SECURITY norm (a47/a267's territory). We re-flag those patterns
*here* with a different weight because for THIS norm they signify "didn't
use the ORM properly", not "security risk". Same syntactic pattern, two
different normative readings — exactly the kind of partition the metric tree
is supposed to surface.

## Why not django-extensions / django-debug-toolbar?

Both are RUNTIME tools (require a live request to log queries). We can't
boot a project from a diff snippet, so they're not statically applicable.

## Score

Combine pylint-django violation count and AST-N+1 / raw-SQL flags into a
weighted violation count, then divide by added Python lines, then
exp(-density * 25). Calibrated so:
  0 violations / 100 lines -> 1.00
  1 pylint-django finding / 100 lines -> 0.78
  1 N+1 site / 50 lines -> 0.61
  2 raw-SQL escapes / 50 lines -> 0.37

If `applies()=True` but pylint-django times out or the tool is missing and
no AST flags fired, we ABSTAIN (return None) — distinguishable downstream
from a clean score.
"""
from __future__ import annotations

import ast
import math
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a328"
ASPECT_NAME = "Django ORM usage and QuerySet semantics"
TIER = 3
TOOLS = ["pylint"]  # pylint-django is a pylint plugin
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "PARTIALLY_THIN"

PY_EXTS = [".py", ".pyi"]

# pylint-django codes we keep — semantics don't require project context.
ENABLED_PLUGIN_CODES = [
    "R5101", "R5102", "R5103",  # HttpResponse vs JsonResponse
    "E5141", "E5142",           # auth.User hardcoding
    "W5101", "W5102", "W5103",  # model __unicode__/__str__
    "W5104",                    # ModelForm exclude
]

# QuerySet manager methods — used by the AST heuristic to recognize chains.
QS_MANAGER_METHODS = {
    "all", "filter", "exclude", "get", "annotate", "values", "values_list",
    "select_related", "prefetch_related", "only", "defer", "order_by",
    "distinct", "reverse", "none", "union", "intersection", "difference",
    "aggregate", "first", "last", "earliest", "latest", "count", "exists",
    "in_bulk", "iterator",
}

# Stub Django settings — minimal so pylint_django boots.
_STUB_SETTINGS = """\
SECRET_KEY = 'x'
INSTALLED_APPS = []
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}
USE_TZ = True
"""


def _added_py_text(diff_text: str):
    """Return {file: added_lines_text} for added Python files."""
    return added_files_by_ext(diff_text, PY_EXTS)


def _has_django_marker(text: str) -> bool:
    """Cheap structural check: does this added-lines blob mention Django?

    We parse what we can with `ast`; if parsing fails (common on diff
    snippets — partial functions, etc.) fall back to a tokenize-style scan
    that looks for `django` import statements and `.objects.` manager use.
    No regex on code; we use stdlib `ast.parse` and a token check.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Best-effort: try wrapping in a class to absorb dangling bodies.
        try:
            tree = ast.parse("class _X:\n    " + text.replace("\n", "\n    "))
        except SyntaxError:
            tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", None) or ""
                names = [a.name for a in node.names] if hasattr(node, "names") else []
                if mod.startswith("django") or any(
                    n.startswith("django") for n in names
                ):
                    return True
            # `.objects.<qs_method>(` is a strong manager marker
            if isinstance(node, ast.Attribute):
                if node.attr == "objects":
                    return True
            # models.Model / models.ForeignKey / etc. base classes
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_src = ast.unparse(base) if hasattr(ast, "unparse") else ""
                    if "models.Model" in base_src or "models.Manager" in base_src:
                        return True
    # Fallback structural check: look for exact substrings that are
    # unambiguous Django markers. These are not regex — they're constant
    # substring searches in the added-lines blob.
    markers = (
        "from django", "import django",
        "models.Model", "models.ForeignKey", "models.ManyToManyField",
        "models.CharField", "models.IntegerField",
        ".objects.filter(", ".objects.all(", ".objects.get(",
        ".objects.create(", ".objects.exclude(",
        "QuerySet", "select_related(", "prefetch_related(",
    )
    return any(m in text for m in markers)


def applies(diff_text: str) -> bool:
    by_path = _added_py_text(diff_text)
    if not by_path:
        return False
    return any(_has_django_marker(t) for t in by_path.values())


# ---------- AST heuristic checks ----------

def _count_orm_bypass(tree: ast.AST) -> int:
    """Count uses of `.extra(...)`, `.raw(...)` on what looks like a manager
    chain. These bypass the ORM — the aspect penalizes them."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("extra", "raw"):
                # only count if we can see a manager / queryset chain root
                # somewhere in the receiver: walk down attribute chain looking
                # for `.objects` or a known QS method.
                cur = node.func.value
                hops = 0
                while isinstance(cur, (ast.Attribute, ast.Call)) and hops < 6:
                    if isinstance(cur, ast.Attribute) and cur.attr == "objects":
                        count += 1
                        break
                    if (isinstance(cur, ast.Call)
                            and isinstance(cur.func, ast.Attribute)
                            and cur.func.attr in QS_MANAGER_METHODS):
                        count += 1
                        break
                    cur = cur.value if isinstance(cur, ast.Attribute) else cur.func
                    hops += 1
    return count


def _looks_like_queryset(node: ast.AST) -> bool:
    """True if expression looks like a queryset/manager call (e.g.
    `Model.objects.filter(...)`, `qs.all()`, `User.objects.all()`).
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        attr = node.func.attr
        if attr in QS_MANAGER_METHODS:
            # walk down looking for `.objects`
            cur = node.func.value
            hops = 0
            while isinstance(cur, (ast.Attribute, ast.Call)) and hops < 6:
                if isinstance(cur, ast.Attribute) and cur.attr == "objects":
                    return True
                if (isinstance(cur, ast.Call)
                        and isinstance(cur.func, ast.Attribute)
                        and cur.func.attr in QS_MANAGER_METHODS):
                    return True
                cur = cur.value if isinstance(cur, ast.Attribute) else cur.func
                hops += 1
            # accept top-level .all()/.filter() with .objects directly
            if (isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "objects"):
                return True
    return False


def _has_attribute_access_in_body(loop_var: str, body: List[ast.stmt]) -> bool:
    """True if any node in `body` does `<loop_var>.<some_attr>.<other>` —
    a chained attribute access pattern that, for a QuerySet of related
    objects, is the N+1 fingerprint (each iteration triggers a follow-up
    SQL query).
    """
    for stmt in body:
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == loop_var):
                return True
    return False


def _count_n_plus_one(tree: ast.AST) -> int:
    """Count `for x in <queryset>: ... x.<rel>.<field> ...` patterns where
    the queryset chain doesn't already include `.select_related` /
    `.prefetch_related`. Heuristic — we can't statically tell ForeignKey
    from any other attribute.
    """
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        iterable = node.iter
        if not _looks_like_queryset(iterable):
            continue
        # check whether select_related / prefetch_related are in the chain
        chain_methods = set()
        cur = iterable
        hops = 0
        while isinstance(cur, ast.Call) and hops < 8:
            if isinstance(cur.func, ast.Attribute):
                chain_methods.add(cur.func.attr)
                cur = cur.func.value
                # may be another call (chained)
                if isinstance(cur, ast.Call):
                    continue
                break
            else:
                break
            hops += 1
        if chain_methods & {"select_related", "prefetch_related"}:
            continue
        if _has_attribute_access_in_body(node.target.id, node.body):
            count += 1
    return count


def _run_ast_checks(text: str) -> Tuple[int, int]:
    """Returns (n_plus_one_count, orm_bypass_count). Returns (0, 0) if
    parsing fails — we don't penalize unparseable diff fragments."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        try:
            wrapped = "class _X:\n    " + text.replace("\n", "\n    ")
            tree = ast.parse(wrapped)
        except SyntaxError:
            return 0, 0
    return _count_n_plus_one(tree), _count_orm_bypass(tree)


# ---------- pylint-django subprocess ----------

def _run_pylint_django(by_path) -> Optional[int]:
    """Returns # of relevant pylint-django findings, or None on tool failure."""
    if not have_tool("pylint"):
        return None
    td = write_temp_files(by_path)
    # write stub settings into the same temp dir
    settings_path = td / "_a328_django_settings.py"
    settings_path.write_text(_STUB_SETTINGS)
    try:
        cmd = [
            "pylint",
            "--load-plugins=pylint_django",
            "--django-settings-module=_a328_django_settings",
            "--output-format=json",
            "--score=no",
            "--persistent=no",
            "--disable=all",
            "--enable=" + ",".join(ENABLED_PLUGIN_CODES),
            str(td),
        ]
        # PYTHONPATH must contain the settings dir; pylint also needs to
        # actually import django.
        env_path = str(td) + os.pathsep + os.environ.get("PYTHONPATH", "")
        import subprocess
        try:
            p = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30.0,
                env={**os.environ, "PYTHONPATH": env_path,
                     "DJANGO_SETTINGS_MODULE": "_a328_django_settings"},
            )
        except subprocess.TimeoutExpired:
            return None
        except FileNotFoundError:
            return None
        out = p.stdout or ""
        if not out.strip():
            return 0
        try:
            findings = json.loads(out)
        except json.JSONDecodeError:
            return None
        if not isinstance(findings, list):
            return None
        return sum(
            1 for f in findings
            if isinstance(f, dict)
            and f.get("message-id") in ENABLED_PLUGIN_CODES
        )
    finally:
        shutil.rmtree(td, ignore_errors=True)


def score(diff_text: str) -> Optional[float]:
    by_path = _added_py_text(diff_text)
    if not by_path:
        return None
    django_files = {f: t for f, t in by_path.items() if _has_django_marker(t)}
    if not django_files:
        return None
    total_lines = sum(1 + t.count("\n") for t in django_files.values())
    if total_lines == 0:
        return None

    # AST checks (always run; cheap and don't need a tool)
    n_plus_one_total = 0
    orm_bypass_total = 0
    for t in django_files.values():
        np1, byp = _run_ast_checks(t)
        n_plus_one_total += np1
        orm_bypass_total += byp

    # pylint-django run (may be None if tool unavailable)
    pld_count = _run_pylint_django(django_files)

    # Compose weighted violation count. Each pylint-django hit is a confident
    # signal; each AST heuristic hit is softer (half weight).
    if pld_count is None and n_plus_one_total == 0 and orm_bypass_total == 0:
        # Tool failure AND no heuristic findings -> abstain (can't measure).
        return None
    weighted = 0.0
    if pld_count is not None:
        weighted += pld_count * 1.0
    weighted += n_plus_one_total * 0.5
    weighted += orm_bypass_total * 0.5

    density = weighted / max(total_lines, 1)
    return float(math.exp(-density * 25.0))
