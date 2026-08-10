"""a268: Component and project structure conventions — THICK.

The aspect: "Organize components and files by feature/role with consistent
naming (e.g., React component directories and index entry points) and follow
repository directory conventions for discoverability and consistency."

Why this is THICK (not even PARTIALLY_THIN):

This norm is, definitionally, *project-relative*. "Follow repository
directory conventions" presupposes a convention that THIS repository has
already established. We only see one PR diff — a few new/modified files —
not the rest of the repository. Without the repository as context, we
cannot know:

  - Whether this project puts React components in `src/components/`,
    `app/`, `features/`, or a feature-sliced layout.
  - Whether it prefers `Button.tsx` (PascalCase file) or
    `button/index.tsx` (lowercase directory with index entry point).
  - Whether Python modules go under `src/`, the package name, or flat
    at the top level.
  - Whether tests live in `tests/`, `__tests__/`, or co-located with sources.
  - Whether the project uses kebab-case, snake_case, or camelCase paths.

The hint suggested possible weak surrogates:
  - PascalCase React component files
  - Presence of `__init__.py` next to new Python files
  - Java package paths matching `com/org/...`

We considered each and rejected:

  1. PascalCase React: A perfectly valid React project may use lowercase
     directory naming with `index.tsx` entries. Flagging either as
     non-conformant would invert the score on half of real projects.

  2. `__init__.py` presence: Many modern Python projects (PEP 420
     namespace packages, `pyproject.toml`-only layouts, single-file
     scripts) intentionally omit `__init__.py`. Penalizing its absence
     would mis-measure the norm.

  3. Java `com/org/...`: This is correct *when* the Java file declares
     `package com.org.…`, but we'd need to read the package declaration
     AND match it against the path. Even then, projects that follow
     domain-prefix conventions (`net.foo.…`, `io.foo.…`,
     `org.foo.…`) are equally valid; "conformance" is to the
     project's *chosen* prefix, which we don't observe.

A self-consistency proxy (do the N new files in THIS diff agree with
each other on casing?) was also considered and rejected: most PRs add
1–3 files, so the within-diff sample is too small to be a reliable
signal, and any score we produce would be dominated by random
single-file PRs landing at 1.0 trivially.

A genuine measurement would require the *repository tree* as context, so
the metric could induce the project's convention and check the new files
against it. That is well beyond what the per-norm verification contract
provides (diff-only, no repo access, no history).

Marking THICK is the honest answer here, and that is itself the
deliverable: this norm sits squarely on the articulability boundary
because the rule is *intersubjective within a project* but not
extractable from one diff.

What would unblock a non-THICK version of this metric:
  - Access to the full repository tree at HEAD before the PR (would
    allow inducing the file-layout convention from existing files).
  - Access to git history for prior accepted PRs (would allow learning
    the empirical convention from precedent).
  - A `CONTRIBUTING.md` or `docs/architecture.md` parser that extracts
    explicit layout rules.
None of these are available in the metric_implementer sandbox today.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a268"
ASPECT_NAME = "Component and project structure conventions"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "Project structure conventions are, by definition, defined by the "
    "project itself. From a single PR diff we cannot observe the rest of "
    "the repository tree, the prior commit history, or the project's "
    "stated conventions, so we cannot know which layout/casing/index-file "
    "rule THIS project follows. Every candidate surrogate (PascalCase "
    "React files, __init__.py presence, com/org Java paths) inverts on "
    "valid alternative conventions and would mis-measure the norm. The "
    "honest measurement is abstention."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
