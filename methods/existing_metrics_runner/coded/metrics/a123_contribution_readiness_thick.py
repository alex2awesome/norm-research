"""a123: Contribution readiness and submission norms — THICK.

The aspect: "Contributions meet baseline entry criteria (tests included, coding
practices followed, debt reduced where possible) and are submitted in a state
suitable for serious review/integration."

Why this is THICK:

"Contribution readiness" is a *project-relative*, *meta-procedural* norm: it
asks whether a PR clears the bar THIS project sets for entry into review. The
bar itself is encoded in artifacts we do NOT receive as part of the diff:

  1. **CONTRIBUTING.md / PULL_REQUEST_TEMPLATE.md** — projects state their
     submission rules here (CLA signed? issue linked? rebased on main?
     changelog updated? screenshots for UI changes?). The diff input does not
     carry the project's contributing guide, so we cannot verify conformance
     to it.

  2. **CI configuration and required checks** — "tests included" is partially
     observable (we can see whether the diff adds a test file), but
     "tests included AT THE LEVEL THIS PROJECT REQUIRES" is not. Some
     projects require unit + integration + e2e + snapshot tests for every
     change; others accept a single regression test. We have no project
     baseline to compare against.

  3. **PR metadata** — "submitted in a state suitable for serious review"
     typically means: linked to an issue, has a description, passes CI, has
     no merge conflicts, is rebased on the target branch, has reviewers
     assigned, has a CLA signature recorded, has labels applied. None of
     these fields are present in a unified-diff artifact; they live on the
     PR object itself (GitHub API fields like `body`, `linked_issues`,
     `mergeable_state`, `check_runs`, `requested_reviewers`).

  4. **"Coding practices followed"** is a tautological re-reference to all
     the project's other code-quality norms (a0..a394). Measuring it here
     would double-count whatever the per-aspect ladder already covers.

  5. **"Debt reduced where possible"** is reader-side judgment about whether
     the diff *could have* simplified more than it did. There is no
     ground-truth refactor to compare against.

Considered and rejected proxies:

  - **"Adds a *_test.* / test_*.py file"** as a "tests included" proxy: this
     proxy fires only on the subset of PRs whose entire purpose is test-
     adjacent; many legitimate bug-fix and refactor PRs are accepted without
     a new test file (e.g., because the existing test suite already covered
     the regression, or the change is a docs/typings adjustment). The proxy
     would systematically score docs-only and pure-refactor PRs as
     non-conformant, which contradicts the norm's intent. It also collapses
     to "this PR modifies a tests/ directory," which is a feature already
     captured by metadata, not a norm measurement.

  - **Diff size as a "ready for serious review" proxy**: contestable both
     directions (a tiny diff can be ill-prepared; a huge diff can be a
     well-staged release branch). The codegen_claude diagnostic showed
     length-correlated heuristics collapse into a text-length signal with
     AUC ~0.5 on this task.

  - **Presence of a changelog/news fragment file**: project-specific. Some
     projects (Twisted, Pip, Sphinx) require it; most do not. Without
     knowing which we are looking at, this either over- or under-fires.

This metric is THICK at the same level as a78 (PR/commit communication, no
substrate in diff) and a268 (project structure, project-relative). Marking it
THICK is the honest measurement and is itself the deliverable: it records that
the "submission norms" boundary is project-defined and not extractable from a
single diff.

What would unblock a non-THICK version:
  - PR metadata alongside the diff (title, body, linked_issues,
    mergeable_state, check_runs, requested_reviewers, labels).
  - The project's CONTRIBUTING.md / PULL_REQUEST_TEMPLATE.md.
  - The project's CI config (which checks are required).
  - Git history for prior accepted PRs in the same project, to induce the
    empirical readiness bar.
None of these are available in the metric_implementer sandbox today.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a123"
ASPECT_NAME = "Contribution readiness and submission norms"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "Contribution readiness is project-relative and procedural: the bar lives "
    "in CONTRIBUTING.md, PULL_REQUEST_TEMPLATE.md, CI required-checks, and PR "
    "metadata (linked issues, CLA, mergeable_state, requested reviewers), none "
    "of which are present in a unified-diff input. 'Tests included' is "
    "partially observable but 'at the level THIS project requires' is not. "
    "'Coding practices followed' tautologically re-references other norms and "
    "would double-count. 'Debt reduced where possible' is reader-side "
    "counterfactual judgment. Every candidate surrogate (presence of a test "
    "file, diff size, changelog fragment) inverts on valid project conventions "
    "and would mis-measure the norm."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
