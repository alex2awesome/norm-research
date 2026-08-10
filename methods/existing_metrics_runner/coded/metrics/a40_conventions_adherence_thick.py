"""a40: Conventions adherence (local and ecosystem) — THICK.

The aspect: "Align with file/project conventions and relevant ecosystem
standards; deviations are rare, well-reasoned, and improve
clarity/consistency."

Why this is THICK:

The norm has TWO clauses, and both are unmeasurable from a single diff:

1. "LOCAL conventions" — by definition project-specific. Whether the new
   code aligns with what THIS repository already does (naming, file
   layout, error-handling idioms, logging style, import ordering rules,
   helper-vs-inline patterns, preferred libraries from the project's
   internal vocabulary) requires reading the rest of the repository. We
   only see the unified diff. This is the same boundary that forces
   a268 (project/component structure) to THICK; a40 is in fact the
   broader, more abstract sibling of a268 and is subsumed by it on the
   "local" axis.

2. "ECOSYSTEM standards" — looks superficially measurable (PEP 8 via
   ruff, gofmt for Go, prettier for JS/TS, rustfmt for Rust, etc.), but
   the norm specifically permits "rare, well-reasoned deviations that
   improve clarity/consistency." Even a strong linter signal cannot
   distinguish a careless violation from a deliberate, locally-correct
   deviation. Worse, many ecosystems have multiple legitimate dialects
   (Black vs. autopep8, Airbnb vs. StandardJS, tabs vs. spaces in Go
   tests) and which one applies is again project-relative.

Why we explicitly rejected building a partial-credit lint-based version:

- Linter score collapses into a "no new violations" detector, which is
  already covered narrowly by a181 (warnings/lints as errors). a40 is
  about CONFORMANCE TO CONVENTION, not absence of lint hits.
- The norm's "deviations are rare, well-reasoned, and improve clarity"
  clause is reader-side judgment — exactly the kind of taste signal we
  flag THICK in a16 and a78.
- Self-consistency within the diff (do new files match each other?) is
  unreliable: most PRs touch 1–3 files, sample size too small.
- The PR text where an author would JUSTIFY a deviation ("well-reasoned")
  is not in the diff input, same blocker that pushes a78 to THICK.

What would unblock a non-THICK version:
  - Repository tree at HEAD before the PR, so we could induce the
    project's local convention from existing files and check the new
    files against it.
  - PR body/commit messages so we could detect explicit rationale for
    any deviation.
  - A `CONTRIBUTING.md` / `.editorconfig` / linter-config parser that
    extracts the project's declared convention and verifies the diff
    against it (this is the most tractable single unblocker but still
    requires repo access).

None of these are available in the metric_implementer sandbox today, and
the hint correctly notes that a40 is subsumed by a268 on the local axis.
The honest answer — and the informative one for the articulability
boundary — is THICK.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a40"
ASPECT_NAME = "Conventions adherence (local and ecosystem)"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "The norm has a 'local' clause that is project-relative (same boundary "
    "as a268 — we cannot observe the repository tree from a single diff to "
    "induce the project's convention) and an 'ecosystem' clause that "
    "explicitly permits 'rare, well-reasoned deviations,' which requires "
    "reader-side judgment about intent that the diff cannot provide. A "
    "linter-based proxy would collapse into a 'no new violations' signal "
    "that is already covered narrowly by a181 and does not measure "
    "conformance-to-convention. Subsumed by a268 on the local axis; THICK "
    "is the honest measurement."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
