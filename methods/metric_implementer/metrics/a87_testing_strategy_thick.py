"""a87: Testing strategy, rigor, and methodology — THICK.

Aspect description (verbatim from aspects.json):

    "A deliberate testing approach (requirements-driven, TDD/BDD/ATDD as
    appropriate) with readable tests, adequate coverage/mix, disciplined
    techniques, and effective test artifacts/fixtures."

Why THICK
---------
This is an *umbrella* norm: it asks whether the PR's testing reflects a
deliberate, disciplined STRATEGY — TDD/BDD/ATDD-appropriate, readable,
adequately covered, well-mixed, with good fixtures. Every concrete,
diff-observable shadow of that umbrella is already captured by an
adjacent metric:

  a131 (ATDD / user-visible behavior) — file-level BDD style + user-boundary
       driver imports (selenium, playwright, supertest, cucumber, .feature
       files). Detects the "B/ATDD as appropriate" shadow.

  a89  (behavior-focused interaction testing) — statement-level assertion
       style: interaction (mock.verify, toHaveBeenCalled) vs. state
       (assertEqual, toBe). Detects the "behavior vs. white-box" shadow.

  a104 (automated tests: presence and design quality) — tanh-blended test
       file presence, test-line ratio, and test-function count. Detects the
       "adequate coverage" shadow.

  a309 (tests included with changes) — per-source-file ↔ per-test-file
       name correspondence. Detects the "tests track the change" shadow.

  a38  (test pyramid) — unit/integration/e2e mix. Detects the "adequate
       mix" shadow.

  a39  (test isolation) — per-test independence and fixture hygiene.
       Detects the "effective fixtures" shadow.

  a9   (design for testability) — production-code structure that enables
       testing. Detects the precondition shadow.

  a99  (builder pattern) — adjacent fixture-construction idiom.

What remains exclusive to a87 after subtracting those shadows is the
*holistic judgment* component: "is this PR's overall testing approach
DELIBERATE? Are the techniques DISCIPLINED? Are the fixtures EFFECTIVE for
the requirement being tested?" Those modifiers — deliberate, disciplined,
effective — are reviewer-mind properties, not diff-observable structure.
Two PRs with identical a131/a89/a104/a309 scores can have wildly
different a87 scores depending on whether a senior engineer judges the
mix and the fixtures to be principled vs. cargo-culted. We cannot read
"deliberate" off a diff without simulating a reviewer.

Could we proxy with a weighted aggregate of a131+a89+a104+a309+a38+a39?
Yes — but that aggregate IS what the downstream RF/LR model already
computes from those columns. Defining a87 as a hand-weighted sum would
double-count signal and add no information; defining it as an unweighted
mean would just dilute it. Neither is a measurement of the norm; both
are a re-derivation the downstream learner does for free.

The principled move is to record this norm as THICK and let the
downstream model combine the adjacent shadow metrics itself. THICK here
is not a failure — it is the precise statement that "deliberate strategy"
is the articulability boundary for this norm cluster, and the adjacent
metrics already cover everything below that boundary.

Tier 0, no tools, applies()=False, score()=None — per GUIDE.md §"When to
mark a metric THICK".
"""
from __future__ import annotations

from typing import Optional

ASPECT_ID = "a87"
ASPECT_NAME = "Testing strategy, rigor, and methodology"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "Umbrella norm subsumed by adjacent metrics: a131 (ATDD/user-visible "
    "behavior), a89 (behavior-focused interaction testing), a104 (test "
    "presence/design), a309 (test-source correspondence), a38 (test "
    "pyramid mix), a39 (test isolation/fixture hygiene), a9 (design for "
    "testability). The residual 'deliberate, disciplined, effective' "
    "strategy judgment is a reviewer-mind property not observable in the "
    "diff; a hand-weighted aggregate of the shadow metrics would double-"
    "count signal the downstream model already combines from those columns."
)

SUBSUMED_BY = ["a131", "a89", "a104", "a309", "a38", "a39", "a9", "a99"]


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
