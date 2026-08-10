"""a102: Tests to drive design and refactoring — THICK.

Aspect (from aspects.json):
    "Use tests (including TDD and doubles) to shape design, document behavior,
    and enable confident refactoring over time."

Why THICK
---------
The norm is fundamentally TEMPORAL. The headline practice it names is
Test-Driven Development: write the test first, watch it fail, then write
just enough production code to make it pass, then refactor under the
green bar. The defining question — "did the tests come before the
production code?" — cannot be answered from a single diff snapshot. We
would need commit-by-commit history showing test-first commits landing
ahead of implementation commits, or at minimum a temporal ordering of
hunks within the change. A unified diff collapses all of that into one
side-by-side view of "before" and "after" — the ordering information is
gone by construction.

The remaining sub-clauses are similarly out of reach from the diff alone:

1. "Use tests to shape design." This is an architectural causality claim:
   the production-code shape (interfaces, parameter lists, seams) was
   *chosen because* the test made that shape necessary. Distinguishing
   "test-shaped code" from "code-then-tested code" needs either commit
   history or an interview with the author. The same final code can be
   produced by either workflow.

2. "Use doubles (mocks/stubs/fakes/spies) appropriately." We can detect
   *presence* of mocking-library calls (`unittest.mock`, `pytest-mock`,
   `sinon`, `mockito`, `gomock`), but the norm asks about *appropriate*
   use — were the doubles placed at meaningful seams, or did they paper
   over poorly-designed coupling? That's a design judgment the diff
   does not contain. Mere mock count is just as easily a code smell
   (over-mocking, "mock everything") as a virtue.

3. "Document behavior." Tests document behavior in proportion to how
   readable and specification-like they are (BDD/Gherkin/given-when-then
   structure, descriptive names). a131 already covers behavior-style
   test naming as a separate aspect; pulling it into a102 would
   double-count.

4. "Enable confident refactoring over time." This is a property of the
   *test suite as a whole over its lifetime* — does running the suite
   give the team confidence to restructure? A single PR cannot exhibit
   or fail this property; only a longitudinal record of refactoring PRs
   landing without regressions can.

Weak proxies considered and rejected
------------------------------------
- "Test file LOC >= source file LOC" as a TDD signal. Tempting, but in
  practice strongly trends with how unit-testy the language community
  is (Go tests run long; Python pytest tests can be terse parameterized
  one-liners) and would mostly measure language convention. It would
  also score high for after-the-fact characterization-test sweeps,
  which are the OPPOSITE of TDD.

- "Mock-library import in test file." Measures use of doubles, but not
  *appropriateness*. Many projects ban mocks in unit tests by policy
  (London- vs Detroit-school TDD) — both schools are TDD, the metric
  would flip sign across them.

- "Test added in same hunk as source change." A diff can show both
  files modified; it cannot show whether the test addition was
  committed before or after the source addition. Same diff, two very
  different workflows.

- "Empty/skipped tests count." Easy to compute (`@pytest.mark.skip`,
  `xit(`, `@Ignore`), but skipped tests are noise about test hygiene,
  not about whether tests drove design.

Per the GUIDE: bias for *honest THICK* over *desperate Tier-1 regex*.

What would unblock this metric
------------------------------
1. **Commit history**, not just the squashed diff. A test-first ordering
   detector would look for: (a) test files appearing in earlier commits
   than the source files they exercise, (b) test commits whose tests
   *fail* against the parent commit's source (red), followed by source
   commits where the tests *pass* (green). GitHub's API exposes the
   per-commit patch list of a PR; metric_implementer's fixture format
   would need to grow a `commits: [...]` field.

2. **CI test-result history per commit.** Even without re-running the
   tests, a red->green commit-pair signature in the CI status stream
   is the TDD watermark.

3. **Coverage delta across the PR.** Coverage that grows in lockstep
   with new source lines — never lagging — is the strongest
   non-temporal proxy: it implies tests landed alongside (or before)
   the code, not as a post-hoc retrofit. Requires a coverage harness.

4. **Author self-report / PR description NLP.** PR templates that
   mention "TDD", "red-green-refactor", or link to a failing test
   issue are a weak surface signal; could be a Tier-1 proxy *for the
   description channel*, but the diff body is silent on it.

Until at least (1) is available, a102 cannot be deterministically
measured. THICK is the informative answer.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a102"
ASPECT_NAME = "Tests to drive design and refactoring"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "TDD is a temporal claim about commit ordering (tests-first, "
    "red->green->refactor) that a single diff snapshot has erased by "
    "construction. The remaining sub-clauses — 'tests shape design', "
    "'doubles used appropriately', 'enables refactoring over time' — are "
    "architectural / longitudinal properties not observable in one diff. "
    "Local proxies (test LOC ratio, mock imports, same-hunk test+source) "
    "either measure language convention or flip sign across schools of "
    "TDD. Unblockers: per-commit patch series, CI red->green sequence per "
    "commit, coverage-delta-vs-source-delta lockstep."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
