"""a12: Appropriate use of established design patterns — THICK.

The norm asks more than "is a Gang-of-Four pattern present?" It demands four
intertwined judgments:

  1. Does the diff face a RECURRING PROBLEM that an established pattern fits?
  2. Is the chosen pattern of APPROPRIATE COMPLEXITY for that problem?
     (Overuse = bad; under-use = bad.)
  3. Is the pattern applied CONSISTENTLY with how the rest of the codebase
     uses similar patterns?
  4. Is the RATIONALE documented?

Items (1)-(3) are taste-laden meta-properties about fit, not about presence;
item (4) is a documentation-style judgment that overlaps with comment-quality
norms already covered elsewhere (a15).

Why structural detection is not enough
--------------------------------------

The hint correctly notes that pattern MARKERS are tree-sitter-detectable:

  - Singleton:   private constructor + static `getInstance()` returning a
                 cached instance.
  - Observer:    `subscribe`/`addListener`/`notify`/`emit` method pairs.
  - Factory:     functions named `create*`, classes `*Factory` / `*Builder`,
                 `new` calls hidden behind a static method.
  - Strategy:    an interface plus several implementations selected by name.
  - Adapter:     a wrapping class delegating most methods to a held instance.
  - Decorator:   a wrapping class adding behavior around delegated calls.

We can count these markers and emit a presence number. The problem is that
**presence is orthogonal to appropriateness**:

  - A diff that adds a Singleton to hold a stateless math constant SATISFIES
    the structural detector and VIOLATES the norm (over-engineering).
  - A diff that inlines a one-shot function VIOLATES the structural detector
    (no pattern) and SATISFIES the norm (appropriately simple).
  - A repository that uses Builder everywhere will score high on "builder
    present" yet may be heavily over-applying the pattern.

So pattern-count -> score correlates more with codebase style than with norm
compliance. The codegen_claude diagnostic warned that such surrogates collapse
into text-length signal once weak.

Why this is distinct from a99 (which IS implemented as PARTIALLY_THIN)
---------------------------------------------------------------------

a99 ("Builder pattern for complex construction") is narrower: it asks only
about the Builder pattern, and pairs builder-class detection with a CANDIDATE
gate (wide-constructor / wide-init count). The candidate gate is what makes
a99 partially measurable — we observe BOTH the problem-shape and the
solution-shape in the same diff and reason about their ratio.

For a12 we would need a candidate gate per pattern family (singleton needs
"global mutable state would otherwise be plumbed everywhere"; observer needs
"multiple consumers of an event stream"; factory needs "construction varies
by type"). None of these candidate conditions is detectable from a diff.

Surrogates considered and rejected
----------------------------------

  - PATTERN-MARKER COUNT (sum of singletons + observers + factories +
    strategies + adapters in the added code). Reflects style, not fit.
    A high count is as likely to indicate over-engineering as appropriateness.

  - PATTERN-DENSITY (markers per added LOC). Same issue, with extra noise on
    small diffs.

  - DOCSTRING-NEAR-PATTERN-MARKER (does each detected pattern have a
    rationale-style comment). Conflates this norm with a15 (comments
    explain *why*); also extremely sparse on the fixture set.

  - LEXICAL HITS for class names like `*Factory`, `*Builder`, `*Strategy`,
    `*Adapter`, `*Decorator`, `*Singleton`. Hits text-length proxy hard;
    contestable (legitimate uses of these suffixes are constant per file
    over time), and on the 50-PR fixture set the marker-class names appear
    in fewer than 10% of diffs.

Per the GUIDE: a regex/marker hit producing ~0.5 for everyone is worse than
a True/None pair that says "I cannot measure this." Marking THICK records
the norm's location precisely on the articulability boundary.

UNBLOCKERS — what would convert this to a deterministic measurement
-------------------------------------------------------------------

  - A repo-level pattern-fit oracle that snapshots WHERE patterns are used
    across the whole codebase, so the diff's pattern choice can be judged
    against existing conventions (item 3).
  - A "candidate problem" detector per pattern family (item 1) — currently
    only Builder has a tractable candidate (wide constructor / wide init).
  - A rationale-presence detector that distinguishes "documents why this
    pattern" from "documents what this code does" (item 4) — currently
    handled (imperfectly) by a15.
  - Item (2) — "appropriate complexity" — requires a counterfactual
    comparison ("would the code be simpler without the pattern?") that no
    static analyzer performs.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a12"
ASPECT_NAME = "Appropriate use of established design patterns"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "Established design-pattern MARKERS (Singleton, Observer, Factory, "
    "Strategy, Adapter, Decorator) are tree-sitter-detectable in principle, "
    "but the norm asks about APPROPRIATENESS — whether the pattern fits the "
    "problem, is of suitable complexity, is applied consistently with the "
    "rest of the codebase, and has documented rationale. Presence is "
    "orthogonal to appropriateness: a Singleton holding a stateless constant "
    "satisfies the detector and violates the norm (over-engineering), while "
    "an inlined one-shot helper violates the detector and satisfies the norm. "
    "Judging fit requires a 'candidate problem' detector per pattern family "
    "and a repo-level view of existing pattern usage, neither available from "
    "a single diff. a99 (Builder) is implementable as PARTIALLY_THIN only "
    "because wide-constructor count is a tractable candidate gate; no "
    "analogous gate exists for the other pattern families."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
