"""a56: Developer tooling usability and formatter UX — THICK.

Aspect description (verbatim):
  "Provide fast, clear, developer-friendly tooling (CLI/IDE) with in-editor
  fixes and opinionated yet composable formatting that respects team needs."

The norm asks about USABILITY of dev tooling and the UX of formatters — a
property of how a tool FEELS to use, not a property that the diff itself
exposes. To pass the norm a change would need to be evaluated on five
intertwined human-experience axes, none of which is observable from the
patch text:

  1. SPEED — "fast" is a benchmark question (wall-clock on the target
     repo). The diff says nothing about whether a CLI flag change shaved
     30 ms or 30 s. Even when a PR touches a tool's hot path we cannot
     measure runtime delta without running benchmarks on a representative
     workload that doesn't ship with the diff.
  2. CLARITY of CLI/IDE output — "clear" error messages and diagnostics
     are a writing/UX judgment. The same diagnostic string is "clear" to
     an expert and obscure to a newcomer; clarity is rated by users, not
     parsed from code.
  3. IN-EDITOR FIX QUALITY — quickfix providers either deliver good
     suggestions or not. Static analysis can confirm a `CodeAction`
     handler exists but cannot judge whether its suggested edits feel
     right in context. This is what LSP user-research studies measure
     manually.
  4. OPINIONATED-YET-COMPOSABLE FORMATTING — a design-tension judgment
     (Prettier-style "no knobs" vs Black's --skip-string-normalization
     compromise vs gofmt's hard line). The norm asks whether the diff
     strikes the *right* tension, which is a taste question debated in
     formatter RFCs (see Prettier issues #40, #1338) for years.
  5. RESPECTING TEAM NEEDS — explicitly relational: a config knob is
     "respectful" if the team that asked for it gets value, "noise" if
     no team uses it. Requires a population of users we cannot observe
     from a single diff.

Why this is distinct from the adjacent aspects flagged in the hint
------------------------------------------------------------------

  a72  ("Formatting layout") — measures whether ARTIFACT CODE conforms
       to a formatter's rules. Implemented PARTIALLY_THIN via
       `ruff format --check` / `black --check`. That is the FORMATTEE
       side; a56 is the FORMATTER-AUTHOR side (does the tool produce a
       good UX?). Different population, different measurement.

  a95  ("Static analysis setup") — measures whether the repo configures
       a linter (ruff/eslint/etc. config file present and well-formed).
       a56 asks whether the linter being configured is itself
       developer-friendly. Configuration presence is detectable;
       configured-tool-UX is not.

  a42  ("Static analysis enforcement") — measures whether the configured
       checks are RUN AND BLOCK on violations (CI hook + non-zero exit
       on findings). a56 sits one level higher: is the tool whose
       enforcement we're checking pleasant to use?

So a56 is the meta-aspect about the QUALITY of the dev tools that a72,
a95, and a42 all presuppose exist. The chain is:

   a56 (tool is good)  -->  a95 (tool is configured)
                       -->  a42 (tool is enforced)
                       -->  a72 (artifact passes tool)

We can verify the last three by running the tool. We cannot verify the
first by examining a diff, because "good" is a user judgment.

Surrogates considered and rejected
----------------------------------

  - "DIFF MENTIONS A FORMATTER OR LINTER" (regex for prettier/black/ruff/
    eslint/gofmt/rustfmt). Hits text-length proxy; co-occurs with a72,
    a95, a42 already; would be a duplicate signal under a different name.
    Worse, presence in the diff is orthogonal to UX: a diff that
    REMOVES a formatter (because the team finds it intolerable) violates
    a56 by improving UX, and a diff that ADDS a formatter (over team
    objection) satisfies the surrogate while violating the norm.

  - "CHANGELOG / README ENTRY DESCRIBES A SPEED OR CLARITY WIN" (regex
    for "faster", "clearer", "improved diagnostic"). Self-reported
    marketing language, not measurement; documented in the
    codegen_claude diagnostic as exactly the kind of regex that collapses
    to text length.

  - "BENCHMARK FILE PRESENT" (look for `bench/`, `benchmarks/`,
    `criterion`, `pytest-benchmark`). Presence of benchmarks would only
    show that the PR can be measured for speed, not that it IS fast.
    Also extremely sparse on the PR fixture set (most PRs do not touch
    bench/).

  - "ADDED CODEACTION / QUICKFIX HANDLER COUNT" (tree-sitter for
    LSP-shaped class/function names). Counts plumbing, not quality.
    A handler that returns wrong suggestions counts the same as one
    that returns useful ones. Also language-specific in a way that
    doesn't survive the cross-repo PR mix.

  - "FORMATTER CONFIG KNOBS ADDED" (lines added to .prettierrc /
    ruff.toml / .editorconfig). Could be evidence of "respecting team
    needs" OR of accreting customizations that violate the
    opinionated-yet-composable balance. Direction unknown.

Each rejected surrogate is at best a 0.55-for-everyone signal and at
worst negatively correlated with the norm. Per the GUIDE: an honest
True/None pair beats a desperate Tier-1 regex.

UNBLOCKERS — what would convert this to a deterministic measurement
-------------------------------------------------------------------

  - A REPRESENTATIVE BENCHMARK harness for the changed tool, run before
    and after the PR, reporting wall-clock delta on a fixed workload.
    Solves axis (1).
  - A USER STUDY pipeline (or a strong LLM-as-novice proxy) rating
    diagnostic clarity on a held-out corpus of error messages. Solves
    axis (2).
  - LSP CodeAction REGRESSION FIXTURES: a corpus of (broken-code,
    expected-fix) pairs to score quickfix providers end-to-end.
    Solves axis (3).
  - FORMATTER OPINIONATEDNESS METRIC: an axiomatic measure (count of
    user-tunable knobs vs hard-coded decisions, plus a consistency check
    that knobs compose without interaction bugs). Solves axis (4)
    partially.
  - TEAM USAGE TELEMETRY (which configuration options are actually used
    by downstream teams). Solves axis (5); fundamentally requires
    population data the diff does not contain.

Until those exist, a56 remains a THICK norm whose value is to mark the
articulability boundary at exactly this point: developer-tool UX is
something experts judge by trying the tool, not by reading its diff.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a56"
ASPECT_NAME = "Developer tooling usability and formatter UX"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "Asks about USABILITY of dev tooling and UX of formatters — five "
    "intertwined human-experience axes (speed felt by users, clarity of "
    "diagnostics, in-editor fix quality, opinionated-yet-composable "
    "formatting tension, respect for team needs) none of which is "
    "observable from a diff. Distinct from a72 (artifact conforms to "
    "formatter — measurable), a95 (formatter configured — measurable), "
    "a42 (formatter enforced — measurable): a56 is the meta-aspect "
    "asking whether the tool those three presuppose is itself "
    "developer-friendly. Surrogates (mentions of formatter names, "
    "changelog speed claims, benchmark file presence, CodeAction handler "
    "counts, config knob deltas) each fail either by collapsing to "
    "text-length signal or by being orthogonal/negatively correlated "
    "with the norm. Honest THICK; would require benchmark harness + "
    "diagnostic-clarity rater + LSP fix regression fixtures + team usage "
    "telemetry to operationalize."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
