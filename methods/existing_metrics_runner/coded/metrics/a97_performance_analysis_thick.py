"""a97: Performance analysis and trade-offs — THICK.

The norm asks reviewers to "instrument/profile to find real bottlenecks,
classify optimization opportunities, and justify time/space/latency/bundle-
size trade-offs against product value and risk." Satisfying it requires:

  (1) PROFILING DATA from a representative workload (cProfile/py-spy/perf
      output, flamegraph, benchmark fixture timings, bundle-analyzer report).
      None of this is present in a code diff.
  (2) BENCHMARK COMPARISONS — before/after measurements on a fixed workload.
      A diff is a one-sided snapshot; we cannot reconstruct the "before"
      runtime, RSS, or bundle size from added/removed source lines.
  (3) PRODUCT-VALUE JUSTIFICATION — whether a trade-off (e.g. trading 200 KB
      bundle size for 30 ms TTI) is *worth it* depends on the product's
      latency budget, user demographics, and risk appetite. That is a
      thoroughly extra-textual judgment.
  (4) COMPLEXITY-ANALYSIS REASONING — even Big-O claims need the reviewer to
      know which input dimension dominates at production scale. The diff
      does not name the input distribution.

There ARE shallow static signals one could collect — e.g. detect nested
`for` loops over collection arguments, naive `+=` on `str` inside loops,
N+1 ORM queries, unbatched HTTP calls. Tools like `semgrep` with a
performance ruleset, `pylint`'s `consider-using-...` checks, and tree-sitter
nested-loop walks can flag candidates. But:

  - Flagged anti-patterns are *candidates for review*, not measurements of
    whether performance was analyzed. A PR can flag zero anti-patterns AND
    still ship a regression because the bottleneck is upstream.
  - More importantly, this metric would conflate "code is fast" with "the
    reviewer thought carefully about performance." A diff that adds careful
    profiling notes in the description but has nested loops would score
    poorly; a diff that silently swaps a fast path for a slow one with no
    loops would score well. Both are wrong-signed for this norm.
  - The norm targets a *meta-property of the review process*, not a property
    of the code. We have no observation of reviewer reasoning in the diff
    text alone — review comments are out of scope of `diff_text`.

A regex that hunts for the words "profile", "benchmark", "O(n)", "bundle
size" in PR text would, per the codegen_claude diagnostic, collapse into
text-length and produce zero signal beyond it.

So we self-classify THICK. The metric file exists for catalog completeness.

UNBLOCKERS — what extra-textual information would convert this to a
deterministic measurement:
  - Attached profiler output or flamegraph artifacts.
  - A benchmark suite with before/after numbers in CI logs.
  - A bundle-size diff report (webpack-bundle-analyzer, source-map-explorer).
  - PR description / review comments (currently outside `diff_text`).
  - A product-value oracle for trade-off scoring.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a97"
ASPECT_NAME = "Performance analysis and trade-offs"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "Performance analysis requires profiling data, benchmark before/after "
    "measurements, and product-value judgment about trade-offs. None of these "
    "are observable in a static code diff. Static anti-pattern hunting "
    "(nested loops, naive string concat, N+1 queries) would measure 'code is "
    "fast' rather than 'reviewer analyzed performance', conflating the wrong "
    "dimension."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
