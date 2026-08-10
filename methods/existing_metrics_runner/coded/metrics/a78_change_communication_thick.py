"""a78: Change/commit/PR communication quality — THICK.

This norm evaluates how well titles, commit messages, and PR descriptions
summarize WHAT changed and WHY: rationale, context, scale-appropriate detail,
avoidance of pure diff-narration, and review-aiding annotations on large CLs.

Why THICK:

1. The unified-diff artifact we receive contains NO PR description, NO PR
   title, and NO commit messages. Those texts live outside the diff hunks and
   are simply not part of the input we score. The metric has no signal to
   read.

2. Even if PR text were available, judging communication QUALITY (clear
   summary vs. diff narration; sufficient rationale; "scaled" detail) requires
   reader-side semantic judgment: did the author convey the *why* in a way
   that a future reviewer can act on? That is the same kind of taste/reader
   judgment we flagged THICK for a16 (maintainability) and a9-style norms.

3. Surface proxies (message length, presence of conventional-commit prefix,
   bullet count) systematically miss the norm: a 400-character message that
   only restates the diff scores high on length but FAILS the norm; a tight
   30-character message that names the bug and links the issue can SATISFY
   it. The codegen_claude diagnostic showed length-correlated regex
   "measurements" of communication quality collapsed to a text-length proxy
   with AUC ~0.50.

Adjacent norm a258 (commit message *structure*) has slightly more deterministic
substrate (conventional-commits regex on the subject line is real) but a78 as
written asks about COMMUNICATION QUALITY, not structural conformance, and so
is THICK at the same level as a16.

UNBLOCKER: if the dataset is extended so each example carries
`(diff, pr_title, pr_body, commit_messages)` jointly, this becomes a
PARTIALLY_THIN candidate: we could measure (a) conventional-commit subject
conformance via regex on the subject line, (b) rationale presence via a
"because/in order to/fixes #" detector, (c) diff-narration penalty via
n-gram overlap between PR body and diff hunk headers. None of those measure
QUALITY directly, but together they would give a non-degenerate signal.
Without the PR text in the input, none of them are even applicable.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a78"
ASPECT_NAME = "Change/commit/PR communication quality"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "The unified-diff input contains no PR title, PR body, or commit messages, "
    "so the norm's substrate is absent from what we score. Even with that text "
    "available, judging clear-summary vs. diff-narration and 'sufficient "
    "rationale scaled to the change' is reader-side semantic judgment that "
    "surface proxies (length, prefix presence) collapse into a text-length "
    "signal. Adjacent norm a258 (commit message structure) has more "
    "deterministic substrate via conventional-commits, but a78 as written "
    "targets communication QUALITY, not structural conformance."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
