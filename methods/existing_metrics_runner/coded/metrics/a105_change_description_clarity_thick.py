"""a105: Change description clarity, completeness, and rationale — THICK.

Norm (from aspects.json[105]):
  "Commit/PR messages and changelogs clearly and politely explain what changed
  and why, include sufficient background/context and links, present a coherent
  scope, and are concise and useful to reviewers and future readers."

Why THICK:

1. SUBSTRATE ABSENT FROM INPUT. The unified-diff text the metric receives
   contains code hunks and file headers. It does NOT contain:
     - the PR title or PR body,
     - per-commit messages (git log -p is stripped to file/hunk headers),
     - CHANGELOG/HISTORY/NEWS files unless the diff itself happens to touch
       one (and even then, the diff shows the *edit*, not the message-quality
       of the change being communicated about the rest of the patch).
   The norm's entire substrate (the prose that explains the change) lives
   outside the artifact we score, so a deterministic score has nothing real
   to read.

2. QUALITY, NOT STRUCTURE. Even if PR/commit/changelog text were available,
   the norm asks about CLARITY, COMPLETENESS, POLITENESS, COHERENT SCOPE,
   and USEFULNESS to future readers. Those are reader-side semantic judgments
   — the same taste-flavored cluster we flagged THICK for a16 (maintainability
   smell) and a78 (change/commit communication quality). Surface proxies
   (message length, presence of "because"/"fixes #N", bullet count, link
   count) systematically miss the norm: a 600-character message that simply
   narrates the diff scores high on length and bullets but FAILS the norm; a
   30-character "fix off-by-one in cursor advance (#412)" can SATISFY it.

3. STRONG OVERLAP WITH a78 AND a258 — BUT a78 IS THE NEAREST NEIGHBOUR.
   - a78 ("Change/commit/PR communication quality") covers the same surface
     (titles, commit messages, PR bodies) and the same QUALITY judgement.
     a78 is THICK for the same two reasons (substrate absent, quality is
     reader judgment); a105 inherits the verdict.
   - a258 ("commit message structure") is THICK in the current pipeline only
     because the substrate (commit subject lines) is also not in the diff. If
     commit metadata were ever joined to the artifact, a258 would become
     PARTIALLY_THIN via conventional-commits regex on the subject; a105 still
     would not, because conventional-commits structure does not measure
     completeness or rationale.

UNBLOCKER (documented for completeness — NOT implemented):
  If the dataset is extended so each example carries
  (diff, pr_title, pr_body, commit_messages, changelog_excerpt), a105 could
  become PARTIALLY_THIN by composing several weak signals:
    a. rationale-presence cue: "because|in order to|so that|fixes #|closes #"
       detector on the PR body / commit body.
    b. context-link cue: count of issue/PR/spec links per N lines of body.
    c. scope-coherence cue: ratio of files touched mentioned in body vs.
       files touched in diff (via diff parsing).
    d. diff-narration penalty: n-gram overlap between body and hunk headers.
  None of those measure QUALITY directly, and "politely" / "useful to future
  readers" remain reader judgments outside any of them. Combined they would
  give a non-degenerate but biased signal — worth doing only if substrate is
  added. With the current diff-only input, none of (a)-(d) are even
  applicable.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a105"
ASPECT_NAME = "Change description clarity, completeness, and rationale (commits, PRs, changelogs)"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "The unified-diff input strips PR titles/bodies, commit messages, and "
    "changelog prose, so the substrate this norm asks about is not in the "
    "artifact we score. Even if it were, judging CLARITY, COMPLETENESS, "
    "POLITENESS, COHERENT SCOPE, and USEFULNESS-to-future-readers is "
    "reader-side semantic judgment — the same taste cluster we flagged THICK "
    "for a78 (change/commit communication quality) and a16 (maintainability). "
    "Surface proxies (length, link count, 'because'/'fixes #' presence) "
    "collapse to a text-length signal, as the codegen_claude diagnostic "
    "showed. Sibling a258 (commit message structure) has a more deterministic "
    "substrate IF commit metadata were joined to examples; a105 does not, "
    "because conventional-commits structure does not measure rationale or "
    "completeness."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
