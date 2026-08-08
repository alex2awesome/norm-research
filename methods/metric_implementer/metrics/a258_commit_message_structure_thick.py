"""a258: Commit message structure and content — THICK.

The norm asks for "accurate, descriptive commit messages with conventional
subject/body separation and structured footers (e.g., issue references and
explicit BREAKING CHANGE notes), optionally using concise tags."

To verify it deterministically we would need the *commit message text* —
the subject line, the blank-line-separated body, and any trailers
(`Fixes #123`, `Co-Authored-By:`, `BREAKING CHANGE:`). Tools exist for
this: `commitlint`, `gitlint`, or a regex over the conventional-commits
grammar (`<type>(<scope>)!: <subject>`) all work *when given a commit
message*.

But this metric receives `diff_text`, which is the produced artifact
content: PR title, PR description, and a `diff --git ...` unified diff.
None of that is the commit message:

  - The PR title is the title of the pull request, not of any commit. A
    squash-merge will use it, but a merge-commit-strategy PR will not.
  - The PR description is free-form Markdown; it carries no "subject /
    body / footer" structure.
  - The diff body shows file content changes; commit metadata
    (`Author:`, `Date:`, message lines beginning a `git log -p` block) is
    stripped from a `diff --git` unified diff.

Adjacent THICKs (a78 commit-history hygiene, a16 maintainability, a97
performance analysis) share this property: the verifying observation is
*metadata not present in the diff text*. A regex search of the
PR description for words like "fix", "feat", or "BREAKING CHANGE"
would, per the codegen_claude diagnostic, collapse into a description-
length proxy with no signal — and would conflate "PR description
mentions fixes" with "commit messages are well-structured", which is a
different norm entirely.

So we self-classify THICK.

UNBLOCKERS — what extra-textual information would convert this to a
deterministic measurement:
  - The raw commit message(s) for the PR (`git log --format=%B <range>`).
    Then `commitlint --config conventional` or `gitlint` would score
    structural compliance directly.
  - Commit trailer parsing for `Fixes:`, `Co-Authored-By:`,
    `BREAKING CHANGE:` footers.
  - For squash-merge PRs, the squash-message preview (still not in the
    diff text).
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a258"
ASPECT_NAME = "Commit message structure and content"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "Commit message structure (conventional subject/body separation, "
    "structured footers, BREAKING CHANGE notes) is metadata not present in "
    "the diff text. A unified `diff --git` block carries file content "
    "changes only; commit subjects, bodies, and trailers are stripped. PR "
    "title and PR description are not commit messages. Tools like "
    "`commitlint` and `gitlint` would verify this norm deterministically "
    "if given the commit message text, but that text is unavailable here."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
