"""New-side line addressing for unified diffs used by verifier witnesses."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .schema import Verdict


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


class DiffAddressError(ValueError):
    """Raised when a diff or witness address is malformed."""


@dataclass(frozen=True)
class DiffLine:
    path: str
    line: int
    text: str
    added: bool


def parse_new_side_lines(diff_text: str) -> tuple[DiffLine, ...]:
    """Return context and added lines with canonical new-file addresses.

    Deleted lines have no new-side address and are omitted.  File metadata and
    the ``\\ No newline`` sentinel are omitted as well.
    """

    if not isinstance(diff_text, str) or not diff_text.startswith("diff --git "):
        raise DiffAddressError("expected a nonempty unified git diff")
    path: str | None = None
    new_line: int | None = None
    in_hunk = False
    result: list[DiffLine] = []
    seen: set[tuple[str, int]] = set()

    for raw in diff_text.splitlines():
        if raw.startswith("diff --git "):
            path = None
            new_line = None
            in_hunk = False
            continue
        if raw.startswith("+++ "):
            marker = raw[4:]
            if marker == "/dev/null":
                path = None
            elif marker.startswith("b/") and len(marker) > 2:
                path = marker[2:]
            else:
                raise DiffAddressError(f"unsupported new-file marker: {raw}")
            continue
        match = _HUNK.match(raw)
        if match:
            if path is None:
                raise DiffAddressError("hunk appeared before a new-file path")
            new_line = int(match.group(1))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if raw.startswith("\\ No newline at end of file"):
            continue
        if raw.startswith("-"):
            continue
        if raw.startswith("+") or raw.startswith(" "):
            if path is None or new_line is None:
                raise DiffAddressError("new-side line has no active hunk")
            identity = (path, new_line)
            if identity in seen:
                raise DiffAddressError(f"duplicate new-side address: {path}:{new_line}")
            seen.add(identity)
            result.append(
                DiffLine(path=path, line=new_line, text=raw[1:], added=raw.startswith("+"))
            )
            new_line += 1
            continue
        # A new metadata section terminates the current hunk.  An otherwise
        # unmarked body line is invalid rather than silently addressable.
        in_hunk = False

    if not result:
        raise DiffAddressError("diff contains no visible new-side source lines")
    return tuple(result)


def visible_line_index(diff_text: str) -> dict[str, frozenset[int]]:
    by_path: dict[str, set[int]] = {}
    for line in parse_new_side_lines(diff_text):
        by_path.setdefault(line.path, set()).add(line.line)
    return {path: frozenset(lines) for path, lines in by_path.items()}


def added_line_index(diff_text: str) -> dict[str, frozenset[int]]:
    by_path: dict[str, set[int]] = {}
    for line in parse_new_side_lines(diff_text):
        if line.added:
            by_path.setdefault(line.path, set()).add(line.line)
    return {path: frozenset(lines) for path, lines in by_path.items()}


def address_is_visible(
    index: dict[str, frozenset[int]], path: str, start_line: int, end_line: int
) -> bool:
    if start_line < 1 or end_line < start_line or path not in index:
        return False
    return all(line in index[path] for line in range(start_line, end_line + 1))


def validate_verdict_addresses(
    diff_text: str, verdict: Verdict, *, require_added: bool = False
) -> None:
    """Require every witness to identify lines actually visible in the item.

    ``require_added`` is useful for mutation controls whose load-bearing
    evidence must be newly introduced.  Real relation verifiers may cite
    unchanged context visible in the diff, so their default is the full
    new-side line index.
    """

    if not isinstance(verdict, Verdict):
        raise TypeError("verdict must be a verifier Verdict")
    index = added_line_index(diff_text) if require_added else visible_line_index(diff_text)
    for witness in verdict.witnesses:
        if not address_is_visible(
            index, witness.path, witness.start_line, witness.end_line
        ):
            scope = "added" if require_added else "visible"
            raise DiffAddressError(
                f"witness is outside {scope} new-side lines: "
                f"{witness.path}:{witness.start_line}-{witness.end_line}"
            )
