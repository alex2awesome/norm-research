"""Soft check: prompt the implementer to double-check tool/library choice.

We list every `re.` use without a `# REGEX_OK: <reason>` annotation within
the 3 lines above it. This is a PROMPT, not a prohibition. The goal is to
make sure we searched for a real parser/tool/library before falling back
to regex.

See GUIDE.md §"Implementation discipline: library-first, regex when honest"
for the actual rule and the allow-list of acceptable reasons.

Exit code: 0 always (this is a soft check). Use `--strict` to exit nonzero
on unannotated finds, e.g. for CI gating.
"""
from __future__ import annotations

# REGEX_OK: tool_output — we're parsing Python source for our own `re.` calls.
# This script self-applies its own rule and annotates honestly.
import re
import sys
from pathlib import Path

METRICS_DIR = Path(__file__).parent / "metrics"
# REGEX_OK: tool_output — matching token-level python identifier names.
RE_USE = re.compile(
    r"\bre\.(compile|match|search|findall|finditer|split|sub|fullmatch)\b")
# REGEX_OK: tool_output — matching our own annotation comment format.
ANNOTATION = re.compile(r"#\s*REGEX_OK:\s*(.+?)\s*$", re.MULTILINE)

HINTS = {
    "py": "→ tree-sitter-python or `ast` for Python source",
    "js": "→ tree-sitter-javascript",
    "ts": "→ tree-sitter-typescript",
    "java": "→ tree-sitter-java",
    "go": "→ tree-sitter-go",
    "prose": "→ spaCy / stanza / textstat (or `# REGEX_OK: prose_surface`)",
    "diff": "→ whatthepatch / unidiff",
}


def file_hint(name: str) -> str:
    low = name.lower()
    for k, h in HINTS.items():
        if k in low:
            return h
    return ""


def check_file(p: Path) -> list[str]:
    lines = p.read_text().splitlines()
    flagged = []
    for i, line in enumerate(lines):
        if not RE_USE.search(line):
            continue
        recent = "\n".join(lines[max(0, i - 3):i])
        if ANNOTATION.search(recent):
            continue
        flagged.append(f"  {p.name}:{i+1}  {line.strip()}")
    return flagged


def main() -> int:
    strict = "--strict" in sys.argv
    total = 0
    for f in sorted(METRICS_DIR.glob("*.py")):
        if f.name.startswith("_"):
            continue
        flagged = check_file(f)
        if not flagged:
            continue
        hint = file_hint(f.name)
        print(f"\n{f.name}: {len(flagged)} unannotated regex use(s) "
              f"{hint}")
        for line in flagged:
            print(line)
        total += len(flagged)
    if total:
        print(f"\n{total} unannotated regex use(s) across metrics.")
        print("If you searched for a real tool/parser first and regex is "
              "genuinely the right choice, add `# REGEX_OK: <reason>` per "
              "GUIDE.md §Implementation discipline.")
    else:
        print("All regex uses are either absent or annotated.")
    return 1 if strict and total else 0


if __name__ == "__main__":
    sys.exit(main())
