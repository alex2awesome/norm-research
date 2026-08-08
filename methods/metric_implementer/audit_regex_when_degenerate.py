"""Audit metrics that don't generalize, looking for regex-on-code shortcuts.

Definition of "fails to generalize" (any of):
  - std < 0.03 across applied fixtures (effectively constant)
  - applied rate > 80% but std == 0 (always returns the same value)
  - applied rate < 10% AND classification == "THIN" (suspiciously narrow)
  - all scores are 0.0 or 1.0 (degenerate binary)

For each such metric, count `re.` uses (annotated and unannotated), inspect
the source, and report whether regex appears to be the load-bearing logic
(vs. a thin shell over tree-sitter or a real CLI tool).

This is the audit the user specifically requested: "check each one for the
unnecessary presence of regex whenever they fail to generalize."
"""
from __future__ import annotations

import json
import math
import re as _re  # REGEX_OK: tool_output — counting `re.` calls in source files
from pathlib import Path
from typing import List

from .metrics import load_all

FIXTURES = Path(__file__).parent / "fixtures" / "sample_prs.json"
METRICS_DIR = Path(__file__).parent / "metrics"
# REGEX_OK: tool_output — count `re.` usages in metric .py files
RE_USE = _re.compile(r"\bre\.(compile|match|search|findall|finditer|split|sub|fullmatch)\b")
# REGEX_OK: tool_output — detect REGEX_OK annotations
ANNOT = _re.compile(r"#\s*REGEX_OK:\s*\S+")
TOOL_HINTS = ("subprocess", "tree_sitter", "tree-sitter", "import ast",
              "from radon", "import lizard", "ruff", "lizard", "radon",
              "pydocstyle", "mypy", "bandit", "semgrep", "eslint",
              "prettier", "gofmt", "sqlfluff", "google-java-format",
              "interrogate", "pydoclint", "chardet")


def main():
    metrics = load_all()
    fixtures = json.loads(FIXTURES.read_text())

    print(f"Auditing {len(metrics)} metrics across {len(fixtures)} fixtures.\n")
    print(f"{'metric':<28} {'class':>16} {'appl':>4} {'std':>6} "
          f"{'mode':>20}  {'re':>3} {'annot':>5}  {'flags'}")
    print("-" * 120)

    flagged_count = 0
    for m in metrics:
        # Score across fixtures
        scs = []
        for f in fixtures:
            a = m.applies(f["text"])
            if a:
                s = m.score(f["text"])
                if s is not None:
                    scs.append(s)

        n_appl = sum(1 for f in fixtures if m.applies(f["text"]))
        if scs:
            mu = sum(scs) / len(scs)
            std = math.sqrt(
                sum((s - mu) ** 2 for s in scs) / max(len(scs) - 1, 1)) \
                if len(scs) >= 2 else 0.0
            uniq = len(set(round(s, 4) for s in scs))
            most = max(set(scs), key=scs.count) if scs else None
            most_frac = sum(1 for s in scs if s == most) / len(scs) \
                if scs else 0.0
            mode_repr = f"{most:.2f}@{most_frac:.0%}" if most is not None else "—"
        else:
            std = float("nan")
            uniq = 0
            mode_repr = "—"

        # Read source, count regex and tool indicators
        src_files = list(METRICS_DIR.glob(f"{m.ASPECT_ID}_*.py"))
        if not src_files:
            print(f"  ! could not find source file for {m.ASPECT_ID}")
            continue
        src = src_files[0].read_text()
        n_re = len(RE_USE.findall(src))
        n_annot = len(ANNOT.findall(src))
        n_unannot = max(n_re - n_annot, 0)
        has_tool = any(h in src for h in TOOL_HINTS)

        # Flag conditions
        flags = []
        is_degenerate = False
        if not math.isnan(std):
            if std < 0.03 and n_appl > 0:
                flags.append("LOW_STD")
                is_degenerate = True
            if uniq == 1 and n_appl > 0:
                flags.append("CONSTANT")
                is_degenerate = True
            if scs and all(s in (0.0, 1.0) for s in scs) and uniq >= 2:
                flags.append("BINARY_ONLY")
        if n_appl > 0 and n_appl / len(fixtures) < 0.10 \
                and getattr(m, "CLASSIFICATION", "") == "THIN":
            flags.append("VERY_NARROW")
        if n_unannot > 0:
            flags.append(f"REGEX_{n_unannot}_UNANNOT")
        if is_degenerate and n_re > 0 and not has_tool:
            flags.append("⚠ REGEX-DEGEN")  # primary audit signal
            flagged_count += 1
        if is_degenerate and not has_tool and n_re == 0:
            flags.append("THIN_FALLBACK")  # stdlib-only and stuck

        name = f"{m.ASPECT_ID} {m.ASPECT_NAME[:18]}"
        print(f"{name:<28} {m.CLASSIFICATION:>16} {n_appl:>4} "
              f"{std:>6.3f} {mode_repr:>20}  {n_re:>3} {n_annot:>5}  "
              f"{','.join(flags)}")

    print()
    print(f"Metrics with REGEX-DEGEN warning (regex + low signal + no "
          f"real tool): {flagged_count}")
    print("\nLegend:")
    print("  LOW_STD       std < 0.03 — effectively constant")
    print("  CONSTANT      all applied scores are identical")
    print("  BINARY_ONLY   only emits 0/1 — possibly missing nuance")
    print("  VERY_NARROW   applies to <10% of fixtures but claims THIN")
    print("  REGEX_N_UNANNOT  unannotated regex uses in source")
    print("  ⚠ REGEX-DEGEN regex-dependent AND degenerate AND no real "
          "tool — fix candidate")
    print("  THIN_FALLBACK no regex, no tool, but degenerate — "
          "may need different angle")


if __name__ == "__main__":
    main()
