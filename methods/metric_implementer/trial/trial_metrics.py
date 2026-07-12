"""Three trial metrics for competitive-code (LC/CC/AC), each with BOTH implementations:
a seed judge rubric (kind="prompt") and a seed Python scorer (kind="code").

The seeds are deliberately mediocre — crude regex/AST heuristics and terse rubrics — so the
improvement loop has real headroom and the scaling grid has something to climb. Invariances
are declared per metric: a transformation that must NOT move one metric (e.g. identifier
renaming for the comments metric) is exactly the *target* of another.
"""

from __future__ import annotations

from typing import List, Tuple

from ..artifact import MetricArtifact

# ---- 1. identifier quality ---------------------------------------------------------------

_ID_PROMPT = """Score how descriptive the variable and function names in this solution are.
1.0 = names communicate each variable's role (e.g. left_boundary, char_counts).
0.0 = names are opaque single letters or meaningless (a, b, x1, tmp) outside loop idioms.
Loop indices i/j/k in short for-loops are acceptable and should not lower the score."""

_ID_CODE = '''import re

def score(text):
    names = set(re.findall(r"\\b([a-zA-Z_][a-zA-Z0-9_]*)\\s*=", text))
    names |= set(re.findall(r"def\\s+(\\w+)", text))
    names -= {"i", "j", "k", "_"}
    if not names:
        return None
    good = sum(1 for n in names if len(n) >= 4)
    return good / len(names)
'''

# ---- 2. explanatory comments ---------------------------------------------------------------

_CM_PROMPT = """Score whether the comments in this solution explain the APPROACH (why), not
just restate the code (what). 1.0 = comments convey the idea/invariant/complexity reasoning.
0.0 = no comments, or comments that only repeat what the line obviously does."""

_CM_CODE = '''import re

def score(text):
    lines = text.splitlines()
    comments = [l for l in lines if l.strip().startswith("#")]
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    if not code_lines:
        return None
    density = len(comments) / len(code_lines)
    return min(1.0, density * 4.0)
'''

# ---- 3. edge-case handling ----------------------------------------------------------------

_EC_PROMPT = """Score whether this solution explicitly handles edge cases: empty input,
single-element input, boundary values, or degenerate cases. 1.0 = clear guarded handling of
the relevant edge cases. 0.0 = no edge-case consideration at all."""

_EC_CODE = '''import re

PATTERNS = [r"if\\s+not\\s+\\w+", r"len\\(\\w+\\)\\s*==\\s*0", r"len\\(\\w+\\)\\s*<\\s*2",
            r"==\\s*\\[\\]", r"is\\s+None", r"return\\s+(?:0|\\[\\]|None|\\"\\")\\s*$"]

def score(text):
    hits = sum(1 for p in PATTERNS if re.search(p, text, re.M))
    return min(1.0, hits / 2.0)
'''


def trial_metrics() -> List[Tuple[MetricArtifact, MetricArtifact]]:
    """Returns [(prompt_artifact, code_artifact), ...] for the three trial metrics."""
    spec = [
        ("identifier_quality", "Descriptive identifier names",
         "Variable and function names are descriptive of their role",
         _ID_PROMPT, _ID_CODE, ["comment_rewording", "blank_lines"]),
        ("explanatory_comments", "Comments explain the approach",
         "Comments explain the approach and reasoning rather than restating the code",
         _CM_PROMPT, _CM_CODE, ["identifier_rename", "blank_lines"]),
        ("edge_case_handling", "Edge cases explicitly handled",
         "The solution explicitly handles edge cases (empty/boundary/degenerate inputs)",
         _EC_PROMPT, _EC_CODE, ["comment_rewording", "identifier_rename", "blank_lines"]),
    ]
    out = []
    for mid, name, desc, prompt, code, inv in spec:
        out.append((
            MetricArtifact(metric_id=mid, kind="prompt", body=prompt.strip(),
                           name=name, description=desc, invariances=inv),
            MetricArtifact(metric_id=mid, kind="code", body=code,
                           name=name, description=desc, invariances=inv),
        ))
    return out
