"""Law trial metrics: a thin -> mid -> thick ladder over fact sections of federal
employment-discrimination opinions (Title VII pool, de-leaked facts only).

The ladder mirrors the E0 thickness intuition on real criteria:
  citation_grounding   thin  — statutory/case citation density is nearly mechanical
  factual_specificity  mid   — concrete dates/actors/events vs. boilerplate; partly countable
  element_mapping      thick — do the narrated facts map onto the legal elements of a
                               discrimination claim (protected class, adverse action,
                               causal nexus)? Recognizing this requires legal knowledge.

Seeds are deliberately crude (same convention as the code/CW trials): the optimizer must
earn any fidelity, and code seeds are the classic crude proxy a real pipeline would start
from.
"""

from __future__ import annotations

from typing import List, Tuple

from ..artifact import MetricArtifact

# ---- 1. citation grounding (thin) ---------------------------------------------------------

_CITE_PROMPT = """Score how well this fact section grounds its assertions in legal \
authority: statutes (e.g. 42 U.S.C. § 2000e), regulations, and case citations. \
1.0 = key assertions tied to cited authority; 0.0 = no citations at all."""

_CITE_CODE = '''import re

PATTERNS = [r"\\d+\\s+U\\.?S\\.?C\\.?", r"\\bF\\.\\s?(?:2d|3d|4th|Supp)",
            r"\\bv\\.\\s+[A-Z]", r"\\u00a7\\s*\\d+", r"C\\.F\\.R\\."]

def score(text):
    words = len(text.split())
    if words < 50:
        return None
    hits = sum(len(re.findall(p, text)) for p in PATTERNS)
    return min(1.0, hits / 6.0)
'''

# ---- 2. factual specificity (mid) ----------------------------------------------------------

_SPEC_PROMPT = """Score the factual specificity of this fact section. 1.0 = concrete \
dates, named actors, particular events and quantities; 0.0 = vague boilerplate \
("at all relevant times", "various actions") with no concrete particulars."""

_SPEC_CODE = '''import re

def score(text):
    words = text.split()
    if len(words) < 50:
        return None
    dates = len(re.findall(r"\\b(19|20)\\d{2}\\b|January|February|March|April|May|June|"
                           r"July|August|September|October|November|December", text))
    numbers = len(re.findall(r"\\$[\\d,]+|\\b\\d+\\b", text))
    boiler = len(re.findall(r"at all relevant times|various|numerous|certain of",
                            text, re.I))
    raw = (dates + 0.5 * numbers - 2.0 * boiler) / (len(words) / 100.0)
    return max(0.0, min(1.0, raw / 8.0))
'''

# ---- 3. element mapping (thick) -------------------------------------------------------------

_ELEM_PROMPT = """Score whether the narrated facts map onto the elements of an \
employment-discrimination claim: membership in a protected class, an adverse employment \
action, and facts suggesting a causal nexus between the two. 1.0 = all elements clearly \
supported by specific facts; 0.0 = the narrative never connects facts to any element."""

_ELEM_CODE = '''import re

KEYWORDS = [r"discriminat", r"retaliat", r"terminat|fired|discharg", r"protected",
            r"because of|on the basis of", r"race|sex|gender|religion|national origin",
            r"harass", r"adverse"]

def score(text):
    words = len(text.split())
    if words < 50:
        return None
    hits = sum(1 for p in KEYWORDS if re.search(p, text, re.I))
    return min(1.0, hits / 5.0)
'''


def trial_metrics_law() -> List[Tuple[MetricArtifact, MetricArtifact]]:
    """Returns [(prompt_artifact, code_artifact), ...] — thin, mid, thick."""
    spec = [
        ("citation_grounding", "Assertions grounded in legal authority",
         "Key factual and legal assertions are tied to cited statutes, regulations, "
         "or case authority",
         _CITE_PROMPT, _CITE_CODE, ["blank_lines"]),
        ("factual_specificity", "Concrete factual specificity",
         "The narrative uses concrete dates, named actors, particular events and "
         "quantities rather than vague boilerplate",
         _SPEC_PROMPT, _SPEC_CODE, ["blank_lines"]),
        ("element_mapping", "Facts map onto claim elements",
         "The narrated facts map onto the legal elements of an employment-discrimination "
         "claim: protected class, adverse employment action, and a causal nexus "
         "between them",
         _ELEM_PROMPT, _ELEM_CODE, ["blank_lines"]),
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
