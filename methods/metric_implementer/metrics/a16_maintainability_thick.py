"""a16: Maintainability (ease of change over time) — THICK.

This norm asks about a property that only manifests AFTER the code is shipped:
how easy will future engineers find it to modify? We have no observable
proxy in the diff itself. Proxies like cyclomatic complexity (lizard, a0) and
function length (already in Tier 3) capture *local* complexity but not the
"ease of FUTURE change" the norm names.

We could regex for "magic numbers", "long names", etc. — but every such
heuristic is contestable, and the codegen_claude diagnostic showed that this
class of regex heuristic produces zero signal beyond text length.

So this metric self-classifies THICK. The metric file exists for catalog
completeness; it records that this norm is NOT deterministically verifiable.
"""
from __future__ import annotations
from typing import Optional

ASPECT_ID = "a16"
ASPECT_NAME = "Maintainability (ease of change over time)"
TIER = 0
TOOLS = []
APPLIES_TO_LANGS = []
CLASSIFICATION = "THICK"

THICK_REASON = (
    "Maintainability is a property of how the code will be USED in the future, "
    "not a property observable in the diff. Local proxies (CCN, function "
    "length, naming) measure *complexity*, not *ease of change*. The latter "
    "depends on coupling to code outside the diff and on engineering culture "
    "we cannot observe."
)


def applies(diff_text: str) -> bool:
    return False


def score(diff_text: str) -> Optional[float]:
    return None
