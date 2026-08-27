"""a49 hybrid: federal-sector employer flag (Title VII 2000e-16 / Babb applies).

Criterion: the employer-DEFENDANT is a U.S. federal-government entity (an
executive department, independent agency, or uniformed service, including
USPS) as opposed to a state/local government, a private company, or a
CONTRACTOR that merely does business with or is funded by a federal client.
Only the direct employer's own sector matters -- a private contractor whose
client is a federal agency (or a nonprofit that receives federal/city
funding) is NOT a federal-sector employer under 2000e-16, even though a
federal agency's name may appear throughout such a narrative.

Baseline (v0_keyword) is a short, fixed enumeration of ~20 agency-name
phrases matched by exact substring. It fails in two directions: (1) it
misses the large majority of the 100+ real federal departments/agencies/
uniformed services it never enumerated (NRC, TSA/DHS, DOL, HHS, EPA, FBI,
NASA, arsenals/military installations, ...), and it under-counts partial
hits (e.g. "Secretary of Labor" alone => 0.5 even though DOL is squarely
federal); (2) its substring matching is brittle to real phrasing variants
("Equal Employment Opportunity (EEO) counselor" does not contain the
literal substring "eeo counselor" because of the parenthetical). No fixed
keyword list can enumerate every federal agency, and no regex can tell "the
client/funder agency named in this narrative" apart from "the entity that
actually employed the plaintiff" (the contractor/funding trap) -- that
needs a read of the passage. So we keep the baseline's keyword idea as a
broadened, generalized code-side backstop (more agencies/uniformed
services, a flexible EEO-counselor regex, and negative state/local/
municipal/contractor markers), and add two THICK-INPUT LLM fields: a
direct classification of the ACTUAL employer-defendant's sector, and
whether the narrative shows the federal-sector ADMINISTRATIVE-PROCESS
fingerprint (agency EEO counselor contact / agency final decision /
MSPB / OPM / EEOC-Office-of-Federal-Operations appeal) tied to this
plaintiff's own claim -- a procedural signature private-sector EEOC-charge
cases do not have. Code keeps the predicate: it never parrots the LLM's
verdict, it maps the two LLM signals into bounded component scores and
blends them with the keyword backstop.
"""
import re
import math


LLM_FIELDS = {
    "employer_class": (
        "Classify the ACTUAL employer-defendant who employed the plaintiff "
        "(not a client, funder, or entity merely mentioned in passing) as "
        "one word: federal, state, local, private, or unclear. Then briefly "
        "name the entity. If the employer is a private contractor whose "
        "client/funder is a federal agency, answer 'private' (not federal)."
    ),
    "federal_procedure": (
        "Does the narrative show THIS plaintiff went through the federal-"
        "sector EEO process (contacting an agency EEO counselor, an "
        "agency's own final decision, or an MSPB/OPM/EEOC-Office-of-"
        "Federal-Operations appeal) for their own claim? Answer yes or no, "
        "plus which marker, in <=15 words."
    ),
}


def _sat(x, k=1.0):
    return 1.0 - math.exp(-x / max(1e-6, k))


# Broadened, generalized federal-entity vocabulary (executive departments,
# independent agencies, uniformed services, USPS) -- this is the actual
# taxonomy of the federal government, not literal train-set facts.
_FED_TERMS = [
    "u.s. army", "u.s. navy", "u.s. air force", "u.s. marine", "u.s. coast guard",
    "u.s. space force", "united states army", "united states navy",
    "united states air force", "united states marine", "army base",
    "air force base", "naval station", "naval base", "arsenal",
    "military installation", "army community hospital",
    "department of veterans affairs", "veterans affairs medical center",
    "department of defense", "department of the army", "department of the navy",
    "department of the air force", "united states postal service", "usps",
    "postal service", "postmaster general", "internal revenue service",
    " irs ", "federal government", "federal agency", "federal employee",
    "federal sector", "secretary of", "eeo office",
    "office of federal operations", "merit systems protection board", " mspb ",
    "office of personnel management", " opm ", "social security administration",
    "department of education", "department of justice", "department of labor",
    "department of health and human services", "department of homeland security",
    "department of state", "department of commerce", "department of energy",
    "department of the interior", "department of agriculture",
    "department of housing and urban development", "department of transportation",
    "environmental protection agency", " epa ", "federal bureau of investigation",
    " fbi ", "central intelligence agency", " cia ",
    "nuclear regulatory commission", " nrc ", "transportation security administration",
    " tsa ", "bureau of apprenticeship and training",
    "national aeronautics and space administration", " nasa ",
    "smithsonian institution", "federal reserve", "general services administration",
    "government accountability office", "2000e-16", "rehabilitation act of 1973",
    "civil service reform act", "final agency decision", "agency's final decision",
]

# Generalized negative markers: state/local/municipal government and
# private-contractor-of-a-government-client patterns (the false-positive
# trap this criterion is prone to).
_NEG_TERMS = [
    "state department of", "county department of", "city department of",
    "department of corrections", "department of correctional services",
    "commonwealth of", "city of ", "county of ", "municipal ",
    "public school district", "board of education", "housing authority",
    "transit authority", "transportation authority", "human rights commission",
    "state university", "community college", "state government",
    "local government", "contractor for", "contracts with",
]

_EEO_COUNSELOR_RE = re.compile(r"eeo\)?\s*counselor")
_EEO_SPELLED_RE = re.compile(
    r"equal employment opportunity\s*(?:\([^)]{0,10}\))?\s*counselor"
)


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        norm = ops.normalize(raw) if raw else ""
        low = norm.lower()

        fed_hits = sum(1 for k in _FED_TERMS if k in low)
        if _EEO_COUNSELOR_RE.search(low) or _EEO_SPELLED_RE.search(low):
            fed_hits += 1
        neg_hits = sum(1 for k in _NEG_TERMS if k in low)

        code_component = _sat(fed_hits, k=1.5)
        if fed_hits == 0 and neg_hits > 0:
            code_component = 0.05

        # --- LLM-grounded predicate (thick-input grounding code can't reach) ---
        ext = extracted or {}
        cls_raw = str(ext.get("employer_class", "") or "").strip().lower()
        proc_raw = str(ext.get("federal_procedure", "") or "").strip().lower()

        is_federal = None
        if cls_raw and "unclear" not in cls_raw:
            if "contractor" in cls_raw:
                is_federal = False
            elif "non-federal" in cls_raw or "nonfederal" in cls_raw:
                is_federal = False
            elif "federal" in cls_raw:
                is_federal = True
            elif any(
                k in cls_raw
                for k in ("state", "local", "private", "municipal", "county", "city")
            ):
                is_federal = False

        proc_yes = proc_raw.startswith("yes")

        if is_federal is True:
            llm_component = 1.0 if proc_yes else 0.92
        elif is_federal is False:
            llm_component = 0.4 if proc_yes else 0.10
        else:
            # LLM field empty/unclear: fall back to the code-side backstop.
            llm_component = max(code_component, 0.55) if proc_yes else code_component

        val = 0.75 * llm_component + 0.25 * code_component
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
