"""u19 Socially Conscious Satire: code validates whether a named target is systemic/institutional (moral-center check) via keyword predicate; LLM fields carry the satire-target extraction and the meta/therapeutic competing read."""

import re

LLM_FIELDS = {
    "satire_target": (
        "In <=8 words, name the social issue, institution, or grand "
        "narrative this joke critiques or satirizes, or say 'none'."
    ),
    "meta_or_therapeutic": (
        "Is this joke self-referential/meta about comedy itself, or does it "
        "process a personal struggle (therapeutic venting)? Answer: meta, therapeutic, or neither."
    ),
}

_SYSTEMIC_RE = re.compile(
    r"\b(government|politic\w*|president|congress\w*|senate|corporat\w*|ceo|"
    r"capitalis\w*|media|news|society|social\w*|religion|church|police|cop\w*|"
    r"system|institution\w*|patriarch\w*|racis\w*|sexis\w*|class\w*|economy|"
    r"war|military|industry|billionaire\w*|wealth|inequality|bureaucrac\w*|"
    r"culture|tradition\w*)\b",
    re.IGNORECASE,
)

_TRIVIAL_TARGET_RE = re.compile(
    r"^(a |an |the )?(man|woman|guy|girl|boy|kid|blonde|dog|cat|wife|husband)\b",
    re.IGNORECASE,
)


def _classify_meta(raw):
    if not raw:
        return None
    s = raw.lower()
    if "meta" in s or "self-referen" in s or "self referen" in s:
        return "meta"
    if "therapeutic" in s or "venting" in s or "process" in s or "personal" in s:
        return "therapeutic"
    if "neither" in s or "none" in s:
        return "neither"
    return None


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        extracted = extracted if isinstance(extracted, dict) else {}

        try:
            norm = ops.normalize(text)
        except Exception:
            norm = text

        # --- primary: satirical target must exist AND read as a systemic /
        # institutional / "grand narrative" target (moral-center predicate
        # lives in code; the target identity itself comes from the field). ---
        target = str(extracted.get("satire_target", "") or "").strip().lower()
        if not target or target in ("none", "n/a"):
            base = 0.15
        elif _SYSTEMIC_RE.search(target) and not _TRIVIAL_TARGET_RE.match(target):
            base = 0.8
        elif _TRIVIAL_TARGET_RE.match(target):
            # names a generic individual/stock character, not a social
            # issue/institution -- weak satire-of-society signal
            base = 0.4
        else:
            base = 0.55  # a target was named but doesn't match known systemic vocab

        # --- code-only corroboration: corpus-typicality guard so degenerate
        # scraped chrome/non-jokes don't get an extreme score by accident ---
        try:
            sims = ops.retrieve_similar(norm, k=5)
            vals = [s[0] for s in sims if isinstance(s, (list, tuple)) and s]
            avg_sim = (sum(vals) / len(vals)) if vals else None
        except Exception:
            avg_sim = None
        if avg_sim is not None and avg_sim < 0.02:
            base = base + (0.5 - base) * 0.3

        # --- secondary competing criterion: therapeutic function / meta
        # self-referential framing (adds a small bonus; does not compete
        # for the same weight as the dominant satire read). ---
        meta = _classify_meta(str(extracted.get("meta_or_therapeutic", "") or ""))
        bonus = 0.1 if meta in ("meta", "therapeutic") else 0.0

        s = base + bonus
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
