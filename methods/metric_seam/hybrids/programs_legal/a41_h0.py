"""a41 hybrid: Replacement trait stated outside class.

Criterion: a successor/selectee is identified and stated to be outside the
plaintiff's protected class.

Baseline (v1_structure) scans for a fixed replacement-verb phrase list
('replaced by', 'filled the position', 'hired instead', ...) within a 200-char
window of a fixed trait-term list, plus a literal "outside ... protected
class" bonus. This gets partial signal (train rho 0.232) but has two concrete
failure modes visible in the train pack:

  1. Phrase coverage is brittle. Real narratives paraphrase the same fact many
     ways the fixed list misses: "replaced her WITH" (active voice, not
     "replaced by"), "filled the role WITH", "instead hired"/"instead
     promoted"/"instead selected" (reversed word order from the baseline's
     list), bare "hired"/"appointed"/"offered the position to" with no
     "instead" at all, or no replacement-verb whatsoever ("A male has worked
     as Marketing Director since her dismissal"). All of these are held-out
     judge_score=1 cases the baseline scores 0.0 on. Code CAN catch more of
     this class of surface variation than the original list did, so we widen
     the phrase lexicon rather than replace it (keeps what already works).
  2. No negation guard. "She was NOT replaced by a person outside ... her
     protected class" contains every keyword the baseline looks for but
     asserts the opposite fact (judge=0, baseline=0.12 false positive). We add
     a cheap left-context negation check before crediting a match.

But no phrase list will ever cover every paraphrase, and the harder part of
the predicate is a COMPARISON the raw text alone can't settle: whether the
successor's stated trait is actually outside the PLAINTIFF's own class (a
white plaintiff replaced by another white employee is not this criterion,
but bare trait-term co-occurrence can't tell the difference). That comparison
is thick-input grounding, so it goes to two LLM fields: what trait (if any)
is stated for the person who got the role instead of the plaintiff, and
whether the text frames that as outside the plaintiff's own class. Code does
the phrase/proximity/negation predicate; the LLM fields plug the paraphrase
gap and do the plaintiff-vs-successor contrast code can't reliably infer.
Combined via noisy-OR so a strong hit on either channel alone is enough,
and a strong baseline-style regex hit is never diluted by a weak LLM answer.
"""
import re

LLM_FIELDS = {
    "successor_trait": (
        "If the text names or describes a specific person who was hired, "
        "promoted, appointed, or selected for the role/position INSTEAD OF "
        "the plaintiff, state that person's protected trait as given in the "
        "text (race, sex, age, national origin, religion, disability) in "
        "under 8 words, e.g. 'white male' or '26-year-old'. Answer NONE if "
        "no such successor is described or no trait is stated for them."
    ),
    "outside_plaintiffs_class": (
        "Only if a successor/replacement person is described: does the text "
        "state or clearly imply that this person's trait is DIFFERENT from "
        "the plaintiff's own protected-class trait (i.e. they are outside "
        "the plaintiff's class)? Answer YES or NO in a few words. Answer "
        "NONE if no successor/replacement person is described at all."
    ),
}

# --- code layer 1: replacement-event phrases (baseline list, widened for the
# paraphrases/word-orders the fixed list missed) ---
_REPL_TERMS = [
    r"replaced (?:her|him|them)?\s*with", r"replaced by", r"\breplacement\b",
    r"\bsuccessor\b", r"succeeded by", r"\bselectee\b",
    r"filled (?:the )?(?:position|role|job)\s*(?:by|with)", r"filled the position",
    r"filled by", r"in (?:his|her|their) place",
    r"hired instead", r"instead hired",
    r"promoted instead", r"instead promoted",
    r"selected instead", r"instead selected",
    r"chosen instead", r"instead chosen",
    r"appointed instead", r"instead appointed",
    r"took over (?:his|her|their) (?:duties|role|position)",
    r"\bsimilarly situated\b", r"\bcomparator\b",
    r"offered (?:the )?(?:position|job|role)\s*to",
    r"was (?:hired|selected|chosen|appointed|promoted)\b",
]

# --- code layer 1: protected-trait terms (baseline list, lightly widened) ---
_TRAIT_TERMS = [
    r"caucasian", r"\bwhite\b", r"\bblack\b", r"african[- ]american",
    r"\bmale\b", r"\bfemale\b", r"\bman\b", r"\bwoman\b",
    r"\byounger\b", r"\bolder\b", r"\d{1,3}[- ]year[- ]old",
    r"national origin", r"hispanic", r"latino", r"latina", r"asian",
    r"\bnon-?disabled\b", r"not (?:a member of|in) the protected class",
    r"outside (?:his|her|their|the) protected class",
    r"\breligion\b", r"christian", r"muslim", r"jewish", r"\bsikh\b",
]

_EXPLICIT_OUTSIDE_RE = re.compile(r"outside (?:his|her|their|the) protected class")
_NEG_RE = re.compile(r"\b(not|never|n't|without)\b")
_NONE_ANSWERS = {"", "none", "n/a", "na", "not stated", "not mentioned", "unknown"}


def _is_negated(t, start, window=45):
    """Cheap left-context negation guard: 'was NOT replaced by ... outside
    her protected class' should not count as an affirmative match."""
    ctx = t[max(0, start - window):start]
    return bool(_NEG_RE.search(ctx))


def _is_none_answer(s):
    return not s or s.strip().lower() in _NONE_ANSWERS


def score(text: str, extracted: dict, ops) -> float:
    try:
        norm = ops.normalize(text) if text else ""
        t = norm.lower()

        repl_spans = [m.span() for pat in _REPL_TERMS for m in re.finditer(pat, t)
                      if not _is_negated(t, m.start())]
        trait_spans = [m.span() for pat in _TRAIT_TERMS for m in re.finditer(pat, t)]

        proximate = 0
        window = 220
        for rs, re_ in repl_spans:
            for ts, te in trait_spans:
                if abs(ts - re_) <= window or abs(rs - te) <= window:
                    proximate += 1
                    break

        explicit_outside = 1 if any(
            not _is_negated(t, m.start()) for m in _EXPLICIT_OUTSIDE_RE.finditer(t)
        ) else 0
        n_named_repl = len(repl_spans)

        code_score = (0.55 * min(1.0, proximate / 1.0)
                      + 0.25 * min(1.0, n_named_repl / 2.0)
                      + 0.2 * explicit_outside)
        code_score = max(0.0, min(1.0, code_score))

        # LLM layer: paraphrase-proof successor detection + the
        # plaintiff-vs-successor class contrast code can't reliably judge.
        ext = extracted or {}
        successor_trait = ext.get("successor_trait", "")
        outside_ans = (ext.get("outside_plaintiffs_class", "") or "").strip().lower()

        llm_score = 0.0
        if not _is_none_answer(successor_trait):
            llm_score = 0.35
        if outside_ans and outside_ans not in _NONE_ANSWERS:
            if re.search(r"\byes\b", outside_ans):
                llm_score = 1.0
            elif re.search(r"\bno\b", outside_ans):
                llm_score = min(llm_score, 0.1)

        # Noisy-OR: a strong hit on either channel is sufficient; a weak LLM
        # answer never dilutes a strong code-side (baseline-style) match.
        combined = 1.0 - (1.0 - code_score) * (1.0 - llm_score)
        return max(0.0, min(1.0, combined))
    except Exception:
        return 0.5
