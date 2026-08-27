"""a222: Inline math delimiters ($...$) used consistently -- hybrid channel.

Design note (from residual analysis of the 30 train examples): the v0 keyword
baseline (regex-count of $...$/\\(..\\) hits) gets rho=0.173 because *presence*
of dollar-delimited math is nearly universal in this corpus and is not what the
judge rewards. The judge instead tracks whether display ($$..$$) vs inline
($..$) was the *right choice* for each formula -- e.g. a one-line equation that
merely continues a sentence's clause ("... is itself, or $$\\Phi(Q)=Q$$ PS...")
reads as broken flow even though the delimiters are well-formed, while a long,
dense derivation crammed into inline $..$ reads as cramped even though nothing
is syntactically wrong. That is a discourse/reading judgment (does the
preceding clause "announce" a standalone equation, or does the equation merely
continue the sentence?) that a regex over spans cannot see without reading
context -- so it is delegated to the LLM field below. Code keeps: (a) a
no-math floor, (b) mechanical delimiter-health (real breakage), (c) the two
anti-patterns the criterion names explicitly (HTML <math>, mixed $ / \\(..\\)
conventions), and (d) a coarse "display-for-clutter" signal (many short,
simple equations pushed into display) as a weak, generalizable proxy.
"""
import re

LLM_FIELDS = {
    "awkward_math": (
        "Quote the single shortest math formula, if any, whose delimiter choice "
        "breaks the sentence's reading flow: either a $$..$$ display equation that "
        "merely continues a clause instead of being introduced as a standalone "
        "equation, or a $..$ inline formula so long or multi-step that it should "
        "have been set apart as a display equation instead. Otherwise answer NONE."
    ),
}

_SHORT_DISPLAY_TOK = 8          # display spans at/under this token count count as "simple"
_CLUTTER_MIN_DISPLAYS = 3        # need several displays before "clutter" is meaningful
_CLUTTER_DISPLAY_FRAC = 0.55     # displays must dominate the span mix
_HTML_MATH = re.compile(r'</?math[ >]', re.IGNORECASE)
_PAREN_INLINE = re.compile(r'\\\(.+?\\\)', re.DOTALL)
_DOLLAR_INLINE_RAW = re.compile(r'(?<!\$)\$(?!\$)[^$\n]{1,400}(?<!\$)\$(?!\$)')


def _clip(x):
    return max(0.0, min(1.0, x))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        if len(t.strip()) < 10:
            return 0.5

        spans = ops.extract_math_spans(t)
        if not spans:
            # No math at all: the criterion has essentially nothing to reward.
            # Training data shows this ranks at the very bottom, but we avoid a
            # hard 0.0 to not overfit a single observed instance.
            return 0.15

        inline_spans = [c for k, c in spans if k == 'inline']
        display_spans = [c for k, c in spans if k == 'display']
        n_total = max(1, len(spans))

        def toklen(c):
            try:
                return len(ops.latex_tokens(c))
            except Exception:
                return len((c or "").split())

        # 1) mechanical breakage from the parser -- real invalid-delimiter cases.
        try:
            health = ops.delimiter_health(t)
            issues = sum(v for v in health.values() if isinstance(v, (int, float)))
        except Exception:
            issues = 0
        health_penalty = min(0.5, 0.12 * issues)

        # 2) explicit negative marker named in the criterion.
        html_penalty = 0.3 if _HTML_MATH.search(t) else 0.0

        # 3) mixed delimiter conventions ($..$ next to \(..\)) -- the criterion
        #    asks for one consistent inline convention across the manuscript.
        uses_dollar = bool(_DOLLAR_INLINE_RAW.search(t))
        uses_paren = bool(_PAREN_INLINE.search(t))
        mixed_penalty = 0.15 if (uses_dollar and uses_paren) else 0.0

        # 4) coarse "display clutter" proxy: many short/simple equations pushed
        #    into display mode reads as over-formatted rather than as genuine
        #    standalone results worth setting apart.
        display_frac = len(display_spans) / n_total
        short_displays = sum(1 for c in display_spans if toklen(c) <= _SHORT_DISPLAY_TOK)
        clutter_penalty = 0.0
        if (len(display_spans) >= _CLUTTER_MIN_DISPLAYS
                and display_frac >= _CLUTTER_DISPLAY_FRAC
                and short_displays >= _CLUTTER_MIN_DISPLAYS - 1):
            clutter_penalty = 0.2

        # 5) LLM-grounded flow check -- the primary discriminating signal, since
        #    it requires reading whether a formula's delimiter choice matches how
        #    it is introduced by the surrounding prose (mechanical rules alone
        #    disagree on this across examples of similar length/complexity).
        llm_flag = ((extracted or {}).get("awkward_math", "") or "").strip()
        llm_bad = bool(llm_flag) and llm_flag.upper() not in ("NONE", "N/A", "NA")
        llm_penalty = 0.35 if llm_bad else 0.0

        penalty = (health_penalty + html_penalty + mixed_penalty
                   + clutter_penalty + llm_penalty)

        base = 0.85
        return _clip(base - penalty)
    except Exception:
        return 0.5
