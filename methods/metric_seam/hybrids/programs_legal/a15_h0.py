"""Hybrid metric channel for a15 (legal_title_vii) --
"HWE based on protected trait and unwelcome".

Criterion: hostile-environment harassment must be BOTH (a) linked to a
protected trait (a slur, trait-based comment, sexual advance) AND (b) shown to
be unwelcome (plaintiff objected, reported it, told the harasser to stop,
rebuffed it, avoided the harasser) -- Oncale v. Sundowner. General incivility,
personality conflicts, "equal-opportunity" bullying, and bare doctrinal labels
("hostile work environment", "harassment") recited in a PROCEDURAL/UNRELATED
context (exhaustion, timeliness, retaliation-only, accommodation-only cases)
do NOT satisfy the criterion.

Baseline failure pattern observed in the train pack (v0_keyword, rho=0.502):
  * FALSE POSITIVES -- narratives that use the word "harass(ment)" or
    "complain(ed) about" only in a procedural aside (a grievance mislabeled as
    "a complaint about harassment" in a religious-accommodation exhaustion
    dispute; a Sunday-scheduling accommodation case where the plaintiff merely
    "complained about" his manager; a retaliation-only claim with a throwaway
    "hostile work environment" recital and no described incident) get partial
    keyword credit even though the judge scores them 0 -- there is no
    described trait-linked conduct at all.
  * FALSE NEGATIVES -- narratives with vivid, concrete, high-scoring
    harassment (quoted derogatory race/sex comments, mimicry of a "black
    female television character", isolation to a supply-closet desk) that
    never happen to contain the exact literal keywords score 0 on the
    baseline despite the judge scoring them ~0.85-0.95.
Both failure modes are the same underlying problem: keyword presence is a
weak proxy for whether the CONSTRUCT -- concrete trait-linked conduct plus a
shown unwelcome reaction to THAT conduct -- is actually present in the facts.
Code cannot tell "a complaint about harassment" (procedural aside) apart from
"she complained to Nelson about the derogatory comments" (a real HWE fact);
that discrimination needs an LLM read of the two narrow constructs. The
keyword density is kept as a code-side hedge/severity tie-breaker (and is the
sole signal when the extractor returns nothing usable), rather than being
discarded, since real, well-documented harassment usually also carries some
of that vocabulary.
"""

LLM_FIELDS = {
    "trait_conduct": (
        "In <=15 words, quote or paraphrase the SPECIFIC protected-trait-linked "
        "conduct alleged AGAINST THE PLAINTIFF (a slur, a race/sex/religion/age/"
        "disability-linked remark or joke, an unwanted sexual advance or "
        "touching). Answer NONE if the facts only use general labels like "
        "'harassment' or 'hostile work environment' WITHOUT describing any "
        "concrete incident directed at the plaintiff."
    ),
    "unwelcome_reaction": (
        "In <=15 words, state how the plaintiff showed that SAME trait-linked "
        "conduct was unwelcome (objected, reported it, told the harasser to "
        "stop, rebuffed/refused it, avoided them). Answer NONE if no such "
        "reaction to trait-linked conduct is described."
    ),
}

_NONE_LIKE = ("", "none", "n/a", "na", "not applicable", "not stated",
              "not mentioned", "no", "nothing", "unclear", "unknown")

# Bare doctrinal labels the extractor sometimes parrots back with no concrete
# content -- treat these as non-answers too.
_LABEL_ONLY = (
    "harassment", "hostile work environment", "discrimination",
    "unwelcome conduct", "unwelcome", "sexual harassment", "harassed",
)

# Code-side hedge/severity signal: the original baseline's lexicon, widened a
# little to also catch some of the false-negative vocabulary (derogatory,
# lewd, mimicry, slurs) without trying to replicate the LLM's judgment.
_KWS = (
    'harass', 'slur', 'unwelcome', 'unwanted', 'offensive comment',
    'sexual advance', 'objected', 'reported the conduct', 'told him to stop',
    'told her to stop', 'complained about', 'derogatory',
    'inappropriate comment', 'lewd', 'obscene', 'demeaning', 'ridicule',
    'mocking', 'mimic', 'imitat', 'racial slur', 'n-word', 'groped', 'fondled',
)


def _clamp(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _present(raw):
    """True if the extractor named something concrete (not a NONE-like
    non-answer and not just the doctrinal label parroted back)."""
    s = (raw or "").strip().strip(".").lower()
    if s in _NONE_LIKE:
        return False
    if s in _LABEL_ONLY:
        return False
    return len(s) >= 3


def score(text, extracted, ops):
    try:
        try:
            t = ops.normalize(text)
        except Exception:
            t = text or ""
        tl = t.lower()

        # ---- code channel: lexical density of harassment-type vocabulary,
        # kept as a hedge/severity tie-breaker (this is what the baseline used).
        hits = sum(1 for k in _KWS if k in tl)
        code_signal = _clamp(hits / 4.0)

        # ---- LLM channel: is the CONSTRUCT itself present -- concrete
        # trait-linked conduct, AND a shown unwelcome reaction to it.
        ext = extracted or {}
        has_conduct = _present(ext.get("trait_conduct", ""))
        has_unwelcome = _present(ext.get("unwelcome_reaction", ""))
        llm_signal = 0.5 * has_conduct + 0.5 * has_unwelcome

        if not has_conduct and not has_unwelcome:
            # No usable construct from the extractor -- either it is broken,
            # or (the informative case) there is genuinely nothing there. A
            # bare keyword hit with no confirmed construct is exactly the
            # baseline's false-positive failure mode, so discount it hard
            # rather than trusting it at face value.
            return _clamp(0.35 * code_signal)

        return _clamp(0.65 * llm_signal + 0.35 * code_signal)
    except Exception:
        return 0.5
