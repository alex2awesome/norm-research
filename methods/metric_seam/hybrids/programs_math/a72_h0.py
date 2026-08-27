"""a72 'Appropriate and economical proof technique' - hybrid channel.

Criterion: reward natural, simple methods suited to the claim; penalize
overkill, clumsy casework, gratuitous machinery, or misguided
symbol-pushing/tedious computation.

Residual analysis of the 30 train examples (baseline_train_rho was only
0.09) suggested two things a regex/LaTeX parser genuinely cannot see, which
the judge nonetheless seems to key on:

  (a) whether the METHOD CHOSEN actually fits the claim, vs. overkill /
      off-topic machinery / misguided symbol-pushing (e.g. an unrelated
      multinomial-distribution derivation dropped onto a simple
      range-mapping question scored 0.0, while a long-but-careful p-adic
      valuation argument, genuinely appropriate for its domain, scored
      1.0 despite being one of the longest answers in the set); and
  (b) whether the answer actually CARRIES OUT a full proof, vs. being a
      bare correction / definitional aside / list of unfinished
      "observations" -- several short, non-overkill answers that never
      complete a proof cluster at 0.15-0.2 rather than high.

Both are semantic/tacit judgments, so they are declared as LLM_FIELDS
(read against the ANSWER only -- the corpus interleaves the asker's own,
sometimes casework-heavy, question text, which must not be mistaken for
the answerer's technique). The code predicate maps the two short label
words onto anchor scores: completeness can only DISCOUNT a good technique
label, never rescue a bad one (a thorough answer to the wrong problem
gets no credit for its thoroughness). A small set of code-only nudges
(equation density/length via equation_stats, case-marker count via
proof_skeleton) act as a bounded safety net for when the LLM field comes
back empty or off-vocabulary, plus a hard override for the
math-free/near-empty "non-answer" pattern.
"""

import re

LLM_FIELDS = {
    "technique_label": (
        "Read the ANSWER only (ignore the question). In one word, "
        "describe its proof technique: elegant, appropriate, overkill, "
        "tedious, mismatched, or vacuous."
    ),
    "completeness_label": (
        "Read the ANSWER only. In one word: complete, partial, or none -- "
        "does it carry out a full proof, a partial/hint-only argument, "
        "or no proof at all?"
    ),
}

_TECH_BUCKETS = [
    ("vacuous", ("vacuous", "no proof", "not a proof", "spam", "nonsense",
                 "irrelevant answer", "non-answer", "nonanswer",
                 "empty answer")),
    ("mismatched", ("mismatch", "off-topic", "off topic", "unrelated",
                    "wrong problem", "doesn't address", "does not address",
                    "irrelevant")),
    ("tedious", ("tedious", "heavy computation", "brute force", "laborious",
                 "excessive computation", "symbol-pushing",
                 "symbol pushing")),
    ("overkill", ("overkill", "unnecessar", "convoluted", "clumsy",
                  "gratuitous", "overcomplicat", "too complex",
                  "heavy machinery")),
    ("appropriate", ("appropriate", "natural", "suited", "fitting",
                      "reasonable", "proper", "well-matched",
                      "well matched")),
    ("elegant", ("elegant", "simple", "clean", "direct", "concise",
                 "economical", "straightforward", "trivial", "minimal",
                 "short proof")),
]

_COMPLETE_BUCKETS = [
    ("none", ("none", "no proof", "not a proof", "empty", "nothing")),
    ("partial", ("partial", "hint", "outline", "sketch", "incomplete")),
    ("complete", ("complete", "full", "entire", "finished", "yes")),
]

_TECH_ANCHOR = {
    "vacuous": 0.05,
    "mismatched": 0.07,
    "tedious": 0.18,
    "overkill": 0.28,
    "appropriate": 0.80,
    "elegant": 0.95,
}
_GOOD_TECH = {"appropriate", "elegant"}


def _classify(s, buckets, default=None):
    s = (s or "").strip().lower()
    if not s:
        return default
    for label, keywords in buckets:
        for kw in keywords:
            if kw in s:
                return label
    return default


def _split_answer(text):
    # Corpus format is consistently "Question: ...\n\nAnswer: ...".
    # Score the ANSWER's technique, not the (sometimes casework-heavy,
    # sometimes off-topic) question the asker wrote.
    m = re.search(r"\bAnswer\s*:\s*", text, flags=re.IGNORECASE)
    if m:
        return text[m.end():]
    return text


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = text or ""
        if len(t.strip()) < 10:
            return 0.0
        try:
            t = ops.normalize(t)
        except Exception:
            pass
        answer = _split_answer(t)
        if len(answer.strip()) < 5:
            answer = t

        ext = extracted or {}
        tech = _classify(ext.get("technique_label", ""), _TECH_BUCKETS)
        comp = _classify(ext.get("completeness_label", ""), _COMPLETE_BUCKETS)

        if tech is None:
            # No usable technique signal -> fall back to completeness
            # alone, kept close to neutral since it's weak evidence on
            # its own.
            if comp == "complete":
                llm_score = 0.55
            elif comp == "partial":
                llm_score = 0.40
            elif comp == "none":
                llm_score = 0.15
            else:
                llm_score = 0.5
        else:
            anchor = _TECH_ANCHOR[tech]
            if tech in _GOOD_TECH:
                if comp == "complete":
                    llm_score = anchor
                elif comp == "partial":
                    llm_score = anchor * 0.30
                elif comp == "none":
                    llm_score = 0.08
                else:
                    llm_score = anchor * 0.70
            else:
                # Bad technique label: completeness cannot rescue it, but
                # "definitely not even a proof" pushes it lower still.
                llm_score = anchor
                if comp == "none":
                    llm_score = min(llm_score, 0.08)

        # --- code safety-net nudges (small, bounded; never the primary
        # signal -- see module docstring for why) ---
        nudge = 0.0
        try:
            n_words = len(answer.split())
        except Exception:
            n_words = 0
        try:
            n_disp, n_inl, _n_num, avg_tok = ops.equation_stats(answer)
        except Exception:
            n_disp = n_inl = 0
            avg_tok = 0.0
        n_spans = (n_disp or 0) + (n_inl or 0)

        if n_words < 15 and n_spans == 0:
            # Near-empty, math-free "answer" -> almost certainly a
            # non-answer regardless of what the LLM field returned.
            nudge -= 0.35
        try:
            skeleton = ops.proof_skeleton(answer)
            n_cases = len(skeleton.get("case", []))
        except Exception:
            n_cases = 0
        if n_cases > 4:
            nudge -= 0.05
        if n_spans > 10 and (avg_tok or 0) > 20:
            nudge -= 0.05

        return max(0.0, min(1.0, llm_score + nudge))
    except Exception:
        return 0.5
