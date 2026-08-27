"""
Hybrid metric channel for a114: "Metacognitive monitoring and fallacy detection"
(Math StackExchange answers).

Design rationale (see pack a114.json train examples):
  - The v0_keyword baseline (rho=0.192) fires on generic hedge words ("note
    that", "verify", "careful") that are usually just math vocabulary or the
    PROBLEM's own request ("verify the identity") -- not evidence the ANSWER
    itself narrates a check. E.g. d01742 is a from-scratch "verify the
    identity" derivation (baseline=0.6 from keyword "verify") but judge=0.25;
    d00019 hits "note that" (baseline=0.2) but judge=0.
  - High-judge answers (0.95-1.0) instead explicitly name a specific claim/
    step/attempt and give an epistemic verdict on it: "Your proof is
    correct" (d01052/d01366/d00012), "there are three main logical gaps"
    (d00012), "your original calculation contained two errors" (d02343),
    "Wikipedia's definition erroneously says..." (d00719), "you cannot use
    associative law...however you can simplify using..." (d04291), or an
    explicit necessity/redundancy audit ("X implies Y, so we can remove Y",
    d01122) / exhaustiveness check ("we need to prove there are no others",
    d04416; "this rules out the choice...", d00671).
  - A clean, correct, from-scratch derivation with NO such narrated audit
    (most 0.0-0.3 examples) scores low even though the math itself is fine
    -- the criterion rewards narrated metacognition, not raw correctness.

So: keep the predicate in code (precise phrase-level detector, restricted to
the ANSWER half of the text so it can't be triggered by the QUESTION's own
"verify/prove" framing), and use two LLM fields for the THICK-INPUT part a
regex can't see reliably: whether a specific verdict-with-reason is actually
given, and whether an independent cross-check/alternate method is used.
"""

import re

LLM_FIELDS = {
    "verdict_reason": "State the specific correctness verdict the answer gives about a claim or step (correct, wrong, or a named error) and why.",
    "alt_check": "Name the independent method, alternate approach, or cross-check the answer uses to confirm its result.",
}

_EMPTY_TOKENS = {
    "", "none", "n/a", "na", "nothing", "no", "not applicable", "unspecified",
    "not stated", "unclear",
}

# High-precision phrases for narrated verification / fallacy-audit language.
# Deliberately excludes the baseline's generic hedge words ("note that",
# "careful", "subtle", "sanity check", "plausibility") which proved to be
# false positives (d00019) or problem-statement leakage (d01742's "verify").
_VERDICT_RE = re.compile(
    r"\b("
    r"correct\w*|incorrect\w*|"
    r"confirm\w*|"
    r"mistakes?|errors?|erroneous\w*|flaws?|fallac\w*|gaps?|"
    r"rules?\s+out|no\s+others?|"
    r"cannot\s+(?:use|apply)|"
    r"redundant|implie[ds]|"
    r"depends?\s+on|"
    r"another\s+approach|independent\w*|converse|"
    r"double[- ]check\w*|cross[- ]check\w*|"
    r"need(?:s|ed)?\s+to\s+(?:also\s+)?(?:prove|show|verify|check)"
    r")\b",
    re.IGNORECASE,
)

_ANSWER_SPLIT_RE = re.compile(r"\banswer\s*:\s*", re.IGNORECASE)


def _answer_only(norm_text):
    parts = _ANSWER_SPLIT_RE.split(norm_text, maxsplit=1)
    if len(parts) == 2 and parts[1].strip():
        return parts[1]
    return norm_text


def _clean_field(v):
    if not isinstance(v, str):
        return ""
    v = v.strip()
    if v.lower() in _EMPTY_TOKENS:
        return ""
    return v


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text if isinstance(text, str) else str(text)
        try:
            norm = ops.normalize(raw)
        except Exception:
            norm = raw

        ans = _answer_only(norm)
        if not ans or not ans.strip():
            ans = norm

        # --- code predicate: narrated verdict / audit-language density,
        # restricted to the answer half so problem-statement words like
        # "verify the identity" (in the QUESTION) can't leak in.
        hits = len(_VERDICT_RE.findall(ans))
        if hits <= 0:
            code_verdict = 0.0
        elif hits == 1:
            code_verdict = 0.6
        else:
            code_verdict = 1.0

        # --- code predicate: explicit case-split / contrastive-connective
        # density, from the math-aware proof_skeleton op. This corpus does
        # its exhaustiveness/necessity audits ("but now consider...", "the
        # other choice works analogously") via case-splits more than via
        # a labeled "Case 1/Case 2" convention.
        case_bonus = 0.0
        connective_bonus = 0.0
        try:
            skel = ops.proof_skeleton(ans) or {}
            n_case = len(skel.get("case", []) or [])
            n_conn = len(skel.get("connective", []) or [])
            if n_case >= 2:
                case_bonus = 0.10
            elif n_case == 1:
                case_bonus = 0.05
            try:
                n_sent, _, _ = ops.sent_stats(ans)
            except Exception:
                n_sent = ans.count(".") + ans.count("!") + ans.count("?")
            n_sent = max(1, n_sent)
            connective_bonus = min(0.05, (n_conn / n_sent) * 0.2)
        except Exception:
            pass

        # --- LLM-grounded verdict: a mid-size reader can tell us whether a
        # concrete verdict-with-reason is actually present (catches
        # paraphrases the regex misses, e.g. "Wikipedia's definition
        # erroneously says..." style corrections of a non-asker source, or
        # implicit corrections of a miscomputed number).
        verdict_txt = _clean_field(
            extracted.get("verdict_reason", "") if isinstance(extracted, dict) else ""
        )
        alt_txt = _clean_field(
            extracted.get("alt_check", "") if isinstance(extracted, dict) else ""
        )

        if not verdict_txt:
            llm_verdict = 0.0
        else:
            n_words = len(verdict_txt.split())
            llm_verdict = 1.0 if n_words >= 4 else 0.5

        llm_alt = 1.0 if alt_txt else 0.0

        s = (
            0.05
            + 0.28 * code_verdict
            + 0.32 * llm_verdict
            + 0.15 * llm_alt
            + case_bonus
            + connective_bonus
        )
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
