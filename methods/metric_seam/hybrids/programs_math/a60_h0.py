"""
Hybrid metric for a60: "Lead with the main results" (Math StackExchange answers).

Design rationale (see reply for full rationale):
The judge rewards answers whose OPENING sentence(s) directly resolve the
asker's question (a verdict, computed value, or directly applicable fact),
and penalizes answers that open with a critique/restatement/question-back,
a bare hint with no resolution ever, or a long derivation whose payoff only
arrives at the very end. This is a semantic "where/whether is the punchline"
judgment a regex can't reliably make on its own -> one LLM_FIELD carries the
core signal. Code supplies: isolating the Answer segment, a parser-visible
early-connective check via ops.proof_skeleton (checkable predicate), and
deterministic regex overrides/fallbacks for robustness when the LLM field is
missing or degenerate.
"""

import re

LLM_FIELDS = {
    "lead_pattern": (
        "In the ANSWER portion only (ignore the Question), classify how it reaches "
        "the asker's core result. Reply exactly one word: IMMEDIATE (states the "
        "result/verdict in the first sentence), SHORT (reaches a clear result within "
        "a few brief steps), LONG (only reaches a result after lengthy setup or "
        "derivation), or NONE (never states a resolved result, only hints/suggestions)."
    ),
}

_ANSWER_SPLIT_RE = re.compile(r'(?:^|\n)\s*Answer\s*:\s*', re.IGNORECASE)

_DIRECT_RE = re.compile(
    r'\b(yes|no)\b\s*[,.]'
    r'|the (short|simple) answer is'
    r'|\bin short\b'
    r'|\bclearly\b'
    r'|\bindeed\b'
    r'|\btherefore\b'
    r'|\bthus\b'
    r'|\bhence\b'
    r'|\bactually\b'
    r'|your (proof|solution|argument|approach|attempt)\s+(is|looks|works)'
    r'|this is (correct|incorrect|wrong|right|false|true)'
    r'|the answer is\b'
    r'|\bnote that\b'
    r'|\bin fact\b',
    re.IGNORECASE,
)

_HEDGE_RE = re.compile(
    r'\bhint\b'
    r'|\btry\b'
    r'|\bconsider\b'
    r'|\bsuppose\b'
    r'|\bassume\b'
    r'|\bi think\b'
    r'|\bi suppose\b'
    r'|you (seem|appear|might|could|should|make a mistake|say)\b'
    r'|why (not|don.t)\b'
    r'|have you (tried|considered)\b'
    r'|what (if|about)\b'
    r'|instead of\b'
    r'|\bmaybe\b'
    r'|\bperhaps\b'
    r'|this reminds me\b'
    r'|one possibility\b',
    re.IGNORECASE,
)

_STRUCTURED_RE = re.compile(
    r'\bstep\s*1\b|\bobservation\s*1\b|^\s*1[).]|\bfirstly\b|\bfirst,',
    re.IGNORECASE,
)

_OPEN_WINDOW = 220

_BUCKET_KEYWORDS = (
    ("immediate", 0.85),
    ("none", 0.05),
    ("no result", 0.05),
    ("unresolved", 0.05),
    ("long", 0.18),
    ("short", 0.5),
)


def _code_open_score(opening: str):
    """Deterministic regex read of the opening window. Returns (score, has_question, direct_hit)."""
    has_question = '?' in opening
    direct_hit = bool(_DIRECT_RE.search(opening))
    hedge_hit = bool(_HEDGE_RE.search(opening))
    structured_hit = bool(_STRUCTURED_RE.search(opening))

    if direct_hit:
        s = 0.82
    elif has_question:
        s = 0.10
    elif hedge_hit:
        s = 0.25
    elif structured_hit:
        s = 0.42
    else:
        s = 0.45
    return s, has_question, direct_hit


def _proof_skeleton_early_bonus(answer_part: str, ops) -> float:
    """Parser-visible check: does a concluding connective / qed marker show up
    early relative to the length of the answer? Returns a value in [0,1],
    0.5 if unavailable/ambiguous (neutral)."""
    try:
        skel = ops.proof_skeleton(answer_part)
        if not isinstance(skel, dict):
            return 0.5
        n = max(len(answer_part), 1)
        offsets = []
        for cat in ("connective", "qed"):
            for item in skel.get(cat, []) or []:
                try:
                    off = item[0]
                    offsets.append(float(off))
                except Exception:
                    continue
        if not offsets:
            return 0.5
        rel = min(offsets) / n
        rel = max(0.0, min(1.0, rel))
        return 1.0 - rel
    except Exception:
        return 0.5


def _parse_lead_pattern(raw: str):
    if not raw:
        return None
    norm = re.sub(r'[^a-z]+', ' ', raw.lower()).strip()
    if not norm:
        return None
    for kw, val in _BUCKET_KEYWORDS:
        if kw in norm:
            return val
    return None


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text:
            return 0.5
        t = ops.normalize(text)
        if not t or not t.strip():
            return 0.5

        m = _ANSWER_SPLIT_RE.search(t)
        answer_part = t[m.end():] if m else t
        if not answer_part.strip():
            answer_part = t

        opening = answer_part[:_OPEN_WINDOW]

        code_score, has_question, direct_hit = _code_open_score(opening)
        early_bonus = _proof_skeleton_early_bonus(answer_part, ops)

        # blend the deterministic parse signals: mostly the opening read,
        # lightly nudged by the early-connective check.
        code_composite = 0.8 * code_score + 0.2 * early_bonus

        raw_field = ""
        try:
            raw_field = extracted.get("lead_pattern", "") if extracted else ""
        except Exception:
            raw_field = ""
        llm_score = _parse_lead_pattern(raw_field)

        if llm_score is None:
            final = code_composite
        else:
            final = 0.7 * llm_score + 0.3 * code_composite

        # deterministic overrides (parser-visible, high-confidence cues)
        if has_question and not direct_hit:
            final = min(final, 0.30)
        if direct_hit:
            final = max(final, 0.55)

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5


if __name__ == "__main__":
    # lightweight self-test with a stubbed `ops` and synthetic extracted values,
    # exercising the buckets described in the design rationale.
    class _StubOps:
        def normalize(self, s):
            return s

        def proof_skeleton(self, s):
            # naive stub: flag "therefore"/"thus"/"hence" occurrences as connectives
            out = {"connective": [], "qed": []}
            for kw in ("therefore", "thus", "hence", "so "):
                idx = s.lower().find(kw)
                if idx >= 0:
                    out["connective"].append((idx, kw))
            return out

    ops = _StubOps()

    cases = [
        ("Answer: You seem to be having trouble. Let me explain at length before I get anywhere near a conclusion...",
         {"lead_pattern": "LONG"}, "low (hedge opening, long buildup)"),
        ("Answer: Yes, coherent states are an overcomplete basis for the Hilbert space...",
         {"lead_pattern": "IMMEDIATE"}, "high (direct yes)"),
        ("Answer: Try f(n)=1 if n is even, f(n)=0 if n is odd, and g(n)=1-f(n).",
         {"lead_pattern": "IMMEDIATE"}, "high (concrete construction despite 'Try')"),
        ("Answer: Step 1: Assume U is irreducible... Step 2: show U1+U2 = V+V.",
         {"lead_pattern": "SHORT"}, "medium (compact structured proof)"),
        ("Answer: Observation 1: ... Observation 2: ... Combining these should give you a proof.",
         {"lead_pattern": "NONE"}, "low (hint-only, never resolves)"),
        ("Answer: You appear to have had the right idea, but handled it too specifically. [long algebra] ... the minimum is -6/17.",
         {"lead_pattern": "LONG"}, "low (buried result after long derivation)"),
    ]
    for txt, ext, note in cases:
        print(f"{score(txt, ext, ops):.3f}  {note}")
