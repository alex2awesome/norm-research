"""
Hybrid metric channel for a18: Expository clarity and readability (Math StackExchange).

Design rationale (see task rationale in reply):
- The naive keyword/connective baseline (train_rho=0.185) is fooled by keyword
  PRESENCE (e.g. "define", "note", paragraph count) even when the surrounding
  math is an unexplained equation-dump, or when a long, jargon-heavy answer
  merely cites a result. The judge instead rewards genuine prose-math
  interleaving (steps connected by reasoning words), penalizes "equation
  soup" (many math spans, ~no connecting prose) and malformed LaTeX, and
  penalizes answers that are complete but not truly *responsive* to the
  question (off-topic, argumentative, or under-explained for the asker's
  stated level) -- a tacit judgment a parser cannot see, hence one LLM field.
- All checkable, LaTeX-structural predicates (span/prose ratio, connective
  density normalized by length, sentence run-ons, delimiter health, token
  density) stay in code using the math-aware ops. Only the "does this answer
  actually explain itself to this asker" judgment is pushed to the LLM field.
"""

import re

LLM_FIELDS = {
    "expo_flaw": (
        "In <=12 words: the main flaw in how the ANSWER explains itself to the "
        "asker -- e.g. off-topic/answers a different question, no real explanation "
        "given, argues with the asker instead of helping, hand-wavy/skips justification "
        "for its steps, or too terse/dense/advanced for what the asker needed -- "
        "or NONE if the explanation is clear and complete."
    ),
}


def _safe_ratio(a, b):
    return a / b if b else 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.0

        t = ops.normalize(text)

        # Corpus convention is "Question: ... \n\nAnswer: ..."; judge the ANSWER's
        # exposition, not the (often messy/short) question prose.
        m = re.search(r"\bAnswer\s*:\s*", t, flags=re.IGNORECASE)
        body = t[m.end():] if m else t
        if not body.strip():
            body = t

        # --- math-aware structural signals ---
        try:
            spans = ops.extract_math_spans(body)
        except Exception:
            spans = []
        n_spans = len(spans)
        math_chars = sum(len(c) for _, c in spans) if spans else 0
        total_chars = max(1, len(body))
        prose_frac = max(0.0, 1.0 - math_chars / total_chars)

        try:
            n_disp, n_inl, n_num, avg_tok = ops.equation_stats(body)
        except Exception:
            n_disp, n_inl, n_num, avg_tok = (0, 0, 0, 0.0)

        try:
            dh = ops.delimiter_health(body)
            broken = sum(v for v in dh.values() if isinstance(v, (int, float)))
        except Exception:
            broken = 0

        try:
            skel = ops.proof_skeleton(body)
            n_connective = len(skel.get("connective", []))
            n_structure = sum(len(v) for k, v in skel.items() if k != "connective")
        except Exception:
            n_connective = 0
            n_structure = 0

        try:
            n_sent, mean_wps, frac_long = ops.sent_stats(body)
        except Exception:
            n_sent, mean_wps, frac_long = (0, 0.0, 0.0)

        # 1) Prose-to-math balance: penalize "equation soup" -- several math
        #    spans with almost no connecting prose between them. Do NOT
        #    penalize short, clean, mostly-prose answers (n_spans small).
        if n_spans >= 2 and prose_frac < 0.25:
            s_balance = 0.15
        elif n_spans >= 1 and prose_frac < 0.40 and n_connective == 0:
            s_balance = 0.40
        else:
            s_balance = 1.0

        # 2) Connective/structure density, saturating and normalized by
        #    sentence count so short-but-clear answers aren't punished for
        #    having few connectives (only reward when they're *missing*
        #    relative to how much text there is).
        expected_conn = max(1.0, n_sent / 3.0)
        conn_ratio = _safe_ratio(n_connective + 0.5 * n_structure, expected_conn)
        s_connect = min(1.0, 0.5 + 0.5 * conn_ratio) if n_sent > 0 else 0.5

        # 3) Sentence-length moderation: penalize run-ons, not brevity.
        if mean_wps <= 0:
            s_len = 0.5
        else:
            s_len = max(0.0, 1.0 - max(0.0, mean_wps - 30.0) / 40.0)
        s_runon = max(0.0, 1.0 - frac_long * 1.5)

        # 4) LaTeX health: broken markup reads as confusing, not clear.
        s_health = max(0.0, 1.0 - min(1.0, broken / 4.0))

        # 5) Density guard: many tokens/span with little prose = "not
        #    distilled" jargon even if some connectives are present.
        density_penalty = 0.15 if (avg_tok and avg_tok > 25 and prose_frac < 0.5) else 0.0

        code_score = (
            0.30 * s_balance
            + 0.25 * s_connect
            + 0.15 * s_len
            + 0.10 * s_runon
            + 0.20 * s_health
        ) - density_penalty
        code_score = max(0.0, min(1.0, code_score))

        # --- LLM grounding: catches non-responsiveness / audience mismatch
        #     that no parser predicate can see. ---
        flaw = (extracted.get("expo_flaw") or "").strip().lower()
        penalty = 0.0
        if flaw and flaw not in ("none", "no flaw", "n/a", "none.", "no", "none found"):
            if any(k in flaw for k in ("off-topic", "off topic", "unrelated", "different question", "doesn't answer", "does not answer")):
                penalty = 0.45
            elif any(k in flaw for k in ("argu", "critiq", "challeng", "refus")):
                penalty = 0.35
            elif any(k in flaw for k in ("no explanation", "no-explanation", "no reasoning", "just states", "asserts", "unexplained")):
                penalty = 0.30
            elif any(k in flaw for k in ("hand-wav", "hand wav", "skip", "elide", "elision", "unjustified", "incomplete", "outline")):
                penalty = 0.25
            elif any(k in flaw for k in ("terse", "too short", "too advanced", "too brief", "too dense", "too complex", "under-explain", "audience", "mismatch", "hard to follow")):
                penalty = 0.20
            else:
                penalty = 0.15

        final = code_score * (1.0 - penalty)
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
