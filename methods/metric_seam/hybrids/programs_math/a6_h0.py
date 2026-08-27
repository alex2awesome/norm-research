"""
Hybrid metric for a6: "Concision and focus without loss of clarity"
(Math StackExchange answers).

Reading of the 30 train examples: raw length / filler-word / sentence-length
signals (the description-compiled baseline, train rho=0.245) do NOT track the
judge well, because in math exposition "concise" does not mean "short" --
a long, dense, no-filler proof (e.g. a multi-step C*-algebra argument) can
still score well (~0.7), while a SHORT answer that dodges the actual question
or pads itself with meta-commentary/apologies/postscripts scores near zero.
The judge appears to reward three things the baseline misses:
  1. FOCUS: does the answer actually resolve the specific question asked, or
     does it digress/evade (a parser can't see this -- it's a semantic
     judgment, so this is the one thing we push to the LLM extractor).
  2. Absence of accretion/hedge markers (postscripts, "Edit:"/"Update:",
     apologies, "I have no idea...") -- structural, catchable in code.
  3. Absence of REDUNDANT restatement of the question's own wording/notation
     inside the answer (economy = don't re-derive what was already given).
  4. Math notation carrying the argument (dense, LaTeX-aware token volume
     per prose word) rather than long expository prose -- this is why we
     need the LaTeX-aware ops instead of raw regex.
"""

import re

LLM_FIELDS = {
    "off_topic_evidence": (
        "Quote the shortest exact phrase (<=15 words) showing the answer "
        "evades, refuses, or fails to address the specific question asked; "
        "else answer NONE."
    ),
}

_HEDGE_SUBSTRINGS = [
    "p.s.", "p.p.s", "pps", "sorry", "i apologize", "unfortunately i",
    "i don't have enough reputation", "i do not have enough reputation",
    "i have no idea", "just kidding", "not sure if this helps",
    "i'm not sure if", "im not sure if",
]
# word-boundary patterns for revision-accretion markers that rarely carry a
# trailing colon once scraped (e.g. "Edit For the evaluation...",
# "Update Concerning..."); "eta"/"note" deliberately excluded since they
# collide with common math variable names (eta) / normal prose (note that).
_HEDGE_PATTERNS = [re.compile(r"\bedit\b", re.I), re.compile(r"\bupdate[sd]?\b", re.I)]

_NEG_NOTES = {"", "none", "n/a", "na", "no", "none.", "no.", "n/a."}


def _sat(x, k=1.0):
    try:
        x = float(x)
    except Exception:
        return 0.0
    return x / (x + k) if x > 0 else 0.0


def _split_answer(text):
    """Best-effort isolation of the answer body from the Q/A pair.
    Falls back to the full text if no 'Answer:' marker is found."""
    parts = re.split(r"\bAnswer\s*:\s*", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", text


def _word_tokens(text):
    return re.findall(r"[A-Za-z]{2,}|\\[A-Za-z]+", text.lower())


def _ngrams(tokens, n=4):
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        try:
            t = ops.normalize(text)
        except Exception:
            t = text

        q_text, a_text = _split_answer(t)
        if not a_text.strip():
            a_text = t

        a_words = re.findall(r"[A-Za-z]+", a_text)
        nw = max(len(a_words), 1)
        a_lower = a_text.lower()

        # --- 1. focus gate (semantic; from LLM field) ---
        gate = 1.0
        try:
            note = ""
            if isinstance(extracted, dict):
                note = str(extracted.get("off_topic_evidence", "") or "")
            note_norm = note.strip().strip(".").lower()
            if note_norm not in _NEG_NOTES:
                gate = 0.15
        except Exception:
            gate = 1.0

        # --- 2. hedge / accretion markers (postscripts, edits, apologies) ---
        try:
            hedge_hits = sum(a_lower.count(p) for p in _HEDGE_SUBSTRINGS)
            hedge_hits += sum(len(p.findall(a_text)) for p in _HEDGE_PATTERNS)
            hedge_hits += 0.5 * a_lower.count("[...]")
            digress_rate = hedge_hits / max(nw / 200.0, 1.0)
            sig_digress = 1.0 - _sat(digress_rate, 1.0)
        except Exception:
            sig_digress = 0.7

        # --- 3. redundant restatement of the question's own wording ---
        try:
            q_tokens = _word_tokens(q_text)
            a_tokens = _word_tokens(a_text)
            q_grams = set(_ngrams(q_tokens, 4))
            a_grams = _ngrams(a_tokens, 4)
            if a_grams:
                overlap = sum(1 for g in a_grams if g in q_grams) / len(a_grams)
            else:
                overlap = 0.0
            sig_restate = 1.0 - _sat(overlap, 0.12)
        except Exception:
            sig_restate = 0.7

        # --- 4. math notation density (LaTeX-aware; not raw regex) ---
        try:
            n_disp, n_inl, n_num, avg_tok = ops.equation_stats(a_text)
            total_math = (n_disp + n_inl) * avg_tok
            density = total_math / nw
            sig_density = _sat(density, 0.35)
        except Exception:
            sig_density = 0.5

        # --- 5. mild run-on-sentence penalty ---
        try:
            n_sent, mean_wps, frac_long = ops.sent_stats(a_text)
            run_on_excess = max(0.0, mean_wps - 35.0)
            sig_sent = 1.0 - min(1.0, run_on_excess / 35.0)
        except Exception:
            sig_sent = 0.7

        base = (
            0.35 * sig_digress
            + 0.30 * sig_restate
            + 0.25 * sig_density
            + 0.10 * sig_sent
        )
        ungated = 0.05 + 0.90 * base
        final = gate * ungated
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
