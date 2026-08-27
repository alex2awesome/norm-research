"""
Hybrid metric channel for aspect a162: "Premise clarity and generative strength"
Criterion: State a clear, opinionated premise that naturally yields strong,
varied comedic situations and interactions.

Design rationale (derived only from the pack's contract + train examples):
- The frozen baseline looks for literal words like "premise"/"idea"/"imagine",
  which almost never appear in reddit-style jokes -- it effectively degenerates
  into length/punctuation/diversity noise (train_rho ~0.29, flat 0.4-0.6 band).
- Inspecting judge scores vs text: jokes the judge rewards tend to (a) build a
  stated scenario through (b) multiple dialogue turns / exchanges that (c) often
  escalate or repeat a pattern with variation (rule-of-three, parallel visits,
  parallel Q&A). Jokes the judge punishes tend to be single-note wordplay/puns
  (one setup -> one twist, no development), rambling meta-commentary with no
  stated scenario, or shock one-liners riding on topic/profanity rather than
  structure. Topic words and offensiveness are NOT scored directly (hazard note).
- Code can see surface dialogue/structure markers but cannot reliably judge
  whether a "premise" is actually stated and whether it is a single pun vs a
  developed scenario -- that discrimination is handed to the LLM fields as
  factual extractions (premise phrase, beat count), while the *predicate*
  (how those facts map to a score, and the anti-single-note-pun correction)
  stays in code.
"""

import re
import math
import statistics
from collections import Counter

LLM_FIELDS = {
    "premise": (
        "State this joke's core comedic premise/scenario in <=8 words; "
        "answer NONE if it is merely a single wordplay/pun setup with no "
        "distinct scenario."
    ),
    "beats": (
        "Answer a single digit 0-3 (3 means three or more) for how many "
        "distinct comedic beats/escalating exchanges develop from the "
        "premise; answer NONE if this cannot be determined."
    ),
}


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not isinstance(text, str):
            return 0.0

        try:
            norm = ops.normalize(text)
            if not norm:
                norm = text
        except Exception:
            norm = text

        tokens = re.findall(r"\b\w+\b", norm)
        nw = max(1, len(tokens))

        # ---------------- LLM-grounded facts ----------------
        premise_raw = (extracted.get("premise") or "").strip() if isinstance(extracted, dict) else ""
        beats_raw = (extracted.get("beats") or "").strip() if isinstance(extracted, dict) else ""

        premise_stated = bool(premise_raw) and premise_raw.upper() != "NONE"

        beats_digit = re.search(r"\d", beats_raw)
        llm_beats_n = int(beats_digit.group()) if beats_digit else None

        # ---------------- code-only structural signals ----------------
        # Dialogue turns: quoted speech spans + reporting verbs (interaction proxy).
        quotes = re.findall(r'["“”][^"“”]{2,300}["“”]', norm)
        n_quotes = len(quotes)
        report_verbs = len(re.findall(
            r"\b(said|says|say|saying|asked|asks|ask|replied|replies|reply|"
            r"answered|answers|answer|shouted|shouts|whispered|muttered|"
            r"exclaim(?:s|ed)?|responded|responds)\b",
            norm, re.I,
        ))
        dialogue_density = min(1.0, (n_quotes + 0.5 * report_verbs) / 6.0)

        # Paragraph segmentation as a coarse beat proxy (independent of LLM).
        paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        n_paras = len(paras)

        # Escalation / repetition markers: numbered or bulleted structure, or
        # explicit sequencing words that mark a repeated-with-variation pattern.
        list_markers = len(re.findall(r"(?m)^\s*(?:\d+[\.\)]|[-*])\s+", text))
        seq_words = len(re.findall(
            r"\b(first|second|third|next|then|finally|after that)\b", norm, re.I
        ))
        escalation = min(1.0, (list_markers + 0.34 * seq_words) / 3.0)

        try:
            n_sent, mean_wps, frac_long = ops.sent_stats(norm)
        except Exception:
            n_sent, mean_wps, frac_long = (0, 0.0, 0.0)

        # Code-only fallback beat estimate, used only if the LLM field is absent.
        code_beats_proxy = max(0.0, min(1.0, (n_paras - 1 + n_quotes) / 5.0))

        if llm_beats_n is not None:
            beats_score = min(1.0, llm_beats_n / 3.0)
        else:
            beats_score = code_beats_proxy

        # "Generative strength": does the premise develop into multiple,
        # interactive, escalating situations (vs. a single flat statement)?
        generative = 0.45 * beats_score + 0.35 * dialogue_density + 0.20 * escalation
        generative = max(0.0, min(1.0, generative))

        # "Premise clarity": trust the LLM's explicit judgment when it answers;
        # otherwise fall back to a coarse code proxy (multi-sentence / multi-
        # paragraph framing suggests *some* scenario is being set up).
        if premise_raw:
            premise_clarity = 1.0 if premise_stated else 0.0
        else:
            premise_clarity = 1.0 if (n_sent >= 2 or n_paras >= 2) else 0.4

        raw = 0.40 * premise_clarity + 0.60 * generative

        # Anti-pattern correction: short text with near-zero dialogue,
        # near-zero beat development, and no escalation structure is almost
        # always a single-note pun/riddle -- the criterion explicitly wants
        # "varied" situations, so a lone setup->twist should not score well
        # even though such texts are lexically clean (the baseline over-scores
        # exactly this class).
        if nw < 70 and dialogue_density < 0.25 and beats_score < 0.4 and escalation < 0.2:
            raw -= 0.22

        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
