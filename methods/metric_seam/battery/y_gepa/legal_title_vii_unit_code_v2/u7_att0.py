import re
import math
from collections import Counter

def score(text: str) -> float:
    """
    Scores a U.S. Title VII employment-discrimination court opinion on the 
    "Genuine Dispute of Material Fact" factor (0-10 scale).

    A high score indicates that the court recites specific facts showing the 
    plaintiff has presented sufficient evidence to create a genuine dispute 
    of material fact for a jury. A low score indicates the court found the 
    facts undisputed, undeveloped, or conclusory.
    """
    if not text or not text.strip():
        return 0.5

    try:
        text_lower = text.lower()
        word_count = len(text.split())
        if word_count < 50:
            return 0.5
        if word_count > 15000:
            text_lower = text_lower[:15000]

        # --- 1. Evidence & Fact Presentation Signals ---
        evidence_words = [
            'evidence', 'testimony', 'deposition', 'sworn', 'affidavit', 
            'exhibit', 'documented', 'witness', 'declaration', 'email', 
            'records', 'demonstrated', 'submitted', 'admissible', 'timestamps'
        ]
        specific_words = [
            'specifically', 'instance', 'instances', 'example', 'examples', 
            'particular', 'specifics', 'details', 'comment', 'comments', 
            'remark', 'remarks', 'slur', 'joke', 'jokes', 'incident', 
            'occurred', 'dates', 'statistics'
        ]

        evidence_counts = [text_lower.count(w) for w in evidence_words]
        evidence_matches = sum(1 for c in evidence_counts if c > 0)
        evidence_density = sum(evidence_counts) / (word_count / 1000.0)

        specific_counts = [text_lower.count(w) for w in specific_words]
        specific_matches = sum(1 for c in specific_counts if c > 0)
        specific_density = sum(specific_counts) / (word_count / 1000.0)

        ev_score = min(2.0, (evidence_matches * 0.2) + (evidence_density * 0.15))
        sp_score = min(2.0, (specific_matches * 0.2) + (specific_density * 0.15))

        # --- 2. Procedural Posture Signals ---
        # Signal that a summary judgment motion is at issue
        sj_terms = ['summary judgment', 'celotex', 'anderson v', 'rule 56', 'moving party']
        sj_term_hits = sum(1 for term in sj_terms if term in text_lower)
        sj_signal = 1.0 if sj_term_hits >= 2 else (0.5 if sj_term_hits == 1 else 0.0)

        # Signal that the court is viewing facts favorably to the non-movant (plaintiff)
        pov_terms = [
            'in the light most favorable', 'light most favorable', 
            'reasonable jury', 'reasonable inference', 'genuine dispute', 
            'material fact'
        ]
        pov_hits = sum(1 for term in pov_terms if term in text_lower)
        pov_signal = min(2.0, pov_hits * 0.4)

        # --- 3. Verdict & Disposition Signals ---
        # Plaintiff win signals (surviving summary judgment)
        deny_words = ['denied', 'deny']
        survive_words = ['survives', 'survive', 'withstand', 'proceed to trial', 'jury could find']
        
        deny_hits = sum(text_lower.count(w) for w in deny_words)
        survive_hits = sum(text_lower.count(w) for w in survive_words)

        if survive_hits > 0 or deny_hits > 0:
            verdict_signal = 2.0
        else:
            verdict_signal = 0.0

        # Plaintiff loss signals (granting summary judgment)
        grant_words = ['granted', 'grant', 'dismiss']
        lose_phrases = [
            'no genuine dispute', 'fails to establish', 'fails to show', 
            'fails to demonstrate', 'no evidence', 'undisputed', 'fails to raise', 
            'scant evidence', 'mere speculation', 'conclusory', 'not enough evidence', 
            'uncontested', 'fails to produce'
        ]

        grant_hits = sum(text_lower.count(w) for w in grant_words)
        lose_phrase_hits = sum(text_lower.count(w) for w in lose_phrases)

        if grant_hits > 0 and lose_phrase_hits > 0:
            verdict_signal = 0.0
        elif lose_phrase_hits >= 2:
            verdict_signal = 1.0
        elif lose_phrase_hits == 1:
            verdict_signal = 1.5

        # --- 4. Combine and Scale ---
        raw_score = ev_score + sp_score + sj_signal + pov_signal + verdict_signal
        
        # Map raw score (typically 0 to 8) to a 0.0 - 10.0 scale
        final_score = max(0.0, min(10.0, raw_score * 1.25))
        
        return final_score

    except Exception:
        return 0.5