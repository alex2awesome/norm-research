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
    facts undisputed, undeveloped, conclusory, or plaintiff failed to present evidence.
    """
    if not text or not text.strip():
        return 0.5

    try:
        text_lower = text.lower()
        word_count = len(text_lower.split())
        if word_count < 20:
            return 0.5
        if word_count > 15000:
            text_lower = text_lower[:15000]

        # --- 1. STRONG NEGATIVE SIGNALS (Suppressed or Nonexistent Evidence) ---
        neg_patterns = [
            r"failed\s+to\s+(?:present|submit|produce|provide|offer|come\s+forward\s+with|establish)",
            r"(?:no|without)\s+evidence",
            r"no\s+genuine\s+(?:issue|dispute)",
            r"unsupported\s+(?:by|conclusory|allegation)",
            r"(?:mere|bare|vague)\s+(?:conclusory|allegation|speculation)",
            r"conclusory",
            r"speculation",
            r"self[-\s]?serving",
            r"fails\s+to\s+establish",
            r"unsubstantiated",
            r"(?:fails|failed|fail)\s+to\s+show",
            r"scintilla",
            r"no\s+showing",
            r"lacks?\s+(?:any\s+)?evidence",
            r"undisputed",
            r"(?:failed|fails)\s+to\s+demonstrate",
            r"no\s+(?:specific|individualized)\s+evidence",
            r"not\s+(?:sufficient|enough)\s+evidence",
            r"(?:insufficient|inadequate)\s+(?:evidence|to)",
            r"points?\s+to\s+no\s+evidence",
            r"relies?\s+(?:solely|only)\s+on",
            r"concedes?\s+(?:that\s+)?(?:no|she\s+has\s+no|he\s+has\s+no|plaintiff)",
            r"admits?\s+(?:that\s+)?(?:she\s+has\s+no|he\s+has\s+no)",
            r"no\s+basis",
            r"fails?\s+to\s+identify",
            r"points?\s+to\s+no\s+(?:admissible|competent)\s+evidence",
            r"(?:fails|failed)\s+to\s+(?:point|direct)\s+the\s+court",
            r"nothing\s+in\s+the\s+record",
            r"plaintiff(?:'s)?\s+(?:offers?|provides?|presents?)\s+no"
        ]
        neg_count = sum(len(re.findall(p, text_lower)) for p in neg_patterns)
        neg_factor = min(5.5, neg_count * 0.45)

        # --- 2. STRONG POSITIVE SIGNALS (Denial of Summary Judgment) ---
        pos_patterns = [
            r"den(?:ying|y|ied)\s+(?:defendants?'?s?\s+)?(?:motion\s+for\s+)?summary\s+judgment",
            r"survives?\s+summary\s+judgment",
            r"genuine\s+dispute\s+of\s+material\s+fact",
            r"genuine\s+issue\s+of\s+material\s+fact",
            r"reasonable\s+jury\s+could",
            r"jury\s+could\s+reasonably",
            r"jury\s+could\s+find",
            r"materially\s+disputed",
            r"(?:sufficient|substantial)\s+(?:evidence|support)",
            r"credibility\s+determination",
            r"triable\s+issue",
            r"(?:resolve|weigh)\s+(?:the\s+)?evidence",
            r"question\s+of\s+fact",
            r"materially\s+genuine",
            r"draw\s+(?:all\s+)?(?:reasonable\s+)?inferences?",
            r"(?:disputes?|disputed|disputed!)\s+(?:the\s+)?(?:defendant|employer|movant)"
        ]
        pos_count = sum(len(re.findall(p, text_lower)) for p in pos_patterns)
        pos_factor = min(6.5, pos_count * 0.55)

        # --- 3. SPECIFIC EVIDENCE & DETAILED FACTS ---
        evidence_words = [
            'deposition', 'sworn', 'affidavit', 'exhibit', 'witness', 
            'declaration', 'records', 'statistics', 'testimony', 'timestamps', 'emails'
        ]
        specific_words = [
            'specifically', 'instance', 'particular', 'comment', 'remark', 
            'slur', 'joke', 'incident', 'occurred', 'dates', 'details', 'specifics', 'pattern'
        ]

        evidence_count = sum(text_lower.count(w) for w in evidence_words)
        specific_count = sum(text_lower.count(w) for w in specific_words)

        evidence_factor = min(2.5, (evidence_count / (word_count / 1000.0)) * 0.7)
        specific_factor = min(2.0, (specific_count / (word_count / 1000.0)) * 0.7)

        # --- 4. MOTION GRANTED SIGNAL ---
        granted_pattern = r"grant(?:ing|ed|s)?\s+(?:defendants?'?s?\s+)?(?:motion\s+for\s+)?summary\s+judgment"
        granted_count = len(re.findall(granted_pattern, text_lower))
        granted_factor = granted_count * 0.6

        # --- 5. PROCEDURAL CONTEXT ---
        sj_mentions = text_lower.count('summary judgment')
        proced_signal = min(1.5, sj_mentions * 0.15) if sj_mentions > 0 else 0

        # --- 6. POOR FACTUAL DEVELOPMENT SIGNALS ---
        underdeveloped_patterns = [
            r"conclusory",
            r"undeveloped",
            r"bare\s+assertion",
            r"vague",
            r"lack(?:s|ing)?\s+detail",
            r"without\s+(?:specifics|detail)",
            r"general",
            r"unsupported",
            r"single\s+(?:instance|comment|remark|incident)",
            r"(?:failed|fails)\s+to\s+(?:develop|elaborate|provide\s+details)"
        ]
        underdev_count = sum(len(re.findall(p, text_lower)) for p in underdeveloped_patterns)
        underdev_factor = min(3.0, underdev_count * 0.5)

        # --- 7. CONTEXT MODIFIERS ---
        time_refs = len(re.findall(r'\b(?:19[89]\d|20[0-2]\d)\b', text_lower))
        time_factor = min(1.5, time_refs * 0.06)

        names_pattern = r'\b[A-Z][a-z]+\s+(?:[A-Z]\.\s+)?[A-Z][a-z]+\b'
        names_count = len(set(re.findall(names_pattern, text)))
        names_factor = min(1.5, names_count * 0.035)

        # --- FINAL ASSEMBLY ---
        raw_score = 5.0 + pos_factor + evidence_factor + specific_factor + proced_signal + time_factor + names_factor
        raw_score -= (neg_factor + underdev_factor + granted_factor)

        # Clamp to 0-10 range
        final_score = max(0.0, min(10.0, raw_score))

        return final_score

    except Exception:
        return 0.5