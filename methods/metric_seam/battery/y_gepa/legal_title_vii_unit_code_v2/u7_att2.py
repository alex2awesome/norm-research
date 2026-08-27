import re
import math

def score(text: str) -> float:
    """
    Scores a U.S. Title VII employment-discrimination court opinion on the 
    "Genuine Dispute of Material Fact" factor (0-10 scale).

    High score: court recites specific facts showing plaintiff presented sufficient 
    evidence to create a genuine dispute of material fact for a jury.
    Low score: facts undisputed, undeveloped, conclusory, or plaintiff failed.
    """
    if not text or not text.strip():
        return 0.5

    try:
        text_lower = text.lower()
        word_count = len(text_lower.split())
        if word_count < 20:
            return 0.5
        if word_count > 20000:
            text_lower = text_lower[:20000]

        # --- 1. PROCEDURAL POSTURE (Factual recitation vs. analysis) ---
        fact_indicators = [
            r"deposition",
            r"declared?\s+under\s+penalty",
            r"exhibit",
            r"affidavit",
            r"interrogator",
            r"e\.e\.o\.c",
            r"personnel\s+file",
            r"time\s+records",
            r"pay\s+records",
            r"performance\s+review",
            r"performance\s+eval",
            r"record\s+(?:evidence|cite|cites)",
            r"submit(?:ted|s|ting)?\s+(?:in\s+support|evidence|a\s+declarat)",
            r"evidence\s+(?:in\s+the\s+record|shows?|indicates?|demonstrates?)",
            r"(?:cites?|points?\s+to)\s+(?:the\s+)?record",
            r"job\s+(?:description|posting|title)",
            r"incident\s+report",
            r"email",
            r"memo(?:randum)?",
            r"complain(?:ed|s)\s+to\s+(?:hr|human\s+resources|supervisor|manager)",
            r"internal\s+complaint",
            r"charge\s+of\s+discrimination"
        ]
        fact_score = sum(min(2, len(re.findall(p, text_lower))) for p in fact_indicators)

        # --- 2. POSITIVE SIGNALS ---
        pos_patterns = [
            (r"genuine\s+(?:issue|dispute)\s+of\s+material\s+fact", 3),
            (r"(?:jury|triable)\s+(?:question|issue)", 2),
            (r"reasonable\s+(?:jury|fact[-\s]?finder|trier\s+of\s+fact)", 2),
            (r"could\s+(?:infer|conclude|find|determine)", 2),
            (r"(?:drawing|draw)\s+all\s+(?:reasonable\s+)?inferences", 2),
            (r"credibility\s+(?:determination|issue|matter|dispute|question)", 2),
            (r"materially\s+disputed", 2),
            (r"disputed?\s+issue", 2),
            (r"issue\s+of\s+fact", 2),
            (r"pretext", 1),
            (r"conflicting\s+(?:evidence|testimony|accounts|statements)", 1),
            (r"direct\s+evidence", 2),
            (r"circumstantial\s+evidence", 1),
            (r"comparator", 1),
            (r"similarly\s+situated", 1),
            (r"discriminatory\s+(?:motive|intent|animus|reason)", 1),
            (r"but[-\s]?for\s+(?:cause|the\s+adverse)", 1),
            (r"protected\s+activity", 1),
            (r"adverse\s+(?:employment\s+)?action", 1),
            (r"prima\s+facie", 1),
            (r"burden\s+shifting", 1),
            (r"sham", 1),
            (r"inconsistent(?:ly)?\s+(?:stated|reasons|explanation)", 1),
            (r"shifting\s+(?:reasons|explanation|justification)", 1),
            (r"temporal\s+proximity", 1),
            (r"causal\s+(?:connection|link|nexus)", 1),
            (r"hostile\s+(?:work|environment)", 1),
            (r"severe\s+(?:or|and)\s+pervasive", 1),
            (r"deny(?:s|ing)?\s+(?:the\s+)?(?:allegation|accusation|charge|claim)", 1),
            (r"disput(?:e|es|ed|ing)\s+(?:defendant|employer)", 1),
            (r"plaintiff(?:'s)?\s+(?:testimony|evidence|version|account)", 1),
        ]
        pos_score = sum(w * min(3, len(re.findall(p, text_lower))) for p, w in pos_patterns)

        # --- 3. NEGATIVE SIGNALS ---
        neg_patterns = [
            (r"(?:no|without)\s+(?:genuine|material)\s+(?:dispute|issue)", 4),
            (r"(?:no|without)\s+evidence", 4),
            (r"no\s+genuine\s+(?:issue|dispute)", 4),
            (r"(?:failed|fails)\s+to\s+(?:present|submit|produce|provide|offer|come\s+forth|point\s+to|identify|show|demonstrate|establish)", 4),
            (r"no\s+(?:specific|individualized|competent|admissible|direct|circumstantial)\s+evidence", 4),
            (r"unsupported(?:\s+by\s+(?:any|the)\s+evidence)?", 3),
            (r"(?:conclusory|conclusory\s+allegations?)", 3),
            (r"(?:mere|bare|vague)\s+(?:conclusory|allegation|speculation)", 3),
            (r"speculation", 2),
            (r"self[-\s]?serving", 2),
            (r"unsubstantiated", 2),
            (r"scintilla", 3),
            (r"no\s+showing", 3),
            (r"lacks?\s+(?:any\s+)?evidence", 3),
            (r"undisputed", 3),
            (r"(?:insufficient|inadequate)\s+(?:evidence|to)", 3),
            (r"(?:nothing|none)\s+in\s+the\s+record", 3),
            (r"points?\s+to\s+no\s+evidence", 3),
            (r"(?:relies?|relying)\s+(?:solely|only|merely)\s+on", 3),
            (r"(?:failed|fails)\s+to\s+(?:point|direct)\s+(?:the\s+)?court", 3),
            (r"conced(?:es?|ed)\s+(?:that\s+)?(?:no|she\s+(?:has\s+)?no|he\s+(?:has\s+)?no|plaintiff\s+(?:has\s+)?no)", 3),
            (r"admit(?:s|ted)\s+(?:that\s+)?(?:no|she\s+(?:has\s+)?no|he\s+(?:has\s+)?no|plaintiff)", 3),
            (r"no\s+basis", 2),
            (r"(?:plaintiff|claimant|appellant)\s+(?:offers?|provides?|presents?)\s+no", 3),
            (r"fails?\s+as\s+a\s+matter\s+of\s+law", 3),
            (r"summary\s+judgment\s+(?:is|shall\s+be|must\s+be|will\s+be|should\s+be)\s+(?:granted|appropriate|entered)", 4),
            (r"(?:grants?|granting|granted)\s+(?:defendant['']s?\s+)?(?:motion\s+for\s+)?summary\s+judgment", 3),
            (r"defendant\s+is\s+entitled\s+to\s+summary\s+judgment", 4),
            (r"motion\s+for\s+summary\s+judgment\s+is\s+granted", 4),
            (r"judgment\s+as\s+a\s+matter\s+of\s+law", 2),
            (r"uncontroverted", 3),
            (r"fails?\s+to\s+(?:create|raise|generate|show)\s+a\s+(?:genuine|triable|material)", 4),
            (r"fails?\s+to\s+(?:raise|create)\s+(?:a\s+)?triable", 3),
            (r"rule\s+56", 1),
            (r"celotex", 2),
            (r"anderson\s+v\.\s+liberty", 2),
            (r"matsushita", 2),
        ]
        neg_score = sum(w * min(3, len(re.findall(p, text_lower))) for p, w in neg_patterns)

        # --- 4. COMBINE ---
        raw = fact_score + pos_score - neg_score

        # --- 5. NORMALIZE ---
        scale = math.log(max(fact_score + pos_score + 1, 1)) + 6.0
        if scale <= 0:
            normalized = 0.0
        else:
            normalized = (raw + scale) / (2.0 * scale)
        normalized = max(0.0, min(1.0, normalized))

        # --- 6. GUARDRAILS ---
        if neg_score >= 15 and pos_score < 4:
            normalized = min(normalized, 0.25)
        elif neg_score >= 10 and pos_score < 4:
            normalized = min(normalized, 0.35)
        elif neg_score >= 6 and pos_score < 2:
            normalized = min(normalized, 0.45)
        if fact_score < 2 and pos_score < 2:
            normalized = min(normalized, 0.35)
        if neg_score >= 12 and fact_score <= 4:
            normalized = min(normalized, 0.30)
        if pos_score >= 20 and neg_score <= 5:
            normalized = max(normalized, 0.70)

        return 0.5 + normalized * 9.5

    except Exception:
        return 0.5