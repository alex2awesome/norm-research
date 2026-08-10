import re
import math
from collections import Counter

def score(text: str) -> float:
    try:
        if not text or not isinstance(text, str):
            return 0.05

        t = text.lower()
        words = re.findall(r"[a-z]+", t)
        total = len(words) if words else 1

        # ---- Phase 1: Is this a retaliation case at all? ----
        ret_strong = sum(t.count(k) for k in ("retaliation", "retaliatory", "retaliated", "retaliate", "retaliates"))
        ret_weak = sum(t.count(k) for k in ("reprisal", "whistleblow", "whistleblower", "s.o.s.h.", "opposition clause", "participation clause", "protected activity", "protected report"))

        prot = sum(t.count(k) for k in (
            "complain", "reported", "reporting", "filed a charge", "filed a complaint",
            "eeoc", "opposed", "grievance", "internal complaint", "charge of discrimination",
            "discrimination complaint", "harassment complaint", "reported harassment",
            "reported discrimination", "opposition to", "protected", "spoke with", "spoke to",
            "notified", "raised concerns", "raised the issue", "brought to"
        ))

        adv = sum(t.count(k) for k in (
            "terminat", "fired", "discharge", "suspended", "suspension", "demot",
            "pay cut", "denied promotion", "reassign", "laid off", "reduction in force",
            "disciplin", "written warning", "negative evaluation", "constructive discharge",
            "non-renewal", "nonrenewal", "not renewed", "adverse employment",
            "adverse action", "forced to resign", "hostile work environment",
            "hostile environment", "transferred", "transfer", "passed over"
        ))

        ret_signal = ret_strong * 2 + ret_weak + (0.3 * min(prot, 6))

        if ret_signal < 0.5 and (prot < 1 or adv < 1):
            return 0.05

        if ret_signal < 0.5:
            ret_signal = 1.0  # minimal retaliation signal but both components present

        # ---- Phase 2: Count concrete causal-evidence categories ----
        evidence_score = 0.0

        # (a) Temporal proximity (strongest single factor)
        temporal_patterns = [
            r"\b(?:days?|weeks?|months?|hours?|one month|two months|three months)\s+(?:after|following|subsequent to)\s+(?:her|his|their|the|plaintiff'?s)\s+(?:complaint|report|charge|grievance|eeoc|opposition|protected)",
            r"\b(?:shortly|soon|immediately|quickly|thereafter)\s+(?:after|following)\s+(?:her|his|their|the|plaintiff'?s)\s+(?:complaint|report|charge|grievance|eeoc|opposition)",
            r"\b(?:temporal|temporal)\s+proximity\b",
            r"\bclose\s+(?:in\s+time|temporal|proximity)\b",
            r"\b(?:after|following|subsequent to)\s+(?:she|he|plaintiff)\s+(?:complain|report|file|oppos)",
            r"\bshortly\s+after\b",
            r"\b(?:weeks?|months?|days?)\s+after\s+(?:she|he|they|plaintiff)\b",
            r"\bwhen\s+(?:she|he|plaintiff)\s+complain",
            r"\bin\s+response\s+to\b",
            r"\b(?:weeks?|months?)\s+later\b",
        ]
        temporal_hits = sum(len(re.findall(p, t, flags=re.IGNORECASE)) for p in temporal_patterns)
        if temporal_hits >= 1:
            evidence_score += min(2.0, 0.8 * temporal_hits)
        if temporal_hits >= 3:
            evidence_score += 0.5

        # (b) But-for / motivating factor causation language
        cause_patterns = [
            r"\bbecause\s+(?:of|he|she|they|it|plaintiff)\s+(?:filed|reported|complain|oppos)",
            r"\bbecause\s+(?:of|her|his|their|plaintiff'?s)\s+(?:complaint|report|charge|grievance|opposition|protected)",
            r"\b(?:but-for|but for)\s+(?:caus|cause)",
            r"\bmotivating\s+factor\b",
            r"\b(?:retaliatory|unlawful)\s+(?:motive|animus|intention)\b",
            r"\b(?:causal|direct)\s+(?:link|connection|nexus)\b",
            r"\bdirect(?:ly)?\s+(?:link|relat)\b",
            r"\bcould\s+infer\b",
            r"\b(?:supports|establishes)\s+(?:an?\s+)?inference\b",
            r"\binfer(?:ence)?\s+of\s+(?:retaliatory|causal|discriminatory)\b",
            r"\b(?:anti-retaliation|retaliatory)\s+(?:purpose|motive|reason|intent)\b",
            r"\bdirect\s+evidence\b",
            r"\bcircumstantial\s+evidence\b",
            r"\bwould\s+not\s+have\s+(?:fired|terminat|demot|disciplin|suspend)\b",
            r"\b(?:causation|causal|cause)\b",
            r"\bdiscriminatory\s+(?:motive|animus|reason|purpose)\b",
            r"\bretaliat\w*\s+(?:motive|animus|intent|purpose)\b",
        ]
        cause_hits = sum(len(re.findall(p, t, flags=re.IGNORECASE)) for p in cause_patterns)
        evidence_score += min(2.5, 0.45 * cause_hits)
        if cause_hits >= 4:
            evidence_score += 0.6

        # (c) Pretext / shifting-justification evidence
        pretext_patterns = [
            r"\bpretext(?:ual)?\b",
            r"\bshifting\s+(?:reasons|justification|explanation|rational)",
            r"\binconsistent\s+(?:reasons|explanation|justification|rationale)",
            r"\bpretext(?:ual)?\s+(?:for|to)\b",
            r"\breason\s+given\b",
            r"\b(?:stated|proffered|offered)\s+reason\b",
            r"\bfalse\s+(?:reason|explanation|justification|pretext)\b",
            r"\bcover[- ]?up\b",
        ]
        pretext_hits = sum(len(re.findall(p, t, flags=re.IGNORECASE)) for p in pretext_patterns)
        evidence_score += min(1.5, 0.45 * pretext_hits)
        if pretext_hits >= 2:
            evidence_score += 0.3

        # (d) Comparator / similarly-situated evidence
        comparator_patterns = [
            r"\bsimilarly[- ]situated\b",
            r"\bsimilarly\s+situated\b",
            r"\bcomparator\b",
            r"\btreated\s+(?:differently|more harshly|less favorably)\b",
            r"\boutside\s+(?:the\s+)?protected\s+(?:class|activity)\b",
            r"\bdid\s+not\s+(?:retaliate|terminat|disciplin)\s+(?:against|other)\b",
            r"\bothers?\s+who\s+did\s+not\b",
        ]
        comp_hits = sum(len(re.findall(p, t, flags=re.IGNORECASE)) for p in comparator_patterns)
        evidence_score += min(1.2, 0.4 * comp_hits)

        # (e) Decision-maker knowledge of protected activity
        knowledge_patterns = [
            r"\b(?:knew|aware)\s+(?:of|about)\s+(?:the|her|his|plaintiff'?s|prior)\s+(?:complaint|report|charge|grievance|protected|eeoc)",
            r"\b(?:knew|aware)\s+(?:that|she|he)\s+(?:had|filed|reported|complain)",
            r"\binformed\s+(?:of|about)\s+(?:the|her|his)\s+(?:complaint|report|charge|grievance)",
            r"\b(?:after|when)\s+(?:she|he|plaintiff)\s+(?:learned|discovered)\b",
            r"\b(?:supervisor|manager|director)\s+(?:knew|was aware)\b",
            r"\bdecision[- ]?maker'?s?\s+(?:knowledge|awareness)\b",
            r"\baware\s+(?:of|that)\b.{0,60}(?:complaint|report|charge|grievance|protected)",
            r"\b(?:he|she|they)\s+(?:knew|was aware)\s+of\b.{0,40}(?:complaint|report)",
        ]
        knowledge_hits = sum(len(re.findall(p, t, flags=re.IGNORECASE)) for p in knowledge_patterns)
        evidence_score += min(1.2, 0.45 * knowledge_hits)

        # (f) McDonnell-Douglas / burden-shifting (suggests formal analysis)
        mcd_patterns = [
            r"\bmcdonnell\s+douglas\b",
            r"\bburden[- ]shifting\b",
            r"\bburden\s+(?:of\s+)?production\b",
            r"\bprima\s+facie\b",
            r"\blegitimate\s+(?:non-?discriminatory|non-?retaliatory)\s+(?:reason|justification)\b",
            r"\bpretext\s+for\s+(?:retaliation|discrimination)\b",
            r"\bultimate\s+burden\b",
            r"\b(?:satisf|meet|establish)\w*\s+(?:the\s+)?prima\s+facie\b",
            r"\b(?:satisf|meet|establish)\w*\s+(?:the\s+)?elements?\s+of\s+(?:a|the)\s+(?:retaliation|claim)\b",
        ]
        mcd_hits = sum(len(re.findall(p, t, flags=re.IGNORECASE)) for p in mcd_patterns)
        if mcd_hits >= 1:
            evidence_score += min(1.0, 0.4 * mcd_hits)

        # (g) Quote / attribution markers (narrative specificity)
        quote_markers = len(re.findall(r'"[^"]{15,200}"', text))
        evidence_score += min(1.0, 0.12 * quote_markers)

        # ---- Phase 3: Combine ----
        prot_factor = min(prot / 4.0, 1.0)
        adv_factor = min(adv / 4.0, 1.0)
        both_factor = prot_factor * adv_factor

        base = min(evidence_score, 9.0)
        final = base * (0.5 + 0.5 * both_factor)

        ret_boost = 0.0
        if ret_strong > 0:
            ret_boost = min(1.2, 0.3 * ret_strong)
        final += ret_boost

        # documents with very little text get damped slightly
        if total < 150:
            final *= 0.8
        elif total < 400:
            final *= 0.95

        # ---- Scale to 0-10 and ensure floor for confirmed-retaliation cases ----
        if ret_strong > 0 and (prot >= 1 or adv >= 1):
            final = max(final, 1.5)

        # If the document has explicit cause + temporal + ret language, ensure decent score
        if ret_strong >= 1 and temporal_hits >= 1 and cause_hits >= 1:
            final = max(final, 4.0)

        final = min(10.0, max(0.0, final))
        return final

    except Exception:
        return 0.5