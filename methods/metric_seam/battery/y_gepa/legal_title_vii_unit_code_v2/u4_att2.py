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

        # ---- Gate 0: Explicit retaliation header (from training twin errors) ----
        gate_patterns = [
            r"retaliat[a-z]*\s+(?:in\s+violation|under\s+(?:title\s+vii|42\s+u\.s\.c))",
            r"retaliat[a-z]*\s+(?:discrimination\s+)?(?:under|pursuant\s+to)",
            r"in\s+retaliation\s+for\s+(?:her|his|their|the|plaintiff'?s)",
            r"retaliat[a-z]*\s+(?:against|toward)\s+(?:her|him|them|the\s+plaintiff)",
            r"(?:claims?|alleges?|asserts?|brought)\s+(?:a\s+)?(?:claim\s+(?:of|for)\s+)?retaliat[a-z]*",
            r"(?:title\s+vii|42\s+u\.s\.c|civil\s+rights\s+act).*retaliat[a-z]*",
            r"retaliat[a-z]*\s+(?:claim|count|cause\s+of\s+action|claim\s+for)",
            r"(?:count|claim)\s+(?:\d+|[ivx]+)\s*:?\s*retaliat[a-z]*",
            r"anti[-\s]?retaliat[a-z]*",
            r"retaliat[a-z]*\s+(?:for|because\s+(?:of|on\s+account\s+of))",
            r"because\s+(?:she|he|they|the\s+plaintiff)\s+(?:had\s+)?(?:engaged\s+in|participated\s+in|opposed)",
            r"protected\s+activity",
            r"(?:after|following|because\s+(?:of|she|he))\s+(?:she|he|the\s+plaintiff)\s+(?:complain|report|filed)",
        ]
        header_hits = sum(1 for p in gate_patterns if re.search(p, t))

        # ---- Gate 1: Identify retaliation case ----
        ret_strong = sum(t.count(k) for k in ("retaliation", "retaliatory", "retaliated", "retaliate", "retaliates"))
        ret_weak = sum(t.count(k) for k in ("reprisal", "whistleblow", "opposition clause", "participation clause", "protected activity", "protected report", "anti-retaliation"))
        ret_signal = ret_strong * 2 + ret_weak + header_hits * 3

        # Protected activity
        prot_terms = (
            "complain", "reported", "reporting", "filed a charge", "filed a complaint",
            "eeoc", "opposed", "grievance", "internal complaint", "charge of discrimination",
            "discrimination complaint", "harassment complaint", "reported harassment",
            "reported discrimination", "opposition to", "protected", "spoke with", "spoke to",
            "notified", "raised concerns", "raised the issue", "brought to", "objected to",
            "blew the whistle", "whistle", "participat"
        )
        prot = sum(t.count(k) for k in prot_terms)

        # Adverse action
        adv_terms = (
            "terminat", "fired", "discharge", "suspended", "suspension", "demot",
            "pay cut", "denied promotion", "reassign", "laid off", "reduction in force",
            "disciplin", "written warning", "negative evaluation", "constructive discharge",
            "non-renewal", "nonrenewal", "not renewed", "adverse employment",
            "adverse action", "forced to resign", "hostile work environment", "passed over"
        )
        adv = sum(t.count(k) for k in adv_terms)

        # Causation inference
        causal_phrases = (
            "because of her complaint", "because of his complaint",
            "because she complained", "because he complained",
            "because she reported", "because he reported",
            "in retaliation for", "as retaliation for",
            "retaliatory motive", "causal connection", "causal link", "causal nexus",
            "because of her report", "because of his report",
            "following her complaint", "following his complaint", "following their complaint",
            "after she complained", "after he complained",
            "after she reported", "after he reported",
            "in response to her complaint", "in response to his complaint",
            "on account of", "pretext for",
        )
        causal_hits = sum(t.count(p) for p in causal_phrases)

        # Gating decisions
        if ret_signal >= 3 or (ret_strong >= 1 and prot >= 1 and adv >= 1):
            is_ret_case = True
        elif ret_signal >= 1 and prot >= 1 and adv >= 1 and causal_hits >= 1:
            is_ret_case = True
        elif ret_strong >= 2 and (prot >= 1 or adv >= 1):
            is_ret_case = True
        else:
            is_ret_case = False

        if not is_ret_case:
            # Non-retaliation cases should score very low
            s = ret_signal * 0.15
            if prot >= 1 and adv >= 1:
                s += 0.15
            if causal_hits > 0:
                s += 0.15
            return max(0.05, min(0.55, s))

        # ---- Phase 2: Score concrete causation evidence ----
        score_val = 0.0

        # Tier 1: Temporal proximity (strongest single factor)
        temporal_patterns = [
            r"\b(?:days?|weeks?|months?|hours?|one month|two months|three months|four months|five months|six months)\s+(?:after|following|subsequent to)\s+(?:her|his|their|the|plaintiff'?s)\s+(?:complaint|report|charge|grievance|eeoc|opposition|protected)",
            r"\b(?:shortly|soon|immediately|quickly|thereafter)\s+(?:after|following)\s+(?:her|his|their|the|plaintiff'?s)\s+(?:complaint|report|charge|grievance|eeoc|opposition|protected)",
            r"(?:shortly|soon|immediately|days?|weeks?|months?)\s+(?:after|following)\s+(?:her|his|their|the)?\s*(?:plaintiff'?s)?\s*(?:filing|making|submitting|lodging)\s+(?:a|her|his|their|the)?\s*(?:complaint|charge|report|grievance)",
            r"(?:shortly|soon|immediately|days?|weeks?|months?)\s+(?:after|following)\s+(?:she|he|plaintiff)\s+(?:complain|report|filed|opposed|notified)",
            r"temporal\s+proximity",
            r"\b(?:within|after)\s+(?:days?|weeks?|months?|hours?)",
        ]
        temp_hits = sum(1 for p in temporal_patterns if re.search(p, t))
        if temp_hits > 0:
            score_val += min(2.5, 0.8 + temp_hits * 0.5)

        # Tier 2: Direct causal statements (strong)
        direct_patterns = [
            r"because\s+(?:of|she|he|the\s+plaintiff|her|his|they)",
            r"on\s+account\s+of\s+(?:her|his|their|the|plaintiff'?s)",
            r"due\s+to\s+(?:her|his|their|the|plaintiff'?s)\s+(?:complaint|report|charge|grievance|opposition|protected|whistleblow)",
            r"in\s+retaliation\s+for",
            r"as\s+retaliation\s+for",
            r"retaliatory\s+(?:animus|motive|intent|purpose|reason)",
            r"(?:in|as)\s+(?:a\s+)?(?:direct\s+|immediate\s+)?response\s+to\s+(?:her|his|their|the|plaintiff'?s)\s+(?:complaint|report|charge|grievance|opposition|protected)",
            r"motivat\w*\s+by\s+(?:her|his|their|the|plaintiff'?s)\s+(?:complaint|report|charge|grievance|opposition|protected|whistleblow)",
            r"desire\s+to\s+(?:retaliate|punish|chill|deter)",
            r"(?:following|after)\s+(?:her|his|their|the|plaintiff'?s)\s+(?:complaint|report|charge|grievance|opposition|protected)",
            r"\bbut[- ]for\s+(?:her|his|their|the|plaintiff'?s)\s+(?:complaint|report|charge|protected)",
            r"if\s+(?:she|he|they|the\s+plaintiff)\s+(?:had\s+)?not\s+(?:complain|report|filed|opposed)",
        ]
        direct_hits = sum(1 for p in direct_patterns if re.search(p, t))
        if direct_hits > 0:
            score_val += min(2.0, 0.5 + direct_hits * 0.4)

        # Tier 3: Causal language already counted
        score_val += min(1.0, causal_hits * 0.25)

        # Tier 4: Evidentiary categories
        cat_patterns = [
            (r"(?:decision[-\s]?maker|supervisor|manager|director|boss|hr|human\s+resources|employer)\s+(?:stated|said|told|remarked|admitted|acknowledged|wrote|emailed|noted|commented|testified)", "decisionmaker"),
            (r"(?:you|she|he|they|plaintiff)\s+(?:wouldn'?t|won'?t|will\s+not|would\s+not)\s+have\s+(?:a\s+)?(?:job|position|work)", "threat"),
            (r"(?:if\s+(?:you|she|he|they)\s+(?:complain|report|file|sue)|don'?t\s+(?:complain|report|file)|stop\s+(?:complain|report))", "threat2"),
            (r"(?:you'?re|you\s+are|she\s+is|he\s+is)\s+(?:fired|terminated|done|out\s+of\s+here|history)", "threat3"),
            (r"(?:admitted|acknowledged|conceded|confessed)\s+(?:that|he|she|they|the)\s*(?:fired|terminated|demoted|suspended|disciplined)", "admission"),
            (r"(?:stated|said|told|remarked|noted|testified)\s+(?:that\s+)?(?:it|this|the)\s+(?:was|is)\s+(?:because|due\s+to|on\s+account\s+of)", "admission2"),
            (r"(?:email|e-mail|memo|memorandum|text|message|letter|note|slack|recording|tape|video)\s+(?:from|by|sent|written)\s+(?:the|a|an)\s+(?:supervisor|manager|director|boss|hr|employer|decision)", "doc"),
            (r"(?:before\s+the\s+)?(?:firing|termination|demotion|suspension|discipline)\s*,?\s*(?:the|a)\s+(?:supervisor|manager|director|boss|hr|employer)\s+(?:said|told|stated|remarked|threatened)", "statement"),
            (r"(?:stated|told|said|remarked|threatened|warned)\s+(?:that\s+)?(?:she|he|they|the\s+plaintiff)\s+(?:should\s+(?:not|never)|'?d?\s*better\s+not)\s+(?:complain|report|file|sue|talk)", "warning"),
            (r"(?:suspicious|questionable|suspect|curious|telling)\s+(?:timing|coincidence|sequence)", "suspicious"),
            (r"(?:pattern|sequence|chronology)\s+of\s+(?:events|adverse)", "pattern"),
            (r"(?:false|pretextual|fabricated|shifting|inconsistent|changing)\s+(?:reason|explanation|justification|pretext)", "pretext"),
            (r"pretext\s+(?:for|to\s+(?:hide|conceal|mask|disguise))", "pretext2"),
            (r"similarly\s+situated", "comparator"),
            (r"(?:treated\s+)?differently\s+than\s+(?:other|similarly|colleagues|coworkers|employees)", "comparator2"),
            (r"(?:deviation|departed|departure)\s+from\s+(?:policy|procedure|practice|standard)", "deviation"),
            (r"(?:inconsistent|contradictory|changed|shifting)\s+(?:explanation|reason|justification|rationale)", "shifting"),
        ]
        cat_hits = sum(1 for _, p in cat_patterns if re.search(p, t))
        score_val += min(1.5, cat_hits * 0.3)

        # Tier 5: McDonnell Douglas / burden shifting framework presence
        burden_patterns = [
            r"mcdonnell\s+douglas",
            r"burden[-\s]shifting",
            r"prima\s+facie\s+(?:case\s+)?(?:of\s+)?retaliat",
            r"(?:legitimate|non[-\s]?retaliatory)\s+(?:reason|explanation|justification)",
            r"(?:pretext|pretextual)",
            r"(?:causation|causal)\s+(?:element|prong|requirement|factor|link|connection|nexus)",
            r"(?:but[-\s]for|but\s+for)\s+(?:causation|cause|factor)",
        ]
        burden_hits = sum(1 for p in burden_patterns if re.search(p, t))
        score_val += min(1.0, burden_hits * 0.2)

        # Tier 6: Detailed protected activity mentions
        score_val += min(0.5, prot * 0.03)

        # Tier 7: Specificity of complaint timing
        specific_complaint_patterns = [
            r"\b(?:on|in)\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}",  # dates
            r"\b\d{4}\b",  # years
        ]
        date_hits = sum(1 for p in specific_complaint_patterns if re.search(p, t))
        if date_hits >= 3:
            score_val += 0.3

        # Penalize vague-only cases
        vague_patterns = [
            r"(?:throughout|during)\s+(?:her|his|their|the)\s+(?:employment|tenure)",
            r"(?:subjected|exposed)\s+to\s+(?:discrimination|harassment)",
            r"(?:ignored|disregarded|dismissed|no\s+action)\s+(?:her|his|their|the)?\s*complaint",
            r"(?:fear|afraid|intimidated)\s+(?:to|of)\s+(?:report|complain)",
            r"(?:allege|claims?|contends?|asserts?)\s+that\s+she\s+was\s+subjected",
        ]
        vague_hits = sum(1 for p in vague_patterns if re.search(p, t))
        if vague_hits >= 2 and temp_hits == 0 and direct_hits == 0:
            score_val -= 0.4

        # Scale: high-quality evidence -> 7-10, moderate -> 4-7, weak -> 1-4
        result = 1.0 + score_val * 1.1
        return max(0.1, min(9.8, result))

    except Exception:
        return 0.5