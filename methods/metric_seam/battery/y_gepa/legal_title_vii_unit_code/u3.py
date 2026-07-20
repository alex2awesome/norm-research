def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.5
        t = text.lower()
        import re

        invest_patterns = [
            r"investigat(?:e[ds]?|ing|ion)",
            r"inquir(?:y|ies)",
            r"look(?:ed)? into",
            r"conduct(?:ed)? an?\s+(?:investigation|inquiry|review)",
            r"internal review",
            r"look(?:ed)? into",
            r"examin(?:e[ds]?|ing|ation)",
            r"independent\s+(?:investigation|examination|review)",
            r"review(?:ed|ing)?\s+(?:the\s+)?(?:complaint|allegation|matter|incident)",
            r"prompt(?:ly)?\s+(?:investigat|look|review|examin|respond)",
            r"prompt\s+(?:investigation|response|corrective)",
            r"corrective\s+action",
            r"remedi(?:al|ate|ated|ating|ation)",
            r"took\s+action",
            r"disciplin(?:e[ds]?|ing|ary)",
            r"warn(?:ed|ing)?",
            r"reprimand(?:ed)?",
            r"suspend(?:ed|ing)?",
            r"terminat(?:e[ds]?|ing|ion)",
            r"discharge[ds]?",
            r"fir(?:e[ds]?|ing)",
            r"train(?:ing|ed)?",
            r"sensitiz(?:e[ds]?|ing|ation)",
            r"counsel(?:ed|ing)?",
            r"transfer(?:red|ring|rring)?",
            r"reassign(?:ed|ing|ment)?",
            r"separat(?:e[ds]?|ed|ing)\s+(?:the|employees|parties)",
            r"no.?contact",
            r"written\s+warning",
            r"verbal\s+warning",
            r"final\s+warning",
            r"policy\s+(?:violation|change|revision|update)",
            r"implement(?:ed|ing)?\s+(?:new\s+)?(?:polic|proced|train)",
            r"train(?:ed|ing)\s+(?:staff|employee|supervisor|workforce)",
        ]

        neg_patterns = [
            r"fail(?:ed|ing|ure)?\s+(?:to\s+)?investigat",
            r"did\s+not\s+investigat",
            r"no\s+investigation",
            r"never\s+investigat",
            r"fail(?:ed|ing|ure)?\s+(?:to\s+)?(?:take|respond|act|remedy|correct)",
            r"did\s+not\s+(?:take|respond|act|remedy|correct|investigate|inquire)",
            r"no\s+(?:corrective|remedial)\s+action",
            r"without\s+(?:investigation|inquiry|corrective|remedial|action)",
            r"delay(?:ed)?\s+(?:in\s+)?(?:investigat|respond|act)",
            r"untimel(?:y|iness)",
            r"unreasonab(?:le|ly)\s+(?:delay|slow)",
            r"took\s+no\s+action",
            r"took\s+(?:no|no further)\s+(?:corrective|remedial|disciplin)",
            r"unresponsive",
            r"inadequate\s+(?:investigation|response|corrective|remedial)",
            r"insufficient\s+(?:investigation|response|corrective|remedial|action)",
            r"ignored\s+(?:the\s+)?(?:complaint|allegation)",
            r"disregard(?:ed|ing)?\s+(?:the\s+)?(?:complaint|allegation)",
            r"brush(?:ed)?\s+(?:it\s+)?off",
            r"dismissed\s+(?:the\s+)?(?:complaint|allegation)",
            r"no\s+remedy",
            r"no\s+(?:action|response|inquiry|follow[- ]?up)",
            r"never\s+(?:respond|act|took|investigat|inquir|follow|correct|remedy)",
            r"failed\s+(?:to\s+)?remedy",
            r"failed\s+(?:to\s+)?prevent",
        ]

        invest_hits = 0
        for pat in invest_patterns:
            ms = re.findall(pat, t)
            invest_hits += len(ms)

        neg_hits = 0
        for pat in neg_patterns:
            ms = re.findall(pat, t)
            neg_hits += len(ms)

        prompt_indicators = 0
        prompt_patterns = [
            r"prompt(?:ly)?",
            r"immediate(?:ly)?",
            r"without\s+delay",
            r"forthwith",
            r"expeditious(?:ly)?",
            r"timely",
            r"swift(?:ly)?",
            r"right\s+away",
        ]
        for pat in prompt_patterns:
            prompt_indicators += len(re.findall(pat, t))

        employer_terms = [
            "employer", "defendant", "company", "management", "supervisor",
            "manager", "hr", "human resources", "human-relations",
            "personnel", "respondent", "company-official"
        ]
        employer_present = any(term in t for term in employer_terms)

        total = invest_hits + neg_hits
        if total == 0:
            return 0.5

        raw = invest_hits / float(total)

        bonus = 0.0
        if invest_hits > 0:
            bonus = min(0.05 * prompt_indicators, 0.15)
            if employer_present:
                bonus += 0.05
            bonus = min(bonus, 0.15)

        s = raw + bonus
        s = min(s, 1.0)
        s = 0.5 + (s - 0.5) * 0.9
        s = max(0.0, min(1.0, s))
        return float(s)

    except Exception:
        return 0.5