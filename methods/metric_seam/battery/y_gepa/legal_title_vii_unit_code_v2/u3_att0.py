import re
import math

_BUDGET = 30000

def _bounded_search(pattern, text):
    safe = r"(?:.|\n){0," + str(_BUDGET) + r"}?"
    return re.compile(pattern.replace(".*?", safe), re.IGNORECASE)

def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = text[:600000]

        investigation_terms = [
            r"investigat", r"inquir", r"examin", r"look(?:ed)? into",
            r"review(?:ed|ing)?\s+(?:the\s+)?(?:matter|alleg|complaint|incid|report|claim|conduct|evidence|facts)",
            r"conduct(?:ed|ing)?\s+an?\s+(?:investigation|inquiry|review|examin|analysis|independent\s+investigation)",
            r"internal\s+investigation", r"prompt\s+investigation",
            r"thorough\s+investigation", r"outside\s+investigator",
            r"third[\s-]?party\s+investigator", r"hired\s+(?:an?\s+)?investigator",
            r"independent\s+investigator", r"external\s+investigation",
            r"investigated\s+the\s+(?:matter|alleg|complaint|incid|report|claim|conduct|facts|accusation)",
            r"interview(?:ed|ing)?", r"took\s+statements?", r"obtained\s+statements?",
            r"gather(?:ed|ing)?\s+(?:information|evidence|facts|witness\s+statements)",
            r"collect(?:ed|ing)?\s+(?:information|evidence|facts)",
            r"made\s+(?:a\s+)?(?:determination|finding)",
            r"determine(?:d|ation)\s+(?:that|whether)",
            r"conclud(?:ed|sion)\s+(?:that|the|no)",
            r"found\s+(?:that|the|no)",
            r"substantiat", r"unsubstantiat", r"unfounded",
            r"could\s+not\s+(?:substantiat|confirm|verify|determin)",
            r"insufficient\s+evidence", r"lack(?:ed|ing)?\s+(?:of\s+)?evidence",
            r"unable\s+to\s+(?:confirm|verify|substantiat|determin)",
            r"personnel\s+(?:file|matter|action)", r"hr\s+(?:review|investigation|inquiry)",
            r"human\s+resources\s+(?:review|investigation|inquiry)",
            r"equal\s+employment\s+opportunity\s+(?:review|investigation|inquiry)",
            r"eeoc\s+(?:review|investigation|inquiry)",
            r"complaint\s+(?:procedure|process|policy)",
            r"grievance\s+process", r"report(?:ed|ing)?\s+(?:to|up)",
            r"escalat", r"brought\s+(?:it|the\s+(?:matter|complaint|issue))\s+to",
            r"notified\s+(?:management|supervisor|hr|human\s+resources|authorities)",
            r"report(?:ed)?\s+(?:the\s+)?(?:incident|conduct|matter|alleg|harass|discrimin)",
        ]

        corrective_terms = [
            r"corrective\s+action", r"remedi(?:ed|al|ate|ating|ation)",
            r"prompt(?:ly)?\s+(?:respond|react|address|correct|investigat|remedi|act|notifi|report|disciplin|suspend|terminat|warn|counsel)",
            r"took\s+(?:action|steps?|measures?|immediate)",
            r"took\s+(?:immediate|prompt|appropriate|reasonable)\s+(?:action|steps?|measures?)",
            r"action\s+(?:was|were)\s+(?:taken|prompt)",
            r"immediate\s+action", r"appropriate\s+action", r"appropriate\s+measures?",
            r"swift\s+(?:action|response|measures?)",
            r"measures?\s+(?:were|was)\s+taken", r"steps?\s+(?:were|was)\s+taken",
            r"disciplin(?:e|ed|ing|ary)",
            r"warn(?:ed|ing)?", r"reprimand", r"counsel(?:ed|ing)?",
            r"suspens(?:ion|ed)", r"suspend(?:ed|ing)?",
            r"terminat(?:ed|ion|ing|e)",
            r"fir(?:ed|ing)", r"discharg(?:ed|ing)",
            r"demot(?:ed|ion|ing)", r"transfer(?:red|ring|ed)?",
            r"reassign(?:ed|ing|ment)",
            r"remov(?:ed|ing|al)\s+(?:the\s+)?(?:employee|supervisor|manager|offender|perpetrator| harasser)",
            r"separat(?:ed|ing)\s+(?:the\s+)?(?:parties|employee|individuals)",
            r"written\s+warning", r"final\s+warning",
            r"employee\s+assistance\s+program",
            r"sensitivity\s+train", r"diversity\s+train", r"anti[\s-]?harassment\s+train",
            r"sexual\s+harassment\s+train", r"preventive\s+train",
            r"train(?:ing|ed)?\s+(?:on|regarding|concerning|related\s+to)",
            r"counseling\s+(?:was|session)",
            r"memorand(?:um|a)\s+of\s+record", r"letter\s+of\s+reprimand",
            r"no[\s-]?contact\s+(?:order|directive|policy|instruction)",
            r"restrict(?:ed|ing)\s+contact", r"work\s+schedule\s+(?:change|adjust|modif)",
            r"schedule\s+(?:change|adjust|modif)",
            r"shift(?:ed|ing)\s+(?:schedule|hours|work)",
            r"chang(?:ed|ing)\s+(?:work\s+)?(?:schedule|hours|assignment|location|shift)",
            r"modif(?:ied|y|ying|ication)\s+(?:of\s+)?(?:schedule|hours|work\s+(?:area|location|environment|assignment))",
            r"transfer(?:red)?\s+(?:the\s+)?(?:employee|supervisor|manager|offender|perpetrator|harasser|plaintiff)",
            r"install(?:ed|ing)?\s+(?:security\s+)?cameras?",
            r"enhanced?\s+security", r"increased?\s+(?:security|supervision)",
            r"policy\s+(?:change|revision|update|modif)",
            r"revis(?:ed|ing|ion)\s+(?:the\s+)?(?:policy|policies|handbook|procedures?)",
            r"updat(?:ed|ing|e)\s+(?:the\s+)?(?:policy|policies|handbook|procedures)",
            r"implement(?:ed|ing|ation)?\s+(?:new\s+)?(?:policies|procedures|measures|guidelines|training)",
            r"enforce(?:d|ment)\s+(?:of\s+)?(?:the\s+)?(?:policy|policies|rules|procedures)",
            r"monitor(?:ed|ing)?", r"supervis(?:ed|ion|ing)\s+(?:the\s+)?(?:situation|employee|offender|perpetrator|workplace)",
            r"follow(?:ed)?[\s-]+up", r"follow\s+up\s+(?:investigation|review|inquiry|action)",
            r"checked?\s+(?:in\s+)?(?:on|with)", r"subsequent(?:ly)?\s+(?:review|monitor|investigat|contact)",
            r"ensur(?:ed|ing)?\s+(?:that\s+)?(?:the\s+)?(?:conduct|behavior|harassment|discrimination)\s+(?:did\s+not|does\s+not|would\s+not|ceased|stopped|cease|stop)",
            r"stop(?:ped|ping)?\s+(?:the\s+)?(?:conduct|behavior|harassment|discrimination)",
            r"ceas(?:ed?|ing)\s+(?:the\s+)?(?:conduct|behavior|harassment|discrimination)",
            r"prevent(?:ed|ing)?\s+(?:future|recurrence|further)\s+(?:incid|conduct|harassment|discrimination|violation)",
            r"preventative?\s+(?:measures?|action|steps?)",
        ]

        def count_terms(terms):
            hits = 0
            positions = []
            for term in terms:
                try:
                    for m in re.finditer(term, t, re.IGNORECASE):
                        hits += 1
                        positions.append(m.start())
                        if hits > 250:
                            return hits, positions
                except re.error:
                    continue
            return hits, positions

        inv_hits, inv_pos = count_terms(investigation_terms)
        cor_hits, cor_pos = count_terms(corrective_terms)

        prompt_count = 0
        for term in [r"prompt(?:ly)?", r"immediate(?:ly)?", r"swift(?:ly)?", r"without\s+delay", r"timely", r"expeditious"]:
            try:
                prompt_count += min(15, len(re.findall(term, t, re.IGNORECASE)))
            except re.error:
                continue

        documented_count = 0
        for term in [r"document(?:ed|ing|ation)?", r"record(?:ed|ing)?", r"report(?:ed|ing)?\s+(?:of|in)"]:
            try:
                documented_count += min(10, len(re.findall(term, t, re.IGNORECASE)))
            except re.error:
                continue

        active_voice_patterns = [
            r"employer\s+(?:conduct|investigat|initiat|launch|undertook|perform|complet|review|examin)",
            r"employer\s+(?:took|implement|adopt|provid|requir|order|direct)",
            r"(?:company|defendant|management|supervisor|hr|human\s+resources)\s+(?:conduct|investigat|initiat|launch|undertook|perform|complet|review|examin|took|implement|provid|requir)",
            r"(?:was|were)\s+(?:conduct|investigat|initiat|launch|undertaken|perform|complet|review|examin|implement|taken)",
            r"following\s+(?:the\s+)?(?:report|complaint|alleg|incident|disclos)",
            r"in\s+response\s+to\s+(?:the\s+)?(?:complaint|alleg|report|incident|conduct|harassment|discrimination)",
            r"after\s+(?:the\s+)?(?:complaint|report|alleg|incident|notification)",
            r"upon\s+(?:receiving|learning|being\s+notified|notification|report|complaint)",
        ]
        active_voice_count = 0
        for term in active_voice_patterns:
            try:
                active_voice_count += min(15, len(re.findall(term, t, re.IGNORECASE)))
            except re.error:
                continue

        negation_patterns = [
            r"(?:did\s+not|failed?\s+to|never|no)\s+(?:investigat|conduct\s+an?\s+investigation|inquir|review|examin|look\s+into|respond|react|address|correct|take\s+action|take\s+(?:any\s+)?(?:corrective|remedial)\s+(?:action|measures?|steps?)|remedi|act|disciplin|warn|reprimand|counsel|suspend|terminat|fire|discharge)",
            r"fail(?:ed|ing|ure)\s+to\s+(?:investigat|conduct|respond|act|address|correct|remedi|take\s+action|disciplin|warn)",
            r"no\s+(?:investigation|inquiry|review|response|action|corrective\s+action|remedial\s+(?:action|measures?|steps?)|discipline|follow[\s-]?up)",
            r"without\s+(?:investigation|inquiry|any\s+(?:investigation|inquiry|action|corrective\s+action|remedial\s+action))",
            r"un(?:reasonabl|necessaril)y\s+delay(?:ed|ing)?",
            r"tard(?:y|ily)",
            r"insufficient\s+(?:corrective|remedial|responsive)\s+action",
            r"inadequate\s+(?:response|investigation|corrective\s+action|remedial\s+action)",
            r"failure\s+to\s+act", r"failure\s+of\s+(?:the\s+)?employer\s+to\s+act",
            r"took\s+no\s+action", r"took\s+no\s+steps",
            r"did\s+nothing", r"no\s+action\s+(?:was|were)\s+taken",
            r"ignored?\s+(?:the\s+)?(?:complaint|alleg|report|incident|matter|conduct|harassment)",
            r"disregard(?:ed|ing)?\s+(?:the\s+)?(?:complaint|alleg|report|incident|matter|conduct)",
            r"dismiss(?:ed|ing)?\s+(?:the\s+)?(?:complaint|alleg|report|incident|matter)",
            r"discredited?\s+(?:the\s+)?(?:complaint|alleg|report|incident|matter|plaintiff)",
            r"discourag(?:ed|ing)?\s+(?:complaints|reporting)",
        ]
        negation_count = 0
        for term in negation_patterns:
            try:
                negation_count += min(15, len(re.findall(term, t, re.IGNORECASE)))
            except re.error:
                continue

        found_something_patterns = [
            r"substantiat(?:ed)?", r"founded", r"found\s+to\s+have\s+(?:violat|engag|harass|discriminat)",
            r"determin(?:ed|ation)\s+(?:that\s+)?(?:the\s+)?(?:allegation|complaint|conduct|harassment|discrimination)\s+(?:was|were)\s+(?:substantiat|found|valid|credible|true)",
            r"sustained?\s+(?:the\s+)?(?:complaint|allegation)",
        ]
        found_something_count = 0
        for term in found_something_patterns:
            try:
                found_something_count += min(8, len(re.findall(term, t, re.IGNORECASE)))
            except re.error:
                continue

        clear_contradictions = [
            r"substantiated[^.]{0,80}(?:no\s+evidence|unsubstantiat|unfounded|without\s+merit|false)",
            r"investigated[^.]{0,80}(?:did\s+not|never|no\s+action|no\s+corrective)",
            r"corrective\s+action[^.]{0,80}(?:was\s+not|not\s+taken|failed)",
            r"(?:warn|suspend|terminat)[^.]{0,80}(?:no\s+evidence|did\s+not|never)",
        ]
        contradiction_count = 0
        for term in clear_contradictions:
            try:
                contradiction_count += min(8, len(re.findall(term, t, re.IGNORECASE)))
            except re.error:
                continue

        raw = (
            (math.sqrt(inv_hits) * 1.7) +
            (math.sqrt(cor_hits) * 1.7) +
            (math.sqrt(active_voice_count) * 2.3) +
            (math.sqrt(prompt_count) * 0.7) +
            (math.sqrt(documented_count) * 0.5) +
            (math.sqrt(found_something_count) * 0.8)
            - (math.sqrt(max(0, negation_count)) * 2.2)
            - (math.sqrt(max(0, contradiction_count)) * 1.5)
        )

        s = 1.4 + raw * 0.75
        s = max(0.3, min(9.9, s))

        if inv_hits + cor_hits >= 6:
            s += 0.3
        if active_voice_count >= 5:
            s += 0.2
        if negation_count >= 8:
            s -= 0.3
        if found_something_count >= 2:
            s += 0.15

        s = max(0.0, min(10.0, s))
        return float(s)

    except Exception:
        return 0.5