import re
import math
from collections import Counter

def score(text: str) -> float:
    try:
        if not text or not isinstance(text, str):
            return 0.5
        t = text.lower()
        words = re.findall(r"[a-z]+", t)
        total = len(words) if words else 1

        retaliation_kw = [
            "retaliat", "reprisal", "whistleblow", "protected activity",
            "protected report", "complaint of discrimination", "opposition clause",
            "participation clause", "after-acquired", " adverse action",
            "adverse employment", "anti-retaliation", "retaliatory motive",
            "chilling effect", "first amendment retaliation"
        ]
        protected_kw = [
            "filed a complaint", "reported discrimination", "complained about",
            "opposed", "participated", "reported harassment", "internal complaint",
            "eeoc charge", "charge of discrimination", "reported fraud", "reported waste",
            "reported abuse", "grievance", "union complaint"
        ]
        adverse_kw = [
            "termination", "fired", "discharge", "suspended", "demotion", "demoted",
            "pay cut", "denied promotion", "reassignment", "reassigned", "laid off",
            "reduction in force", "disciplined", "written warning", "negative evaluation"
        ]

        ret_hits = sum(t.count(k) for k in retaliation_kw)
        prot_hits = sum(t.count(k) for k in protected_kw)
        adv_hits = sum(t.count(k) for k in adverse_kw)

        if ret_hits == 0 and prot_hits == 0:
            return 0.25

        causal_phrases = [
            "because he filed", "because she filed", "because they filed",
            "because of his complaint", "because of her complaint",
            "because of their complaint", "because he complained",
            "because she complained", "because he reported",
            "because she reported", "because of his report",
            "because of her report", "in retaliation for",
            "causal link", "causal connection", "causal nexus",
            "but-for causation", "but-for cause", "motivating factor",
            "animus", "pretext for", "close temporal proximity",
            "temporal proximity", "after she complained",
            "after he complained", "after they complained",
            "shortly after", "days after he complained",
            "days after she complained", "weeks after he complained",
            "weeks after she complained", "days after the complaint",
            "weeks after the complaint", "terminated shortly after",
            "fired shortly after", "demoted shortly after",
            "following his complaint", "following her complaint",
            "following his report", "following her report",
            "soon after", "knowledge of the protected",
            "was aware of the complaint", "knew of the complaint",
            "knew of his complaint", "knew of her complaint",
            "was aware that", "causation", "causation standard",
            "retaliatory animus"
        ]
        causal_hits = sum(t.count(k) for k in causal_phrases)

        timeline_pattern = re.compile(
            r"\b(within|after)\s+(?:a\s+|a\s+few\s+)?(days?|weeks?|months?)\s+of\s+"
            r"(?:her|his|their|the|plaintiff'?s?)\s+(?:complaint|report|charge|grievance|petition)"
        )
        timeline_hits = len(timeline_pattern.findall(t))

        evidence_kw = [
            "evidence", "record shows", "record reflects", "demonstrates",
            "establishes", "proves", "proof of", "direct evidence",
            "circumstantial evidence", "inference", "supports a finding",
            "genuine issue", "material fact", "jury could find",
            "jury could conclude", "reasonable jury", "sufficient evidence",
            "showing of", "testified", "email", "emails", "memo",
            "memorandum", "documented", "contemporaneous"
        ]
        evidence_hits = sum(t.count(k) for k in evidence_kw)

        word_counter = Counter(words)
        causation_signal = causal_hits + 2 * timeline_hits

        base = 0.0
        if ret_hits > 0:
            base += min(1.5, 0.3 + 0.25 * math.log1p(ret_hits))
        if prot_hits > 0:
            base += min(1.5, 0.3 + 0.25 * math.log1p(prot_hits))
        if adv_hits > 0:
            base += min(1.0, 0.2 + 0.2 * math.log1p(adv_hits))
        if causation_signal > 0:
            base += min(3.0, 0.5 + 0.5 * math.log1p(causation_signal))
        if evidence_hits > 0:
            base += min(1.5, 0.2 + 0.2 * math.log1p(evidence_hits))

        has_ret = ret_hits > 0
        has_prot = prot_hits > 0
        has_adv = adv_hits > 0
        has_causal = causation_signal > 0

        if has_ret and has_prot and has_adv and has_causal:
            base += 1.5
        elif has_ret and (has_prot or has_adv) and has_causal:
            base += 1.0
        elif has_ret and (has_prot or has_adv):
            base += 0.5
        elif has_prot and has_adv:
            base += 0.3

        density_bonus = min(1.5, (ret_hits + prot_hits + adv_hits + causation_signal) * 250.0 / total)
        base += density_bonus

        if total < 200:
            penalty = 0.0
        elif total < 500:
            penalty = 0.15
        else:
            penalty = 0.0

        final = max(0.0, min(10.0, base - penalty))

        if not (has_ret or has_prot):
            return min(final, 3.0) / 10.0
        if has_causal:
            return max(final, 4.0) / 10.0
        return final / 10.0
    except Exception:
        return 0.5