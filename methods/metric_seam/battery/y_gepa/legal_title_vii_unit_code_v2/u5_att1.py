import re
import math
from collections import Counter

def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        t_low = text.lower()
        length = len(t_low)
        if length < 200:
            return 0.0

        # Strong markers - high specificity for Title VII statistical/systemic evidence
        strong_markers = [
            "disparate impact", "adverse impact", "pattern or practice",
            "pattern of discrimination", "four-fifths rule", "four fifths rule",
            "4/5 rule", "80% rule", "80 percent rule", "standard deviation",
            "standard deviations", "statistically significant", "regression analys",
            "multiple regression", "binomial distribution", "chi-square",
            "chi square", "griggs v", "hazelwood", "teamsters v",
            "compensable factor", "flagrant mistake",
            "underrepresent", "under-represent", "underutili",
            "workforce analy", "availability data", "applicant flow",
            "labor market", "labour market", "comparison group",
            "selection rate", "hiring rate", "pass rate", "promoted at a lower rate",
            "hired at a lower rate", "promoted less", "comparators",
            "pool of applicants", "qualified applicant pool",
            "disparity", "disparities", "statistical disparit",
            "impact ratio", "disproportionate", "disproportionately",
            "systemic discriminat", "class action",
        ]

        strong_hits = 0
        for m in strong_markers:
            c = t_low.count(m)
            if c:
                strong_hits += min(c, 4)

        # Moderate markers - statistical context phrases
        moderate_markers = [
            "statistic", "percentages of", "proportion", "demographic",
            "minority", "female", "women", "african american",
            "hispanic", "asian american", "caucasian", "protected class",
            "protected group", "class of", "similarly situated",
            "comparator", "comparable", "numbers of", "percentage",
            "comparative", "representative sample",
            "mean", "median", "average", "incumbent",
            "turnover", "attrition", "composition",
            "evidence of discriminat", "comparative evidence",
        ]

        moderate_hits = 0
        for m in moderate_markers:
            c = t_low.count(m)
            if c:
                moderate_hits += min(c, 5)

        # Numerical evidence: percentages near discrimination context
        pct_matches = re.findall(r'\d+\.?\d*\s*%', t_low)
        pct_count = len(pct_matches)

        # Ratio/fraction patterns
        ratio_matches = re.findall(
            r'\b\d+\.?\d*\s*(?:percent|%)\s*(?:of|fewer|less|more|greater)\b',
            t_low)
        ratio_count = len(ratio_matches)

        # Explicit findings of discrimination
        finding_markers = [
            "finding of discrimination", "finding of discriminat",
            "found that", "prior finding", "previously found",
            "convicted of discriminat", "liability for discriminat",
            "determine(d|s)? .* discriminat",
        ]
        finding_hits = 0
        for m in finding_markers:
            if m in t_low:
                finding_hits += 1
            else:
                try:
                    if re.search(m, t_low):
                        finding_hits += 1
                except Exception:
                    pass

        # Compute raw score
        raw = (
            min(strong_hits, 18) * 1.8
            + min(moderate_hits, 25) * 0.35
            + min(pct_count, 12) * 0.4
            + min(ratio_count, 8) * 0.6
            + finding_hits * 1.5
        )

        # Title VII / employment discrimination context check
        title_vii_ctx = any(k in t_low for k in [
            "title vii", "title 7", "employment discriminat",
            "discriminat", "disparate", "equal employment",
            "eeoc", "civil rights act", "employer", "employee",
            "hiring", "promotion", "termination", "workplace",
            "work force", "workforce", "qualified",
            "retaliat", "hostile work", "reasonable accommodation",
            "pay", "compensation", "salary",
        ])

        if not title_vii_ctx:
            return 0.0

        # Boost when multiple strong signals cluster
        if strong_hits >= 3:
            raw += 1.5
        if strong_hits >= 6:
            raw += 1.0
        if strong_hits >= 10:
            raw += 1.5

        # Penalize purely anecdotal / individual cases
        anecdote_markers = [
            "individual disparate treatment",
            "individual claim",
            "single incident",
            "stray remark",
            "stray remarks",
            "anecdotal",
            "individualized assessment",
        ]
        anecdote_hits = sum(1 for m in anecdote_markers if m in t_low)
        if anecdote_hits and strong_hits < 2:
            raw -= 1.0

        # Non-Title-VII topics that trigger false positives
        non_eeo = [
            "first amendment", "free speech", "freedom of speech",
            "freedom of religion", "establishment clause",
            "fourth amendment", "search and seizure",
            "due process clause",
            "criminal", "defendant", "prosecution",
            "convicted of a crime",
            "patent", "copyright", "trademark",
            "bankruptcy", "antitrust", "securities",
            "environmental", "clean water act",
            "immigration", "deportation",
            "administrative procedure",
            "tax code", "internal revenue",
        ]
        non_eeo_hits = sum(1 for m in non_eeo if m in t_low)
        if non_eeo_hits >= 2 and strong_hits < 2:
            raw *= 0.3

        # Map to 0-1 scale
        score_val = raw / 22.0
        score_val = max(0.0, min(score_val, 1.0))
        return score_val

    except Exception:
        return 0.5