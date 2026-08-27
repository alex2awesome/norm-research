import re
import math

def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        t = text.lower()
        tl = len(t)
        if tl < 200:
            return 0.0

        s = 0.0

        tiers = [
            (
                [
                    "disparate impact", "adverse impact", "pattern or practice",
                    "pattern of discrimination", "four-fifths rule", "four fifths rule",
                    "4/5 rule", "80% rule", "80 percent rule",
                    "standard deviation", "standard deviations",
                    "statistically significant", "regression analys",
                    "multiple regression", "binomial distribution",
                    "chi-square", "chi square", "t-test",
                    "two standard deviation", "griggs v",
                    "hazelwood", "teamsters v", "wards cove",
                    "disparity", "disparities", "statistical disparit",
                    "impact ratio", "flawed promotional formula",
                    "promotion formula", "selection rate", "hiring rate",
                    "pass rate", "passing rate", "selection ratio",
                    "underrepresent", "under-represent", "underutili",
                    "under-utili", "workforce analy", "work force analy",
                    "availability data", "applicant flow", "applicant pool",
                    "comparison group", "comparison pool",
                    "qualified applicant", "labor market data",
                    "labor market statistic", "composition of the work",
                    "prior finding", "prior findings",
                    "previous finding", "earlier finding",
                    "found to have discriminat", "finding of discriminat",
                    "systemic discriminat", "history of discriminat",
                    "history of unlawful", "history of racial",
                    "history of segrega", "history of exclusion",
                    "history of bias",
                    "in the relevant labor market", "relevant labor market",
                    "relevant market", "by a margin of",
                    "disproportionate", "disproportionately",
                    "under a federal court order", "consent decree",
                    "affirmative action plan", "affirmative-action plan",
                    "court-ordered", "remedial order",
                    "quota", "set-aside", "minority", "blacks",
                    "african american", "african-american", "hispanic",
                    "latino", "asian american", "asian-american",
                    "white employee", "white applicant",
                    "female applicant", "women applicant",
                    "protected class", "protected group", "women",
                    "females", "male", "class of", "class action",
                ],
                3.2,
            ),
            (
                [
                    "statistic", "percentage", "percent", "ratio",
                    "proportion", "demographic", "comparators", "comparator",
                    "similarly situated", "comparison", "comparative evidence",
                    "evidence of discriminat", "representative sample",
                    "incumbent", "turnover", "attrition",
                    "evidence of bias", "evidence of a pattern",
                    "evidence of systemic", "formula", "weighted",
                    "scoring system", "test result", "test score",
                    "pass rate", "test", "examination",
                    "mean", "median", "average", "analysis",
                    "data", "representation",
                ],
                0.5,
            ),
        ]

        for markers, wt in tiers:
            for m in markers:
                c = t.count(m)
                if c:
                    s += min(c, 8) * wt

        digit_ratio = sum(1 for ch in t if ch.isdigit()) / tl
        s += min(digit_ratio * 28.0, 2.0)

        pct_hits = len(re.findall(r"\d+(\.\d+)?\s*(%|percent|pct)", t))
        s += min(pct_hits, 10) * 0.28

        ratio_hits = len(re.findall(r"\d+(\.\d+)?\s*(:|/|to)\s*\d+(\.\d+)?", t))
        s += min(ratio_hits, 5) * 0.4

        year_hits = len(re.findall(r"\b(19[5-9]\d|20[0-2]\d)\b", t))
        s += min(year_hits, 5) * 0.08

        system_hits = len(re.findall(
            r"(hiring|promotion|promotional|promotion|pay|wage|salary|termination|recruitment|recruiting|admission|transfer|assignment|seniority|benefit|leave|layoff|evaluation|testing|test|examination|training|advancement|compensation)\s+(practice|practices|policy|policies|procedure|procedures|formula|formulas|formulae|system|systems|test|tests|requirement|requirements|criterion|criteria|standard|standards)",
            t,
        ))
        s += min(system_hits, 6) * 0.55

        pat_hits = 0
        for pre in ("hiring", "promotion", "promotional", "pay", "salary",
                    "termination", "recruitment", "admission", "transfer",
                    "workplace", "department", "company", "employer",
                    "agency", "promotion"):
            for suf in ("discriminat", "bias", "imbalance"):
                pat_hits += t.count(pre + " " + suf)
                pat_hits += t.count(pre + "-" + suf)
        s += min(pat_hits, 4) * 0.7

        impact_phrases = [
            "effect of", "effects of", "result of",
            "legacy of", "impact of", "impact on",
            "consequence of", "due to past",
            "resulted in", "caused by",
        ]
        ip = sum(t.count(p) for p in impact_phrases)
        s += min(ip, 4) * 0.28

        context_phrase = [
            "evidence of a", "pattern of", "practice of",
            "history of", "statistics", "statistical",
            "percentages", "proportions", "data",
            "comparison", "disparit", "underrepresent",
            "under-represent", "formula",
        ]
        cp = sum(t.count(p) for p in context_phrase)
        s += min(cp, 10) * 0.16

        doc_indicators = [
            "eeoc", "equal employment", "title vii",
            "civil rights act", "class action", "class of",
            "decree", "findings of fact", "expert witness",
            "laboratory", "sociologist", "statistician",
        ]
        di = sum(1 for p in doc_indicators if p in t)
        s += min(di, 5) * 0.18

        if "systemic" in t or "systematic" in t:
            s += 1.0
        if "discriminat" in t and (
            "finding" in t or "found" in t or "prior" in t
        ):
            s += 0.7

        if s > 5.5:
            s += math.log(s / 5.5) * 1.8

        if tl < 1500:
            s *= 0.92
        elif tl < 6000:
            s *= 0.98

        return min(round(s / 10.0, 4), 1.0)
    except Exception:
        return 0.5