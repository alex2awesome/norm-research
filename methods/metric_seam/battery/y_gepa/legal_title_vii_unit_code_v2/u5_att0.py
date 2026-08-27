import re
import math
from collections import Counter

def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = text
        t_low = text.lower()
        length = len(t_low)
        if length < 200:
            return 0.5

        # -----------------------------------------------------------
        # Pattern banks (lowercase);  words keyed with r"\b" context
        # -----------------------------------------------------------
        statistical_indicators = [
            "statistic", "standard deviation", "two or three std",
            "p-value", "p value", "p<", "p <", "significance level",
            "significantly", "chi-square", "chi square",
            "regression analys", "regression model", "logit", "probit",
            "binomial", "multivariate", "multivariable",
            "correlation", "coefficient", "variable",
            "confidence interval", "odds ratio", "relative risk",
            "t-test", "t test", "anova", "z-score", "z score",
            "observed", "expected", "disparity ratio",
            "four-fifths", "four fifths", "4/5", "80%",
            "80 percent", "80% rule", "adverse impact",
            "standard error", "r-squared", "r squared",
            "median", "mean", "average", "percentile",
            "sampling", "sample", "proportional", "proportion",
            "percentage", "ratio", "distribution",
            "measures", "measured", "quantif",
        ]

        systemic_workforce = [
            "underrepresent", "under-represent", "overrepresent",
            "over-represent", "underutili", "workforce analy",
            "work force analy", "workforce compos",
            "demographic", "labor pool", "labour pool",
            "applicant pool", "applicant flow", "relevant labor market",
            "relevant labour market", "relevant labor", "comparison population",
            "population comparison", "representation", "compositional data",
            "hire rate", "hiring rate", "selection rate", "promotion rate",
            "termination rate", "attrition rate", "turnover rate",
            "pass rate", "fail rate", "acceptance rate",
            "incumbency", "incumbent", "headcount",
            "minority", "female", "women", "african american",
            "hispanic", "asian", "protected class", "protected group",
        ]

        numerical_pattern = (
            r"\b(?:\d{1,3}(?:,\d{3})+|\d+\.?\d*)\s*%'
            r"|\b\d{1,3}(?:,\d{3})+\s+(?:employee|worker|applicant|person|people|individual|hire|hiring|promotion|female|male|minorit)"
            r"|\b\d+\.?\d*\s*(?:std|standard deviation|sigma|z-score|p-value|p<|p <|z score)"
            r"|\bstandard deviation[s]?\s+(?:of|<|>|=|less|more|greater|approximately)"
            r"|\bmore than\s+\d+\s+std"
            r"|\bunder\s+\d+\.?\d*\s*%'
            r"|\bless than\s+\d+\.?\d*\s*%'
            r"|\bonly\s+\d+\.?\d*\s*%'
            r"|\bn\s*=\s*\d"
            r"|\btable\s+[0-9ivxlcdm]+\b"
        )

        findings_bias = [
            "finding of discrimination", "found to discriminate",
            "finding of disparate", "found to have discriminated",
            "previously found", "prior finding", "pattern or practice of discrimination",
            "pattern of discrimination", "practice of discrimination",
            "history of discrimination", "engaged in discrimination",
            "liable for discrimination", "adjudicated", "consent decree",
            "back pay", "front pay", "class-wide relief",
            "systemic discriminat", "monell", "evidence of a pattern",
            "evidence of systemic", "widespread discriminat",
            "repeated acts of discriminat", "recurring discriminat",
            "discriminatory pattern", "pattern of bias",
        ]

        formulas_flawed = [
            "promotional formula", "promotion formula", "scoring system",
            "selection criter", "selection procedure", "selection device",
            "promotion criter", "hiring criter", "test format",
            "cut score", "cutoff score", "cutoff", "passing score",
            "validity stud", "validation stud", "job relatedness",
            "job-related", "business necessity", "arbitrary",
            "flawed", "defective", "manipulat", "biased test",
            "biased exam", "biased formula", "weighted",
            "weighting", "composite score",
        ]

        class_collective = [
            "class action", "collective action", "class member",
            "class-wide", "class certification", "rule 23",
            "pattern or practice", "systemic",
        ]

        evidentiary = [
            "expert report", "statistical expert", "labor economist",
            "econometric", "labor statist", "dr.", "professor",
            "exhibit", "table", "appendix", "chart", "figure",
            "plt's", "def's", "plaintiff's expert",
            "defendant's expert", "government's expert",
        ]

        # -----------------------------------------------------------
        # Count helper with word-boundary safety
        # -----------------------------------------------------------
        def count_terms(term_list):
            total = 0
            for term in term_list:
                if len(term) <= 3 or term.startswith(("p<", "p <", "p ", "n=")):
                    idx = t_low.find(term)
                    while idx != -1:
                        total += 1
                        idx = t_low.find(term, idx + 1)
                else:
                    pattern = r'\b' + re.escape(term)
                    total += len(re.findall(pattern, t_low))
            return total

        def count_regex(pattern):
            return len(re.findall(pattern, t_low))

        stat_count   = count_terms(statistical_indicators)
        systemic_count = count_terms(systemic_workforce)
        num_count    = count_regex(numerical_pattern)
        finding_count = count_terms(findings_bias)
        formula_count = count_terms(formulas_flawed)
        class_count  = count_terms(class_collective)
        evid_count   = count_terms(evidentiary)

        # Normalise to per-10k characters (robust across opinion length)
        scale = max(length / 10000.0, 1e-6)

        stat_norm   = stat_count / scale
        systemic_norm = systemic_count / scale
        num_norm    = num_count / scale
        finding_norm = finding_count / scale
        formula_norm = formula_count / scale
        class_norm  = class_count / scale

        # -----------------------------------------------------------
        # Sub-scores
        # -----------------------------------------------------------
        raw = 0.55 * (1 - math.exp(-stat_norm / 3.0))
        raw += 0.35 * (1 - math.exp(-systemic_norm / 4.0))
        raw += 0.45 * (1 - math.exp(-num_norm / 4.0))
        raw += 0.30 * (1 - math.exp(-finding_norm / 1.5))
        raw += 0.15 * (1 - math.exp(-formula_norm / 2.0))
        raw += 0.15 * (1 - math.exp(-class_norm / 1.5))
        raw += 0.10 * (1 - math.exp(-evid_count / 4.0))

        # Bonus: stat + systemic co-occurrence implies a workforce stat analysis
        if stat_count >= 3 and systemic_count >= 3:
            raw += 0.10
        if stat_count >= 5 and systemic_count >= 5 and num_count >= 3:
            raw += 0.10

        # -----------------------------------------------------------
        # Strong phrase presence (near-max signal regardless of density)
        # -----------------------------------------------------------
        strong_phrases = [
            "pattern or practice of discrimination",
            "pattern of discrimination",
            "systemic discriminat",
            "found to discriminat",
            "finding of discriminat",
            "history of discriminat",
            "four-fifths rule", "four fifths rule", "80% rule",
            "four-fifths", "four fifths",
            "adverse impact",
            "standard deviation",
            "underrepresent", "under-represent",
            "consent decree", "back pay", "front pay",
            "disparate impact",
            "statistically significant",
            "regression analys", "regression model",
            "selection rate",
            "class-wide relief", "class wide relief",
        ]

        strong_hits = 0
        for phrase in strong_phrases:
            if phrase in t_low:
                strong_hits += 1

        raw += min(0.15, strong_hits * 0.04)

        # -----------------------------------------------------------
        # Signal floor: strong statistical terms present
        # -----------------------------------------------------------
        if stat_count >= 5 and num_count >= 3:
            raw = max(raw, 6.0)
        elif stat_count >= 3 and num_count >= 2:
            raw = max(raw, 4.0)

        # -----------------------------------------------------------
        # Gate: absence of any bias evidence caps score low
        # -----------------------------------------------------------
        if stat_count == 0 and systemic_count == 0 and num_count == 0 and finding_count == 0:
            raw = min(raw, 1.5)

        # -----------------------------------------------------------
        # Clamp to [0, 10]
        # -----------------------------------------------------------
        raw = max(0.0, min(10.0, raw))

        if raw < 0.01:
            return 0.0
        if raw > 9.99:
            return 10.0

        return round(raw, 2)

    except Exception:
        return 0.5