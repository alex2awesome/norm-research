"""a46 hybrid: specific neutral employment practice identified.

Criterion: a specific facially-neutral employment practice (test,
requirement, screen) is named in the narrative — e.g. a named test
("Wonderlic Test", "Drug Analysis Proficiency Test"), a degree/rank/
experience prerequisite, a background/medical/polygraph screen, a formal
multi-factor performance-evaluation, or a written company policy applied
to the plaintiff. It is NOT satisfied by generic discrimination/harassment/
retaliation language, or by procedural machinery (EEOC charge, grievance
process, statute of limitations) that isn't itself the challenged practice.

Baseline (v0_holistic) is a fixed keyword/phrase list (named test-type
nouns, a short enumerated list of named practices, a narrow "requirement
language" regex, a numeric-cutoff regex, "facially neutral" boilerplate).
It fires on the small set of cases that use its exact vocabulary
(train_rho only 0.286) but misses most real positives because legal
narratives name practices in open-ended, non-formulaic prose: a proper-noun
test the list never enumerated, a job posting's requirement list, a
multi-factor evaluation described narratively ("evaluating ... on the
basis of X, Y, and Z"), a bachelor's-degree prerequisite, a fitness-for-
duty evaluation, a named written policy invoked for discipline. Regex
cannot recognize an open-ended set of practice names/descriptions — that
needs an LLM read of the passage. So we keep the baseline's regex families
as a code-side backstop (broadened slightly to structural patterns —
Proper-Noun + Test/Policy/Program/Evaluation/..., degree/prerequisite
language, an evaluation-criteria list construction — rather than
enumerating the literal train instances), and add two THICK-INPUT LLM
fields: what specific practice (if any) is named, and what employment
decision it gated. The gating question is the actual predicate that
separates a genuine challenged practice from incidental procedural nouns
(a "grievance procedure" or "EEOC charge" is not an employment practice
under this criterion even though it sounds official) — that AND/OR
combination is done in code on the LLM's short answers, not by the LLM.
"""
import re
import math


LLM_FIELDS = {
    "named_practice": (
        "State the specific facially-neutral employment test, requirement, "
        "screening procedure, or evaluation practice named in this document "
        "(e.g. a named test, a degree/experience/rank requirement, a "
        "background/medical/psychological/polygraph screen, or a written "
        "policy applied to the plaintiff) in <=10 words; answer NONE if no "
        "such specific practice is named."
    ),
    "practice_decision": (
        "What employment decision did that named practice gate or lead to "
        "(e.g. hiring, promotion, termination, discipline, assignment)? "
        "Answer in a few words, or NONE if no such practice is described."
    ),
}


def _sat(x, k=1.0):
    return 1.0 - math.exp(-x / max(1e-6, k))


_NONE_MARKERS = (
    "none", "n/a", "na", "not applicable", "not specified", "not identified",
    "not named", "unclear", "unknown", "no specific", "not present",
    "does not name", "not describe", "not mentioned",
)


def _none_ish(s):
    if not s:
        return True
    s2 = s.strip().lower()
    if s2 == "":
        return True
    return any(m in s2 for m in _NONE_MARKERS)


_CONCRETE_RE = re.compile(
    r"\b(test|exam|assess|screen|check|evaluat|polic|require|criteri|"
    r"standard|degree|certif|physical|written|credit|background|polygraph|"
    r"drug|height|weight|score|diploma|licen[sc]e|proficiency)"
)

# Structural, generalized patterns (not literal train answers): a
# capitalized proper-noun phrase followed by a practice-type noun. Catches
# named tests/policies/programs the fixed baseline list never enumerated.
_NAMED_TEST_POLICY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z'-]*\s+){1,4}(?:Test|Exam|Examination|Assessment|"
    r"Evaluation|Policy|Program|Plan|Statement|Report|Requirement|Standard|"
    r"Screening|Check)\b"
)

# Baseline's original signal families, kept as a code-side backstop.
_GENERIC_PRACTICE_RE = re.compile(
    r"\b(?:written|physical|aptitude|entrance|qualifying|standardized|"
    r"physical agility|cognitive|polygraph|psychological|medical|"
    r"fitness[- ]for[- ]duty)\s+(?:test|exam|examination|assessment|"
    r"evaluation|screen(?:ing)?)\b"
)
_NAMED_PRACTICE_RE = re.compile(
    r"\b(?:seniority system|credit check|background check|background "
    r"investigation|polygraph|drug test|physical agility test|written "
    r"examination|high school diploma requirement|minimum height "
    r"requirement|minimum weight requirement|screening criteria|degree "
    r"requirement|certification requirement)\b"
)
_REQUIREMENT_LANG_RE = re.compile(
    r"\brequir(?:e|es|ed|ement|ements|ing).{0,50}\b(?:applicants|candidates|"
    r"employees|for (?:the|this) position)\b"
)
_NUMERIC_SPECIFICITY_RE = re.compile(
    r"\b(?:cutoff|passing|minimum)\s+(?:score|grade)\s+of\s+\d+|\bscore of "
    r"\d+|\d+%?\s*(?:or (?:higher|above|lower))"
)
_NEUTRALITY_LANG_RE = re.compile(
    r"\b(?:facially neutral|applies to all|objective criteria|applicable "
    r"to all employees)\b"
)

# New generalized additions: degree/prerequisite language, a job-posting
# "requirements including ..." list, and a narrative multi-factor
# evaluation construction ("evaluated ... on the basis of X, Y, and Z").
_DEGREE_PREREQ_RE = re.compile(
    r"\b(?:bachelor|associate|master|high school)(?:'s)?\s+(?:degree|"
    r"diploma)\b|\bprerequisite\b"
)
_REQUIREMENT_LIST_RE = re.compile(
    r"\brequirements?\s+(?:includ\w*|of|such as)\b"
)
_EVAL_CRITERIA_LIST_RE = re.compile(
    r"\bevaluat\w*.{0,100}\bbasis of\b"
)


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        norm = ops.normalize(raw) if raw else ""
        low = norm.lower()

        # --- code-side backstop / structural signals ---
        named_hits = (
            len(_NAMED_TEST_POLICY_RE.findall(norm))
            + len(_NAMED_PRACTICE_RE.findall(low))
            + 0.5 * len(_GENERIC_PRACTICE_RE.findall(low))
        )
        requirement_hits = (
            len(_REQUIREMENT_LANG_RE.findall(low))
            + len(_REQUIREMENT_LIST_RE.findall(low))
        )
        numeric_hits = len(_NUMERIC_SPECIFICITY_RE.findall(low))
        degree_hits = len(_DEGREE_PREREQ_RE.findall(low))
        eval_hits = len(_EVAL_CRITERIA_LIST_RE.findall(low))
        neutrality_hits = len(_NEUTRALITY_LANG_RE.findall(low))

        code_component = (
            0.30 * _sat(named_hits)
            + 0.15 * _sat(requirement_hits)
            + 0.15 * _sat(numeric_hits)
            + 0.15 * _sat(degree_hits)
            + 0.15 * _sat(eval_hits)
            + 0.10 * _sat(neutrality_hits)
        )
        code_component = max(0.0, min(1.0, code_component))

        # --- LLM-grounded predicate: what practice, gating what decision ---
        ext = extracted or {}
        practice_raw = str(ext.get("named_practice", "") or "")
        decision_raw = str(ext.get("practice_decision", "") or "")

        practice_present = not _none_ish(practice_raw)
        decision_present = not _none_ish(decision_raw)
        concrete = practice_present and bool(
            _CONCRETE_RE.search(practice_raw.lower())
        )

        llm_component = 0.0
        if practice_present:
            llm_component = 0.7 if concrete else 0.4
            if decision_present:
                llm_component = min(1.0, llm_component + 0.3)

        val = 0.55 * llm_component + 0.45 * code_component
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
