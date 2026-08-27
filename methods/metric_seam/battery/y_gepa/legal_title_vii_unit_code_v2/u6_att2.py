import re
import math

def score(text: str) -> float:
    if not text or not isinstance(text, str):
        return 0.0
    try:
        t = text.lower()
        N = len(t)

        # --- 1. Government entity (defendant must be govt) ---
        gov_entities = [
            "city of", "county of", "state of", "commonwealth of",
            "united states", "department of", "u.s. department",
            "school district", "board of education", "board of supervisors",
            "municipal", "public employer", "governmental", "government",
            "state agency", "public agency", "public entity",
            "police department", "fire department", "fdny", "nypd",
            "school board", "regents of", "trustees of", "public university",
            "civil service", "public safety",
            "mayor", "city council", "governor", "state legislature",
            "attorney general", "secretary of", "commissioner",
            "department of health", "department of labor",
            "department of justice", "department of education",
            "department of transportation",
            "emergency medical", "paramedic", "emt", "first responder",
        ]
        gov_hits = sum(1 for g in gov_entities if g in t)
        gov_score = min(gov_hits / 3.0, 1.0)

        # --- 2. Federal/civil-rights regulatory body ---
        regulatory = [
            "eeoc", "equal employment opportunity commission",
            "office of federal contract compliance", "ofccp",
            "civil rights act", "title vii",
            "section 1981", "section 1983",
        ]
        reg_hits = sum(1 for r in regulatory if r in t)
        reg_score = min(reg_hits / 2.0, 1.0)

        # --- 3. Affirmative action / remedial plan keywords ---
        aa_plan = [
            "affirmative action", "affirmative-action",
            "remedial plan", "remedial program", "remedial measure",
            "remedy past", "remedying past", "remediate past",
            "race-conscious", "race conscious", "race-conscious plan",
            "gender-conscious",
            "set-aside", "set aside",
            "hiring goal", "promotion goal", "representation goal",
            "workforce goal", "minority business",
            "preference program", "preference plan",
            "diversity plan", "diversity initiative",
            "equity initiative", "inclusion plan", "inclusionary",
            "narrowly tailored", "compelling interest", "strict scrutiny",
            "remedial purpose", "remedial objective",
            "manifest imbalance", "underrepresentation", "under-representation",
            "disparity study", "utilization analysis",
            "race-neutral alternative", "raceneutral",
            "address past discrimination", "remedy the effects",
            "effects of past discrimination", "past discriminatory",
            "correct the effects",
            "50/50", "one-for-one", "goals and timetables",
            "croson", "adarand", "grutter", "gratz", "fisher",
            "bakke", "wygant", "fullilove", "metro broadcasting",
        ]
        aa_hits = sum(1 for term in aa_plan if term in t)
        aa_score = min(aa_hits / 2.0, 1.0)

        # --- 4. Data collection / reporting mandate ---
        data_mandate = [
            "eeo-1", "eeo1", "employer information report",
            "pay data", "workforce data", "data collection",
            "demographic data", "collect pay data",
            "component 2", "pay-data",
            "employer survey", "reporting requirement",
            "recordkeeping", "record-keeping",
            "equal employment opportunity data",
            "minority utilization", "workforce composition",
            "labor availability",
        ]
        data_hits = sum(1 for d in data_mandate if d in t)
        data_score = min(data_hits / 2.0, 1.0)

        # --- 5. Public health / emergency safety mandate ---
        emergency = [
            "covid", "covid-19", "coronavirus", "pandemic",
            "state of emergency", "local emergency", "public health",
            "emergency declaration", "emergency order",
            "vaccination", "vaccine", "vaccinated", "immunization",
            "face covering", "mask", "ppe", "social distancing",
            "stay-at-home", "shelter-in-place", "lockdown",
            "reopening plan", "reopen plan",
            "essential worker", "first responder", "emergency responder",
            "emergency safety", "public health emergency",
            "emergency measure", "emergency mandate",
            "department of public health",
            "health officer", "health authority",
            "quarantine", "isolation",
        ]
        emerg_hits = sum(1 for e in emergency if e in t)
        emerg_score = min(emerg_hits / 2.0, 1.0)

        # --- 6. Government defense posture ---
        defense = [
            "defendant", "appellee", "respondent",
            "moved to dismiss", "motion to dismiss",
            "summary judgment", "qualified immunity",
            "government interest", "legitimate interest",
            "compelling governmental", "public interest",
            "defends", "defended", "justified",
            "court upholds", "court sustained",
            "court affirmed",
        ]
        defense_hits = sum(1 for d in defense if d in t)
        defense_score = min(defense_hits / 3.0, 1.0)

        # --- 7. Plaintiff challenge (must be challenging the plan) ---
        challenge = [
            "plaintiff", "challenged", "challenge", "challenges",
            "contends", "argues", "alleges", "alleged",
            "discriminatory", "reverse discrimination",
            "unconstitutional", "equal protection",
            "fourteenth amendment", "equal protection clause",
            "sued", "lawsuit", "class action",
            "injunction", "declaratory",
            "moved for summary",
        ]
        challenge_hits = sum(1 for c in challenge if c in t)
        challenge_score = min(challenge_hits / 3.0, 1.0)

        # --- 8. Individual employment dispute (negative signal) ---
        individual_emp = [
            "terminated", "fired", "suspended",
            "sexual harassment", "hostile work environment",
            "reasonable accommodation", "ada",
            "age discrimination", "adea",
            "pregnancy", "pda",
            "retaliation", "whistleblower",
            "overtime", "flsa", "wage",
            "wrongful termination", "wrongful discharge",
            "individual capacity",
        ]
        indiv_hits = sum(1 for i in individual_emp if i in t)

        # === Scoring logic ===
        govt_basis = gov_score * 0.55 + reg_score * 0.45
        plan_basis = aa_score * 0.45 + data_score * 0.25 + emerg_score * 0.30

        if gov_score == 0 and reg_score == 0:
            return 0.0

        if plan_basis < 0.08:
            return 0.0

        challenge_factor = 0.55 + 0.45 * challenge_score

        defense_factor = max(
            defense_score * 0.40,
            min(gov_score + reg_score, 1.0) * 0.30
        )

        penalty = 1.0
        if indiv_hits >= 3 and aa_score == 0:
            penalty *= 0.4
        if emerg_score == 0 and data_score == 0 and aa_score == 0:
            penalty *= 0.5

        raw = govt_basis * (0.45 + 0.55 * plan_basis) * challenge_factor * (0.75 + 0.25 * defense_factor) * penalty

        return min(raw, 1.0)
    except Exception:
        return 0.0