def score(text: str) -> float:
    if not text or not isinstance(text, str):
        return 0.5
    try:
        t = text.lower()
        import re

        # ---- A) Government/defendant-entity signal ----
        gov_gov = [
            "city of", "county of", "state of", "commonwealth of",
            "united states", "department of ", "u.s. department",
            "school district", "board of education", "board of supervisors",
            "municipal", "public employer", "government", "governmental",
            "state agency", "state actor", "public agency", "public entity",
            "police department", "fire department", "fdny", "nypd",
            "school board", "regents of", "trustees of", "public university",
            "civil service", "public safety", "firefighter", "first responder",
            "emergency medical", "paramedic", "emt", "public health",
        ]
        gov_hits = sum(1 for g in gov_gov if g in t)
        gov_score = min(gov_hits / 3.0, 1.0)

        # ---- B) Affirmative-action / remedial-plan signal ----
        aa_plan = [
            "affirmative action", "affirmative-action",
            "remedial plan", "remedy past", "remedying past",
            "remediate past", "remedy discrimination",
            "race-conscious", "race conscious", "race-conscious plan",
            "minority preference", "gender-conscious",
            "set-aside", "set aside",
            "quota", "50/50", "one-for-one", "goals and timetables",
            "hiring goal", "promotion goal", "representation goal",
            "workforce goal", "minority business",
            "preference program", "preference plan",
            "croson", "adarand", "grutter", "gratz", "fisher",
            "bakke", "wygant", "fullilove", "metro broadcasting",
            "narrowly tailored", "compelling interest", "strict scrutiny",
            "remedial purpose", "remedial objective", "remedial measure",
            "remedial program", "diversity plan", "diversity initiative",
            "equity initiative", "inclusion plan", "inclusionary",
            "manifest imbalance", "underrepresentation", "under-representation",
            "disparity study", "utilization analysis",
            "race-neutral alternative", "raceneutral",
            "address past discrimination", "remedy the effects",
            "effects of past discrimination", "past discriminatory",
            "correct the effects",
        ]
        aa_hits = sum(1 for term in aa_plan if term in t)
        aa_score = min(aa_hits / 2.0, 1.0)

        # ---- C) Plaintiff-as-government-official (strong 0-signal) ----
        pl_gov = [
            "filed charges with the eeo",
            "charges with the equal employment",
            "charge of discrimination with",
            "filed a charge",
            "filed charges",
            "plaintiffs filed",
            "filed a complaint with the eeo",
            "eeoc took no action",
            "eeoc has not investigated",
            "reasonable cause determination",
            "right to sue",
            "notice of right to sue",
            "the eeo",
        ]
        plaintiff_gov = sum(1 for g in pl_gov if g in t)
        plaintiff_is_gov_official = plaintiff_gov >= 1

        # ---- D) Mere EEOC-data-collection (0-signal override) ----
        eeoc_data = [
            "eeo-1", "eeo1", "eeo 1",
            "employer information report",
            "pay data", "component 2", "paydata",
            "data collection", "data-collection",
            "collecting pay data",
        ]
        eeoc_data_hits = sum(1 for g in eeoc_data if g in t)
        aa_plan_signals = sum(
            1 for term in ["affirmative action", "remedial plan", "race-conscious",
                           "50/50", "quota", "hiring goal", "promotion goal",
                           "preference program", "croson", "adarand",
                           "grutter", "bakke", "wygant", "fullilove"]
            if term in t
        )
        eeoc_data_only = (eeoc_data_hits >= 2 and aa_plan_signals == 0)

        # ---- E) Individual-plaintiff-discrimination 0-signal ----
        ind_disc = [
            "failed to hire", "failed to promote", "terminated",
            "discharged", "demoted", "suspended", "wrongful termination",
            "retaliated against", "hostile work environment",
            "individual disparate", "age discrimination",
            "disability discrimination", "religious accommodation",
            "pregnancy discrimination", "harassed",
        ]
        ind_disc_hits = sum(1 for g in ind_disc if g in t)
        individual_discrimination = ind_disc_hits >= 2

        # ---- F) EEOC-as-prosecutor (0-signal) ----
        eeoc_prosecutor = (
            ("eeoc" in t or "equal employment opportunity commission" in t) and
            ("on behalf of" in t or "brought suit" in t or
             "the commission" in t or "intervened" in t) and
            aa_plan_signals == 0
        )

        # ---- Overrides -> 0-signal ----
        if eeoc_data_only or plaintiff_is_gov_official or eeoc_prosecutor:
            return 0.05

        # ---- Combine A + B ----
        if aa_score == 0:
            aa_combined = 0.0
        else:
            aa_combined = aa_score * (0.45 + 0.55 * gov_score)

        # ---- Apply individual-discrimination penalty ----
        if individual_discrimination:
            aa_combined *= 0.4

        # ---- Final mapping ----
        if aa_combined <= 0.01:
            return 0.03
        elif aa_combined <= 0.05:
            return 0.10
        elif aa_combined <= 0.15:
            return 0.22
        elif aa_combined <= 0.30:
            return 0.40
        elif aa_combined <= 0.50:
            return 0.58
        elif aa_combined <= 0.70:
            return 0.72
        elif aa_combined <= 0.85:
            return 0.82
        elif aa_combined <= 0.95:
            return 0.90
        else:
            return 0.95

    except Exception:
        return 0.5