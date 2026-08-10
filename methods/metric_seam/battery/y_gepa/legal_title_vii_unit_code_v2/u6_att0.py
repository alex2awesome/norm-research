def score(text: str) -> float:
    import re

    if not text or not isinstance(text, str):
        return 0.5
    try:
        t = text.lower()
        low = t
        up = text.upper()

        # 1) Government/defendant-entity signal
        gov_terms = [
            "united states", "state of ", "commonwealth of", "city of ",
            "county of", "department of", "u.s. department", "federal",
            "municipality", "school district", "board of education",
            "board of supervisors", "public employer", "government",
            "state agency", "state employer", "public agency",
            "state defendant", "governmental", "state actor",
            "public safety", "emergency management", "public health",
            "secretary of", "attorney general", "civil service",
            "public university", "regents of", "trustees of",
            "commission", "authority", "metropolitan",
        ]
        gov_hits = sum(1 for g in gov_terms if g in low)
        gov_score = min(gov_hits / 4.0, 1.0)

        # 2) Affirmative-action / remedial-plan signal
        aa_terms = [
            "affirmative action", "affirmative-action",
            "remedial plan", "remedy past", "remedying past",
            "remediate past", "remedy discrimination",
            "remedial measure", "remedial program",
            "diversity plan", "inclusion plan",
            "set-aside", "set aside program", "minority business",
            "minority contracting", "minority preference",
            "race-conscious", "race conscious", "gender-conscious",
            "concordia", "croson", "adarand", "grutter", "gratz",
            "fisher", "bakke", "wygant", "fullilove", "metro broadcasting",
            "preference program", "preference plan",
            "compelling interest", "narrowly tailored",
            "strict scrutiny", "intermediate scrutiny",
            "diversity interest", "educational diversity",
            "race-neutral alternative", "raceneutral alternative",
            "underrepresentation", "manifest imbalance",
            "remedial purpose", "remedial objective",
            "correct past", "address past discrimination",
            "inclusionary", "outreach program",
            "diversity initiative", "equity initiative",
            "representation goal", "workforce goal",
            "data collection", "data-collection",
            "disparity study", "utilization analysis",
            "equal employment opportunity plan",
        ]
        aa_hits = sum(1 for term in aa_terms if term in low)
        aa_score = min(aa_hits / 3.0, 1.0)

        # 3) Title VII / employment context
        title_terms = [
            "title vii", "title 7", "employment discrimination",
            "equal employment", "eeoc", "employer",
            "employment action", "adverse employment",
            "workplace", "hiring", "promotion",
        ]
        title_hits = sum(1 for term in title_terms if term in low)
        title_score = min(title_hits / 3.0, 1.0)

        # 4) Defendant-defending signal (motion posture)
        defense_terms = [
            "defendant argues", "defendants argue",
            "defendant contends", "defendants contend",
            "defendant maintains", "defendants maintain",
            "defendant asserts", "defendants assert",
            "defendant responds", "defendants respond",
            "defendant's position", "defendants' position",
            "government argues", "government contends",
            "state argues", "state contends",
            "city argues", "city contends",
            "county argues", "county contends",
            "summary judgment", "motion to dismiss",
            "defendant moves", "defendants move",
            "qualified immunity",
        ]
        defense_hits = sum(1 for term in defense_terms if term in low)
        defense_score = min(defense_hits / 3.0, 1.0)

        # 5) Plaintiff challenging signal
        challenge_terms = [
            "plaintiff challenges", "plaintiffs challenge",
            "plaintiff alleges", "plaintiffs allege",
            "plaintiff contends", "plaintiffs contend",
            "plaintiff asserts", "plaintiffs assert",
            "challenges the", "challenge to the",
            "attacks the", "contests the",
            "unconstitutional", "violates equal protection",
            "equal protection", "reverse discrimination",
            "reverse-discrimination", "race discrimination",
            "title vii claim", "discrimination claim",
            "plaintiff sued", "plaintiffs sued",
        ]
        challenge_hits = sum(1 for term in challenge_terms if term in low)
        challenge_score = min(challenge_hits / 3.0, 1.0)

        # 6) Past-discrimination remedy emphasis
        remedy_terms = [
            "past discrimination", "prior discrimination",
            "historical discrimination", "history of discrimination",
            "effects of past", "lingering effects",
            "systemic discrimination", "pattern of discrimination",
            "societal discrimination", "remedial justification",
            "compelling interest in diversity",
            "underrepresentation of", "historical exclusion",
            "legacy of discrimination",
        ]
        remedy_hits = sum(1 for term in remedy_terms if term in low)
        remedy_score = min(remedy_hits / 2.0, 1.0)

        # 7) Emergency / public-health / safety overlay
        emergency_terms = [
            "emergency", "public health", "public safety",
            "pandemic", "epidemic", "covid", "vaccination mandate",
            "vaccine mandate", "emergency declaration",
            "disaster response", "crisis response",
            "first responder", "essential worker",
            "emergency responder", "life safety",
        ]
        emergency_hits = sum(1 for term in emergency_terms if term in low)
        emergency_score = min(emergency_hits / 3.0, 1.0)

        # --- Require: government + (affirmative action OR emergency context) + challenge ---
        # Core required: government entity AND remedial/aa plan AND plaintiff challenging it
        has_gov = gov_hits >= 2
        has_aa = aa_hits >= 1
        has_title = title_hits >= 1
        has_challenge = challenge_hits >= 1
        has_defense = defense_hits >= 1

        # Gate: must have government, title-vii-ish context, and at least one AA/remedial signal
        gate_passed = has_gov and has_title and (has_aa or emergency_hits >= 2)

        if not gate_passed:
            # weak partial signal if some ingredients present
            partial = (
                0.10 * gov_score
                + 0.05 * aa_score
                + 0.05 * title_score
                + 0.05 * challenge_score
            )
            return round(min(0.25, partial) * 10, 2)

        # Weighted combination
        raw = (
            0.22 * gov_score
            + 0.25 * aa_score
            + 0.10 * title_score
            + 0.10 * defense_score
            + 0.13 * challenge_score
            + 0.12 * remedy_score
            + 0.08 * emergency_score
        )

        # Strong indicator bonuses
        strong_indicators = [
            "affirmative action",
            "race-conscious",
            "remedy past discrimination",
            "remedial plan",
            "narrowly tailored",
            "compelling interest",
            "strict scrutiny",
            "croson",
            "adarand",
            "grutter",
            "fisher",
            "fullilove",
            "wygant",
            "vaccine mandate",
            "public health emergency",
            "vaccination requirement",
        ]
        strong_count = sum(1 for s in strong_indicators if s in low)
        if strong_count >= 2:
            raw += 0.10

        # Cap and scale to 0-10
        raw = max(0.0, min(1.0, raw))
        result = round(raw * 10, 2)
        return result

    except Exception:
        return 0.5