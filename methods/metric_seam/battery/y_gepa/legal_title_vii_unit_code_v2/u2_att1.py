def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        import re
        t = text.lower()
        tl = len(t)

        if tl < 50:
            return 0.0

        # --- Severe harassment indicators (physical/sexual violence) ---
        severe = [
            "raped", "sexual assault", "sexually assaulted", "sodomized",
            "physical assault", "physically attacked", "struck ", "struck her",
            "punched", "slapped", "shoved", "grabbed her by", "grabbed him by",
            "grabbed her breast", "grabbed his crotch", "strangled", "choked her",
            "choked him", "burned a cross", "noose", "death threat",
            "threatened to kill", "threatened to shoot", "threatened to rape",
            "brandished", "pointed a gun", "pulled a knife", "exposed himself",
            "exposed herself", "forced her to", "forced him to", "quid pro quo",
            "demanded sexual", "masturbat", "pornography on", "groped her",
            "groped his", "fondled her", "fondled his", "indecent exposure",
            "criminal sexual conduct",
        ]
        severe_hits = 0
        for term in severe:
            c = t.count(term)
            severe_hits += min(c, 3)

        # --- Pervasive harassment (repeated conduct) ---
        pervasive_terms = [
            "repeatedly", "daily", "weekly", "constantly", "continuously",
            "numerous occasions", "numerous times", "several times",
            "regularly", "on a daily basis", "every day", "for months",
            "for years", "throughout her employment", "throughout his employment",
            "ongoing", "persistent", "over a period", "systematic",
        ]
        pervasive_hits = 0
        for term in pervasive_terms:
            if term in t:
                pervasive_hits += 1

        # --- Moderate verbal harassment (slurs, comments) ---
        moderate = [
            "n-word", "n word", "nigger", "racial slur", "racial slurs",
            "racial epithet", "racial epithets", "racist remarks",
            "racist comments", "racist jokes", "monkey", "spic", "wetback",
            "chink", "gook", "kike", "spook", "coon", "porch monkey",
            "sexual comments", "lewd comments", "vulgar comments",
            "obscene comments", "derogatory comments", "sexist remarks",
            "sexist comments", "derogatory remarks", "humiliating",
            "degrading", "dehumanizing", "intimidating", "menacing",
            "threatening", "hostile", "abusive",
        ]
        moderate_hits = 0
        for term in moderate:
            c = t.count(term)
            moderate_hits += min(c, 3)

        # --- Specific evidence markers ---
        specificity_markers = [
            "testified that", "sworn statement", "affidavit", "witnessed",
            "corroborat", "documented", "e-mail", "email", "text message",
            "text messages", "texted", "sent a text", "recorded",
            "recording", "surveillance", "video", "photograph", "photo",
            "screenshot", "performance review", "incident report",
            "wrote a letter", "complained to", "reported the", "human resources",
            "specifically stated", "on one occasion", "on another occasion",
            "for example", "such as when", "including when", "instance",
            "incidents", "episode", "episodes", "occurred on",
        ]
        specificity_hits = 0
        for term in specificity_markers:
            c = t.count(term)
            specificity_hits += min(c, 3)

        # --- Physical (non-sexual) harassment ---
        physical = [
            "blocked her", "blocked him", "blocked the door", "cornered her",
            "cornered him", "followed her", "followed him", "stood so close",
            "invaded her personal space", "invaded his personal space",
            "touched her", "touched his", "brushed against", "bumped her",
            "rubbed her", "massaged her shoulders", "pinched her",
            "put his arm around", "put his hand on", "put his hands on",
            "tried to kiss", "kissed her", "hugged her", "made a pass at",
            "propositioned her",
        ]
        physical_hits = 0
        for term in physical:
            c = t.count(term)
            physical_hits += min(c, 3)

        # --- Hostile work environment general ---
        hwe_mentions = t.count("hostile work environment")
        harassment_mentions = t.count("harassment")
        discrimination_mentions = t.count("discrimination")

        # --- Compute weighted score ---
        raw = (
            severe_hits * 2.0
            + pervasive_hits * 1.2
            + moderate_hits * 0.8
            + specificity_hits * 0.7
            + physical_hits * 1.5
            + hwe_mentions * 0.5
        )

        # Mild boost for harassment mentions (cap so pure "pay disparity" type docs stay low)
        raw += min(harassment_mentions, 8) * 0.15

        # Discount if mostly about pay/election/policy/constitutional issues
        non_harassment_signals = 0
        for term in [
            "pay disparity", "wage", "salary", "compensation", "equal pay",
            "promotional", "promotion", "demotion", "termination", "fired",
            "discharge", "layoff", "lay off", "reduction in force",
            "qualified", "pretext", "burden", "prima facie", "bona fide",
            "election", "covid", "pandemic", "vaccin", "constitutional",
            "first amendment", "free speech", "due process",
            "report eeo", "eeo-1", "information report",
            "administrative", "regulatory",
        ]:
            if term in t:
                non_harassment_signals += 1

        if non_harassment_signals >= 4 and severe_hits == 0 and physical_hits == 0:
            raw *= 0.3
        elif non_harassment_signals >= 3 and severe_hits == 0 and physical_hits == 0:
            raw *= 0.5

        result = min(raw / 10.0, 1.0)

        if result < 0.01:
            result = 0.0

        return round(result, 3)

    except Exception:
        return 0.5