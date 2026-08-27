def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        import re
        t = text.lower()
        tl = len(t)

        if tl < 40:
            return 0.0

        words = t.split()
        word_count = len(words)
        if word_count < 30:
            return 0.0

        score_val = 0.0

        # === ANTI-SIGNAL: non-employment contexts ===
        anti_patterns = [
            r"undergraduate student", r"freshman", r"student at",
            r"university", r"college student", r"campus",
            r"medical (?:procedure|treatment|malpractice)",
            r"doctor (?:harass|abuse)", r"patient", r"nursing home",
            r"school district", r"elementary school", r"high school",
            r"dormitor", r"roommate",
        ]
        anti_count = 0
        for pat in anti_patterns:
            if re.search(pat, t):
                anti_count += 1
        if anti_count >= 2:
            score_val -= 2.5
        elif anti_count >= 1:
            score_val -= 1.2

        # === SEVERE: explicit physical/sexual conduct ===
        severe_terms = [
            "grabbed his", "grabbed her", "grabbed .{0,20}butt", "grabbed .{0,20}breast",
            "grabbed .{0,20}thigh", "grabbed .{0,20}crotch", "grabbed .{0,20}groin",
            "groped", "fondled", "touched her", "touched his", "touched .{0,20}butt",
            "touched .{0,20}breast", "touched .{0,20}thigh", "touched .{0,20}genitals",
            "brushed against", "rubbed against", "rubbed .{0,20}(?:thigh|butt|breast|crotch|groin)",
            "forced (?:him|her) (?:to|into)", "forced (?:oral|sex|intercourse)",
            "demanded (?:oral|sex|sexual)", "demanded sexual",
            "requests? for oral", "oral (?:sex|intercourse)", "vaginal intercourse",
            "anal intercourse", "sexual intercourse", "attempted rape", "raped",
            "sexual assault", "sexually assaulted", "criminal sexual",
            "sodomized", "indecent exposure", "exposed himself", "exposed herself",
            "exposed (?:his|her) genitals", "masturbat", "ejaculated",
            "pulled down", "lifted .{0,20}skirt", "lifted .{0,20}dress",
            "put .{0,20}hand (?:on|down|up)", "slid .{0,20}hand",
            "massaged .{0,20}(?:shoulder|back|neck|leg|thigh)",
            "kissed her", "kissed him", "tried to kiss",
            "unbuttoned", "unzipped", "pulled out (?:his|her)",
            "genitals?", "penis", "vagina", "breasts?", "buttocks?", "crotch",
            "quid pro quo", "sleep with", "have sex with",
            "sexual favor", "sexual advances?", "sexual proposition",
            " propositioned",
            "physical assault", "physically attack", "punched", "slapped",
            "shoved", "strangled", "choked", "brandished",
            "threatened to (?:kill|rape|shoot|harm)", "death threat",
            "pointed (?:a )?gun", "pulled (?:a )?knife",
            "burned .{0,15}cross", "noose", "hanged? .{0,15}effigy",
            "hit her", "hit him", "struck her", "struck him",
            "grabbed her by", "grabbed him by",
        ]
        severe_hits = 0
        severe_set = set()
        for term in severe_terms:
            ms = re.findall(term, t)
            if ms:
                severe_hits += min(len(ms), 4)
                severe_set.add(term)
        score_val += severe_hits * 0.55

        # === SPECIFICITY: body parts + actions ===
        body_parts = ["breast", "buttocks", "thigh", "crotch", "groin", "genitals",
                       "penis", "vagina", "leg", "butt", "chest", "neck", "shoulder"]
        body_hits = 0
        for bp in body_parts:
            c = t.count(bp)
            body_hits += min(c, 3)
        if body_hits >= 2:
            score_val += min(body_hits, 8) * 0.2

        # === SEXUAL HARASSMENT LANGUAGE ===
        sh_terms = [
            "sexual harassment", "sexual advances?", "sexual conduct",
            "sexual comments?", "sexual remarks?", "sexual jokes?",
            "sexually harass", "sexually (?:comment|harass|proposition)",
            "unwelcome (?:sexual|touching|advances|conduct|comments)",
            "inappropriate (?:sexual|touching|conduct|comments|remarks)",
            "sexual behavior", "sexual nature", "of a sexual nature",
            "strip (?:club|joint|joints)", "strip joints?",
            "sexual performance", "sexual virility", "sexual prowess",
            "sexual relations?", "sexual activity",
            "pornograph", "obscene", "lewd", "lascivious",
            "dirty (?:joke|picture|magazine|comment)",
            "sexual pictures?", "sexual images?", "sexual photos?",
            "sexual texts?", "sexting", "inappropriate texts?",
        ]
        sh_hits = 0
        for term in sh_terms:
            if re.search(term, t):
                sh_hits += 1
        score_val += sh_hits * 0.3

        # === SLURS & EPITHETS ===
        slur_terms = [
            "nigger", "nigga", "n-word", "n word", "spic", "wetback", "chink",
            "gook", "kike", "spook", "coon", "porch monkey", "jungle bunny",
            "raghead", "towelhead", "sand nigger", "beaner", "zipperhead",
            "slope", "wop", "dago", "mick", "kraut", "jap", "wet back",
            "faggot", "fag", "dyke", "tranny", "shemale",
            "cunt", "bitch", "whore", "slut", "hooker",
        ]
        slur_hits = 0
        for term in slur_terms:
            c = t.count(term)
            if c > 0:
                slur_hits += min(c, 4)
        score_val += slur_hits * 0.5

        slur_generic = [
            "racial slur", "racial slurs", "racial epithet", "racial epithets",
            "racist remarks?", "racist comments?", "racist jokes?",
            "racially motivat", "racial jokes?", "derogatory slurs?",
            "ethnic slur", "ethnic slurs",
        ]
        sg_hits = 0
        for term in slur_generic:
            if re.search(term, t):
                sg_hits += 1
        score_val += sg_hits * 0.3

        # === PERVASIVENESS: repeated/ongoing ===
        pervasive_terms = [
            "repeatedly", "daily", "weekly", "constantly", "continuously",
            "numerous occasions", "numerous times", "multiple times",
            "several times", "regularly", "on a daily basis", "every day",
            "for months", "for years", "throughout .{0,20}(?:employment|tenure|time)",
            "ongoing", "persistent", "over a period", "systematic",
            "pattern", "course of conduct", "continuing",
            "several years", "multiple (?:occasions|instances|incidents|times)",
            "frequent", "frequency", "constant",
        ]
        pervasive_hits = 0
        for term in pervasive_terms:
            if re.search(term, t):
                pervasive_hits += 1
        score_val += pervasive_hits * 0.25

        duration_matches = re.findall(r'over\s+(?:a\s+)?(?:period\s+of\s+)?(\d+)\s+(year|month|week|day)', t)
        for num_str, unit in duration_matches:
            try:
                num = int(num_str)
                if unit in ("year", "years"):
                    score_val += min(num * 0.3, 1.5)
                elif unit in ("month", "months"):
                    score_val += min(num * 0.15, 1.0)
            except:
                pass

        # === MODERATE: specific discriminatory conduct ===
        moderate_terms = [
            "humiliat", "degrading", "dehumaniz", "intimidat",
            "threatened", "menac", "hostile work", "hostile environment",
            "derogatory comments?", "derogatory remarks?",
            "offensive comments?", "offensive remarks?",
            "vulgar comments?", "vulgar remarks?", "vulgar language",
            "obscene comments?", "obscene remarks?", "obscene language",
            "sexist remarks?", "sexist comments?",
            "demeaning", "belittling", "mocking", "ridicule",
            "offensive jokes?", "vulgar jokes?",
            "explicit (?:language|comments|remarks|photos|pictures)",
            "inappropriate (?:comments|remarks|jokes|touching|behavior|conduct)",
            "unwelcome (?:comments|remarks|touching|advances|conduct|behavior|attention)",
            "made comments? about", "made remarks? about",
            "comments? about .{0,20}(?:body|appearance|sex|breast|butt|looks)",
            "asked about", "inquired about", "questions? about",
            "discriminat", "retaliat",
        ]
        moderate_hits = 0
        moderate_set = set()
        for term in moderate_terms:
            ms = re.findall(term, t)
            if ms:
                moderate_hits += min(len(ms), 4)
                moderate_set.add(term)
        score_val += moderate_hits * 0.18

        # === WITNESS/CORROBORATION ===
        corroboration = [
            r"witness", r"corroborat", r"testified", r"admitted",
            r"confirmed", r"documented", r"reported .{0,20}(?:to|it)",
            r"complained", r"filed .{0,15}complaint", r"e-mail",
            r"email", r"memo", r"letter", r"note", r"recording",
            r"surveillance", r"photograph", r"video",
        ]
        corr_hits = 0
        for pat in corroboration:
            if re.search(pat, t):
                corr_hits += 1
        score_val += min(corr_hits, 5) * 0.15

        # === SPECIFICITY MARKERS (dates, names, quotes, particulars) ===
        date_count = len(re.findall(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}', t))
        date_count += len(re.findall(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}', t))
        date_count += len(re.findall(r'\bin\s+\d{4}\b', t))
        score_val += min(date_count, 8) * 0.08

        quote_count = len(re.findall(r'[A-Z][a-z]+ (?:stated|testified|said|told|wrote|reported)\s+(?:that\s+)?"', text))
        score_val += min(quote_count, 5) * 0.3

        # === EEOC / TITLE VII (employment discrimination context) ===
        eeoc_terms = [
            "title vii", "eeoc", "equal employment",
            "employment discrimination", "hostile work environment",
            "supervisor", "manager", "co-?worker", "coworker",
            "employer", "employee", "workplace", "work environment",
            "termination", "fired", "discharged", "resigned",
            "department", "division", "hired", "position",
        ]
        eeoc_hits = 0
        for term in eeoc_terms:
            if re.search(term, t):
                eeoc_hits += 1
        score_val += min(eeoc_hits, 6) * 0.12

        # === CONCLUSORY ANTI-SIGNAL: bare legal allegations ===
        conclusory = [
            "generalized", "conclusory", "bare allegation", "vague",
            "unsupport", "speculative", "mere speculation",
            "no specific", "without (?:specific|detail|evidence|support)",
            "failed to (?:allege|establish|show|demonstrate|provide)",
            "no evidence", "fails to (?:allege|establish|show|demonstrate|provide)",
            "mere assertion", "unsupported",
        ]
        conc_hits = 0
        for term in conclusory:
            if term in t:
                conc_hits += 1
        score_val -= min(conc_hits, 4) * 0.2

        # === PROSE CUTTOFF ===
        if word_count > 1500:
            score_val *= 0.8
        if word_count > 3000:
            score_val *= 0.85

        # === FINAL CLAMP ===
        if score_val < 0:
            score_val = 0.0
        if score_val > 10:
            score_val = 10.0

        return score_val

    except:
        return 5.0