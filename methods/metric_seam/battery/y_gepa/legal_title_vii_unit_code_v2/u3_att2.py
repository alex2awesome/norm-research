import re
import math

_BUDGET = 25000

def _has_neg_near(text, match, window=90):
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    snippet = text[start:end].lower()
    return any(neg in snippet for neg in [
        'no ', 'not ', 'never', 'failed to', 'did not', "didn't",
        'lack of', 'without', 'failure to', 'refused', 'untimely',
        'inadequate', 'failed', 'no reasonable', 'no adequate',
        'cannot ', "can't ", 'could not', "couldn't", 'nothing',
        'unreasonab', 'deficien'
    ])

def _sentence_around(text, pos, radius=120):
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    return text[start:end].lower()

def score(text: str) -> float:
    try:
        if not isinstance(text, str) or len(text) < 30:
            return 0.5

        t = text[:_BUDGET]
        tl = t.lower()

        # ===================================================================
        # CRITICAL FILTERS: Catch false-positives where "investigation" refers
        # to the plaintiff's job duties, criminal matters, or investigations
        # OF the plaintiff rather than BY the employer.
        # ===================================================================
        is_law_enforcement = bool(re.search(
            r'\b(police (dept|department|officer)|sheriff|detective|'
            r'fbi agent|law enforcement|border patrol|ice agent|'
            r'public safety|state trooper|constable)\b', tl))

        plaintiff_investigates = bool(re.search(
            r'\b(plaintiff|claimant|employee|officer|he|she|petitioner)\s+'
            r'(began\s+)?investigat', tl))

        investigated_of = bool(re.search(
            r'\b(investigated|investigation of)\s+(plaintiff|claimant|'
            r'the employee|him|her)\b', tl))

        criminal_ctx = bool(re.search(
            r'\b(criminal|prosecut|indict|grand jury|arrested|convict|'
            r'defendant (was|had) (charged|arrested)|fraud|bribery|'
            r'embezzlement|extortion|racketeer|wiretap)\b', tl))

        # Drops score to near-zero for clear mismatches
        law_enforce_penalty = 0.0
        if is_law_enforcement:
            law_enforce_penalty = min(0.45, 0.15 + tl.count('police') * 0.04 + tl.count('detective') * 0.04)
        if plaintiff_investigates:
            law_enforce_penalty = max(law_enforce_penalty, 0.4)
        if investigated_of:
            law_enforce_penalty = max(law_enforce_penalty, 0.45)
        if criminal_ctx:
            law_enforce_penalty = max(law_enforce_penalty, 0.3)

        # ===================================================================
        # STRONG EVIDENCE: Direct employer investigation or corrective action
        # ===================================================================
        strong_inv_patterns = [
            r'employer\s+investigated',
            r'employer\s+(conducted|initiated|launched)\s+(an?\s+)?investigation',
            r'company\s+(conducted|launched|initiated|began|started)\s+(an?\s+)?investigation',
            r'(hr|human\s+resources)\s+(conducted|launched|initiated|began)\s+(an?\s+)?investigation',
            r'(hr|human\s+resources)\s+investigated',
            r'(conducted|launched|initiated|began|started)\s+(a|an|its|the)\s+\w*\s*investigation',
            r'(prompt|thorough|immediate|timely|reasonable|adequate|internal|independent)\s+investigation',
            r'investigation\s+(revealed|found|determined|concluded|confirmed|showed|substantiat)',
            r'investigator\s+(found|concluded|determined|interviewed)',
            r'(outside|third-party|external)\s+investigator',
            r'hired\s+(an?\s+)?(outside|independent|third-party)\s+investigator',
            r'(internal|outside)\s+investigation\s+(was|into|of)',
            r'investigation\s+(was|into|of|by)\s',
            r'department\s+investigated',
            r'management\s+(conducted|investigated|reviewed)',
            r'(interviewed|questioned)\s+(witnesses|the (complainant|accused|employee|parties))',
        ]

        strong_action_patterns = [
            r'(took|implemented|initiated)\s+(prompt\s+)?corrective\s+action',
            r'(took|implemented|initiated)\s+(prompt\s+)?remedial\s+(action|measures|steps)',
            r'responded\s+promptly',
            r'(promptly|immediately)\s+(investigat|remedied|addressed|corrected|responded|suspended|terminated|disciplined|warned)',
            r'(disciplined|suspended|terminated|reprimanded|warned|transferred)\s+(the\s+)?(harasser|accused|alleged|perpetrator)',
            r'(verbal|written)\s+warning',
            r'(sensitivity|anti-harassment|diversity|workplace)\s+training',
            r'mandatory\s+training',
            r'(corrective|remedial)\s+(action|measures|steps)',
        ]

        strong_inv = 0
        for pat in strong_inv_patterns:
            for m in re.finditer(pat, tl):
                ctx = _sentence_around(tl, m.start())
                if _has_neg_near(ctx, re.search(pat, ctx) or m):
                    continue
                # Penalize if law enforcement investigation found in same context
                if re.search(r'\b(police|detective|officer|criminal|prosecut|arrest)\b', ctx):
                    continue
                strong_inv += 1
                if strong_inv >= 4:
                    break
            if strong_inv >= 4:
                break

        strong_action = 0
        for pat in strong_action_patterns:
            for m in re.finditer(pat, tl):
                ctx = _sentence_around(tl, m.start())
                if _has_neg_near(ctx, re.search(pat, ctx) or m):
                    continue
                strong_action += 1
                if strong_action >= 4:
                    break
            if strong_action >= 4:
                break

        # ===================================================================
        # MODERATE SIGNALS: investigation process steps or procedural signs
        # ===================================================================
        moderate_signals = 0
        moderate_patterns = [
            r'investigation\s+into\s+(the|her|his|these|plaintiff)',
            r'interviewed\s+\w+',
            r'witness\s+statement',
            r'took\s+statements',
            r'gathered\s+(evidence|information|statements)',
            r'reviewed\s+(the\s+)?(evidence|complaint|surveillance|video|footage)',
            r'investigat\w+\s+(the\s+)?(allegation|complaint|claim|incident|conduct)',
            r'(complaint|grievance)\s+process',
            r'(internal|company)\s+review',
            r'investigat\w+\s+and\s+(found|determined|concluded)',
            r'personnel\s+department\s+(review|investigat)',
            r'eeoc\s+(charge|investigation|determination)',
            r'determined\s+(that\s+)?(no\s+)?(harassment|discrimination|violation)',
            r'concluded\s+(that\s+)?(no\s+)?(harassment|discrimination)',
            r'(policy|company|handbook)\s+(violation|violated)',
            r'substantiat\w+\s+(the\s+)?(allegation|claim|complaint)',
            r'could\s+not\s+substantiate',
            r'found\s+(to\s+have\s+)?(violat|engag)',
        ]
        for pat in moderate_patterns:
            for m in re.finditer(pat, tl):
                ctx = _sentence_around(tl, m.start())
                if _has_neg_near(ctx, re.search(pat, ctx) or m):
                    continue
                if re.search(r'\b(police|detective|officer|criminal|prosecut|arrest)\b', ctx):
                    continue
                moderate_signals += 1
                if moderate_signals >= 6:
                    break
            if moderate_signals >= 6:
                break

        # ===================================================================
        # WEAK SIGNALS: Generic "investigation" mentions
        # ===================================================================
        weak_inv = 0
        for m in re.finditer(r'investigat\w+', tl):
            ctx = _sentence_around(tl, m.start())
            if _has_neg_near(ctx, re.search(r'investigat\w+', ctx) or m):
                continue
            weak_inv += 1
            if weak_inv >= 8:
                break

        weak_corrective = 0
        for m in re.finditer(r'(corrective|remedial|remedi\w+)', tl):
            ctx = _sentence_around(tl, m.start())
            if _has_neg_near(ctx, re.search(r'(corrective|remedial|remedi\w+)', ctx) or m):
                continue
            weak_corrective += 1
            if weak_corrective >= 6:
                break

        weak_discipline = 0
        for m in re.finditer(r'\b(disciplin\w+|suspend\w+|terminat\w+|reprimand\w+|warning)\b', tl):
            ctx = _sentence_around(tl, m.start())
            if _has_neg_near(ctx, re.search(r'\b(disciplin\w+|suspend\w+|terminat\w+|reprimand\w+|warning)\b', ctx) or m):
                continue
            # Skip if plaintiff was disciplined/terminated
            if re.search(r'(plaintiff|claimant|employee|him|her|petitioner)\s+(was\s+)?(terminat|suspend|disciplin|reprimand)', ctx):
                continue
            weak_discipline += 1
            if weak_discipline >= 5:
                break

        # ===================================================================
        # COMBINE AND SCORE
        # ===================================================================
        raw_score = 0.0
        raw_score += min(strong_inv, 3) * 1.6
        raw_score += min(strong_action, 3) * 1.7
        raw_score += min(moderate_signals, 6) * 0.65
        raw_score += min(weak_inv, 5) * 0.18
        raw_score += min(weak_corrective, 4) * 0.25
        raw_score += min(weak_discipline, 4) * 0.30

        # Cap and apply penalties
        capped = min(raw_score, 9.0)
        final_score = capped * (1.0 - law_enforce_penalty)

        if final_score < 0.5:
            return 0.5
        if final_score > 9.5:
            return 9.5
        return round(final_score, 2)

    except Exception:
        return 0.5