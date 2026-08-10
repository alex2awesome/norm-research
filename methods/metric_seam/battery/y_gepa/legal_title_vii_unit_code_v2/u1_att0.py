def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        t = text.lower()
        length = len(t)
        if length < 200:
            return 1.0

        import re

        stage_keywords = [
            'motion to dismiss', 'motion for summary judgment', 'summary judgment',
            'rule 12(b)(6)', '12(b)(6)', 'rule 56', 'trial', 'bench trial',
            'jury trial', 'appeal', 'motion for judgment', 'motion for a new trial',
            'motion in limine', 'motion to remand', 'motion to compel',
            'preliminary injunction', 'temporary restraining order',
            'motion for reconsideration', 'petition for certiorari', 'mandamus',
            'remittitur', 'motion for remittitur', 'j.n.o.v.', 'judgment as a matter of law',
            'motion for judgment as a matter of law', 'rule 50', 'rule 59',
            'motion to strike', 'motion to vacate', 'motion for class certification'
        ]
        stage_count = 0
        for kw in stage_keywords:
            stage_count += t.count(kw)
            if stage_count >= 40:
                break
        has_stage = stage_count > 0

        lfmf_patterns = [
            r'light most favorable to the non[- ]mov',
            r'light most favorable to the nonmov',
            r'light most favorable to (?:the )?(?:plaintiff|non[- ]?moving|oppos)',
            r'(?:viewed|considered|construed|read) (?:all (?:the )?)?(?:facts|evidence|record)(?:[^.]{0,80}?)light most favorable',
            r'light most favorable',
            r'in a light most favorable',
            r'favorable to the party oppos',
            r'favorable to the non[- ]?mov'
        ]
        lfmf_hits = 0
        for pat in lfmf_patterns:
            try:
                matches = re.findall(pat, t)
                lfmf_hits += len(matches)
            except Exception:
                pass

        sj_indicators = [
            'genuine issue of material fact',
            'genuine dispute of material fact',
            'no genuine issue',
            'no genuine dispute',
            'material fact',
            'celotex',
            'anderson v',
            'summary judgment',
            'moving party',
            'burden of production',
            'shifts to the non[- ]?moving'
        ]
        sj_score = 0
        for ind in sj_indicators:
            sj_score += t.count(ind)
            if sj_score > 60:
                break

        if lfmf_hits >= 3 and sj_score >= 3:
            base = 10.0
        elif lfmf_hits >= 3:
            base = 9.5
        elif lfmf_hits >= 2 and sj_score >= 3:
            base = 9.5
        elif lfmf_hits >= 2:
            base = 9.0
        elif lfmf_hits == 1 and sj_score >= 4:
            base = 9.0
        elif lfmf_hits == 1:
            base = 8.0
        elif sj_score >= 6:
            base = 7.5
        elif sj_score >= 3:
            base = 7.0
        elif sj_score >= 1:
            base = 6.0
        else:
            base = 0.0

        if base == 0.0 and not has_stage:
            return 1.0
        if base == 0.0 and has_stage:
            return 4.0

        stage_bonus = 0.0
        if 'summary judgment' in t or 'motion for summary judgment' in t:
            stage_bonus = max(stage_bonus, 1.5)
        if 'motion to dismiss' in t or '12(b)(6)' in t:
            stage_bonus = max(stage_bonus, 1.2)
        if re.search(r'\btrial\b', t) or 'bench trial' in t or 'jury trial' in t:
            stage_bonus = max(stage_bonus, 0.8)
        if 'appeal' in t or 'appellate' in t:
            stage_bonus = max(stage_bonus, 1.0)
        if 'preliminary injunction' in t:
            stage_bonus = max(stage_bonus, 0.7)
        if 'class certification' in t:
            stage_bonus = max(stage_bonus, 0.5)
        if 'motion to compel' in t:
            stage_bonus = max(stage_bonus, 0.4)

        standard_patterns = [
            r'de novo', r'abuse of discretion', r'clearly erroneous',
            r'substantial evidence', r'arbitrary and capricious',
            r'plain error', r'standard of review'
        ]
        standard_found = False
        for pat in standard_patterns:
            try:
                if re.search(pat, t):
                    standard_found = True
                    break
            except Exception:
                pass

        if standard_found:
            stage_bonus += 0.5

        result = base + stage_bonus

        if result >= 10:
            return 10.0
        if result < 1 and has_stage:
            return 4.0
        return max(0.0, min(10.0, result))
    except Exception:
        return 0.5