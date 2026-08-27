def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        t = text.lower()
        if len(t) < 200:
            return 0.5

        import re

        lfmf_count = len(re.findall(r'light most favorable', t))
        viewed_favorable = len(re.findall(
            r'(?:viewed|considered|construed|read|taken|resolve|draw|inferred?)(?:[^.]{0,60})light most favorable', t))
        lfmf_signal = min(lfmf_count, 3) * 1.2 + min(viewed_favorable, 3) * 1.5

        genuine_issue = len(re.findall(r'(?:genuine|triable)\s+(?:issue|dispute)\s+(?:of|as to)\s+material\s+fact', t))
        no_genuine = len(re.findall(r'no\s+(?:genuine|triable)\s+(?:issue|dispute)\s+(?:of|as to)?\s*material\s+fact', t))
        no_genuine_count = len(re.findall(r'no\s+genuine', t))
        celotex = t.count('celotex')
        summary_judgment = t.count('summary judgment') + t.count('summary judgement')
        rule56 = len(re.findall(r'rule\s*56|fed\.\s*r\.\s*civ\.\s*p\.\s*56', t))
        sj_signal = min(summary_judgment, 5) * 1.4 + min(genuine_issue + no_genuine, 4) * 1.3 + min(celotex, 2) * 0.8 + min(rule56, 2) * 0.6

        motion_dismiss = len(re.findall(r'(?:motion to dismiss|rule\s*12\(b\)|12\(b\)\(6\)|failure to state a claim|motion to dismiss for failure to state)', t))
        iqb_twombly = len(re.findall(r'(?:iqbal|twombly|plausib(?:le|ility)|facial(?:ly)? (?:challeng|attack)|well-?pleaded)', t))
        twombly_signal = min(motion_dismiss, 5) * 1.5 + min(iqb_twombly, 4) * 0.7

        appeal_terms = t.count('appeal') + t.count('appellate') + t.count('affirm') + t.count('reverse')
        circuit_count = len(re.findall(r'\b\d+(?:st|nd|rd|th)\s+circuit\b', t))
        affirm_reverse = len(re.findall(r'\b(?:affirm(?:ed|ing)?|revers(?:ed|ing)|vacat(?:ed|ing)|remand(?:ed|ing)?)\b', t))
        appeal_signal = min(appeal_terms, 8) * 0.4 + min(circuit_count, 3) * 1.0 + min(affirm_reverse, 5) * 0.8

        trial_signal = min(t.count('trial') + t.count('verdict'), 6) * 0.5
        jury_instr = len(re.findall(r'(?:jury instruction|verdict form|deliberation|hung jury)', t))
        witness = len(re.findall(r'(?:testif(?:ied|y)|cross-?exam|direct exam|witness stand|eyewitness)', t))
        trial_signal += min(jury_instr + witness, 4) * 0.6

        other_signals = (
            len(re.findall(r'preliminary injunction|temporary restraining order|(?:permanent )?injunction', t)) * 0.7 +
            len(re.findall(r'motion for judgment as a matter of law|j\.?n\.?o\.?v\.?|rule\s*50|judgment as a matter', t)) * 0.7 +
            len(re.findall(r'class certification|class action|f\.?r\.?c\.?p\.?\s*23|typicality|commonality|numerosity|adequacy of', t)) * 0.5 +
            len(re.findall(r'motion to compel|discovery|interrogator|deposition|protective order', t)) * 0.4
        )

        standards = len(re.findall(
            r'(?:standard of review|de novo review|de novo|abuse of discretion|clearly erroneous|substantial evidence|arbitrary and capricious|rational basis|strict scrutiny|intermediate scrutiny)', t))
        federal_court = len(re.findall(r'\b(?:title vii|42 u\.?s\.?c\.?|civil rights act|equal employment|eeoc|district court|magistrate)\b', t))

        total_stage = sj_signal + twombly_signal + appeal_signal + trial_signal + other_signals
        explicit_signal = lfmf_signal + min(standards, 5) * 0.6

        total = total_stage + explicit_signal

        if total_stage < 1.5 and explicit_signal < 1.0:
            return 0.0

        if lfmf_count >= 2 and (sj_signal >= 3 or twombly_signal >= 2):
            total += 2.5

        result = min(total / 18.0, 1.0)

        if total_stage >= 6 and explicit_signal < 0.5:
            result *= 0.7
        if total_stage >= 8 and lfmf_count == 0:
            result *= 0.55
        if result < 0.15 and explicit_signal < 0.5:
            return 0.05

        return round(result * 10.0) / 10.0

    except Exception:
        return 0.5