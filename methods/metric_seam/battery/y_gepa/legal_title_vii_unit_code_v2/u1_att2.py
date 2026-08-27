import re
import math

def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        t = text.lower()
        if len(t) < 80:
            return 0.15

        # Stage signals
        sj_terms = len(re.findall(r'summary judg', t))
        genuine = len(re.findall(r'genuine\s+(?:issue|dispute|triable)\s+(?:of|as to)?\s*(?:material)?\s*fact', t))
        no_genuine = len(re.findall(r'no\s+genuine', t))
        rule56 = len(re.findall(r'rule\s*56|fed\.?\s*r\.?\s*civ\.?\s*p\.?\s*56', t))
        celotex = t.count('celotex')
        sj_signal = min(sj_terms, 6) * 1.4 + min(genuine + no_genuine, 5) * 1.3 + min(rule56, 2) * 0.6 + min(celotex, 2) * 0.4

        mtd_terms = len(re.findall(r'motion to dismiss', t))
        rule12 = len(re.findall(r'rule\s*12\(b\)|12\(b\)\(6\)', t))
        iqbal = t.count('iqbal') + t.count('twombly')
        plaus = len(re.findall(r'plausib', t))
        fail_state = len(re.findall(r'failure to state', t))
        mtd_signal = min(mtd_terms, 5) * 1.5 + min(rule12, 2) * 0.8 + min(iqbal, 3) * 0.5 + min(plaus, 3) * 0.4 + min(fail_state, 2) * 0.5

        appeal_count = t.count('appeal') + t.count('appellate')
        circuit = len(re.findall(r'\b\d+(?:st|nd|rd|th)\s+circuit\b', t))
        affirm_rev = len(re.findall(r'\b(?:affirm(?:ed|ing)?|revers(?:ed|ing)|vacat(?:ed|ing)|remand(?:ed|ing)?)\b', t))
        appeal_signal = min(appeal_count, 8) * 0.35 + min(circuit, 3) * 0.9 + min(affirm_rev, 5) * 0.7

        trial_count = t.count('trial') + t.count('verdict')
        jury_instr = len(re.findall(r'jury instruction|verdict form|deliberation|hung jury', t))
        witness = len(re.findall(r'testif(?:ied|y)|cross-?exam|witness stand|eyewitness', t))
        trial_signal = min(trial_count, 8) * 0.4 + min(jury_instr + witness, 4) * 0.5

        # Procedural identification in general
        proced = len(re.findall(r'before\s+(?:the\s+)?court\s+is|pending before|presently before', t))
        proced += len(re.findall(r'motion (?:for|to)|defendant\'?s motion|plaintiff\'?s motion|cross-?motion', t))
        proced += len(re.findall(r'\bden(?:y|ied|ying)\s+(?:defendant\'?s|plaintiff\'?s|the)?\s*motion', t))
        proced += len(re.findall(r'grant(?:ed|ing|s)?\s+(?:defendant\'?s|plaintiff\'?s|the)?\s*motion', t))
        proced_signal = min(proced, 6) * 0.3

        # Standard of review / light most favorable
        lfmf = len(re.findall(r'light most favorable', t))
        viewed_lfmf = len(re.findall(
            r'(?:viewed|considered|construed|read|taken|resolv(?:e|ed|ing)|draw(?:n|ing)?|inferred?)(?:[^.]{0,60})light most favorable', t))
        standard = len(re.findall(r'standard of review|de novo|abuse of discretion|clearly erroneous', t))
        sor_signal = min(lfmf, 3) * 1.2 + min(viewed_lfmf, 3) * 1.6 + min(standard, 3) * 0.5

        # Court procedural posture declaration
        posture_decl = len(re.findall(
            r'this matter is before|comes now|before the court|presently before|pending before|the court must (?:first )?determine', t))
        posture_signal = min(posture_decl, 3) * 0.4

        combined = sj_signal + mtd_signal + appeal_signal + trial_signal + proced_signal + sor_signal + posture_signal

        # Factual background bonus - Title VII procedural docs often recite facts
        facts = len(re.findall(r'plaintiff alleges|defendant (?:contends|argues)|the facts|background|procedural (?:history|posture)', t))
        combined += min(facts, 4) * 0.2

        # Moderate baseline for all docs (most court docs identify a stage)
        base = 0.2
        total = base + combined * 0.12

        # Moderate signal gets moderate score
        if combined >= 1.5:
            total = 0.4 + combined * 0.1
        if combined >= 4:
            total = 0.55 + (combined - 4) * 0.07
        if combined >= 8:
            total = 0.75 + (combined - 8) * 0.05
        if combined >= 14:
            total = 0.92

        return max(0.05, min(1.0, total))
    except Exception:
        return 0.5