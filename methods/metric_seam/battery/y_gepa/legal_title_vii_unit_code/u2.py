import re
from collections import Counter

def score(text: str) -> float:
    try:
        if not text or len(text.strip()) < 200:
            return 0.5

        t = text.lower()
        length = len(t)
        words = re.findall(r"[a-z']+", t)
        wc = max(len(words), 1)

        harass_kw = [
            "harass", "harassment", "harassed", "harassing",
            "slur", "slurs", "epithet", "epithets", "derogatory",
            "racial", "racist", "racism", "sexist", "sexism",
            "sexually", "sexual", "offensive", "abuse", "abusive",
            "intimidat", "threat", "hostile", "degrad", "humiliat",
            "assault", "assaulted", "inappropriate", "vulgar",
            "profanity", "obscene", "taunt", "mock", "belittle",
            "insult", "denigrate",
        ]
        harass_hits = sum(t.count(k) for k in harass_kw)
        if harass_hits == 0:
            return 0.05

        slur_terms = ["n-word", "n word", "the n-word", "nigger", "spic", "wetback",
                      "kike", "chink", "gook", "raghead", "coon", "faggot",
                      "dyke", "tranny", "cunt", "bitch", "whore", "slut"]
        slur_count = sum(t.count(s) for s in slur_terms)

        physical_terms = ["grabbed", "grabbing", "touched", "touching", "touched her",
                          "groped", "groping", "rubbed", "rubbing", "stroked",
                          "pinched", "hit", "struck", "shoved", "pushed",
                          "spat", "spit", "punched", "slapped", "kissed",
                          "hugged", "blocked", "cornered", "followed"]
        physical_count = sum(t.count(p) for p in physical_terms)

        date_num = len(re.findall(r"\b(19|20)\d{2}\b", t))

        quotation_marks = t.count('"') + t.count('“') + t.count('”') + t.count('“') + t.count('„')

        quote_patterns = [
            r"testif", r"depos", r"declar", r"affidavit", r"statement",
            r"evidence", r"admitted", r"acknowledged", r"recalled",
            r"reported", r"complain", r"witness",
        ]
        evidence_hits = sum(t.count(p) for p in quote_patterns)

        freq_patterns = [
            "repeatedly", "frequent", "frequently", "daily", "weekly",
            "every day", "numerous times", "multiple times", "continually",
            "regularly", "constantly", "ongoing", "persistent", "over a period",
            "several times", "many times", "on multiple occasions",
        ]
        freq_hits = sum(t.count(f) for f in freq_patterns)

        behavior_kw = [
            "comment", "comments", "joke", "jokes", "remark", "remarks",
            "statement", "statements", "email", "e-mail", "emails", "text",
            "text message", "note", "notes", "letter", "memo", "picture",
            "image", "photo", "cartoon", "poster", "noose", "graffiti",
        ]
        concrete_behavior = sum(t.count(b) for b in behavior_kw)

        title_vii = ("title vii" in t) or ("title 7" in t)

        def capped(x, cap):
            return min(x / cap, 1.0)

        s_slurs = capped(slur_count * 1.5, 3)
        s_physical = capped(physical_count, 5)
        s_dates = capped(date_num, 10)
        s_quotes = capped(quotation_marks / 6.0, 8)
        s_evidence = capped(evidence_hits, 12)
        s_freq = capped(freq_hits, 4)
        s_behavior = capped(concrete_behavior, 8)
        s_harass = capped(harass_hits, 15)

        concrete_score = (
            0.18 * s_slurs + 0.18 * s_physical + 0.12 * s_dates
            + 0.12 * s_quotes + 0.12 * s_evidence + 0.10 * s_freq
            + 0.10 * s_behavior + 0.08 * s_harass
        )

        vagueness_terms = [
            "allegedly", "rumor", "rumors", "reputation", "generalized",
            "conclusory", "conclus", "speculative", "vague", "general allegation",
            "unsupported", "unsubstantiated", "bare assertion", "boilerplate",
            "general hostility", "perceived", "she believed", "he believed",
            "felt", "felt that", "opinion", "subjective",
        ]
        vagueness_hits = sum(t.count(v) for v in vagueness_terms)

        dismissive_terms = [
            "no evidence", "fails to show", "fails to demonstrate",
            "insufficient evidence", "no showing", "unsupported by",
            "fails to establish", "no basis", "mere speculation",
            "no indication", "without merit", " conclusory ",
        ]
        dismissive_hits = sum(t.count(d) for d in dismissive_terms)

        vagueness_score = (
            0.6 * capped(vagueness_hits, 8) + 0.4 * capped(dismissive_hits, 5)
        )

        if title_vii:
            concrete_score = min(concrete_score * 1.05, 1.0)

        base = concrete_score * (1.0 - 0.55 * vagueness_score)
        base = max(0.0, min(1.0, base))

        if harass_hits >= 3 and concrete_score > 0.45:
            base = max(base, 0.55)
        if slur_count > 0 or physical_count >= 2:
            base = max(base, 0.65)
        if concrete_score < 0.15 and vagueness_score > 0.4:
            base = min(base, 0.25)

        return round(base, 4)

    except Exception:
        return 0.5