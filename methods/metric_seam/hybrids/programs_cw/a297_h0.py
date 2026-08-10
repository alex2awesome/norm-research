"""
a297: Authenticity and honesty of voice
"A truthful, unforced voice - vulnerable when needed - that feels uniquely
the writer's and sustains reader trust."

Design notes (from train-residual study):
  - baseline_train_rho = 0.042 (near-zero). The baseline bet on first-person
    density + "confession" keywords + a generic-phrase penalty. That bet is
    wrong: many of the LOWEST-judged docs are heavy first-person confessional
    text (fake AP-news joke, wedding-toast reveal, dark-humor suicide bit),
    while some of the HIGHEST-judged docs have almost no first-person voice
    at all (a third-person "twenty dollar bill" narrator). Keyword presence
    is topic, not craft/authenticity - exactly what the corpus notes warn.
  - What actually separates the train extremes is not pronoun choice, it's
    whether the piece leans on a shallow, load-bearing GIMMICK (celebrity/
    franchise parody, shock-for-its-own-sake vulgarity, wish-fulfillment
    power-fantasy, or a visibly dashed-off/low-effort framing like "I guess
    I gotta contribute now") vs. a fully realized bit/scene that earns its
    tone. Almost every judge_score<=0.2 train doc has an identifiable tell
    of this kind; almost none of the judge_score>=0.6 docs do.
  - A secondary, weaker signal: genuine private vulnerability/admission
    (either in-story or in an authorial aside) nudges score up, but it is
    NOT sufficient by itself (several low/mid docs contain real admissions
    too) - so it's a small modifier, not the main predicate.
  - Regex can't see either of these (they require reading the whole story
    and judging tone/register), so both live in LLM_FIELDS; the actual
    predicate/weighting stays in code.
"""

import re
import statistics  # noqa: F401  (kept available per contract; not required)

LLM_FIELDS = {
    "vulnerable_admission": (
        "In <=12 words: the narrator's or author's single most genuine, "
        "private vulnerable feeling or admission in this piece, or NONE if "
        "there isn't one."
    ),
    "surface_tell": (
        "In <=10 words: name ONE shallow device this piece leans on - "
        "celebrity/franchise parody, shock-for-its-own-sake vulgarity, "
        "wish-fulfillment power fantasy, or a dashed-off/low-effort tone - "
        "or NONE if the piece feels fully realized on its own terms."
    ),
}


def _is_none(val):
    v = (val or "").strip()
    if not v:
        return True
    vu = v.upper().strip(" .!\"'")
    return vu in ("NONE", "N/A", "NA", "NULL", "-")


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        try:
            t = ops.normalize(text)
        except Exception:
            t = text
        if not t or not t.strip():
            return 0.5

        vuln = extracted.get("vulnerable_admission", "") if extracted else ""
        tell = extracted.get("surface_tell", "") if extracted else ""
        vuln_hit = not _is_none(vuln)
        tell_hit = not _is_none(tell)

        s = 0.55
        if tell_hit:
            s -= 0.30
        if vuln_hit:
            s += 0.10

        # Defensive craft floor: truly degenerate/near-empty fragments
        # shouldn't ride the baseline's coattails.
        try:
            n_sent, mean_wps, frac_long = ops.sent_stats(t)
        except Exception:
            n_sent, mean_wps, frac_long = (10, 15.0, 0.2)
        if n_sent is not None and n_sent <= 2:
            s -= 0.08

        # Weak originality tiebreaker via TF-IDF retrieval: a piece that
        # sits almost on top of many corpus neighbors reads as more
        # derivative/tropey ("uniquely the writer's" cuts the other way);
        # a piece with few close neighbors reads as more distinctive.
        # Small, bounded effect only - this is corroborating evidence, not
        # the predicate.
        try:
            sims = ops.retrieve_similar(t, k=5, exclude_id=None)
            others = [sim for sim, _id in (sims or []) if sim is not None and sim < 0.98]
            if others:
                avg_sim = sum(others) / len(others)
                if avg_sim >= 0.45:
                    s -= 0.05
                elif avg_sim < 0.12:
                    s += 0.05
        except Exception:
            pass

        return float(max(0.0, min(1.0, s)))
    except Exception:
        return 0.5


if __name__ == "__main__":
    # Minimal self-constructed smoke test (no external files/corpus).
    class _FakeOps:
        def normalize(self, x):
            return x

        def extract_dates(self, x):
            return []

        def sent_stats(self, x):
            n = max(1, x.count(".") + x.count("!") + x.count("?"))
            words = x.split()
            mean_wps = len(words) / n
            frac_long = sum(1 for w in words if len(w) >= 7) / max(1, len(words))
            return (n, mean_wps, frac_long)

        def retrieve_similar(self, x, k=5, exclude_id=None):
            return [(1.0, "self"), (0.2, "d1"), (0.15, "d2")]

    ops = _FakeOps()

    gimmick_text = "PARIS (AP) Leaders gathered to thank the detective who defeated ISIS. " * 5
    sincere_text = (
        "I stood in the bathroom and pressed the towel against my mouth so my "
        "roommate wouldn't hear. I had never said the word out loud before. "
        "I said it anyway, quietly, to see if it would break something in me. "
        "It didn't. That was the strange part."
    )

    print("gimmick, tell_hit only:",
          score(gimmick_text, {"vulnerable_admission": "NONE", "surface_tell": "celebrity parody joke"}, ops))
    print("sincere, vuln_hit only:",
          score(sincere_text, {"vulnerable_admission": "fear of saying the word aloud", "surface_tell": "NONE"}, ops))
    print("neither field fires:",
          score(sincere_text, {"vulnerable_admission": "NONE", "surface_tell": "NONE"}, ops))
    print("empty text:", score("", {}, ops))
