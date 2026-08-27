"""a44 hybrid: Protected activity present (opposition or participation).

Criterion: plaintiff engaged in identifiable opposition (complained, reported,
objected) or participation (filed a charge, testified, cooperated with an
investigation) activity regarding discrimination/harassment.

Baseline (v2_holistic) counts institutional-channel words (EEOC/HR/ombuds),
filing verbs, opposition verbs, and participation verbs anywhere in the doc
(train rho 0.54). The train pack shows two concrete failure shapes:

  1. FALSE POSITIVES from term presence without the construct. "EEOC"/"filed
     ... claim" fire even when: (a) EEOC is named as a co-plaintiff/enforcement
     party in a systemic case with no personal act by this plaintiff described
     (d00734, judge 0, baseline 0.28); (b) the filing is a workers'-comp claim,
     not Title VII opposition (d01107, judge 0, baseline 0.29); (c) "reported"
     fires on reporting a MEDICAL FACT (pregnancy, absence reason) rather than
     opposing discrimination (d00914/d00324, judge 0, baseline 0.1/0). This is
     exactly the known failure pattern: legal-term presence is a weak proxy for
     the construct being factually present.
  2. FALSE NEGATIVES from phrase brittleness. Genuine opposition/participation
     paraphrased outside the fixed verb list scores near zero even at judge=1:
     "made multiple complaints to supervisors" (d00816, judge 0.9, baseline 0),
     "filed an internal complaint" (d00408, judge 1.0, baseline 0), "contacted
     an EEO counselor" (d00871, judge 1.0, baseline 0.22), "instituted this
     action" / "filed a sworn charge" (d00856, judge 1.0, baseline 0.3),
     "repeatedly requested help from ... superiors" (d00271, judge 0.75,
     baseline 0).

Per the pack's own guidance, the construct itself (did THIS plaintiff actually
oppose/participate, and about what) is thick-input grounding an LLM extractor
reaches better than regex; code keeps the quantity/temporal signal (how many
dated instances -> more "identifiable") and the institutional-channel word
list as a *secondary* confirmation, never the primary gate. Concretely: the
LLM's on/off-topic read GATES the score (multiplicative), so an off-topic or
absent act cannot be rescued by strong institutional-keyword counts (fixes
failure shape 1), while a genuine on-topic act reported informally still
clears a reasonably high floor even if the code channel is silent (fixes
failure shape 2). The old regex channel survives as a bounded +/-20% modifier,
so it still adds value when it agrees, but is never allowed to dominate.
"""
import re, math

LLM_FIELDS = {
    "protected_act": (
        "In <=15 words, name the SPECIFIC act by which THIS plaintiff (not a "
        "government agency, not a co-worker) opposed, complained about, "
        "reported, or participated in a proceeding regarding discrimination "
        "or harassment (e.g. 'filed EEOC charge', 'complained to HR about "
        "harassment', 'testified in coworker's investigation'). Answer NONE "
        "if no such act by this plaintiff is described."
    ),
    "protected_topic": (
        "In <=10 words, what was that act ABOUT (e.g. 'race discrimination', "
        "'sexual harassment', 'workers comp injury', 'attendance policy', "
        "'pregnancy disclosure'). Answer NONE if no act."
    ),
}

# ---- code layer: baseline institutional/filing/opposition/participation
# channels, lightly widened for a couple of the clearest missed paraphrases,
# kept as a bounded confirmatory signal (see _code_channel / combination). ----
_CHANNEL_PATS = [
    r'\beeoc\b', r'eeo counselor', r'eeo office', r'eeo hotline',
    r'office of federal operations', r'\bofo\b', r'human resources',
    r'\bhr department\b', r'ombuds', r'equal employment opportunity',
    r'labor relations',
]
_FILING_PATS = [
    r'filed (?:a |an )?(?:\w+\s+){0,2}charge', r'filed (?:a |an )?(?:\w+\s+){0,2}complaint',
    r'filed (?:a |an )?grievance', r'filed (?:a |an )?claim', r'filed (?:a |an )?appeal',
    r'instituted (?:this |the )?(?:action|lawsuit|suit)',
]
_OPPOSITION_PATS = [
    r'\bcomplained (?:to|about)\b', r'made.{0,15}complaints?',
    r'\breported\b', r'\bopposed\b', r'\bobjected to\b', r'\bprotested\b',
    r'\bwhistleblow', r'requested help',
]
_PARTICIPATION_PATS = [
    r'participated in.{0,30}investigation', r'\btestified\b',
    r'gave a statement', r'cooperated with.{0,30}investigation',
    r'contacted (?:an |the )?eeo counselor',
]


def _sat(x, k):
    return 1.0 - math.exp(-x / max(1e-6, k))


def _code_channel(t):
    channel = sum(len(re.findall(p, t)) for p in _CHANNEL_PATS)
    filing = sum(len(re.findall(p, t)) for p in _FILING_PATS)
    opposition = sum(len(re.findall(p, t)) for p in _OPPOSITION_PATS)
    participation = sum(len(re.findall(p, t)) for p in _PARTICIPATION_PATS)
    s = (0.3 * _sat(channel, 1.5)
         + 0.3 * _sat(filing, 1.0)
         + 0.25 * _sat(opposition, 2.0)
         + 0.15 * _sat(participation, 1.0))
    return max(0.0, min(1.0, s))


# ---- LLM-field predicate: on/off-topic gate + formality read ----
_ON_TOPIC_RE = re.compile(
    r'discriminat|harass|retaliat|\bhostile\b|\bracial\b|\brace\b|\bsex\b|sexual|'
    r'gender|disab|\bage\b|national origin|religio|\beeo\b|title vii|civil rights|'
    r'equal employment',
    re.I,
)
_OFF_TOPIC_RE = re.compile(
    r"workers?[' ]?comp|workman|contract dispute|wage claim|overtime|zoning|"
    r'land use|\bfmla\b|attendance polic|independent contractor|pension|'
    r'retirement benefit|medical (?:history|condition)|pregnan\w* to\b',
    re.I,
)
_FORMAL_RE = re.compile(
    r'\beeoc\b|\bfiled\b|\bcharge\b|hotline|counselor|investigation|testif|'
    r'hearing|labor relations|human resources|\bhr\b|formal complaint|'
    r'instituted|ofccp',
    re.I,
)
_INFORMAL_RE = re.compile(
    r'report|complain|told|inform|request|oppose|object|protest|grievance',
    re.I,
)
_NONE_ANSWERS = {"", "none", "n/a", "na", "no", "unknown", "not stated", "not mentioned"}


def _is_none_answer(s):
    return not s or s.strip().lower() in _NONE_ANSWERS


def score(text: str, extracted: dict, ops) -> float:
    try:
        norm = ops.normalize(text) if text else ""
        t = norm.lower()
        code_s = _code_channel(t)

        ext = extracted or {}
        act = (ext.get("protected_act") or "").strip()
        topic = (ext.get("protected_topic") or "").strip()
        act_l, topic_l = act.lower(), topic.lower()
        has_act = not _is_none_answer(act_l)

        if not has_act:
            # No personally-attributed act -> the construct is likely absent
            # even if institutional keywords fired elsewhere in the doc
            # (e.g. EEOC named as a third-party enforcement plaintiff).
            base = 0.12
        else:
            on_topic = bool(_ON_TOPIC_RE.search(topic_l)) or bool(_ON_TOPIC_RE.search(act_l))
            off_topic = bool(_OFF_TOPIC_RE.search(topic_l)) and not on_topic

            if off_topic:
                base = 0.08
            elif on_topic:
                if _FORMAL_RE.search(act_l):
                    base = 0.85
                elif _INFORMAL_RE.search(act_l):
                    base = 0.72
                else:
                    base = 0.45
                # temporal/quantity structure: multiple dated instances make
                # the activity more concretely "identifiable" (per criterion
                # name), a code-side signal LLM short answers don't carry.
                try:
                    n_dates = len(ops.extract_dates(text))
                except Exception:
                    n_dates = 0
                if n_dates >= 2:
                    base = min(1.0, base + 0.08)
            else:
                # act present but topic unresolved/ambiguous
                base = 0.4

        # Code channel acts as a bounded +/-20% confirmatory modifier only
        # -- it can nudge the LLM-gated base up or down but never override
        # the gate (fixes the false-positive shape without discarding the
        # baseline's genuine partial signal on agreement).
        modifier = 0.8 + 0.2 * code_s
        final = base * modifier
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
