"""a324_h0 -- hybrid channel for "Character Dimensionality and Development".

Judge gradient observed on train:
  low  (0.05-0.2): overview/chronicle narration (collective 'we'/humanity, history
                   logs), one-note caricatures, apologetic newbie author-notes;
  mid  (0.3)     : individual characters in one scene, twist but shallow arc;
  high (0.5-0.65): named characters with interiority + backstory whose feelings/
                   relationships evolve; often practiced serial writers.

Design: two cheap, well-posed LLM fields give thick-input grounding
(individual-vs-overview focus; one grounded character whose interior changes);
the predicate stays in code (grounding check against name recurrence /
first-person density) and is blended with class-level code features
(interiority density, individual-vs-collective pronoun ratio, recurring names,
past-perfect backstory, dialogue, length) plus small author-note adjustments.
"""

import re
import math
from collections import Counter

LLM_FIELDS = {
    "arc_character": (
        "Name ONE character whose inner feelings, motives, or relationships "
        "clearly change or deepen between the story's start and end, plus 3-6 "
        "words on how; if no character's interior changes, answer NONE."
    ),
    "story_focus": (
        "Answer 'individual' if the story dramatizes specific characters in "
        "scenes; answer 'overview' if it mostly summarizes history, groups, "
        "or humanity in aggregate."
    ),
}

# ---------------------------------------------------------------- lexicons --

_INTERIOR_RE = re.compile(
    r"\b(?:felt|feel(?:ing)?|thought|wonder(?:ed|ing)?|remember(?:ed|ing)?|"
    r"realiz(?:ed|ing)|realis(?:ed|ing)|regret(?:ted|s)?|hoped?|hoping|"
    r"fear(?:ed|s)?|afraid|scared|wanted|wish(?:ed|ing)?|longed|hated?|"
    r"loved?|believ(?:ed|ing)|understood|decid(?:ed|ing)|noticed|imagin(?:ed|ing)|"
    r"dream(?:ed|t|ing)?|worr(?:ied|ying)|ashamed|guilt(?:y)?|grief|sorrow|"
    r"lonel(?:y|iness)|anger|angry|jealous|proud|shame|despair|dread(?:ed)?|"
    r"relief|relieved|yearn(?:ed|ing)?|craved?|panick(?:ed|ing)|nervous|"
    r"anxious|memor(?:y|ies))\b",
    re.IGNORECASE,
)

_CHANGE_RE = re.compile(
    r"(?:\bno longer\b|\bfor the first time\b|\bnever again\b|\bused to\b|"
    r"\bbecame\b|\bbecoming\b|\bgrew\b|\bfinally\b|\bat last\b|\blearn(?:ed|t)\b|"
    r"\bchanged\b|\bnever be the same\b|\bnot anymore\b)",
    re.IGNORECASE,
)

_BACKSTORY_RE = re.compile(
    r"(?:\bhad\s+(?:been|never|always|once|long)\b|\bhad\s+\w+(?:ed|en)\b|"
    r"\byears? ago\b|\bas a child\b|\bwhen (?:i|he|she) was\b|\bback then\b|"
    r"\bchildhood\b|\bonce been\b|\bin those days\b)",
    re.IGNORECASE,
)

_COLLECTIVE_RE = re.compile(
    r"\b(?:we|us|our|ours|ourselves|humanity|mankind|humankind|everyone|"
    r"citizens|nations|species|civilizations?|empires?|the world|the public|"
    r"the masses)\b",
    re.IGNORECASE,
)

_INDIVIDUAL_RE = re.compile(
    r"\b(?:he|she|him|her|his|hers|himself|herself|i|me|my|mine|myself)\b",
    re.IGNORECASE,
)

_FIRST_PERSON_RE = re.compile(r"\b(?:i|me|my|myself)\b", re.IGNORECASE)

_RELATION_RE = re.compile(
    r"\b(?:mother|father|mom|dad|parent|sister|brother|sibling|daughter|son|"
    r"child|wife|husband|spouse|partner|lover|boyfriend|girlfriend|fiance|"
    r"friend|best friend|neighbor|neighbour|colleague|coworker|co-worker|"
    r"boss|mentor|student|teacher|rival|enemy|stranger|ally|companion|"
    r"family|relationship|marriage|friendship)\b",
    re.IGNORECASE,
)

_APOLOGY_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"english is(?:n'?t| not) my (?:first|native)",
        r"first (?:prompt|post|story|submission)",
        r"(?:very )?new to writing",
        r"haven'?t written",
        r"been a (?:few|while|long|couple|year)[^.!?]{0,60}(?:written|writing)",
        r"apolog(?:y|ies|ise|ize|izing)",
        r"sorry (?:for|about|if)",
        r"gotta contribute",
        r"criticism (?:is )?(?:welcome|appreciated)",
        r"\bedit\s*:",
        r"thank you[^.!?]{0,40}made it this far",
        r"not (?:going to|gonna) be great",
        r"forgive me",
        r"hope (?:you|someone) enjoy",
    )
]

_SIGNATURE_RE = re.compile(r"(?:^|[\s(\[/])r/([A-Za-z0-9_]{3,21})")

_NAME_STOP = frozenset(
    "The And But Then They She He His Her Him It Its You Your When What Why "
    "Who Where How Not Now One Two Three That This There These Those Was Were "
    "Are Is Had Has Have Been From With After Before While Just Like Well Yes "
    "No Oh Okay Maybe Monday Tuesday Wednesday Thursday Friday Saturday Sunday "
    "January February March April May June July August September October "
    "November December Mr Mrs Ms Dr Sir All For Our Out Some Everyone People "
    "Suddenly Finally Eventually Meanwhile Later Today Tomorrow Yesterday "
    "Because If So As At In On Of To Do Did Will Would Could Should Can Cant "
    "Im Ive Id Youre Hes Shes Dont Wait Look Listen Hey Hello Please Thanks "
    "Thank God Lord Everything Nothing Something Anything Nobody Somebody "
    "Edit Update Part".split()
)


def _words(text):
    return re.findall(r"[A-Za-z']+", text)


def _clip01(x):
    return max(0.0, min(1.0, float(x)))


def _name_counts(text):
    """Capitalized tokens seen at least once mid-sentence -> occurrence counts."""
    counts = Counter()
    mid_sentence = set()
    for m in re.finditer(r"[A-Z][a-z]{2,}\b", text):
        tok = m.group(0)
        if tok in _NAME_STOP:
            continue
        counts[tok] += 1
        j = m.start() - 1
        while j >= 0 and text[j] in " \t":
            j -= 1
        if j >= 0 and (text[j].islower() or text[j] in ",;:"):
            mid_sentence.add(tok)
    return {t: c for t, c in counts.items() if t in mid_sentence}


def _is_none_answer(ans):
    a = (ans or "").strip().strip(".!\"'").lower()
    if not a:
        return True
    if a.startswith("none") or a in ("n/a", "na", "no", "nothing", "nobody"):
        return True
    if a.startswith("no ") and ("character" in a or re.search(r"\bone\b", a)):
        return True
    return False


def score(text, extracted, ops):
    try:
        if not text or not isinstance(text, str):
            return 0.5
        try:
            t = ops.normalize(text)
            if not t or not isinstance(t, str):
                t = text
        except Exception:
            t = text

        words = _words(t)
        w = max(1, len(words))
        per100 = 100.0 / w

        # ------------------------------------------------- code features --
        # Individual-vs-collective POV: overview/chronicle pieces (history
        # logs, 'we/humanity' infodumps) are the judge's lowest band.
        indiv = len(_INDIVIDUAL_RE.findall(t))
        coll = len(_COLLECTIVE_RE.findall(t))
        ratio = indiv / float(indiv + coll + 1)
        s_pov = _clip01((ratio - 0.30) / 0.55)

        # Social positioning: relationship nouns (criterion mentions
        # "social positioning" explicitly).
        rel = len(_RELATION_RE.findall(t)) * per100
        s_rel = _clip01(rel / 1.0)

        # Interiority density (weak but principled; keep small).
        inter = len(_INTERIOR_RE.findall(t)) * per100
        s_inter = _clip01(inter / 2.0)

        quotes = len(re.findall(r'[\"“”]', t)) * per100
        s_dial = _clip01(quotes / 2.5)

        # Room to develop: very short pieces rarely earn "reveal facets
        # over time"; saturates by ~600 words.
        s_len = _clip01(w / 600.0)

        code = (
            0.35 * s_pov
            + 0.20 * s_rel
            + 0.25 * s_len
            + 0.10 * s_inter
            + 0.10 * s_dial
        )

        # -------------------------------------------------- LLM features --
        extracted = extracted if isinstance(extracted, dict) else {}

        focus = str(extracted.get("story_focus", "") or "").lower()
        if "individual" in focus:
            s_focus = 1.0
        elif "overview" in focus or "collective" in focus or "summar" in focus:
            s_focus = 0.0
        else:
            s_focus = 0.5  # unanswered / off-schema

        arc = str(extracted.get("arc_character", "") or "")
        if _is_none_answer(arc):
            s_arc = 0.0
        else:
            base = 0.5
            cap_toks = [
                tok for tok in re.findall(r"\b[A-Z][A-Za-z'-]{2,}\b", arc)
                if tok not in _NAME_STOP
            ]
            if cap_toks:
                best = max(
                    len(re.findall(r"\b" + re.escape(tok) + r"\b", t))
                    for tok in cap_toks
                )
                if best >= 3:
                    ground = 0.5
                elif best == 2:
                    ground = 0.35
                elif best == 1:
                    ground = 0.2
                else:
                    ground = 0.0
                    base = 0.3          # likely hallucinated name
            else:
                # unnamed protagonist / "the narrator": ground on 1st person
                fp = len(_FIRST_PERSON_RE.findall(t)) * per100
                low = arc.lower()
                if "narrator" in low or fp >= 1.5:
                    ground = 0.4
                elif fp >= 0.5:
                    ground = 0.25
                else:
                    ground = 0.1
            s_arc = _clip01(base + ground)

        llm = 0.45 * s_focus + 0.55 * s_arc

        # ---------------------------------------------------- adjustments --
        adjust = 0.0
        head_zone = t[:400].lower()
        tail_zone = t[-700:].lower()
        hits = sum(
            1 for rgx in _APOLOGY_RES
            if rgx.search(head_zone) or rgx.search(tail_zone)
        )
        adjust -= 0.06 * min(2, hits)

        for m in _SIGNATURE_RE.finditer(t[-500:]):
            if m.group(1).lower() != "writingprompts":
                adjust += 0.08
                break

        spam = len(re.findall(r"\n[ \t]*\n[ \t]*\n[ \t]*\n", t))
        if spam > 5:
            adjust -= 0.10
        elif spam > 2:
            adjust -= 0.05

        return _clip01(0.48 * code + 0.52 * llm + adjust)
    except Exception:
        return 0.5
