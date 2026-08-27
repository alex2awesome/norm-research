"""a252 hybrid channel: Information management — withholding and payoff.

Design (from train-residual study):
  High-judge stories withhold a key fact/identity and pay it off near the END,
  with the reveal EARNED by earlier seeding. Low-judge stories either explain
  everything upfront, chain random "it turns out" gotchas, land a cheap
  punchline twist, or carry amateur chrome (author notes, "Edit:", emoticons,
  sloppy typography). Keyword proxies fail (baseline rho ~0.09); the construct
  is structural, so two forced-choice LLM fields classify reveal POSITION and
  reveal CRAFT, and code keeps the predicate: mapping those labels to a score
  and applying robust surface penalties the judge demonstrably punishes.
"""

import re

LLM_FIELDS = {
    "reveal_position": (
        "Where does the story's most important reveal or payoff land? "
        "Answer one word: EARLY, MIDDLE, END, or NONE."
    ),
    "reveal_craft": (
        "Is the story's key reveal foreshadowed and earned (EARNED), a random "
        "cheap twist (GOTCHA), a direct explanatory info-dump (DUMP), or absent (NONE)?"
    ),
}

_END_TOKENS = ("FINAL", "CLOSING", "CONCLUSION")
_EARLY_TOKENS = ("EARLY", "BEGINNING", "OPENING")
_MID_TOKENS = ("MIDDLE", "HALFWAY")

# "END"/"LAST"/"START"/"MID" are bare 3-4 letter fragments that collide with
# common unrelated words as substrings (FRIENDS/SENDING/EXTENDED contain END;
# PLASTIC/BLASTED contain LAST; STARTLED contains START; TIMIDLY/INTIMIDATED
# contain MID) -- anchor them to real word boundaries so only a genuine
# standalone token in the LLM's answer fires.
_END_WORD_RE = re.compile(r"\b(?:END|LAST)\b")
_START_WORD_RE = re.compile(r"\bSTART\b")
_MID_WORD_RE = re.compile(r"\bMID\b")

_META_PATTERNS = (
    r"(?m)^\s*(?:final\s+)*edit\s*[:\-]",          # "Edit: formatting" author notes
    r"\[\s*wp\s*\]",                                # prompt echo tag
    r"guess i gotta contribute",
    r"if no one else will post",
    r"since i'?ve written",                         # "been a few years since I've written"
    r"haven'?t written (?:anything )?in",
    r"sorry for (?:the )?formatting",
    r"english is not my first",
    r"thanks for reading",
    r"\bobligatory\b",
)


def _pos_label(v):
    u = (v or "").strip().upper()
    if not u or "NONE" in u:
        return "none"
    if any(t in u for t in _EARLY_TOKENS) or _START_WORD_RE.search(u):
        return "early"
    if any(t in u for t in _MID_TOKENS) or _MID_WORD_RE.search(u):
        return "middle"
    if any(t in u for t in _END_TOKENS) or _END_WORD_RE.search(u):
        return "end"
    return "unknown"


# "EARN"/"FAIR"/"INFO" are bare fragments that collide with unrelated words as
# substrings (LEARNED/LEARNING/EARNEST contain EARN; AFFAIRS/FAIRLAWN/
# FAIRYTALES contain FAIR; REINFORCEMENTS contains INFO) -- anchor to word
# boundaries (with an explicit inflection whitelist for EARN, since the bare
# boundary alone would still catch "EARNEST") so only the intended concept fires.
_EARN_RE = re.compile(r"\bEARN(?:ED|S|ING)?\b")
_FAIR_RE = re.compile(r"\bFAIR\b")
_INFO_RE = re.compile(r"\bINFO\w*\b")


def _craft_label(v):
    u = (v or "").strip().upper()
    if not u or u.startswith("NONE"):
        return "none"
    # negatives first so "NOT EARNED"/"UNEARNED" never read as earned
    if ("UNEARNED" in u or "NOT EARNED" in u or "GOTCHA" in u
            or "CHEAP" in u or "RANDOM" in u or "NOWHERE" in u):
        return "gotcha"
    if (_EARN_RE.search(u) or "FORESHADOW" in u or _FAIR_RE.search(u)
            or "SETUP" in u or "SET UP" in u):
        return "earned"
    if "DUMP" in u or _INFO_RE.search(u) or "EXPLAIN" in u:
        return "dump"
    if "NONE" in u:
        return "none"
    return "unknown"


_POS_DELTA = {"none": -0.10, "early": -0.12, "middle": 0.02, "end": 0.18, "unknown": 0.0}
_CRAFT_DELTA = {"none": -0.08, "gotcha": -0.22, "earned": 0.22, "dump": -0.18, "unknown": 0.0}


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        try:
            t = ops.normalize(text)
        except Exception:
            t = text
        low = t.lower()
        words = re.findall(r"[A-Za-z']+", t)
        n = max(1, len(words))

        s = 0.5

        # --- LLM-grounded structural predicate (the criterion itself) ---
        pos = _pos_label(extracted.get("reveal_position", ""))
        craft = _craft_label(extracted.get("reveal_craft", ""))
        pos_delta = _POS_DELTA[pos]
        # a gotcha/info-dump landing at the end is still unfair withholding:
        # the END bonus only accrues when the reveal is not judged cheap
        if craft in ("gotcha", "dump") and pos_delta > 0:
            pos_delta = 0.0
        s += pos_delta + _CRAFT_DELTA[craft]

        # --- code-level robust penalties (judge-punished surface classes) ---

        # 1) author-note / meta chrome (train: 0.0-0.35 band only)
        meta_hits = sum(1 for p in _META_PATTERNS if re.search(p, low))
        # emoticon in the opening chrome region only (subreddit-plug ":)" at
        # the tail of good stories must NOT be penalized)
        if re.search(r"[;:]\)|:p\b", low[:200]):
            meta_hits += 1
        s -= 0.12 * min(meta_hits, 2)

        # 2) serial "it turns out" reveal chains = gotcha cascade / premature
        #    explanation (d04046 has 4; high scorers have 0-1)
        c_turns = len(re.findall(r"\bturn(?:s|ed)\s+out\b", low))
        if c_turns >= 3:
            s -= 0.15
        elif c_turns == 2:
            s -= 0.06

        # 3) explicit info-dump framing
        c_dump = len(re.findall(r"\bas you know\b|\blet me explain\b", low))
        s -= 0.06 * min(c_dump, 2)

        # 4) too short to withhold anything (skits / fragments)
        if n < 160:
            s -= 0.10
        elif n < 300:
            s -= 0.04

        # 5) sloppy craft: unpunctuated lowercase "i" pronoun density
        #    (typo-laden lows like d00120; threshold spares chat-format fiction)
        ci = len(re.findall(r"\bi\b", t))  # case-sensitive: lowercase only
        if ci / float(n) > 0.02:
            s -= 0.08

        # 6) space-before-punctuation typos
        if len(re.findall(r"\s[,.!?;](?:\s|$)", t)) >= 3:
            s -= 0.05

        return float(max(0.0, min(1.0, s)))
    except Exception:
        return 0.5
