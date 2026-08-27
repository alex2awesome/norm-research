"""Hybrid metric channel for a333 — Book and Layout Design: Clarity, Readability, and Promise.

Insight from train residuals: on plain-text short fiction, the judge rewards
DELIBERATE VISUAL / DOCUMENT FORM — stories written as emails, chat logs,
log entries, letter-spaced typography — and pushes messy prose (typos,
author apologies, hard line-wrapping, no paragraphing) toward zero.
Keyword mentions of "design/layout" are irrelevant (baseline rho ~0.17).

Predicate stays in code (line-level structural census); one LLM field grounds
the document-format read on the FULL story (excerpt-invisible middles), one
names layout devices. Both map through fixed keyword buckets in code.
"""

import re
import math

LLM_FIELDS = {
    "doc_format": (
        "If this story is written as a non-prose document or transcript "
        "(chat log, email, letter, diary, forum thread, news report, "
        "screenplay, list), name that format in 2-4 words; otherwise answer NONE."
    ),
    "layout_devices": (
        "Name any deliberate visual layout devices used in this text "
        "(section headers, divider lines, letter-spaced words, all-caps lines, "
        "dropcap, bulleted list); otherwise answer NONE."
    ),
}

# --- fixed vocab: LLM answer -> format strength (token-stem matched) ----------
# NOTE: "log", "list", "memo" (STRONG), "play", "verse" (WEAK), "tale" (NON)
# were pulled out of the open-suffix startswith() stems below because they
# collide with unrelated tokens ("logic", "listening", "memory", "playstation",
# "versed", "talent") under prefix matching (scanner-caught bugs). They are
# re-checked via the explicit whitelists/regexes just below instead.
_STRONG_FORMAT = (
    "chat", "email", "mail", "letter", "epistolar", "correspond",
    "messag", "forum", "thread", "transcript", "diary", "journal",
    "report", "entry", "entries", "article", "interview", "review",
    "wiki", "encyclopedia", "document", "record", "bulletin", "newslet",
)
_WEAK_FORMAT = ("screenplay", "script", "dialog", "poem")
_NON_FORMAT = ("none", "prose", "story", "stories", "narrat", "fiction",
               "novel")

# Explicit whitelist of safe inflections for the stems pulled out above
# (checked against whole tokens, same as the startswith() stems, but closed
# instead of open-ended).
_STRONG_FORMAT_EXACT = {
    "log", "logs", "logging", "logbook", "logfile", "logfiles",
    "list", "lists", "listing", "listings", "checklist", "checklists",
    "memo", "memos", "memorandum", "memoranda",
}
_WEAK_FORMAT_EXACT = {
    "play", "plays", "playscript", "playscripts",
    "verse", "verses", "versify", "versifying", "versified",
}
_NON_FORMAT_EXACT = {"tale", "tales", "taletelling"}

_DEVICE_WORDS = (
    "header", "heading", "divider", "rule", "spaced", "spacing", "letter-spaced",
    "all-caps", "all caps", "uppercase", "dropcap", "drop cap", "bullet",
    "column", "ascii", "centered", "indent",
)
# "list" pulled out of _DEVICE_WORDS: was a bare (unbounded) substring check
# that fired on "listening"/"glistening"/"ballistic"/etc (scanner-caught bug).
_DEVICE_LIST_RE = re.compile(r"\b(?:list|lists|listing|listings|checklist|checklists|"
                              r"bulleted|numbered)\b")

# --- line-level structural patterns -------------------------------------------
_LABEL_RE = re.compile(r"^\**([A-Z][A-Za-z0-9 .'@_\-]{0,28})\**\s*:\s*(\S.*)$")
_META_LINE_RE = re.compile(r"^\*[^*]{4,120}\*$")          # *server msg* / *stage dir*
_HR_RE = re.compile(r"^[\s\-*_=~#]{3,}$")
# field-style document apparatus: "To:", "Subject:", "Entry 102394",
# "Accessing ... Log", "Dear X", sign-offs. NOT loose prefixes ("To be...").
_FIELD_RES = (
    re.compile(
        r"^(to|from|cc|bcc|subject|date|time|location|server|players?|"
        r"re|sent|attn|status)\s*[:—\-]\s*\S", re.I),
    re.compile(r"^(entry|log|day|chapter|record|report|case|file|episode)\s*"
               r"(no\.?\s*|#\s*)?\d", re.I),
    re.compile(r"^accessing\b|^begin (transmission|log|recording)\b|"
               r"^end (of )?(transmission|log|recording)\b", re.I),
    re.compile(r"^dear\s+[A-Z]"),
    re.compile(r"^(sincerely|regards|best regards|yours truly|signed)\b[,.]?\s*$",
               re.I),
)
_JOINLEAVE_RE = re.compile(
    r"\bhas (joined|left|disconnected|connected|logged (in|out|on|off))\b", re.I
)
_SPACED_RUN_RE = re.compile(r"(?:\b\w \w \w(?: \w)+)")     # "w h y", "v e r y"
_DROPCAP_RE = re.compile(r"#+\s*\[?\]?\(#?dropcap\)|^#{2,6}\s", re.M)
_APOLOGY_RE = re.compile(
    r"sorry for (the )?(bad |my )?(formatting|grammar|spelling|english)|"
    r"first (wp |writing ?prompts? )?post|please be kind|"
    r"isn'?t going to be great|not (very )?good at writ|"
    r"haven'?t written (in|for)|wrote this on my phone|still crap|"
    r"i guess i gotta contribute|be gentle|constructive criticism is welcome",
    re.I,
)
_EDIT_NOTE_RE = re.compile(r"^\s*\**\s*edit\s*\d*\s*[:\-]", re.I | re.M)
_WORD_RE = re.compile(r"[A-Za-z']+")


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _llm_format_flag(extracted):
    ans = (extracted.get("doc_format") or "").strip().lower()
    if not ans:
        return 0.0
    toks = re.findall(r"[a-z]+", ans)

    def hit(vocab):
        return any(t.startswith(stem) for t in toks for stem in vocab)

    def hit_exact(exact_set):
        return any(t in exact_set for t in toks)

    if hit(_STRONG_FORMAT) or hit_exact(_STRONG_FORMAT_EXACT):
        return 1.0
    if hit(_WEAK_FORMAT) or hit_exact(_WEAK_FORMAT_EXACT):
        return 0.35
    if hit(_NON_FORMAT) or hit_exact(_NON_FORMAT_EXACT):
        return 0.0
    return 0.2  # named something, but off-vocab: mild credit only


def _llm_device_score(extracted):
    ans = (extracted.get("layout_devices") or "").strip().lower()
    if not ans or ans.startswith("none"):
        return 0.0
    hits = sum(1 for w in _DEVICE_WORDS if w in ans)
    if _DEVICE_LIST_RE.search(ans):
        hits += 1
    return _clip(hits / 3.0)


def _structural_census(text):
    """Line-level census of document apparatus vs prose."""
    lines = [ln.rstrip() for ln in text.split("\n")]
    nonempty = [ln for ln in lines if ln.strip()]
    n = max(1, len(nonempty))

    chat_lines = 0      # Name: message   (no quoted dialogue after label)
    script_lines = 0    # **Name**: "quoted line"
    meta_lines = 0      # *system message / stage direction*
    hr_lines = 0
    field_hits = 0
    caps_lines = 0

    for ln in nonempty:
        s = ln.strip()
        if _HR_RE.match(s) and len(s) >= 3 and not _WORD_RE.search(s):
            hr_lines += 1
            continue
        if _META_LINE_RE.match(s):
            meta_lines += 1
        bare = s.strip("*# ")
        if len(bare) < 90 and any(rx.match(bare) for rx in _FIELD_RES):
            field_hits += 1
        m = _LABEL_RE.match(s)
        if m and len(m.group(1).split()) <= 3:
            body = m.group(2)
            if body.lstrip().startswith(('"', "“", "'", "‘")):
                script_lines += 1
            else:
                chat_lines += 1
        letters = re.sub(r"[^A-Za-z]", "", s)
        if len(letters) >= 8 and letters.isupper():
            caps_lines += 1

    apparatus_hits = field_hits + len(_JOINLEAVE_RE.findall(text))
    chat_frac = chat_lines / n
    script_frac = script_lines / n
    meta_frac = meta_lines / n
    # sustained structure: chat/meta lines are full-strength, script demoted
    structure = _clip((chat_frac + meta_frac + 0.35 * script_frac) / 0.5)
    caps_frac = caps_lines / n
    return apparatus_hits, structure, chat_frac, script_frac, caps_frac, hr_lines


def _prose_hygiene(text, ops):
    """Small, robust orderings inside the prose mass (judge 0.0-0.25)."""
    adj = 0.0
    # author apologies / self-deprecation about craft or formatting
    if _APOLOGY_RE.search(text):
        adj -= 0.10
    if _EDIT_NOTE_RE.search(text):
        adj -= 0.02
    # lowercase mess: sentences/lines starting lowercase, lowercase "i"
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        low_starts = sum(1 for ln in lines if ln[0].islower())
        if low_starts / len(lines) > 0.3:
            adj -= 0.08
    if len(re.findall(r"(?<![A-Za-z])i(?=[ ,'])", text)) >= 3:
        adj -= 0.05
    # paragraphing: blank-line separated paragraphs of sane size
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) >= 3:
        wcounts = [len(_WORD_RE.findall(p)) for p in paras]
        mean_w = sum(wcounts) / len(wcounts)
        if 15 <= mean_w <= 220:
            adj += 0.04
    else:
        # hard-wrapped or single-block wall
        adj -= 0.04
    # readability: moderate sentence length (criterion mentions readability)
    try:
        n_sent, mean_wps, _fl = ops.sent_stats(text)
        if n_sent >= 5:
            adj += 0.03 * (1.0 - _clip(abs(mean_wps - 16.0) / 20.0))
    except Exception:
        pass
    # very short fragments read as low-effort
    if len(_WORD_RE.findall(text)) < 220:
        adj -= 0.04
    return adj


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        try:
            t = ops.normalize(text)
        except Exception:
            t = text
        if not isinstance(t, str) or not t.strip():
            t = text

        apparatus_hits, structure, chat_frac, script_frac, caps_frac, hr_lines = (
            _structural_census(t)
        )
        llm_flag = _llm_format_flag(extracted or {})
        device = _llm_device_score(extracted or {})

        # code-side format flag: needs REAL document apparatus (>=2 field-style
        # header lines / join-leave metadata) or sustained unquoted labeling.
        code_flag = 0.0
        if apparatus_hits >= 3 or chat_frac >= 0.25:
            code_flag = 1.0
        elif apparatus_hits >= 2 or chat_frac >= 0.12:
            code_flag = 0.6
        elif script_frac >= 0.25:
            code_flag = 0.35

        flag = max(llm_flag, code_flag)
        # gated form score: format flag opens the gate, structural density
        # inside the document decides how high (log-entry-then-prose ~0.3,
        # fully structured chat/email ~0.75-0.8)
        s_form = flag * (0.35 + 0.65 * structure) if flag > 0 else 0.0

        # typography flair (letter-spaced words, dropcaps, dividers, caps lines)
        spaced = len(_SPACED_RUN_RE.findall(t))
        flair = 0.0
        flair += 0.10 * _clip(spaced / 2.0)
        if _DROPCAP_RE.search(t):
            flair += 0.04
        if hr_lines >= 1:
            flair += 0.02
        if 0 < caps_frac <= 0.15:
            flair += 0.02
        flair += 0.05 * device

        base = 0.14 + _prose_hygiene(t, ops)
        out = base + 0.60 * s_form + flair * (1.0 - s_form)
        return float(_clip(out))
    except Exception:
        return 0.5
