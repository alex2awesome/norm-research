"""
Hybrid channel for aspect a324: "Brevity and ellipsis-based wit"

Criterion: concise phrasing and omission that let the audience infer the
elided material; a paraphrase that keeps the sense but changes the wording
should lose the laugh -- i.e. the wit lives in the FORM (specific words,
what is left unsaid), not in the propositional content of the joke.

Design notes (derived only from the pack's contract/hazard-notes/examples):
- Code alone cannot judge whether a punchline "states" its point outright
  or "implies" it and leaves inferential work to the reader, nor whether a
  meaning-preserving paraphrase would kill the joke. Those are exactly the
  THICK constructs the contract says to hand to an LLM field, keeping the
  scoring predicate itself in code.
- Structural/computable correlates of economy-of-form that code CAN see:
  overall brevity, literal ellipsis punctuation, and a "compressed final
  line" pattern (the punchline being much shorter than the setup that
  precedes it). These provide a deterministic floor/adjustment and keep the
  channel from collapsing to a pure LLM lookup.
- Scraped-corpus hazard: trailing credit/citation/"edit:" boilerplate lines
  can masquerade as a short "punchline" line and corrupt the compression
  signal, so trailing footer-like lines are stripped before structural
  analysis.
"""

import re

LLM_FIELDS = {
    "resolution_mode": (
        "Does the punchline STATE its point explicitly, or IMPLY it so the "
        "reader must infer the unstated meaning? Answer STATES or IMPLIES."
    ),
    "paraphrase_kills_joke": (
        "If reworded with different words but the same meaning, would this "
        "joke stop being funny? Answer YES or NO."
    ),
}

_FOOTER_RE = re.compile(
    r"^\s*(?:https?://\S+|(?:credit|source|edit|update)s?\s*[:\-]|"
    r"\^?\(.*\)\s*$|/?[ur]/\w+)",
    re.IGNORECASE,
)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def _strip_trailing_footer(text):
    lines = text.split("\n")
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if _FOOTER_RE.search(last) or "reddit.com" in last.lower():
            lines.pop()
            continue
        break
    out = "\n".join(lines).strip()
    return out if out else text


def _brevity_signal(n_chars):
    if n_chars < 50:
        return 0.3
    if n_chars <= 700:
        return 1.0
    if n_chars <= 1200:
        return max(0.15, 0.6 - (n_chars - 700) / 1250.0)
    return 0.15


def _ellipsis_signal(text):
    hits = text.count("...") + text.count("…")
    return min(1.0, hits / 2.0)


def _compression_signal(core):
    pieces = [p.strip() for p in _SENT_SPLIT_RE.split(core) if p.strip()]
    if len(pieces) < 3:
        return 0.5
    last_len = len(pieces[-1].split())
    head_lens = [len(p.split()) for p in pieces[:-1]]
    head_mean = sum(head_lens) / len(head_lens) if head_lens else last_len
    if head_mean <= 0:
        return 0.5
    ratio = last_len / head_mean
    if ratio <= 0.6:
        return 1.0
    if ratio <= 1.0:
        return 0.7
    return 0.3


def _sentence_economy_signal(ops, core):
    try:
        stats = ops.sent_stats(core)
    except Exception:
        return 0.5
    try:
        if isinstance(stats, dict):
            mean_wps = float(stats.get("mean_words_per_sent", 0) or 0)
        else:
            mean_wps = float(stats[1]) if len(stats) > 1 else 0.0
    except Exception:
        return 0.5
    if mean_wps <= 0:
        return 0.5
    if 4 <= mean_wps <= 16:
        return 1.0
    if mean_wps < 4:
        return 0.6
    return max(0.2, 1.0 - (mean_wps - 16) / 20.0)


def _llm_resolution_signal(extracted):
    val = str(extracted.get("resolution_mode", "") or "").strip().upper()
    if not val:
        return 0.5
    if "IMPLI" in val:
        return 1.0
    if "STATE" in val:
        return 0.0
    return 0.5


def _llm_paraphrase_signal(extracted):
    val = str(extracted.get("paraphrase_kills_joke", "") or "").strip().upper()
    if not val:
        return 0.5
    if val.startswith("Y") or "YES" in val:
        return 1.0
    if val.startswith("N") or "NO" in val:
        return 0.0
    return 0.5


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not isinstance(text, str):
            return 0.0

        try:
            norm = ops.normalize(text)
            if not isinstance(norm, str) or not norm:
                norm = text
        except Exception:
            norm = text

        core = _strip_trailing_footer(norm)
        extracted = extracted if isinstance(extracted, dict) else {}

        implies_signal = _llm_resolution_signal(extracted)
        wordplay_signal = _llm_paraphrase_signal(extracted)

        brevity = _brevity_signal(len(core))
        elli = _ellipsis_signal(norm)
        compression = _compression_signal(core)
        sent_economy = _sentence_economy_signal(ops, core)

        code_struct = (
            0.40 * brevity
            + 0.25 * elli
            + 0.25 * compression
            + 0.10 * sent_economy
        )
        code_struct = max(0.0, min(1.0, code_struct))

        raw = 0.45 * implies_signal + 0.30 * wordplay_signal + 0.25 * code_struct
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
