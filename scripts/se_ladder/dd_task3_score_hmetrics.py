"""Deep dive task 3: h-metrics — anatomy-derived candidate metrics for the
SO[python] articulability residual (2026-06-12).

Derived from hand-reading 100 residual examples (dense-FT confident-correct,
bank-ENS wrong; `dd_anatomy_sample.jsonl`). The dense model reads EPISTEMIC
REGISTER and EVIDENCE QUALITY, which g1-g6 (presentation geometry) and the
aNNN code metrics miss:

  POSITIVE register: mechanism explanations, exact-replacement directives
  ("use X instead", before/after blocks), quoted documentation, deep doc /
  changelog / commit links, version-aware statements.
  NEGATIVE register: self-doubt, personal anecdote ("I had the same
  problem"), meta-moderation ("Welcome to StackOverflow... guidelines"),
  unverified suggestions ("try this, it should work"), questions back at
  the asker, EDIT markers, deprecated pandas/numpy APIs.

All metrics are candidate-only (body + code), regex/string level, cheap.
Each emits {id}_score + {id}_applied.

Usage:
  python dd_task3_score_hmetrics.py <input.parquet> <output.parquet>
input must have columns row_id, body, code.
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

FENCE_RE = re.compile(r"^[ \t]*(?:```|~~~)[^\n]*\n.*?^[ \t]*(?:```|~~~)[ \t]*$",
                      re.DOTALL | re.MULTILINE)
FENCE_BLOCK_RE = re.compile(
    r"^[ \t]*(?:```|~~~)[^\n]*\n(.*?)^[ \t]*(?:```|~~~)[ \t]*$",
    re.DOTALL | re.MULTILINE)
INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")

SELF_DOUBT_RE = re.compile(
    r"\b(i (?:do not|don'?t) (?:exactly )?know|"
    r"i[' a]m not sure|not sure (?:what|if|why|how)|"
    r"i guess|i suppose|no idea|i can[' no]*t tell|don[']?t ask me|"
    r"i[' a]m not (?:an? )?expert|off the top of my head)\b", re.I)
ANECDOTE_RE = re.compile(
    r"\b(i had (?:the |a )?(?:same|similar)|in my case|worked for me|"
    r"works for me|i ended up|i was able to (?:fix|solve)|i (?:fixed|solved) "
    r"(?:it|this|the)|my (?:problem|issue|error) was|after (?:a bit|some) of? "
    r"?(?:trouble ?shooting|debugging|searching)|i (?:also )?(?:ran|run) into)"
    r"\b", re.I)
META_MOD_RE = re.compile(
    r"\b(welcome to stack ?overflow|minimal reproducible|please (?:kindly )?"
    r"(?:provide|share|post|add|include|follow)|these guidelines|"
    r"hope (?:this|it) helps|hth\b|please accept|up ?vote|"
    r"mark (?:it |this )?as (?:the )?answer|as (?:mentioned|stated) in "
    r"(?:the )?comments?)\b", re.I)
ASKER_Q_RE = re.compile(
    r"\b(can you (?:give|provide|share|post|explain|handle|clarify)|"
    r"could you (?:give|provide|share|post|explain|clarify)|"
    r"what (?:exactly )?(?:do you|are you trying)|do you (?:want|mean|need)|"
    r"if you could expand|what is your)\b", re.I)
UNVERIFIED_RE = re.compile(
    r"\b(try this|you can try|you could try|try the following|"
    r"(?:it |this |that )?should (?:work|do the (?:job|trick)|be able|solve|"
    r"fix|help|suffice|take)|hopefully|probably work|might work|may work|"
    r"not tested|untested|i have ?n[o']t tested|something like this should)"
    r"\b", re.I)
DIRECTIVE_RE = re.compile(
    r"^(?:you (?:need|should|have|must|want)|use |you can use|don[']?t use|"
    r"instead of|replace |change |remove |add |the (?:problem|issue|error|"
    r"reason|method|main problem|handler|default)|because |it is because|"
    r"this (?:is|error|happens|behavi)|there (?:is|are))", re.I)
EDIT_RE = re.compile(r"^[ \t]*(?:\*\*)?(?:edit|update)\s*[:\d]", re.I | re.M)
VERSION_RE = re.compile(
    r"\b(?:since|as of|in|before|after|until)\s+(?:numpy|pandas|python|"
    r"tensorflow|keras|django|matplotlib|scipy|scikit-learn|sklearn|flask)?"
    r"\s*v?\d+\.\d+|release notes|changelog|deprecated|was removed in|"
    r"backwards?[- ]incompat", re.I)
DEEP_LINK_RE = re.compile(
    r"https?://\S*?(?:#[\w.-]{3,}|/generated/|/api/|/reference/|"
    r"github\.com/\S+/(?:commit|issues|pull|blob)/)", re.I)
ANY_LINK_RE = re.compile(r"https?://\S+")
SHALLOW_LINK_RE = re.compile(
    r"https?://(?:www\.)?(?:w3schools|tutorialspoint|geeksforgeeks)\.\S+", re.I)
QUOTE_LINE_RE = re.compile(r"^[ \t]*> \S", re.M)
OUTPUT_SHOWN_RE = re.compile(
    r"(^|\n)\s*(?:>>> |In \[\d+\]|Out\[\d+\]|#\s*(?:output|=>|prints?:)|"
    r"output:)", re.I)
DEPRECATED_TOKENS = [
    ".ix[", "pd.rolling_", "rolling_sum(", "rolling_mean(", "rolling_std(",
    ".as_matrix(", "pd.TimeGrouper", ".set_value(", "np.asscalar(",
    "pd.scatter_matrix", ".sort(columns", "pd.tools.", ".convert_objects(",
    "from pandas.io.data", "pd.ewma(", "df.append(", "np.matrix(",
]
PRONOUN_I_RE = re.compile(r"(?:^| )i(?=[ '])")
MECHANISM_RE = re.compile(
    r"\b(because|the (?:reason|problem|issue|error|cause) (?:is|here|was)|"
    r"this (?:is because|happens|means)|which means|due to|so that|"
    r"under the hood|internally|by default|the default)\b", re.I)
BEFORE_AFTER_RE = re.compile(
    r"\b(instead of|replace (?:this|it|that|your)?|rather than|change "
    r"(?:this|it|your)|use this instead|should be)\b", re.I)


def strip_blocks(body: str) -> str:
    return FENCE_RE.sub(" <CODEBLOCK> ", body)


def words(text: str) -> int:
    return max(1, len(text.split()))


def hmetrics(body: str, code: str) -> dict:
    out = {}
    ok = isinstance(body, str) and len(body) >= 5
    if not ok:
        for i in range(1, 20):
            out[f"h{i}_score"] = np.nan
            out[f"h{i}_applied"] = 0
        return out
    prose = strip_blocks(body)
    nw = words(prose)
    has_code = isinstance(code, str) and len(code) > 0

    def put(i, val, applied=1):
        out[f"h{i}_score"] = float(val) if applied else np.nan
        out[f"h{i}_applied"] = applied

    # negative-register metrics (per 1000 prose words)
    put(1, 1000.0 * len(SELF_DOUBT_RE.findall(prose)) / nw)   # self-doubt
    put(2, 1000.0 * len(ANECDOTE_RE.findall(prose)) / nw)     # anecdote
    put(3, 1000.0 * len(META_MOD_RE.findall(prose)) / nw)     # meta-moderation
    put(4, 1000.0 * (len(ASKER_Q_RE.findall(prose))
                     + prose.count("?")) / nw)                # asker-questions
    put(5, 1000.0 * len(UNVERIFIED_RE.findall(prose)) / nw)   # unverified-try
    put(12, float(len(EDIT_RE.findall(body))))                # edit markers
    put(14, 1000.0 * len(PRONOUN_I_RE.findall(prose)) / nw)   # lowercase i

    # positive-register metrics
    put(6, float(bool(DIRECTIVE_RE.match(prose.strip()[:120]))))  # directive opener
    put(7, float(len(QUOTE_LINE_RE.findall(body))))           # doc-quote lines
    n_deep = len(DEEP_LINK_RE.findall(body))
    n_links = len(ANY_LINK_RE.findall(body))
    n_shallow = len(SHALLOW_LINK_RE.findall(body))
    put(8, float(n_deep - n_shallow))                         # link specificity
    put(9, float(len(VERSION_RE.findall(prose))))             # version-aware
    put(13, float(len(OUTPUT_SHOWN_RE.findall(body))))        # output shown

    # code-dependent metrics
    if has_code:
        low = (body or "").lower()
        put(10, float(sum(low.count(t.lower()) for t in DEPRECATED_TOKENS)))
        # prose-code coherence: share of inline-code tokens that appear in code
        inl = set()
        for m in INLINE_CODE_RE.findall(strip_blocks(body)):
            for tok in re.split(r"[^\w.]+", m):
                if len(tok) >= 3:
                    inl.add(tok.lower())
        if inl:
            code_low = code.lower()
            hits = sum(1 for t in inl if t in code_low)
            put(11, hits / len(inl))
        else:
            put(11, np.nan, 0)
    else:
        put(10, np.nan, 0)
        put(11, np.nan, 0)

    # g-extensions: length / block-shape distribution
    put(15, float(np.log1p(nw)))                              # prose words
    put(16, float(np.log1p(len(code) if has_code else 0)))    # code chars
    blocks = FENCE_BLOCK_RE.findall(body)
    if blocks:
        lines = [b.count("\n") + 1 for b in blocks]
        put(17, float(max(lines)))                            # max block lines
    else:
        put(17, np.nan, 0)
    put(18, 1000.0 * len(MECHANISM_RE.findall(prose)) / nw)   # mechanism
    put(19, float(len(BEFORE_AFTER_RE.findall(prose))
                  * (1 if len(blocks) >= 2 else 0)
                  + len(BEFORE_AFTER_RE.findall(prose))))     # before/after
    return out


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    df = pd.read_parquet(inp, columns=["row_id", "body", "code"])
    recs = []
    for r in df.itertuples(index=False):
        rec = {"row_id": r.row_id}
        rec.update(hmetrics(r.body, r.code))
        recs.append(rec)
    out = pd.DataFrame(recs)
    out.to_parquet(outp, index=False)
    print(f"wrote {outp} ({len(out)} rows, {len(out.columns)-1} cols)")


if __name__ == "__main__":
    main()
