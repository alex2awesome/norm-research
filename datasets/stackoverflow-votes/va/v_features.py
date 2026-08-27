#!/usr/bin/env python3
"""Deterministic, label-blind V features for StackOverflow [python] ANSWERS
(V6 vote cell).

Adapted from `datasets/math/stackexchange/va/v_features.py` -- same module
shape, same generic-surface tail (length / structure / register), so the two
StackExchange vote cells share a comparable surface channel. The math-specific
block (LaTeX density, display math, proof framing, heavy machinery) is replaced
by the CODE-specific block this corpus actually carries: fenced and indented
code blocks, inline code spans, tracebacks, shell/pip invocations, imports,
version pins, output blocks, and documentation links.

Input is the ANSWER BODY ONLY. The question title/body is stripped by the
caller so question length cannot leak into an "answer style" feature. Neither
the vote score, the median, the accept flag, the position, nor any date is
accepted or inspected -- every column is a regex count/ratio over the answer
text.

BODY FORMAT: Markdown, not HTML. Measured over 400K answers in this mirror,
<p>/<pre> appear on 0.21% of rows while ``` fences appear on 61.3%. The
extractors below therefore parse Markdown; nothing calls an HTML stripper
(the legacy pool builder's re.sub(r"<[^>]+>", " ") would delete `List<int>`,
`<module>`, `<class 'x'>` and any `a < b ... c > d` span -- a code shredder on
this corpus).
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List

LEX = {
    # --- explanatory register (shared shape with the math.SE module) --------
    "v_deductive": r"\b(?:thus|therefore|hence|because|so that|which means|"
                   r"consequently|the reason)\b",
    "v_hedging": r"\b(?:i think|maybe|perhaps|probably|not sure|might be|seems|"
                 r"i believe|afaik|as far as i know|should work)\b",
    "v_first_person": r"\b(?:i|my|me|i'm|i've)\b",
    "v_second_person": r"\b(?:you|your|you're|you've)\b",
    "v_imperative_hint": r"\b(?:note that|notice|observe that|keep in mind|"
                         r"remember|beware|caveat|caution|warning)\b",
    "v_instruction_verb": r"\b(?:try|run|install|use|replace|change|add|remove|"
                          r"set|call|import|create|open|edit|check)\b",
    "v_meta_edit": r"\b(?:edit|update[d]?|added|corrected|typo|oops|thanks|"
                   r"as of|deprecated since)\b",
    # --- diagnosis / cause language (the "diagnoses the actual cause" probe)-
    "v_cause_language": r"\b(?:the (?:problem|issue|error|cause|reason) is|"
                        r"this happens because|root cause|that's why|"
                        r"the culprit)\b",
    "v_error_mention": r"\b(?:error|exception|traceback|fails?|failure|crash|"
                       r"bug|warning|stacktrace)\b",
    # --- code-domain content markers ---------------------------------------
    "v_import_mention": r"(?m)^\s*(?:import|from)\s+[A-Za-z_][\w.]*",
    "v_shell_invocation": r"(?m)^\s*(?:\$|>>>|#\s)?\s*(?:pip3?|conda|apt-get|"
                          r"brew|python3?|npm|sudo|docker)\s+\S",
    "v_repl_prompt": r">>>|\.\.\.\s",
    "v_traceback_marker": r"(?m)Traceback \(most recent call last\)|"
                          r"^\s*File \"[^\"]+\", line \d+",
    "v_version_pin": r"\b(?:python\s*[23](?:\.\d+)+|"
                     r"v?\d+\.\d+\.\d+|>=\s*\d+\.\d+|==\s*\d+\.\d+)\b",
    "v_doc_link": r"https?://(?:docs\.python\.org|docs\.|"
                  r"[\w.-]*readthedocs|.*\.readthedocs)",
    "v_so_selflink": r"https?://(?:[\w.-]*\.)?stackoverflow\.com|"
                     r"https?://stackexchange\.com",
    "v_github_link": r"https?://(?:www\.)?github\.com",
    "v_generality_marker": r"\b(?:in general|more generally|for any|"
                           r"any (?:number|version|case)|works for all|"
                           r"regardless of)\b",
    "v_edge_case_marker": r"\b(?:edge case|corner case|caveat|however,? if|"
                          r"note that if|unless|only if|except when|"
                          r"in case)\b",
    "v_perf_marker": r"\b(?:faster|slower|performance|efficient|O\(|"
                     r"complexity|memory|vectoriz|timeit|benchmark)\b",
    "v_alternative_marker": r"\b(?:alternatively|another (?:way|option|"
                            r"approach)|or you (?:can|could)|"
                            r"a better way|instead)\b",
}
_LEX_RE = {k: re.compile(v, re.I) for k, v in LEX.items()}

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
SENTENCE_END_RE = re.compile(r"[.!?]+(?:\s|$)")
LIST_MARK_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s")
FENCE_RE = re.compile(r"(?s)```.*?```|~~~.*?~~~")
FENCE_OPEN_RE = re.compile(r"(?m)^\s*(?:```|~~~)")
INDENT_CODE_RE = re.compile(r"(?m)^(?: {4}|\t)\S")
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
MD_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")
BARE_URL_RE = re.compile(r"https?://\S+")
HEADING_RE = re.compile(r"(?m)^#{1,6}\s|\S\n(?:=+|-{2,})\s*$")
# NB: `(?!>)` is load-bearing -- without it the Python REPL prompt `>>>`, which
# appears on a large share of this corpus, is counted as a Markdown blockquote.
BLOCKQUOTE_RE = re.compile(r"(?m)^\s*>(?!>)")
COMMENT_RE = re.compile(r"(?m)^\s*#[^!]")

V_NAMES = list(LEX) + [
    # generic structure/length tail (mirrors the math.SE module)
    "v_log_len", "v_word_count", "v_sentence_count", "v_avg_sentence_words",
    "v_numeral_density", "v_alpha_share", "v_question_marks",
    "v_uppercase_letter_ratio", "v_linebreak_count", "v_paragraph_count",
    "v_list_marker_count", "v_type_token_ratio",
    # code-structure tail (the math cell's LaTeX tail, re-pointed at code)
    "v_n_code_fences", "v_code_fence_chars", "v_code_char_share",
    "v_n_indent_code_lines", "v_n_inline_code", "v_inline_code_density",
    "v_n_md_links", "v_n_urls", "v_n_headings", "v_n_blockquotes",
    "v_n_code_comments", "v_prose_char_count", "v_prose_to_code_ratio",
]


def _split_code_prose(b: str):
    """Return (code_text, prose_text) using Markdown fences + 4-space indents."""
    fences = FENCE_RE.findall(b)
    rest = FENCE_RE.sub(" ", b)
    indented = [ln for ln in rest.split("\n")
                if INDENT_CODE_RE.match(ln + "\n") or
                (ln.startswith("    ") and ln.strip())]
    code = "\n".join(fences + indented)
    prose = "\n".join(ln for ln in rest.split("\n")
                      if not (ln.startswith("    ") and ln.strip())
                      and not ln.startswith("\t"))
    return code, prose


def v_features(body: str) -> Dict[str, float]:
    b = body or ""
    n = len(b)
    words = WORD_RE.findall(b)
    nw = len(words)
    letters = [c for c in b if c.isalpha()]
    n_sent = max(len(SENTENCE_END_RE.findall(b)), 1 if b else 0)
    code, prose = _split_code_prose(b)

    out = {k: float(len(_LEX_RE[k].findall(b))) for k in LEX}
    out["v_log_len"] = float(math.log1p(n))
    out["v_word_count"] = float(nw)
    out["v_sentence_count"] = float(n_sent)
    out["v_avg_sentence_words"] = float(nw / max(n_sent, 1))
    out["v_numeral_density"] = float(sum(c.isdigit() for c in b) / (n + 1))
    out["v_alpha_share"] = float(len(letters) / (n + 1))
    out["v_question_marks"] = float(b.count("?"))
    out["v_uppercase_letter_ratio"] = float(
        sum(c.isupper() for c in letters) / max(len(letters), 1))
    out["v_linebreak_count"] = float(b.count("\n"))
    out["v_paragraph_count"] = float(b.count("\n\n") + 1)
    out["v_list_marker_count"] = float(len(LIST_MARK_RE.findall(b)))
    lows = [w.lower() for w in words]
    out["v_type_token_ratio"] = float(len(set(lows)) / max(nw, 1))

    out["v_n_code_fences"] = float(len(FENCE_OPEN_RE.findall(b)) // 2
                                   + len(FENCE_RE.findall(b)) * 0)
    out["v_n_code_fences"] = float(len(FENCE_RE.findall(b)))
    out["v_code_fence_chars"] = float(sum(len(x) for x in FENCE_RE.findall(b)))
    out["v_code_char_share"] = float(len(code) / (n + 1))
    out["v_n_indent_code_lines"] = float(len(INDENT_CODE_RE.findall(b)))
    out["v_n_inline_code"] = float(len(INLINE_CODE_RE.findall(b)))
    out["v_inline_code_density"] = float(len(INLINE_CODE_RE.findall(b)) / (nw + 1))
    out["v_n_md_links"] = float(len(MD_LINK_RE.findall(b)))
    out["v_n_urls"] = float(len(BARE_URL_RE.findall(b)))
    out["v_n_headings"] = float(len(HEADING_RE.findall(b)))
    out["v_n_blockquotes"] = float(len(BLOCKQUOTE_RE.findall(b)))
    out["v_n_code_comments"] = float(len(COMMENT_RE.findall(code)))
    out["v_prose_char_count"] = float(len(prose))
    out["v_prose_to_code_ratio"] = float(len(prose) / (len(code) + 1))

    ordered = {k: out[k] for k in V_NAMES}
    assert list(ordered) == V_NAMES
    assert all(math.isfinite(v) for v in ordered.values()), \
        [k for k, v in ordered.items() if not math.isfinite(v)]
    return ordered


def vector(body: str, names: Iterable[str] = V_NAMES) -> List[float]:
    vals = v_features(body)
    return [vals[k] for k in names]


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        s = ("The problem is that `dict.keys()` returns a view in Python 3.\n\n"
             "```python\n>>> d = {'a': 1}\n>>> list(d.keys())\n['a']\n```\n\n"
             "See the [docs](https://docs.python.org/3/library/stdtypes.html).\n")
        print(json.dumps(v_features(s), indent=1))
    print(f"n_features = {len(V_NAMES)}")
