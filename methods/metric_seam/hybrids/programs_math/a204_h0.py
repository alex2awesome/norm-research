import re

LLM_FIELDS = {
    "broken_spacing_example": "Quote verbatim (<=15 words) one spot in this answer where math spacing, an ellipsis, or a line/formula break looks visually awkward or confusing; say NONE if there is no such spot."
}

_ASCII_ELLIPSIS_RE = re.compile(r'(?<!\\)\.{3,}')
_THIN_CMDS = ('\\,', '\\;', '\\:', '\\!')
_PROPER_ELLIPSIS_CMDS = ('\\ldots', '\\cdots', '\\dots', '\\vdots')
_BACKSLASH_SPACE_RE = re.compile(r'\\ ')
_UNICODE_ELLIPSIS = '…'
_DISPLAY_BLOCK_RE = re.compile(r'\$\$.+?\$\$', re.S)
# scrape/conversion artifacts around inline math boundaries: a stray space
# stranding the following punctuation ("$3$ ," / "$n$ .") -- the classic
# "poor spacing around math" defect for this corpus.
_SPACE_BEFORE_PUNCT_RE = re.compile(r'\$[ \t]+[,.;:!?)\]]')
# closing $ glued directly onto the next sentence with no space at all
# ("$G$.So") -- the opposite failure, also a spacing/line-break hazard.
_GLUED_AFTER_RE = re.compile(r'\$[.,;:!?][A-Za-z]')

# (unicode glyph, [equivalent latex macro spellings]) -- used only to flag
# same-operation typeset two different ways within one document.
_UNI_MACRO_PAIRS = (
    ('∪', ['\\cup']), ('∩', ['\\cap']), ('∖', ['\\setminus']),
    ('∈', ['\\in']), ('∉', ['\\notin']), ('⊂', ['\\subset']), ('⊆', ['\\subseteq']),
    ('→', ['\\to', '\\rightarrow']), ('⇒', ['\\Rightarrow', '\\implies']),
    ('⇔', ['\\Leftrightarrow', '\\iff']), ('≤', ['\\le', '\\leq']), ('≥', ['\\ge', '\\geq']),
    ('≠', ['\\ne', '\\neq']), ('∞', ['\\infty']), ('∂', ['\\partial']), ('∇', ['\\nabla']),
    ('∑', ['\\sum']), ('∏', ['\\prod']), ('∫', ['\\int']), ('√', ['\\sqrt']),
    ('×', ['\\times']), ('÷', ['\\div']),
)


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = _safe(lambda: ops.normalize(text), text) or text

        spans = _safe(lambda: ops.extract_math_spans(t), [])
        n_spans = len(spans)

        penalty = 0.0
        bonus = 0.0

        # 1. raw ASCII ellipsis ("...." / "...") vs proper LaTeX ellipsis macros
        bad_dots = len(_ASCII_ELLIPSIS_RE.findall(t))
        proper_ellipsis = sum(t.count(c) for c in _PROPER_ELLIPSIS_CMDS) + t.count(_UNICODE_ELLIPSIS)
        if bad_dots > 0:
            penalty += min(0.8, 0.55 + 0.15 * (bad_dots - 1))
        elif proper_ellipsis > 0:
            bonus += min(0.08, 0.03 * proper_ellipsis)

        # 2. thin-space macro (\, \; \: \!) misuse: padding the boundary of every
        #    inline span, or dense repetition within a span -- overuse the naive
        #    baseline rewards but that actually clutters inline math in prose.
        inline_spans = [c for (k, c) in spans if k == 'inline']
        if inline_spans:
            padded = 0
            dense = 0
            for c in inline_spans:
                s = c.strip()
                starts = any(s.startswith(cmd) for cmd in _THIN_CMDS) or s.startswith('\\ ')
                ends = any(s.endswith(cmd) for cmd in _THIN_CMDS) or s.endswith('\\ ')
                if starts or ends:
                    padded += 1
                thin_ct = sum(c.count(cmd) for cmd in _THIN_CMDS)
                toks = max(1, len(c.split()))
                if thin_ct >= 2 and thin_ct / toks > 0.3:
                    dense += 1
            pad_frac = padded / len(inline_spans)
            dense_frac = dense / len(inline_spans)
            penalty += min(0.35, 0.55 * pad_frac)
            penalty += min(0.25, 0.4 * dense_frac)

        # 3. backslash-space filler spam: literal "\ " used repeatedly as a
        #    manual spacer around every symbol (distinct from occasional
        #    legitimate use, e.g. "\, dx").
        bs_space_ct = len(_BACKSLASH_SPACE_RE.findall(t))
        if n_spans > 0 and (bs_space_ct / max(1, n_spans)) > 1.0:
            penalty += min(0.35, 0.05 * bs_space_ct)

        # 4. raw unicode math operators substituting for LaTeX macros, and the
        #    specific inconsistency case of the SAME operation typeset two
        #    different ways in one document (unicode glyph + its \macro twin).
        census = _safe(lambda: ops.notation_census(t), {}) or {}
        unicode_op_ct = 0
        for k, v in census.items():
            try:
                if isinstance(k, str) and k != _UNICODE_ELLIPSIS and any(ord(ch) > 127 for ch in k):
                    unicode_op_ct += v
            except Exception:
                continue
        if unicode_op_ct > 0:
            penalty += min(0.4, 0.12 * unicode_op_ct)

        pair_hits = 0
        for uni_sym, macro in _UNI_MACRO_PAIRS:
            if census.get(uni_sym, 0) > 0 and any(census.get(m, 0) > 0 for m in macro):
                pair_hits += 1
        if pair_hits > 0:
            penalty += min(0.3, 0.15 * pair_hits)

        # 4b. boundary punctuation glued to (or stranded by) closing "$" -- a
        #     pervasive scraped-MSE artifact directly matching this criterion.
        space_before_punct = len(_SPACE_BEFORE_PUNCT_RE.findall(t))
        glued_after = len(_GLUED_AFTER_RE.findall(t))
        boundary_ct = space_before_punct + glued_after
        if boundary_ct > 0:
            penalty += min(0.5, 0.06 * boundary_ct)
        # whitespace stuck just inside INLINE delimiters ("$ x $" mid-sentence)
        # -- display blocks ("$$ ... $$") legitimately pad with inner spaces,
        # so only inline spans count here.
        padded_delims = sum(1 for c in inline_spans if c != c.strip() and c.strip() != "")
        if inline_spans:
            pd_frac = padded_delims / len(inline_spans)
            penalty += min(0.2, 0.3 * pd_frac)

        # 5. structural delimiter health (parity/brace/env/left-right mismatches,
        #    naked commands, unseparated adjacent formulas) -- sum of whatever
        #    int-valued counters the op reports, defect-agnostic to key names.
        dh = _safe(lambda: ops.delimiter_health(t), {}) or {}
        dh_total = 0
        try:
            dh_total = sum(v for v in dh.values() if isinstance(v, (int, float)))
        except Exception:
            dh_total = 0
        if dh_total > 0:
            penalty += min(0.45, 0.1 * dh_total)

        # 6. un-narrated "formula wall": consecutive display blocks stacked with
        #    ~0 connecting prose -- a direct readability/line-break failure
        #    (reader has no guiding text between derivation steps).
        disp_blocks = list(_DISPLAY_BLOCK_RE.finditer(t))
        wall_gaps = 0
        for a, b in zip(disp_blocks, disp_blocks[1:]):
            gap = t[a.end():b.start()]
            if len(gap.split()) <= 2:
                wall_gaps += 1
        if wall_gaps > 0:
            penalty += min(0.3, 0.12 * wall_gaps)

        # 7. dense math = more surface area for residual micro-typographic
        #    friction; only amplifies an already-nonzero penalty.
        if n_spans >= 8 and penalty > 0:
            penalty *= 1.15

        # 8. LLM tacit read: a concretely pointed-to awkward spot beyond what
        #    the code-side predicates enumerate.
        note = ""
        if isinstance(extracted, dict):
            note = (extracted.get("broken_spacing_example") or "").strip()
        if note and note.upper() not in ("NONE", "N/A", "NA"):
            penalty += 0.12

        s = 1.0 - penalty + bonus
        if s < 0.0:
            s = 0.0
        if s > 1.0:
            s = 1.0
        return s
    except Exception:
        return 0.5
