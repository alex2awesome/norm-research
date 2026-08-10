"""Hybrid metric channel for a216: Equation numbering and LaTeX referencing.

Judge rewards genuine, consistently-applied LaTeX cross-referencing machinery
(\\label placed inside a display/equation environment, later pulled by
\\eqref/\\ref/\\autoref/\\cref with a matching key) -- NOT mere keyword presence.
Bare \\tag usage (no \\label/\\ref link) earns much smaller, secondary credit,
and only when the tag content looks like a real equation number/short key
(digits, "2a", "*") rather than a prose caption ("Eq. 2a", "Equation 4") stuffed
into \\tag for cosmetic purposes. Consistency (fraction of display equations
that carry the numbering/labeling) matters as much as bare presence: a single
isolated \\label+\\eqref pair buried among many unlabeled display equations is
worth much less than the same pair when it is the only display equation, or
when most/all display equations are uniformly labeled.

No LLM_FIELDS declared: the predicate here is fully mechanical (existence and
placement of \\label/\\eqref/\\ref/\\tag keys, and their internal consistency),
so a code-only parser resolves it without any tacit judgment an LLM would need
to supply.
"""

import re

LLM_FIELDS = {}  # predicate is fully checkable from LaTeX structure; no LLM grounding needed

_LABEL_RE = re.compile(r'\\label\s*\{([^}]*)\}')
_REF_RE = re.compile(r'\\(?:eqref|ref|autoref|cref)\s*\{([^}]*)\}')
# \tag{...} (braced, arbitrary content) OR bare \tag1 / \tag* / \tag2a (broken/no-brace form)
_TAG_RE = re.compile(r'\\tag\s*\{([^}]*)\}|\\tag\s*(\*|\d+[a-zA-Z]?)\b')
_SUBEQ_RE = re.compile(r'\\begin\{subequations\}|\\numberwithin\s*\{equation\}|\\setcounter\s*\{equation\}')
_SHORT_TAG_RE = re.compile(r'^\s*\*?\s*\d{0,3}[a-zA-Z]?\s*\*?\s*$')
_WORDY_TAG_RE = re.compile(r'[A-Za-z]{2,}')

_FLOOR = 0.15
_MATCH_SLOPE = 0.45
_TAG_SLOPE = 0.20
_SUBEQ_BONUS = 0.03
_PENALTY_CAP = 0.10


def _safe_call(fn, default):
    try:
        return fn()
    except Exception:
        return default


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = _safe_call(lambda: ops.normalize(text), text) or text

        # --- LaTeX-aware span extraction (not raw keyword regex over the whole doc) ---
        spans = _safe_call(lambda: ops.extract_math_spans(t), [])
        if not isinstance(spans, list):
            spans = []
        display_contents = [c for item in spans
                             if isinstance(item, (tuple, list)) and len(item) >= 2
                             and item[0] == 'display' for c in [item[1]]]
        n_display_spans = len(display_contents)

        es = _safe_call(lambda: ops.equation_stats(t), None)
        es_n_display = 0
        if isinstance(es, (tuple, list)) and len(es) >= 1:
            try:
                es_n_display = int(es[0])
            except Exception:
                es_n_display = 0

        eff_n_display = max(n_display_spans, es_n_display, 1)

        # --- \label placed INSIDE a display span (correct placement per contract) ---
        label_keys = set()
        n_labeled_display = 0
        tag_contents = []
        n_tagged_display = 0
        for c in display_contents:
            labs = _LABEL_RE.findall(c)
            if labs:
                n_labeled_display += 1
                label_keys.update(s.strip() for s in labs if s.strip())
            tag_matches = _TAG_RE.findall(c)
            this_tags = [a if a else b for (a, b) in tag_matches] if tag_matches else []
            if this_tags:
                n_tagged_display += 1
                tag_contents.extend(x.strip() for x in this_tags if x.strip())

        # --- references live in prose, so search the whole normalized text ---
        ref_keys = set(s.strip() for s in _REF_RE.findall(t) if s.strip())
        matched_keys = label_keys & ref_keys

        has_subeq_machinery = bool(_SUBEQ_RE.search(t))

        # --- mild structural-quality penalty (broken delimiters undercut "required number placement") ---
        dh = _safe_call(lambda: ops.delimiter_health(t), {})
        issues = 0
        if isinstance(dh, dict):
            for v in dh.values():
                try:
                    issues += max(0, int(v))
                except Exception:
                    pass
        penalty = min(_PENALTY_CAP, 0.015 * issues)

        if matched_keys:
            coverage = min(1.0, n_labeled_display / eff_n_display)
            s = _FLOOR + _MATCH_SLOPE * coverage
        elif n_tagged_display > 0:
            coverage = min(1.0, n_tagged_display / eff_n_display)
            short_like = [c for c in tag_contents if _SHORT_TAG_RE.match(c)]
            wordy = [c for c in tag_contents if _WORDY_TAG_RE.search(c)]
            genuine_frac = len(short_like) / max(len(tag_contents), 1)

            nums = []
            for c in tag_contents:
                m = re.match(r'^(\d+)', c)
                if m:
                    nums.append(int(m.group(1)))
            seq_bonus = 0.0
            if len(nums) >= 2:
                uniq = sorted(set(nums))
                span = uniq[-1] - uniq[0] + 1
                completeness = len(uniq) / max(span, 1)
                seq_bonus = 0.03 * completeness

            wordy_penalty = 0.05 * (len(wordy) / max(len(tag_contents), 1))
            # any \tag machinery at all (even prose-caption abuse of it) is still weak positive
            # evidence of *attempted* numbering -- never rate it below the no-evidence floor.
            s = _FLOOR + max(0.0, _TAG_SLOPE * coverage * genuine_frac + seq_bonus - wordy_penalty)
        else:
            s = _FLOOR

        if has_subeq_machinery:
            s += _SUBEQ_BONUS

        s -= penalty
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
