"""a102 Document preparation and labeling -- LaTeX-aware hybrid metric.

Rationale: the naive regex baseline rewards mere PRESENCE of \\label{/\\ref{/
\\begin{equation,align}/\\usepackage/\\documentclass/\\section. That both
over-credits isolated, non-functional instances (one \\begin{align} block with
no labels/refs at all still gets partial credit) and completely misses the
idiom Math StackExchange answers actually use for "clear numbering so
cross-references are reliable under revision": \\tag{N} on a display equation,
referenced later either via \\eqref{N}/\\ref{N} or by a bare "(N)" mention in
prose. We detect that functional definition-then-backreference pattern
directly, reward its EXTENT and GENUINENESS (not raw keyword count), add a
weaker secondary signal for general display-math typesetting effort and
informal proof-structure organization (cases/steps -- the MSE-answer stand-in
for "sections/subparts"), and penalize broken LaTeX delimiter health. Answers
with no math typesetting at all cannot satisfy this criterion and are floored
at 0.
"""
import re

LLM_FIELDS = {}


def _safe(fn, default):
    try:
        val = fn()
        return val if val is not None else default
    except Exception:
        return default


_TAG_BRACE_RE = re.compile(r'\\tag\{([^{}]{1,24})\}')
_TAG_BARE_RE = re.compile(r'\\tag(\d+)')
_LABEL_RE = re.compile(r'\\label\{([^{}]{1,48})\}')
_TOKEN_OK_RE = re.compile(r'^[A-Za-z0-9.\-* ]{1,16}$')


def _genuine_backrefs(t: str):
    """Find \\tag{}/\\label{} definitions and check whether each is ever
    referenced again later in the doc, either via a formal \\eqref/\\ref/
    \\autoref/\\cref macro or a bare "(token)" mention in prose. Returns
    (n_distinct_definitions, n_definitions_with_a_genuine_backreference).
    """
    defs = []
    for m in _TAG_BRACE_RE.finditer(t):
        defs.append((m.group(1).strip(), m.end()))
    for m in _TAG_BARE_RE.finditer(t):
        defs.append((m.group(1).strip(), m.end()))
    for m in _LABEL_RE.finditer(t):
        defs.append((m.group(1).strip(), m.end()))

    seen = set()
    n_backref = 0
    for tok, pos in defs:
        if not tok or tok in seen:
            continue
        seen.add(tok)
        esc = re.escape(tok)
        tail = t[pos:]
        formal = re.search(r'\\(?:eqref|ref|autoref|cref)\{' + esc + r'\}', tail)
        prose = None
        if _TOKEN_OK_RE.match(tok):
            prose = re.search(r'\(\s*' + esc + r'\s*\)', tail)
        if formal or prose:
            n_backref += 1
    return len(seen), n_backref


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = text or ""
        if len(t.strip()) < 30:
            return 0.0

        t = _safe(lambda: ops.normalize(t), t) or t

        n_display, n_inline, n_numbered, _avg_tok = _safe(
            lambda: ops.equation_stats(t), (0, 0, 0, 0.0)
        )
        n_display = n_display or 0
        n_inline = n_inline or 0
        if n_display == 0 and n_inline == 0:
            return 0.0  # no math typesetting at all: can't satisfy this criterion

        n_defs, n_backref = _safe(lambda: _genuine_backrefs(t), (0, 0))
        extent = min(1.0, (n_numbered or 0) / 6.0)
        backref_frac = (n_backref / n_defs) if n_defs else 0.0
        numbering_component = 0.6 * extent + 0.4 * backref_frac

        display_component = min(1.0, n_display / 6.0)

        skeleton = _safe(lambda: ops.proof_skeleton(t), {}) or {}

        def _n(cat):
            try:
                return len(skeleton.get(cat, []) or [])
            except Exception:
                return 0

        n_struct = _n('case') + _n('theorem') + _n('definition') + _n('proof') + _n('qed')
        n_conn = _n('connective')
        structure_component = min(1.0, n_struct / 6.0 + 0.2 * min(1.0, n_conn / 6.0))

        health = _safe(lambda: ops.delimiter_health(t), {}) or {}
        bad = 0
        if isinstance(health, dict):
            for v in health.values():
                try:
                    bad += max(0, int(v))
                except Exception:
                    pass
        penalty = min(0.35, 0.06 * bad)

        raw = (0.50 * numbering_component
               + 0.35 * display_component
               + 0.15 * structure_component)
        raw -= penalty

        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
