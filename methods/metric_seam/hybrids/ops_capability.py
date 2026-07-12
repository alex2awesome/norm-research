"""E2L capability-ops library — real compiled instruments for hybrid metric channels
(metric-seam agentic program, WS1 E2L amendment, see
notes/2026-07-10__seam-agentic-program-runbook.md "E2L — LIBRARY-AUGMENTED E2").

Motivating design note (user-directed, pre-registered before any E2L data): the 9
function-wall E2 kills were all REGEX-genre code — a bare-string-match or window-search
implementation of a construct that actually requires structure (a dependency parse, a
symbolic entailment check, a distributional recompute). This module gives E2L crews a
sanctioned, label-free capability library so a candidate can EXECUTE and VERIFY instead
of pattern-match. Every op here targets a SPECIFIC decisive_reason on record in
outputs/metric_seam_pilot/battery/effort_ladder/e2/*/meta.json:

  press_releases/a31 (ATTRIBUTION): the rejected candidate's `_ATTRIB_RE` scanned a
    3-sentence window (prev/sent/next) for a bare reporting-verb keyword ("believes"),
    so an unrelated "believes" 1-2 sentences away falsely marked a genuine self-voiced
    overclaim as third-party-attributed (train d00964, judge=0.00: h0 0.000 -> cand
    0.303). Its `_EXEC_TITLE_RE` was a closed vocabulary of executive titles (CEO/CFO/
    ...) that missed "principal scientist at GE Research" as company-self (d01806).
    `attributions()` below ties a claim span to the EXACT syntactic complement (ccomp/
    xcomp) of the reporting verb that governs it — never a window — and
    `speaker_is_first_person_org` is decided from the document's own most-frequent ORG
    entity (+ acronym/substring alias), not a fixed title list.
  math/a150, math/a30 (VACUITY): both rejected candidates cleared a "logical scaffolding"
    gate with REPEATED SURFACE MARKERS (2x "note that", a bare "since"-clause, a
    citation-name stack) that carried zero checkable content — the gate/signal
    double-counted the same surface tokens it was meant to corroborate, and a150's
    vacuity guard was a 25-character LENGTH check, not a content check. `parse_math` +
    `licensing_does_work` below refuse to be fooled by surface form: they either
    symbolically verify that a consequent follows from a premise (checkable subset:
    equation rescaling, same-LHS value comparison, plain algebraic/numeric identity) or
    honestly report checkable=False. A content-free "note that ... since ..." string
    that never parses as math never gets checkable=True.
  humor/a117, humor/a315 (REPETITION-FUNCTION / laundering): a117's near-dup detector
    penalized rule-of-three escalation because Jaccard treats a genuine callback and
    copy-paste padding identically (both are "high sentence overlap"); a315's AND-NOT
    laundering gate keyed a closed deflation vocabulary that any paraphrase evaded.
    `is_refrain()` below distinguishes the two by measuring NOVELTY BETWEEN repeat
    occurrences (does new material actually happen between the callbacks, i.e. does the
    text progress) rather than by overlap magnitude or a fixed lexicon, and
    `discourse_position()` locates a span structurally (opening/middle/coda) instead of
    by a fixed trailing-percentage cut (the a117 "tail-excision blind-deletes a
    punchline paragraph" failure).

Scope discipline (applies to every op): label-free (never sees judge scores or gold),
deterministic (no sampling / no temperature), corpus-loaded conventions match ops.py /
ops_math.py (Ops/MathOps: single-document Z=f(X) computation ops), and every op degrades
gracefully — a missing optional dependency (spaCy model, scipy, ...) makes an op return
its documented "unavailable" value (empty list / None / checkable=False), it NEVER
raises out of a candidate's score() call. Honest scoping over guessing: `checkable`
flags exist so a candidate can WEIGHT confidence rather than being fed a silent wrong
answer (math/a150's own lesson: "a corroboration gate must be independent of the signal
it gates").

VERSION = 'e2l-v1' (see CAPABILITIES dict below). This library is FROZEN per E2L wave —
crews may not edit it mid-cell (toolsmith evolution of shared libraries is WS2's
separate question); bump VERSION on any behavior change.

---------------------------------------------------------------------------------------
USAGE (the E2L wiring convention, confirmed under battery/agentic_run.py sandboxing):

    # inside a candidate .py file (methods/metric_seam/hybrids/programs_*/<aid>_XX.py)
    from ops_capability import CapabilityOps
    _cap = CapabilityOps()          # free: shares the module-level spaCy singleton

    LLM_FIELDS = {...}              # unchanged, optional

    def score(text, extracted, ops):
        # `ops` is still the task's Ops/MathOps instance (normalize/extract_dates/...);
        # `_cap` is the E2L capability layer, imported once at module scope.
        attrs = _cap.attributions(text)
        self_voiced = _cap.self_attributed(text, some_span)
        ...
        return value_in_0_1

This works because battery_common.py (imported by every eval/iteration entry point —
agentic_run.py, eval_hybrids_task.py) inserts methods/metric_seam/hybrids onto sys.path
BEFORE any candidate module is loaded via importlib (battery_common.load_mod /
eval_hybrids_task.load_mod), so `from ops_capability import CapabilityOps` resolves
exactly like the existing `from ops import Ops` pattern in *_h0.py baselines — no new
sandboxing hole, no path hacking inside the candidate. CapabilityOps itself is stateless
(every method is a staticmethod delegating to a module-level function): instantiating it
per-candidate-module is free, and multiple candidates in the same process share the one
lazily-loaded spaCy pipeline (see _get_nlp).
---------------------------------------------------------------------------------------

Import-time budget: this module imports only `re`, `collections`, `functools`, `math` at
top level. spaCy, sympy, scipy, networkx, and dateutil are ALL lazy-imported inside the
functions that need them (measured: bare `import ops_capability` <0.05s; see
test_ops_capability.py for the enforced <5s smoke check including first spaCy load).
"""
import functools
import math
import re
from collections import Counter

VERSION = "e2l-v1"

CAPABILITIES = {
    "attributions":          {"group": "attribution", "requires": "spacy",
                               "degrades_to": "[]"},
    "self_attributed":       {"group": "attribution", "requires": "spacy",
                               "degrades_to": "None (unknown, not a guess)"},
    "parse_math":            {"group": "math", "requires": "sympy",
                               "degrades_to": "None"},
    "licensing_does_work":   {"group": "math", "requires": "sympy",
                               "degrades_to": "{'checkable': False, 'follows': None}"},
    "restates_definition":   {"group": "math", "requires": "sympy",
                               "degrades_to": "{'checkable': False, 'match_index': None}"},
    "stat_consistency":      {"group": "consistency", "requires": "scipy (or sympy.stats)",
                               "degrades_to": "rows with checkable=False"},
    "number_consistency":    {"group": "consistency", "requires": "stdlib only",
                               "degrades_to": "n/a (pure regex + arithmetic)"},
    "date_chain":            {"group": "dates", "requires": "dateutil",
                               "degrades_to": "[]"},
    "deadline_satisfied":    {"group": "dates", "requires": "dateutil",
                               "degrades_to": "None"},
    "sentence_graph":        {"group": "discourse", "requires": "networkx (+spacy optional)",
                               "degrades_to": "empty graph, or None if networkx missing"},
    "is_refrain":            {"group": "discourse", "requires": "spacy optional",
                               "degrades_to": "[] (regex sentence split fallback)"},
    "discourse_position":    {"group": "discourse", "requires": "spacy optional",
                               "degrades_to": "None (regex sentence split fallback)"},
    "fact_density":          {"group": "ner", "requires": "spacy",
                               "degrades_to": "{'counts': {}, 'total': 0, 'per_1000_words': 0.0}"},
    "entities_with_evidence": {"group": "ner", "requires": "spacy",
                               "degrades_to": "[]"},
}

# ===========================================================================
# shared lazy singletons / caps
# ===========================================================================
_SPACY_CHAR_CAP = 8000     # matches ops.py's corpus-loader truncation convention
_MATH_STR_CAP = 400        # a "checkable" span is a short algebraic snippet, not a essay
_STAT_TEXT_CAP = 20000
_PARSE_CACHE_SIZE = 16     # small: one document is usually re-used a handful of times
                           # within a single score() call (attributions + self_attributed
                           # + sentence_graph on the SAME text), not across the corpus.

_nlp = None
_nlp_tried = False


def _get_nlp():
    """Module-level spaCy singleton, loaded once on first use (lazy). Returns None
    (never raises) if spacy or en_core_web_sm is unavailable -- every op that depends
    on this degrades to its documented fallback rather than crashing the harness."""
    global _nlp, _nlp_tried
    if _nlp_tried:
        return _nlp
    _nlp_tried = True
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        _nlp.max_length = 2_000_000
    except Exception:
        _nlp = None
    return _nlp


@functools.lru_cache(maxsize=_PARSE_CACHE_SIZE)
def _parse(text_capped):
    """Cached spaCy Doc for a (already-capped) text string. None if spaCy unavailable
    or parsing failed. Cache is keyed on the exact string so repeated ops on the SAME
    document within one score() call reuse the parse; it is NOT a corpus-wide cache."""
    nlp = _get_nlp()
    if nlp is None:
        return None
    try:
        return nlp(text_capped)
    except Exception:
        return None


def _doc_for(text):
    if not text or not isinstance(text, str) or not text.strip():
        return None
    return _parse(text[:_SPACY_CHAR_CAP])


def _safe(default):
    """Never-raise guard (same convention as ops_math.py's _safe): on any exception,
    return default() if callable else default. Ops feed unvetted corpus text; a hybrid
    candidate must never crash the harness because of a library edge case."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception:
                return default() if callable(default) else default
        return wrapped
    return deco


# ===========================================================================
# 1. ATTRIBUTION  (press_releases/a31 axis)
# ===========================================================================
_REPORTING_VERBS = {
    "say", "state", "note", "claim", "argue", "tell", "announce", "add",
    "explain", "report", "disclose", "reveal", "confirm", "deny", "insist",
    "believe", "think", "suggest", "indicate", "assert", "declare", "remark",
    "opine", "write", "tweet", "post", "respond", "reply", "warn", "predict",
    "acknowledge", "maintain", "observe", "conclude",
}
_FIRST_PERSON_WORDS = {"i", "we", "us", "our", "ours", "myself", "ourselves"}
_SELF_PHRASE_RE = re.compile(
    r"\b(the company('s)?|our own|the firm('s)?|the organization('s)?)\b", re.I)
_ORG_SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|corp|corporation|llc|ltd|co|company|group|holdings|plc)\.?\b",
    re.I)


def _norm_org(s):
    s = (s or "").strip().lower()
    s = re.sub(r"^\s*the\s+", "", s)
    s = re.sub(r"[’']s\b", "", s)
    s = _ORG_SUFFIX_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _home_orgs(doc):
    """The document's own organization(s): the most-frequent ORG entities, plus the
    earliest-occurring one (headline/dateline convention). Frozenset of normalized
    strings; short (<=4 char) aliases are only admitted if the ORIGINAL entity text was
    upper-cased (an acronym like "GE"/"FAIR"), to avoid a short common word colliding by
    accident."""
    counts, first_pos, raw_upper = Counter(), {}, {}
    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        norm = _norm_org(ent.text)
        if len(norm) < 2:
            continue
        if len(norm) <= 4 and not ent.text.isupper():
            continue  # short alias only trusted if it was written as an acronym
        counts[norm] += 1
        first_pos.setdefault(norm, ent.start_char)
    if not counts:
        return frozenset()
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], first_pos[kv[0]]))
    top_n = max(1, min(3, len(ranked)))
    home = {ranked[i][0] for i in range(top_n)}
    earliest = min(counts, key=lambda k: first_pos[k])
    home.add(earliest)
    return frozenset(home)


def _speaker_is_self(speaker_span, home_orgs):
    low = " " + re.sub(r"[^a-z' ]", " ", speaker_span.lower()) + " "
    first_words = re.findall(r"[a-z']+", low)[:2]
    if any(w in _FIRST_PERSON_WORDS for w in first_words):
        return True
    if _SELF_PHRASE_RE.search(speaker_span):
        return True
    norm_speaker = _norm_org(speaker_span)
    for h in home_orgs:
        if h and re.search(r"\b" + re.escape(h) + r"\b", norm_speaker):
            return True
    return False


def _subtree_bounds(token):
    idxs = [tk.i for tk in token.subtree]
    return min(idxs), max(idxs)


@_safe(list)
def attributions(text):
    """All reporting-verb attributions in the document, via spaCy dependency parse.

    For every VERB whose lemma is a reporting verb (said/claims/believes/according-to/
    ...) AND that has both a subject (nsubj/nsubjpass/csubj) and a genuine propositional
    complement (ccomp/xcomp) -- i.e. it actually governs a claim, not just an object of
    belief ("believes in borders" has no ccomp and is correctly SKIPPED, the exact a31
    kill-case axis: a bare reporting verb with no governed clause must not attribute
    anything, in this sentence or any other) -- returns one record:

        {"verb": lemma, "verb_text": surface form,
         "speaker_span": text of the subject's full subtree (name + appositive title,
                          e.g. "Radislav Potyrailo, a principal scientist at GE
                          Research"),
         "span": text of the governed clause (the actual claim),
         "span_start": char offset (start), "span_end": char offset (end) IN THE
                          CAPPED/parsed text -- these are the offsets attributions()
                          used internally; self_attributed() re-derives its own so a
                          span found by string search still works,
         "sentence": the full sentence text (context only, not used for scoping --
                          the "span" bounds are what get compared, never the sentence),
         "speaker_is_first_person_org": bool, decided from the document's own most-
                          frequent ORG entity (see _home_orgs), not a closed title list,
         "quote_like": bool, True if the sentence contains quotation marks}

    Also detects the "According to X, <clause>" construction (a distinct dependency
    shape: "According" is a `prep` child of the main clause's root verb, not a verb
    itself) as a second attribution pattern.

    Every attribution is scoped to the EXACT clause a verb governs, never a sentence
    window -- this is the direct fix for the a31 decisive_reason ("_ATTRIB_RE
    false-match on bare believes leaks third-party discount onto self-voiced
    overclaims"): a reporting verb elsewhere in the document, or even in the same
    sentence but with no clausal complement, contributes nothing here.

    Degrades to [] if spaCy/en_core_web_sm is unavailable or the doc is empty.
    """
    doc = _doc_for(text)
    if doc is None:
        return []
    home = _home_orgs(doc)
    out = []
    for tok in doc:
        if tok.pos_ != "VERB" or tok.lemma_.lower() not in _REPORTING_VERBS:
            continue
        subj = next((c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass", "csubj")),
                    None)
        comp = next((c for c in tok.children if c.dep_ in ("ccomp", "xcomp")), None)
        if subj is None or comp is None:
            continue
        s0, s1 = _subtree_bounds(subj)
        c0, c1 = _subtree_bounds(comp)
        speaker_span = doc[s0:s1 + 1].text
        sent = tok.sent
        out.append({
            "verb": tok.lemma_.lower(),
            "verb_text": tok.text,
            "speaker_span": speaker_span,
            "span": doc[c0:c1 + 1].text,
            "span_start": doc[c0].idx,
            "span_end": doc[c1].idx + len(doc[c1].text),
            "sentence": sent.text,
            "speaker_is_first_person_org": _speaker_is_self(speaker_span, home),
            "quote_like": ('"' in sent.text or "“" in sent.text
                            or "‘" in sent.text),
        })
    for sent in doc.sents:
        toks = list(sent)
        if len(toks) < 4:
            continue
        if toks[0].text.lower() != "according" or toks[1].text.lower() != "to":
            continue
        to_tok = toks[1]
        pobj = next((c for c in to_tok.children if c.dep_ == "pobj"), None)
        if pobj is None:
            continue
        s0, s1 = _subtree_bounds(pobj)
        rest_start = s1 + 1
        while rest_start <= toks[-1].i and doc[rest_start].text in (",", ":"):
            rest_start += 1
        if rest_start > toks[-1].i:
            continue
        speaker_span = doc[s0:s1 + 1].text
        c0, c1 = rest_start, toks[-1].i
        out.append({
            "verb": "according_to",
            "verb_text": "according to",
            "speaker_span": speaker_span,
            "span": doc[c0:c1 + 1].text,
            "span_start": doc[c0].idx,
            "span_end": doc[c1].idx + len(doc[c1].text),
            "sentence": sent.text,
            "speaker_is_first_person_org": _speaker_is_self(speaker_span, home),
            "quote_like": False,
        })
    return out


def self_attributed(text, span):
    """Is `span` (a substring of `text`, OR a (start, end) char-offset pair into the
    CAPPED text used internally -- pass a substring unless you already have offsets
    from attributions()) in the document's OWN voice, or is it embedded inside a
    third-party-attributed clause?

    Returns True (self-voiced: not governed by any attribution(), OR governed by one
    whose speaker resolves to the document's own org/first-person -- a company's own
    spokesperson quote still counts as the release's own voice), False (embedded in a
    genuine third-party attribution), or None if this cannot be determined honestly
    (spaCy unavailable, or `span` given as a string that does not appear verbatim in
    `text`) -- None is a real answer here, not a fallback default.

    This is the op that directly targets the a31 kill test: a neighboring sentence that
    merely sits near an unrelated reporting-verb mention (e.g. "...believes in
    borders...") must resolve to True (self-voiced), not be falsely discounted as
    attributed, because attributions() never produced a claim record for that verb (no
    ccomp) in the first place -- there is nothing for this span to overlap.
    """
    if not text:
        return None
    if isinstance(span, (tuple, list)) and len(span) == 2:
        try:
            s0, s1 = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            return None
    elif isinstance(span, str):
        idx = text.find(span)
        if idx == -1:
            return None
        s0, s1 = idx, idx + len(span)
    else:
        return None
    if _get_nlp() is None:
        return None
    attrs = attributions(text)
    for a in attrs:
        a0, a1 = a["span_start"], a["span_end"]
        if a0 < s1 and s0 < a1:  # half-open interval overlap
            return bool(a["speaker_is_first_person_org"])
    return True


# ===========================================================================
# 2. MATH ENTAILMENT  (math/a150, math/a30 vacuity axis)
# ===========================================================================
_DELIM_STRIP_RE = re.compile(
    r"^\s*(?:\${1,2}|\\\(|\\\[)\s*|\s*(?:\${1,2}|\\\)|\\\])\s*$")


def _strip_math_delims(s):
    prev = None
    while prev != s:
        prev = s
        s = _DELIM_STRIP_RE.sub("", s)
    return s.strip()


# sympy's lark-backend LaTeX grammar has NO notion of "not math": with no
# recognized structure it silently degrades to reading every letter as its own
# one-character symbol and every adjacent pair as implicit multiplication --
# "note that the result follows since n exists" parses "successfully" to
# a*c*e**5*f*h**2*i**2*l**3*n**3*o**3*r*s**5*t**6*u*w*x. That is EXACTLY the
# math/a150 vacuity failure mode (syntax fires without semantic work) inside
# our own parser, so it is guarded against explicitly: any multi-letter
# "word" (>=3 alphabetic chars) that is not a recognized math/function/greek
# token makes the snippet PROSE, not math, and parse_math refuses it before
# ever calling into sympy.
_MATH_WORDS = {
    "sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "lim",
    "sup", "inf", "det", "gcd", "lcm", "deg", "dim", "ker", "mod", "arg",
    "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "min", "max",
    "sqrt", "var", "cov", "abs",
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "infty",
}
_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _looks_like_prose(s):
    stripped_cmds = re.sub(r"\\[A-Za-z]+", " ", s)  # drop \frac \cdot \left etc.
    for w in _WORD_RE.findall(stripped_cmds):
        if w.lower() not in _MATH_WORDS:
            return True
    return False


@_safe(lambda: None)
def parse_math(s):
    """Parse a short LaTeX/math snippet into a sympy object, or None (honest failure --
    never guess). Uses sympy's built-in lark-backend LaTeX parser (no antlr4
    dependency). Handles the CHECKABLE subset this library scopes to: arithmetic,
    algebraic equations/rewrites, fractions, roots, basic comparisons, sums with
    literal bounds. Explicitly does NOT attempt semantic-only LaTeX (\\text{...},
    modular-arithmetic notation, custom macros, multi-line derivations) -- those raise
    inside the parser and this returns None rather than a wrong parse.

    Ambiguous parses (sympy's lark grammar can return an internal `_ambig` Tree node
    for genuinely ambiguous surface forms like "f(x)" reading as either a function call
    or implicit multiplication) are also treated as unparseable: the result is only
    returned if it is a resolved sympy object (`isinstance(x, sympy.Basic)`).

    PROSE GUARD: the underlying grammar has no notion of "not math" -- with no
    recognized structure it silently reads every letter as a one-character symbol and
    concatenation as implicit multiplication, so plain English ("note that the result
    follows since n exists") "parses" as a monomial of single-letter variables. This is
    exactly the math/a150 vacuity failure mode reproduced inside the parser itself, so
    it is guarded explicitly: any run of >=3 alphabetic characters that is not a
    recognized math/function/greek token (sin, log, alpha, ...) makes the input PROSE
    and parse_math returns None before ever calling sympy.

    Input is capped at 400 chars -- a "checkable" span is a short symbolic snippet, not
    a multi-line derivation; longer input returns None immediately.
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s or len(s) > _MATH_STR_CAP:
        return None
    s = _strip_math_delims(s)
    if not s or _looks_like_prose(s):
        return None
    import sympy as sp
    from sympy.parsing.latex import parse_latex
    expr = parse_latex(s, backend="lark")
    if not isinstance(expr, sp.Basic):
        return None
    return expr


def _equation_ratio(eq1, eq2):
    """sympy.cancel((lhs2-rhs2)/(lhs1-rhs1)), or None if eq1 is a tautology (0=0, can't
    divide by it) -- shared by licensing_does_work and restates_definition."""
    import sympy as sp
    d1 = sp.simplify(eq1.lhs - eq1.rhs)
    d2 = sp.simplify(eq2.lhs - eq2.rhs)
    if d1 == 0:
        return None
    if d2 == 0:
        return sp.Integer(0)
    return sp.cancel(d2 / d1)


@_safe(lambda: {"checkable": False, "follows": None, "detail": "internal error"})
def licensing_does_work(premise_math, consequent_math):
    """Does `consequent_math` actually follow from `premise_math` -- for the CHECKABLE
    subset only: {"checkable": bool, "follows": bool|None, "detail": str}.

    Checkable cases (each independently verified, not string/keyword matched):
      1. Both are boolean literals (sympy auto-evaluates e.g. "3*4=12" to a bare
         BooleanTrue/False at parse time) -- direct logical check: a false premise
         licenses anything (vacuous truth, flagged as such in `detail`); a true premise
         licenses only a true consequent.
      2. Both are equations (sympy.Eq) -- follows iff consequent's (lhs-rhs) is the
         premise's (lhs-rhs) scaled by a nonzero CONSTANT (no free symbols in the
         ratio): the same constraint, just algebraically rearranged/rescaled. A
         consequent whose ratio to the premise still depends on a free variable is NOT
         a rescaling of the same equation -- checkable=True, follows=False (it is an
         independent, unjustified claim, exactly the math/a150-and-a30 pattern: real
         parseable syntax that performs no actual licensing work).
      3. Both are equations sharing the identical LHS symbol/expression ("x = ...") --
         follows iff the two claimed RHS values are symbolically equal.
      4. Neither is an equation (plain expressions/numbers) -- follows iff their
         difference simplifies to exactly 0 (arithmetic/algebraic identity).

    Anything outside these four (e.g. one operand None/unparseable, mismatched
    equation/expression types not covered above, or a sympy error mid-check) returns
    checkable=False, follows=None. This is deliberate: general symbolic entailment is
    undecidable, and a wrong "follows=True" is worse than an honest "can't check" --
    the exact lesson on record from math/a150 (a corroboration gate must be independent
    of, and stricter than, the surface signal it gates).
    """
    import sympy as sp
    from sympy.logic.boolalg import BooleanAtom
    if premise_math is None or consequent_math is None:
        return {"checkable": False, "follows": None, "detail": "unparseable operand"}

    if isinstance(premise_math, BooleanAtom) or isinstance(consequent_math, BooleanAtom):
        if not (isinstance(premise_math, BooleanAtom) and isinstance(consequent_math, BooleanAtom)):
            return {"checkable": False, "follows": None,
                    "detail": "one operand is a boolean literal, the other is not"}
        p_val, c_val = bool(premise_math), bool(consequent_math)
        follows = (not p_val) or c_val
        detail = ("premise is False: vacuously licenses anything" if not p_val
                  else f"premise True, consequent {c_val}")
        return {"checkable": True, "follows": follows, "detail": detail}

    if isinstance(premise_math, sp.Equality) and isinstance(consequent_math, sp.Equality):
        if premise_math.lhs == consequent_math.lhs:
            same = sp.simplify(premise_math.rhs - consequent_math.rhs) == 0
            return {"checkable": True, "follows": bool(same),
                    "detail": "same-LHS value comparison"}
        ratio = _equation_ratio(premise_math, consequent_math)
        if ratio is None:
            return {"checkable": False, "follows": None,
                    "detail": "premise is a tautology (0=0), cannot license"}
        if not ratio.free_symbols:
            return {"checkable": True, "follows": bool(ratio != 0),
                    "detail": f"consequent = premise scaled by {ratio}"}
        return {"checkable": True, "follows": False,
                "detail": f"consequent/premise ratio depends on {ratio.free_symbols} "
                          f"-- not a rescaling of the same equation"}

    if not isinstance(premise_math, sp.Equality) and not isinstance(consequent_math, sp.Equality):
        diff = sp.simplify(premise_math - consequent_math)
        return {"checkable": True, "follows": bool(diff == 0),
                "detail": "expression identity"}

    return {"checkable": False, "follows": None,
            "detail": "mismatched equation/expression types"}


def _symbolic_equal(a, b):
    import sympy as sp
    from sympy.logic.boolalg import BooleanAtom
    if isinstance(a, BooleanAtom) or isinstance(b, BooleanAtom):
        return (isinstance(a, BooleanAtom) and isinstance(b, BooleanAtom)
                and bool(a) == bool(b))
    if isinstance(a, sp.Equality) and isinstance(b, sp.Equality):
        ratio = _equation_ratio(a, b)
        if ratio is None:
            return sp.simplify(b.lhs - b.rhs) == 0
        return not ratio.free_symbols and ratio != 0
    if isinstance(a, sp.Equality) != isinstance(b, sp.Equality):
        return False
    return sp.simplify(a - b) == 0


@_safe(lambda: {"checkable": False, "match_index": None, "detail": "internal error"})
def restates_definition(clause, defn_candidates):
    """Does `clause` (raw text/LaTeX snippet) symbolically restate one of
    `defn_candidates` (list of raw text/LaTeX snippets)? SYMBOLIC equivalence via
    parse_math + algebraic-rescaling check (see _symbolic_equal), never a string/
    lexical comparison -- that is the whole point of this op (a candidate reaching for
    string similarity here is exactly the mechanism this library exists to replace).

    Returns {"checkable": bool, "match_index": int|None, "detail": str}.
      - clause itself unparseable -> checkable=False (can't even start).
      - clause parses but NO candidate parses -> checkable=False (nothing to compare
        against; do not fall back to string matching).
      - clause parses, >=1 candidate parses -> checkable=True, match_index = the first
        symbolically-equal candidate's index, or None if none match.
    """
    c_expr = parse_math(clause)
    if c_expr is None:
        return {"checkable": False, "match_index": None, "detail": "clause unparseable"}
    any_parsed = False
    for i, cand in enumerate(defn_candidates or []):
        d_expr = parse_math(cand)
        if d_expr is None:
            continue
        any_parsed = True
        if _symbolic_equal(c_expr, d_expr):
            return {"checkable": True, "match_index": i,
                    "detail": f"symbolically equal to candidate {i}"}
    if not any_parsed:
        return {"checkable": False, "match_index": None,
                "detail": "no candidate parseable"}
    return {"checkable": True, "match_index": None,
            "detail": "clause parseable but matches no candidate"}


# ===========================================================================
# 3. CONSISTENCY RECOMPUTATION  (peer_review axis, statcheck-style)
# ===========================================================================
_STAT_PATTERNS = [
    ("t", re.compile(r"\bt\s*\(\s*(?P<df>\d+(?:\.\d+)?)\s*\)\s*=\s*(?P<stat>-?\d+(?:\.\d+)?)")),
    ("f", re.compile(r"\bF\s*\(\s*(?P<df1>\d+)\s*,\s*(?P<df2>\d+)\s*\)\s*=\s*(?P<stat>\d+(?:\.\d+)?)")),
    ("chi2", re.compile(
        r"(?:chi2|chi-square|χ2|χ²|\\chi\^?2|\\chi2)"
        r"\s*(?:\(\s*(?P<df>\d+)\s*\))?\s*=\s*(?P<stat>\d+(?:\.\d+)?)", re.I)),
    ("r", re.compile(r"\br\s*\(\s*(?P<df>\d+)\s*\)\s*=\s*(?P<stat>-?(?:0?\.\d+|\d+(?:\.\d+)?))")),
    ("z", re.compile(r"\bz\s*=\s*(?P<stat>-?\d+(?:\.\d+)?)")),
]
_P_NEAR_RE = re.compile(
    r"[,;\s]*p\s*(?P<cmp><=|>=|=|<|>)\s*(?P<p>0?\.\d+|\d+(?:\.\d+)?)", re.I)
_P_WINDOW = 45


def _recompute_p(test, stat, df=None, df2=None):
    """p-value recomputation for one test statistic. scipy if available (fast, exact
    special functions); else sympy.stats as a dependency-light fallback (slower, still
    exact, no scipy required). Returns None (not a guess) if neither is usable, the
    inputs are out of domain (e.g. |r|>=1), or df is missing where required."""
    try:
        stat = float(stat)
        if df is not None:
            df = float(df)
        if df2 is not None:
            df2 = float(df2)
    except (TypeError, ValueError):
        return None
    try:
        from scipy import stats as sst
        if test == "t" and df:
            return float(2 * sst.t.sf(abs(stat), df))
        if test == "f" and df and df2:
            return float(sst.f.sf(stat, df, df2))
        if test == "chi2" and df:
            return float(sst.chi2.sf(stat, df))
        if test == "z":
            return float(2 * sst.norm.sf(abs(stat)))
        if test == "r" and df and abs(stat) < 1:
            tt = stat * math.sqrt(df / (1 - stat * stat))
            return float(2 * sst.t.sf(abs(tt), df))
        return None
    except ImportError:
        pass
    except Exception:
        return None
    try:
        import sympy as sp
        from sympy.stats import StudentT, F, ChiSquared, Normal, cdf
        if test == "t" and df:
            X = StudentT("X", df)
            return float(2 * (1 - cdf(X)(abs(stat))))
        if test == "f" and df and df2:
            X = F("X", df, df2)
            return float(1 - cdf(X)(stat))
        if test == "chi2" and df:
            X = ChiSquared("X", df)
            return float(1 - cdf(X)(stat))
        if test == "z":
            X = Normal("X", 0, 1)
            return float(2 * (1 - cdf(X)(abs(stat))))
        if test == "r" and df and abs(stat) < 1:
            tt = stat * math.sqrt(df / (1 - stat * stat))
            X = StudentT("X", df)
            return float(2 * (1 - cdf(X)(abs(tt))))
    except Exception:
        return None
    return None


def _p_significant(cmp, val):
    """Best-effort 'was this reported as significant at alpha=.05' reading of a
    reported p-value + comparator; None if the comparator itself is ambiguous (bare
    '=' close to .05 is genuinely ambiguous and left unresolved rather than guessed)."""
    if cmp in ("<", "<="):
        return val <= 0.05 + 1e-9
    if cmp in (">", ">="):
        return False
    if cmp == "=":
        if val < 0.05 - 1e-9:
            return True
        if val > 0.05 + 1e-9:
            return False
        return None
    return None


@_safe(list)
def stat_consistency(text):
    """Extract (test statistic, df, reported p) triples via regex, RECOMPUTE the
    p-value from the statistic + degrees of freedom, and flag inconsistencies --
    statcheck-style. Supports t(df)=, F(df1,df2)=, chi2/χ2(df)=, r(df)=, z=.

    Each match searches a short forward window (45 chars) for a co-occurring p
    comparison (p<.05 / p=.02 / ...); a stat with no nearby p, or a df-requiring test
    with no df captured (chi2 is often reported without df in prose), is returned with
    checkable=False rather than silently dropped or guessed.

    Returns a list of:
        {"test": "t"/"f"/"chi2"/"r"/"z", "stat": float, "df": float|None,
         "df2": float|None (F only), "reported_cmp": str, "reported_p": float,
         "recomputed_p": float|None, "checkable": bool,
         "numeric_consistent": bool|None (recomputed within tolerance of reported),
         "decision_inconsistent": bool|None (recomputed crosses the alpha=.05
             significance boundary the OPPOSITE way from what was reported -- the
             headline statcheck-style flag),
         "context": ~120-char snippet around the match}

    Recomputation uses scipy if available, else sympy.stats (see _recompute_p) -- never
    a hand-rolled approximation of a distribution CDF (that would defeat the point of
    "recompute", not just approximate it).
    """
    if not text:
        return []
    t = text[:_STAT_TEXT_CAP]
    out = []
    seen_spans = set()
    for test, pat in _STAT_PATTERNS:
        for m in pat.finditer(t):
            key = (m.start(), m.end())
            if key in seen_spans:
                continue
            seen_spans.add(key)
            gd = m.groupdict()
            df = gd.get("df") or gd.get("df1")
            df2 = gd.get("df2")
            stat = gd.get("stat")
            window = t[m.end():m.end() + _P_WINDOW]
            pm = _P_NEAR_RE.match(window) or _P_NEAR_RE.search(window[:20])
            row = {
                "test": test, "stat": float(stat),
                "df": float(df) if df else None,
                "df2": float(df2) if df2 else None,
                "reported_cmp": None, "reported_p": None,
                "recomputed_p": None, "checkable": False,
                "numeric_consistent": None, "decision_inconsistent": None,
                "context": t[max(0, m.start() - 40):m.end() + _P_WINDOW].replace("\n", " "),
            }
            if pm is None:
                out.append(row)
                continue
            cmp, p_str = pm.group("cmp"), pm.group("p")
            reported_p = float(p_str)
            row["reported_cmp"], row["reported_p"] = cmp, reported_p
            recomputed = _recompute_p(test, row["stat"], row["df"], row["df2"])
            row["recomputed_p"] = recomputed
            if recomputed is None:
                out.append(row)
                continue
            row["checkable"] = True
            tol = max(0.005, 0.15 * reported_p)
            if cmp in ("<", "<="):
                row["numeric_consistent"] = recomputed <= reported_p * 1.5 + tol
            elif cmp in (">", ">="):
                row["numeric_consistent"] = recomputed >= reported_p * 0.5 - tol
            else:
                row["numeric_consistent"] = abs(recomputed - reported_p) <= tol
            reported_sig = _p_significant(cmp, reported_p)
            recomputed_sig = recomputed <= 0.05 + 1e-9
            row["decision_inconsistent"] = (
                None if reported_sig is None else reported_sig != recomputed_sig)
            out.append(row)
    return out


_COUNT_PCT_RE = re.compile(
    r"(?P<num>\d[\d,]*)\s*(?:out of|of|/)\s*(?P<denom>\d[\d,]*)"
    r"[^.\n%]{0,25}?\(?\s*(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent)\)?", re.I)
_DELTA_PCT_RE = re.compile(
    r"from\s+(?P<a>\d[\d,]*(?:\.\d+)?)\s+to\s+(?P<b>\d[\d,]*(?:\.\d+)?)"
    r"[^.\n%]{0,40}?(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent)", re.I)


@_safe(list)
def number_consistency(text):
    """Re-derivable arithmetic in prose: percentages vs. counts. Two checkable
    patterns, both scoped to a single sentence-ish window (no cross-sentence pairing of
    unrelated numbers):

      1. "X out of/of Y (Z%)" -- checks Z ~= 100*X/Y.
      2. "increased/changed from A to B, a Z% ..." -- checks Z ~= 100*(B-A)/A.

    Returns a list of {"kind": "count_pct"|"delta_pct", ..., "computed_pct": float,
    "stated_pct": float, "consistent": bool, "context": str}. Tolerance is 1.0
    percentage point OR 8% relative (whichever is larger) to absorb reporting-side
    rounding (e.g. "83.3%" rounded from a slightly different raw count some documents
    also don't show); anything outside that is flagged inconsistent=False, not dropped.
    """
    if not text:
        return []
    t = text[:_STAT_TEXT_CAP]
    out = []
    for m in _COUNT_PCT_RE.finditer(t):
        num = float(m.group("num").replace(",", ""))
        denom = float(m.group("denom").replace(",", ""))
        stated = float(m.group("pct"))
        if denom <= 0:
            continue
        computed = 100.0 * num / denom
        tol = max(1.0, 0.08 * stated)
        out.append({
            "kind": "count_pct", "num": num, "denom": denom,
            "stated_pct": stated, "computed_pct": round(computed, 2),
            "consistent": abs(computed - stated) <= tol,
            "context": t[max(0, m.start() - 20):m.end() + 5].replace("\n", " "),
        })
    for m in _DELTA_PCT_RE.finditer(t):
        a = float(m.group("a").replace(",", ""))
        b = float(m.group("b").replace(",", ""))
        stated = float(m.group("pct"))
        if a == 0:
            continue
        computed = 100.0 * (b - a) / a
        tol = max(1.0, 0.08 * abs(stated))
        out.append({
            "kind": "delta_pct", "a": a, "b": b,
            "stated_pct": stated, "computed_pct": round(computed, 2),
            "consistent": abs(abs(computed) - abs(stated)) <= tol,
            "context": t[max(0, m.start() - 10):m.end() + 5].replace("\n", " "),
        })
    return out


# ===========================================================================
# 4. PROCEDURAL DATES  (legal axis)
# ===========================================================================
_DATE_SCAN_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+"
    r"\d{1,2}(?:st|nd|rd|th)?(?:\s*,\s*\d{4}|\s+\d{4})?"
    r"|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s*,?\s*\d{4}"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b")


@_safe(list)
def date_chain(text):
    """All dates in document order: dateutil.parser parses each surface match into a
    calendar date, guarded per-match (an unparseable match, e.g. a false-positive
    fragment, is skipped rather than raising). Computes the interval to the PREVIOUS
    successfully-parsed date.

    Returns a list of {"text": surface match, "start": char offset, "date": ISO date
    string, "days_since_prev": int|None (None for the first date, or if parse order
    goes backward -- negative intervals ARE reported, they are informative: a document
    that states dates out of chronological order is itself a signal)}.

    Degrades to [] if python-dateutil is unavailable.
    """
    if not text:
        return []
    try:
        from dateutil import parser as dtparser
    except ImportError:
        return []
    out = []
    prev_date = None
    for m in _DATE_SCAN_RE.finditer(text[:_STAT_TEXT_CAP]):
        raw = m.group(0)
        try:
            d = dtparser.parse(raw, fuzzy=True, default=_DATE_DEFAULT)
        except Exception:
            continue
        days_since = (d.date() - prev_date).days if prev_date else None
        out.append({"text": raw, "start": m.start(), "date": d.date().isoformat(),
                    "days_since_prev": days_since})
        prev_date = d.date()
    return out


def _make_default_date():
    import datetime
    return datetime.datetime(2000, 1, 1)


_DATE_DEFAULT = None


def _ensure_date_default():
    global _DATE_DEFAULT
    if _DATE_DEFAULT is None:
        _DATE_DEFAULT = _make_default_date()
    return _DATE_DEFAULT


@_safe(lambda: None)
def deadline_satisfied(event_date, filing_date, days):
    """Was `filing_date` within `days` days of `event_date`? Both dates may be str
    (parsed via dateutil) or datetime.date/datetime objects. Returns True/False, or
    None if either date fails to parse -- never guesses a compliance verdict from an
    unparseable date."""
    from dateutil import parser as dtparser
    import datetime

    def _coerce(d):
        if isinstance(d, (datetime.date, datetime.datetime)):
            return d if isinstance(d, datetime.date) and not isinstance(d, datetime.datetime) \
                else d.date()
        return dtparser.parse(str(d), fuzzy=True, default=_ensure_date_default()).date()

    ev, fl = _coerce(event_date), _coerce(filing_date)
    return (fl - ev).days <= int(days)


# ===========================================================================
# 5. DISCOURSE / STRUCTURE  (humor/general axis)
# ===========================================================================
_STOP = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "i", "you", "he", "she",
    "it", "we", "they", "this", "that", "these", "those", "at", "by", "as",
    "so", "if", "then", "than", "do", "does", "did", "have", "has", "had",
    "not", "no", "just", "my", "your", "his", "her", "its", "our", "their",
    "me", "him", "them", "us", "up", "out", "about", "into", "over",
}


def _split_sentences(text):
    """List of (start, end, text) sentence spans, bounded by _SPACY_CHAR_CAP.

    A newline run is ALWAYS treated as a hard sentence boundary first (script/dialogue
    formatting -- "Monkey: Do you have a banana?\\n\\nBartender: No\\n\\nMonkey: ..." --
    uses newlines, not terminal punctuation, as its turn separator; a plain spaCy-only
    sentencizer under-segments this exact real-corpus construction, folding several
    dialogue turns into one "sentence" and hiding the very repetition structure
    is_refrain() needs to see -- the same insight already on record in the a31 h0's own
    `_split_sents`: "nav-chrome/link-list text uses newlines, not periods"). Within each
    newline-delimited block, spaCy's sentencizer further splits on terminal punctuation
    if available, else a regex fallback."""
    if not text:
        return []
    text = text[:_SPACY_CHAR_CAP]
    nlp = _get_nlp()
    out = []
    for m in re.finditer(r"[^\n]+", text):
        block = m.group(0)
        if not block.strip():
            continue
        base = m.start()
        if nlp is not None:
            try:
                bdoc = nlp(block)
                for s in bdoc.sents:
                    if s.text.strip():
                        out.append((base + s.start_char, base + s.end_char, s.text))
                continue
            except Exception:
                pass
        pos = 0
        for piece in re.split(r"(?<=[.!?])\s+", block):
            if piece.strip():
                start = block.find(piece, pos)
                if start == -1:
                    start = pos
                out.append((base + start, base + start + len(piece), piece))
                pos = start + len(piece)
    return out


def _content_words(s):
    doc = _doc_for(s) if len(s) < 400 else None
    if doc is not None:
        return frozenset(
            tok.lemma_.lower() for tok in doc
            if tok.is_alpha and not tok.is_stop and len(tok.text) > 1)
    toks = re.findall(r"[A-Za-z']+", s.lower())
    return frozenset(w for w in toks if len(w) > 1 and w not in _STOP)


def _all_words(s):
    """Every alphabetic token (stopwords INCLUDED), lemmatized if spaCy is available.
    Used for near-duplicate CLUSTERING in is_refrain(): a short dialogue line like
    "Do you have a banana?" is almost entirely function words after stopword removal
    (only "banana" survives _content_words), so clustering must compare on the FULL
    wording, not the content-word-only basis used for the novelty-between measurement."""
    doc = _doc_for(s) if len(s) < 400 else None
    if doc is not None:
        return frozenset(tok.lemma_.lower() for tok in doc if tok.is_alpha)
    return frozenset(re.findall(r"[A-Za-z']+", s.lower()))


@_safe(lambda: None)
def sentence_graph(text):
    """networkx.Graph over the document's sentences: node i = {"text", "start", "end"};
    edge (i, j, weight=Jaccard-overlap-of-content-words) for every pair with
    overlap > 0.08 (sparse by construction -- most sentence pairs share nothing).
    Content words = spaCy lemmas (non-stopword, alphabetic) if spaCy is available, else
    a stdlib regex-tokenize + small stoplist fallback. Capped at the first 240
    sentences (O(n^2) pairs). Returns None if networkx is unavailable.

    This is the shared substrate for is_refrain() and discourse_position() below --
    exposed directly too, since "which earlier sentence does this one echo, and how
    strongly" is itself a useful structural signal beyond the two derived ops.
    """
    import networkx as nx
    sents = _split_sentences(text)
    G = nx.Graph()
    for i, (a, b, s) in enumerate(sents):
        G.add_node(i, text=s, start=a, end=b)
    bows = [_content_words(s) for _, _, s in sents]
    n = min(len(sents), 240)
    for i in range(n):
        wi = bows[i]
        if not wi:
            continue
        for j in range(i + 1, n):
            wj = bows[j]
            if not wj:
                continue
            inter = wi & wj
            if not inter:
                continue
            w = len(inter) / len(wi | wj)
            if w > 0.08:
                G.add_edge(i, j, weight=w, shared=len(inter))
    return G


@_safe(list)
def is_refrain(text):
    """Recurring sentences, classified as CRAFT (rule-of-three-style callback with
    escalating/progressing context between occurrences) vs. PADDING (near-duplicate
    with nothing new happening between occurrences) -- the a117 axis ("repetition-as-
    craft vs repetition-as-padding is not Jaccard-separable").

    Method: cluster near-duplicate sentences (FULL-wording Jaccard >= 0.55 -- stopwords
    included, since a short dialogue refrain like "Do you have a banana?" is almost
    entirely function words and would never cluster on content words alone) into
    occurrence groups. For each group with >=2 occurrences, the
    key signal is NOVELTY BETWEEN occurrences, not the overlap magnitude of the repeat
    itself: for each gap between consecutive occurrences, compute the fraction of
    content words in the BETWEEN-material that were not already used anywhere before
    that gap started. High between-novelty (>=0.4) with a non-trivial gap (>=1
    sentence) means the text actually progressed between callbacks -- a refrain. Near-
    zero novelty (adjacent or content-free gaps) means nothing happened between the
    repeats -- padding. A gap sentence that is ITSELF part of a different >=2-occurrence
    cluster (a second static refrain filling the space, e.g. alternating A/B boilerplate
    -- "great deal... buy now... great deal... buy now...") contributes no novelty
    credit at all: two interleaved refrains are still padding, not progression. A final
    occurrence that is a near- (not exact-) duplicate of
    the earlier ones (a "twist" line, e.g. "Then, do you have a banana?" after two
    verbatim "Do you have a banana?" lines) is itself counted as evidence of
    escalation, since exact-copy padding never varies the line.

    Returns a list of:
        {"sentence": the first occurrence's text, "occurrences": [sentence indices],
         "is_refrain": bool, "reason": str,
         "novelty_between": [float, ...] (one per gap),
         "varied_final": bool (last occurrence not verbatim-identical to the first)}
    """
    sents = _split_sentences(text)
    n = len(sents)
    if n < 3:
        return []
    all_bows = [_all_words(s) for _, _, s in sents]     # clustering basis
    bows = [_content_words(s) for _, _, s in sents]      # novelty-measurement basis
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        if len(all_bows[i]) < 3:
            continue
        for j in range(i + 1, n):
            if len(all_bows[j]) < 3:
                continue
            inter = all_bows[i] & all_bows[j]
            if not inter:
                continue
            jac = len(inter) / len(all_bows[i] | all_bows[j])
            if jac >= 0.55:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # sentences that are THEMSELVES part of some other >=2-occurrence cluster don't
    # count as "new" material when they show up inside another cluster's gap -- a
    # second static refrain filling the gap is not narrative progression, it's more
    # padding (the exact AABB alternating-boilerplate failure mode this guards).
    repeated_idxs = {i for occ in groups.values() if len(occ) >= 2 for i in occ}
    out = []
    for occ in groups.values():
        if len(occ) < 2:
            continue
        occ = sorted(occ)
        seen_before = set(bows[occ[0]])
        novelty = []
        for k in range(len(occ) - 1):
            gap_start, gap_end = occ[k] + 1, occ[k + 1]
            between_words = set()
            for gi in range(gap_start, gap_end):
                if gi in repeated_idxs:
                    continue
                between_words |= bows[gi]
            if between_words:
                new = between_words - seen_before
                novelty.append(len(new) / len(between_words))
            else:
                novelty.append(0.0)
            seen_before |= between_words | bows[occ[k + 1]]
        exact_first = all_bows[occ[0]]
        exact_last = all_bows[occ[-1]]
        varied_final = exact_first != exact_last and bool(exact_first & exact_last)
        avg_gap_len = sum(occ[k + 1] - occ[k] for k in range(len(occ) - 1)) / (len(occ) - 1)
        avg_novelty = sum(novelty) / len(novelty) if novelty else 0.0
        is_craft = (avg_gap_len >= 1.0 and (avg_novelty >= 0.4 or varied_final))
        reason = ("escalating callback: new material between repeats"
                  if is_craft and avg_novelty >= 0.4 else
                  "escalating callback: varied final occurrence (twist line)"
                  if is_craft else
                  "near-adjacent or content-free gaps: no progression between repeats"
                  if avg_gap_len < 1.0 or avg_novelty < 0.1 else
                  "low novelty between repeats")
        out.append({
            "sentence": sents[occ[0]][2], "occurrences": occ,
            "is_refrain": is_craft, "reason": reason,
            "novelty_between": [round(x, 3) for x in novelty],
            "varied_final": varied_final,
        })
    return out


@_safe(lambda: None)
def discourse_position(text, span):
    """Where does `span` (a substring of `text`, or a (start, end) char-offset pair)
    fall structurally: "opening" (first ~15% of sentences, at least 1), "coda" (last
    ~10% of sentences, at least 1), or "middle"? Returns None if `span` cannot be
    located (string not found verbatim) or the document has no sentences.

    This targets the a117 "tail-excision blind-deletes a punchline paragraph" failure:
    a fixed trailing-character-percentage cut has no relationship to sentence
    boundaries and can slice through the middle of the actual final beat. Position here
    is computed over SENTENCE COUNT, so "coda" always resolves to whole sentences.
    """
    sents = _split_sentences(text)
    n = len(sents)
    if n == 0:
        return None
    if isinstance(span, (tuple, list)) and len(span) == 2:
        s0 = int(span[0])
    elif isinstance(span, str):
        idx = text.find(span)
        if idx == -1:
            return None
        s0 = idx
    else:
        return None
    idx_sent = None
    for i, (a, b, _s) in enumerate(sents):
        if a <= s0 < b:
            idx_sent = i
            break
    if idx_sent is None:
        idx_sent = n - 1 if s0 >= sents[-1][1] else 0
    opening_cut = max(1, math.ceil(n * 0.15))
    coda_cut = n - max(1, math.ceil(n * 0.10))
    if idx_sent < opening_cut:
        return "opening"
    if idx_sent >= coda_cut:
        return "coda"
    return "middle"


# ===========================================================================
# 6. NER FACTS
# ===========================================================================
_NUM_NEARBY_RE = re.compile(r"\d")
_URL_NEARBY_RE = re.compile(r"https?://|www\.", re.I)
_EVIDENCE_WINDOW = 80


@_safe(lambda: {"counts": {}, "total": 0, "per_1000_words": 0.0})
def fact_density(text):
    """spaCy NER entity counts by type: {"counts": {"ORG": n, "PERSON": n, ...},
    "total": sum, "per_1000_words": total normalized by document word count}.
    Degrades to an all-zero dict if spaCy is unavailable."""
    doc = _doc_for(text)
    if doc is None:
        return {"counts": {}, "total": 0, "per_1000_words": 0.0}
    counts = Counter(ent.label_ for ent in doc.ents)
    total = sum(counts.values())
    n_words = max(1, sum(1 for tok in doc if tok.is_alpha))
    return {"counts": dict(counts), "total": total,
            "per_1000_words": round(1000.0 * total / n_words, 2)}


@_safe(list)
def entities_with_evidence(text):
    """Named entities that appear near a NUMBER, a DATE-like string, or a URL (within
    _EVIDENCE_WINDOW=80 chars) -- a cheap, label-free proxy for "this mention is
    grounded in a specific fact" vs. a bare name-drop. Returns a list of:
        {"text": entity surface form, "label": spaCy NER type, "start": char offset,
         "end": char offset, "has_number_nearby": bool, "has_date_nearby": bool,
         "has_url_nearby": bool, "evidenced": bool (any of the three)}
    A DATE-type entity does not need a second co-occurring date to count -- only
    number/date/url nearby matters for the OTHER entity types; DATE entities are
    still returned (evidenced by number/url if present) for completeness.
    Degrades to [] if spaCy is unavailable.
    """
    doc = _doc_for(text)
    if doc is None:
        return []
    t = doc.text
    out = []
    for ent in doc.ents:
        lo = max(0, ent.start_char - _EVIDENCE_WINDOW)
        hi = min(len(t), ent.end_char + _EVIDENCE_WINDOW)
        window = t[lo:ent.start_char] + t[ent.end_char:hi]
        has_num = bool(_NUM_NEARBY_RE.search(window))
        has_date = bool(_DATE_SCAN_RE.search(window))
        has_url = bool(_URL_NEARBY_RE.search(window))
        out.append({
            "text": ent.text, "label": ent.label_,
            "start": ent.start_char, "end": ent.end_char,
            "has_number_nearby": has_num, "has_date_nearby": has_date,
            "has_url_nearby": has_url,
            "evidenced": has_num or has_date or has_url,
        })
    return out


# ===========================================================================
# instantiable wrapper (E2L wiring convention -- see module docstring USAGE)
# ===========================================================================
class CapabilityOps:
    """Stateless wrapper bundling every op above as a bound (static)method, so a
    candidate can do `from ops_capability import CapabilityOps; _cap = CapabilityOps()`
    at module scope and call `_cap.attributions(text)` etc. from inside score(). Free to
    instantiate: all state (the spaCy pipeline, the parse cache) lives at module level
    and is shared across every CapabilityOps() instance and every candidate in the
    process. See CAPABILITIES for the op -> {group, requires, degrades_to} index."""

    VERSION = VERSION
    CAPABILITIES = CAPABILITIES

    attributions = staticmethod(attributions)
    self_attributed = staticmethod(self_attributed)
    parse_math = staticmethod(parse_math)
    licensing_does_work = staticmethod(licensing_does_work)
    restates_definition = staticmethod(restates_definition)
    stat_consistency = staticmethod(stat_consistency)
    number_consistency = staticmethod(number_consistency)
    date_chain = staticmethod(date_chain)
    deadline_satisfied = staticmethod(deadline_satisfied)
    sentence_graph = staticmethod(sentence_graph)
    is_refrain = staticmethod(is_refrain)
    discourse_position = staticmethod(discourse_position)
    fact_density = staticmethod(fact_density)
    entities_with_evidence = staticmethod(entities_with_evidence)
