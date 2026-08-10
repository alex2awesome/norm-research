"""
a144: Existential proofs: explicit witnesses and scope.

Judge rewards answers that (a) introduce a CONCRETE witness (a specific value,
function, or set) rather than a hint/sketch, (b) VERIFY that witness satisfies
the required property, and (c) handle quantifier scope precisely (fix/choose,
for all / there exists, case splits). The baseline (train rho 0.225) counts
raw keywords like "let"/"there exist" over the WHOLE text (question+answer),
which rewards keyword presence regardless of whether the derivation actually
completes. Two concrete train failures motivate the redesign here:
  - d00629 (judge 0.7, baseline 0.05): builds an explicit closed-form formula
    via decomposition/verification with none of the baseline's magic words.
  - d04973 (judge 0.0, baseline 0.28): question text contains "\\exists"/
    "there exists" and answer lists "Observations" but explicitly punts
    ("Combining these observations should give you a proof") -- never
    produces or verifies a witness.
So: (1) score the Answer-only span, not the Question; (2) use the math-aware
ops (proof_skeleton, equation_stats, extract_math_spans) to detect concrete,
verified conclusions instead of raw keyword counts; (3) use one LLM field to
read whether the derivation actually COMPLETES to a definite witness -- a
hint-vs-completed judgment that keyword regex cannot make reliably (e.g.
"Hint:"-prefixed answers span the full quality range in the training data).
"""

import re

LLM_FIELDS = {
    "witness": "In <=10 words, the specific value/function/set the answer constructs as the witness, or NONE if none is given",
    "completed": "Answer only yes or no: does the answer finish the derivation to a definite conclusion rather than stop at a hint or sketch",
}

_ANSWER_MARK = re.compile(r"\bAnswer\s*:", re.IGNORECASE)

_HEDGE_PATS = [
    re.compile(r"should (give|work|prove|do the trick)", re.IGNORECASE),
    re.compile(r"i(?:'m| am) not (?:really )?sure", re.IGNORECASE),
    re.compile(r"i don't (?:see|know) how", re.IGNORECASE),
    re.compile(r"left (?:to|as) (?:the reader|an exercise)", re.IGNORECASE),
    re.compile(r"without success", re.IGNORECASE),
]

_QUANT_PATS = [
    re.compile(r"there\s+exists?", re.IGNORECASE),
    re.compile(r"\\exists"),
    re.compile(r"for\s+(all|each|every)\b", re.IGNORECASE),
    re.compile(r"\\forall"),
    re.compile(r"\barbitrary\b", re.IGNORECASE),
    re.compile(r"\bfix\b", re.IGNORECASE),
]

_UNIQ_PATS = [
    re.compile(r"\bunique(ness)?\b", re.IGNORECASE),
    re.compile(r"exactly one", re.IGNORECASE),
    re.compile(r"at most one", re.IGNORECASE),
    re.compile(r"is the only", re.IGNORECASE),
]


def _safe_call(fn, default, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def _clip01(x):
    return max(0.0, min(1.0, x))


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.0

        norm = _safe_call(ops.normalize, text, text)
        if not isinstance(norm, str) or not norm.strip():
            norm = text

        m = _ANSWER_MARK.search(norm)
        ans = norm[m.end():] if m else norm
        if not ans.strip():
            ans = norm

        # ---- code-only structural signals (Answer-only text) ----
        skel = _safe_call(ops.proof_skeleton, {}, ans)
        if not isinstance(skel, dict):
            skel = {}

        def cat_count(name):
            try:
                return len(skel.get(name, []) or [])
            except Exception:
                return 0

        n_def = cat_count("definition")
        n_qed = cat_count("qed")
        n_case = cat_count("case")
        n_thm = cat_count("theorem")
        n_conn = cat_count("connective")
        n_proof = cat_count("proof")
        structure = n_case + n_thm + min(n_conn, 5) + min(n_proof, 3)

        eq_stats = _safe_call(ops.equation_stats, (0, 0, 0, 0.0), ans)
        try:
            n_display, n_inline, n_numbered, avg_tokens = eq_stats
        except Exception:
            n_display, n_inline, n_numbered, avg_tokens = (0, 0, 0, 0.0)

        spans = _safe_call(ops.extract_math_spans, [], ans)
        if not isinstance(spans, list):
            spans = []
        eq_spans = 0
        boxed = False
        for item in spans:
            try:
                kind, content = item
            except Exception:
                continue
            if not isinstance(content, str):
                continue
            if "\\boxed" in content:
                boxed = True
            if "=" in content and len(content.strip()) > 1:
                eq_spans += 1

        dh = _safe_call(ops.delimiter_health, {}, ans)
        bad = 0
        if isinstance(dh, dict):
            for v in dh.values():
                if isinstance(v, (int, float)):
                    bad += v

        hedges = sum(len(p.findall(ans)) for p in _HEDGE_PATS)
        quant = sum(len(p.findall(ans)) for p in _QUANT_PATS)
        uniq = sum(len(p.findall(ans)) for p in _UNIQ_PATS)

        raw = 0.05
        raw += 0.15 * min(1.0, n_def / 3.0)
        raw += 0.12 * min(1.0, n_qed / 2.0)
        raw += 0.08 * min(1.0, structure / 4.0)
        raw += 0.15 * min(1.0, (eq_spans + (1 if boxed else 0)) / 3.0)
        raw += 0.08 * min(1.0, (n_display + n_numbered) / 2.0)
        raw += 0.07 * min(1.0, quant / 3.0)
        raw += 0.05 * min(1.0, uniq / 2.0)
        raw -= 0.10 * min(1.0, hedges / 2.0)
        raw -= 0.05 * min(1.0, bad / 5.0)

        # ---- LLM-grounded signal: hint vs completed, and witness presence ----
        extracted = extracted or {}
        witness_txt = str(extracted.get("witness", "") or "").strip()
        has_witness = bool(witness_txt) and witness_txt.strip().upper() not in ("NONE", "N/A", "NA")

        completed_txt = str(extracted.get("completed", "") or "").strip().lower()
        if completed_txt.startswith("y"):
            comp_adj = 0.15
        elif completed_txt.startswith("n"):
            comp_adj = -0.20
        else:
            comp_adj = 0.0

        wit_adj = 0.15 if has_witness else -0.05

        final = raw + wit_adj + comp_adj
        return _clip01(final)
    except Exception:
        return 0.5


if __name__ == "__main__":
    # Minimal smoke test with a mock ops object matching the documented
    # interface, exercised on self-constructed strings (no corpus access).
    import statistics  # noqa: F401 (allowed stdlib, unused here but permitted)

    class MockOps:
        def normalize(self, t):
            return t

        def extract_dates(self, t):
            return []

        def sent_stats(self, t):
            sents = re.split(r"(?<=[.!?])\s+", t.strip())
            sents = [s for s in sents if s]
            n = len(sents) or 1
            words = t.split()
            mean_w = len(words) / n
            long_frac = sum(1 for w in words if len(w) > 8) / (len(words) or 1)
            return (n, mean_w, long_frac)

        def retrieve_similar(self, t, k=5, exclude_id=None):
            return [(1.0, "self")]

        def extract_math_spans(self, t):
            spans = []
            for m in re.finditer(r"\$\$(.+?)\$\$|\$(.+?)\$", t, re.DOTALL):
                content = m.group(1) or m.group(2) or ""
                kind = "display" if m.group(1) is not None else "inline"
                spans.append((kind, content))
            return spans

        def latex_tokens(self, span_content):
            return re.findall(r"\\[a-zA-Z]+|[A-Za-z]+|[0-9]+|\S", span_content)

        def notation_census(self, t):
            from collections import Counter
            c = Counter(re.findall(r"\\[a-zA-Z]+", t))
            return dict(c)

        def equation_stats(self, t):
            spans = self.extract_math_spans(t)
            n_display = sum(1 for k, c in spans if k == "display")
            n_inline = sum(1 for k, c in spans if k == "inline")
            n_numbered = sum(1 for k, c in spans if "\\tag" in c)
            toks = [len(self.latex_tokens(c)) for k, c in spans]
            avg_tokens = (sum(toks) / len(toks)) if toks else 0.0
            return (n_display, n_inline, n_numbered, avg_tokens)

        def proof_skeleton(self, t):
            cats = {
                "theorem": [r"\bclaim\b", r"\btheorem\b", r"\blemma\b"],
                "definition": [r"\blet\b", r"\bdefine\b", r"\bdenote\b", r"\bset\b", r"\bchoose\b", r"\btake\b", r"\bfix\b"],
                "proof": [r"\bproof\b"],
                "qed": [r"\bqed\b", r"as required", r"this proves", r"hence proved", r"∠"],
                "case": [r"\bcase\s*\d"],
                "connective": [r"\btherefore\b", r"\bthus\b", r"\bhence\b", r"\bso\b"],
            }
            out = {}
            for cat, pats in cats.items():
                hits = []
                for p in pats:
                    for m in re.finditer(p, t, re.IGNORECASE):
                        hits.append((m.start(), m.group(0)))
                out[cat] = hits
            return out

        def delimiter_health(self, t):
            return {
                "odd_dollar_parity": t.count("$") % 2,
                "brace_imbalance": abs(t.count("{") - t.count("}")),
            }

    mock = MockOps()

    completed_case = (
        "Question: does there exist x? Answer: Let r = min(|x|,|y|,|z|). "
        "Then $|u| > 0$ as required. So $(u,v,w) \\in A$."
    )
    hint_case = (
        "Question: hypothesize there exists k. Answer: Observation 1: blah. "
        "Combining these observations should give you a proof."
    )

    print("completed_case, no LLM fields:", score(completed_case, {}, mock))
    print("completed_case, with LLM fields:", score(completed_case, {"witness": "r = min(|x|,|y|,|z|)", "completed": "yes"}, mock))
    print("hint_case, no LLM fields:", score(hint_case, {}, mock))
    print("hint_case, with LLM fields:", score(hint_case, {"witness": "NONE", "completed": "no"}, mock))
    print("empty text:", score("", {}, mock))
    print("garbage extracted:", score(completed_case, None, mock))
