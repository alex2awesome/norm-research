"""Hybrid metric channel for a126: 'Heuristic discovery via patterns, induction,
and analogy' (Math StackExchange answers).

Design note: the description-compiled baseline (train_rho=0.253) scans raw
keyword phrases ('conjecture', 'notice that', ...) over the WHOLE text
(question + answer) with no LaTeX awareness. Two failure modes follow:
  (a) those words appear as ordinary filler in the asker's QUESTION or in
      throwaway phrases in a purely rigorous ANSWER ("it appears you accept
      0 in N") -- presence != quality;
  (b) the judge's actual high scorers (numeric simulation tables, guessed
      finite-difference formulas, explicit analogy statements, named general
      "tricks", Socratic guided proofs) are under-weighted because they show
      the *process* structurally, not via those exact words.

This program (i) isolates the responder's ANSWER (splits off the asker's
Question), (ii) replaces raw phrase-counting with structural evidence of an
actual discovery process -- numeric experiment tables, literal
pattern/sequence lists, gated guess->verify combinations, named heuristics,
counterexample construction, Socratic question density, and case-based
exploration (via proof_skeleton) -- and (iii) adds one LLM field for the
tacit, paraphrase-robust version of the same judgment.
"""

import re

LLM_FIELDS = {
    "discovery_move": (
        "In <=6 words, name the discovery/analogy move the ANSWER makes "
        "(e.g. 'guess then verify', 'cross-domain analogy', 'counterexample "
        "test', 'numeric experiment', 'named general trick'); empty if the "
        "answer is a direct derivation or citation with no such move."
    ),
}


def _sat(x, k=1.0):
    try:
        x = float(x)
        return x / (x + k) if x > 0 else 0.0
    except Exception:
        return 0.0


def _split_answer(t):
    # Isolate the responder's answer so question-side chatter ("any ideas?",
    # "I conjecture that...") doesn't leak into the discovery-process signal.
    tl = t.lower()
    idx = tl.rfind("\nanswer:")
    if idx == -1:
        idx = tl.rfind("answer:")
    return t[idx:] if idx != -1 else t


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = ops.normalize(text)
        ans = _split_answer(t)
        ans_l = ans.lower()

        sigs = []

        # 1) Numeric-experimentation tables: lines carrying >=2 precise decimals
        #    each (simulation output / numeric verification of a conjectured
        #    asymptotic or probability), the strongest, hardest-to-fake signal.
        lines = ans.split("\n")
        decimal_lines = sum(
            1 for ln in lines if len(re.findall(r"-?\d+\.\d{2,}", ln)) >= 2
        )
        sigs.append((1.2, _sat(decimal_lines, 2.0)))

        # 2) Literal numeric pattern/sequence lists (finite-difference style
        #    "guess the formula from the sequence" discovery), >=4 numbers run.
        seq_hits = len(
            re.findall(r"(?:-?\d+(?:\.\d+)?\s*,\s*){3,}-?\d+(?:\.\d+)?", ans)
        )
        sigs.append((0.9, _sat(seq_hits, 1.0)))

        # 3) Explicit cross-domain / structural analogy language in the answer.
        an = sum(
            ans_l.count(p)
            for p in [
                "similar to", "analogous", "reminiscent", "resembles",
                "same idea", "same trick", "same technique", "as in the case",
                "parallel to",
            ]
        )
        sigs.append((0.8, _sat(an, 1.0)))

        # 4) Gated guess/conjecture -> verify combination (both sides must be
        #    present; a bare "guess" with no verification is only weakly credited).
        guess_words = ["conjectur", "guess", "hypothes", "suspect"]
        # NOTE (hygiene patch): bare "check" as a substring also matches
        # "checkmark" -- a StackExchange accepted-answer UI feature in this
        # corpus (e.g. "give one person the checkmark"), unrelated to
        # numeric/logical verification. Boundary-anchor just this one term
        # with a small inflection whitelist so genuine "check/checks/
        # checked/checking" still fire but the compound noun "checkmark"
        # does not.
        check_words = [
            "verify", "verif", "confirm", "consisten", "numerically",
            "simulat", "matches", "agrees",
        ]
        _CHECK_WORD_RE = re.compile(r"\bcheck(?:s|ed|ing)?\b")
        has_guess = any(w in ans_l for w in guess_words)
        has_check = any(w in ans_l for w in check_words) or bool(_CHECK_WORD_RE.search(ans_l))
        gv = 1.0 if (has_guess and has_check) else (0.35 if has_guess else 0.0)
        sigs.append((0.9, gv))

        # 5) Naming the general heuristic/trick behind the specific move.
        hn = sum(
            ans_l.count(p)
            for p in [
                "the trick is", "the idea is", "the point is", "the key idea",
                "key observation", "in general,",
            ]
        )
        sigs.append((0.7, _sat(hn, 1.0)))

        # 6) Counterexample / consequence-checking construction: an "example"
        #    trigger near a negation (testing whether a claim can hold).
        cx = 0
        for m in re.finditer(r"\b(consider|for example|e\.g\.)\b", ans_l):
            window = ans_l[max(0, m.start() - 200): m.end() + 200]
            if re.search(r"\b(cannot|never|false|fails|no such|not true|impossible)\b", window):
                cx += 1
        sigs.append((0.6, _sat(cx, 1.0)))

        # 6b) Empirical/graphical observation (reading off behavior from a
        #     plot or numeric scan rather than a closed-form proof step).
        eo = sum(
            ans_l.count(p)
            for p in ["graph of", "plot of", "illustrates", "plotting", "graphically", "numerically observe"]
        )
        sigs.append((0.5, _sat(eo, 1.0)))

        # 7) Socratic / guided-discovery style: rhetorical questions inside the
        #    answer itself (not the asker's question).
        try:
            n_sent, _mean_w, _frac_long = ops.sent_stats(ans)
        except Exception:
            n_sent = max(ans.count(".") + ans.count("?"), 1)
        qmarks = ans.count("?")
        sigs.append((0.5, _sat((qmarks / max(n_sent, 1)) * 3.0, 1.0)))

        # 8) Case-based exploration (multiple worked cases ~ inductive search
        #    across scenarios), via the LaTeX-aware proof-structure parser.
        try:
            skel = ops.proof_skeleton(ans)
            n_cases = len(skel.get("case", []))
        except Exception:
            n_cases = 0
        sigs.append((0.4, _sat(max(n_cases - 1, 0), 1.5)))

        wsum = sum(s[0] for s in sigs) or 1.0
        agg = sum(a * b for a, b in sigs) / wsum

        # 9) LLM thick-input field: paraphrase-robust version of the same
        #    judgment, for discovery moves the surface patterns above miss.
        dm = (extracted.get("discovery_move") or "").strip().lower()
        llm_boost = 0.0
        if dm and dm not in ("none", "n/a", "na", "no", "none.", "no move"):
            hit = any(
                k in dm
                for k in [
                    "guess", "conjectur", "analog", "counterexam", "experiment",
                    "numeric", "simulat", "pattern", "induct", "verif", "check",
                    "hypothes", "trick",
                ]
            )
            llm_boost = 0.15 if hit else 0.05

        final = 0.08 + 0.77 * agg + llm_boost
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
