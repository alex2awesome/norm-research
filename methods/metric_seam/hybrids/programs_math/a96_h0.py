"""
a96: Demarcate proof from conjecture or heuristic support.

Design rationale (see accompanying report): the baseline keyword-count regex
(train rho=0.033) fails because presence of words like "conjecture" or
"heuristic" is a PROXY, not the target -- the judge actually rewards honest
CALIBRATION between an answer's closing certainty language and how much
justification actually precedes it. Reading the 30 train excerpts:
  - High judge scores (~1.0) go to answers that are either (a) fully worked,
    correctly-concluded derivations, OR (b) explicitly modest partial work
    (e.g. "Hint:", ending with a question back to the asker) that never
    pretends to be more finished than it is.
  - Low judge scores go to answers that OVERCLAIM: a "Q.E.D." or "clearly"
    or "should give you a proof" tag glued onto a derivation that actually
    has an unaddressed gap, OR a bare formula/assertion stated as fact with
    zero derivation and no hedge at all.
This is a semantic judgment a regex cannot make (it requires reading whether
the logic actually closes), so we outsource exactly that reading to the LLM
extractor and keep the combination/prediction rule in code.
"""

import re
import math

LLM_FIELDS = {
    "rigor_label": (
        "In the ANSWER portion only (ignore the Question), is the main "
        "mathematical result presented as a COMPLETE proof, a PARTIAL "
        "hint/sketch left open, a CONJECTURE/empirical claim, or a bare "
        "unproved ASSERTION with no derivation? Reply one word: COMPLETE, "
        "PARTIAL, CONJECTURE, or ASSERTION."
    ),
    "overclaim": (
        "Does the ANSWER use definitive closing language (e.g. QED, "
        "\"therefore proved\", \"clearly\", \"this shows\") for a step "
        "whose justification is actually missing or has a logical gap? "
        "Reply YES, NO, or UNSURE."
    ),
}


def _sat(x, k=1.0):
    return x / (x + k) if x > 0 else 0.0


def _answer_part(text):
    m = re.search(r"\bAnswer\s*:\s*", text)
    if m:
        return text[m.end():]
    return text


def _classify(raw, keywords):
    s = (raw or "").strip().upper()
    for kw in keywords:
        if kw in s:
            return kw
    return ""


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text:
            return 0.5

        try:
            t = ops.normalize(text)
        except Exception:
            t = text

        ans = _answer_part(t)

        # --- LLM-grounded epistemic-calibration signal (core predicate) ---
        rigor_label = _classify((extracted or {}).get("rigor_label", ""),
                                 ["COMPLETE", "PARTIAL", "CONJECTURE", "ASSERTION"])
        overclaim = _classify((extracted or {}).get("overclaim", ""),
                               ["YES", "NO", "UNSURE"])

        rigor_map = {
            "COMPLETE": 0.35,
            "PARTIAL": 0.28,   # honest hints/sketches score nearly as well as full proofs
            "CONJECTURE": 0.05,
            "ASSERTION": -0.28,
        }

        # --- code-only structural signals: cheap regularizer / fallback ---
        try:
            skeleton = ops.proof_skeleton(ans)
        except Exception:
            skeleton = {}
        n_proof_markers = len(skeleton.get("proof", [])) + len(skeleton.get("qed", []))
        n_connective = len(skeleton.get("connective", []))

        try:
            n_display, n_inline, n_numbered, avg_tokens = ops.equation_stats(ans)
        except Exception:
            n_display, n_inline, n_numbered, avg_tokens = 0, 0, 0, 0.0

        try:
            n_sent, mean_wps, frac_long = ops.sent_stats(ans)
        except Exception:
            n_sent, mean_wps, frac_long = 1, 10.0, 0.0

        base = 0.5
        # Overclaiming (a confident close glued onto an unaddressed gap) is
        # the sharpest failure mode in the train data -- it dominates the
        # rigor_label credit rather than just partially offsetting it, since
        # a dishonest QED is judged worse than an honest partial sketch.
        if overclaim == "YES":
            val = 0.15 + 0.05 * _sat(n_proof_markers, 1.0)
        elif overclaim == "UNSURE":
            val = base + 0.6 * rigor_map.get(rigor_label, 0.0)
        else:  # "NO" or field unavailable -> trust rigor_label at full weight
            val = base + rigor_map.get(rigor_label, 0.0)
            if n_proof_markers > 0:
                val += 0.03 * _sat(n_proof_markers, 1.0)

        # thin support: almost no math spans and no derivation-length text,
        # and no hedged PARTIAL/CONJECTURE framing -> bare assertion, even
        # if the LLM field came back empty/unparseable.
        thin = (n_display + n_inline) <= 1 and avg_tokens < 4 and n_sent <= 3
        if thin and rigor_label not in ("PARTIAL", "CONJECTURE"):
            val -= 0.07
        # some connective/derivation structure is a mild positive regardless
        val += 0.02 * _sat(n_connective, 2.0)

        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
