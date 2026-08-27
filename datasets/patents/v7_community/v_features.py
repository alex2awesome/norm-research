#!/usr/bin/env python3
"""Deterministic, label-blind V features for GRANTED US PATENTS (V7 forward-
citation community cell).

Same module shape as `datasets/stackoverflow-votes/va/v_features.py` and
`datasets/math/stackexchange/va/v_features.py` (V_NAMES / v_features / vector),
and it carries the SAME generic length-structure-register tail so the surface
channel is comparable across cells. The domain block is patent-specific: claim-1
grammar (preamble kind, transitional phrase, limitation count, means-plus-
function, functional "configured to" language), the classic textual
claim-BREADTH proxies from the patent-scope literature (claim-1 word count,
number of limitations, "plurality of" / "at least one" quantifiers), and the
definiteness/relative-term markers the MPEP flags.

INPUT IS THE DOCUMENT TEXT ONLY -- title, abstract, claim 1. This is the cell
whose sibling (patents claim-fell) was closed as a metadata-leak post-mortem, so
the exclusions are load-bearing and asserted, not merely intended:

  NOTHING here may read, and none of these is passed in: examiner identity or
  art unit, assignee/inventor/attorney identity, filing or grant date, the
  patent number, CPC codes, num_claims, or any citation count. Every column
  below is a regex count or ratio over the three text fields.

`num_claims` in particular is a DECLARED NUISANCE channel for this cell
(alone-AUC ~.60 against y, measured in population_manifest.json's leak battery):
it is deliberately NOT a feature here, because it is not recoverable from the
text the instruments actually see (claim 1 only).
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List

# --------------------------------------------------------------- lexicons ---
# Claim-1 grammar and scope language. Sources for the choices: MPEP 2111.03
# (transitional phrases: "comprising" is open-ended, "consisting of" closed),
# MPEP 2173.05(b) (relative terms and definiteness), 35 USC 112(f) (means-plus-
# function), and the claim-scope measurement literature (word count and
# limitation count as breadth proxies).
CLAIM_LEX = {
    "v_c1_comprising": r"\bcomprising\b|\bcomprises\b",
    "v_c1_consisting": r"\bconsisting (?:essentially )?of\b",
    "v_c1_including": r"\bincluding\b|\bincludes\b",
    "v_c1_wherein": r"\bwherein\b",
    "v_c1_means_for": r"\bmeans for\b",
    "v_c1_configured_to": r"\bconfigured to\b|\badapted to\b|\boperable to\b",
    "v_c1_plurality": r"\ba plurality of\b|\bplural\b",
    "v_c1_at_least_one": r"\bat least one\b|\bone or more\b",
    "v_c1_said": r"\bsaid\b",
    "v_c1_the_antecedent": r"\bthe said\b|\bthereof\b|\btherein\b|\bthereto\b|"
                           r"\bthereby\b|\bthereon\b",
    "v_c1_relative_term": r"\b(?:about|substantially|approximately|generally|"
                          r"essentially|relatively|significantly|nearly)\b",
    "v_c1_optional": r"\b(?:optionally|preferably|if desired|as needed|may be)\b",
    "v_c1_range": r"\b(?:between\s+\d|from\s+\d+\s+to\s+\d|less than|greater than|"
                  r"at most|no more than)\b",
    "v_c1_step_of": r"\bstep(?:s)? of\b",
    "v_c1_first_second": r"\b(?:first|second|third|fourth)\b",
    "v_c1_respectively": r"\brespectively\b",
    "v_c1_negative_limit": r"\b(?:free of|absent|without|excluding|other than|"
                           r"non-)\b",
}
# Abstract / title register.
DOC_LEX = {
    "v_doc_invention_framing": r"\b(?:the present invention|this invention|"
                               r"the invention relates|disclosed herein|"
                               r"in one embodiment|according to)\b",
    "v_doc_problem_language": r"\b(?:problem|drawback|disadvantage|limitation|"
                              r"difficulty|shortcoming|need for|challenge)\b",
    "v_doc_benefit_language": r"\b(?:improve[sd]?|enhanc\w+|reduc\w+|increas\w+|"
                              r"efficien\w+|advantage|benefit|optimiz\w+)\b",
    "v_doc_embodiment": r"\b(?:embodiment|implementation|aspect|variant|"
                        r"example)\b",
    "v_doc_hedging": r"\b(?:may|can|might|could|possibly|optionally|"
                     r"in some cases)\b",
    "v_doc_system_noun": r"\b(?:system|apparatus|device|assembly|module|unit|"
                         r"circuit|mechanism)\b",
    "v_doc_method_noun": r"\b(?:method|process|technique|procedure|algorithm|"
                         r"protocol)\b",
    "v_doc_composition_noun": r"\b(?:composition|compound|formulation|mixture|"
                              r"alloy|polymer|solution)\b",
    "v_doc_data_noun": r"\b(?:data|signal|information|memory|processor|"
                       r"network|server|database|software)\b",
}

_CLAIM_RE = {k: re.compile(v, re.I) for k, v in CLAIM_LEX.items()}
_DOC_RE = {k: re.compile(v, re.I) for k, v in DOC_LEX.items()}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
SENT_END_RE = re.compile(r"[.!?](?:\s|$)")
# A claim's limitations are delimited by semicolons; the preamble ends at the
# transitional phrase. Reference numerals look like "(12)" or "12a".
SEMI_RE = re.compile(r";")
REFNUM_RE = re.compile(r"\(\s*\d{1,3}[a-z]?\s*\)")
CLAIM_NUM_PREFIX_RE = re.compile(r"^\s*\d+\s*\.\s*")
TRANSITION_RE = re.compile(r"\b(?:comprising|consisting (?:essentially )?of|"
                           r"including|having|containing)\b", re.I)

V_NAMES: List[str] = (
    list(CLAIM_LEX)
    + list(DOC_LEX)
    + [
        # --- claim-1 structure / breadth proxies -------------------------
        "v_c1_word_count", "v_c1_log_len", "v_c1_n_limitations",
        "v_c1_preamble_word_count", "v_c1_body_word_count",
        "v_c1_mean_limitation_words", "v_c1_n_refnums",
        "v_c1_is_method", "v_c1_is_crm", "v_c1_comma_count",
        "v_c1_type_token_ratio", "v_c1_numeral_density",
        "v_c1_avg_word_len", "v_c1_long_word_share",
        # --- abstract ----------------------------------------------------
        "v_abs_word_count", "v_abs_log_len", "v_abs_sentence_count",
        "v_abs_avg_sentence_words", "v_abs_type_token_ratio",
        "v_abs_numeral_density", "v_abs_alpha_share",
        "v_abs_uppercase_letter_ratio", "v_abs_avg_word_len",
        # --- title -------------------------------------------------------
        "v_title_word_count", "v_title_char_count", "v_title_avg_word_len",
        "v_title_has_and", "v_title_uppercase_ratio",
        # --- cross-field --------------------------------------------------
        "v_abs_to_claim_word_ratio", "v_title_claim_word_overlap",
        "v_abs_claim_word_overlap", "v_doc_total_word_count",
    ]
)


def _toks(s: str) -> List[str]:
    return WORD_RE.findall(s or "")


def _ttr(ws: List[str]) -> float:
    return len({w.lower() for w in ws}) / max(len(ws), 1)


def v_features(title: str, abstract: str, claim1: str) -> Dict[str, float]:
    t, ab = title or "", abstract or ""
    c1 = CLAIM_NUM_PREFIX_RE.sub("", claim1 or "")   # drop the leading "1."
    out: Dict[str, float] = {}

    # lexicon counts: claim lexicon on claim 1, doc lexicon on title+abstract
    for k, rx in _CLAIM_RE.items():
        out[k] = float(len(rx.findall(c1)))
    doc_text = t + " " + ab
    for k, rx in _DOC_RE.items():
        out[k] = float(len(rx.findall(doc_text)))

    # ---- claim 1 -------------------------------------------------------
    cw = _toks(c1)
    out["v_c1_word_count"] = float(len(cw))
    out["v_c1_log_len"] = float(math.log1p(len(c1)))
    out["v_c1_n_limitations"] = float(len(SEMI_RE.findall(c1)) + 1)
    m = TRANSITION_RE.search(c1)
    pre = c1[: m.start()] if m else c1
    body = c1[m.end():] if m else ""
    out["v_c1_preamble_word_count"] = float(len(_toks(pre)))
    out["v_c1_body_word_count"] = float(len(_toks(body)))
    out["v_c1_mean_limitation_words"] = float(
        len(cw) / max(len(SEMI_RE.findall(c1)) + 1, 1))
    out["v_c1_n_refnums"] = float(len(REFNUM_RE.findall(c1)))
    out["v_c1_is_method"] = float(bool(re.match(
        r"^\s*(?:a|an|the)?\s*(?:computer[- ]implemented\s+)?(?:method|process)\b",
        pre, re.I)))
    out["v_c1_is_crm"] = float(bool(re.search(
        r"\b(?:computer[- ]readable|machine[- ]readable|storage medium|"
        r"program product)\b", pre, re.I)))
    out["v_c1_comma_count"] = float(c1.count(","))
    out["v_c1_type_token_ratio"] = float(_ttr(cw))
    out["v_c1_numeral_density"] = float(
        sum(ch.isdigit() for ch in c1) / (len(c1) + 1))
    out["v_c1_avg_word_len"] = float(
        sum(len(w) for w in cw) / max(len(cw), 1))
    out["v_c1_long_word_share"] = float(
        sum(len(w) >= 9 for w in cw) / max(len(cw), 1))

    # ---- abstract ------------------------------------------------------
    aw = _toks(ab)
    a_letters = [ch for ch in ab if ch.isalpha()]
    n_sent = max(len(SENT_END_RE.findall(ab)), 1 if ab else 0)
    out["v_abs_word_count"] = float(len(aw))
    out["v_abs_log_len"] = float(math.log1p(len(ab)))
    out["v_abs_sentence_count"] = float(n_sent)
    out["v_abs_avg_sentence_words"] = float(len(aw) / max(n_sent, 1))
    out["v_abs_type_token_ratio"] = float(_ttr(aw))
    out["v_abs_numeral_density"] = float(
        sum(ch.isdigit() for ch in ab) / (len(ab) + 1))
    out["v_abs_alpha_share"] = float(len(a_letters) / (len(ab) + 1))
    out["v_abs_uppercase_letter_ratio"] = float(
        sum(ch.isupper() for ch in a_letters) / max(len(a_letters), 1))
    out["v_abs_avg_word_len"] = float(
        sum(len(w) for w in aw) / max(len(aw), 1))

    # ---- title ---------------------------------------------------------
    tw = _toks(t)
    t_letters = [ch for ch in t if ch.isalpha()]
    out["v_title_word_count"] = float(len(tw))
    out["v_title_char_count"] = float(len(t))
    out["v_title_avg_word_len"] = float(
        sum(len(w) for w in tw) / max(len(tw), 1))
    out["v_title_has_and"] = float(bool(re.search(r"\band\b", t, re.I)))
    out["v_title_uppercase_ratio"] = float(
        sum(ch.isupper() for ch in t_letters) / max(len(t_letters), 1))

    # ---- cross-field ----------------------------------------------------
    st, sa, sc = ({w.lower() for w in tw}, {w.lower() for w in aw},
                  {w.lower() for w in cw})
    out["v_abs_to_claim_word_ratio"] = float(len(aw) / (len(cw) + 1))
    out["v_title_claim_word_overlap"] = float(
        len(st & sc) / max(len(st), 1))
    out["v_abs_claim_word_overlap"] = float(
        len(sa & sc) / max(len(sa | sc), 1))
    out["v_doc_total_word_count"] = float(len(tw) + len(aw) + len(cw))

    ordered = {k: out[k] for k in V_NAMES}
    assert list(ordered) == V_NAMES, "V_NAMES / v_features key mismatch"
    assert all(math.isfinite(v) for v in ordered.values()), \
        [k for k, v in ordered.items() if not math.isfinite(v)]
    return ordered


def vector(title: str, abstract: str, claim1: str,
           names: Iterable[str] = V_NAMES) -> List[float]:
    vals = v_features(title, abstract, claim1)
    return [vals[k] for k in names]


if __name__ == "__main__":
    import json
    demo = v_features(
        "Coherent LADAR using intra-pixel quadrature detection",
        "A frequency modulated coherent laser detection and ranging system "
        "includes a detector array. In one embodiment the system improves "
        "range resolution substantially.",
        "1. A ladar system comprising: a laser source configured to emit a "
        "frequency modulated optical signal; a detector array (12) having a "
        "plurality of pixels, wherein each pixel includes at least one "
        "quadrature detection circuit; and a processor coupled to said "
        "detector array.")
    print(json.dumps(demo, indent=1))
    print(f"n_features = {len(V_NAMES)}")
