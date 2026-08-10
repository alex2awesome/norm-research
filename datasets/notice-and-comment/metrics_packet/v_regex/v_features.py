#!/usr/bin/env python3
"""Standalone copy of the 27 N&C V (verifiability) regex features.
Extracted verbatim from v4/aggregate_vat_nc.py (2026-07 VAT campaign).
Usage: from v_features import v_features, V_NAMES; feats = v_features(comment_text)
"""
import re

NUMTOK  = re.compile(r"\b\d[\d,\.]*\b")
SENT_RE = re.compile(r"[.!?]+")
KW = {
    "v_kw_cfr":        re.compile(r"\bC\.?\s?F\.?\s?R\.?\b|\bU\.?\s?S\.?\s?C\.?\b|\bFed\.?\s?Reg\b", re.I),
    "v_kw_section":    re.compile(r"§|\bsection\s+\d|\bsubpart\b|\bsubsection\b", re.I),
    "v_kw_docket":     re.compile(r"\bdocket\b|\bRIN\b|federal register|proposed rule|\bNPRM\b", re.I),
    "v_kw_legal":      re.compile(r"arbitrary and capricious|administrative procedure act|\bAPA\b|statutor|\bunlawful\b|exceed.{0,20}authority|\bcongress\b", re.I),
    "v_kw_data":       re.compile(r"\bdata\b|\bstudy\b|\bstudies\b|\bresearch\b|\bevidence\b|\bsurvey\b|peer[- ]reviewed|\banalysis\b", re.I),
    "v_kw_econ":       re.compile(r"\bcost\b|\bbenefit\b|economic|\bburden\b|\$\s?\d|compliance cost|small business", re.I),
    "v_kw_alt":        re.compile(r"alternativ|\binstead\b|we recommend|we suggest|should consider|we propose|we urge", re.I),
    "v_kw_support":    re.compile(r"\bsupport\b|\bagree\b|applaud|commend|\bendorse\b", re.I),
    "v_kw_oppose":     re.compile(r"\boppose\b|\bobject\b|disagree|\breject\b|\bwithdraw\b|rescind", re.I),
    "v_kw_org":        re.compile(r"\bassociation\b|\borganization\b|\bcoalition\b|\binstitute\b|on behalf of|\bmembers\b|\bindustry\b", re.I),
    "v_kw_change_req": re.compile(r"should be|must be|\brevise\b|\bamend\b|\bmodify\b|clarif|\bstrike\b|\bdelete\b", re.I),
    "v_kw_specific":   re.compile(r"\bpage\s?\d|\bp\.\s?\d|\bpart\s+\d|paragraph|\btable\s?\d|\bfigure\s?\d|line\s?\d", re.I),
    "v_kw_procedural": re.compile(r"comment period|\bextension\b|\bextend\b|\bdeadline\b|public hearing", re.I),
    "v_kw_attach":     re.compile(r"\battach\b|attached|enclosed|appendix|exhibit|see enclosure", re.I),
    "v_kw_experience": re.compile(r"\bI am a\b|\bmy experience\b|years of experience|\bas a\b|first[- ]hand", re.I),
    "v_kw_hedge":      re.compile(r"\bmay\b|\bmight\b|\bcould\b|\bperhaps\b|\bpossibly\b|\bconcern\b", re.I),
}
def v_features(text):
    t = text or ""
    words = t.split()
    nw = max(len(words), 1)
    sents = [s for s in SENT_RE.split(t) if s.strip()]
    ns = max(len(sents), 1)
    alpha = [c for c in t if c.isalpha()]
    feats = {
        "v_char_len":     float(len(t)),
        "v_word_len":     float(nw),
        "v_sent_count":   float(ns),
        "v_avg_word_len": float(sum(len(w) for w in words) / nw),
        "v_avg_sent_len": float(nw / ns),
        "v_num_density":  float(100.0 * len(NUMTOK.findall(t)) / nw),
        "v_pct_count":    float(t.count("%")),
        "v_question":     float(t.count("?")),
        "v_exclaim":      float(t.count("!")),
        "v_caps_ratio":   float(sum(c.isupper() for c in alpha) / max(len(alpha), 1)),
        "v_first_person": float(100.0 * len(re.findall(r"\b(I|my|me|mine)\b", t)) / nw),
    }
    for name, rgx in KW.items():
        feats[name] = float(len(rgx.findall(t)))
    return feats

V_NAMES = ["v_char_len","v_word_len","v_sent_count","v_avg_word_len","v_avg_sent_len",
           "v_num_density","v_pct_count","v_question","v_exclaim","v_caps_ratio",
           "v_first_person"] + list(KW.keys())

