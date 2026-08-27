#!/usr/bin/env python
"""MECHBAT: mechanical-capability battery (user 2026-07-08 night: "test a lot more calibration
metrics... nail down what the LLM can/cannot do that code CAN, when it comes to just
scanning / regexing"). ~34 code-verifiable rules per task spanning:
  presence_char / presence_word(dynamic mid-freq) / casing / pattern / length / count /
  position(prefix|suffix-local) / negation / parity(hard) / order(two-token relation)
Truth is computed BY CODE on the SHOWN slice text[:max_text_chars] (the executor's view —
lesson from the truncation audit). Rules with truth base-rate outside [.15,.85] on the probe
set are dropped (balanced-acc needs both classes). Output freeze is zxa_glm.py- and
osl_sweep-compatible: freeze_zxa_mechbat_<task>_v1.json (entries carry mech.truth inline →
self-contained; scorer never recomputes).
Run ON sk3: python build_mechbat.py humor peer_review
"""
import json
import re
import sys

import numpy as np

B = "/lfs/skampere3/0/alexspan"
sys.path.insert(0, f"{B}/norm-research")
OM = f"{B}/outputs/osl_multi"


def words(t):
    return re.findall(r"[A-Za-z']+", t)


def build_rules(probes):
    """Return list of (id, family, locality, statement, truth_fn)."""
    R = []

    def add(rid, fam, loc, stmt, fn):
        R.append((rid, fam, loc, stmt, fn))

    # --- presence: single characters -----------------------------------------------------
    for ch, nm in [("?", "question mark"), ("!", "exclamation mark"), (":", "colon"),
                   ("%", "percent sign"), ("(", "opening parenthesis")]:
        add(f"presence-{nm.replace(' ', '')}", "presence_char", "global",
            f"This text contains at least one {nm} ('{ch}').", lambda t, c=ch: c in t)
    add("presence-digit", "presence_char", "global",
        "This text contains at least one digit (0-9).", lambda t: bool(re.search(r"\d", t)))
    add("presence-quote", "presence_char", "global",
        "This text contains quoted speech (a quotation mark).",
        lambda t: bool(re.search(r'["“”]', t)))
    # --- presence: dynamic mid-frequency words -------------------------------------------
    from collections import Counter
    df = Counter()
    for t in probes:
        df.update({w.lower() for w in words(t) if len(w) >= 3})
    n = len(probes)
    cands = [(w, c / n) for w, c in df.items() if 0.25 <= c / n <= 0.75]
    cands.sort(key=lambda x: abs(x[1] - 0.5))
    for i, (w, r) in enumerate(cands[:4]):
        add(f"presence-word-{w}", "presence_word", "global",
            f"This text contains the word '{w}'.",
            lambda t, w=w: bool(re.search(rf"\b{re.escape(w)}\b", t, re.I)))
    # --- casing ---------------------------------------------------------------------------
    add("casing-allcaps", "casing", "global",
        "This text contains a word written in ALL CAPITAL LETTERS that is at least 3 letters long.",
        lambda t: bool(re.search(r"\b[A-Z]{3,}\b", t)))
    add("casing-two-capitalized", "casing", "global",
        "This text contains two consecutive capitalized words (each starting with a capital letter).",
        lambda t: bool(re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", t)))
    # --- pattern --------------------------------------------------------------------------
    add("pattern-4digit", "pattern", "global",
        "This text contains a number written with exactly four digits in a row (like a year).",
        lambda t: bool(re.search(r"(?<!\d)\d{4}(?!\d)", t)))
    add("pattern-url", "pattern", "global",
        "This text contains 'http' or 'www'.", lambda t: ("http" in t.lower()) or ("www" in t.lower()))
    add("pattern-hyphenated", "pattern", "global",
        "This text contains a hyphenated word (a hyphen directly between two letters).",
        lambda t: bool(re.search(r"[A-Za-z]-[A-Za-z]", t)))
    # --- length / aggregate medians on SHOWN text -----------------------------------------
    wc = sorted(len(words(t)) for t in probes)
    kw = wc[len(wc) // 2]
    cc = sorted(len(t) for t in probes)
    kc = cc[len(cc) // 2]
    pp = sorted(t.count(".") for t in probes)
    kp = pp[len(pp) // 2]
    km = sorted(t.count(",") for t in probes)[len(probes) // 2]
    add("length-words", "length", "aggregate",
        f"This text is longer than {kw} words.", lambda t, k=kw: len(words(t)) > k)
    add("length-chars", "length", "aggregate",
        f"This text is longer than {kc} characters (counting spaces).", lambda t, k=kc: len(t) > k)
    add("length-periods", "length", "aggregate",
        f"This text contains more than {kp} period characters ('.').",
        lambda t, k=kp: t.count(".") > k)
    # --- counting -------------------------------------------------------------------------
    add("count-2qmark", "count", "aggregate",
        "This text contains at least two question marks.", lambda t: t.count("?") >= 2)
    add("count-3digit", "count", "aggregate",
        "This text contains at least three digit characters (0-9) in total.",
        lambda t: len(re.findall(r"\d", t)) >= 3)
    add("count-commas", "count", "aggregate",
        f"This text contains more than {km} commas.", lambda t, k=km: t.count(",") > k)
    if cands:
        w0 = cands[0][0]
        add(f"count-word2-{w0}", "count", "aggregate",
            f"The word '{w0}' appears at least twice in this text.",
            lambda t, w=w0: len(re.findall(rf"\b{re.escape(w)}\b", t, re.I)) >= 2)
    # --- position-local -------------------------------------------------------------------
    add("pos-digit-first30", "position", "prefix",
        "Within the first 30 words of this text there is at least one digit (0-9).",
        lambda t: bool(re.search(r"\d", " ".join(words(t)[:30]))))
    add("pos-qmark-last30", "position", "suffix",
        "Within the last 30 words of this text there is at least one question mark.",
        lambda t: "?" in " ".join(re.split(r"\s+", t.strip())[-30:]))
    if len(cands) > 1:
        w1 = cands[1][0]
        add(f"pos-word-first50-{w1}", "position", "prefix",
            f"The word '{w1}' appears within the first 50 words of this text.",
            lambda t, w=w1: bool(re.search(rf"\b{re.escape(w)}\b", " ".join(words(t)[:50]), re.I)))
    add("pos-ends-terminal", "position", "suffix",
        "The very last non-space character of this text is a period, question mark, or exclamation mark.",
        lambda t: t.strip()[-1:] in ".?!")
    # --- negation -------------------------------------------------------------------------
    add("neg-noqmark", "negation", "global",
        "This text does NOT contain any question mark.", lambda t: "?" not in t)
    add("neg-nodigit", "negation", "global",
        "This text does NOT contain any digit (0-9).", lambda t: not re.search(r"\d", t))
    add("neg-noexclaim", "negation", "global",
        "This text does NOT contain any exclamation mark.", lambda t: "!" not in t)
    # --- parity (deliberately hard for scanners-without-state) ----------------------------
    add("parity-qmark", "parity", "aggregate",
        "This text contains an even number of question marks (zero counts as even).",
        lambda t: t.count("?") % 2 == 0)
    add("parity-digit", "parity", "aggregate",
        "This text contains an even number of digit characters (zero counts as even).",
        lambda t: len(re.findall(r"\d", t)) % 2 == 0)
    # --- order (two-token relations) -------------------------------------------------------
    def first_pos(t, pat):
        m = re.search(pat, t)
        return m.start() if m else None

    def order_digit_qmark(t):
        d = first_pos(t, r"\d")
        if d is None:
            return False
        q = t.find("?")
        return True if q == -1 else d < q
    add("order-digit-before-qmark", "order", "global",
        "This text contains at least one digit, and the first digit appears earlier in the text "
        "than the first question mark (if there is no question mark at all, answer YES as long "
        "as there is a digit).", order_digit_qmark)
    if len(cands) > 3:
        wa, wb = cands[2][0], cands[3][0]
        def order_words(t, wa=wa, wb=wb):
            a = re.search(rf"\b{re.escape(wa)}\b", t, re.I)
            if a is None:
                return False
            b = re.search(rf"\b{re.escape(wb)}\b", t, re.I)
            return True if b is None else a.start() < b.start()
        add(f"order-{wa}-before-{wb}", "order", "global",
            f"This text contains the word '{wa}', and its first occurrence appears earlier in "
            f"the text than the first occurrence of the word '{wb}' (if '{wb}' is absent, answer "
            f"YES as long as '{wa}' is present).", order_words)
    return R


def main(task):
    from methods.metric_implementer.experiments.run_real_test import _load_texts
    from methods.metric_implementer import config as cfgmod
    v2 = json.load(open(f"{OM}/freeze_{task}_v2.json"))
    meta = v2["meta"]
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), meta["task"])
    n = int(meta["n_probes"])
    texts, _ = _load_texts(meta["task"], 60 + n, cfg)
    shown = [t[: cfg.max_text_chars] for t in texts[60: 60 + n]]
    rules = build_rules(shown)
    entries, dropped = [], []
    for rid, fam, loc, stmt, fn in rules:
        truth = [bool(fn(t)) for t in shown]
        r = float(np.mean(truth))
        if not (0.15 <= r <= 0.85):
            dropped.append((rid, round(r, 3)))
            continue
        entries.append({"name": f"MECHBAT-{rid}", "kind": f"mechbat|{fam}|{loc}",
                        "rubric": stmt, "criteria": [],
                        "mech": {"family": fam, "locality": loc, "base_rate": round(r, 3),
                                 "truth": [int(x) for x in truth]}})
    out = {"meta": {**meta, "mechbat": {"v": 1, "n_rules": len(entries),
                                        "truth_on": "shown slice text[:max_text_chars]"}},
           "metrics": entries}
    path = f"{OM}/freeze_zxa_mechbat_{task}_v1.json"
    json.dump(out, open(path, "w"), indent=1)
    fams = {}
    for e in entries:
        fams[e["mech"]["family"]] = fams.get(e["mech"]["family"], 0) + 1
    print(f"{task}: {len(entries)} rules kept ({fams}); dropped extreme-base-rate: {dropped}")
    print(f"  -> {path}")


if __name__ == "__main__":
    for task in (sys.argv[1:] or ["humor", "peer_review"]):
        main(task)
