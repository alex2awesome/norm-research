#!/usr/bin/env python3
"""Audit patents_first_draft_with_applicant_cites_balanced.csv.gz for leakage.

Checks:
 1. Length features by label.
 2. Distinctive section markers — does the text reveal whether IDS cites
    are present at all (the citation enrichment may be label-leaky if
    citation count or presence correlates with outcome)?
 3. Top distinguishing trigrams by label.
 4. Specific patent boilerplate / shortcut features (CLAIMS, ABSTRACT,
    "claim 1 of US ...", citation patterns).
 5. Near-duplicates via shingle Jaccard on 1K sample (catches CPC-family
    duplicates and continuation/divisional filings).
 6. Class-conditional citation counts (the *number* of cited references
    may itself be a confound — apps with more art tend to be harder
    to approve first-draft).
 7. Length of the "APPLICANT CITES" suffix by label.
"""
import csv
import gzip
import hashlib
import math
import re
import statistics
from collections import Counter, defaultdict

csv.field_size_limit(2**31 - 1)

INPUT = ("/lfs/skampere3/0/alexspan/norm-research/datasets/patents/"
         "patents_first_draft_with_applicant_cites_balanced.csv.gz")


def split_text(t):
    """Splits text on the applicant-cites section header if present."""
    # Common variants — find whichever section divider is used
    parts = re.split(r"\n\nAPPLICANT[_ ]CITE[SD]?:?\n", t, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0], parts[1]
    # Fallback: look for "CITED REFERENCES" or "REFERENCES"
    parts = re.split(r"\n\nCITED[ _]REFERENCES?:?\n", t, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0], parts[1]
    return t, ""


def count_citations(cites_section):
    """Counts likely citation entries — patent numbers and DOIs."""
    if not cites_section:
        return 0
    # Patent numbers: e.g., US12345678, US 12,345,678 B2, 12,345,678
    pats = re.findall(r"\b(?:US|EP|WO|JP|CN)[\s-]?\d{4,}[A-Z]?\d?\b", cites_section)
    return len(pats)


def main():
    rows = []
    with gzip.open(INPUT, "rt") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            t = r["text"]
            main_text, cites = split_text(t)
            rows.append({
                "text": t,
                "main": main_text,
                "cites": cites,
                "label": int(r["judgement"]),
            })
    n = len(rows)
    pos = sum(r["label"] for r in rows)
    print(f"Total rows: {n}")
    print(f"Positives (first-draft approved): {pos} ({pos/n*100:.1f}%)")
    print()

    # Inspect splitter behavior on first 3 rows
    print("--- 0. Sample row structure ---")
    for i in range(min(3, n)):
        r = rows[i]
        print(f"row {i}: label={r['label']}, total_len={len(r['text'])}, "
              f"main_len={len(r['main'])}, cites_len={len(r['cites'])}")
        print(f"  text head 300: {repr(r['text'][:300])}")
        print(f"  text tail 300: {repr(r['text'][-300:])}")
    print()

    print("--- 1. Length by label ---")
    for label in (0, 1):
        s = [r for r in rows if r["label"] == label]
        full = [len(r["text"]) for r in s]
        main = [len(r["main"]) for r in s]
        cites = [len(r["cites"]) for r in s]
        print(f"  label={label}: n={len(s)}")
        print(f"    full chars: mean={statistics.mean(full):.0f} med={statistics.median(full):.0f}")
        print(f"    main chars: mean={statistics.mean(main):.0f} med={statistics.median(main):.0f}")
        print(f"    cites chars: mean={statistics.mean(cites):.0f} med={statistics.median(cites):.0f}")
    print()

    print("--- 2. Citation count by label ---")
    by_label_cite_counts = defaultdict(list)
    for r in rows:
        c = count_citations(r["cites"])
        by_label_cite_counts[r["label"]].append(c)
    for label in (0, 1):
        cs = by_label_cite_counts[label]
        if cs:
            print(f"  label={label}: mean={statistics.mean(cs):.2f}  med={statistics.median(cs)}  "
                  f"p90={sorted(cs)[int(0.9*len(cs))]}  zero_rate={sum(1 for x in cs if x == 0)/len(cs):.3f}")
    print()

    print("--- 3. Section-header presence by label ---")
    headers = {
        "ABSTRACT":          re.compile(r"\bABSTRACT\b"),
        "CLAIMS":            re.compile(r"\bCLAIMS?\b"),
        "BACKGROUND":        re.compile(r"\bBACKGROUND\b"),
        "SUMMARY":           re.compile(r"\bSUMMARY\b"),
        "APPLICANT_CITES":   re.compile(r"\bAPPLICANT[ _]CITE[SD]?\b", re.IGNORECASE),
        "CITED_REFERENCES":  re.compile(r"\bCITED[ _]REFERENCES?\b", re.IGNORECASE),
        "REFERENCES":        re.compile(r"\bREFERENCES?\b"),
        "DETAILED":          re.compile(r"\bDETAILED\s+DESCRIPTION\b"),
    }
    n_pos = pos; n_neg = n - pos
    print(f"  {'header':18s} | {'pos':>8s} | {'neg':>8s} | {'lift':>6s}")
    for name, pat in headers.items():
        np_ = sum(1 for r in rows if r["label"] == 1 and pat.search(r["text"]))
        nn_ = sum(1 for r in rows if r["label"] == 0 and pat.search(r["text"]))
        p1, p0 = np_ / n_pos, nn_ / n_neg
        lift = (p1 - p0) / (p1 + p0) if (p1 + p0) else 0
        print(f"  {name:18s} | {np_:>8d} | {nn_:>8d} | {lift:+.3f}")
    print()

    print("--- 4. Top distinguishing trigrams (lift) ---")
    c0, c1 = Counter(), Counter()
    # Use only the MAIN text (excluding cites) so we don't conflate citation tokens
    for r in rows:
        toks = re.findall(r"\b[a-z]+\b", r["main"].lower())
        for k in range(len(toks) - 2):
            tg = " ".join(toks[k:k + 3])
            if r["label"] == 1:
                c1[tg] += 1
            else:
                c0[tg] += 1
    total1, total0 = sum(c1.values()), sum(c0.values())
    candidates = [g for g, c in c1.items() if c > 200 and c0.get(g, 0) > 200]
    scored = []
    for g in candidates:
        p1 = c1[g] / total1
        p0 = c0[g] / total0
        if p0 == 0: continue
        lift = (p1 - p0) / (p1 + p0)
        scored.append((lift, g, c1[g], c0.get(g, 0)))
    scored.sort()
    print("  Top neg-leaning trigrams (label=0 → NOT first-draft approved):")
    for s, g, n1, n0 in scored[:10]:
        print(f"    {s:+.3f} '{g}' pos={n1} neg={n0}")
    print("  Top pos-leaning trigrams (label=1 → first-draft approved):")
    for s, g, n1, n0 in scored[-10:][::-1]:
        print(f"    {s:+.3f} '{g}' pos={n1} neg={n0}")
    print()

    print("--- 5. Exact text duplicates ---")
    text_hash = Counter()
    for r in rows:
        text_hash[hashlib.md5(r["text"].encode("utf-8")).hexdigest()] += 1
    dupes = sum(c - 1 for c in text_hash.values() if c > 1)
    print(f"Exact dup rows: {dupes} ({dupes/n*100:.2f}%)")
    # Same main, different cites? (continuations / divisionals)
    main_hash = Counter()
    for r in rows:
        main_hash[hashlib.md5(r["main"][:5000].encode("utf-8")).hexdigest()] += 1
    main_dupes = sum(c - 1 for c in main_hash.values() if c > 1)
    print(f"Same-main-text-prefix rows (5K-char prefix): {main_dupes} ({main_dupes/n*100:.2f}%)")
    print()

    print("--- 6. Near-duplicate via 9-char shingle Jaccard (1K sample) ---")
    import random
    rng = random.Random(0)
    sample = rng.sample(rows, min(1000, n))
    def shingles(s, k=9):
        s = re.sub(r"\s+", " ", s).strip()
        return set(s[i:i + k] for i in range(0, max(1, len(s) - k + 1)))
    sl = [shingles(r["main"][:2000]) for r in sample]
    inv = defaultdict(list)
    for i, sh in enumerate(sl):
        for s in sh:
            inv[s].append(i)
    high_overlap = 0
    cross_label_overlap = 0
    for i, sh in enumerate(sl):
        if not sh: continue
        cand = Counter()
        for s in sh:
            for j in inv[s]:
                if j > i:
                    cand[j] += 1
        for j, common in cand.most_common(5):
            sj = sl[j]
            if not sj: continue
            jac = common / max(1, len(sh | sj))
            if jac > 0.5:
                high_overlap += 1
                if sample[i]["label"] != sample[j]["label"]:
                    cross_label_overlap += 1
    print(f"  Pairs with main-text Jaccard>0.5: {high_overlap}, of which cross-label: {cross_label_overlap}")
    print()


if __name__ == "__main__":
    main()
