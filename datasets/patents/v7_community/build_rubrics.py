#!/usr/bin/env python3
"""V7 patents forward-citation cell: merge the four label-blind proposer batches
into the final A bank (`rubrics.jsonl`).

PROVENANCE. Four proposer subagents each read a DISJOINT batch of 14 real
TRAIN-SPLIT patents (title + abstract + claim 1, no label attached, sampled in
sha256("v7-mine|" + patent_id) order -- see v7_exemplars.py) and proposed 17
candidates each, self-labelling every one Track A (a real substantive quality
property) or Track B (a declared surface correlate we expect to be nuisance).
Each proposer was given a different angle so the four batches would not collapse
onto the same ideas:
    b0  claim definiteness / 112-style disclosure quality
    b1  relationship between title, abstract and claim; contribution framing
    b2  claim-construction craft and the boundary the claim draws
    b3  inventive substance and the reach of the disclosure
The raw proposals are kept verbatim in `proposals_b{0,1,2,3}.jsonl` beside this
file, and every surviving entry carries its source batch in `origin`.

THIS SCRIPT IS THE MERGE + AUDIT + GEPA-PHRASING STEP. It is deterministic and
takes no arguments:

  1. METADATA BAN (hard, non-negotiable). The sibling patents cell (claim-fell)
     was closed as a metadata-leak post-mortem: examiner-identity and
     metadata-no-text instruments reached .756 by leakage. Any criterion whose
     text mentions citations, owner/assignee/inventor, examiner or art unit,
     litigation, fees/renewals, dates, patent numbers, CPC codes, families or
     continuations, market data, or grant outcomes is DROPPED here, with the
     offending term recorded. A judge cannot leak what it is never asked about.
  2. CIRCULARITY BAN. Anything asking the judge to predict importance, value,
     influence, adoption or how often the patent was later cited is dropped:
     that is the quantity under test.
  3. DEDUP. Near-duplicate concepts across batches are collapsed (token-overlap
     on name + description); every drop is logged with the survivor it merged
     into.
  4. GEPA PHRASING REWRITE. The scored `description` is rebuilt to a fixed
     shape carrying the two-sided match test and a real NA branch, because the
     judge sees only `name` + `description`:
         <property statement>
         HIGH (1.0): ...  LOW (0.0): ...  NA: ...
     The five phrasing rules enforced (same rules as the so_votes bank):
       (1) judgeable from title + abstract + claim 1 alone;
       (2) never references any excluded metadata or any outcome;
       (3) a two-sided match test, not "more is better", so it cannot be read
           off length or formatting;
       (4) an NA branch that is a genuine "attempts nothing bearing on this"
           case rather than a synonym for 0.0;
       (5) one property per criterion.

  python3 datasets/patents/v7_community/build_rubrics.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "rubrics.jsonl"
AUDIT = HERE / "rubrics_audit.json"

# --- 1/2. the two hard bans ---------------------------------------------------
BAN_METADATA = re.compile(
    r"\b(?:citation|cited|cites|assignee|owner|company|corporat|inventor|"
    r"attorney|examiner|art unit|litigat|renewal|maintenance fee|"
    r"filing date|priority date|grant date|patent number|cpc|classification code|"
    r"family|continuation|divisional|market|revenue|licens|portfolio|"
    r"prosecution|office action|allowance|granted or rejected)\b", re.I)
BAN_CIRCULAR = re.compile(
    r"\b(?:important|importance|valuable|value of|influential|influence|impact|"
    r"widely (?:used|adopted)|adoption|commercial success|significan(?:ce|t) of "
    r"the invention|how often|later cited|downstream)\b", re.I)

STOP = set("the a an of to and or in on for with that this is are be by as it "
           "its whether versus vs not no any all its from at into than then "
           "claim claims patent abstract document text one two".split())

# --- 3b. CURATED SEMANTIC MERGE ----------------------------------------------
# Token-overlap dedup collapses only near-identical WORDING; the four proposers
# were given different angles and re-derived the same concepts in different
# words, so it fired 0 times on 68 raw candidates. These merges were made by
# reading all 66 survivors. Key: dropped name -> (survivor kept, reason).
# Nothing here is dropped for being weak; only for being a restatement, for
# being so rare that the column would be near-constant, or to hold the judge
# budget at the program's usual scale (~600K calls).
CURATED_MERGE = {
    # antecedent basis, proposed twice
    "Antecedent Basis Clarity": ("Antecedent Basis Discipline", "same concept"),
    # functional language needing structural backing, proposed three times
    "Bare Functional Recitation":
        ("Functional Language Grounded In Structure", "same concept"),
    "Structural Backing For Functional Language":
        ("Functional Language Grounded In Structure", "same concept"),
    # element interdependency, proposed three times
    "Structural Interdependency": ("Interdependent Limitations", "same concept"),
    "Claim States Interactive Relationships":
        ("Interdependent Limitations", "same concept"),
    # claim scope vs the disclosed contribution, proposed three times
    "Claim Breadth Matches Abstract Framing":
        ("Claim Scope Matches Disclosed Contribution", "same concept"),
    "Claim Scope Alignment With Abstract":
        ("Claim Scope Matches Disclosed Contribution", "same concept"),
    # causal mechanism, proposed at three loci; the abstract-side (a08) and the
    # why-side (a53) are kept as genuinely different questions, the claim-side
    # restatement is not
    "Mechanism Identified Not Just Result":
        ("Causal Mechanism Articulation", "same concept, claim-side restatement"),
    "Structure To Effect Linkage":
        ("Causal Mechanism Articulation", "whereby/thereby variant of the same"),
    # definiteness of degree terms, proposed twice
    "Definiteness Of Claim Terms": ("Relative Term Anchoring", "same concept"),
    # numeric rationale, proposed twice
    "Numeric Threshold Tied To Rationale":
        ("Numeric Parameters Tied To Function", "same concept"),
    # single inventive concept / necessity, proposed twice each
    "Concentrated Locus Of Invention":
        ("Single Inventive Concept Coherence", "same concept"),
    "Necessity Of Recited Limitations": ("Orphan Element Test", "same concept"),
    "Claim Free Of Undisclosed Core Limitation":
        ("Claim Scope Matches Disclosed Contribution", "same concept"),
    "Explicit Engineering Trade Off":
        ("Competing Constraint Resolution", "same concept"),
    "Jargon Explained In Context": ("In Claim Term Definition", "same concept"),
    "Conditional Logic Specificity":
        ("Condition Responsive Behavior", "narrower case of the same"),
    "Abstract Plain Register":
        ("Said/Aforementioned Legalese Register", "same register axis"),
    "Open-Ended Quantifier Boundedness":
        ("Claim Connective Density", "same claim-shape axis"),
    "Non-Redundant Limitation Recitation":
        ("Orphan Element Test", "same concept"),
}
# Dropped as too rare to carry variance (the column would be near-constant or
# almost entirely NA), or as pure grammar checks that granted claims all pass.
CURATED_RARE = {
    "Single Sentence Resolution": "granted claims essentially all pass; near-constant",
    "Explicit Exclusionary Limitation": "negative limitations are rare; near-constant 0",
    "Alternative Operating Mode": "rare; near-constant 0",
    "Index Variable Relationship Definiteness": "indexed claims are rare; ~all NA",
    "Markush Style Alternative Listing": "rare outside chemistry; near-constant 0",
    "Formal Notation For Key Limitation": "rare; near-constant 0",
    "Preamble States Application Context":
        "budget trim; overlaps Preamble Definitional Work",
    "Novelty In Structure Not Field Of Use":
        "budget trim; overlaps Concrete Versus Abstract Boundary",
    "Title Names The Mechanism": "budget trim; keeps one title-side Track B probe",
}


def toks(s):
    return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if w not in STOP}


def main():
    props, audit = [], {"dropped": [], "merged": [], "kept": []}
    for b in range(4):
        p = HERE / f"proposals_b{b}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        for ln in p.read_text().splitlines():
            if ln.strip():
                d = json.loads(ln)
                d["origin"] = f"proposer batch {b}"
                props.append(d)
    print(f"raw proposals: {len(props)}")

    # --- bans ---
    survivors = []
    for d in props:
        blob = " ".join(str(d.get(k, "")) for k in
                        ("name", "description", "high", "low", "na"))
        m1, m2 = BAN_METADATA.search(blob), BAN_CIRCULAR.search(blob)
        if m1:
            audit["dropped"].append({"name": d["name"], "reason": "metadata ban",
                                     "term": m1.group(0), "origin": d["origin"]})
        elif m2:
            audit["dropped"].append({"name": d["name"], "reason": "circular with y",
                                     "term": m2.group(0), "origin": d["origin"]})
        else:
            survivors.append(d)
    print(f"after metadata + circularity bans: {len(survivors)} "
          f"(dropped {len(audit['dropped'])})")

    # --- curated semantic merge (3b), applied before the lexical pass ---
    staged = []
    for d in survivors:
        nm = d["name"]
        if nm in CURATED_MERGE:
            into, why = CURATED_MERGE[nm]
            audit["merged"].append({"dropped": nm, "into": into,
                                    "reason": f"curated: {why}",
                                    "origin": d["origin"]})
        elif nm in CURATED_RARE:
            audit["dropped"].append({"name": nm, "reason": "curated: low variance",
                                     "term": CURATED_RARE[nm],
                                     "origin": d["origin"]})
        else:
            staged.append(d)
    missing = ([n for n in CURATED_MERGE if n not in {d["name"] for d in survivors}]
               + [n for n in CURATED_RARE if n not in {d["name"] for d in survivors}])
    assert not missing, f"curation names not found in proposals: {missing}"
    for _, (into, _) in CURATED_MERGE.items():
        assert into in {d["name"] for d in staged}, f"merge survivor missing: {into}"
    print(f"after curated merge: {len(staged)}")
    survivors = staged

    # --- lexical dedup (catches near-identical wording the curation missed) ---
    kept = []
    for d in survivors:
        tn, td = toks(d["name"]), toks(d["name"] + " " + d["description"])
        dup = None
        for k in kept:
            kn, kd = toks(k["name"]), toks(k["name"] + " " + k["description"])
            jn = len(tn & kn) / max(len(tn | kn), 1)
            jd = len(td & kd) / max(len(td | kd), 1)
            if jn >= 0.6 or jd >= 0.5:
                dup = k
                break
        if dup is None:
            kept.append(d)
        else:
            audit["merged"].append({"dropped": d["name"], "into": dup["name"],
                                    "origin": d["origin"]})
    print(f"after dedup: {len(kept)} (merged {len(audit['merged'])})")

    # --- GEPA phrasing rewrite ---
    rubrics = []
    na_ctr = 0
    for i, d in enumerate(kept):
        track = "B" if str(d.get("track", "A")).upper().startswith("B") else "A"
        pre = "s" if track == "B" else "a"
        prop = re.sub(r"\s+", " ", d["description"]).strip().rstrip(".")
        high = re.sub(r"\s+", " ", d["high"]).strip().rstrip(".")
        low = re.sub(r"\s+", " ", d["low"]).strip().rstrip(".")
        na = re.sub(r"\s+", " ", d["na"]).strip().rstrip(".")
        # rule 4: an NA branch that merely restates the 0.0 branch is not a real
        # NA; flag it rather than silently shipping it.
        if len(toks(na) & toks(low)) / max(len(toks(na) | toks(low)), 1) > 0.72:
            na_ctr += 1
            audit["dropped"].append({"name": d["name"],
                                     "reason": "NA branch duplicates the 0.0 branch",
                                     "origin": d["origin"]})
            continue
        desc = (f"{prop}. HIGH (1.0): {high}. LOW (0.0): {low}. "
                f"NA: {na}. Judge only from the title, abstract and claim 1 shown; "
                f"do not reward or penalise raw length.")
        rubrics.append({
            "rubric_id": f"{pre}{len(rubrics) + 1:02d}",
            "name": d["name"], "description": desc, "track": track,
            "origin": d["origin"], "proposer_rationale": d.get("rationale", ""),
            "gepa_revision": ("rebuilt to the fixed two-sided shape "
                              "<property> / HIGH / LOW / NA, with the explicit "
                              "'judge only from the shown fields, do not reward "
                              "length' clause appended (phrasing rules 1-5)"),
        })
        audit["kept"].append({"rubric_id": rubrics[-1]["rubric_id"],
                              "name": d["name"], "track": track,
                              "origin": d["origin"]})

    with open(OUT, "w") as f:
        for r in rubrics:
            f.write(json.dumps(r) + "\n")
    nA = sum(r["track"] == "A" for r in rubrics)
    audit["summary"] = {"raw": len(props), "after_bans": len(survivors),
                        "after_dedup": len(kept),
                        "dropped_fake_na": na_ctr,
                        "final": len(rubrics), "track_A": nA,
                        "track_B": len(rubrics) - nA}
    AUDIT.write_text(json.dumps(audit, indent=2))
    print(f"\nFINAL BANK: {len(rubrics)} criteria ({nA} Track A / "
          f"{len(rubrics) - nA} Track B)")
    print("wrote", OUT, "and", AUDIT)
    for r in rubrics:
        print(f"  {r['rubric_id']} [{r['track']}] {r['name']}")


if __name__ == "__main__":
    main()
