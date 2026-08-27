#!/usr/bin/env python3
"""Build the blind pairwise concept-identity adjudication batch for the round-0
bank census, with an authored anchor battery.

Anchors follow the missing-mass note's PART-4 fix (recommendation 5): the SAME
anchors are DELIBERATELY AUTHORED PARAPHRASE PAIRS, not high-cosine pairs (the
pilot's cosine-derived "SAME" anchors were rejected by both judges and taught
nothing).  The DIFFERENT anchors are lexical look-alikes whose concepts genuinely
differ.  Anchors are interleaved with the real pairs by stable hash and carry the
same id scheme, so the judge cannot tell them apart.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

ANCHORS = [
    # --- authored SAME (paraphrase of one concept in two registers) -------------
    {"label": "SAME",
     "X": {"name": "Quantified compliance-cost estimate supplied by the commenter",
           "description": "The comment supplies its own numeric estimate of what complying with "
                          "the proposed rule would cost, with the basis for the number stated."},
     "Y": {"name": "Commenter provides own dollar figures for cost of compliance",
           "description": "The submission puts forward concrete monetary compliance-cost numbers "
                          "of its own and explains where those figures come from."}},
    {"label": "SAME",
     "X": {"name": "Proposes a specific alternative regulatory approach",
           "description": "The comment names a concrete different way the agency could achieve "
                          "the rule's objective, described specifically enough to be adopted."},
     "Y": {"name": "Offers a concrete substitute policy design the agency could adopt",
           "description": "Rather than only objecting, the submission sets out an actionable "
                          "alternative design for the rule that the agency could implement."}},
    # --- authored DIFFERENT (lexical look-alikes, different concepts) -----------
    {"label": "DIFFERENT",
     "X": {"name": "Cites the Code of Federal Regulations by part and section",
           "description": "The comment contains explicit CFR citations identifying the exact "
                          "provisions it addresses."},
     "Y": {"name": "Argues the rule exceeds the agency's statutory authority",
           "description": "The comment makes a legal argument that the proposal is not authorized "
                          "by the statute the agency relies on."}},
    {"label": "DIFFERENT",
     "X": {"name": "Reports first-hand operational experience in the regulated activity",
           "description": "The commenter describes what they themselves do in the regulated "
                          "sector and what the rule would change about it."},
     "Y": {"name": "Expresses strong personal feeling about the proposal",
           "description": "The comment conveys the intensity of the commenter's approval or "
                          "disapproval, independent of any evidence offered."}},
]

INSTRUCTION = """You are auditing a bank of scoring criteria written for judging PUBLIC COMMENTS
submitted to United States federal agencies on proposed rules.

For each PAIR below, decide ONE question:

  Would an independent judge, scoring public comments against criterion X and against
  criterion Y, be measuring THE SAME UNDERLYING CONCEPT?

Answer SAME only if the two criteria would produce essentially interchangeable scores
because they name one concept in different words. Answer DIFFERENT if a comment could
plausibly score high on one and low on the other -- including when the two are closely
related, overlapping, or members of the same topic family. Related is not the same.

Be strict. Two criteria about the same regulatory topic are DIFFERENT if they target
different properties of the comment (e.g. "cites the statute" vs "argues the statute is
exceeded").

Emit exactly one JSON object and nothing else:

{"verdicts": [
  {"pair_id": "CP001", "verdict": "SAME" or "DIFFERENT", "confidence": "high"/"medium"/"low",
   "reason": "<one sentence>"},
  ... one entry for EVERY pair, in any order ...
]}
"""


def main():
    real = json.loads((HERE / "census_pairs_blind.json").read_text())["pairs"]
    items = [{"kind": "real", **p} for p in real]
    for n, a in enumerate(ANCHORS):
        flip = int(hashlib.sha256(f"anchor|{n}".encode()).hexdigest(), 16) % 2
        X, Y = (a["X"], a["Y"]) if not flip else (a["Y"], a["X"])
        items.append({"kind": "anchor", "anchor_label": a["label"],
                      "pair_id": f"CP{len(real) + n + 1:03d}", "X": X, "Y": Y})

    items.sort(key=lambda p: hashlib.sha256(f"census-adj|{p['pair_id']}".encode()).hexdigest())
    # renumber so anchor ids are not trivially the last four
    key = []
    for k, it in enumerate(items):
        new = f"CP{k + 1:03d}"
        key.append({"shown_id": new, "orig_id": it["pair_id"], "kind": it["kind"],
                    "anchor_label": it.get("anchor_label")})
        it["shown_id"] = new

    body = INSTRUCTION + "\n\n" + "\n\n".join(
        f"--- PAIR {it['shown_id']} ---\n"
        f"X NAME: {it['X']['name']}\nX DESCRIPTION: {it['X']['description']}\n"
        f"Y NAME: {it['Y']['name']}\nY DESCRIPTION: {it['Y']['description']}"
        for it in items
    ) + "\n"

    (HERE / "census_adjudication_prompt.txt").write_text(body)
    (HERE / "census_adjudication_key.json").write_text(json.dumps({"key": key}, indent=1))
    print(f"{len(items)} pairs ({len(real)} real + {len(ANCHORS)} anchors) "
          f"-> census_adjudication_prompt.txt ({len(body)} chars)")


if __name__ == "__main__":
    main()
