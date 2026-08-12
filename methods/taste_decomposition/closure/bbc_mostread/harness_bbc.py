#!/usr/bin/env python3
"""BBC most-read closure — sealed dual-track proposer harness.

Builds one SEALED prompt file per (track, proposer) into a scratch directory OUTSIDE
the repo, so a proposer can reach neither the bank, the labels, nor another proposer's
output. Proposers see: the headline, the dense percentile, the articulated-instrument
percentile. They never see y, the most-read rank, the capture day, the per-criterion
bank scores, or each other.

FLEET: P = 8 across 2 families (codex gpt-5.6-luna x4, GLM-5.2 x4), each with its own
salt so the card ORDER differs per proposer and they do not all anchor on the same
first item. The freeze's target is 3 families; the Claude legs are unavailable this
session (subagent cap 500/500), so the round is recorded as degraded to the 2-family
floor. Composition is written into fleet_manifest.json for the round record.

  python3 harness_bbc.py build   --round 1
  python3 harness_bbc.py collect --round 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/bbc_closure")
K_A, K_B = 15, 10
FLEET = [("codex_luna", "a"), ("codex_luna", "b"), ("codex_luna", "c"),
         ("codex_luna", "d"), ("glm", "a"), ("glm", "b"), ("glm", "c"), ("glm", "d")]

CORPUS = ("headlines as they appeared on the BBC News home page, captured at intervals "
          "between 2017 and 2024")
CONSTRUCT = ("whether BBC's own readers made this article one of the ten most-read on "
             "the site that day -- a same-outlet readership signal, i.e. what readers "
             "actually clicked, as opposed to what editors placed prominently")

COMMON = """You are helping to audit a measurement instrument. You will be shown {n} news
headlines. Each carries two scores on a 0-1 PERCENTILE scale, computed by two different
systems that DISAGREE about it:

  * `dense percentile` -- a neural model that reads the raw headline text and was
    trained on this outcome. It is accurate overall but its reasoning is opaque.
  * `articulated percentile` -- a transparent instrument built from 23 surface features
    plus 14 explicitly-worded editorial criteria (news values: elite actors,
    institutional action, crisis, violence, scale, recency, legal proceedings, economic
    impact, celebrity, human interest, novelty, domestic relevance, hard-vs-soft news,
    top-tier running story).

The items below are the ones where the two systems disagree MOST. The construct both are
trying to measure is: {construct}.

The corpus is {corpus}.

IMPORTANT CONSTRAINTS:
  * You are NOT told the outcome for any item, and you must not guess it item-by-item.
    Your job is to name GENERAL, TESTABLE properties, not to label these specific rows.
  * Do not propose anything that requires information outside the headline text itself.
  * Each property must be judgeable by a careful reader from the headline alone, and
    must be phrased so two independent judges would score it the same way.

DIRECTION NOTE (added for this round, and labelled as an augmentation to the standing
instruction -- the standing instruction is unchanged): the disagreement in this slice
runs in BOTH directions, and is asymmetric. On 43 of the 60 items the ARTICULATED
instrument scores HIGHER than the dense model, and on 17 the dense model scores higher.
So consider both: (i) what the articulated criteria may be OVER-crediting in items the
dense model ranks low, and (ii) what the dense model may be perceiving in items the
articulated criteria rank low.

THE ITEMS:

{cards}
"""

TRACK_A = """
YOUR TASK (Track A). Propose exactly {k} candidate QUALITY-RELEVANT criteria: properties
that plausibly bear on the construct itself and that the 14 criteria listed above do not
already capture. Composite / interaction criteria are allowed and welcome -- "X together
with Y", "X in the absence of Y".

Return EXACTLY {k} items, one per line, in this format and nothing else:

NAME: <=8 words | DESCRIPTION: one or two sentences a judge could apply, stating what
scores high, what scores low, and when it does not apply | RATIONALE: <=20 words on why
this bears on the construct
"""

TRACK_B = """
YOUR TASK (Track B). Propose exactly {k} candidate SPURIOUS channels: textual properties
that would PREDICT the outcome without being part of the construct's merit -- length or
format proxies, boilerplate, house-style markers, topic or beat markers, temporal tells,
formatting habits.

Work in UPSTREAM-REASONING mode:
  1. enumerate factors BEYOND the text that could causally affect the outcome (the
     desk or beat that produced it, time of day or week, editorial promotion elsewhere
     on the site, seasonal news cycle, a story running for several days, staffing);
  2. for each, ask what textual FINGERPRINT it would leave in a headline;
  3. propose those fingerprints as channels.

Also consider explicitly: the item's POSITION or ORDER within its container (its place
in the day's run of stories, or in a multi-day running story) and any textual
fingerprint of it.

TAG each channel with its conjectured upstream parent, or "surface-only" if none. If a
channel's parent plausibly causes GENUINE merit as well (e.g. a beat that both gets
promoted and produces better stories), tag it MIXED -- do not force it to one side.

Return EXACTLY {k} items, one per line, in this format and nothing else:

NAME: <=8 words | DESCRIPTION: one or two sentences a judge could apply | PARENT:
<conjectured upstream factor, or surface-only> | MIXED: yes|no
"""


def build(round_no: int):
    cards_path = HERE / f"slice_r{round_no}_cards.txt"
    slice_json = json.loads((HERE / f"slice_r{round_no}.json").read_text())
    raw = cards_path.read_text().rstrip("\n").split("\n")
    # cards are 2 lines each
    cards = ["\n".join(raw[i:i + 2]) for i in range(0, len(raw), 2)]
    assert len(cards) == slice_json["n_slice"], (len(cards), slice_json["n_slice"])
    # blindness re-assert at seal time
    bad = re.compile(r"judgement|most.?read|\brank\b|label|y=", re.I)
    for c in cards:
        assert not bad.search(c), f"label leak at seal: {c[:60]}"

    d = SCRATCH / f"bbc_r{round_no}"
    d.mkdir(parents=True, exist_ok=True)
    manifest = {"round": round_no, "P": len(FLEET),
                "families": sorted({f for f, _ in FLEET}),
                "n_families": len({f for f, _ in FLEET}),
                "target_families": 3,
                "degraded": True,
                "degradation_reason": "Claude legs unavailable (subagent cap 500/500); "
                                      "recorded as the freeze's 2-family floor",
                "k_A": K_A, "k_B": K_B, "slots": 2 * len(FLEET),
                "slice": f"slice_r{round_no}.json",
                "slice_sha1": hashlib.sha1(cards_path.read_bytes()).hexdigest(),
                "direction_note": "labelled augmentation; standing instruction unchanged",
                "proposers": []}
    for fam, pid in FLEET:
        name = f"{fam}_{pid}"
        salt = f"bbc-r{round_no}-{name}"
        rng = random.Random(int(hashlib.sha256(salt.encode()).hexdigest()[:12], 16))
        order = cards[:]
        rng.shuffle(order)
        body = COMMON.format(n=len(order), construct=CONSTRUCT, corpus=CORPUS,
                             cards="\n".join(order))
        for track, tmpl, k in (("A", TRACK_A, K_A), ("B", TRACK_B, K_B)):
            (d / f"prompt_{track}_{name}.txt").write_text(body + tmpl.format(k=k))
        manifest["proposers"].append({"name": name, "family": fam, "salt": salt})
    (d / "fleet_manifest.json").write_text(json.dumps(manifest, indent=1))
    (HERE / f"fleet_manifest_r{round_no}.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "proposers"}, indent=1))
    print(f"\nsealed {2*len(FLEET)} prompts under {d}")


# Proposers do not reliably emit the literal NAME:/DESCRIPTION: prefixes. Observed in
# round 1: codex wrote "<name>: <description> | RATIONALE: ...", GLM Track B wrote
# "<NAME-IN-CAPS>: <description> | PARENT: ... | MIXED: ...", GLM Track A used the
# literal form. A strict parser silently dropped 120 of 200 proposals and zeroed three
# codex slots whose outputs were fine, which would have corrupted the species table and
# the Good-Turing accounting downstream. The rule below is deliberately format-tolerant
# but still precise: a proposal line must carry at least one KNOWN FIELD marker, so
# prose and preamble cannot be mistaken for a proposal.
FIELD = r"(?:RATIONALE|PARENT|MIXED|DESCRIPTION)"
LINE_STRICT = re.compile(r"^NAME:\s*(?P<name>[^|]+?)\s*\|\s*DESCRIPTION:\s*(?P<desc>.+)",
                         re.I | re.S)
LINE_LOOSE = re.compile(r"^(?P<name>[^:|]{3,90}?):\s*(?P<desc>.+)", re.S)
# third variant seen in round 1 (codex Track B): "<name> | <description> | PARENT: ..."
# -- pipe-separated with no colon after the name.
LINE_PIPE = re.compile(r"^(?P<name>[^|:]{3,90}?)\s*\|\s*(?P<desc>.+)", re.S)
HAS_FIELD = re.compile(r"\|\s*" + FIELD + r"\s*:", re.I)


def collect(round_no: int):
    d = SCRATCH / f"bbc_r{round_no}"
    man = json.loads((d / "fleet_manifest.json").read_text())
    out, missing = [], []
    for p in man["proposers"]:
        for track in ("A", "B"):
            f = d / f"out_{track}_{p['name']}.txt"
            if not f.exists() or len(f.read_text().strip()) < 100:
                missing.append(f"{track}/{p['name']}")
                continue
            for ln in f.read_text().splitlines():
                ln = ln.strip().lstrip("-*0123456789. ")
                if not HAS_FIELD.search(ln):
                    continue                      # not a proposal line
                m = (LINE_STRICT.match(ln) or LINE_LOOSE.match(ln)
                     or LINE_PIPE.match(ln))
                if not m:
                    continue
                rest = m.group("desc")
                item = {"track": track, "proposer": p["name"], "family": p["family"],
                        "name": m.group("name").strip().strip("*").strip()[:120]}
                for fld in ("RATIONALE", "PARENT", "MIXED"):
                    mm = re.search(fld + r":\s*([^|]+)", rest, re.I)
                    if mm:
                        item[fld.lower()] = mm.group(1).strip()[:200]
                item["description"] = re.split(r"\|\s*(RATIONALE|PARENT|MIXED):",
                                               rest, flags=re.I)[0].strip()[:600]
                out.append(item)
    res = {"round": round_no, "n_proposals": len(out), "missing_slots": missing,
           "by_track": {t: sum(1 for o in out if o["track"] == t) for t in "AB"},
           "by_family": {f: sum(1 for o in out if o["family"] == f)
                         for f in man["families"]},
           "by_proposer": {p["name"]: sum(1 for o in out if o["proposer"] == p["name"])
                           for p in man["proposers"]},
           "proposals": out}
    (HERE / f"proposals_r{round_no}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "proposals"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "collect"])
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    (build if a.cmd == "build" else collect)(a.round)
