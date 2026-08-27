#!/usr/bin/env python3
"""Sealed multi-family proposer fleet for the CW-community closure campaign.

Direct port of methods/taste_decomposition/closure/robust_mm/harness.py (the
sealed-fleet seal contract) with:
  * the CW domain preamble (writing prompt + story, community upvotes),
  * BOTH tracks: TRACK A (quality-relevant, k_A=15) and TRACK B (suspected
    spurious, k_B=10), issued as SEPARATE sealed prompts in separate contexts,
  * per-proposer stable-hash slice ordering (never a seeded shuffle).

Seal contract, unchanged from the pilot's robustification:
  proposers see row text + the two models' percentile ranks and NOTHING else --
  no bank, no other proposer, no label y.

Usage:
  python fleet_cw.py build --slice round1_slice.json --tag r1
  python fleet_cw.py collect --tag r1 --track A
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get(
    "CW_FLEET_SCRATCH",
    "/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
    "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/cw_fleet"))
K_A = 15
K_B = 10

PROPOSERS = [
    {"id": "claude_sonnet", "family": "claude", "model": "claude-sonnet (sealed subagent)", "salt": "cw01"},
    {"id": "claude_opus", "family": "claude", "model": "claude-opus (sealed subagent)", "salt": "cw02"},
    {"id": "glm_a", "family": "glm", "model": "glm-5.2 (thinking, key A)", "salt": "cw03"},
    {"id": "glm_b", "family": "glm", "model": "glm-5.2 (thinking, key B)", "salt": "cw04"},
    {"id": "codex_luna_a", "family": "openai", "model": "gpt-5.6-luna (codex exec, effort high)", "salt": "cw05"},
    {"id": "codex_luna_b", "family": "openai", "model": "gpt-5.6-luna (codex exec, effort high)", "salt": "cw06"},
]

_SETUP = """You are reading {n} short stories posted to an online creative-writing
community. Each was written in response to a one-line writing prompt.

Two very different models score these stories:

  * a DENSE model, which reads the raw text end-to-end and has learned to perceive
    what this community rewards (it cannot say what it perceives);
  * a SCORECARD model, built only from a fixed bank of explicitly written craft
    criteria, each scored by a judge, and then aggregated.

The {n} stories below are the ones where the two models DISAGREE MOST. For each you
are given both models' percentile ranks within the mining pool:

  dense_pct  = the dense model's percentile (1.00 = it likes this one most)
  card_pct   = the scorecard model's percentile

Half the slice is `dense_high_card_low` (the dense model sees something the scorecard
misses) and half is `dense_low_card_high` (the scorecard is impressed by something the
dense model is not).

Long stories carry a deterministic middle omission; judge what is shown.
"""

PREAMBLE_A = _SETUP + """
YOUR TASK. Propose exactly {k} NEW quality-relevant criteria: properties of a story
that a careful editor would treat as evidence of craft or reader value, which would
help explain what the dense model is perceiving and the scorecard is missing (or, in
the other direction, what the scorecard over-credits).

HARD CONSTRAINTS.
1. You have NOT been shown the scorecard's criterion bank and you must not ask for it.
   Propose what you actually think matters, from the evidence in front of you. Do not
   hedge toward "probably already covered" -- propose it anyway.
2. LABEL-BLIND. You have not been shown how any of these stories scored with readers,
   and no criterion may reference upvotes, popularity, ranking, awards, or any other
   outcome variable. Criteria must be judgeable from the story text alone.
3. Each criterion must be SCORABLE by an independent judge reading only the writing
   prompt and the story, with a clear high end and a clear low end.
4. Composite / interaction-shaped criteria are allowed and encouraged ("X together
   with Y"), as are simple single-property criteria.
5. Ground each proposal in the slice: cite at least two row ids that motivate it.
6. Aim for {k} DISTINCT criteria -- do not restate one idea {k} ways.

OUTPUT FORMAT. Emit exactly one JSON object and nothing else:

{{"criteria": [
  {{"id": "A01",
   "name": "<short criterion name, <= 12 words>",
   "instruction": "<the scoring instruction an independent judge would follow; say what clearly satisfying it looks like and what clearly failing it looks like>",
   "rationale": "<why this explains the disagreement; cite >= 2 row ids>"}},
  ... exactly {k} entries ...
]}}
"""

PREAMBLE_B = _SETUP + """
YOUR TASK IS DIFFERENT FROM CRITICISM. Propose exactly {k} SUSPECTED-SPURIOUS
channels: properties of these texts that you believe would PREDICT how this community
responds WITHOUT being evidence of story quality.

Work in BOTH of these modes and draw proposals from each.

MODE 1 -- SURFACE PATTERN HUNTING. Look directly at the texts for:
  * length, format and typography habits (paragraphing, line breaks, dividers,
    all-caps, italic/bold markup, ellipses, em-dashes);
  * community boilerplate and platform furniture (author notes, "thanks for reading",
    part-1/continuation markers, subreddit conventions, edit notes);
  * genre / topic / content markers that are fashionable in this community;
  * narrative-mode tells (first person, present tense, dialogue-heavy openings).

MODE 2 -- UPSTREAM REASONING (do this deliberately, it is not optional).
  (a) Enumerate unseen factors BEYOND the text that could causally affect how a story
      is received here: author reputation or seniority in the community, an established
      following, posting time and thread position, how early the story was posted under
      the prompt, whether the writer had editing help or is a professional, series
      momentum from earlier instalments, cross-posting and social networks, moderator
      or curation dynamics.
  (b) For EACH such factor, ask: what textual FINGERPRINT would it leave in the story
      itself -- what would a reader be able to see in the text that betrays it?
  (c) Propose those fingerprints as channels.

TAGGING (required on every channel).
  * `upstream_parent`: the unseen factor you think produces this fingerprint, or the
    exact string "surface-only" if you think the channel has no upstream cause.
  * `mixed`: true if the same upstream parent would plausibly ALSO cause genuinely
    better writing (e.g. an experienced writer both attracts a following AND writes
    better), false if the parent affects reception without improving the work.
    Be honest -- `mixed: true` is not a failure, it is the interesting case.

HARD CONSTRAINTS.
1. You have NOT been shown any criterion bank and must not ask for one.
2. LABEL-BLIND. You have not been shown how any story scored with readers; no channel
   may reference upvotes, popularity, ranking, or any other outcome variable.
3. Each channel must be SCORABLE by an independent judge reading only the writing
   prompt and the story text. A fingerprint you cannot see in the text is not a
   proposal -- discard it and say so in the rationale of a neighbouring channel.
4. Name the CHANNEL, not the verdict: describe what to count or look for, not whether
   it is good.
5. Ground each proposal in the slice: cite at least two row ids.
6. Aim for {k} DISTINCT channels, with at least 4 coming from MODE 2.

OUTPUT FORMAT. Emit exactly one JSON object and nothing else:

{{"criteria": [
  {{"id": "B01",
   "name": "<short channel name, <= 12 words>",
   "instruction": "<the scoring instruction an independent judge would follow; say what a high count/degree looks like and what a low one looks like>",
   "upstream_parent": "<the unseen factor this fingerprints, or 'surface-only'>",
   "mixed": <true or false>,
   "rationale": "<why you think this is predictive but not quality; cite >= 2 row ids>"}},
  ... exactly {k} entries ...
]}}
"""


def order_for(rows, salt):
    return sorted(rows, key=lambda r: hashlib.sha256(f"{salt}|{r['i']}".encode()).hexdigest())


def build_prompt(rows, salt, track, k):
    ordered = order_for(rows, salt)
    body = []
    for n, r in enumerate(ordered):
        d = ("dense_high_card_low" if r["direction"] == "dense_high_va_low"
             else "dense_low_card_high")
        body.append(
            f"--- ROW R{n+1:02d}  [{d}]  dense_pct={r['dense_pct']:.3f}  "
            f"card_pct={r['va_nl_pct']:.3f} ---\n"
            f"WRITING PROMPT: {r['prompt'].strip()}\n\nSTORY:\n{r['story'].strip()}")
    pre = (PREAMBLE_A if track == "A" else PREAMBLE_B)
    return pre.format(n=len(rows), k=k) + "\n\n" + "\n\n".join(body) + "\n"


def cmd_build(args):
    rows = json.loads((HERE / args.slice).read_text())
    manifest = []
    for track, k in (("A", K_A), ("B", K_B)):
        outdir = SCRATCH / f"{args.tag}{track}"
        outdir.mkdir(parents=True, exist_ok=True)
        for p in PROPOSERS:
            salt = f"{args.tag}|{track}|{p['salt']}"
            txt = build_prompt(rows, salt, track, k)
            f = outdir / f"prompt_{p['id']}.txt"
            f.write_text(txt)
            manifest.append({**p, "tag": args.tag, "track": track, "k": k,
                             "prompt_path": str(f), "n_chars": len(txt),
                             "slice": args.slice,
                             "order_sha": hashlib.sha256(
                                 "|".join(str(r["i"]) for r in order_for(rows, salt)
                                          ).encode()).hexdigest()[:16]})
        (outdir / "manifest.json").write_text(
            json.dumps([m for m in manifest if m["track"] == track], indent=1))
    shas = {m["order_sha"] for m in manifest}
    print(f"{args.tag}: {len(manifest)} sealed prompts "
          f"({manifest[0]['n_chars']//1024} KB each), {len(shas)} distinct orderings")
    print(f"  dirs: {SCRATCH}/{args.tag}A , {SCRATCH}/{args.tag}B")


JSON_RE = re.compile(r"\{[\s\S]*\}")


def parse_output(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    m = JSON_RE.search(text)
    if not m:
        raise ValueError("no JSON object found")
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        s = m.group(0)
        for cut in range(len(s), 0, -1):
            frag = s[:cut]
            try:
                d = json.loads(frag if frag.rstrip().endswith("]}") else frag + "]}")
                break
            except json.JSONDecodeError:
                continue
        else:
            raise
    crit = d["criteria"] if isinstance(d, dict) else d
    out = []
    for c in crit:
        if not c.get("name"):
            continue
        out.append({"name": str(c["name"]).strip(),
                    "instruction": str(c.get("instruction", "")).strip(),
                    "rationale": str(c.get("rationale", "")).strip(),
                    # FREEZE ADDENDUM 2: Track-B upstream tagging
                    "upstream_parent": str(c.get("upstream_parent", "")).strip()
                    or None,
                    "mixed": (bool(c["mixed"]) if isinstance(c.get("mixed"),
                                                             (bool, int))
                              else (str(c.get("mixed", "")).strip().lower()
                                    in ("true", "yes", "1"))
                              if c.get("mixed") is not None else None)})
    return out


def cmd_collect(args):
    outdir = SCRATCH / f"{args.tag}{args.track}"
    manifest = json.loads((outdir / "manifest.json").read_text())
    pool, report = [], []
    for m in manifest:
        f = outdir / f"out_{m['id']}.txt"
        if not f.exists():
            report.append({"proposer": m["id"], "status": "MISSING"})
            continue
        try:
            crit = parse_output(f.read_text())
        except Exception as e:  # noqa: BLE001
            report.append({"proposer": m["id"], "status": f"PARSE_FAIL {e}"})
            continue
        names = [c["name"].lower() for c in crit]
        rec = {"proposer": m["id"], "family": m["family"], "model": m["model"],
               "status": "ok", "n": len(crit),
               "n_distinct_names": len(set(names)),
               "mean_instruction_chars": round(
                   sum(len(c["instruction"]) for c in crit) / max(1, len(crit)))}
        if args.track == "B":  # FREEZE ADDENDUM 2 compliance check
            rec["n_upstream_tagged"] = sum(
                1 for c in crit if c.get("upstream_parent")
                and c["upstream_parent"].lower() != "surface-only")
            rec["n_surface_only"] = sum(
                1 for c in crit if (c.get("upstream_parent") or "").lower()
                == "surface-only")
            rec["n_untagged"] = sum(1 for c in crit if not c.get("upstream_parent"))
            rec["n_mixed"] = sum(1 for c in crit if c.get("mixed"))
            rec["mode2_quota_met"] = rec["n_upstream_tagged"] >= 4
        report.append(rec)
        for j, c in enumerate(crit):
            pool.append({"tag": args.tag, "track": args.track, "proposer": m["id"],
                         "family": m["family"], "model": m["model"],
                         "pid": f"{args.track}|{m['id']}#{j+1:02d}", **c})
    (HERE / f"{args.tag}_fleet_{args.track}.json").write_text(
        json.dumps({"tag": args.tag, "track": args.track,
                    "k_requested": K_A if args.track == "A" else K_B,
                    "proposers": report, "n_proposals": len(pool),
                    "proposals": pool}, indent=1))
    print(json.dumps(report, indent=1))
    print(f"{args.tag}/{args.track}: {len(pool)} proposals -> "
          f"{args.tag}_fleet_{args.track}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--slice", required=True)
    b.add_argument("--tag", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--tag", required=True)
    c.add_argument("--track", required=True, choices=["A", "B"])
    a = ap.parse_args()
    {"build": cmd_build, "collect": cmd_collect}[a.cmd](a)
