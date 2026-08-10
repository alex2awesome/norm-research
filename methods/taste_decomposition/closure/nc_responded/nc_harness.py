#!/usr/bin/env python3
"""Sealed multi-proposer fleet harness -- N&C RESPONDED closure campaign.

Direct port of methods/taste_decomposition/closure/robust_mm/harness.py with the
N&C register substituted and BOTH tracks built (the pilot's harness was Track-A
only; the freeze runs the fleet on Track B as well, per the FREEZE ADDENDUM
"the fleet species machinery runs on Track-B proposals too").

SEAL CONTRACT (unchanged from M1):
  * every proposer sees the SAME disagreement slice: comment texts + the two
    models' percentile ranks, nothing else;
  * NO sight of the 198-rubric criterion bank;
  * NO sight of any other proposer's output;
  * NO sight of the label y (`responded`) -- slice files carry no y;
  * slice ORDERING permuted per proposer by stable sha256 over a proposer salt.

Track A and Track B run in SEPARATE contexts with SEPARATE instructions (prereg
round-structure items 2 and 3): a proposer never sees the other track's brief.

Usage:
  python nc_harness.py build --slice round1_disagreement_slice.json --tag r1
  python nc_harness.py collect --tag r1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/nc_closure")
K_A = 15
K_B = 10

PROPOSERS = [
    {"id": "claude_sonnet", "family": "claude", "model": "claude-sonnet (sealed subagent)", "salt": "ncp01"},
    {"id": "claude_opus", "family": "claude", "model": "claude-opus (sealed subagent)", "salt": "ncp02"},
    {"id": "codex_luna_a", "family": "openai", "model": "gpt-5.6-luna (codex exec, effort high)", "salt": "ncp03"},
    {"id": "codex_luna_b", "family": "openai", "model": "gpt-5.6-luna (codex exec, effort high)", "salt": "ncp04"},
    {"id": "glm_a", "family": "glm", "model": "glm-5.2 (thinking, key A)", "salt": "ncp05"},
    {"id": "glm_b", "family": "glm", "model": "glm-5.2 (thinking, key B)", "salt": "ncp06"},
]

_SETUP = """You are reading {n} PUBLIC COMMENTS submitted to United States federal agencies on
proposed rules (the notice-and-comment rulemaking process).

Two very different models score these comments:

  * a DENSE model, which reads the raw comment end-to-end and has learned to perceive
    what makes a comment consequential (it cannot say what it perceives);
  * a SCORECARD model, built only from a fixed bank of explicitly written criteria,
    each scored by a judge, then aggregated.

The {n} comments below are the ones where the two models DISAGREE MOST. Each carries
both models' percentile ranks within the mining pool:

  dense_pct  = the dense model's percentile (1.00 = it rates this comment highest)
  card_pct   = the scorecard model's percentile

Half the slice is `dense_high_card_low` (the dense model sees something the scorecard
misses) and half is `dense_low_card_high` (the scorecard is impressed by something the
dense model is not).
"""

TRACK_A = _SETUP + """
YOUR TASK. Propose exactly {k} NEW QUALITY-RELEVANT criteria: properties of a public
comment that a careful regulatory analyst would treat as evidence that the comment is
substantively good regulatory input -- criteria that would help explain what the dense
model is perceiving and the scorecard is missing (or, in the other direction, what the
scorecard over-credits).

HARD CONSTRAINTS.
1. You have NOT been shown the scorecard's criterion bank and must not ask for it.
   Propose what you actually think matters. Do not hedge toward "probably already
   covered" -- propose it anyway.
2. LABEL-BLIND. You have not been shown any outcome for these comments, and no
   criterion may reference agency responses, rule changes, acceptance, or any other
   outcome variable. Criteria must be judgeable from the comment text alone.
3. Each criterion must be SCORABLE 0-10 by an independent judge reading only the
   comment text, with a clear high end and a clear low end.
4. QUALITY-RELEVANT, not incidental. Do not propose length, formatting, document
   shape, boilerplate, topic labels, or submitter-identity markers: those are handled
   elsewhere. Propose things a reviewer would call merit.
5. Composite / interaction-shaped criteria are allowed and encouraged
   ("X stated TOGETHER WITH Y"), as are simple single-property criteria.
6. Ground each proposal in the slice: cite at least two row ids that motivate it.
7. Aim for {k} DISTINCT criteria -- do not restate one idea {k} ways.

OUTPUT FORMAT. Emit exactly one JSON object and nothing else:

{{"criteria": [
  {{"id": "C01",
   "name": "<short criterion name, <= 12 words>",
   "instruction": "<the 0-10 scoring instruction an independent judge would follow; say what a 10 looks like and what a 0 looks like>",
   "rationale": "<why this explains the disagreement; cite >= 2 row ids>"}},
  ... exactly {k} entries ...
]}}
"""

TRACK_B = _SETUP + """
YOUR TASK. Propose exactly {k} SUSPECTED-SPURIOUS channels: features of these comments
that you expect to be PREDICTIVE of how the dense model scores them, but that are NOT
evidence of the comment's substantive merit. "Predictive but not quality."

Work in TWO modes and draw from both.

MODE 1 -- SURFACE PATTERN HUNTING. Look directly at the text for tells: length and
verbosity proxies; document shape and formatting; boilerplate and template/form-letter
similarity; submission era or date tells; topic or agency style markers; density of
statutory or CFR citation used as decoration; tone and register markers.

MODE 2 -- UPSTREAM REASONING (do this deliberately, it is the more valuable mode).
Step (a): enumerate factors BEYOND the text that could causally affect how a federal
agency treats a comment -- for example the submitter's organisational resources, whether
a law firm or professional lobbyist drafted it, existing relationships between the
submitter and the agency, where in the docket's lifecycle the comment arrived, whether
it is part of a coordinated campaign, whether it was professionally edited, the
submitter's seniority or standing in the sector.
Step (b): for each such factor, ask what textual FINGERPRINT it would leave in the
comment -- what would you actually SEE in the words if that factor were present?
Step (c): propose that fingerprint as a scorable channel.

You are explicitly asked to look BEYOND the obvious surface tells -- the obvious ones
are already declared. Novel, specific, mechanistic channels are worth more than
restatements of "longer is better".

HARD CONSTRAINTS.
1. LABEL-BLIND. You have not been shown any outcome, and no channel may reference
   agency responses, rule changes, or any other outcome variable.
2. Each channel must be SCORABLE 0-10 by an independent judge reading ONLY the comment
   text, with a clear high end and a clear low end. The channel is the fingerprint, not
   the unseen factor itself -- you may not ask the judge to know who the submitter is.
3. TAG every channel with `upstream_parent`: the unseen factor you conjecture produces
   it, or the string "surface-only" if you conjecture none.
4. TAG every channel with `mixed`: true if the conjectured upstream parent plausibly
   ALSO causes genuine comment quality (e.g. a well-resourced organisation both gets
   more agency attention AND writes better-evidenced comments), false otherwise.
   Be honest here -- a `true` does not disqualify the channel, it changes how it is
   reported.
5. Ground each proposal in the slice: cite at least two row ids.
6. Aim for {k} DISTINCT channels.

OUTPUT FORMAT. Emit exactly one JSON object and nothing else:

{{"criteria": [
  {{"id": "N01",
   "name": "<short channel name, <= 12 words>",
   "instruction": "<the 0-10 scoring instruction an independent judge would follow; say what a 10 looks like and what a 0 looks like>",
   "upstream_parent": "<the unseen beyond-text factor you conjecture causes this fingerprint, or \\"surface-only\\">",
   "mixed": true or false,
   "rationale": "<why you expect this to be predictive without being merit; cite >= 2 row ids>"}},
  ... exactly {k} entries ...
]}}
"""


def order_for(rows, salt):
    return sorted(rows, key=lambda r: hashlib.sha256(f"{salt}|{r['i']}".encode()).hexdigest())


def build_prompt(rows, salt, track, k):
    ordered = order_for(rows, salt)
    body = []
    for n, r in enumerate(ordered):
        d = "dense_high_card_low" if r["direction"] == "dense_high_va_low" else "dense_low_card_high"
        body.append(
            f"--- ROW R{n+1:02d}  [{d}]  dense_pct={r['dense_pct']:.3f}  card_pct={r['va_nl_pct']:.3f} ---\n"
            f"{r['text'].strip()}"
        )
    tmpl = TRACK_A if track == "a" else TRACK_B
    return tmpl.format(n=len(rows), k=k) + "\n\n" + "\n\n".join(body) + "\n"


def cmd_build(args):
    rows = json.loads((HERE / args.slice).read_text())
    outdir = SCRATCH / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for track, k in (("a", K_A), ("b", K_B)):
        for p in PROPOSERS:
            salt = f"{args.tag}|{track}|{p['salt']}"
            txt = build_prompt(rows, salt, track, k)
            f = outdir / f"prompt_{track}_{p['id']}.txt"
            f.write_text(txt)
            manifest.append({**p, "tag": args.tag, "track": track, "k": k,
                             "prompt_path": str(f), "n_chars": len(txt), "slice": args.slice,
                             "order_sha": hashlib.sha256(
                                 "|".join(str(r["i"]) for r in order_for(rows, salt)).encode()
                             ).hexdigest()[:16]})
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    shas = {m["order_sha"] for m in manifest}
    print(f"{args.tag}: {len(manifest)} sealed prompts "
          f"({manifest[0]['n_chars']} chars each), {len(shas)} distinct orderings")
    print(f"  dir: {outdir}")


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
            try:
                d = json.loads(s[:cut] + "]}" if not s[:cut].rstrip().endswith("]}") else s[:cut])
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
        rec = {"name": str(c["name"]).strip(),
               "instruction": str(c.get("instruction", "")).strip(),
               "rationale": str(c.get("rationale", "")).strip()}
        # FREEZE ADDENDUM 2 tags (Track B only; absent on Track A by design)
        if "upstream_parent" in c:
            rec["upstream_parent"] = str(c.get("upstream_parent", "")).strip()
        if "mixed" in c:
            rec["mixed"] = bool(c.get("mixed"))
        out.append(rec)
    return out


def cmd_collect(args):
    outdir = SCRATCH / args.tag
    manifest = json.loads((outdir / "manifest.json").read_text())
    pool, report = [], []
    for m in manifest:
        f = outdir / f"out_{m['track']}_{m['id']}.txt"
        if not f.exists():
            report.append({"proposer": m["id"], "track": m["track"], "status": "MISSING"})
            continue
        try:
            crit = parse_output(f.read_text())
        except Exception as e:
            report.append({"proposer": m["id"], "track": m["track"], "status": f"PARSE_FAIL {e}"})
            continue
        names = [c["name"].lower() for c in crit]
        report.append({"proposer": m["id"], "track": m["track"], "family": m["family"],
                       "model": m["model"], "status": "ok", "n": len(crit),
                       "n_distinct_names": len(set(names)),
                       "mean_instruction_chars": round(
                           sum(len(c["instruction"]) for c in crit) / max(1, len(crit)))})
        for j, c in enumerate(crit):
            pool.append({"tag": args.tag, "track": m["track"], "proposer": m["id"],
                         "family": m["family"], "model": m["model"],
                         "pid": f"{m['track']}:{m['id']}#{j+1:02d}", **c})
    (HERE / f"fleet_{args.tag}.json").write_text(json.dumps(
        {"tag": args.tag, "k_a": K_A, "k_b": K_B, "proposers": report,
         "n_proposals": len(pool), "proposals": pool}, indent=1))
    print(json.dumps(report, indent=1))
    print(f"{args.tag}: {len(pool)} proposals -> fleet_{args.tag}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--slice", required=True); b.add_argument("--tag", required=True)
    c = sub.add_parser("collect"); c.add_argument("--tag", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "collect": cmd_collect}[a.cmd](a)
