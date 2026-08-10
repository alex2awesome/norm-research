#!/usr/bin/env python3
"""B-SIDE M1 -- sealed multi-proposer harness, TRACK-B (spurious-mindset) mirror of
../harness.py.

SEAL CONTRACT (identical to the A-side M1, see ../harness.py docstring):
  * every proposer sees the SAME disagreement slice: row texts + the two model
    percentile ranks, nothing else;
  * NO sight of the B-channel census / the bside_census.json holdout design -- the
    fleet is sealed by construction and never sees a declared set regardless of which
    channels this run is scoring detection against;
  * NO sight of any other proposer's output, no label;
  * slice ORDERING is permuted per proposer by a stable sha256 sort over a
    proposer-specific + replicate-specific salt (never a seeded shuffle).

The TRACK_B instruction text below is copied VERBATIM from the current PRODUCTION
Track-B brief (methods/taste_decomposition/closure/maps_hw_si/harness_maps.py),
i.e. it already carries FREEZE ADDENDUM 2 (MODE 2, upstream-factor reasoning +
upstream_parent/mixed tagging) and FREEZE ADDENDUM 4 (MODE 3, position-in-container),
adapted only for the peer-abstract item/construct framing already used by the A-side
M1/M3 battery (methods/taste_decomposition/closure/robust_mm/harness.py).

Unlike A-side M3, there is no depletion step: Track-B channels never enter VA_nl, so
holding one out changes nothing about the score matrices or the disagreement slice.
All 3 replicates therefore read the IDENTICAL standard slice
(../slice_round5_fullbank.json, the full-bank round-4 disagreement slice) -- only the
proposer-salt ordering differs per (replicate, proposer), giving 3 independent fleet
draws on the same input, which is what the sensitivity-by-replicate readout needs.

Usage:
  python harness_b.py build --tag bside_rep1     # writes sealed prompts
  python harness_b.py collect --tag bside_rep1    # normalises outputs -> proposals_bside_rep1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROBUST_MM = HERE.parent
SLICE = ROBUST_MM / "slice_round5_fullbank.json"
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/robust_mm/bside")
K_B = 10

# P = 4 across 2 families (Claude sealed subagents + gpt-5.6-luna via codex exec),
# matching the A-side M3's ACTUAL fleet (GLM rate-limited out of the M3 replicates
# there too -- design note P>=4/>=2 families is the floor, not the target). A GLM
# supplement (glm_a) is attempted separately and appended if it lands in time.
PROPOSERS = [
    {"id": "claude_sonnet", "family": "claude", "model": "claude-sonnet (sealed subagent)", "salt": "bs01"},
    {"id": "claude_opus", "family": "claude", "model": "claude-opus (sealed subagent)", "salt": "bs02"},
    {"id": "codex_luna_a", "family": "openai", "model": "gpt-5.6-luna (codex exec, effort high)", "salt": "bs05"},
    {"id": "codex_luna_b", "family": "openai", "model": "gpt-5.6-luna (codex exec, effort high)", "salt": "bs06"},
]
PROPOSERS_GLM = [
    {"id": "glm_a", "family": "glm", "model": "glm-5.2 (thinking, key A)", "salt": "bs03"},
]

HEADER = """You are reading {n} abstracts of machine-learning conference paper submissions.

Two very different models score these submissions for quality:

  * a DENSE model, which reads the raw text end-to-end and has learned to perceive
    quality directly (it cannot say what it perceives);
  * a SCORECARD model, which is built only from a fixed bank of explicitly written
    quality criteria, each scored 0-10 by a judge, and then aggregated.

The {n} abstracts below are the ones where the two models DISAGREE MOST. For each
abstract you are given both models' percentile ranks within the mining pool:

  dense_pct  = the dense model's percentile (1.00 = it likes this one most)
  card_pct   = the scorecard model's percentile

Half the slice is `dense_high_card_low` (the dense model sees quality the scorecard
misses) and half is `dense_low_card_high` (the scorecard is impressed by something
the dense model is not).
"""

# Verbatim from methods/taste_decomposition/closure/maps_hw_si/harness_maps.py
# TRACK_B (production, carries Addendum-2 MODE 2 + Addendum-4 MODE 3), with
# {item}="abstract", {construct}="quality" substituted for the peer-verdict cell.
TRACK_B = """
YOUR TASK IS THE OPPOSITE OF QUALITY-HUNTING. Propose exactly {k} SUSPECTED-SPURIOUS
channels: features of an abstract that are plausibly PREDICTIVE of how these items fare,
but are NOT quality. These become DECLARED NUISANCES: they are never allowed to
join the scorecard's criterion bank; they are used only to DISCOUNT the other models.

Work in ALL THREE of these modes and mix the results:

MODE 1 -- SURFACE PATTERN-HUNTING. Length and verbosity proxies, formatting and layout
habits, typography (capitalisation, punctuation runs, emoji, hashtags, handles),
boilerplate and template phrases, community/venue style markers, topic and
subject-matter markers, temporal tells, orthographic quirks.

MODE 2 -- UPSTREAM-FACTOR REASONING (required; do this explicitly).
  (a) Enumerate factors BEYOND the text that could causally affect how such an item is
      received -- for example who produced it and their reputation, following, seniority
      or practice in this format; the timing of its production or submission; editing or
      assistance; process dynamics on the receiving side; social or audience effects.
      Adapt the list to THIS corpus.
  (b) For each factor ask: what TEXTUAL FINGERPRINT would it leave in the item text?
      An unseen factor that leaves NO trace in the text cannot bias a text-only model
      and is out of scope here; you are hunting the ones that DO leave a trace.
  (c) Propose those fingerprints as channels, phrased so a judge can score the
      FINGERPRINT (what is visible in the text) and never the unobservable factor.

MODE 3 -- POSITION IN CONTAINER (required; propose at least one). Consider the item's
POSITION or ORDER within its container -- where it falls in the stream of entries for
this contest, how early or late it was submitted relative to the others, how crowded
the container is, and where the container itself falls in the run of contests -- and
any TEXTUAL FINGERPRINT such a position would leave (for example: an item that reads as
one of the first, most obvious readings of the prompt, versus one that reads as a
late-arriving riff written by someone who has already seen the obvious takes taken).
Score the fingerprint in the text; you cannot see the actual position.

TAGGING (required on every channel).
  * `upstream_parent`: the unseen factor you conjecture is upstream of this fingerprint
    ("author's practice in the format", "audience/following", "submission timing",
    "position in the entry stream", ...), or exactly "surface-only" when you claim none.
  * `mixed`: true when that conjectured parent plausibly ALSO causes genuine
    quality (e.g. a practised entrant may both be favoured AND actually write
    better entries), false when the parent is purely a nuisance. Channels tagged mixed
    are reported in BOTH the discounted and the undiscounted readouts as a sensitivity
    band -- so tag honestly; `mixed` is not a demotion.

HARD CONSTRAINTS.
1. LABEL-BLIND. You have not been shown any outcome for these items, and no channel may
   reference an outcome, decision, vote, score or selection. Score the text only.
2. Each channel must be SCORABLE 0-10 by an independent judge reading only the item
   text. Phrase it as an EXTENT question ("how much of X is present"), never as a
   quality judgement, and say explicitly that the judge must not judge whether the
   feature is good.
3. Ground each proposal in the slice: cite at least two row ids.
4. Aim for {k} DISTINCT channels -- do not restate one idea {k} ways. At least four of
   the {k} must come from MODE 2 and at least one from MODE 3.

OUTPUT FORMAT. Emit exactly one JSON object and nothing else:

{{"channels": [
  {{"id": "B01",
   "name": "<short channel name, <= 12 words>",
   "instruction": "<the 0-10 scoring instruction an independent judge would follow; extent only, explicitly not a quality judgement>",
   "upstream_parent": "<conjectured unseen factor, or 'surface-only'>",
   "mixed": true or false,
   "rationale": "<why this is predictive-but-not-quality; cite >= 2 row ids>"}},
  ... exactly {k} entries ...
]}}
"""


def order_for(rows, salt):
    return sorted(rows, key=lambda r: hashlib.sha256(f"{salt}|{r['i']}".encode()).hexdigest())


def build_prompt(rows, salt, k=K_B):
    ordered = order_for(rows, salt)
    body = []
    for n, r in enumerate(ordered):
        d = "dense_high_card_low" if r["direction"] == "dense_high_va_low" else "dense_low_card_high"
        body.append(
            f"--- ROW R{n+1:02d}  [{d}]  dense_pct={r['dense_pct']:.3f}  card_pct={r['va_nl_pct']:.3f} ---\n"
            f"{r['text'].strip()}"
        )
    head = HEADER.format(n=len(rows))
    task = TRACK_B.format(k=k)
    return head + task + "\n\n" + "\n\n".join(body) + "\n"


def cmd_build(args):
    rows = json.loads(SLICE.read_text())
    proposers = PROPOSERS + (PROPOSERS_GLM if args.with_glm else [])
    outdir = SCRATCH / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for p in proposers:
        salt = f"{args.tag}|{p['salt']}"
        txt = build_prompt(rows, salt)
        f = outdir / f"prompt_{p['id']}.txt"
        f.write_text(txt)
        manifest.append({**p, "tag": args.tag, "prompt_path": str(f), "n_chars": len(txt),
                         "slice": str(SLICE),
                         "order_sha": hashlib.sha256(
                             "|".join(str(r["i"]) for r in order_for(rows, salt)).encode()
                         ).hexdigest()[:16]})
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    shas = {m["order_sha"] for m in manifest}
    print(f"{args.tag}: wrote {len(manifest)} sealed prompts ({manifest[0]['n_chars']} chars each), "
          f"{len(shas)} distinct orderings")
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
    s = m.group(0)
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        d = None
        for cut in range(len(s), 0, -1):
            frag = s[:cut]
            for close in ("", "]}", "\"}]}"):
                try:
                    d = json.loads(frag + close)
                    break
                except json.JSONDecodeError:
                    continue
            if d is not None:
                break
        if d is None:
            raise
    items = d.get("channels") if isinstance(d, dict) else d
    if items is None:
        items = d.get("criteria") if isinstance(d, dict) else None
    out = []
    for c in items or []:
        if not c.get("name"):
            continue
        out.append({
            "name": str(c["name"]).strip(),
            "instruction": str(c.get("instruction", "")).strip(),
            "upstream_parent": str(c.get("upstream_parent", "surface-only")).strip() or "surface-only",
            "mixed": bool(c.get("mixed", False)),
            "rationale": str(c.get("rationale", "")).strip(),
        })
    return out


def cmd_collect(args):
    outdir = SCRATCH / args.tag
    manifest = json.loads((outdir / "manifest.json").read_text())
    pool, report = [], []
    for m in manifest:
        f = outdir / f"out_{m['id']}.txt"
        if not f.exists() or len(f.read_text().strip()) < 50:
            report.append({"proposer": m["id"], "family": m["family"], "status": "MISSING"})
            continue
        try:
            crit = parse_output(f.read_text())
        except Exception as e:
            report.append({"proposer": m["id"], "family": m["family"],
                           "status": f"PARSE_FAIL {type(e).__name__}: {e}"})
            continue
        names = [c["name"].lower() for c in crit]
        report.append({"proposer": m["id"], "family": m["family"], "model": m["model"],
                       "status": "ok", "n": len(crit),
                       "n_distinct_names": len(set(names)),
                       "n_mixed": sum(1 for c in crit if c["mixed"]),
                       "n_surface_only": sum(1 for c in crit if c["upstream_parent"].lower() == "surface-only"),
                       "mean_instruction_chars": round(sum(len(c["instruction"]) for c in crit) / max(1, len(crit)))})
        for j, c in enumerate(crit):
            pool.append({"tag": args.tag, "proposer": m["id"], "family": m["family"],
                         "model": m["model"], "pid": f"{m['id']}#{j+1:02d}", **c})
    (HERE / f"proposals_{args.tag}.json").write_text(
        json.dumps({"tag": args.tag, "k_requested": K_B, "proposers": report,
                    "n_proposals": len(pool), "proposals": pool}, indent=1))
    print(json.dumps(report, indent=1))
    print(f"{args.tag}: {len(pool)} proposals -> proposals_{args.tag}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tag", required=True)
    b.add_argument("--with-glm", action="store_true")
    c = sub.add_parser("collect")
    c.add_argument("--tag", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "collect": cmd_collect}[a.cmd](a)
