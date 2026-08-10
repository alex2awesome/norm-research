#!/usr/bin/env python3
"""Sealed dual-track proposer harness for the REDDIT-JOKES COMMUNITY cell.

Inherited from mathse_vote/harness_maps.py (same SEAL CONTRACT, same Track-A
interaction-shaped steer held constant across rounds, same Track-B UPSTREAM-REASONING
mode from FREEZE ADDENDUM 2, same FREEZE ADDENDUM 4 position-in-container requirement).

TWO CELL-SPECIFIC REWRITES, recorded rather than silent:

  MODE 3 (FREEZE ADDENDUM 4).  A forum joke has no sibling-container the way a math.SE
  answer has its question.  The honest ordinal on this corpus is POSITION IN THE
  SUBREDDIT'S OWN POSTING STREAM -- which era of the forum's conventions, running gags
  and repost cycle the post comes from, and where it sits in the retelling chain of a
  joke that has been posted many times before.  The brief asks for the textual
  FINGERPRINT of that position, exactly as the addendum requires.  An observed
  `created_utc` covariate exists for 86.1% of the population, so `era_line.py` can check
  the fingerprint against the real ordinal -- the same audit the math.SE cell ran on
  answer arrival order.

  MODE 4.  Upstream priors rewritten for a joke-sharing forum (poster's practice and
  repeat-posting habit, sourcing/retelling versus origination, audience-management
  furniture, delivery-format habit, taboo/edginess register).

WHAT THE FLEET IS NOT TOLD.  y here is a top-quartile-versus-bottom-quartile split taken
INSIDE (length_bin x format x topic) strata, so length/format/topic are already partly
matched out of the label.  That is a fact about the instrument and telling a proposer
would be a design steer outside the freeze; it is used only when the map is interpreted.

Usage:
  python harness_maps.py build --cell jokes_community --round 1
  python harness_maps.py collect --cell jokes_community --round 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import cells as C

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/jokes_community")
K_A = 15
K_B = 10

# FLEET SIZE, raised for round 3 (coordinator, 2026-08-08, registered pre-Delta_2).
# Basis: notes/2026-08-08__aside_recovery_audit.md dose-response -- rediscovery ~70% at
# P ~ 6-7 and ~80% at P ~ 8 (beta-binomial; zero-inflated fit with asymptote pi=.88 puts
# 70% at P~7). That audit is explicit that raising P is the ONLY route that keeps the
# Good-Turing estimator valid, at ~2-6 extra sealed calls per round. Extra slots are the
# same models under fresh salts, exactly as codex_luna_a/b and glm_a/b already were, so
# every slot still reads its own independently ordered slice.
# TIER: all of these are TIER S (sealed) and are the only proposals that may feed the
# Good-Turing estimator. Any taxonomy-DIRECTED sweep runs as TIER D in its own file and is
# excluded from every estimator quantity (two-tier rule, design note S8 addendum).
PROPOSERS = [
    {"id": "claude_opus", "family": "claude", "model": "claude-opus (sealed subagent)", "salt": "jk01"},
    {"id": "claude_sonnet", "family": "claude", "model": "claude-sonnet (sealed subagent)", "salt": "jk02"},
    {"id": "claude_opus_b", "family": "claude", "model": "claude-opus (sealed subagent, fresh salt)", "salt": "jk07"},
    {"id": "codex_luna_a", "family": "openai", "model": "gpt-5.6-luna (codex exec, effort high)", "salt": "jk05"},
    {"id": "codex_luna_b", "family": "openai", "model": "gpt-5.6-luna (codex exec, effort high)", "salt": "jk06"},
    {"id": "codex_luna_c", "family": "openai", "model": "gpt-5.6-luna (codex exec, effort high)", "salt": "jk08"},
    {"id": "glm_a", "family": "glm", "model": "glm-5.2 (thinking, key A)", "salt": "jk03"},
    {"id": "glm_b", "family": "glm", "model": "glm-5.2 (thinking, key B)", "salt": "jk04"},
]
TIER = "S"          # sealed; the only tier that feeds Good-Turing

HEADER = """You are reading {n} {items} drawn from a corpus of {corpus}.

Two very different models score these items:

  * a DENSE model, which reads the raw text end-to-end and has learned to perceive
    {construct} directly (it cannot say what it perceives);
  * a SCORECARD model, built only from a fixed bank of explicitly written criteria,
    each scored 0-10 by a judge, and then aggregated.

The {n} items below are the ones where the two models DISAGREE MOST. Each carries both
models' percentile ranks within the mining pool:

  dense_pct  = the dense model's percentile (1.00 = it ranks this one highest)
  card_pct   = the scorecard model's percentile

Half the slice is `dense_high_card_low` (the dense model sees something the scorecard
misses) and half is `dense_low_card_high` (the scorecard is impressed by something the
dense model is not).
"""

TRACK_A = """
YOUR TASK. Propose exactly {k} NEW quality-relevant criteria: properties of a {item}
that a careful expert would treat as evidence of {construct}, which would help explain
what the dense model is perceiving and the scorecard is missing (or, in the other
direction, what the scorecard over-credits).

HARD CONSTRAINTS.
1. You have NOT been shown the scorecard's criterion bank and you must not ask for it.
   Propose what you actually think matters, from the evidence in front of you. Do not
   hedge toward "probably already covered" -- propose it anyway.
2. LABEL-BLIND. You have not been shown any outcome for these items, and no criterion
   may reference an outcome, a decision, a vote, a score, a selection or any other
   downstream result. Criteria must be judgeable from the item text alone.
3. Each criterion must be SCORABLE 0-10 by an independent judge reading only the item
   text, with a clear high end and a clear low end.
4. COMPOSITE / INTERACTION-SHAPED criteria are encouraged ("X together with Y"), as are
   simple single-property criteria. This shape instruction is held constant across all
   rounds of this campaign.
5. Ground each proposal in the slice: cite at least two row ids that motivate it.
6. Aim for {k} DISTINCT criteria -- do not restate one idea {k} ways.

OUTPUT FORMAT. Emit exactly one JSON object and nothing else:

{{"criteria": [
  {{"id": "A01",
   "name": "<short criterion name, <= 12 words>",
   "instruction": "<the 0-10 scoring instruction an independent judge would follow; say what a 10 looks like and what a 0 looks like>",
   "rationale": "<why this explains the disagreement; cite >= 2 row ids>"}},
  ... exactly {k} entries ...
]}}
"""

TRACK_B = """
YOUR TASK IS THE OPPOSITE OF QUALITY-HUNTING. Propose exactly {k} SUSPECTED-SPURIOUS
channels: features of a {item} that are plausibly PREDICTIVE of how these items fare,
but are NOT {construct}. These become DECLARED NUISANCES: they are never allowed to
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

MODE 3 -- POSITION IN CONTAINER (required; propose at least one). These items were not
written in a vacuum: each was posted into a long-running forum at some point in that
forum's history, and many are retellings of jokes the forum has already seen many times.
Consider where an item sits in that stream -- which ERA of the forum's conventions,
running gags, current-events references and taste it comes from, and where it falls in
the RETELLING CHAIN of a joke that has circulated before (a first posting, a variant, a
knowing re-post, a joke built on top of another joke the audience is assumed to know).
You cannot see the actual date or the actual retelling count; you must propose the
TEXTUAL FINGERPRINT such a position would leave. Examples of the SHAPE of such a
fingerprint (do not just copy these): topical references that date the item to a
particular period; vocabulary, slang or platform idiom that belongs to one era of the
forum; framing that presupposes the audience already knows the joke's form or its
original ("you know the one", "a remix of", "reposting because"); framing that presents
the item as newly minted. Score the fingerprint in the text.

MODE 4 -- UPSTREAM PRIORS FOR THIS CORPUS (use these as starting points, not a
checklist, and go beyond them). Consider the textual fingerprints of: the POSTER'S
PRACTICE in this format (a habitual poster who has learned the forum's rhythm versus a
one-off poster) -- what does a practised poster's text look like, in how it opens, how
it manages the reader, how much throat-clearing it allows itself, how the punch is set
off; SOURCING VERSUS ORIGINATION (a joke transcribed from a comedian, a translated joke,
a joke lifted from elsewhere on the internet, versus one written for this post) and the
traces each leaves; AUDIENCE-MANAGEMENT FURNITURE (apologies, disclaimers, edit notes,
requests for upvotes, tags and content warnings, "sorry if repost", "first post"); the
DELIVERY-FORMAT HABIT the item commits to (one-liner, question-and-answer riddle,
dialogue, narrative build, list) and how cleanly it commits to it; and the REGISTER OF
TABOO (how far the item leans on shock, slur-adjacency, sexual or scatological content,
or dark subject matter) as a circulation channel distinct from craft. For each, name the
visible textual trace and score the trace.

TAGGING (required on every channel).
  * `upstream_parent`: the unseen factor you conjecture is upstream of this fingerprint
    ("author's practice in the format", "audience/following", "submission timing",
    "position in the entry stream", ...), or exactly "surface-only" when you claim none.
  * `mixed`: true when that conjectured parent plausibly ALSO causes genuine
    {construct} (e.g. a practised entrant may both be favoured AND actually write
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
   the {k} must come from MODE 2, at least one from MODE 3 and at least one from MODE 4.

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


def build_prompt(rows, meta, salt, track):
    ordered = order_for(rows, salt)
    body = []
    for n, r in enumerate(ordered):
        d = "dense_high_card_low" if r["direction"] == "dense_high_va_low" else "dense_low_card_high"
        body.append(
            f"--- ROW R{n+1:02d}  [{d}]  dense_pct={r['dense_pct']:.3f}  "
            f"card_pct={r['va_nl_pct']:.3f} ---\n{(r['text'] or '')[:meta['text_trunc']].strip()}")
    head = HEADER.format(n=len(rows), items=meta["item"] + "s", corpus=meta["corpus"],
                         construct=meta["construct"])
    task = (TRACK_A if track == "A" else TRACK_B).format(
        k=K_A if track == "A" else K_B, item=meta["item"], construct=meta["construct"])
    return head + task + "\n\n" + "\n\n".join(body) + "\n"


def cmd_build(args):
    tag = f"{args.cell}_r{args.round}"
    rows = json.loads((HERE / f"{tag}_slice.json").read_text())
    meta = C.CELL_META[args.cell]
    outdir = SCRATCH / tag
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for p in PROPOSERS:
        for track in ("A", "B"):
            salt = f"{tag}|{track}|{p['salt']}"
            txt = build_prompt(rows, meta, salt, track)
            f = outdir / f"prompt_{track}_{p['id']}.txt"
            f.write_text(txt)
            manifest.append({**p, "track": track, "tag": tag, "cell": args.cell,
                             "round": args.round, "prompt_path": str(f),
                             "n_chars": len(txt),
                             "order_sha": hashlib.sha256("|".join(
                                 str(r["i"]) for r in order_for(rows, salt)).encode()
                             ).hexdigest()[:16]})
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    shas = {m["order_sha"] for m in manifest}
    print(f"{tag}: {len(manifest)} sealed prompts, {len(shas)} distinct orderings, "
          f"{manifest[0]['n_chars']} chars (A) / {manifest[1]['n_chars']} chars (B)")
    print(f"  dir: {outdir}")


JSON_RE = re.compile(r"\{[\s\S]*\}")


def parse_output(text, track):
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
    items = d.get("criteria") if track == "A" else d.get("channels")
    if items is None:
        items = d.get("channels") or d.get("criteria") or (d if isinstance(d, list) else None)
    out = []
    for c in items or []:
        if not c.get("name"):
            continue
        rec = {"name": str(c["name"]).strip(),
               "instruction": str(c.get("instruction", "")).strip(),
               "rationale": str(c.get("rationale", "")).strip()}
        if track == "B":
            rec["upstream_parent"] = str(c.get("upstream_parent", "surface-only")).strip() or "surface-only"
            rec["mixed"] = bool(c.get("mixed", False))
        out.append(rec)
    return out


def cmd_collect(args):
    tag = f"{args.cell}_r{args.round}"
    outdir = SCRATCH / tag
    manifest = json.loads((outdir / "manifest.json").read_text())
    pool, report = [], []
    for m in manifest:
        f = outdir / f"out_{m['track']}_{m['id']}.txt"
        if not f.exists() or len(f.read_text().strip()) < 100:
            report.append({"proposer": m["id"], "track": m["track"], "status": "MISSING"})
            continue
        try:
            crit = parse_output(f.read_text(), m["track"])
        except Exception as e:
            report.append({"proposer": m["id"], "track": m["track"],
                           "status": f"PARSE_FAIL {type(e).__name__}: {e}"})
            continue
        names = [c["name"].lower() for c in crit]
        rec = {"proposer": m["id"], "track": m["track"], "family": m["family"],
               "model": m["model"], "status": "ok", "n": len(crit),
               "n_distinct_names": len(set(names)),
               "mean_instruction_chars": round(sum(len(c["instruction"]) for c in crit)
                                               / max(1, len(crit)))}
        if m["track"] == "B":
            rec["n_mixed"] = sum(1 for c in crit if c.get("mixed"))
            rec["n_surface_only"] = sum(1 for c in crit
                                        if c.get("upstream_parent", "").lower() == "surface-only")
        report.append(rec)
        for j, c in enumerate(crit):
            pool.append({"tag": tag, "cell": args.cell, "round": args.round,
                         "track": m["track"], "proposer": m["id"], "family": m["family"],
                         "model": m["model"], "pid": f"{m['track']}:{m['id']}#{j+1:02d}", **c})
    out = {"tag": tag, "cell": args.cell, "round": args.round,
           "k_A": K_A, "k_B": K_B, "proposers": report,
           "n_proposals": len(pool), "proposals": pool}
    (HERE / f"{tag}_proposals_fleet.json").write_text(json.dumps(out, indent=1))
    ok = [r for r in report if r["status"] == "ok"]
    print(json.dumps(report, indent=1))
    print(f"{tag}: {len(pool)} proposals from {len(ok)}/{len(manifest)} slots "
          f"-> {tag}_proposals_fleet.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "collect"):
        s = sub.add_parser(name)
        s.add_argument("--cell", required=True)
        s.add_argument("--round", default="1")
    a = ap.parse_args()
    {"build": cmd_build, "collect": cmd_collect}[a.cmd](a)
