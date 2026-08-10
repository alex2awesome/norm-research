#!/usr/bin/env python3
"""M1 -- sealed multi-proposer harness for the robustified missing-mass battery.

SEAL CONTRACT (the whole point of M1; see notes/2026-08-06 Part 3 and design Sec 8):
  * every proposer sees the SAME disagreement slice: row texts + the two model
    percentile ranks, nothing else;
  * NO sight of the criterion bank (the pilot's sequential rounds showed each
    proposer the current bank and told it not to duplicate -- that drives observed
    recapture to ~0 and makes Good-Turing/Chao1 undefined);
  * NO sight of any other proposer's output;
  * NO sight of the label `judgement` (label-blindness rule) -- the slice files
    carry no y and the prompt says so;
  * slice ORDERING is permuted per proposer by a stable sha256 sort over a
    proposer-specific salt, so two calls to the same model on the same slice are
    genuinely independent draws (never a seeded shuffle).

Usage:
  python harness.py build --slice slice_rep1.json --tag rep1     # writes prompts
  python harness.py collect --tag rep1                           # normalises outputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/robust_mm")
K = 15

# P = 6 proposers, >= 3 model families.  Two calls to the same model with different
# slice orderings count as two independent proposers (they are independent draws
# from that model's proposal distribution, which is what the species estimator needs).
PROPOSERS = [
    {"id": "claude_sonnet", "family": "claude", "model": "claude-sonnet (subagent)", "salt": "ps01"},
    {"id": "claude_opus", "family": "claude", "model": "claude-opus (subagent)", "salt": "ps02"},
    {"id": "glm_a", "family": "glm", "model": "glm-5.2 (thinking, key A)", "salt": "ps03"},
    {"id": "glm_b", "family": "glm", "model": "glm-5.2 (thinking, key B)", "salt": "ps04"},
    {"id": "codex_luna_a", "family": "openai", "model": "gpt-5.6-luna (codex exec, effort high)", "salt": "ps05"},
    {"id": "codex_luna_b", "family": "openai", "model": "gpt-5.6-luna (codex exec, effort high)", "salt": "ps06"},
]

PREAMBLE = """You are reading {n} abstracts of machine-learning conference paper submissions.

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

YOUR TASK. Propose exactly {k} NEW quality-relevant criteria: properties of an
abstract that a careful reviewer would treat as evidence of research quality, which
would help explain what the dense model is perceiving and the scorecard is missing
(or, in the other direction, what the scorecard over-credits).

HARD CONSTRAINTS.
1. You have NOT been shown the scorecard's criterion bank and you must not ask for
   it. Propose what you actually think matters, from the evidence in front of you.
   Do not hedge toward "probably already covered" -- propose it anyway.
2. LABEL-BLIND. You have not been shown whether any of these papers was accepted or
   rejected, and no criterion may reference acceptance, rejection, reviewer scores,
   venue decisions, or any other outcome variable. Criteria must be judgeable from
   the abstract text alone.
3. Each criterion must be SCORABLE 0-10 by an independent judge reading only the
   abstract, with a clear high end and a clear low end.
4. Composite / interaction-shaped criteria are allowed and encouraged
   ("X stated TOGETHER WITH Y"), as are simple single-property criteria.
5. Ground each proposal in the slice: cite at least two row ids that motivate it.
6. Aim for {k} DISTINCT criteria -- do not restate one idea {k} ways.

OUTPUT FORMAT. Emit exactly one JSON object and nothing else:

{{"criteria": [
  {{"id": "C01",
   "name": "<short criterion name, <= 12 words>",
   "instruction": "<the 0-10 scoring instruction an independent judge would follow; say what a 10 looks like and what a 0 looks like>",
   "rationale": "<why this explains the disagreement; cite >= 2 row ids>"}},
  ... exactly {k} entries ...
]}}
"""


def order_for(rows, salt):
    return sorted(rows, key=lambda r: hashlib.sha256(f"{salt}|{r['i']}".encode()).hexdigest())


def build_prompt(rows, salt, k=K):
    ordered = order_for(rows, salt)
    body = []
    for n, r in enumerate(ordered):
        d = "dense_high_card_low" if r["direction"] == "dense_high_va_low" else "dense_low_card_high"
        body.append(
            f"--- ROW R{n+1:02d}  [{d}]  dense_pct={r['dense_pct']:.3f}  card_pct={r['va_nl_pct']:.3f} ---\n"
            f"{r['text'].strip()}"
        )
    return PREAMBLE.format(n=len(rows), k=k) + "\n\n" + "\n\n".join(body) + "\n"


def cmd_build(args):
    rows = json.loads((HERE / args.slice).read_text())
    outdir = SCRATCH / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for p in PROPOSERS:
        txt = build_prompt(rows, f"{args.tag}|{p['salt']}")
        f = outdir / f"prompt_{p['id']}.txt"
        f.write_text(txt)
        manifest.append({**p, "tag": args.tag, "prompt_path": str(f), "n_chars": len(txt),
                         "slice": args.slice,
                         "order_sha": hashlib.sha256(
                             "|".join(str(r["i"]) for r in order_for(rows, f"{args.tag}|{p['salt']}")
                                      ).encode()).hexdigest()[:16]})
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    # sanity: orderings really differ
    shas = {m["order_sha"] for m in manifest}
    print(f"{args.tag}: wrote {len(manifest)} sealed prompts ({manifest[0]['n_chars']} chars each), "
          f"{len(shas)} distinct orderings")
    print(f"  dir: {outdir}")


JSON_RE = re.compile(r"\{[\s\S]*\}")


def parse_output(text):
    """Tolerant extraction of the {"criteria": [...]} object."""
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
        # last resort: trim to the final closing brace of the criteria array
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
        out.append({"name": str(c["name"]).strip(),
                    "instruction": str(c.get("instruction", "")).strip(),
                    "rationale": str(c.get("rationale", "")).strip()})
    return out


def cmd_collect(args):
    outdir = SCRATCH / args.tag
    manifest = json.loads((outdir / "manifest.json").read_text())
    pool, report = [], []
    for m in manifest:
        f = outdir / f"out_{m['id']}.txt"
        if not f.exists():
            report.append({"proposer": m["id"], "status": "MISSING"})
            continue
        try:
            crit = parse_output(f.read_text())
        except Exception as e:
            report.append({"proposer": m["id"], "status": f"PARSE_FAIL {e}"})
            continue
        names = [c["name"].lower() for c in crit]
        report.append({"proposer": m["id"], "family": m["family"], "model": m["model"],
                       "status": "ok", "n": len(crit),
                       "n_distinct_names": len(set(names)),
                       "mean_instruction_chars": round(sum(len(c["instruction"]) for c in crit) / max(1, len(crit)))})
        for j, c in enumerate(crit):
            pool.append({"tag": args.tag, "proposer": m["id"], "family": m["family"],
                         "model": m["model"], "pid": f"{m['id']}#{j+1:02d}", **c})
    (HERE / f"proposals_{args.tag}.json").write_text(
        json.dumps({"tag": args.tag, "k_requested": K, "proposers": report,
                    "n_proposals": len(pool), "proposals": pool}, indent=1))
    print(json.dumps(report, indent=1))
    print(f"{args.tag}: {len(pool)} proposals -> proposals_{args.tag}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--slice", required=True); b.add_argument("--tag", required=True)
    c = sub.add_parser("collect"); c.add_argument("--tag", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "collect": cmd_collect}[a.cmd](a)
