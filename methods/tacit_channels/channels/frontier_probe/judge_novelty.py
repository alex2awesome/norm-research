"""Phase 0.3 — LLM-judge pass: recombination vs novel formalization (measurement of record).

Two subcommands, provider-neutral (the API call is api_field_runner.py's job):
  prepare  — build the judging prompts JSONL, WITH blinded known-label anchors mixed in
             (feedback_anchor_test_annotation_passes)
  ingest   — join runner results, validate anchor separation, emit per-domain rates

Judge tier: Sonnet+ or GLM only (feedback_judges_sonnet_or_better). Runner:
  python methods/metric_seam/battery/api_field_runner.py --backend zai_anthropic \
      --model glm-4.7 --prompts <prepare output> --out <results jsonl>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

from methods.tacit_channels.channels.common import read_jsonl, write_jsonl

PROMPT = """You judge whether an evaluation-criterion articulation introduces genuinely novel
formalization or merely recombines standard evaluation vocabulary for its domain.

Domain: {domain}
Criterion name: {construct}
Articulation (the guidance added beyond the bare name):
---
{articulation}
---

Score the articulation's conceptual novelty relative to standard {domain} evaluation practice:
0 = pure recombination — every concept is standard vocabulary for this domain, reworded
5 = a non-obvious combination or operationalization of standard concepts
10 = genuinely novel formalization — introduces a measurable construct, decomposition, or
     decision procedure that is NOT part of standard {domain} evaluation vocabulary

Reply with a one-sentence justification, then on the final line exactly: SCORE: <0-10>"""

# Blinded anchors: judge sees them as ordinary rows. Recombination anchors are boilerplate
# rubric prose; novel anchors introduce invented-but-coherent formal procedures.
ANCHORS = [
    {"label": "recombination", "domain": "humor", "construct": "Comedic timing",
     "articulation": "The joke should be funny and well paced, with a clear setup and an "
                     "effective punchline that lands for the audience."},
    {"label": "recombination", "domain": "peer-review", "construct": "Clarity of writing",
     "articulation": "The paper should be clearly written and well organized, with arguments "
                     "that are easy to follow and free of grammatical errors."},
    {"label": "recombination", "domain": "notice-and-comment", "construct": "Responsiveness",
     "articulation": "The comment should address the proposed rule directly and provide "
                     "relevant supporting evidence for its position."},
    {"label": "novel", "domain": "humor", "construct": "Incongruity budget",
     "articulation": "Assign each joke an incongruity budget B = number of frame shifts the "
                     "setup licenses. A joke passes iff the punchline spends exactly B: fewer "
                     "is flat, more is noise. Count frame shifts by paraphrasing setup and "
                     "punchline into event schemas and diffing their role bindings."},
    {"label": "novel", "domain": "peer-review", "construct": "Claim-evidence flux",
     "articulation": "For each numbered claim, compute flux = (independent evidence sources "
                     "cited for the claim) minus (load-bearing uses of the claim later in the "
                     "paper). The paper passes iff no claim has negative flux; report the "
                     "minimum flux over claims as the score."},
    {"label": "novel", "domain": "notice-and-comment", "construct": "Docket displacement",
     "articulation": "Score a comment by displacement: the number of sentences in the final "
                     "rule preamble whose most similar draft-stage sentence changes when the "
                     "comment is removed from the simulated docket. Estimate via "
                     "leave-one-out nearest-neighbor matching over sentence embeddings."},
]


def _anchor_id(a: dict) -> str:
    return "anchor_" + hashlib.sha256(
        (a["construct"] + a["articulation"]).encode()).hexdigest()[:12]


def prepare(rescues_path: str, out_path: str, max_rows: int | None,
            seed: int = 20260722) -> None:
    rows = [r for r in read_jsonl(rescues_path) if r.get("articulation_text")]
    if max_rows:
        rows = rows[:max_rows]
    prompts = []
    for r in rows:
        added = r["articulation_text"]
        name = r.get("construct_name_text") or ""
        if added.startswith(name):
            added = added[len(name):].strip()
        prompts.append({
            "channel": "novelty",
            "aspect_id": f'{r["family"]}::{r["executor_job"]}::{r["domain"]}',
            "datapoint_id": r["cell_id"],
            "prompt": PROMPT.format(domain=r["domain"], construct=r.get("construct") or "",
                                    articulation=added),
            "rescued": r["rescued"],
        })
    for a in ANCHORS:
        prompts.append({
            "channel": "novelty_anchor", "aspect_id": f'anchor::{a["label"]}',
            "datapoint_id": _anchor_id(a),
            "prompt": PROMPT.format(domain=a["domain"], construct=a["construct"],
                                    articulation=a["articulation"]),
            "rescued": None,
        })
    random.Random(seed).shuffle(prompts)  # blind the anchors positionally
    n = write_jsonl(out_path, prompts)
    print(f"wrote {n} prompts ({len(ANCHORS)} anchors) -> {out_path}")


def ingest(prompts_path: str, results_path: str, out_path: str,
           anchor_margin: float = 3.0) -> None:
    prompts = {(p["channel"], p["aspect_id"], p["datapoint_id"]): p
               for p in read_jsonl(prompts_path)}
    results = read_jsonl(results_path)
    joined, anchor_scores = [], {"recombination": [], "novel": []}
    for res in results:
        key = (res.get("channel", ""), res["aspect_id"], res["datapoint_id"])
        p = prompts.get(key)
        if p is None or res.get("score") in (None, "NA"):
            continue
        score = float(res["score"])
        if p["channel"] == "novelty_anchor":
            anchor_scores[p["aspect_id"].split("::")[1]].append(score)
        else:
            joined.append({**{k: p[k] for k in ("aspect_id", "datapoint_id", "rescued")},
                           "novelty_score": score})
    rec = anchor_scores["recombination"]
    nov = anchor_scores["novel"]
    rec_mean = sum(rec) / len(rec) if rec else None
    nov_mean = sum(nov) / len(nov) if nov else None
    anchors_pass = (rec_mean is not None and nov_mean is not None
                    and (nov_mean - rec_mean) >= anchor_margin)
    write_jsonl(out_path, joined)
    summary = {
        "n_judged": len(joined),
        "anchor_recombination_mean": rec_mean, "anchor_novel_mean": nov_mean,
        "anchors_pass": anchors_pass,
        "rescued_mean": _mean([j["novelty_score"] for j in joined if j["rescued"]]),
        "contrast_mean": _mean([j["novelty_score"] for j in joined if not j["rescued"]]),
    }
    print(json.dumps(summary, indent=2))
    if not anchors_pass:
        raise SystemExit("ANCHOR FAILURE: judge does not separate known recombination from "
                         "known novelty — batch is not interpretable, do not use.")


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare")
    p1.add_argument("--rescues", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--max-rows", type=int, default=None)
    p2 = sub.add_parser("ingest")
    p2.add_argument("--prompts", required=True)
    p2.add_argument("--results", required=True)
    p2.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.rescues, args.out, args.max_rows)
    else:
        ingest(args.prompts, args.results, args.out)


if __name__ == "__main__":
    main()
