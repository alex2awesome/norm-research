"""Build recoverability smoke-test prompts.

For each (task, aspect_id), collect k (text, score) pairs from Claude judge
labels, then write a single subagent prompt asking the subagent to guess what
rubric/aspect produced those labels.

Hides: aspect name, aspect description, aspect_id.
Reveals: texts + scores.

Output: runs/validity_full/v2/_recoverability/<task>__<aid>__k<k>.txt
        runs/validity_full/v2/_recoverability/_truth.json (ground truth + key map)
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

REPO = Path("/Users/spangher/Projects/stanford-research/norm-research")
V2 = REPO / "runs/validity_full/v2"
OUT = V2 / "_recoverability"
OUT.mkdir(exist_ok=True, parents=True)


def parse_resp(raw):
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    if not raw.startswith("{"):
        s, e = raw.find("{"), raw.rfind("}")
        if s >= 0 and e > s: raw = raw[s:e+1]
    return json.loads(raw)


def collect_per_aspect(task: str):
    """Return: {aspect_id: [(dp_id, score), ...]}, only non-null scores."""
    cd = V2 / task / "judge_responses_claude"
    per = collections.defaultdict(list)
    for f in cd.glob("*.json"):
        try:
            obj = parse_resp(f.read_text())
            for tr in obj.get("results", []):
                tid = tr.get("text_id")
                if not tid: continue
                for sc in tr.get("scores", []):
                    aid = sc.get("aspect_id")
                    if aid and sc.get("score") is not None:
                        per[aid].append((tid, float(sc["score"])))
        except Exception:
            pass
    return per


def pick_aspects(per_aspect, n_pick=3, min_n=35, rng=None):
    """Pick aspects with enough non-null + variance, mixing mean score levels."""
    rng = rng or random.Random(0)
    cands = []
    for aid, pairs in per_aspect.items():
        n = len(pairs)
        if n < min_n: continue
        scores = [s for _, s in pairs]
        if len(set(scores)) < 2: continue
        mean = sum(scores) / n
        std = (sum((s - mean) ** 2 for s in scores) / n) ** 0.5
        if std < 0.15: continue   # too uniform
        cands.append((aid, n, mean, std))
    if len(cands) <= n_pick:
        return cands
    # Diverse mean buckets: low/mid/high
    cands.sort(key=lambda x: x[2])
    third = len(cands) // 3
    buckets = [cands[:third], cands[third:2*third], cands[2*third:]]
    picks = []
    for b in buckets:
        if b: picks.append(rng.choice(b))
        if len(picks) >= n_pick: break
    return picks[:n_pick]


PROMPT_TEMPLATE = """You are participating in a rubric-recovery experiment.

We applied a single hidden rubric to {k} texts. For each text, the rubric judge\
 assigned a score: 0.0, 0.5, or 1.0 (1.0 = strongly satisfies the rubric; 0.0 =\
 strongly violates). Your job: looking only at the texts and their scores,\
 propose what single rubric was being applied.

Hidden rubric properties:
- It is ONE specific quality dimension of the texts (not a meta-judgment like\
 "this is good")
- It has a clear, narrow scope (not "overall quality")
- A trained rater could agree on individual scores

Your answer must be a single rubric in JSON with two fields:
- "name": 3-8 words, like a section heading
- "description": 1-2 sentences, what the rubric checks

Output ONLY valid JSON, no commentary, no markdown fences.

---
Domain: {task}

The {k} (text, score) pairs (each text truncated to {text_max_chars} chars):
{pairs_block}

Now propose the hidden rubric. Output ONLY valid JSON:
{{"name": "...", "description": "..."}}
"""


def build_pairs_block(pairs, datapoints_by_id, text_max_chars=1500):
    parts = []
    for i, (dp_id, score) in enumerate(pairs):
        d = datapoints_by_id.get(dp_id)
        if not d: continue
        text = d["text"][:text_max_chars]
        ellip = "..." if len(d["text"]) > text_max_chars else ""
        parts.append(f"--- Text {i+1} | score = {score} ---\n{text}{ellip}\n")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["peer_review", "math"])
    ap.add_argument("--n-per-task", type=int, default=3,
                    help="aspects to pick per task")
    ap.add_argument("--k", type=int, default=30,
                    help="texts per recoverability prompt")
    ap.add_argument("--text-max-chars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    truth = {}   # filename -> {task, aspect_id, name, description}
    n_written = 0

    for task in args.tasks:
        per = collect_per_aspect(task)
        aspects = json.loads((V2 / task / "aspects.json").read_text())
        asp_by_id = {a["aspect_id"]: a for a in aspects}
        dps = json.loads((V2 / task / "datapoints.json").read_text())
        dp_by_id = {d["datapoint_id"]: d for d in dps}

        picks = pick_aspects(per, n_pick=args.n_per_task, min_n=args.k + 5, rng=rng)
        print(f"\n{task}: picked {len(picks)} aspects:")
        for aid, n, mean, std in picks:
            print(f"  {aid}: n={n}, mean={mean:.2f}, std={std:.2f} "
                  f"| {asp_by_id[aid]['name'][:60]}")

            # Sample k pairs, stratified across score values
            pairs = per[aid]
            by_score = collections.defaultdict(list)
            for p in pairs:
                by_score[p[1]].append(p)
            # Take proportional samples
            sampled = []
            for s, plist in by_score.items():
                want = round(args.k * len(plist) / len(pairs))
                rng.shuffle(plist)
                sampled.extend(plist[:want])
            # Pad if rounding lost a few
            if len(sampled) < args.k:
                extra = [p for p in pairs if p not in sampled]
                rng.shuffle(extra)
                sampled.extend(extra[:args.k - len(sampled)])
            sampled = sampled[:args.k]
            rng.shuffle(sampled)

            pairs_block = build_pairs_block(sampled, dp_by_id,
                                             text_max_chars=args.text_max_chars)
            prompt = PROMPT_TEMPLATE.format(
                k=len(sampled), task=task,
                text_max_chars=args.text_max_chars,
                pairs_block=pairs_block,
            )

            fname = f"{task}__{aid}__k{args.k}.txt"
            (OUT / fname).write_text(prompt)
            truth[fname] = {
                "task": task,
                "aspect_id": aid,
                "name": asp_by_id[aid]["name"],
                "description": asp_by_id[aid]["description"],
                "n_pairs": len(sampled),
                "mean_score": sum(s for _, s in sampled) / len(sampled),
            }
            n_written += 1

    (OUT / "_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"\nwrote {n_written} recoverability prompts to {OUT}")
    print(f"  truth at {OUT}/_truth.json")
    print(f"  sample prompt chars: {len((OUT / next(iter(truth))).read_text())}")


if __name__ == "__main__":
    main()
