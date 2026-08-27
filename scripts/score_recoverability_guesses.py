"""Score recoverability guesses with an LLM judge.

Reads:
  _recoverability/_truth.json (filename -> {task, aspect_id, name, description, ...})
  _recoverability/<file>.guess.json (single subagent guess: {name, description})

For each (truth, guess) pair: build a single Claude prompt that asks for a 0-3
semantic-match score with rationale.

We assemble all pairs into ONE prompt for a single judge subagent call,
to minimize cost and keep evaluations consistent.

Output:
  _recoverability/_judge_prompt.txt   <- the prompt to feed a subagent
  _recoverability/_judge_scores.json  <- (written by subagent) per-file score
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/Users/spangher/Projects/stanford-research/norm-research"
           "/runs/validity_full/v2/_recoverability")

JUDGE_PROMPT = """You are scoring rubric-recovery attempts.

For each pair below, you see:
- TRUTH: the actual rubric (name + description)
- GUESS: a subagent's attempt to recover what rubric produced certain labels,
  given only the labeled texts.

Score the GUESS on a 0-3 scale:
- 3 = essentially the same rubric (same construct, possibly different wording)
- 2 = clearly captures the core idea, partially overlapping construct
- 1 = related but materially different construct (correct general domain, wrong specific axis)
- 0 = unrelated or contradictory construct

Output ONLY valid JSON, no commentary, no markdown fences. Format:
{{"scores": [
  {{"key": "...", "score": 0-3, "reason": "short justification"}},
  ...
]}}

---
Pairs:
{pairs_block}
"""

PAIR_TEMPLATE = """=== {i}) key = {key} ===
TRUTH name:        {truth_name}
TRUTH description: {truth_desc}
GUESS name:        {guess_name}
GUESS description: {guess_desc}
"""


def main():
    truth = json.loads((OUT / "_truth.json").read_text())
    pairs = []
    keys_in_order = []
    for i, (fname, info) in enumerate(sorted(truth.items()), 1):
        key = fname.replace(".txt", "")
        guess_path = OUT / f"{key}.guess.json"
        if not guess_path.exists():
            print(f"SKIP {key}: no guess found")
            continue
        try:
            guess = json.loads(guess_path.read_text())
        except Exception as ex:
            print(f"SKIP {key}: guess unparseable ({ex})")
            continue
        pairs.append(PAIR_TEMPLATE.format(
            i=len(keys_in_order)+1, key=key,
            truth_name=info["name"], truth_desc=info["description"],
            guess_name=guess.get("name", "?"),
            guess_desc=guess.get("description", "?"),
        ))
        keys_in_order.append(key)

    prompt = JUDGE_PROMPT.format(pairs_block="\n".join(pairs))
    (OUT / "_judge_prompt.txt").write_text(prompt)
    (OUT / "_judge_keys.json").write_text(json.dumps(keys_in_order, indent=2))
    print(f"wrote judge prompt with {len(keys_in_order)} pairs to {OUT}/_judge_prompt.txt")
    print(f"  prompt chars: {len(prompt)}")


if __name__ == "__main__":
    main()
