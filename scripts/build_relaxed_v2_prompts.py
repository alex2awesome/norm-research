"""Build careful, task-specific relaxed-applicability prompts for the 260
RELAX+BORDERLINE creative_writing aspects (from cw_aspect_categorization.json).

For each scored datapoint (2611 total), generate one prompt per block of 15
aspects → 2611 × 18 ≈ 47K prompts. Sharded across the available GPUs.

Critical checks the builder applies:
  - System prompt is creative-writing-specific (no peer-review boilerplate).
  - Rubric block uses the aspects.json name + description verbatim.
  - Output schema example uses actual datapoint_id format ("d#####").
  - Aspect ID list in task instruction matches the rubric block exactly.
  - Datapoint text is truncated only at a sentence/paragraph boundary
    if necessary (preserve reading flow for the judge).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
TASK = "creative_writing"
CATEGORIZATION_FILE = REPO / "outputs/v2_analysis/cw_aspect_categorization.json"
ASPECTS_FILE = REPO / f"runs/validity_full/v2/{TASK}/aspects.json"
DATAPOINTS_FILE = REPO / f"runs/validity_full/v2/{TASK}/datapoints.json"

# Use prefix "v2relax" so output dir is distinct from the earlier batch.
PROMPT_DIR = REPO / "runs/cw_relax_appl/v2relax_prompts"
LIST_DIR = REPO / "runs/cw_relax_appl/v2relax_lists"


SYSTEM_PROMPT = """\
You are an expert evaluator of CREATIVE WRITING — short fiction posted to
platforms where readers vote on what they enjoy. You will score one piece
of fiction on multiple craft rubrics. The rubrics below cover plot,
character, prose style, voice, structure, openings/endings, emotional
resonance, and reader engagement.

For each rubric below, you are given:
  - A rubric NAME and an EXPANDED DEFINITION (what the rubric measures).

For each (text, rubric) pair, output ONE record:

  - "applicable": true if the rubric is meaningfully assessable on this text.

    *** IMPORTANT — RELAXED APPLICABILITY INSTRUCTION ***
    Default to "applicable: true" for the rubrics in this prompt. These
    rubrics describe craft and engagement concerns that apply to nearly
    any piece of fiction.

    Mark non-applicable ONLY if the text is structurally outside the
    rubric's domain. Examples:
       * Rubric about "opening hook effectiveness" + text is a single
         disconnected sentence with no recognizable opening.
       * Rubric about "ending payoff" + text breaks off mid-scene with
         no attempt at closure.
       * Rubric about "dialogue craft" + text contains zero dialogue.

    Do NOT mark non-applicable merely because the text:
       - is short, unpolished, or amateur in tone
       - is genre fiction rather than literary fiction
       - doesn't announce itself as engagement-oriented
       - is reddit-style prose rather than published-magazine style

    Every story has a plot (even thin), a voice (even weak), an opening,
    an ending, characters, and a tonal stance — all assessable.

    *** Note on classical / genre framing in rubric text ***
    Several rubrics below come from classical or Aristotelian poetics
    and use terms like "tragedy", "epic", "mythos", "catharsis". Read
    these as the UNIVERSAL PRINCIPLE the rubric encodes, not as a genre
    restriction. For example:
       * "Tragedy aims to arouse pity and fear" → does THIS story
         produce earned emotional response from the reader?
       * "Tragic unity over epic sprawl" → does THIS story maintain
         unified scope rather than sprawl?
       * "Primacy of plot (mythos)" → does THIS story have a plot that
         drives events more than spectacle or static character?
       * A rubric mentioning "the marvelous" / "speculative elements"
         → assess the rubric's principle (verisimilitude, coherence)
         even on realist fiction.
    Apply the principle universally; the classical vocabulary is the
    rubric's SOURCE, not its DOMAIN. A common failure to avoid is
    marking applicable=false because the rubric text invokes a specific
    genre when the underlying craft concern applies to the work at hand.

  - "score": one of 0.0, 0.5, 1.0
      0.0 = clearly fails / violates the rubric
      0.5 = borderline / partial / mixed signal / hard to tell
      1.0 = clearly satisfies the rubric
      (use null only when applicable=false)

  - "reason": 5-15 words pointing to concrete textual feature(s) driving
    the score — quote phrases or name specific moments from the text.

Output VALID JSON ONLY in this exact shape (no markdown, no commentary):

{
  "results": [
    {"text_id": "<DPID>", "scores": [
      {"aspect_id": "<AID>", "applicable": true,  "score": 1.0, "reason": "..."},
      {"aspect_id": "<AID>", "applicable": false, "score": null, "reason": "..."}
    ]}
  ]
}

Be STRICT on the SCORE (don't give 1.0 unless the text clearly satisfies
the rubric — use 0.5 for mixed). Be GENEROUS on the APPLICABILITY (default
to applicable=true unless the text is structurally outside scope).
"""


def load_aspects_to_rescore() -> list[dict]:
    cats = json.loads(CATEGORIZATION_FILE.read_text())
    relax_ids = set(cats["RELAX"]) | set(cats["BORDERLINE"])
    all_aspects = json.loads(ASPECTS_FILE.read_text())
    return [a for a in all_aspects if a["aspect_id"] in relax_ids]


def load_scored_datapoints() -> list[tuple[str, str]]:
    """Return list of (datapoint_id, text) for datapoints already in cells_v1
    (we want to re-score those, not new ones)."""
    import pandas as pd
    judges = ["qwen_thinking_fp8", "claude"]
    dps = set()
    for j in judges:
        f = REPO / f"outputs/v2_db/cells_v1/task={TASK}/judge={j}/data.parquet"
        if f.exists():
            dps |= set(pd.read_parquet(f, columns=["datapoint_id"])["datapoint_id"])
    dps_all = json.loads(DATAPOINTS_FILE.read_text())
    dp_text = {d["datapoint_id"]: d.get("text", "") for d in dps_all}
    out = []
    for dp_id in sorted(dps):
        text = dp_text.get(dp_id, "")
        if text:
            out.append((dp_id, text))
    return out


def truncate_text(text: str, max_chars: int = 4000) -> str:
    """Truncate text at a paragraph boundary if possible, then sentence boundary."""
    if len(text) <= max_chars:
        return text
    # Try paragraph break
    head = text[:max_chars]
    p = head.rfind("\n\n")
    if p > max_chars * 0.7:
        return text[:p].rstrip()
    # Try sentence break
    for terminator in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
        p = head.rfind(terminator)
        if p > max_chars * 0.7:
            return text[:p + 1].rstrip()
    # Fallback: hard cut
    return head


def format_rubric_block(aspects: list[dict]) -> str:
    """Format aspects as in the existing prompt style."""
    out = []
    for a in aspects:
        out.append(f"### Rubric {a['aspect_id']}: {a['name']}")
        desc = (a.get("description") or "").strip()
        out.append(f"DEFINITION: {desc}")
        out.append("")
    return "\n".join(out)


def build_prompt(dp_id: str, text: str, aspects: list[dict]) -> str:
    aspect_ids = [a["aspect_id"] for a in aspects]
    rubric_section = format_rubric_block(aspects)
    user = (
        f"You will score 1 TEXT on each of {len(aspects)} CRAFT RUBRICS.\n\n"
        f"================================================\n"
        f"RUBRICS ({len(aspects)} total)\n"
        f"================================================\n"
        f"{rubric_section}\n"
        f"================================================\n"
        f"TEXT TO SCORE (1 total)\n"
        f"================================================\n"
        f"--- {dp_id} ---\n{truncate_text(text)}\n\n"
        f"================================================\n"
        f"TASK\n"
        f"================================================\n"
        f"For the text \"{dp_id}\", score it on EACH of the following rubrics:\n"
        f"  {aspect_ids}\n\n"
        f"Output the JSON now. The text_id MUST be exactly \"{dp_id}\".\n"
        f"Every rubric in the list above MUST appear in the scores array.\n"
    )
    return SYSTEM_PROMPT + "\n=== USER ===\n" + user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_size", type=int, default=15)
    ap.add_argument("--n_shards", type=int, default=6)
    ap.add_argument("--sample_only", action="store_true",
                    help="Build 2 sample prompts and print them for review.")
    args = ap.parse_args()

    aspects = load_aspects_to_rescore()
    blocks = [aspects[i:i + args.block_size]
              for i in range(0, len(aspects), args.block_size)]
    print(f"aspects to re-score: {len(aspects)}  -> {len(blocks)} blocks of "
          f"≤ {args.block_size}")

    dps = load_scored_datapoints()
    print(f"datapoints: {len(dps)}")
    total = len(blocks) * len(dps)
    print(f"total prompts: {total:,}")

    if args.sample_only:
        # Print one full prompt for each of two different blocks
        dp_id, text = dps[0]
        for bi in [0, len(blocks) // 2]:
            p = build_prompt(dp_id, text, blocks[bi])
            print(f"\n{'=' * 80}\nSAMPLE PROMPT  block {bi}, dp {dp_id} "
                  f"({len(blocks[bi])} aspects)\n{'=' * 80}\n")
            print(p[:6000])
            if len(p) > 6000:
                print(f"\n... [truncated; full length {len(p)} chars]")
        return

    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    LIST_DIR.mkdir(parents=True, exist_ok=True)

    stems = []
    for bi, block in enumerate(blocks):
        for dp_id, text in dps:
            stem = f"{TASK}__b15p_v2relax_{bi}__p0__dd{dp_id[1:]}"
            (PROMPT_DIR / f"{stem}.txt").write_text(build_prompt(dp_id, text, block))
            stems.append(stem)
    print(f"wrote {len(stems)} prompts -> {PROMPT_DIR}")

    shards = [[] for _ in range(args.n_shards)]
    for i, s in enumerate(stems):
        shards[i % args.n_shards].append(s)
    for i, sh in enumerate(shards):
        (LIST_DIR / f"shard_{i}.txt").write_text("\n".join(sh) + "\n")
    print(f"sharded across {args.n_shards} → ~{len(shards[0])} prompts/shard")


if __name__ == "__main__":
    main()
