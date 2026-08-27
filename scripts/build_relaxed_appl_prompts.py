"""Build relaxed-applicability re-scoring prompts for one task.

For each audience-appeal aspect in the task's aspects.json (matched by
keyword categories: hook/voice/novelty/emotion/payoff/audience/humor),
group into blocks of 15 and generate prompts for every datapoint already
in cells_v1 (so we re-score the audience-appeal aspects only).

Differences from the original prompt:
  * The rubric block contains ONLY audience-appeal aspects.
  * The system instructions explicitly relax applicability: "Default to
    applicable=True; mark non-applicable only if the work is structurally
    outside the criterion's domain."

Outputs:
  runs/cw_relax_appl/prompts/<task>__b15p_relax_<block>__p0__dd<dpid>.txt
  runs/cw_relax_appl/prompt_lists/<task>_shard_<i>.txt   (split across N gpus)
"""
import argparse
import json
import re
from pathlib import Path

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")

CATEGORIES = {
    "hook": [r"\bhook\b", r"opening", r"attention", r"grab", r"compelling start"],
    "voice": [r"\bvoice\b", r"narrative voice", r"authorial", r"persona", r"distinctive style"],
    "novelty": [r"novel", r"original", r"\bfresh\b", r"surprise", r"unexpected",
                r"twist", r"subver", r"unconventional", r"innovative"],
    "share": [r"shareab", r"viral", r"memorab", r"sticky", r"contagious"],
    "emotional": [r"emotional", r"resonance", r"\bimpact\b", r"poignan", r"moving",
                  r"heart", r"empathy.*reader", r"engagement"],
    "audience": [r"audience.*experien", r"reader.*experien", r"engag.*reader",
                 r"reader.*engag", r"reader.*invest"],
    "payoff": [r"payoff", r"punchline", r"climax", r"\btwist\b", r"reversal", r"reveal",
               r"\bturn\b", r"peripeteia"],
    "humor_mechanics": [r"setup.*punch", r"callback", r"comedic timing", r"incongru",
                        r"\babsurd", r"subvert"],
}


# --- relaxed system prompt ------------------------------------------------

RELAXED_SYSTEM = """\
You are an expert evaluator scoring creative artifacts on multiple
ENGAGEMENT-CRITERION rubrics simultaneously. The rubrics below are about
how the artifact *engages its reader* — hooks, voice, novelty, surprise,
payoff, emotional resonance, audience fit, and (where relevant) humor
mechanics.

For each rubric below, you are given:
  - A rubric NAME and an EXPANDED DEFINITION.
  - A WHAT-TO-LOOK-FOR checklist (concrete textual signals).
  - An APPLICABILITY note (when to mark non-applicable).
  - Three CALIBRATION EXEMPLARS (real excerpts) showing what a
    clearly-violating, neutral/non-applicable, and clearly-satisfying
    instance looks like, with explanations.

For each (text, rubric) pair, output ONE record:
  - "applicable": true if the rubric is engaged by the text's content.

    ** IMPORTANT — RELAXED APPLICABILITY INSTRUCTION **
    Default to "applicable: true" for these engagement-criterion rubrics.
    Mark non-applicable ONLY if the text is structurally outside the
    rubric's domain (e.g., the rubric is about "opening hook" and the
    text is a single-sentence fragment with no opening to assess; the
    rubric is about "comedic timing" and the text is plainly not humor).
    Do NOT mark non-applicable merely because the text doesn't announce
    itself as engagement-oriented. Every story has a hook (even a weak
    one); every voice can be assessed; every ending has an effect.

  - "score": one of 0.0, 0.5, 1.0
      0.0 = clearly violates / fails the criterion
      0.5 = borderline / partial / mixed signal
      1.0 = clearly satisfies
      (use null if applicable=false)
  - "reason": 5-15 words pointing to concrete textual feature(s).

Output VALID JSON ONLY in this exact shape:
{
  "results": [
    {"text_id": "d0", "scores": [
      {"aspect_id": "a0", "applicable": true,  "score": 1.0, "reason": "..."},
      ...
    ]},
    {"text_id": "d1", "scores": [...]},
    ...
  ]
}

Be strict but fair on the SCORE. Be GENEROUS on the APPLICABILITY for
these engagement criteria.
"""


def find_audience_aspects(task: str):
    arr = json.loads(
        (REPO / f"runs/validity_full/v2/{task}/aspects.json").read_text())
    matched_ids = set()
    for a in arr:
        text = (a.get("name", "") + " " + a.get("description", "")).lower()
        for pats in CATEGORIES.values():
            for pat in pats:
                if re.search(pat, text):
                    matched_ids.add(a["aspect_id"])
                    break
    print(f"[{task}] {len(matched_ids)} audience-appeal aspects "
          f"out of {len(arr)} total")
    return [a for a in arr if a["aspect_id"] in matched_ids]


def extract_aspect_block(aspects: list[dict]) -> str:
    """Format aspects in the same style as existing prompts."""
    out = []
    for a in aspects:
        out.append(f"### Rubric {a['aspect_id']}: {a['name']}")
        out.append(f"DEFINITION: {a.get('description', '')}")
        # The existing prompts include WHAT-TO-LOOK-FOR / APPLICABILITY /
        # CALIBRATION EXEMPLARS. The aspects.json doesn't have those fields
        # individually; we use just the description as the rubric content
        # because the relaxed-applicability instruction in the system
        # prompt dominates the application decision anyway.
        out.append("")
    return "\n".join(out)


def datapoint_text(task: str, dp_id: str) -> str:
    if not hasattr(datapoint_text, "_cache"):
        datapoint_text._cache = {}
    cache = datapoint_text._cache
    if task not in cache:
        dps = json.loads(
            (REPO / f"runs/validity_full/v2/{task}/datapoints.json").read_text())
        cache[task] = {d["datapoint_id"]: d.get("text", "") for d in dps}
    return cache[task].get(dp_id, "")


def scored_datapoints(task: str) -> list[str]:
    import pandas as pd
    dfs = []
    for j in ["qwen_thinking_fp8", "claude"]:
        f = REPO / f"outputs/v2_db/cells_v1/task={task}/judge={j}/data.parquet"
        if f.exists():
            dfs.append(pd.read_parquet(f)[["datapoint_id"]])
    if not dfs:
        return []
    return sorted(set(pd.concat(dfs)["datapoint_id"]))


def build_prompts(task: str, n_shards: int, block_size: int = 15,
                  max_text_chars: int = 4000):
    audience = find_audience_aspects(task)
    # Group into blocks of block_size
    blocks = [audience[i:i + block_size]
              for i in range(0, len(audience), block_size)]
    print(f"[{task}] grouped into {len(blocks)} blocks of up to {block_size}")

    dps = scored_datapoints(task)
    print(f"[{task}] {len(dps)} scored datapoints to re-score")

    out_dir = REPO / "runs/cw_relax_appl/prompts"
    list_dir = REPO / "runs/cw_relax_appl/prompt_lists"
    out_dir.mkdir(parents=True, exist_ok=True)
    list_dir.mkdir(parents=True, exist_ok=True)

    stems = []
    for bidx, block in enumerate(blocks):
        rubric_section = extract_aspect_block(block)
        aspect_ids = [a["aspect_id"] for a in block]
        for dp_id in dps:
            text = datapoint_text(task, dp_id)[:max_text_chars]
            prompt = RELAXED_SYSTEM + "\n=== USER ===\n"
            prompt += (
                f"You will score 1 TEXT on each of {len(block)} ENGAGEMENT "
                f"RUBRICS.\n\n"
                "================================================\n"
                f"RUBRICS ({len(block)} total)\n"
                "================================================\n"
            )
            prompt += rubric_section + "\n"
            prompt += (
                "================================================\n"
                f"TEXTS TO SCORE (1 total)\n"
                "================================================\n"
                f"--- {dp_id} ---\n{text}\n\n"
                "================================================\n"
                "TASK\n"
                "================================================\n"
                f"For EACH text in ['{dp_id}'],\n"
                f"score it on EACH rubric in {aspect_ids}.\n\n"
                "Output the JSON now (one entry per text, with one score-record per rubric).\n"
            )
            stem = f"{task}__b15p_relax_{bidx}__p0__dd{dp_id[1:]}"
            (out_dir / f"{stem}.txt").write_text(prompt)
            stems.append(stem)
    print(f"[{task}] wrote {len(stems)} prompt files -> {out_dir}")

    # Shard the stem list
    shards = [[] for _ in range(n_shards)]
    for i, s in enumerate(stems):
        shards[i % n_shards].append(s)
    for i, sh in enumerate(shards):
        (list_dir / f"{task}_shard_{i}.txt").write_text("\n".join(sh) + "\n")
    print(f"[{task}] sharded across {n_shards} → "
          f"~{len(shards[0])} prompts/shard")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative_writing")
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--block_size", type=int, default=15)
    args = ap.parse_args()
    build_prompts(args.task, args.n_shards, args.block_size)


if __name__ == "__main__":
    main()
