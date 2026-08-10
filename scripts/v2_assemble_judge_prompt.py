"""Assemble multi-rubric judge prompts from enrichment files + bundle defs.

Each call scores B aspects (bundle) × T texts (chunk) at once.
Three-point scale {0, 0.5, 1}, applicability gate, brief reasoning per score.

Usage as library:
  from v2_assemble_judge_prompt import build_prompt
  system, user = build_prompt(bundle, enrichments, texts)
"""
from __future__ import annotations

import json
from pathlib import Path


# ============================================================
# System prompt template
# ============================================================
SYSTEM_PROMPT = """You are an expert peer-review evaluator scoring scientific papers on multiple evaluation rubrics simultaneously.

For each rubric below, you are given:
  - A rubric NAME and an EXPANDED DEFINITION (what it means, what counts, edge cases).
  - A WHAT-TO-LOOK-FOR checklist (concrete textual signals).
  - An APPLICABILITY note (when to mark non-applicable).
  - Three CALIBRATION EXEMPLARS (real paper excerpts) showing what a clearly-violating, neutral/non-applicable, and clearly-satisfying instance looks like, with explanations.

For each (text, rubric) pair, output ONE record:
  - "applicable": true if the rubric is engaged by the text's content, false if the text is on a topic where the rubric has no purchase (e.g., a rubric about clinical trial reporting on an ML methodology paper).
  - "score": one of 0.0, 0.5, 1.0
      0.0 = clearly violates / fails to engage when it should
      0.5 = borderline / partial / mixed signal
      1.0 = clearly satisfies
      (use null if applicable=false)
  - "reason": 5-15 words pointing to the concrete textual feature(s) driving the score.

Output VALID JSON ONLY in this exact shape:
{
  "results": [
    {"text_id": "d0", "scores": [
      {"aspect_id": "a0", "applicable": true,  "score": 1.0, "reason": "explicit methods section with power analysis"},
      {"aspect_id": "a3", "applicable": false, "score": null, "reason": "purely theoretical paper, no empirical claims"},
      ...
    ]},
    {"text_id": "d1", "scores": [...]},
    ...
  ]
}

Be strict but fair. The 0.5 score is FOR mixed signals AND for "couldn't tell" — don't punish a paper for not engaging with a rubric outside its scope (mark applicable=false instead)."""


# ============================================================
# Block builders
# ============================================================
def _format_aspect_block(enr: dict) -> str:
    """One rubric's enrichment as a prompt block."""
    aid = enr["aspect_id"]
    name = enr["name"]
    edef = enr.get("expanded_definition", "")
    wtlf = enr.get("what_to_look_for", [])
    appl = enr.get("applicability_note", "")
    ex = enr.get("exemplars", {})

    out = [f"### Rubric {aid}: {name}",
           f"DEFINITION: {edef}"]
    if wtlf:
        out.append("WHAT TO LOOK FOR:")
        for b in wtlf:
            out.append(f"  - {b}")
    if appl:
        out.append(f"APPLICABILITY: {appl}")
    out.append("CALIBRATION EXEMPLARS:")
    for kind, label in [("violates", "0.0 (violates)"),
                         ("neutral", "0.5 (neutral/N-A)"),
                         ("satisfies", "1.0 (satisfies)")]:
        e = ex.get(kind, {})
        if not e: continue
        excerpt = (e.get("text_excerpt") or "").strip().replace("\n", " ")[:350]
        explain = (e.get("explanation") or "").strip()
        out.append(f"  [{label}] \"{excerpt}\"")
        out.append(f"    → {explain}")
    return "\n".join(out)


def _format_text_block(text_id: str, text: str, max_chars: int = 4000) -> str:
    text = (text or "").strip().replace("\r", "")
    if len(text) > max_chars:
        text = text[:max_chars] + " [...TRUNCATED]"
    return f"--- {text_id} ---\n{text}"


def build_prompt(bundle: dict, enrichments: list[dict],
                  texts: list[tuple[str, str]],
                  text_max_chars: int = 4000) -> tuple[str, str]:
    """
    bundle: {"bundle_id": "b0", "aspect_ids": [...], "aspects": [...]}
    enrichments: list of enrichment dicts (one per aspect in bundle, ordered)
    texts: [(text_id, text), ...]
    Returns (system_prompt, user_message).
    """
    aspect_blocks = [_format_aspect_block(e) for e in enrichments]
    aspects_section = "\n\n".join(aspect_blocks)

    text_blocks = [_format_text_block(tid, t, text_max_chars) for tid, t in texts]
    texts_section = "\n\n".join(text_blocks)

    aspect_ids = [e["aspect_id"] for e in enrichments]
    text_ids = [tid for tid, _ in texts]

    user = f"""You will score {len(texts)} TEXTS on each of {len(enrichments)} RUBRICS.

================================================
RUBRICS ({len(enrichments)} total)
================================================
{aspects_section}

================================================
TEXTS TO SCORE ({len(texts)} total)
================================================
{texts_section}

================================================
TASK
================================================
For EACH text in {text_ids},
score it on EACH rubric in {aspect_ids}.

Output the JSON now (one entry per text, with one score-record per rubric)."""

    return SYSTEM_PROMPT, user


# ============================================================
# Standalone test
# ============================================================
def _smoke_test():
    v2 = Path("runs/validity_full/full_v2")
    bundles = json.loads((v2 / "judge_bundles.json").read_text())
    enr_dir = v2 / "judge_enrichment"
    datapoints = json.loads((v2 / "datapoints.json").read_text())

    # Find first bundle with at least 3 enrichments ready
    for b in bundles:
        enrs = []
        for aid in b["aspect_ids"]:
            p = enr_dir / f"{aid}.json"
            if p.exists():
                enrs.append(json.loads(p.read_text()))
        if len(enrs) >= 3:
            break
    else:
        print("No bundles ready yet — wait for subagents to write enrichments")
        return

    texts = [(d["datapoint_id"], d["text"]) for d in datapoints[:5]]
    system, user = build_prompt(b, enrs, texts)
    print("=== SYSTEM ===")
    print(system[:500])
    print(f"... [total system: {len(system)} chars]")
    print()
    print("=== USER (first 2000 chars) ===")
    print(user[:2000])
    print(f"... [total user: {len(user)} chars]")
    print()
    print(f"=== STATS ===")
    print(f"  bundle: {b['bundle_id']}  aspects in bundle: {len(b['aspect_ids'])}  enrichments loaded: {len(enrs)}")
    print(f"  texts in chunk: {len(texts)}")
    print(f"  est tokens (system+user): {(len(system) + len(user)) // 4}")


if __name__ == "__main__":
    _smoke_test()
