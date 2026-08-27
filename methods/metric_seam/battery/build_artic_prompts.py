"""E8-ARTIC prompt builder — pre-registered articulation battery.

Each extractor family is asked, for every DISTINCT field it extracts across all 5
tasks, to articulate (WITHOUT seeing any document) the working definition / decision
rule it applies for that field. We later compare articulations across families
(gemma / llama-70B / qwen wrappers run this same prompt file on sk3).

Source field-prompt inventories (one row per distinct aspect_id = "<aid>__<field>"):
  - fleet tasks: outputs/metric_seam_pilot/tasks/<task>/field_prompts.jsonl
    for task in {creative_writing, math, humor, legal_title_vii}
  - press releases (v2 machinery): outputs/metric_seam_pilot/v2/field_prompts_v2.jsonl

All 5 files share the identical field-prompt template:
  "From the document below, {instruction}\n\n<document>\n{text}\n</document>\n\n
   Reply with ONE short line (max 20 words): the answer only, or NONE if absent."

We take the FIRST prompt seen per distinct aspect_id, regex out {instruction}, and
emit a FIXED articulation prompt (same wrapper text for every field/task — only the
embedded {instruction} varies).

Output: outputs/metric_seam_pilot/battery/artic_prompts.jsonl
  {"channel": "field", "aspect_id": "<task>::<aid>__<field>", "datapoint_id": "artic",
   "prompt": <articulation prompt>}

This script ONLY builds prompts. It does not touch sk3 and does not run any model.

Usage: python3 build_artic_prompts.py
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"

# task label -> source field_prompts file
FIELD_FILES = {
    "creative_writing": BASE / "tasks/creative_writing/field_prompts.jsonl",
    "math": BASE / "tasks/math/field_prompts.jsonl",
    "humor": BASE / "tasks/humor/field_prompts.jsonl",
    "legal_title_vii": BASE / "tasks/legal_title_vii/field_prompts.jsonl",
    "press_releases": BASE / "v2/field_prompts_v2.jsonl",
}

OUT_PATH = BASE / "battery/artic_prompts.jsonl"

# Matches the fixed field-prompt template shared by all 5 sources; captures the
# {instruction} span between the fixed prefix and the <document> opening tag.
INSTR_PAT = re.compile(r"^From the document below, (.*?)\n\n<document>\n", re.DOTALL)
TAIL_PAT = re.compile(
    r"\n</document>\n\nReply with ONE short line \(max 20 words\): "
    r"the answer only, or NONE if absent\.\s*$"
)

ARTIC_TEMPLATE = (
    "You are a field extractor in a document-evaluation pipeline. For each document "
    "you are asked:\n\nFrom the document below, {instruction}\n\nWithout seeing any "
    "document, articulate the working definition / decision rule you apply when "
    "answering. In 2-4 sentences: what exactly do you look for in the text, and what "
    "makes you answer one way rather than another? Reply with ONLY the rule, no "
    "preamble."
)


def load_first_per_aspect(path):
    """Return {aspect_id: prompt} using the FIRST row seen per distinct aspect_id,
    preserving first-seen order."""
    seen = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            aid = row["aspect_id"]
            if aid not in seen:
                seen[aid] = row["prompt"]
    return seen


def extract_instruction(prompt, task, aid):
    m = INSTR_PAT.match(prompt)
    if not m:
        raise ValueError(f"[{task}::{aid}] prompt does not match expected template "
                          f"prefix; first 200 chars: {prompt[:200]!r}")
    if not TAIL_PAT.search(prompt):
        raise ValueError(f"[{task}::{aid}] prompt does not match expected template "
                          f"tail; last 200 chars: {prompt[-200:]!r}")
    return m.group(1).strip()


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    expected_counts = {
        "press_releases": 39, "creative_writing": 71, "math": 48,
        "humor": 62, "legal_title_vii": 39,
    }

    per_task_rows = {}
    n_written = 0
    bad_leak = []
    bad_short = []
    examples = {}

    with open(OUT_PATH, "w") as out_f:
        for task, path in FIELD_FILES.items():
            by_aid = load_first_per_aspect(path)
            rows = []
            for aid, src_prompt in by_aid.items():
                instruction = extract_instruction(src_prompt, task, aid)
                artic_prompt = ARTIC_TEMPLATE.format(instruction=instruction)

                if "<document>" in artic_prompt or len(artic_prompt) > 1500:
                    bad_leak.append((task, aid, len(artic_prompt)))
                if len(instruction) < 20:
                    bad_short.append((task, aid, len(instruction)))

                out_aid = f"{task}::{aid}"
                row = {"channel": "field", "aspect_id": out_aid,
                       "datapoint_id": "artic", "prompt": artic_prompt}
                out_f.write(json.dumps(row) + "\n")
                rows.append(row)
                n_written += 1

            per_task_rows[task] = rows
            if task not in examples and rows:
                examples[task] = rows[0]

    # ---- validation ----
    print("=== E8-ARTIC build validation ===")
    print(f"-> {OUT_PATH}  ({n_written} total rows)\n")

    print("1) Field counts per task (distinct aspect_ids):")
    any_deviation = False
    for task, rows in per_task_rows.items():
        exp = expected_counts.get(task)
        n = len(rows)
        flag = ""
        if exp is not None and abs(n - exp) > max(3, round(0.15 * exp)):
            flag = "  <-- DEVIATION from expected ~%d" % exp
            any_deviation = True
        print(f"   {task:18s} n={n:4d}  (expected ~{exp}){flag}")
    if not any_deviation:
        print("   OK: all counts within tolerance of expectations.")

    print("\n2) Leaked <document> / oversized (>1500 chars) prompts:")
    if bad_leak:
        print(f"   FAIL: {len(bad_leak)} rows")
        for t, a, ln in bad_leak[:10]:
            print(f"     {t}::{a}  len={ln}")
    else:
        print("   OK: 0 rows.")

    print("\n3) Empty/truncated instructions (< 20 chars):")
    if bad_short:
        print(f"   FAIL: {len(bad_short)} rows")
        for t, a, ln in bad_short[:10]:
            print(f"     {t}::{a}  len={ln}")
    else:
        print("   OK: 0 rows.")

    print("\n4) Example prompts (first row per task):")
    for task in ("creative_writing", "math", "press_releases", "humor", "legal_title_vii"):
        row = examples.get(task)
        if row is None:
            continue
        print(f"\n--- {task} :: {row['aspect_id']} ---")
        print(row["prompt"])

    print("\n=== done ===")


if __name__ == "__main__":
    main()
