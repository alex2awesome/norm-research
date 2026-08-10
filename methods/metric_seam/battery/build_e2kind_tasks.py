"""E2-KIND prompt builder for fleet tasks (scale-up of build_e2kind_prompts.py).

Same design as the PR pilot: cells 4/5/6 per e2-bearing field, documents taken
VERBATIM from the task's battery stip prompts (same items, same truncation), so cells
are text-identical to the run E2 rows except for the instruction.

Leak guard: refuses to emit a nonce-cell prompt whose INSTRUCTION contains the
community field name's content words (defense in depth on top of author validation).

Usage: python3 build_e2kind_tasks.py <task>
-> outputs/metric_seam_pilot/tasks/<task>/e2kind_prompts.jsonl
"""
import json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"

T = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""

DOC_RE = re.compile(r"<document>\n(.*)\n</document>", re.DOTALL)
STOP = {"the", "a", "an", "of", "with", "and", "or", "verdict", "label", "mode",
        "quality", "type", "kind", "class"}


def content_words(field):
    return [w for w in field.split("_") if len(w) > 3 and w not in STOP]


def main():
    task = sys.argv[1]
    kv = json.load(open(BASE / f"battery/variants_e2kind_{task}.json"))
    outdir = BASE / "tasks" / task

    docs = {}
    for line in open(outdir / "battery_prompts.jsonl"):
        r = json.loads(line)
        aidc, field = r["aspect_id"].split("__", 1)
        if not aidc.endswith(".stip"):
            continue
        aid = aidc.split(".")[0]
        m = DOC_RE.search(r["prompt"])
        if m:
            docs[(aid, field, r["datapoint_id"])] = m.group(1)

    n = {"cell4": 0, "cell5": 0, "cell6": 0}
    leaks = []
    with open(outdir / "e2kind_prompts.jsonl", "w") as f:
        for key, v in kv.items():
            if key.startswith("_"):
                continue
            aid, field = key.split("__", 1)
            cells = {"cell4": v["cell4_nonce_deviant"],
                     "cell5": v["cell5_name_neutral"],
                     "cell6": v["cell6_nonce_neutral"]}
            for cell, ins in cells.items():
                if cell in ("cell4", "cell6"):
                    for w in content_words(field):
                        if re.search(r"\b" + re.escape(w[:6]), ins, re.I):
                            leaks.append((key, cell, w))
                for (a, fl, d), text in docs.items():
                    if a != aid or fl != field:
                        continue
                    f.write(json.dumps({
                        "channel": "field",
                        "aspect_id": f"{aid}.{cell}__{field}",
                        "datapoint_id": d,
                        "prompt": T.format(instruction=ins, text=text)}) + "\n")
                    n[cell] += 1
    if leaks:
        print("LEAK WARNINGS (community stem in nonce-cell instruction):")
        for l in leaks:
            print("  ", l)
    print(f"{task}: {n} (total {sum(n.values())}) -> tasks/{task}/e2kind_prompts.jsonl")


if __name__ == "__main__":
    main()
