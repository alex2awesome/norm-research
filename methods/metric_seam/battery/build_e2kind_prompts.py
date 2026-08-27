"""E2-KIND prompt builder — grid cells 4/5/6 (seam note §7), PR pilot.

Cells (per e2-bearing field, from variants_e2kind_pr.json):
  cell4 nonce+deviant   — deviant rule, community name replaced by nonce
  cell5 name+neutral    — real name, arbitrary fresh-label rule
  cell6 nonce+neutral   — nonce name, same fresh-label rule (pure execution control)

Documents are taken VERBATIM from the existing battery stip prompts (same items, same
truncation) so cells are text-identical to the run E2 rows except for the instruction.

Usage: python3 build_e2kind_prompts.py
-> outputs/metric_seam_pilot/v2/e2kind_prompts.jsonl
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


def main():
    kv = json.load(open(BASE / "battery/variants_e2kind_pr.json"))
    docs = {}  # (aid, field, dpid) -> doc text from the stip prompt
    for line in open(BASE / "v2/battery_prompts.jsonl"):
        r = json.loads(line)
        aidc, field = r["aspect_id"].split("__", 1)
        if not aidc.endswith(".stip"):
            continue
        aid = aidc.split(".")[0]
        m = DOC_RE.search(r["prompt"])
        if m:
            docs[(aid, field, r["datapoint_id"])] = m.group(1)

    n = {"cell4": 0, "cell5": 0, "cell6": 0}
    smoke = []
    with open(BASE / "v2/e2kind_prompts.jsonl", "w") as f:
        for key, v in kv.items():
            aid, field = key.split("__", 1)
            cells = {"cell4": v["cell4_nonce_deviant"],
                     "cell5": v["cell5_name_neutral"],
                     "cell6": v["cell6_nonce_neutral"]}
            for cell, ins in cells.items():
                for (a, fl, d), text in docs.items():
                    if a != aid or fl != field:
                        continue
                    row = {"channel": "field",
                           "aspect_id": f"{aid}.{cell}__{field}",
                           "datapoint_id": d,
                           "prompt": T.format(instruction=ins, text=text)}
                    f.write(json.dumps(row) + "\n")
                    n[cell] += 1
                    if n[cell] == 1:
                        smoke.append(row["prompt"][:200])
    print(f"e2kind prompts: {n} (total {sum(n.values())}) -> v2/e2kind_prompts.jsonl")
    for s in smoke[:2]:
        print("SMOKE:", s.replace("\n", " ")[:180])


if __name__ == "__main__":
    main()
