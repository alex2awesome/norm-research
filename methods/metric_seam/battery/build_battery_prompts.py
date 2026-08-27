"""Assemble E1 (KEY) + E2 (STIP) extraction prompts from agent-authored variants.

Reads outputs/metric_seam_pilot/battery/variants_<task>.json (agent-authored, gated) and
the task's items; emits battery prompts with namespaced aspect_ids:
  {aid}.keyname__{field}   E1 condition A (name-only instruction)
  {aid}.keynonce__{field}  E1 condition B (nonce name + full operational definition)
  {aid}.stip__{field}      E2 (deviant stipulated definition)
Same field template as build_field_prompts_task.py so extraction is apples-to-apples
with the certified condition.

Usage: python3 build_battery_prompts.py <task>
-> outputs/metric_seam_pilot/{v2|tasks/<task>}/battery_prompts.jsonl
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"

T = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""

ITEMS = {"press_releases": BASE / "v1/items_v1.json"}
OUTDIR = {"press_releases": BASE / "v2"}


def main():
    task = sys.argv[1]
    items_p = ITEMS.get(task, BASE / "tasks" / task / "items.json")
    outdir = OUTDIR.get(task, BASE / "tasks" / task)
    variants = json.load(open(BASE / "battery" / f"variants_{task}.json"))
    items = json.load(open(items_p))

    out = outdir / "battery_prompts.jsonl"
    n = {"keyname": 0, "keynonce": 0, "stip": 0}
    with open(out, "w") as f:
        for key, v in variants.items():
            aid, field = v["aid"], v["field"]
            conds = {"keyname": v["nameonly_instruction"],
                     "keynonce": v["nonce_instruction"]}
            if v.get("e2"):
                conds["stip"] = v["e2"]["deviant_instruction"]
            for cond, ins in conds.items():
                for it in items:
                    f.write(json.dumps({
                        "channel": "field",
                        "aspect_id": f"{aid}.{cond}__{field}",
                        "datapoint_id": it["datapoint_id"],
                        "prompt": T.format(instruction=ins, text=it["ctext"])}) + "\n")
                    n[cond] += 1
    print(f"{task}: {sum(n.values())} prompts ({n}) -> {out}")


if __name__ == "__main__":
    main()
