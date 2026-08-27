"""LOCAL API PROBE prompt builder (small-scale battery pilot while sk3 is down).

PR only, held-out test items only (100), criteria = the four e2-bearing high-fm ones.
Conditions: full (original instruction) / keyname / keynonce / stip.
The 'full' condition is extracted per probe model and serves as that extractor's own
reference (within-extractor comparisons; the certified Gemma extraction stays the LCC
anchor). Docs truncated to 12k chars to bound API cost.

-> outputs/metric_seam_pilot/battery/probe/probe_prompts_pr.jsonl        (all conditions)
-> outputs/metric_seam_pilot/battery/probe/probe_prompts_pr_full.jsonl   (full only, for
   scale-rung extractors that only need the original instruction)
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
from harness import split_ids  # noqa: E402

CRITERIA = ["a76", "a87", "a104", "a112"]
MAXC = 12000

T = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""


def main():
    inv = json.load(open(BASE / "battery/inventory.json"))["press_releases"]
    variants = json.load(open(BASE / "battery/variants_press_releases.json"))
    items = {x["datapoint_id"]: x["ctext"]
             for x in json.load(open(BASE / "v1/items_v1.json"))}
    _, test = split_ids()
    test = sorted(test)

    outdir = BASE / "battery/probe"
    outdir.mkdir(exist_ok=True)
    n = {"full": 0, "keyname": 0, "keynonce": 0, "stip": 0}
    fall = open(outdir / "probe_prompts_pr.jsonl", "w")
    ffull = open(outdir / "probe_prompts_pr_full.jsonl", "w")
    for aid in CRITERIA:
        for field, fmeta in inv[aid]["fields"].items():
            v = variants.get(f"{aid}__{field}", {})
            conds = {"full": fmeta["instruction"],
                     "keyname": v.get("nameonly_instruction"),
                     "keynonce": v.get("nonce_instruction")}
            if v.get("e2"):
                conds["stip"] = v["e2"]["deviant_instruction"]
            for cond, ins in conds.items():
                if not ins:
                    continue
                for d in test:
                    row = json.dumps({"channel": "field",
                                      "aspect_id": f"{aid}.{cond}__{field}",
                                      "datapoint_id": d,
                                      "prompt": T.format(instruction=ins,
                                                         text=items[d][:MAXC])})
                    fall.write(row + "\n")
                    if cond == "full":
                        ffull.write(row + "\n")
                    n[cond] += 1
    fall.close()
    ffull.close()
    print(f"probe prompts: {n} (total {sum(n.values())}) -> {outdir}")


if __name__ == "__main__":
    main()
