"""Collect LLM_FIELDS from hybrid programs and emit extraction prompts for the sk3 batch.

Output: outputs/metric_seam_pilot/v1/field_prompts.jsonl
        (channel=field, aspect_id=<aid>__<field>, datapoint_id, prompt)
After scoring, split_field_results.py writes llm_fields/<aid>__<field>.json
"""
import importlib.util, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot/v1"
HYB = pathlib.Path(__file__).parent / "programs"

T = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""


def main():
    items = json.load(open(OUT / "items_v1.json"))
    n = 0
    with open(OUT / "field_prompts.jsonl", "w") as f:
        for prog in sorted(HYB.glob("*_h*.py")):
            aid = prog.stem.split("_")[0]
            spec = importlib.util.spec_from_file_location(prog.stem, prog)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                print(f"{prog.name}: load error {e}", file=sys.stderr)
                continue
            fields = getattr(mod, "LLM_FIELDS", {}) or {}
            cf_items = (json.load(open(OUT / "cf_items_a86.json"))
                        if aid == "a86" else [])
            for field, instruction in list(fields.items())[:2]:
                done = OUT / "llm_fields" / f"{aid}__{field}.json"
                if done.exists():
                    continue
                for it in items:
                    f.write(json.dumps({
                        "channel": "field",
                        "aspect_id": f"{aid}__{field}",
                        "datapoint_id": it["datapoint_id"],
                        "prompt": T.format(instruction=instruction,
                                           text=it["ctext"])}) + "\n")
                    n += 1
                for it in cf_items:  # CF texts too, so full-hybrid CF gate can use fields
                    f.write(json.dumps({
                        "channel": "field",
                        "aspect_id": f"a86cf__{field}",
                        "datapoint_id": it["datapoint_id"],
                        "prompt": T.format(instruction=instruction,
                                           text=it["ctext"])}) + "\n")
                    n += 1
                print(f"{aid}__{field}: queued")
    print(f"wrote {n} field prompts")


if __name__ == "__main__":
    main()
