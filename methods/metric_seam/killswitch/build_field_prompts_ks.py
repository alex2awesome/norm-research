"""Collect LLM_FIELDS from kill-switch hybrid programs -> Gemma extraction prompts.

Output: outputs/metric_seam_pilot/killswitch/field_prompts_ks.jsonl
        (channel=field, aspect_id=<pid>__<field>, datapoint_id, prompt)
"""
import importlib.util, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot/killswitch"
HYB = pathlib.Path(__file__).parent / "programs_ks"

T = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""


def main():
    items = json.load(open(ROOT / "outputs/metric_seam_pilot/v1/items_v1.json"))
    n = 0
    with open(OUT / "field_prompts_ks.jsonl", "w") as f:
        for prog in sorted(HYB.glob("p9*_h*.py")):
            pid = prog.stem.split("_")[0]
            spec = importlib.util.spec_from_file_location(prog.stem, prog)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                print(f"{prog.name}: load error {e}", file=sys.stderr)
                continue
            fields = getattr(mod, "LLM_FIELDS", {}) or {}
            for field, instruction in list(fields.items())[:2]:
                for it in items:
                    f.write(json.dumps({
                        "channel": "field", "aspect_id": f"{pid}__{field}",
                        "datapoint_id": it["datapoint_id"],
                        "prompt": T.format(instruction=instruction,
                                           text=it["ctext"])}) + "\n")
                    n += 1
            print(f"{prog.stem}: {len(fields)} fields")
    print(f"field_prompts_ks.jsonl: {n} prompts")


if __name__ == "__main__":
    main()
