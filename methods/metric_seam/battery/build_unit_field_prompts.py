"""Collect LLM fields declared by unit-h0 programs (programs_units/u*_h0.py) and emit
extractor prompts over the humor pilot items.

-> outputs/metric_seam_pilot/tasks/humor_units/field_prompts.jsonl
"""
import importlib.util, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
PU = ROOT / "methods/metric_seam/hybrids/programs_units"
OUT = ROOT / "outputs/metric_seam_pilot/tasks/humor_units"

T = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""


def load_mod(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    items = json.load(open(OUT / "items.json"))
    n, nasp = 0, 0
    with open(OUT / "field_prompts.jsonl", "w") as f:
        for prog in sorted(PU.glob("u*_h0.py"), key=lambda p: int(p.stem[1:].split("_")[0])):
            aid = prog.stem.split("_")[0]
            mod = load_mod(prog)
            fields = list((getattr(mod, "LLM_FIELDS", {}) or {}).items())[:2]
            if fields:
                nasp += 1
            for field, instruction in fields:
                for it in items:
                    f.write(json.dumps({
                        "channel": "field", "aspect_id": f"{aid}__{field}",
                        "datapoint_id": it["datapoint_id"],
                        "prompt": T.format(instruction=instruction, text=it["ctext"])})
                        + "\n")
                    n += 1
    print(f"{nasp} unit programs declare fields; {n} field prompts -> {OUT/'field_prompts.jsonl'}")


if __name__ == "__main__":
    main()
