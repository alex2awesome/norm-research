"""Collect LLM_FIELDS from wave-2 hybrids (programs_v2/) -> field_prompts_v2.jsonl."""
import importlib.util, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
V1 = ROOT / "outputs/metric_seam_pilot/v1"
OUT = ROOT / "outputs/metric_seam_pilot/v2"
HYB = pathlib.Path(__file__).parent / "programs_v2"

T = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""


def main():
    items = json.load(open(V1 / "items_v1.json"))
    n = 0
    with open(OUT / "field_prompts_v2.jsonl", "w") as f:
        for prog in sorted(HYB.glob("*_h0.py")):
            aid = prog.stem.split("_")[0]
            spec = importlib.util.spec_from_file_location(prog.stem, prog)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                print(f"{prog.name}: load error {e}", file=sys.stderr)
                continue
            for field, instruction in list((getattr(mod, "LLM_FIELDS", {}) or {}).items())[:2]:
                for it in items:
                    f.write(json.dumps({"channel": "field",
                                        "aspect_id": f"{aid}__{field}",
                                        "datapoint_id": it["datapoint_id"],
                                        "prompt": T.format(instruction=instruction,
                                                           text=it["ctext"])}) + "\n")
                    n += 1
    print(f"wrote {n} field prompts")


if __name__ == "__main__":
    main()
