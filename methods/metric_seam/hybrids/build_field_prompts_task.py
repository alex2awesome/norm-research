"""Collect LLM_FIELDS from a task-generic hybrid fleet (programs_<progdir>/) -> field
prompts for the Gemma extractor. Usage: python3 build_field_prompts_task.py <task> <progdir>
  <task>    e.g. creative_writing, math   (reads outputs/metric_seam_pilot/tasks/<task>/items.json)
  <progdir> e.g. programs_cw, programs_math  (under methods/metric_seam/hybrids/)
Output: outputs/metric_seam_pilot/tasks/<task>/field_prompts.jsonl
"""
import importlib.util, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]

T = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""


def main():
    task, progdir = sys.argv[1], sys.argv[2]
    OUT = ROOT / "outputs/metric_seam_pilot/tasks" / task
    HYB = pathlib.Path(__file__).parent / progdir
    items = json.load(open(OUT / "items.json"))
    n, aspects_with_fields = 0, 0
    with open(OUT / "field_prompts.jsonl", "w") as f:
        for prog in sorted(HYB.glob("*_h0.py")):
            aid = prog.stem.split("_")[0]
            spec = importlib.util.spec_from_file_location(prog.stem, prog)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                print(f"{prog.name}: load error {e}", file=sys.stderr)
                continue
            fields = list((getattr(mod, "LLM_FIELDS", {}) or {}).items())[:2]
            if fields:
                aspects_with_fields += 1
            for field, instruction in fields:
                for it in items:
                    f.write(json.dumps({"channel": "field",
                                        "aspect_id": f"{aid}__{field}",
                                        "datapoint_id": it["datapoint_id"],
                                        "prompt": T.format(instruction=instruction,
                                                           text=it["ctext"])}) + "\n")
                    n += 1
    print(f"{aspects_with_fields} aspects declare fields; wrote {n} field prompts -> "
          f"{OUT/'field_prompts.jsonl'}")


if __name__ == "__main__":
    main()
