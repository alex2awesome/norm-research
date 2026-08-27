"""Collect the NEW (budget-4) fields from programs_b4/<task>__<aid>_h4.py and emit
extractor prompts ONLY for field names not already extracted in the fleet f_orig.

-> outputs/metric_seam_pilot/battery/b4_field_prompts.jsonl
   (aspect_id encoded as "<task>::<aid>__<field>" so the merge step can route it)
"""
import importlib.util, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
B4 = ROOT / "methods/metric_seam/hybrids/programs_b4"
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx  # noqa: E402

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
    ctxs = {}
    n = 0
    with open(ROOT / "outputs/metric_seam_pilot/battery/b4_field_prompts.jsonl", "w") as f:
        for prog in sorted(B4.glob("*_h4.py")):
            task, rest = prog.stem.split("__", 1)
            aid = rest.split("_h4")[0]
            if task not in ctxs:
                ctxs[task] = load_ctx(task)
            ctx = ctxs[task]
            existing = set()
            for row in ctx["f_orig"].get(aid, {}).values():
                existing.update(row.keys())
            mod = load_mod(prog)
            fields = getattr(mod, "LLM_FIELDS", {}) or {}
            new_fields = {k: v for k, v in fields.items() if k not in existing}
            for field, instruction in new_fields.items():
                for dpid, text in ctx["items"].items():
                    f.write(json.dumps({
                        "channel": "field",
                        "aspect_id": f"{task}::{aid}__{field}",
                        "datapoint_id": dpid,
                        "prompt": T.format(instruction=instruction, text=text)}) + "\n")
                    n += 1
            print(f"{task}.{aid}: {len(new_fields)} new fields {list(new_fields)}")
    print(f"-> {n} b4 field prompts")


if __name__ == "__main__":
    main()
