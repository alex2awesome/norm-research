"""Field prompts for h1 modules — ONLY fields that are new or whose instruction changed vs h0.

h1 extractions are namespaced as aspect_id "<aid>.h1__<field>" so they never collide with h0's
stored values (several h1s reuse a field name with a revised instruction). Unchanged fields
(same name AND same instruction) are reused from the h0 extraction at eval time.

Usage: python3 build_field_prompts_h1.py <task> <progdir> <aid1,aid2,...>
-> outputs/metric_seam_pilot/tasks/<task>/field_prompts_h1.jsonl
"""
import importlib.util, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]

T = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""


def load_fields(p):
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(getattr(mod, "LLM_FIELDS", {}) or {})


def main():
    task, progdir, aids = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    OUT = ROOT / "outputs/metric_seam_pilot/tasks" / task
    HYB = pathlib.Path(__file__).parent / progdir
    items = json.load(open(OUT / "items.json"))
    n = 0
    with open(OUT / "field_prompts_h1.jsonl", "w") as f:
        for aid in aids:
            f0 = load_fields(HYB / f"{aid}_h0.py")
            f1 = load_fields(HYB / f"{aid}_h1.py")
            need = {k: v for k, v in list(f1.items())[:2] if f0.get(k) != v}
            print(f"{aid}: {len(need)}/{len(f1)} fields need extraction {sorted(need)}")
            for field, instruction in need.items():
                for it in items:
                    f.write(json.dumps({"channel": "field",
                                        "aspect_id": f"{aid}.h1__{field}",
                                        "datapoint_id": it["datapoint_id"],
                                        "prompt": T.format(instruction=instruction,
                                                           text=it["ctext"])}) + "\n")
                    n += 1
    print(f"wrote {n} prompts -> {OUT / 'field_prompts_h1.jsonl'}")


if __name__ == "__main__":
    main()
