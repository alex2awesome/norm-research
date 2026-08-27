"""W1.3 gate-resolution expansion: 400 NEW held-out PR items for the four v1 hybrid aspects.

a110/a80 gate verdicts were UNRESOLVED at n_test=100 (P(gate)=.59/.31, B=2000); this adds
400 fresh items (disjoint from the v1 250, same sampling frame: len>=1000, same canonical
head5000+tail2500) so the frozen h0 hybrids + frozen baselines can be re-bootstrapped on
n_test up to 500. Pure evaluation expansion — nothing is retrained; train split unchanged.

Emits: outputs/metric_seam_pilot/v1/expansion/{items_exp.json, prompts_exp.jsonl (judge
2-pass + scope), field_prompts_exp.jsonl (LLM fields declared by the frozen h0 programs)}.
"""
import importlib.util, json, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from build_task import T1, T2, TSCOPE, ROLE, canonical  # noqa: E402

V1 = ROOT / "outputs/metric_seam_pilot/v1"
EXP = V1 / "expansion"
EXP.mkdir(exist_ok=True)
ASPECTS = ["a80", "a86", "a105", "a110"]
HYB = ROOT / "methods/metric_seam/hybrids/programs"
N_NEW, SEED = 400, 101

FIELD_T = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""


def main():
    v1_ids = {x["datapoint_id"] for x in json.load(open(V1 / "items_v1.json"))}
    data = json.load(open(ROOT / "runs/validity_full/v2/press_releases/datapoints.json"))
    pool = [d for d in data
            if len(d.get("text", "")) >= 1000 and d["datapoint_id"] not in v1_ids]
    random.Random(SEED).shuffle(pool)
    items = pool[:N_NEW]
    for it in items:
        it["ctext"] = canonical(it["text"])
    json.dump(items, open(EXP / "items_exp.json", "w"))
    print(f"{len(items)} new items (pool {len(pool)}, excluded {len(v1_ids)} v1 ids)")

    role, doctype = ROLE["press_releases"]
    aspects = {x["aspect_id"]: x for x in json.load(
        open(ROOT / "runs/validity_full/v2/press_releases/aspects.json"))}
    n = 0
    with open(EXP / "prompts_exp.jsonl", "w") as f:
        for aid in ASPECTS:
            a = aspects[aid]
            for it in items:
                for ch, T in [("pass1", T1), ("pass2", T2)]:
                    f.write(json.dumps({
                        "channel": ch, "aspect_id": aid,
                        "datapoint_id": it["datapoint_id"],
                        "prompt": T.format(role=role, doctype=doctype, name=a["name"],
                                           description=a["description"],
                                           text=it["ctext"])}) + "\n")
                    n += 1
        for it in items:
            f.write(json.dumps({
                "channel": "scope", "aspect_id": "scope",
                "datapoint_id": it["datapoint_id"],
                "prompt": TSCOPE.format(doctype=doctype, text=it["ctext"])}) + "\n")
            n += 1
    print(f"prompts_exp.jsonl: {n}")

    n = 0
    with open(EXP / "field_prompts_exp.jsonl", "w") as f:
        for aid in ASPECTS:
            prog = HYB / f"{aid}_h0.py"
            spec = importlib.util.spec_from_file_location(prog.stem, prog)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for field, instruction in list(getattr(mod, "LLM_FIELDS", {}).items())[:2]:
                for it in items:
                    f.write(json.dumps({
                        "channel": "field", "aspect_id": f"{aid}__{field}",
                        "datapoint_id": it["datapoint_id"],
                        "prompt": FIELD_T.format(instruction=instruction,
                                                 text=it["ctext"])}) + "\n")
                    n += 1
    print(f"field_prompts_exp.jsonl: {n}")


if __name__ == "__main__":
    main()
