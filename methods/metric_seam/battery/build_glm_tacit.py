"""GLM-family tacitity probe (2026-07-08) — semantic-gravity across GLM endpoints.

Self-contained E2-STIP: for each e2-bearing field, build TWO prompts per (subsampled)
item — the FULL community instruction (name+definition, = the certified field prompt)
and the DEVIANT stipulation. Run BOTH through each GLM version so snap-back is measured
against each version's OWN community baseline (not Gemma's). Thick task (humor) vs thin
task (math) contrast. Sparing: 50 items/task (stable-hash subsample).

-> outputs/metric_seam_pilot/battery/glm_tacit/prompts.jsonl
   + checkers.json (aid.field -> {checker_expr, task, construct})
"""
import hashlib, importlib.util, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"
OUT = BASE / "battery/glm_tacit"
OUT.mkdir(exist_ok=True)
TASKS = {"humor": "programs_humor", "math": "programs_math"}
N_ITEMS = 50

T = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""


def llm_fields(progdir, aid):
    p = ROOT / "methods/metric_seam/hybrids" / progdir / f"{aid}_h0.py"
    if not p.exists():
        return {}
    s = importlib.util.spec_from_file_location(p.stem + progdir, p)
    m = importlib.util.module_from_spec(s)
    try:
        s.loader.exec_module(m)
        return getattr(m, "LLM_FIELDS", {}) or {}
    except Exception:
        return {}


def subsample(items, n):
    keyed = sorted(items, key=lambda it: hashlib.md5(
        it["datapoint_id"].encode()).hexdigest())
    return keyed[:n]


def main():
    checkers = {}
    n = 0
    with open(OUT / "prompts.jsonl", "w") as f:
        for task, progdir in TASKS.items():
            v = json.load(open(BASE / f"battery/variants_{task}.json"))
            items = subsample(json.load(open(BASE / f"tasks/{task}/items.json")), N_ITEMS)
            for k, x in v.items():
                if not (isinstance(x, dict) and x.get("e2")):
                    continue
                aid, field = x["aid"], x["field"]
                lf = llm_fields(progdir, aid)
                if field not in lf or not x["e2"].get("checker_expr") \
                        or not x["e2"].get("deviant_instruction"):
                    continue
                conds = {"comm": lf[field], "stip": x["e2"]["deviant_instruction"]}
                ck = f"{task}::{aid}__{field}"
                checkers[ck] = {"task": task, "aid": aid, "field": field,
                                "checker_expr": x["e2"]["checker_expr"],
                                "construct": x.get("construct_name", ""),
                                "rule": x["e2"].get("rule_gloss", "")}
                for cond, ins in conds.items():
                    for it in items:
                        f.write(json.dumps({
                            "channel": "field",
                            "aspect_id": f"{task}::{aid}.{cond}__{field}",
                            "datapoint_id": it["datapoint_id"],
                            "prompt": T.format(instruction=ins, text=it["ctext"])})
                            + "\n")
                        n += 1
    json.dump(checkers, open(OUT / "checkers.json", "w"), indent=1)
    per_task = {}
    for ck in checkers:
        t = checkers[ck]["task"]
        per_task[t] = per_task.get(t, 0) + 1
    print(f"{n} prompts, {len(checkers)} e2 fields {per_task}, {N_ITEMS} items/task "
          f"x 2 conds -> {OUT}")


if __name__ == "__main__":
    main()
