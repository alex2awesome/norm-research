"""GLM E2-KIND grid probe (2026-07-08) — deficit-corrected semantic gravity.

Reuses the existing e2kind cell4/5/6 prompts (byte-identical to what Gemma/Llama/Qwen got)
filtered to a 50-item stable-hash subsample, PLUS the community condition (comm = full
field instruction) so each GLM version has its OWN community baseline for the cell4
conflict/phantom readout. gravity_effect = cell6.acc - cell4.comply subtracts the
instruction-following-deficit -> separates genuine gravity from rule-following capacity.

-> outputs/metric_seam_pilot/battery/glm_e2kind/prompts.jsonl
"""
import hashlib, importlib.util, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"
OUT = BASE / "battery/glm_e2kind"
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


def sub_ids(items, n):
    keyed = sorted(items, key=lambda it: hashlib.md5(
        it["datapoint_id"].encode()).hexdigest())
    return {it["datapoint_id"] for it in keyed[:n]}


def main():
    n = 0
    with open(OUT / "prompts.jsonl", "w") as f:
        for task, progdir in TASKS.items():
            items_l = json.load(open(BASE / f"tasks/{task}/items.json"))
            keep = sub_ids(items_l, N_ITEMS)
            itext = {it["datapoint_id"]: it["ctext"] for it in items_l}
            # 1) existing cell4/5/6 prompts, filtered + task-namespaced
            for line in open(BASE / f"tasks/{task}/e2kind_prompts.jsonl"):
                r = json.loads(line)
                if r["datapoint_id"] not in keep:
                    continue
                r["aspect_id"] = f"{task}::{r['aspect_id']}"
                f.write(json.dumps(r) + "\n")
                n += 1
            # 2) comm condition (own community baseline) for each e2 field
            fields = set()
            for line in open(BASE / f"tasks/{task}/e2kind_prompts.jsonl"):
                r = json.loads(line)
                aid = r["aspect_id"].split(".")[0]
                field = r["aspect_id"].split("__", 1)[1]
                fields.add((aid, field))
            for aid, field in sorted(fields):
                ins = llm_fields(progdir, aid).get(field)
                if not ins:
                    continue
                for d in keep:
                    f.write(json.dumps({
                        "channel": "field",
                        "aspect_id": f"{task}::{aid}.comm__{field}",
                        "datapoint_id": d,
                        "prompt": T.format(instruction=ins, text=itext[d])}) + "\n")
                    n += 1
    print(f"{n} prompts -> {OUT}  ({N_ITEMS} items/task, cells 4/5/6 + comm)")


if __name__ == "__main__":
    main()
