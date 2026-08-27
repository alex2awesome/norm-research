"""E2-KIND extension prompt builder: re-authored fields only (those carrying a
community_checker_expr — the 5 math fields whose original e2 checkers were
corpus-degenerate). Same template + leak guard as build_e2kind_tasks.py; output is
a separate file so completed runs are never re-burned.

Usage: python3 build_e2kind_ext.py <task>
-> outputs/metric_seam_pilot/tasks/<task>/e2kind_prompts_ext.jsonl
"""
import json, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from build_e2kind_tasks import T, DOC_RE, content_words  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"


def main():
    task = sys.argv[1]
    kv = json.load(open(BASE / f"battery/variants_e2kind_{task}.json"))
    ext = {k: v for k, v in kv.items()
           if not k.startswith("_") and v.get("community_checker_expr")}
    if not ext:
        sys.exit(f"no community_checker_expr entries in variants_e2kind_{task}.json")
    outdir = BASE / "tasks" / task

    docs = {}
    for line in open(outdir / "battery_prompts.jsonl"):
        r = json.loads(line)
        aidc, field = r["aspect_id"].split("__", 1)
        if not aidc.endswith(".stip"):
            continue
        aid = aidc.split(".")[0]
        m = DOC_RE.search(r["prompt"])
        if m:
            docs[(aid, field, r["datapoint_id"])] = m.group(1)

    n = {"cell4": 0, "cell5": 0, "cell6": 0}
    leaks = []
    with open(outdir / "e2kind_prompts_ext.jsonl", "w") as f:
        for key, v in ext.items():
            aid, field = key.split("__", 1)
            cells = {"cell4": v["cell4_nonce_deviant"],
                     "cell5": v["cell5_name_neutral"],
                     "cell6": v["cell6_nonce_neutral"]}
            for cell, ins in cells.items():
                if cell in ("cell4", "cell6"):
                    for w in content_words(field):
                        if re.search(r"\b" + re.escape(w[:6]), ins, re.I):
                            leaks.append((key, cell, w))
                for (a, fl, d), text in docs.items():
                    if a != aid or fl != field:
                        continue
                    f.write(json.dumps({
                        "channel": "field",
                        "aspect_id": f"{aid}.{cell}__{field}",
                        "datapoint_id": d,
                        "prompt": T.format(instruction=ins, text=text)}) + "\n")
                    n[cell] += 1
    if leaks:
        print("LEAK WARNINGS (community stem in nonce-cell instruction):")
        for l in leaks:
            print("  ", l)
        sys.exit(1)
    print(f"{task} ext ({len(ext)} fields): {n} (total {sum(n.values())}) "
          f"-> tasks/{task}/e2kind_prompts_ext.jsonl")


if __name__ == "__main__":
    main()
