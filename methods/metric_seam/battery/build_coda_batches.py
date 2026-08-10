"""CODA probe (priors note §3.1) — build blinded annotation batches.

143 fleet criteria (name+description only, task identity hidden, opaque ids,
shuffled across tasks) + 6 synthetic anchors interleaved per batch. Anchor keys
are written to a SEPARATE file never shown to annotators.

-> outputs/metric_seam_pilot/battery/coda/batch_<k>.json      (annotator input)
-> outputs/metric_seam_pilot/battery/coda/coda_manifest.json  (id -> task/aid + anchor keys)
"""
import json, pathlib, random

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"
OUT = BASE / "battery/coda"
OUT.mkdir(exist_ok=True)

TASKS = ["press_releases", "creative_writing", "math", "humor", "legal_title_vii"]

# anchors: (name, description, keyed feature checks {feature: allowed values}, hard?)
ANCHORS = [
    ("Length requirement", "The document's word count exceeds 500 words.",
     {"F1": (2,), "F5": (1, 2), "F3": (0,)}, True),
    ("Vitality", "The piece feels alive.",
     {"F3": (1, 2), "F4": (1, 2), "F1": (0,)}, True),
    ("Precedent citation", "Cites at least two precedent cases by name.",
     {"F1": (1, 2), "F2": (2,), "F5": (1, 2)}, True),
    ("Professional register", "Maintains a professional register throughout the document.",
     {"F3": (1, 2), "F6": (1, 2), "F1": (0,)}, True),
    ("Supported claims", "Uses concrete examples to support at least some of its claims.",
     {}, False),
    ("Follow the argument", "Explanations are organized so a reader can follow the main "
     "argument without rereading.", {}, False),
]

def main():
    rows = []
    for t in TASKS:
        cam = json.load(open(BASE / "cam_profile.json"))[t]["per_criterion"]
        aspects = {a["aspect_id"]: a for a in
                   json.load(open(ROOT / f"runs/validity_full/v2/{t}/aspects.json"))}
        for r in cam:
            a = aspects.get(r["aspect"])
            if not a:
                continue
            rows.append({"task": t, "aid": r["aspect"], "name": a.get("name", ""),
                         "description": a.get("description", "")})
    rng = random.Random(20260708)
    rng.shuffle(rows)

    manifest, batches = {}, []
    per = 20
    nb = (len(rows) + per - 1) // per
    ctr = 0
    for b in range(nb):
        chunk = rows[b * per:(b + 1) * per]
        items = []
        for r in chunk:
            cid = f"c{ctr:03d}"; ctr += 1
            manifest[cid] = {"kind": "real", "task": r["task"], "aid": r["aid"]}
            items.append({"id": cid, "name": r["name"], "description": r["description"]})
        for name, desc, checks, hard in ANCHORS:
            cid = f"c{ctr:03d}"; ctr += 1
            manifest[cid] = {"kind": "anchor", "anchor_name": name,
                             "checks": {k: list(v) for k, v in checks.items()}, "hard": hard}
            items.append({"id": cid, "name": name, "description": desc})
        rng.shuffle(items)
        batches.append(items)
        json.dump(items, open(OUT / f"batch_{b}.json", "w"), indent=1)
    json.dump(manifest, open(OUT / "coda_manifest.json", "w"), indent=1)
    print(f"{len(rows)} real criteria, {nb} batches (+6 anchors each) -> {OUT}")

if __name__ == "__main__":
    main()
