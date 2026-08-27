"""UNIT->CODE builder (2026-07-08): census lexicon units as seam criteria.

Selects 30 humor R1 constructs (n_sources>=3; all MECHANICAL + top-13 CRAFT +
top-13 TASTE by n_sources), formats each as a criterion (unit name + gloss), and
emits 2-form judge prompts over the EXISTING humor pilot items (same 250 docs,
same splits as the fleet -> outcomes directly comparable).

-> outputs/metric_seam_pilot/tasks/humor_units/{units_selected.json, prompts.jsonl}
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot/tasks/humor_units"
OUT.mkdir(parents=True, exist_ok=True)

ROLE = ("an experienced comedy writer and humor editor",
        "attempted-humor text (a joke, humorous story, or comedic piece)")

T1 = """You are {role} grading ONE {doctype} against ONE criterion.

Criterion: {name}
Definition: {description}

Document (may be truncated; '[...]' marks an elided middle):
<document>
{text}
</document>

Grade how well the document satisfies this criterion on a 0-10 integer scale \
(0 = completely fails, 10 = exemplary). If the criterion genuinely does not apply to this \
kind of document, answer NA. Reply with exactly one line in the form:
SCORE: <integer 0-10 or NA>"""

T2 = """As {role}, evaluate the document below on a single quality criterion, ignoring all \
other qualities.

The criterion — {name}: {description}

Document ('[...]' = elided middle):
<document>
{text}
</document>

Give an integer 0-10 (0 = criterion badly violated or absent, 10 = criterion fully \
exemplified). If this criterion simply cannot apply to a document of this kind, answer NA. \
Your entire reply must be one line:
SCORE: <integer 0-10 or NA>"""


def main():
    nn = json.load(open(ROOT / "outputs/lexicon/node_names_humor_R1.json"))
    cod = json.load(open(ROOT / "outputs/lexicon/codability/codability_humor.json"))
    multi = [r for r in cod if (r.get("n_sources") or 0) >= 3
             and r.get("type") in ("MECHANICAL", "CRAFT", "TASTE")
             and r["construct"] in nn]
    sel = [r for r in multi if r["type"] == "MECHANICAL"]
    for ty, k in [("CRAFT", 13), ("TASTE", 13)]:
        rows = sorted([r for r in multi if r["type"] == ty],
                      key=lambda r: -r["n_sources"])
        sel += rows[:k]
    units = []
    for i, r in enumerate(sel):
        node = nn[r["construct"]]
        units.append({"aspect_id": f"u{i}", "construct": r["construct"],
                      "type": r["type"], "n_sources": r["n_sources"],
                      "name": node["name"], "description": node["gloss"]})
    json.dump(units, open(OUT / "units_selected.json", "w"), indent=1)

    items = json.load(open(ROOT / "outputs/metric_seam_pilot/tasks/humor/items.json"))
    role, doctype = ROLE
    n = 0
    with open(OUT / "prompts.jsonl", "w") as f:
        for u in units:
            for it in items:
                for ch, T in [("pass1", T1), ("pass2", T2)]:
                    f.write(json.dumps({
                        "channel": ch, "aspect_id": u["aspect_id"],
                        "datapoint_id": it["datapoint_id"],
                        "prompt": T.format(role=role, doctype=doctype, name=u["name"],
                                           description=u["description"],
                                           text=it["ctext"])}) + "\n")
                    n += 1
    by = {}
    for u in units:
        by[u["type"]] = by.get(u["type"], 0) + 1
    print(f"{len(units)} units {by}, {len(items)} items, {n} prompts -> {OUT}")


if __name__ == "__main__":
    main()
