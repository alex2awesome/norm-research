"""Retrieval-battery inventory (E1 KEY / E2 STIP selection), per pre-reg note
notes/2026-07-05__seam-position-retrieval-and-codability-priors.md §2.3.

Collects, for every hybrid criterion across the 4 fleet corpora:
  task, aid, criterion name/description, fm + transport ratios (3fam), the program's
  LLM_FIELDS instructions, and the Gemma answer-shape stats per field (distinct raws,
  top values) needed to pick binary-ish fields for E2.

Selection rules (pre-registered here, BEFORE authoring):
  E1: criteria with fm >= 0.10, capped at top-15 per task by fm.
  E2: among E1-eligible criteria, fields whose Gemma answers are near-binary
      (<=4 distinct values covering >=90% of non-empty answers), top-8 per task by fm.

-> outputs/metric_seam_pilot/battery/inventory.json
"""
import importlib.util, json, pathlib, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"
HYB = ROOT / "methods/metric_seam/hybrids"

TASKS = {
    "press_releases": {"tfam": BASE / "v2/transport_eval_3fam.json",
                       "fields": BASE / "v2/field_results_v2.jsonl",
                       "aspects": ROOT / "runs/validity_full/v2/press_releases/aspects.json",
                       "progdir": "programs_v2"},
    "creative_writing": {"tfam": BASE / "tasks/creative_writing/transport_eval_3fam.json",
                         "fields": BASE / "tasks/creative_writing/field_results.jsonl",
                         "aspects": ROOT / "runs/validity_full/v2/creative_writing/aspects.json",
                         "progdir": "programs_cw"},
    "math": {"tfam": BASE / "tasks/math/transport_eval_3fam.json",
             "fields": BASE / "tasks/math/field_results.jsonl",
             "aspects": ROOT / "runs/validity_full/v2/math/aspects.json",
             "progdir": "programs_math"},
    "humor": {"tfam": BASE / "tasks/humor/transport_eval_3fam.json",
              "fields": BASE / "tasks/humor/field_results.jsonl",
              "aspects": ROOT / "runs/validity_full/v2/humor/aspects.json",
              "progdir": "programs_humor"},
}


def load_fields_shape(path):
    """(aid__field) -> answer-shape stats from the certified Gemma extraction."""
    vals = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("channel") != "field":
            continue
        vals.setdefault(r["aspect_id"], []).append((r.get("raw") or "").strip().lower())
    out = {}
    for k, v in vals.items():
        nonempty = [x for x in v if x]
        c = Counter(nonempty)
        top4 = c.most_common(4)
        cover = sum(n for _, n in top4) / max(1, len(nonempty))
        out[k] = {"n": len(v), "n_distinct": len(c),
                  "top_values": [f"{val} x{n}" for val, n in top4],
                  "top4_coverage": round(cover, 3),
                  "near_binary": len(c) <= 4 or cover >= 0.90}
    return out


def main():
    OUT = BASE / "battery"
    OUT.mkdir(exist_ok=True)
    inv = {}
    for task, cfg in TASKS.items():
        aspects = {a["aspect_id"]: a for a in json.load(open(cfg["aspects"]))}
        tfam = json.load(open(cfg["tfam"]))["aspects"]
        shapes = load_fields_shape(cfg["fields"])
        rows = {}
        for prog in sorted((HYB / cfg["progdir"]).glob("*_h0.py")):
            aid = prog.stem.split("_")[0]
            v = tfam.get(aid)
            if not isinstance(v, dict) or "error" in v:
                continue
            spec = importlib.util.spec_from_file_location(prog.stem, prog)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                print(f"{task}/{aid}: load error {e}", file=sys.stderr)
                continue
            fields = dict(list((getattr(mod, "LLM_FIELDS", {}) or {}).items())[:2])
            if not fields:
                continue
            a = aspects.get(aid, {})
            rows[aid] = {
                "criterion_name": a.get("name", ""),
                "criterion_description": a.get("description", ""),
                "fm": v.get("field_marginal"),
                "ratio_llama": v.get("ratio_llama"), "ratio_qwen": v.get("ratio_qwen"),
                "rho_gemma": v["rho"]["gemma"], "rho_blank": v["rho"]["blank"],
                "fields": {fn: {"instruction": ins,
                                "shape": shapes.get(f"{aid}__{fn}", {})}
                           for fn, ins in fields.items()},
            }
        # E1 selection: fm >= .10, top-15 by fm
        elig = sorted([a for a, r in rows.items()
                       if r["fm"] is not None and r["fm"] >= 0.10],
                      key=lambda a: -rows[a]["fm"])[:15]
        for a in elig:
            rows[a]["E1_selected"] = True
        # E2 selection: within E1-eligible, near-binary fields, top-8 criteria by fm
        e2 = []
        for a in elig:
            bin_fields = [fn for fn, f in rows[a]["fields"].items()
                          if f["shape"].get("near_binary")]
            if bin_fields:
                e2.append((a, bin_fields))
            if len(e2) == 8:
                break
        for a, fns in e2:
            rows[a]["E2_selected_fields"] = fns
        inv[task] = rows
        n_f = sum(len(rows[a]["fields"]) for a in elig)
        print(f"{task}: {len(rows)} criteria with fields | E1 {len(elig)} criteria "
              f"({n_f} fields) | E2 {len(e2)} criteria "
              f"({sum(len(f) for _, f in e2)} fields)")
    json.dump(inv, open(OUT / "inventory.json", "w"), indent=1)
    print(f"-> {OUT / 'inventory.json'}")


if __name__ == "__main__":
    main()
