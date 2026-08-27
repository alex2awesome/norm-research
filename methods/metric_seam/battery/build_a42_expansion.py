"""WS6.1: a42 (math) fresh-item gate expansion (runbook 2026-07-10; pilot-results §"a42
disposition": h1 crossed certification P_gate=.52 but missed promotion by .011 at n=100 —
NEVER rule-bend; resolve on fresh items per the a110/a80 gate-expansion protocol).

PRE-REGISTERED RESOLUTION RULE (frozen before any new data lands): 400 FRESH items (stable
md5-hash order over the 4,750 unused pool dpids), judged 2-pass on a42 only; h1 promotes iff
BOTH, on the combined 500-item held-out set (100 original test + 400 fresh), paired
bootstrap B=2000:
  (a) P(h1 > h0) >= .80
  (b) G1-floor: P(rho_h1 >= .60) >= .95   [the margin arm of G1 needs code-flavor scores,
      which don't exist for fresh items and won't be recomputed — disclosed limitation]
Anything else = h0 stays head. Judges use intersection-only 2-pass items (eval-v2 rule).

Builds ONE Gemma jsonl (judge pass1/pass2 rows + field rows for the union of
a42_h0/a42_h1 LLM_FIELDS) + the fresh-items file.
Usage: python3 build_a42_expansion.py
Then (sk3): gemma_score_v1.py --prompts a42_expansion_prompts.jsonl
            --out a42_expansion_results.jsonl (gpu_waiter pattern, 1 GPU, never contend)
Then: eval_a42_expansion.py
"""
import hashlib
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot/tasks/math"
HYB = ROOT / "methods/metric_seam/hybrids/programs_math"
POOL = ROOT / "runs/validity_full/v2/math/datapoints.json"
N_FRESH = 400

FIELD_T = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""


def load_prog(name):
    spec = importlib.util.spec_from_file_location(name, HYB / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recover_templates(items):
    """Split existing a42 pass1/pass2 prompts on the item's ctext -> (prefix, suffix)."""
    by_dpid = {it["datapoint_id"]: it["ctext"] for it in items}
    tpl = {}
    for line in open(OUT / "prompts.jsonl"):
        r = json.loads(line)
        if r["aspect_id"] != "a42" or r["channel"] not in ("pass1", "pass2"):
            continue
        if r["channel"] in tpl:
            continue
        ct = by_dpid[r["datapoint_id"]]
        assert ct in r["prompt"], f"ctext not verbatim in {r['channel']} prompt"
        pre, suf = r["prompt"].split(ct, 1)
        tpl[r["channel"]] = (pre, suf)
        if len(tpl) == 2:
            return tpl
    raise SystemExit("could not recover both pass templates")


def make_ctext(existing, row):
    """Replicate the ctext derivation (observed: ctext is a prefix-cap of text)."""
    cap = max(len(it["ctext"]) for it in existing)
    assert all(it["text"].startswith(it["ctext"][:50]) for it in existing[:20])
    return row["text"][:cap]


def main():
    existing = json.load(open(OUT / "items.json"))
    used = {it["datapoint_id"] for it in existing}
    pool = [r for r in json.load(open(POOL)) if r["datapoint_id"] not in used]
    pool.sort(key=lambda r: hashlib.md5(r["datapoint_id"].encode()).hexdigest())
    fresh = [{"datapoint_id": r["datapoint_id"], "judgement": r["judgement"],
              "text": r["text"], "ctext": make_ctext(existing, r)} for r in pool[:N_FRESH]]
    json.dump(fresh, open(OUT / "a42_expansion_items.json", "w"))

    tpl = recover_templates(existing)
    fields = {}
    for prog in ("a42_h0", "a42_h1"):
        for fname, instr in list((getattr(load_prog(prog), "LLM_FIELDS", {}) or {}).items())[:2]:
            fields.setdefault(fname, instr)

    n = 0
    with open(OUT / "a42_expansion_prompts.jsonl", "w") as f:
        for it in fresh:
            for ch, (pre, suf) in tpl.items():
                f.write(json.dumps({"channel": ch, "aspect_id": "a42",
                                    "datapoint_id": it["datapoint_id"],
                                    "prompt": pre + it["ctext"] + suf}) + "\n")
                n += 1
            for fname, instr in fields.items():
                f.write(json.dumps({"channel": "field", "aspect_id": f"a42__{fname}",
                                    "datapoint_id": it["datapoint_id"],
                                    "prompt": FIELD_T.format(instruction=instr,
                                                             text=it["ctext"])}) + "\n")
                n += 1
    print(f"{len(fresh)} fresh items, {len(fields)} fields ({sorted(fields)}), "
          f"{n} prompts -> {OUT / 'a42_expansion_prompts.jsonl'}")


if __name__ == "__main__":
    main()
