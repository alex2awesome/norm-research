"""Generalized seed-G F extension: for ANY task with a hybrid_gate_report (base+ceiling),
per-criterion judge scores (load_ctx), and improver packs (criterion name/description),
build the arm-G held-out prompts (SEED prompt only -- no GEPA) and compute the seam width
F = A/(V+A) against the judge.

Seed-G is a CONSERVATIVE LOWER BOUND on the domain's seam: GEPA can only raise G, so true
F >= seed-G F. Cheap: one Gemma-4-31b pass/task, zero GLM.

  python3 seed_g_extend.py build <task>              -> seed_g_<task>_prompts.jsonl (TEST rows)
  python3 seed_g_extend.py eval  <task> <results>    -> seed_g_<task>_final.json + printed F
"""
import json, math, sys, pathlib, glob

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam/battery"))
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
import battery_common as bc
from certificates import spearman

BASE = ROOT / "outputs/metric_seam_pilot"
OUT = HERE / "seed_g_extend"
OUT.mkdir(exist_ok=True)

# load_ctx needs a PROGDIR entry (hyb dir; unused for seed-G but dereferenced) -- patch all.
for t, pd in {"legal_title_vii": "programs_legal", "legal_ss_disability": "programs_legal_ss",
              "peer_review": "programs_peer", "creative_writing": "programs_cw",
              "math": "programs_math", "humor": "programs_humor"}.items():
    bc.PROGDIR.setdefault(t, pd)

TASK_DOMAIN = {
    "legal_title_vii": "U.S. Title VII employment-discrimination court opinion",
    "legal_ss_disability": "U.S. Social Security disability benefits determination / opinion",
    "peer_review": "academic peer-review report of a submitted paper",
    "creative_writing": "creative-writing story or story excerpt",
    "math": "Math StackExchange question/answer post",
    "humor": "stand-up comedy / humor-craft feedback post",
}
DOC_MARKER = "<<<DOCUMENT>>>"
MAXCHARS = 6000
FOOTER = "\n\nReply with exactly one line: SCORE: <integer 0-10>"
SEED_TEMPLATE = """You are scoring a single {domain} on one specific quality criterion.

Criterion: {name} -- {desc}

Document:
{marker}

Score how well this document satisfies the criterion above, from 0 (fails completely) to \
10 (fully satisfies it). Judge the criterion only; ignore unrelated aspects of quality."""


def ceiling(rel1, k=2):
    r = max(0.0, min(1.0, rel1)); relk = k * r / (1 + (k - 1) * r)
    return math.sqrt(relk) if relk > 0 else float("nan")


def clip01(x):
    return max(0.0, min(1.0, x))


def gate_path(task):
    d = "v2" if task == "press_releases" else f"tasks/{task}"
    return BASE / d / "hybrid_gate_report.json"


def pack_path(task, aid):
    d = "v2" if task == "press_releases" else f"tasks/{task}"
    return BASE / d / "improver_packs" / f"{aid}.json"


def certified_aids(task):
    g = json.load(open(gate_path(task)))
    out = []
    for aid, v in g.items():
        b = v.get("full", {}).get("rho_baseline"); j = v.get("judge_rel1")
        if b is not None and j is not None and pack_path(task, aid).exists():
            out.append(aid)
    return sorted(out), g


def cmd_build(task):
    ctx = bc.load_ctx(task)
    aids, _ = certified_aids(task)
    test_ids = sorted(ctx["test"])
    dom = TASK_DOMAIN[task]
    outp = OUT / f"seed_g_{task}_prompts.jsonl"
    n = 0
    with open(outp, "w") as f:
        for aid in aids:
            pack = json.load(open(pack_path(task, aid)))
            body = SEED_TEMPLATE.format(domain=dom, name=pack["criterion_name"],
                                        desc=pack["criterion_description"], marker=DOC_MARKER)
            for dpid in test_ids:
                text = ctx["items"].get(dpid, "")[:MAXCHARS]
                prompt = body.replace(DOC_MARKER, text) + FOOTER
                f.write(json.dumps({"channel": "field", "aspect_id": f"{task}.{aid}.final",
                                    "datapoint_id": dpid, "prompt": prompt}) + "\n")
                n += 1
    print(f"{task}: {len(aids)} criteria x {len(test_ids)} test = {n} rows -> {outp}")


MIN_MINORITY = 5  # both instrument AND judge need >=5 items off their modal value on the
                  # scored selection, else spearman is tie-dominated / single-item driven


def variance_ok(vals, min_minority=MIN_MINORITY):
    from collections import Counter
    c = Counter(vals)
    return len(vals) - c.most_common(1)[0][1] >= min_minority


def cmd_eval(task, results):
    ctx = bc.load_ctx(task)
    _, gate = certified_aids(task)
    test_ids = sorted(ctx["test"])
    g_by = {}
    for line in open(results):
        r = json.loads(line); a = r.get("aspect_id", "")
        if not a.endswith(".final"):
            continue
        aid = a.split(".")[1]; sc = r.get("score")
        if isinstance(sc, int):
            g_by.setdefault(aid, {})[r["datapoint_id"]] = sc
    rows = []; skipped = []
    print(f"{'aid':6s} {'base':>6s} {'G':>7s} {'ceil':>6s} | {'r_base':>7s} {'r_G':>6s} {'seamW':>6s}  n")
    for aid, col_g in sorted(g_by.items()):
        judge = ctx["judge"].get(aid, {})
        sel = [d for d in test_ids if d in judge and col_g.get(d) is not None]
        if len(sel) < 20:
            skipped.append((aid, f"n={len(sel)}<20")); continue
        gv = [col_g[d] for d in sel]; jv = [judge[d] for d in sel]
        if not variance_ok(gv) or not variance_ok(jv):
            skipped.append((aid, "degenerate: <5 off-modal items (instrument or judge)")); continue
        rho_g = spearman(gv, jv)
        if rho_g != rho_g:  # nan guard — clip01(nan) would silently become 1.0
            skipped.append((aid, "spearman nan")); continue
        base = gate[aid]["full"]["rho_baseline"]; ceil = ceiling(gate[aid]["judge_rel1"])
        r_base = clip01(base / ceil); r_g = clip01(rho_g / ceil)
        seamw = (r_g - r_base) / r_g if r_g > 0 else float("nan")
        rows.append(dict(aid=aid, base=round(base, 4), G=round(rho_g, 4), ceil=round(ceil, 4),
                         r_base=round(r_base, 3), r_G=round(r_g, 3), seam_width=round(seamw, 3),
                         n_test=len(sel)))
        print(f"{aid:6s} {base:6.3f} {rho_g:7.3f} {ceil:6.3f} | {r_base:7.3f} {r_g:6.3f} {seamw:6.3f}  {len(sel)}")
    for aid, why in skipped:
        print(f"SKIP {aid}: {why}")
    import statistics as st
    mb = st.median(r["r_base"] for r in rows); mg = st.median(r["r_G"] for r in rows)
    F = (mg - mb) / mg if mg else float("nan")
    summ = dict(task=task, n=len(rows), n_skipped=len(skipped), med_r_base=round(mb, 3),
                med_r_G=round(mg, 3), seam_width_F=round(F, 3), arm="seed-G (lower bound)",
                skipped=[f"{a}:{w}" for a, w in skipped])
    print(f"\n{task}: V=med_r_base {summ['med_r_base']}  V+A=med_r_G {summ['med_r_G']}  "
          f"-> F={summ['seam_width_F']}  (n={summ['n']}, skipped {len(skipped)})")
    json.dump({"rows": rows, "domain_summary": {task: summ}},
              open(OUT / f"seed_g_{task}_final.json", "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("build", "eval"):
        print("usage: seed_g_extend.py build <task> | eval <task> <results.jsonl>"); sys.exit(1)
    if sys.argv[1] == "build":
        cmd_build(sys.argv[2])
    else:
        cmd_eval(sys.argv[2], sys.argv[3])
