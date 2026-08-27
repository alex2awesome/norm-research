"""Seam-pilot analysis: per-aspect kappa_e(L) profile, disagreement mass, regime call.

Inputs: outputs/metric_seam_pilot/{items.json, code_scores.json, results.jsonl}
Output: printed seam table + outputs/metric_seam_pilot/seam_table.json
"""
import json, math, pathlib, statistics as st
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot"
CODEGEN = ROOT / "runs/validity_full/v2/press_releases/codegen_claude"

ASPECTS = ["a79", "a80", "a110", "a100", "a101", "a86", "a105", "a118", "a117", "a73"]
FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]
NAMES = {
    "a79": "Wire-ready format", "a80": "Media contacts", "a110": "Boilerplate/metadata",
    "a100": "Lede 5Ws", "a101": "Inverted pyramid", "a86": "Quote quality",
    "a105": "Plain language", "a118": "Timeliness signaled", "a117": "Newsworthiness/hook",
    "a73": "Empathy/sensitivity",
}
EXPECT = {
    "a79": "V", "a80": "V", "a110": "V",
    "a100": "boundary", "a101": "boundary", "a86": "boundary",
    "a105": "boundary", "a118": "boundary",
    "a117": "A", "a73": "A",
}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 10:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return k, 1 - po  # kappa, disagreement mass delta


def median_split(vals):
    med = st.median(vals)
    return [1 if v > med else 0 for v in vals]


def loc(path):
    return sum(1 for l in open(path) if l.strip())


def main():
    code = json.load(open(OUT / "code_scores.json"))
    llm = {}          # aspect -> {dpid: score 0-10}
    na = Counter()
    pfail = Counter()
    for line in open(OUT / "results.jsonl"):
        r = json.loads(line)
        s = r["score"]
        if s == "NA":
            na[r["aspect_id"]] += 1
        elif s is None:
            pfail[r["aspect_id"]] += 1
        else:
            llm.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = s

    table = []
    for aid in ASPECTS:
        scores = llm.get(aid, {})
        vals = list(scores.values())
        row = {"aspect": aid, "name": NAMES[aid], "expected": EXPECT[aid],
               "n": len(vals), "na": na[aid], "parse_fail": pfail[aid]}
        if len(vals) < 30:
            row["verdict"] = "LLM-CHANNEL-DEGENERATE (too few scores)"
            table.append(row)
            continue
        hist = Counter(vals)
        modal_frac = hist.most_common(1)[0][1] / len(vals)
        row["llm_mean"] = round(st.mean(vals), 2)
        row["llm_sd"] = round(st.pstdev(vals), 2)
        row["llm_modal_frac"] = round(modal_frac, 2)
        row["collapsed"] = modal_frac > 0.9 or st.pstdev(vals) < 0.4

        profile = {}
        for fl in FLAVORS:
            col = code.get(f"{aid}_{fl}")
            if col is None:
                profile[fl] = {"status": "BROKEN"}
                continue
            pairs = [(col[d], scores[d]) for d in scores
                     if col.get(d) is not None]
            if len(pairs) < 30:
                profile[fl] = {"status": "too-few"}
                continue
            cx = [p[0] for p in pairs]
            ly = [p[1] for p in pairs]
            rho = spearman(cx, ly)
            kap, delta = cohen_kappa(median_split(cx), median_split(ly))
            profile[fl] = {"rho": round(rho, 3), "kappa": round(kap, 3),
                           "delta": round(delta, 3),
                           "loc": loc(CODEGEN / f"{aid}_{fl}.py")}
        row["profile"] = profile

        rhos = [(p.get("rho"), fl) for fl, p in profile.items()
                if isinstance(p.get("rho"), float) and not math.isnan(p["rho"])]
        if row["collapsed"]:
            row["verdict"] = "LLM-CHANNEL-DEGENERATE (collapse)"
        elif not rhos:
            row["verdict"] = "NO-WORKING-PROGRAM (codegen failed all rungs)"
        else:
            best_rho, best_fl = max(rhos)
            row["best"] = f"{best_fl} rho={best_rho:.2f}"
            if best_rho >= 0.6:
                row["verdict"] = "CODABLE-NOW (candidate V; run MIGRATE gate)"
            elif best_rho >= 0.35:
                row["verdict"] = "BOUNDARY (partial code signal; adjudicate residual)"
            else:
                row["verdict"] = "A-LAYER (resisted codegen at all rungs tried)"
        table.append(row)

    json.dump(table, open(OUT / "seam_table.json", "w"), indent=1)

    print(f"{'aspect':6} {'name':22} {'exp':8} {'NA':>3} {'sd':>4} "
          f"{'rho v0':>7} {'rho v1':>7} {'rho v2':>7} verdict")
    for row in table:
        p = row.get("profile", {})
        def r(fl):
            q = p.get(fl, {})
            v = q.get("rho")
            return f"{v:+.2f}" if isinstance(v, float) else q.get("status", "-")[:5]
        print(f"{row['aspect']:6} {row['name'][:22]:22} {row['expected']:8} "
              f"{row.get('na',0):>3} {row.get('llm_sd','-'):>4} "
              f"{r('v0_keyword'):>7} {r('v1_structure'):>7} {r('v2_holistic'):>7} "
              f"{row.get('verdict','')}")


if __name__ == "__main__":
    main()
