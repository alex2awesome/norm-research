"""Bonus: on the competition corpus, does judged 'code quality' predict actual CORRECTNESS?

The competition corpus carries an EXTERNAL ground-truth anchor (verdict AC/WA/TLE/RE/CE) that
the judge never saw. For each measurable code_review quality aspect, correlate:
  - judge score vs is_correct(AC)      -- does the LLM's quality judgment track correctness?
  - best code-program vs is_correct    -- does the description-compiled program track it?
This is the first anchored check in the seam project (everything else certifies reproduction of
the judge, not correctness).
"""
import json, math, pathlib, statistics as st

R = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research")
OUT = R / "outputs/metric_seam_pilot/tasks/code_competition"


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def main():
    items = json.load(open(OUT / "items.json"))
    verdict = {it["datapoint_id"]: it.get("verdict") for it in items}
    correct = {d: (1 if v == "AC" else 0) for d, v in verdict.items() if v and v != "unknown"}
    names = {a["aspect_id"]: a["name"]
             for a in json.load(open(R / "runs/validity_full/v2/code_review/aspects.json"))}
    code = json.load(open(OUT / "code_scores.json"))

    p1, p2 = {}, {}
    for line in open(OUT / "results.jsonl"):
        r = json.loads(line)
        if not isinstance(r["score"], int):
            continue
        d = p1 if r["channel"] == "pass1" else p2 if r["channel"] == "pass2" else None
        if d is not None:
            d.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    judge = {}
    for aid in set(p1) | set(p2):
        for d in set(p1.get(aid, {})) | set(p2.get(aid, {})):
            vals = [m[aid][d] for m in (p1, p2) if d in m.get(aid, {})]
            judge.setdefault(aid, {})[d] = sum(vals) / len(vals)

    seam = {r["aspect"]: r for r in json.load(open(OUT / "seam_table.json"))["table"]}
    ac_rate = st.mean(correct.values())
    print(f"anchor: {len(correct)} graded submissions, AC-rate {ac_rate:.2f} "
          f"(verdicts: { {v: sum(1 for x in verdict.values() if x==v) for v in set(verdict.values())} })")
    print(f"{'aspect':7} {'judge~AC':>9} {'code~AC':>9} {'judge~code':>11}  name")
    rows = []
    for aid in sorted(judge, key=lambda a: -abs(_safe(_corr_ac(judge.get(a, {}), correct)))):
        jsel = [d for d in judge[aid] if d in correct]
        if len(jsel) < 40:
            continue
        j_ac = spearman([judge[aid][d] for d in jsel], [correct[d] for d in jsel])
        best = seam.get(aid, {}).get("best_flavor")
        c_ac = float("nan")
        jc = float("nan")
        if best and code.get(f"{aid}_{best}"):
            col = code[f"{aid}_{best}"]
            csel = [d for d in jsel if col.get(d) is not None]
            if len(csel) >= 40:
                c_ac = spearman([col[d] for d in csel], [correct[d] for d in csel])
                jc = spearman([judge[aid][d] for d in csel], [col[d] for d in csel])
        rows.append({"aspect": aid, "name": names.get(aid, ""), "judge_ac": _r(j_ac),
                     "code_ac": _r(c_ac), "judge_code": _r(jc), "n": len(jsel)})
        print(f"{aid:7} {_f(j_ac):>9} {_f(c_ac):>9} {_f(jc):>11}  {names.get(aid,'')[:40]}")
    json.dump({"ac_rate": ac_rate, "n_graded": len(correct), "rows": rows},
              open(OUT / "verdict_anchor.json", "w"), indent=1)
    js = [r["judge_ac"] for r in rows if r["judge_ac"] is not None]
    cs = [r["code_ac"] for r in rows if r["code_ac"] is not None]
    print(f"\nmedian |judge~AC| across {len(js)} aspects: {st.median([abs(x) for x in js]):.3f}"
          f"   median |code~AC|: {st.median([abs(x) for x in cs]):.3f}")
    print(f"most NEGATIVE judge~AC: {min(js):.3f} "
          f"({[r['aspect'] for r in rows if r['judge_ac']==min(js)][0]})   "
          f"most POSITIVE: {max(js):.3f} "
          f"({[r['aspect'] for r in rows if r['judge_ac']==max(js)][0]})")
    print(f"aspects where judge~AC and code~AC DISAGREE IN SIGN: "
          f"{[r['aspect'] for r in rows if r['code_ac'] and r['judge_ac'] and r['judge_ac']*r['code_ac']<0]}")


def _corr_ac(jm, correct):
    sel = [d for d in jm if d in correct]
    return spearman([jm[d] for d in sel], [correct[d] for d in sel]) if len(sel) >= 40 else 0.0


def _safe(x):
    return 0.0 if x != x else x


def _r(x):
    return None if x != x else round(x, 3)


def _f(x):
    return "  ." if x != x else f"{x:.3f}"


if __name__ == "__main__":
    main()
