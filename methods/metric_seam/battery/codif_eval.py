"""CODIF merge + anchor QC + outcome-panel cross (R11).

1. Loads codif/batch_*.jsonl (12 Sonnet batches; anchors a104/press_releases and
   a42/math blinded into every batch).
2. Anchor QC per batch: tag-presence agreement vs hand-coded ground truth; batches
   below threshold flagged (re-run, don't trust).
3. Inter-annotator reliability on the anchors (11-12 independent annotations each).
4. Merge to one row per (task, aid) — native batch preferred — then cross tag
   composition with the outcome panel: fm (inventory + eval_scale), transport ratios
   (inventory), E1 fracs (key_eval_*), aperture kept (seampos_eval).

Usage: python3 codif_eval.py
-> outputs/metric_seam_pilot/battery/codif/codif_merged.jsonl + codif_summary.json
"""
import json, pathlib, sys
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import BASE  # noqa: E402

CODIF = BASE / "battery/codif"
TAGS = [f"C{i}" for i in range(1, 9)]

# hand-coded anchor truth; None = borderline, either accepted
ANCHORS = {
    ("press_releases", "a104"): {"C1": True, "C2": True, "C3": True, "C4": None,
                                 "C5": False, "C6": False, "C7": True, "C8": True,
                                 "c2_role": "GATE"},
    ("math", "a42"): {"C1": None, "C2": True, "C3": True, "C4": False,
                      "C5": True, "C6": False, "C7": True, "C8": True,
                      "c8_share": "HIGH"},
}
NATIVE = {("press_releases", "a104"): "batch_prA", ("math", "a42"): "batch_maC"}

# self-reported anchor contamination (agent read earlier batch files and copied or
# harmonized its anchor rows): excluded from QC + reliability even if not byte-identical
NONINDEP = {("batch_maB", ("press_releases", "a104")),
            ("batch_prB", ("press_releases", "a104")),
            ("batch_prB", ("math", "a42")),
            ("batch_leB", ("press_releases", "a104")),
            ("batch_leB", ("math", "a42")),
            ("batch_huB", ("press_releases", "a104")),
            ("batch_huB", ("math", "a42")),
            ("batch_leA", ("press_releases", "a104")),
            ("batch_leA", ("math", "a42")),
            ("batch_cwA", ("press_releases", "a104")),
            ("batch_cwA", ("math", "a42")),
            ("batch_maC", ("press_releases", "a104")),
            ("batch_huA", ("press_releases", "a104")),
            ("batch_huA", ("math", "a42"))}


def anchor_score(row, truth):
    ok = tot = 0
    for t in TAGS:
        want = truth.get(t)
        if want is None:
            continue
        tot += 1
        got = bool(row.get("tags", {}).get(t, {}).get("present"))
        ok += got == want
    if "c2_role" in truth:
        tot += 1
        ok += row.get("tags", {}).get("C2", {}).get("role") == truth["c2_role"]
    if "c8_share" in truth:
        tot += 1
        ok += row.get("c8_share") == truth["c8_share"]
    return ok, tot


def med(xs):
    xs = sorted(x for x in xs if x is not None and x == x)  # drop None AND NaN
    return round(xs[len(xs) // 2], 3) if xs else None


def main():
    batches = {}
    for p in sorted(CODIF.glob("batch_*.jsonl")):
        rows = []
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        batches[p.stem] = rows

    # ---- anchor QC (with copy detection: later batches can see earlier batch
    # files on disk; byte-identical evidence strings = copied, not independent) ----
    bad_batches = set()
    anchor_rows = defaultdict(list)   # (task,aid) -> [(batch, row, copied)]
    seen_sigs = defaultdict(dict)     # (task,aid) -> sig -> first batch
    print("== anchor QC ==")
    for bname, rows in sorted(batches.items()):
        for key, truth in ANCHORS.items():
            task, aid = key
            cand = [r for r in rows if r.get("aid") == aid and r.get("task") == task]
            if not cand:
                print(f"{bname}: MISSING anchor {aid}")
                bad_batches.add(bname)
                continue
            row = cand[0]
            sig = json.dumps([row.get("tags"), row.get("dominant_code_tags"),
                              row.get("c8_share"), row.get("llm_fields")],
                             sort_keys=True)
            copied = sig in seen_sigs[key] or (bname, key) in NONINDEP
            if copied:
                src = seen_sigs[key].get(sig, "self-reported harmonization")
                print(f"{bname} {aid}: NON-INDEPENDENT ({src}) — "
                      f"excluded from QC/reliability")
            else:
                seen_sigs[key][sig] = bname
                ok, tot = anchor_score(row, truth)
                flag = "" if ok >= tot - 1 else "  <-- FLAG"
                if ok < tot - 1:
                    bad_batches.add(bname)
                print(f"{bname} {aid}: {ok}/{tot}{flag}")
            anchor_rows[key].append((bname, row, copied))

    # ---- inter-annotator reliability on anchors ----
    print("== anchor inter-annotator tag-presence agreement ==")
    reliab = {}
    for key, triples in anchor_rows.items():
        indep = [(b, r) for b, r, copied in triples if not copied]
        if len(indep) < 2:
            continue
        agr = []
        for t in TAGS:
            vals = [bool(r.get("tags", {}).get(t, {}).get("present")) for _, r in indep]
            agr.append(max(Counter(vals).values()) / len(vals))
        reliab["/".join(key)] = round(sum(agr) / len(agr), 3)
        print(f"{key}: mean per-tag modal agreement {reliab['/'.join(key)]} "
              f"(n={len(indep)} independent)")

    # ---- merge (native batch wins; skip anchor dupes) ----
    merged = {}
    for bname, rows in batches.items():
        for r in rows:
            key = (r.get("task"), r.get("aid"))
            if key in ANCHORS and NATIVE[key] != bname:
                continue
            if key in merged:
                continue
            merged[key] = {**r, "_batch": bname,
                           "_flagged": bname in bad_batches}
    print(f"merged: {len(merged)} programs "
          f"({sum(1 for v in merged.values() if v['_flagged'])} from flagged batches)")
    with open(CODIF / "codif_merged.jsonl", "w") as f:
        for key in sorted(merged):
            f.write(json.dumps(merged[key]) + "\n")

    # ---- outcome panel ----
    inv = json.load(open(BASE / "battery/inventory.json"))
    scale = json.load(open(BASE / "battery/eval_scale.json"))
    keyev = {}
    for t in ["press_releases", "creative_writing", "math", "humor"]:
        p = BASE / f"battery/key_eval_{t}.json"
        if p.exists():
            keyev[t] = json.load(open(p))
    seam = {}
    p = BASE / "battery/seampos_eval.json"
    if p.exists():
        seam = json.load(open(p)).get("aspects", {})

    panel = {}
    for (task, aid), row in merged.items():
        o = {}
        iv = inv.get(task, {}).get(aid)
        if iv:
            o["fm"] = iv.get("fm")
            o["ratio_llama"] = iv.get("ratio_llama")
            o["ratio_qwen"] = iv.get("ratio_qwen")
        sc = scale.get(task, {}).get("criteria", {}).get(aid)
        if sc and "fm" in sc:
            o.setdefault("fm", sc["fm"].get("gemma"))
            o["fm_llama70"] = sc["fm"].get("llama70")
        ke = keyev.get(task, {})
        ker = ke.get(aid) or ke.get("aspects", {}).get(aid) if isinstance(ke, dict) else None
        if isinstance(ker, dict):
            o["frac_name"] = ker.get("frac_name")
            o["frac_nonce"] = ker.get("frac_nonce")
        if task == "press_releases" and aid in seam and "error" not in seam[aid]:
            o["aperture_kept"] = seam[aid].get("aperture_frac_kept")
        panel[(task, aid)] = o

    # ---- descriptive crosses ----
    summ = {"anchor_reliability": reliab,
            "flagged_batches": sorted(bad_batches)}
    # tag prevalence by task
    prev = defaultdict(lambda: defaultdict(int))
    ntask = Counter()
    for (task, aid), row in merged.items():
        ntask[task] += 1
        for t in TAGS:
            if row.get("tags", {}).get(t, {}).get("present"):
                prev[task][t] += 1
    summ["tag_prevalence"] = {task: {t: round(prev[task][t] / n, 2) for t in TAGS}
                              for task, n in ntask.items()}
    summ["c8_share_by_task"] = {task: dict(Counter(
        r.get("c8_share") for (tk, _), r in merged.items() if tk == task))
        for task in ntask}
    # thick-predicate census
    preds = Counter()
    preds_by_task = defaultdict(Counter)
    for (task, aid), row in merged.items():
        for f, tag in (row.get("llm_fields") or {}).items():
            preds[tag] += 1
            preds_by_task[task][tag] += 1
    summ["thick_predicates"] = dict(preds.most_common())
    summ["thick_predicates_by_task"] = {t: dict(c.most_common())
                                        for t, c in preds_by_task.items()}
    # per-tag outcome contrast: median fm with vs without tag
    contrasts = {}
    for t in TAGS:
        w = [panel[k].get("fm") for k, r in merged.items()
             if r.get("tags", {}).get(t, {}).get("present") and k in panel]
        wo = [panel[k].get("fm") for k, r in merged.items()
              if not r.get("tags", {}).get(t, {}).get("present") and k in panel]
        contrasts[t] = {"fm_with": med(w), "fm_without": med(wo),
                        "n_with": sum(1 for x in w if x is not None and x == x),
                        "n_without": sum(1 for x in wo if x is not None and x == x)}
    # transport ratio for C6 (exemplar-match) and C2 (signifier)
    for t in ("C2", "C6"):
        w = [panel[k].get("ratio_llama") for k, r in merged.items()
             if r.get("tags", {}).get(t, {}).get("present") and k in panel]
        wo = [panel[k].get("ratio_llama") for k, r in merged.items()
              if not r.get("tags", {}).get(t, {}).get("present") and k in panel]
        contrasts[t]["ratio_llama_with"] = med(w)
        contrasts[t]["ratio_llama_without"] = med(wo)
    # E1 nonce survival by c8_share
    by_share = defaultdict(list)
    for k, r in merged.items():
        fn = panel.get(k, {}).get("frac_nonce")
        if fn is not None:
            by_share[r.get("c8_share")].append(fn)
    contrasts["frac_nonce_by_c8_share"] = {s: (med(v), len(v))
                                           for s, v in by_share.items()}
    summ["contrasts"] = contrasts

    json.dump(summ, open(CODIF / "codif_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items()
                      if k in ("tag_prevalence", "thick_predicates")}, indent=1))
    print(f"-> {CODIF / 'codif_merged.jsonl'}, codif_summary.json")


if __name__ == "__main__":
    main()
