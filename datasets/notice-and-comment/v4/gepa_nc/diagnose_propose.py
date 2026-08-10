#!/usr/bin/env python3
"""GEPA round driver: diagnose Gemma-vs-Sonnet fidelity per rubric, have Sonnet
propose a revised Gemma-facing description for low-fidelity rubrics.

Label-free objective per rubric:
  fidelity = 0.5 * categorical agreement ({1.0,0.5,0.0,NA} exact match)
           + 0.5 * Spearman on items numeric under BOTH judges (0 if <8 such items)
computed on the 60-item dev set (+ anchors, which also gate degeneracy).

Usage:
  python3 diagnose_propose.py --round 1 --dev-npz dev_scores_r0.npz
Reads bank_r{K-1}.jsonl (r0 = original nc_rubrics.jsonl), writes bank_r{K}.jsonl
+ history.json (per-rubric per-round fidelity, best variant tracking).
Rubrics with fidelity >= --good-enough (default 0.75) are carried forward unchanged.
"""
import argparse, json, re, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
CONC = 6
NUM = {"1.0": 1.0, "0.5": 0.5, "0.0": 0.0}

def to_cat(v):
    if v is None: return None
    s = str(v).strip()
    if s in NUM: return s
    if s.upper().startswith("NA"): return "NA"
    try:
        f = float(s)
        return min(NUM, key=lambda k: abs(NUM[k] - f))
    except Exception:
        return None

def fidelity(ref_map, gem_map):
    keys = [k for k in ref_map if k in gem_map and to_cat(ref_map[k]) and to_cat(gem_map[k])]
    if len(keys) < 20:
        return None, []
    r = [to_cat(ref_map[k]) for k in keys]
    g = [to_cat(gem_map[k]) for k in keys]
    cat = float(np.mean([a == b for a, b in zip(r, g)]))
    num_keys = [k for k, a, b in zip(keys, r, g) if a != "NA" and b != "NA"]
    if len(num_keys) >= 8:
        rv = [NUM[to_cat(ref_map[k])] for k in num_keys]
        gv = [NUM[to_cat(gem_map[k])] for k in num_keys]
        rho = spearmanr(rv, gv).statistic
        rho = 0.0 if np.isnan(rho) else float(rho)
    else:
        rho = 0.0
    fid = 0.5 * cat + 0.5 * max(rho, 0.0)
    dis = sorted(keys, key=lambda k: -abs(NUM.get(to_cat(ref_map[k]), 0.5 if to_cat(ref_map[k])=="NA" else 0)
                                          - NUM.get(to_cat(gem_map[k]), 0.5 if to_cat(gem_map[k])=="NA" else 0))
                 if True else 0)
    worst = [(k, to_cat(ref_map[k]), to_cat(gem_map[k])) for k in dis if to_cat(ref_map[k]) != to_cat(gem_map[k])][:6]
    return fid, worst

PROPOSE_SYS = """You are improving the DESCRIPTION of a quality criterion so that a smaller judge model (Gemma-4-31B), scoring public comments on proposed federal rules with the one-token protocol (1.0 / 0.5 / 0.0 / NA), agrees better with a reference expert judge scoring the SAME construct.

Rules:
- The CONSTRUCT must not change: you are clarifying what to look for, adding crisp decision boundaries for 1.0 vs 0.5 vs 0.0, and stating explicitly when to answer NA vs 0.0. Do not broaden or narrow what the criterion measures.
- Keep the NAME unchanged. Rewrite only the description, <= 90 words, imperative and concrete.
- The disagreements below show where the smaller judge diverges from the reference; target those failure patterns.
Return ONLY JSON: {"description": "..."}"""

def build_propose_prompt(rubric, fid, worst, dev_texts):
    lines = [PROPOSE_SYS, "",
             f"CRITERION NAME: {rubric['name']}",
             f"ORIGINAL CONSTRUCT DESCRIPTION: {rubric.get('orig_description', rubric.get('description',''))}",
             f"CURRENT GEMMA-FACING DESCRIPTION: {rubric.get('description','')}",
             f"CURRENT FIDELITY: {fid:.2f}", "", "DISAGREEMENTS (reference vs gemma):"]
    for k, rv, gv in worst:
        snip = (dev_texts.get(k, "") or "")[:400].replace("\n", " ")
        lines.append(f"- ref={rv} gemma={gv} :: {snip}")
    return "\n".join(lines)

def call_sonnet(prompt):
    for _ in range(3):
        try:
            r = subprocess.run(["claude", "-p", "--model", "sonnet"], input=prompt,
                               capture_output=True, text=True, timeout=300)
            m = re.search(r"\{.*\}", r.stdout, re.S)
            if m:
                d = json.loads(m.group(0))
                if d.get("description"):
                    return d["description"].strip()
        except Exception:
            pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--dev-npz", required=True)
    ap.add_argument("--good-enough", type=float, default=0.75)
    a = ap.parse_args()

    prev_bank = HERE / (f"bank_r{a.round-1}.jsonl" if a.round > 1 else "bank_r0.jsonl")
    if a.round == 1 and not prev_bank.exists():
        # seed r0 = original bank, remembering the original construct text
        rs = [json.loads(l) for l in open(HERE.parent / "nc_rubrics.jsonl") if l.strip()]
        for r in rs:
            r["orig_description"] = r.get("description", "")
        prev_bank.write_text("\n".join(json.dumps(r) for r in rs))
    rubrics = [json.loads(l) for l in open(prev_bank) if l.strip()]

    ref = json.loads((HERE / "ref_scores.json").read_text())
    d = np.load(a.dev_npz, allow_pickle=True)
    a_names = [str(x) for x in d["a_names"]]
    gem = {}
    for j, nm in enumerate(a_names):
        col = {}
        for i, did in enumerate(d["doc_id"]):
            v = d["X"][i, j]
            col[str(did)] = "NA" if np.isnan(v) else f"{v:.1f}"
        gem[nm] = col

    dev_texts = {r["doc_id"]: r["text"] for r in (json.loads(l) for l in open(HERE / "dev.jsonl"))}
    for x in json.loads((HERE / "anchors.json").read_text()):
        dev_texts[x["doc_id"]] = x["text"]

    hist_p = HERE / "history.json"
    hist = json.loads(hist_p.read_text()) if hist_p.exists() else {}

    jobs, keep = [], []
    for r in rubrics:
        rid = str(r["rubric_id"])
        if rid not in ref or r["name"] not in gem:
            keep.append(r); continue
        fid, worst = fidelity(ref[rid], gem[r["name"]])
        if fid is None:
            keep.append(r); continue
        h = hist.setdefault(rid, {"best_fid": -1, "best_description": r.get("description", ""),
                                  "orig_description": r.get("orig_description", r.get("description", "")),
                                  "rounds": {}})
        h["rounds"][str(a.round - 1)] = fid
        if fid > h["best_fid"]:
            h["best_fid"] = fid
            h["best_description"] = r.get("description", "")
        if fid >= a.good_enough:
            keep.append(r)
        else:
            jobs.append((r, fid, worst))
    print(f"round {a.round}: {len(jobs)} rubrics below {a.good_enough}, {len(keep)} kept", flush=True)

    def work(job):
        r, fid, worst = job
        newd = call_sonnet(build_propose_prompt(r, fid, worst, dev_texts))
        return r, newd

    out = list(keep)
    with ThreadPoolExecutor(CONC) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for k, f in enumerate(as_completed(futs)):
            r, newd = f.result()
            r2 = dict(r)
            if newd:
                r2["description"] = newd
            out.append(r2)
            if (k + 1) % 20 == 0:
                print(f"proposed {k+1}/{len(jobs)}", flush=True)

    out.sort(key=lambda r: r["rubric_id"])
    (HERE / f"bank_r{a.round}.jsonl").write_text("\n".join(json.dumps(r) for r in out))
    hist_p.write_text(json.dumps(hist, indent=2))
    fids = [h["rounds"].get(str(a.round - 1)) for h in hist.values() if str(a.round - 1) in h["rounds"]]
    fids = [f for f in fids if f is not None]
    print(f"fidelity r{a.round-1}: mean={np.mean(fids):.3f} median={np.median(fids):.3f} "
          f"<0.5: {sum(f < 0.5 for f in fids)} -> bank_r{a.round}.jsonl", flush=True)
    print("PROPOSE_DONE", flush=True)

if __name__ == "__main__":
    main()
