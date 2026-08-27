"""Skeptical line-by-line audit of the code_competition verdict-anchor claims (2026-07-03).

Checks, per claim:
  C1 "255 graded, 47% AC"            -> stratified BY CONSTRUCTION; population AC-rate differs
  C2 "median |judge~AC| = 0.103"      -> clustering (canonical_pid), significance, attenuation
  C3 "a180 clean-formatting -0.437"   -> language confound, verdict-class breakdown, code-length
                                         confound, difficulty confound, judge-collapse check,
                                         permutation p, within-language stratified estimate
  C4 "a36/a153 positive"              -> is a153 driven purely by TLE?
  C5 "9 aspects disagree in sign"     -> require BOTH sides Bonferroni-significant
"""
import json, math, random, re, statistics as st, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot/tasks/code_competition"
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman

items = json.load(open(OUT / "items.json"))
meta = {}
for it in items:
    m = re.search(r"SUBMITTED SOLUTION \(([^)]*)\)", it["text"])
    d = re.search(r"difficulty: ([^|]*)\|", it["text"])
    code = it["text"].split("SUBMITTED SOLUTION", 1)[-1]
    meta[it["datapoint_id"]] = {
        "verdict": it.get("verdict"), "pid": it.get("canonical_pid"),
        "lang": (m.group(1) if m else "?").strip().lower(),
        "difficulty": (d.group(1).strip() if d else ""),
        "code_len": len(code)}

def langfam(l):
    if "py" in l: return "python"
    if l in ("cpp", "c++", "gnu c++", "c"): return "cpp"
    return "other"

correct = {d: 1 if v["verdict"] == "AC" else 0 for d, v in meta.items()
           if v["verdict"] and v["verdict"] != "unknown"}

p1, p2 = {}, {}
for line in open(OUT / "results.jsonl"):
    r = json.loads(line)
    if not isinstance(r["score"], int) or r["aspect_id"] == "scope":
        continue
    dd = p1 if r["channel"] == "pass1" else p2 if r["channel"] == "pass2" else None
    if dd is not None:
        dd.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
judge = {}
for aid in set(p1) | set(p2):
    for d in set(p1.get(aid, {})) | set(p2.get(aid, {})):
        vals = [m[aid][d] for m in (p1, p2) if d in m.get(aid, {})]
        judge.setdefault(aid, {})[d] = sum(vals) / len(vals)

names = {a["aspect_id"]: a["name"] for a in
         json.load(open(ROOT / "runs/validity_full/v2/code_review/aspects.json"))}

def perm_p(x, y, B=4000, seed=0):
    obs = abs(spearman(x, y))
    rng = random.Random(seed); yy = list(y); hits = 0
    for _ in range(B):
        rng.shuffle(yy)
        if abs(spearman(x, yy)) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (B + 1)

# ---------------- C1: prevalence by construction --------------------------------
print("C1  sample AC-rate:", round(st.mean(correct.values()), 3), f"(n={len(correct)})",
      "| stratified sample {AC:120, WA:60, TLE:30, RE:30, CE:15} -> BY CONSTRUCTION.")
print("    population (candidates.parquet, known verdicts): AC 293777/425190 = 0.691")

# ---------------- C2: clustering + significance ---------------------------------
pids = [meta[d]["pid"] for d in correct]
print(f"\nC2  clustering: {len(set(pids))} unique problems over {len(correct)} graded items",
      f"(max per problem {max(st.multimode([pids.count(p) for p in set(pids)]))if pids else 0})")
langs = st.multimode
n = len(correct)
bonf = 0.05 / 17
print(f"    n={n}; Spearman SE~1/sqrt(n-3)={1/math.sqrt(n-3):.3f}; "
      f"Bonferroni(17) alpha={bonf:.4f} -> |r| threshold ~{2.94/math.sqrt(n-3):.3f}")

rows = []
for aid in sorted(judge):
    sel = [d for d in judge[aid] if d in correct]
    if len(sel) < 40:
        continue
    j = [judge[aid][d] for d in sel]; y = [correct[d] for d in sel]
    r = spearman(j, y)
    p = perm_p(j, y, B=2000, seed=hash(aid) % 9999)
    modal = max(j.count(v) for v in set(j)) / len(j)
    rows.append((aid, r, p, len(sel), modal))
sig = [x for x in rows if x[2] < bonf]
print(f"    Bonferroni-significant judge~AC aspects: "
      f"{[(a, round(r,3)) for a, r, p, _, _ in sig]}")
print(f"    NOT significant: {[(a, round(r,3)) for a, r, p, _, _ in rows if p >= bonf]}")

# ---------------- C3: a180 deep-dive ---------------------------------------------
aid = "a180"
sel = [d for d in judge[aid] if d in correct]
j = {d: judge[aid][d] for d in sel}
print(f"\nC3  a180 '{names[aid][:50]}' n={len(sel)}")
modal = max(list(j.values()).count(v) for v in set(j.values())) / len(sel)
print(f"    judge collapse check: modal-score fraction {modal:.2f} "
      f"(sd {st.pstdev(j.values()):.2f}, {len(set(j.values()))} distinct) -> "
      f"{'COLLAPSED' if modal > 0.8 else 'ok spread'}")
print(f"    pooled judge~AC = {spearman([j[d] for d in sel], [correct[d] for d in sel]):.3f} "
      f"(perm p={perm_p([j[d] for d in sel], [correct[d] for d in sel]):.4f})")
print("    verdict-class mean judge score:")
for v in ("AC", "WA", "TLE", "RE", "CE"):
    g = [j[d] for d in sel if meta[d]["verdict"] == v]
    if g:
        print(f"      {v:3} n={len(g):3}  mean={st.mean(g):.2f}")
print("    language strata (judge~AC within stratum):")
wsum = 0.0; wtot = 0
for fam in ("python", "cpp", "other"):
    g = [d for d in sel if langfam(meta[d]["lang"]) == fam]
    if len(g) >= 20:
        ac = st.mean(correct[d] for d in g)
        r = spearman([j[d] for d in g], [correct[d] for d in g])
        print(f"      {fam:7} n={len(g):3} AC-rate={ac:.2f}  judge~AC={r: .3f}")
        if r == r:
            wsum += r * len(g); wtot += len(g)
print(f"      within-language weighted judge~AC = {wsum/max(wtot,1): .3f} "
      f"(vs pooled -0.437; gap = language confound share)")
r_len_j = spearman([j[d] for d in sel], [meta[d]["code_len"] for d in sel])
r_len_ac = spearman([meta[d]["code_len"] for d in sel], [correct[d] for d in sel])
print(f"    code-length confound: judge~len={r_len_j:.3f}, len~AC={r_len_ac:.3f}")
diffs_num = {d: float(meta[d]["difficulty"]) for d in sel
             if re.fullmatch(r"\d+(\.\d+)?", meta[d]["difficulty"] or "")}
if len(diffs_num) >= 40:
    dsel = sorted(diffs_num)
    print(f"    difficulty confound (n={len(dsel)}): "
          f"judge~diff={spearman([j[d] for d in dsel], [diffs_num[d] for d in dsel]):.3f}, "
          f"diff~AC={spearman([diffs_num[d] for d in dsel], [correct[d] for d in dsel]):.3f}")

# ---------------- C4: a153 -- TLE-driven? -----------------------------------------
aid = "a153"
sel = [d for d in judge[aid] if d in correct]
j = {d: judge[aid][d] for d in sel}
no_tle = [d for d in sel if meta[d]["verdict"] != "TLE"]
print(f"\nC4  a153 hot-path perf: pooled judge~AC="
      f"{spearman([j[d] for d in sel], [correct[d] for d in sel]):.3f}; "
      f"excluding TLE (n={len(no_tle)}): "
      f"{spearman([j[d] for d in no_tle], [correct[d] for d in no_tle]):.3f}")

# ---------------- C5: sign disagreements, both-significant ------------------------
code = json.load(open(OUT / "code_scores.json"))
seam = {r["aspect"]: r for r in json.load(open(OUT / "seam_table.json"))["table"]}
both_sig = []
for aid, rj, pj, nn, _ in rows:
    fl = seam.get(aid, {}).get("best_flavor")
    col = code.get(f"{aid}_{fl}") if fl else None
    if not col:
        continue
    csel = [d for d in judge[aid] if d in correct and col.get(d) is not None]
    if len(csel) < 40:
        continue
    c = [col[d] for d in csel]; y = [correct[d] for d in csel]
    rc = spearman(c, y)
    pc = perm_p(c, y, B=2000, seed=hash(aid + "c") % 9999)
    if rj * rc < 0 and pj < bonf and pc < bonf:
        both_sig.append((aid, round(rj, 3), round(rc, 3)))
print(f"\nC5  sign-disagreements with BOTH sides Bonferroni-significant: "
      f"{both_sig or 'NONE'} (claimed 9 -> raw sign flips without significance)")
