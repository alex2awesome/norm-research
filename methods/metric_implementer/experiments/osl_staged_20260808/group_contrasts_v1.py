"""Group-contrast analysis v1 (user 2026-08-11 night): per-metric master feature table,
then three contrasts, each profiled on the OTHER features (within-task where possible):
  A. codable vs rest
  B. best-channel groups (humor bank: name / definition / examples winner)
  C. scaling groups (OSL fit verdict, 8 domains; staircase satgroup, 5 tasks)
Descriptive (report results, not verdicts). Output: outputs/analyses/group_contrasts_v1/.
"""
import csv
import glob
import json
import os
import re
from collections import Counter

import numpy as np
from scipy import stats as sps

D = "outputs/articulation_story_20260810"
OUT = "outputs/analyses/group_contrasts_v1"
os.makedirs(OUT, exist_ok=True)
TASKS8 = ["humor", "creative_writing", "math", "peer_review", "news_homepages",
          "press_releases", "notice_and_comment", "patents"]

# ---------------- features ----------------
M = {}   # (task, name) -> dict

for t in TASKS8:
    for name, units in json.load(open(f"{D}/code_metrics/unit_codability_{t}.json"))["metrics"].items():
        labs = [u["label"] for u in units]
        M[(t, name)] = {"task": t, "name": name,
                        "pct_cod": 100 * sum(l == "MECHANICAL" for l in labs) / len(labs),
                        "pct_taste": 100 * sum(l == "TASTE" for l in labs) / len(labs)}

for t in TASKS8:
    for n, v in json.load(open(f"notebooks/data/2026-07-07-osl-multi/curves_{t}.json")).items():
        if v.get("kind") == "bank" and (t, n) in M:
            M[(t, n)]["oslv"] = v.get("verdict")
            M[(t, n)]["L"] = v.get("L")

fj = json.load(open(f"{D}/analyses/family_verdict_join_v1.json"))["full_rows"]
for r in fj:
    tm = [v for v in r["top_minus_mid"].values() if v is not None]
    if len(tm) >= 3 and (r["task"], r["name"]) in M:
        g = "rising" if all(v > .02 for v in tm) else ("plateaued" if all(v <= .02 for v in tm)
                                                       else "family-dep")
        M[(r["task"], r["name"])]["sat"] = g

# 9-type labels via the O-id join (ladder_p123 logic)
types = json.load(open("outputs/analyses/osl_metric_types_20260728.json"))
labels, key = types["labels"], types["key"]
AX = {"MECHANICAL_CHECK": "in-text", "CRAFT_OPERATION": "in-text", "EVIDENCE_RIGOR": "in-text",
      "HOLISTIC_TASTE": "in-text", "AUDIENCE_FIT": "interface", "COMMUNITY_TRANSFORM": "interface",
      "IDENTITY_PERSONA": "beyond", "EXTERNAL_BUNDLE": "beyond", "RECEPTION_OUTCOME": "beyond"}
oid = 0
for f in sorted(glob.glob("notebooks/data/2026-07-07-osl-multi/curves_*.json")):
    dom = os.path.basename(f)[7:-5]
    for name, v in json.load(open(f)).items():
        if v.get("verdict") not in ("RISING", "REACHES", "BOUNDED"):
            continue
        i = "O%04d" % oid
        assert key[i]["verdict"] == v["verdict"] and key[i]["dom"] == dom
        lab = labels[i]["label"] if isinstance(labels[i], dict) else labels[i]
        if (dom, name) in M:
            M[(dom, name)]["type9"] = lab
            M[(dom, name)]["axis"] = AX[lab]
        oid += 1

for r in json.load(open("outputs/analyses/ladder_p123_v1/ladder_p123_v1.json"))["rows"]:
    if (r["task"], r["name"]) in M:
        M[(r["task"], r["name"])]["rung"] = r["rung"]

# concreteness (audited instrument)
CONC = {}
for rec in csv.DictReader(open(f"{D}/analyses/brysbaert_concreteness.csv"), delimiter="\t"):
    try:
        CONC[rec["Word"].lower()] = float(rec["Conc.M"])
    except (ValueError, KeyError):
        pass
FUNC = set(('''the a an and or but nor of to in for with on by at from into over under as that this
these those it its they them their there here is are was were be been being am do does did done
have has had having will would shall should can could may might must not no yes if then else when
while which who whom whose what where why how all any some none each every either neither both few
many much more most other another such only own same so than too very just also again further once
about against between through during before after above below up down out off very'''.split()))


def lemma(w):
    for suf, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e")):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            c = w[:len(w) - len(suf)] + rep
            if c in CONC:
                return c
    return w


for t in TASKS8:
    for m in json.load(open(f"{D}/code_metrics/defs_{t}.json")):
        if (t, m["name"]) in M:
            toks = [w for w in re.findall(r"[a-z]+", (m["name"] + " " + m["definition"]).lower())
                    if w not in FUNC]
            vals = [CONC[lemma(w)] for w in toks if lemma(w) in CONC]
            if len(vals) >= 5:
                M[(t, m["name"])]["conc"] = float(np.mean(vals))

# humor bank channels: name/def/examples holdout at the frontier2v ref (v3)
v3 = json.load(open(f"{D}/flips/flip_functional_v3_llama70b.json"))["results"]["humor"]
for b, rec in v3.items():
    h = rec["objectives"].get("frontier2v", {}).get("holdout", {})
    if all(h.get(k) is not None for k in ("name", "definition", "functional")) and ("humor", b) in M:
        M[("humor", b)].update(ch_name=h["name"], ch_def=h["definition"], ch_ex=h["functional"])
        arms = {"name": h["name"], "definition": h["definition"], "examples": h["functional"]}
        M[("humor", b)]["best_ch"] = max(arms, key=arms.get)

led = json.load(open(f"{D}/analyses/v3sets_ledger_v1.json"))
for r in led["qwen25-72b"]:
    if ("humor", r["b"]) in M:
        M[("humor", r["b"])]["content_q"] = r["functional"] - r["functionalmm"]
cat = json.load(open(f"{D}/analyses/metric_categories_blind_v1.json"))
for b, c in cat.items():
    if ("humor", b) in M:
        M[("humor", b)]["cat"] = c

rows = list(M.values())
print(f"master table: {len(rows)} metrics; feature coverage:",
      {k: sum(1 for r in rows if k in r) for k in
       ("oslv", "sat", "type9", "rung", "conc", "best_ch", "content_q", "cat")})
json.dump(rows, open(f"{OUT}/master_table_v1.json", "w"), indent=0)


def prof(sub, all_, label, feats=("type9", "axis", "oslv", "sat", "rung", "cat", "best_ch")):
    """enrichment profile: for categorical feats, share in sub vs share in rest + top enrichments"""
    rest = [r for r in all_ if r not in sub]
    print(f"\n### {label}: n={len(sub)} vs rest n={len(rest)}")
    for f in feats:
        a = Counter(r[f] for r in sub if f in r)
        b = Counter(r[f] for r in rest if f in r)
        na, nb = sum(a.values()), sum(b.values())
        if na < 10 or nb < 10:
            continue
        ers = []
        for k in set(a) | set(b):
            pa, pb = a[k] / na, b[k] / nb
            if a[k] + b[k] >= 8 and (pa > 0.03 or pb > 0.03):
                ers.append((k, pa, pb, pa - pb))
        ers.sort(key=lambda x: -abs(x[3]))
        line = "; ".join(f"{k}: {100*pa:.0f}% vs {100*pb:.0f}%" for k, pa, pb, _ in ers[:4])
        if line:
            print(f"  {f:8s} {line}")
    for f in ("pct_cod", "pct_taste", "conc", "ch_def", "ch_ex", "content_q", "L"):
        va = [r[f] for r in sub if f in r and r[f] is not None]
        vb = [r[f] for r in rest if f in r and r[f] is not None]
        if len(va) >= 10 and len(vb) >= 10:
            u = sps.mannwhitneyu(va, vb)
            star = "*" if u.pvalue < .05 else " "
            print(f"  {f:9s} med {np.median(va):7.3f} vs {np.median(vb):7.3f} (MW p={u.pvalue:.3g}){star}")


print("\n" + "=" * 90)
print("A. CODABLE vs REST  (codable = any MECHANICAL unit; also shown: strongly-codable >= 40%)")
print("=" * 90)
cod = [r for r in rows if r["pct_cod"] > 0]
prof(cod, rows, "any-codable (pooled, 8 domains)")
strong = [r for r in rows if r["pct_cod"] >= 40]
prof(strong, rows, "strongly-codable >=40% (pooled)")
print("\nwithin-task any-codable share:",
      {t: f"{100*np.mean([r['pct_cod']>0 for r in rows if r['task']==t]):.0f}%" for t in TASKS8})
# within-task type enrichment for codable (pooled across tasks but computed within, then summed)
wt = Counter()
wtn = Counter()
for t in TASKS8:
    tr = [r for r in rows if r["task"] == t and "type9" in r]
    for r in tr:
        wt[(r["type9"], r["pct_cod"] > 0)] += 1
print("\n9-type x any-codable (all-domain counts):")
for ty in sorted({k[0] for k in wt}):
    c1, c0 = wt[(ty, True)], wt[(ty, False)]
    if c1 + c0 >= 20:
        print(f"  {ty:22s} codable {c1:3d} / uncodable {c0:3d}  ({100*c1/(c1+c0):.0f}% codable)")

print("\n" + "=" * 90)
print("B. BEST-CHANNEL groups (humor bank, name/definition/examples holdout, frontier ref)")
print("=" * 90)
hum = [r for r in rows if r["task"] == "humor" and "best_ch" in r]
for chn in ("name", "definition", "examples"):
    sub = [r for r in hum if r["best_ch"] == chn]
    prof(sub, hum, f"best-channel = {chn}", feats=("type9", "axis", "oslv", "sat", "rung", "cat"))
    names = sorted(sub, key=lambda r: -(r.get("ch_" + {"name": "name", "definition": "def",
                                                       "examples": "ex"}[chn], 0)))[:8]
    print("   e.g.:", "; ".join(n["name"][:42] for n in names[:6]))

print("\n" + "=" * 90)
print("C. SCALING groups")
print("=" * 90)
osl = [r for r in rows if r.get("oslv") in ("RISING", "REACHES", "BOUNDED")]
for g in ("RISING", "REACHES", "BOUNDED"):
    prof([r for r in osl if r["oslv"] == g], osl, f"OSL verdict = {g}",
         feats=("type9", "axis", "sat", "rung", "cat", "best_ch"))
sat5 = [r for r in rows if "sat" in r]
for g in ("rising", "family-dep", "plateaued"):
    prof([r for r in sat5 if r["sat"] == g], sat5, f"staircase = {g} (5 tasks)",
         feats=("type9", "axis", "oslv", "rung", "cat", "best_ch"))
print("\nDONE ->", OUT)
