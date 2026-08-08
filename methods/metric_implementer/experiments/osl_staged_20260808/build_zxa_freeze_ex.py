"""1c: build freeze_zxa_ex_humor_v1.json — exemplar arms for the z×a mode grid.

Arms per slate humor base: exemplars / def_exemplars / exemplars_mm (content placebo: partner
base's exemplars under this base's name). Exemplar labels come from the FROZEN LOCAL_MID crowd
(11 local mid-scale executors, same set qwen3_adjudicate.py froze) majority on the v2
definition panel (mbar285_<exec>.npz) — reconstruction-only, no human labels. Selected items'
probe indices are recorded in each entry's zxa.exemplar_idx: the fitter MUST mask those items
for the arms of that base (they appear verbatim in the rubric).
Entry format identical to zxa v1 ("<base>||<arm>", kind "<class>|<arm>") so
`osl_sweep --mbar-only` runs unchanged.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research")
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer import config as cfgmod

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
O = f"{B}/outputs/osl"
LOCAL_MID = ["llama1b", "llama3b", "llama8b", "llama70b", "qwen25-3b", "qwen25-7b",
             "qwen25-14b", "qwen25-72b", "mistral7b", "phi4", "gemma2-27b"]
N_POS = N_NEG = 4
LEN_CAP = 400

import re
slate = [m for m in json.load(open(f"{OM}/zxa_slate_v1.json")) if m["task"] == "humor"]
v1 = json.load(open(f"{OM}/freeze_zxa_humor_v1.json"))
doss_section = {}
for e in v1["metrics"]:
    if e["zxa"]["arm"] == "dossier":
        mm_ = re.search(r"CONTRAST EXEMPLARS[:\s]*(.*?)(?=BOUNDARY CASES)", e["rubric"], re.S)
        if mm_:
            doss_section[e["zxa"]["base"]] = mm_.group(1).strip()
v2 = json.load(open(f"{OM}/freeze_humor_v2.json"))
v2rub = {m["name"]: m["rubric"] for m in v2["metrics"]}

panels, names_ref = {}, None
for ex in LOCAL_MID:
    z = np.load(f"{O}/mbar285_{ex}.npz", allow_pickle=True)
    nm = [str(n) for n in z["names"]]
    if names_ref is None:
        names_ref = nm
    assert nm == names_ref, f"name order differs for {ex}"
    panels[ex] = z["m_bar"]
row = {n: i for i, n in enumerate(names_ref)}

cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
texts, _ = _load_texts("humor", 360, cfg)
probes = texts[60:360]
assert len(probes) == panels[LOCAL_MID[0]].shape[1] == 300

def pick_exemplars(base):
    if base not in row:
        return None
    i = row[base]
    votes = np.stack([(panels[ex][i] > 0.5).astype(float) for ex in LOCAL_MID])
    cons = votes.mean(0)  # fraction of crowd saying YES per item
    ok = [j for j, t in enumerate(probes) if len(t) <= LEN_CAP and t.strip()]
    pos = sorted(ok, key=lambda j: -cons[j])[:N_POS]
    neg = sorted(ok, key=lambda j: cons[j])[:N_NEG]
    if cons[pos[-1]] < 0.7 or cons[neg[-1]] > 0.3:
        return None  # crowd not decisive enough for clean exemplars
    return pos, neg

def block(pos, neg):
    p = "\n".join("- " + probes[j].strip() for j in pos)
    n = "\n".join("- " + probes[j].strip() for j in neg)
    return ("\nExamples that satisfy this criterion:\n%s\n"
            "Examples that do NOT satisfy it:\n%s" % (p, n))

chosen, skipped = {}, []
for m in slate:
    got = pick_exemplars(m["name"])
    if got is None:
        skipped.append(m["name"])
    else:
        chosen[m["name"]] = got

# mismatch partners: rotate within class among chosen bases
by_cls = {}
for m in slate:
    if m["name"] in chosen:
        by_cls.setdefault(m["class"], []).append(m["name"])
partner = {}
for cls, ns in by_cls.items():
    for a, b in zip(ns, ns[1:] + ns[:1]):
        partner[a] = b

entries = []
for m in slate:
    base = m["name"]
    if base not in chosen:
        continue
    pos, neg = chosen[base]
    ex_idx = sorted(pos + neg)
    blk = block(pos, neg)
    defn = v2rub.get(base, base)
    mm = partner[base]
    mp, mn = chosen[mm]
    for arm, rub, msrc in (
            ("exemplars", base + blk, None),
            ("def_exemplars", defn + blk, None),
            ("exemplars_mm", base + block(mp, mn), mm)):
        entries.append({
            "name": f"{base}||{arm}", "kind": f"{m['class']}|{arm}", "rubric": rub,
            "criteria": [],
            "zxa": {"base": base, "arm": arm, "class": m["class"],
                    "exemplar_idx": ex_idx if msrc is None else sorted(mp + mn),
                    "mismatch_src": msrc}})

out = {"meta": dict(v2["meta"], exemplar_freeze="v1", crowd=LOCAL_MID,
                    n_pos=N_POS, n_neg=N_NEG, len_cap=LEN_CAP,
                    note="mask zxa.exemplar_idx items at fit time (verbatim in rubric)",
                    skipped_bases=skipped),
       "metrics": entries}
# authored-exemplar arms: ALL slate bases (contested included) via the dossier's own
# CONTRAST EXEMPLARS section; placebo = partner base's section (rotate within class, all bases)
all_by_cls = {}
for m in slate:
    if m["name"] in doss_section:
        all_by_cls.setdefault(m["class"], []).append(m["name"])
apartner = {}
for cls, ns in all_by_cls.items():
    for a_, b_ in zip(ns, ns[1:] + ns[:1]):
        apartner[a_] = b_
n_auth = 0
for m in slate:
    base = m["name"]
    if base not in doss_section:
        continue
    sec = doss_section[base]
    mmb = apartner[base]
    for arm, rub, msrc in (
            ("exemplars_authored", base + "\nContrast examples:\n" + sec, None),
            ("exemplars_authored_mm", base + "\nContrast examples:\n" + doss_section[mmb], mmb)):
        entries.append({
            "name": f"{base}||{arm}", "kind": f"{m['class']}|{arm}", "rubric": rub,
            "criteria": [],
            "zxa": {"base": base, "arm": arm, "class": m["class"], "exemplar_idx": [],
                    "mismatch_src": msrc}})
        n_auth += 1
out["metrics"] = entries
out["meta"]["authored_arm"] = "CONTRAST EXEMPLARS section extracted from v1 dossiers; covers all bases incl. contested"
out["meta"]["crowd_exemplar_coverage_by_class"] = {c: len(v) for c, v in by_cls.items()}
json.dump(out, open(f"{OM}/freeze_zxa_ex_humor_v1.json", "w"), indent=1)
print(f"authored-arm entries added: {n_auth}")
print(f"bases kept {len(chosen)}/{len(slate)}; skipped (indecisive crowd): {len(skipped)}")
for s in skipped:
    print("  SKIP:", s[:60])
print(f"entries written: {len(entries)} -> freeze_zxa_ex_humor_v1.json")
