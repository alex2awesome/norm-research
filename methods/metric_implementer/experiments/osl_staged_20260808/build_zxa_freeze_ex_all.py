"""1c all-domains expansion: exemplar-arm freezes for creative_writing / peer_review / math /
news_homepages (humor already done). Same design as humor: corpus exemplars from the frozen
LOCAL_MID crowd consensus (decisiveness gate), authored exemplars from the v1 dossiers'
CONTRAST EXEMPLARS section (all bases incl. contested), mismatched placebos for both.
Long-text tasks (median probe > 800 chars): 3+3 exemplars truncated to 500 chars, else 4+4
at 400. Definition text: task v2 freeze rubric if present, else the slate rubric.
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research")
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer import config as cfgmod

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
LOCAL_MID = ["llama1b", "llama3b", "llama8b", "llama70b", "qwen25-3b", "qwen25-7b",
             "qwen25-14b", "qwen25-72b", "mistral7b", "phi4", "gemma2-27b"]
TASKS = ["creative_writing", "peer_review", "math", "news_homepages"]

slate_all = json.load(open(f"{OM}/zxa_slate_v1.json"))
slate_all += json.load(open(f"{OM}/news_slate_v1.json"))

for task in TASKS:
    slate = [m for m in slate_all if m["task"] == task]
    if not slate:
        print(f"{task}: no slate bases, skip")
        continue
    v1 = json.load(open(f"{OM}/freeze_zxa_{task}_v1.json"))
    doss_section = {}
    for e in v1["metrics"]:
        if e["zxa"]["arm"] == "dossier":
            mm_ = re.search(r"CONTRAST EXEMPLARS[:\s]*(.*?)(?=BOUNDARY CASES)", e["rubric"], re.S)
            if mm_:
                doss_section[e["zxa"]["base"]] = mm_.group(1).strip()
    v2rub = {}
    v2f = f"{OM}/freeze_{task}_v2.json"
    if os.path.exists(v2f):
        v2rub = {m["name"]: m["rubric"] for m in json.load(open(v2f))["metrics"]}
    for m in slate:
        v2rub.setdefault(m["name"], m.get("rubric") or m["name"])

    panels, names_ref, used = {}, None, []
    for ex in LOCAL_MID:
        p = f"{OM}/mbar2_{task}_{ex}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        nm = [str(n) for n in z["names"]]
        if names_ref is None:
            names_ref = nm
        if nm != names_ref:
            continue
        panels[ex] = z["m_bar"]
        used.append(ex)
    if len(used) < 7:
        print(f"{task}: only {len(used)} crowd panels, skip corpus arms")
    row = {n: i for i, n in enumerate(names_ref or [])}

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), task.replace("_", "-"))
    texts, _ = _load_texts(task.replace("_", "-"), 360, cfg)
    probes = texts[60:360]
    n_items = panels[used[0]].shape[1] if used else 300
    probes = probes[:n_items]
    med = int(np.median([len(t) for t in probes]))
    long_task = med > 800
    n_pos = n_neg = 3 if long_task else 4
    cap = 500 if long_task else 400

    def trunc(t):
        t = t.strip()
        return t if len(t) <= cap else t[:cap].rsplit(" ", 1)[0] + " ..."

    def pick(base):
        if base not in row or len(used) < 7:
            return None
        cons = np.stack([(panels[ex][row[base]] > 0.5).astype(float) for ex in used]).mean(0)
        ok = [j for j, t in enumerate(probes) if t.strip()]
        pos = sorted(ok, key=lambda j: -cons[j])[:n_pos]
        neg = sorted(ok, key=lambda j: cons[j])[:n_neg]
        if cons[pos[-1]] < 0.7 or cons[neg[-1]] > 0.3:
            return None
        return pos, neg

    def block(pos, neg):
        p = "\n".join("- " + trunc(probes[j]) for j in pos)
        n = "\n".join("- " + trunc(probes[j]) for j in neg)
        return ("\nExamples that satisfy this criterion:\n%s\n"
                "Examples that do NOT satisfy it:\n%s" % (p, n))

    chosen, skipped = {}, []
    for m in slate:
        got = pick(m["name"])
        (chosen.__setitem__(m["name"], got) if got else skipped.append(m["name"]))
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
        if base in chosen:
            pos, neg = chosen[base]
            blk = block(pos, neg)
            mmb = partner[base]
            mp, mn = chosen[mmb]
            for arm, rub, msrc, idx in (
                    ("exemplars", base + blk, None, sorted(pos + neg)),
                    ("def_exemplars", v2rub[base] + blk, None, sorted(pos + neg)),
                    ("exemplars_mm", base + block(mp, mn), mmb, sorted(mp + mn))):
                entries.append({"name": f"{base}||{arm}", "kind": f"{m['class']}|{arm}",
                                "rubric": rub, "criteria": [],
                                "zxa": {"base": base, "arm": arm, "class": m["class"],
                                        "exemplar_idx": idx, "mismatch_src": msrc}})
    all_by_cls = {}
    for m in slate:
        if m["name"] in doss_section:
            all_by_cls.setdefault(m["class"], []).append(m["name"])
    apartner = {}
    for cls, ns in all_by_cls.items():
        for a_, b_ in zip(ns, ns[1:] + ns[:1]):
            apartner[a_] = b_
    for m in slate:
        base = m["name"]
        if base not in doss_section:
            continue
        mmb = apartner[base]
        for arm, rub, msrc in (
                ("exemplars_authored", base + "\nContrast examples:\n" + doss_section[base], None),
                ("exemplars_authored_mm", base + "\nContrast examples:\n" + doss_section[mmb], mmb)):
            entries.append({"name": f"{base}||{arm}", "kind": f"{m['class']}|{arm}",
                            "rubric": rub, "criteria": [],
                            "zxa": {"base": base, "arm": arm, "class": m["class"],
                                    "exemplar_idx": [], "mismatch_src": msrc}})

    meta = dict(task=task, exemplar_freeze="v1-alltasks", crowd=used, n_pos=n_pos, n_neg=n_neg,
                len_cap=cap, long_task=long_task, skipped_bases=skipped,
                note="mask zxa.exemplar_idx at fit time")
    json.dump({"meta": meta, "metrics": entries},
              open(f"{OM}/freeze_zxa_ex_{task}_v1.json", "w"), indent=1)
    print(f"{task}: slate {len(slate)}, corpus-bases {len(chosen)}, skipped {len(skipped)}, "
          f"entries {len(entries)}, crowd {len(used)}, long={long_task}")
