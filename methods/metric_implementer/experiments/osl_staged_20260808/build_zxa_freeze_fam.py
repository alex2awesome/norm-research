#!/usr/bin/env python
"""Assemble freeze_zxa_<task>_fam_v1.json — same-family decompression arms.

Arms (per slate metric): expl_fam_llama / doss_fam_llama / expl_fam_qwen / doss_fam_qwen,
authored by llama70b / qwen25-72b (author_fam_arms.py), gates identical to v1
(build_zxa_freeze.py): explanation 130-180w, dossier 360-450w + 4 section labels in order,
planted rule verbatim in both texts. Entry format identical to v1 ("<base>||<arm>",
kind "<class>|<arm>") so osl_sweep --mbar-only runs unchanged. y_ref for fits stays the
v1 frontier dossier consensus (same probes: meta copied from the task v2 freeze).
"""
import hashlib, json, sys
from collections import defaultdict

import numpy as np

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
LABELS = ["DEFINITION", "WHAT COUNTS", "CONTRAST EXEMPLARS", "BOUNDARY CASES"]
AUTHORS = {"llama": "llama70b", "qwen": "qwen25-72b"}
TASK_FILE = {"humor": "freeze_humor_v2.json", "creative_writing": "freeze_creative_writing_v2.json",
             "peer_review": "freeze_peer_review_v2.json", "math": "freeze_math_v2.json"}


def wc(s):
    return len(s.split())


def main():
    slate = json.load(open(f"{OM}/zxa_slate_v1.json"))
    authored = {}
    for fam, author in AUTHORS.items():
        for r in json.load(open(f"{OM}/zxa_authoring_fam/{author}.json")):
            authored[(fam, r["task"], r["name"])] = r
    errs = []
    bad_bases = set()
    by_task = defaultdict(list)
    for m in slate:
        by_task[m["task"]].append(m)
        for fam in AUTHORS:
            n_err0 = len(errs)
            a = authored.get((fam, m["task"], m["name"]))
            if a is None or not a.get("valid", False):
                errs.append(f"MISSING/invalid {fam} authored: {m['task']} / {m['name'][:50]}")
                bad_bases.add((m["task"], m["name"]))
                continue
            we, wd = wc(a["explanation"]), wc(a["dossier"])
            if not (130 <= we <= 180):
                errs.append(f"{fam} explanation {we}w out of window: {m['name'][:50]}")
            if not (360 <= wd <= 450):
                errs.append(f"{fam} dossier {wd}w out of window: {m['name'][:50]}")
            pos = [a["dossier"].find(L) for L in LABELS]
            if any(p < 0 for p in pos) or pos != sorted(pos):
                errs.append(f"{fam} dossier labels bad: {m['name'][:50]}")
            if m["class"] == "PLANTED":
                rule = m["rubric"].strip()
                for fld in ("explanation", "dossier"):
                    if rule not in a[fld]:
                        errs.append(f"{fam} planted rule NOT verbatim in {fld}: {m['name'][:50]}")
            if len(errs) > n_err0:
                bad_bases.add((m["task"], m["name"]))
    if errs:
        print(f"WARN {len(errs)} gate failures -> dropping {len(bad_bases)} bases "
              f"(2x2 kept only where BOTH authors fully valid):")
        for e in errs[:40]:
            print(" -", e)
    kept = {t: [m for m in ms if (t, m["name"]) not in bad_bases]
            for t, ms in by_task.items()}
    for t in kept:
        print(f"[fam-freeze] {t}: kept {len(kept[t])}/{len(by_task[t])} bases")
        if len(kept[t]) < 8:  # slates hold 10-41 bases/task; 8 = min for a usable 2x2
            print(f"FATAL: {t} has <8 bases with both authors valid; fix authoring first")
            sys.exit(1)
    by_task = kept

    for task, ms in by_task.items():
        meta = json.load(open(f"{OM}/{TASK_FILE[task]}"))["meta"]
        entries = []
        for m in ms:
            for fam, author in AUTHORS.items():
                a = authored[(fam, task, m["name"])]
                arm_text = {f"expl_fam_{fam}": f"{m['name']}: {a['explanation']}",
                            f"doss_fam_{fam}": f"{m['name']}:\n{a['dossier']}"}
                for arm, txt in arm_text.items():
                    entries.append({"name": f"{m['name']}||{arm}", "kind": f"{m['class']}|{arm}",
                                    "rubric": txt, "criteria": [],
                                    "zxa": {"base": m["name"], "arm": arm, "class": m["class"],
                                            "n_words": wc(txt), "author": author,
                                            "mismatch_src": None}})
        out = {"meta": {**meta, "zxa": {"slate": "zxa_slate_v1.json",
                                        "arms": sorted({e['zxa']['arm'] for e in entries}),
                                        "authors": AUTHORS,
                                        "note": "same-family decompression arms; run with "
                                                "--n-forms 1; y_ref = v1 dossier consensus"}},
               "metrics": entries}
        path = f"{OM}/freeze_zxa_{task}_fam_v1.json"
        blob = json.dumps(out, indent=1)
        open(path, "w").write(blob)
        sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
        nw = [e["zxa"]["n_words"] for e in entries]
        print(f"{task}: {len(ms)} metrics x 4 fam arms = {len(entries)} entries; "
              f"arm-words min/med/max {min(nw)}/{int(np.median(nw))}/{max(nw)}; sha {sha}")


if __name__ == "__main__":
    main()
