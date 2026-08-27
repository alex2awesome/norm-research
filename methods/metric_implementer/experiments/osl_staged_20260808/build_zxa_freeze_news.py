#!/usr/bin/env python
"""Assemble freeze_zxa_news_homepages_v1.json (journalism z×a expansion, 2026-07-09).

Same construction + gates as the original build_zxa_freeze.py: 6 arms per slate metric
(name / definition / explanation / dossier / dossier_mismatched / definition_padded),
derangement within task (real->real, planted->planted, no fixed points), inert FILLER
padding trimmed to each metric's dossier word count. Inputs: news_slate_v1.json +
news_authoring/agent_{1..4}.json (Sonnet-authored, self-validated). Probes: the curated
pool news_probes.jsonl (junk-filtered, English-gated) — recorded in meta so every run and
readout sets OSL_PROBES_FILE accordingly.
"""
import hashlib, json, sys
from collections import defaultdict

import numpy as np

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
ARMS = ["name", "definition", "explanation", "dossier", "dossier_mismatched", "definition_padded"]
LABELS = ["DEFINITION", "WHAT COUNTS", "CONTRAST EXEMPLARS", "BOUNDARY CASES"]
FILLER = (
    "Before forming a judgment, read the entire text once without stopping, then a second time "
    "at a slower pace. Base your assessment only on what is actually present in the text rather "
    "than on what a similar text might typically contain. Give the same attention to the opening, "
    "the middle sections, and the ending, since important material can appear anywhere. Do not "
    "let the length of the text, its formatting, or its topic influence your judgment in either "
    "direction. If parts of the text are ambiguous, consider the most reasonable reading of the "
    "passage in context before deciding. Avoid being swayed by surface polish or by minor errors "
    "that are unrelated to the question at hand. Treat every text as coming from an unknown "
    "author, and do not attempt to guess at the source, venue, or intended audience beyond what "
    "the text itself states. Your judgment should concern this single text on its own terms, not "
    "a comparison with other texts you may have seen. Take as much care with a short text as "
    "with a long one, and re-read any section you found difficult before finalizing. When you "
    "have weighed the relevant considerations, commit to a single overall judgment and answer "
    "in the required format without hedging. Apply the same standard consistently from one text "
    "to the next, and be neither systematically lenient nor systematically strict. Remember that "
    "a careful, consistent reading is more informative than a fast impression, and that your "
    "answer should reflect the text as written rather than an improved or degraded version of "
    "it. Consider the text as a whole as well as its parts, since the answer may depend on "
    "material that only becomes clear once the full passage has been read. Do not defer to how "
    "often a property tends to occur in general; attend to whether it occurs here. If you remain "
    "genuinely uncertain after careful consideration, choose the answer that is better supported "
    "by the specific wording of the text itself."
).split()


def wc(s):
    return len(s.split())


def main():
    slate = json.load(open(f"{OM}/news_slate_v1.json"))
    authored = {}
    for i in (1, 2, 3, 4):
        for r in json.load(open(f"{OM}/news_authoring/agent_{i}.json")):
            authored[(r["task"], r["name"])] = r
    errs, warns = [], []
    for m in slate:
        if "||" in m["name"]:
            errs.append(f"'||' in metric name: {m['name']}")
        a = authored.get((m["task"], m["name"]))
        if a is None:
            errs.append(f"MISSING authored: {m['name']}"); continue
        we, wd = wc(a["explanation"]), wc(a["dossier"])
        if not (130 <= we <= 180):
            errs.append(f"explanation {we}w out of [130,180]: {m['name'][:50]}")
        if not (360 <= wd <= 450):
            errs.append(f"dossier {wd}w out of [360,450]: {m['name'][:50]}")
        pos = [a["dossier"].find(L) for L in LABELS]
        if any(p < 0 for p in pos) or pos != sorted(pos):
            errs.append(f"dossier section labels bad: {m['name'][:50]}")
        if m["class"] == "PLANTED":
            rule = m["rubric"].strip()
            for fld in ("explanation", "dossier"):
                if rule not in a[fld]:
                    errs.append(f"planted rule NOT verbatim in {fld}: {m['name'][:50]}")
    for m in slate:
        a = authored.get((m["task"], m["name"]))
        if not a:
            continue
        for other in slate:
            if other["name"] != m["name"] and len(other["name"]) >= 12 \
                    and other["name"].lower() in a["dossier"].lower():
                warns.append(f"name leak: '{other['name'][:40]}' in dossier of '{m['name'][:40]}'")
    if errs:
        print(f"VALIDATION FAILED ({len(errs)}):")
        for e in errs[:40]:
            print(" -", e)
        sys.exit(1)
    for w in warns:
        print("WARN:", w)

    rng = np.random.default_rng(20260709)

    def derange(group):
        idx = np.arange(len(group))
        if len(group) < 2:
            return {group[0]["name"]: group[0]["name"]} if group else {}
        while True:
            p = rng.permutation(idx)
            if not np.any(p == idx):
                break
        return {group[i]["name"]: group[p[i]]["name"] for i in idx}

    real = [m for m in slate if m["class"] != "PLANTED"]
    planted = [m for m in slate if m["class"] == "PLANTED"]
    swap = {**derange(real), **derange(planted)}
    probes = [json.loads(l)["text"] for l in open(f"{OM}/news_probes.jsonl")][60:360]
    k_med = int(np.median([len(t[:4000].split()) for t in probes]))
    meta = {"k_criteria": len(slate), "k_med_words": k_med, "n_probes": 300,
            "probe_window": "60:360", "task": "news-homepages",
            "probes_file": f"{OM}/news_probes.jsonl",
            "src": "freeze_news_homepages_v2.json + bounded_audit + curated probe pool"}
    entries = []
    for m in slate:
        a = authored[(m["task"], m["name"])]
        other = authored[(m["task"], swap[m["name"]])]
        pad_n = max(0, wc(a["dossier"]) - wc(m["rubric"]))
        filler = " ".join((FILLER * (pad_n // len(FILLER) + 1))[:pad_n])
        arm_text = {
            "name": m["name"],
            "definition": m["rubric"],
            "explanation": f"{m['name']}: {a['explanation']}",
            "dossier": f"{m['name']}:\n{a['dossier']}",
            "dossier_mismatched": f"{m['name']}:\n{other['dossier']}",
            "definition_padded": f"{m['rubric']}\n\n{filler}",
        }
        for arm in ARMS:
            entries.append({"name": f"{m['name']}||{arm}", "kind": f"{m['class']}|{arm}",
                            "rubric": arm_text[arm], "criteria": [],
                            "zxa": {"base": m["name"], "arm": arm, "class": m["class"],
                                    "n_words": wc(arm_text[arm]),
                                    "mismatch_src": swap[m["name"]] if arm == "dossier_mismatched" else None}})
    out = {"meta": {**meta, "zxa": {"slate": "news_slate_v1.json", "arms": ARMS,
                                    "derangement_seed": 20260709,
                                    "note": "journalism expansion; run with --n-forms 1 AND "
                                            "OSL_PROBES_FILE=news_probes.jsonl"}},
           "metrics": entries}
    path = f"{OM}/freeze_zxa_news_homepages_v1.json"
    blob = json.dumps(out, indent=1, ensure_ascii=False)
    open(path, "w").write(blob)
    sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    nw = [e["zxa"]["n_words"] for e in entries]
    print(f"news_homepages: {len(slate)} metrics x {len(ARMS)} arms = {len(entries)} entries; "
          f"arm-words min/med/max {min(nw)}/{int(np.median(nw))}/{max(nw)}; sha {sha}")
    by_base = defaultdict(dict)
    for e in entries:
        by_base[e["zxa"]["base"]][e["zxa"]["arm"]] = e["zxa"]["n_words"]
    d = [abs(v["definition_padded"] - v["dossier"]) for v in by_base.values()]
    print(f"  |padded - dossier| words: med {int(np.median(d))} max {max(d)}")


if __name__ == "__main__":
    main()
