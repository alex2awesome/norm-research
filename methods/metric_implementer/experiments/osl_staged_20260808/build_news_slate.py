#!/usr/bin/env python
"""news_homepages z×a slate v1 (journalism expansion, 2026-07-09).

Composition mirrors humor v1 at smaller scale:
  10 TACIT-RECRUIT   voice/judgment/identity constructs recruited by TYPE (humor's audited
                     tacit cluster analog: voice, newsworthiness intuition, proximity,
                     tabloid-boundary, curation). Recruited, NOT audit-certified — the class
                     name keeps that honest; z×a curves will classify them empirically.
   2 DIALECT-SUSPECT audited (bounded_audit.json 2026-07-08).
   4 REACHES-ANCHOR  audited REACHES-OK anchors.
   5 PLANTED         mechanical rules generated on the CURATED probe pool
                     (news_probes.jsonl: junk-filtered 38%-pass, English-gated, 360 rows).
Rubrics come from freeze_news_homepages_v2.json by exact name; audited names missing from
the v2 freeze fall back to their bounded-audit description.
"""
import json, sys

import numpy as np

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
sys.path.insert(0, f"{B}/norm-research")
from methods.metric_implementer.experiments.osl_sweep import planted_metrics

TACIT_RECRUIT = [
    "Authorial presence and distinctive point of view",
    "Tone, voice, and epistemic stance",
    "News values and newsworthiness criteria",
    "Audience identification/proximity as a news value",
    "Localism and community orientation",
    "Avoid sensationalism and clickbait",
    "Proportionality and restraint vs. sensationalism",
    "Curation breadth and composition",
    "Prominence/elite‑actor focus",
    "Conflict/tension as narrative and news value",
]
DIALECT = {
    "Communication as social construction and symbolic process": None,
    "Explain responses and offer actionable insights (Solutions)":
        "Explain responses and offer actionable insights: coverage moves beyond problem "
        "description to explain how people/institutions are responding and what readers can do.",
}
REACHES = {
    "Headline and framing integrity: accuracy, alignment, and non-clickbait": None,
    "Accountability journalism (watchdog)":
        "Accountability journalism (watchdog): coverage scrutinizes powerful actors and "
        "institutions, documents wrongdoing or failure, and holds them to account.",
    "Multimodality and multimedia presentation": None,
    "Permissible, non‑deceptive visual edits and labeling":
        "Permissible, non-deceptive visual edits and labeling: any visual alteration stays "
        "within accepted bounds (cropping/toning) and manipulations are disclosed/labeled.",
}


def main():
    frz = json.load(open(f"{OM}/freeze_news_homepages_v2.json"))
    rub = {m["name"]: m["rubric"] for m in frz["metrics"]}
    probes = [json.loads(l)["text"] for l in open(f"{OM}/news_probes.jsonl")][60:360]
    probes = [t[:4000] for t in probes]
    k_med = int(np.median([len(t.split()) for t in probes]))
    print(f"probes: {len(probes)}, k_med_words={k_med}")

    slate = []

    def add(name, klass, fallback=None):
        r = rub.get(name) or fallback
        if not r:
            print(f"  !! no rubric for: {name}")
            return
        slate.append({"task": "news_homepages", "class": klass, "name": name, "rubric": r})

    for n in TACIT_RECRUIT:
        add(n, "TACIT-RECRUIT")
    for n, fb in DIALECT.items():
        add(n, "DIALECT-SUSPECT", fb)
    for n, fb in REACHES.items():
        add(n, "REACHES-ANCHOR", fb)
    for pm in planted_metrics(probes, k_med):
        slate.append({"task": "news_homepages", "class": "PLANTED",
                      "name": pm["name"], "rubric": pm["rubric"]})
    out = f"{OM}/news_slate_v1.json"
    json.dump(slate, open(out, "w"), indent=1, ensure_ascii=False)
    from collections import Counter
    print(Counter(s["class"] for s in slate))
    print(f"wrote {out}: {len(slate)} metrics (meta k_med_words={k_med})")


if __name__ == "__main__":
    main()
