"""EXP-EAP-1 batch builder (prereg a545e0c3f). Runs on sk3.
Contested-5 metrics; all-y1 (cap 40) + matched y0 by stable md5; k=8 leave-one-out
anchors; ~15% distant-metric negative controls. Arbiter input carries NO definitions,
NO reconstructions, NO metric names. Sealed key stays on sk3."""
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

MD = Path("/lfs/skampere3/0/alexspan/mention_auc")
OUT = MD / "eap_v1"
(OUT / "batches").mkdir(parents=True, exist_ok=True)

METRICS = [("humor", "a20"), ("humor", "a263"), ("humor", "a99"),
           ("peer", "a49"), ("peer", "a50")]
TEXTF = {"humor": ("humor_score_texts.jsonl", "source_id"),
         "peer": ("peer_paper_texts.jsonl", "paper_id")}
YFILE = {"humor": "humor_ypos.json", "peer": "peer_y_pos.json"}
H = lambda s: hashlib.md5(s.encode()).hexdigest()


def ymap(task):
    raw = json.load(open(MD / YFILE[task]))
    k = next(iter(raw))
    m2d = defaultdict(set)
    if re.fullmatch(r"a\d+", k):
        for m, docs in raw.items():
            m2d[m] = set(docs)
    else:
        for d, ms in raw.items():
            for m in ms:
                m2d[m].add(d)
    return m2d


texts, corpus = {}, {}
for task in ("humor", "peer"):
    tf, key = TEXTF[task]
    texts[task] = {}
    for line in open(MD / tf):
        r = json.loads(line)
        texts[task][r[key]] = r["text"]
    corpus[task] = json.load(open(MD / f"mo_{task}_corpus8b.json"))["post_ids"]

items, key_rows = [], []
for task, m in METRICS:
    m2d = ymap(task)
    cset = [d for d in corpus[task] if d in texts[task]]
    pos = sorted([d for d in cset if d in m2d[m]], key=lambda d: H(f"eap:{m}:{d}"))[:40]
    y0 = [d for d in cset if d not in m2d[m]]
    neg = sorted(y0, key=lambda d: H(f"eap:{m}:{d}"))[:len(pos)]
    # distant same-task metric: >=15 corpus positives, minimal overlap with m
    best = None
    for x, docs in m2d.items():
        if x == m:
            continue
        xp = docs & set(cset)
        if len(xp) < 15:
            continue
        ov = len(xp & set(pos)) / max(1, len(xp))
        if best is None or ov < best[0]:
            best = (ov, x, xp)
    nctl_n = math.ceil(0.15 * (len(pos) + len(neg)))
    nctl = sorted([d for d in best[2] if d not in m2d[m]],
                  key=lambda d: H(f"eapctl:{m}:{d}"))[:nctl_n]
    print(f"{task}/{m}: pos {len(pos)} neg {len(neg)} negctl {len(nctl)} "
          f"(distant={best[1]} overlap={best[0]:.2f})")
    for d, role in ([(d, "pos") for d in pos] + [(d, "neg") for d in neg]
                    + [(d, "negctl") for d in nctl]):
        pool = [a for a in pos if a != d]
        anch = sorted(pool, key=lambda a: H(f"eapanch:{m}:{d}:{a}"))[:8]
        if len(anch) < 8:
            continue
        iid = f"eap_{m}_{H(f'{task}:{m}:{d}')[:10]}"
        items.append({"item_id": iid,
                      "anchors": [texts[task][a][:600] for a in anch],
                      "document": texts[task][d][:6000]})
        key_rows.append({"item_id": iid, "task": task, "metric": m, "doc": d,
                         "role": role})

items.sort(key=lambda it: H("eapshuf:" + it["item_id"]))
for i in range(0, len(items), 6):
    json.dump(items[i:i + 6], open(OUT / "batches" / f"batch_{i // 6:03d}.json", "w"))
json.dump(key_rows, open(OUT / "eap_key_SEALED.json", "w"), indent=0)
print(f"TOTAL items {len(items)} batches {math.ceil(len(items) / 6)} -> {OUT}")
