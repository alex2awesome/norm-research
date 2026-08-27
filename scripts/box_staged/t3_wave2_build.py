import hashlib
import json
import re
from collections import Counter, defaultdict

import numpy as np

MD = "/lfs/skampere3/0/alexspan/mention_auc"
HIER = "/lfs/skampere3/0/alexspan/norm-research/outputs/hierarchy"
h = lambda s: hashlib.md5(("t3v3:" + s).encode()).hexdigest()
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

HIERFILE = {"cw": "creative-writing", "pr": "press-releases", "humor": "humor",
            "crx": "code-review", "peer": "peer-review"}


BANKDIR = "/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful/banks"


def construct_of(task):
    """metric_id -> (name, definition). cw/pr/humor: hierarchy group gi=int(mid[1:]).
    crx: code-review bank index (verified against crx matches choices). peer: bank index."""
    out = {}
    if task in ("cw", "pr", "humor"):
        g = json.load(open(f"{HIER}/{HIERFILE[task]}_general_r2_expanded.json"))["merged_groups"]
        for i, x in enumerate(g):
            out[f"a{i}"] = (x.get("merged_name", f"a{i}"), x.get("merged_description", "")[:400])
    elif task == "crx":
        bank = json.load(open(f"{BANKDIR}/code-review.json"))["metrics"]
        for i, m in enumerate(bank):
            out[f"a{i}"] = (m["name"] if isinstance(m, dict) else str(m),
                            (m.get("description", "")[:400] if isinstance(m, dict) else ""))
        choices = {norm(str(r.get("choice", ""))) for r in
                   json.load(open(f"{MD}/crx_matches_consolidated.json"))} if True else set()
        hits = sum(1 for n, _ in out.values() if norm(n) in choices)
        print(f"crx bank-order check: {hits}/{len(out)} bank names appear in matches choices")
        assert hits >= len(out) * 0.5, "crx bank-order mapping FAILED verification"
    elif task == "peer":
        bank = json.load(open(f"{BANKDIR}/peer-review.json"))["metrics"]
        for i, m in enumerate(bank):
            out[f"a{i}"] = (m["name"] if isinstance(m, dict) else str(m),
                            (m.get("description", "")[:400] if isinstance(m, dict) else ""))
    return out


def load_texts(tf, key):
    out = {}
    for line in open(f"{MD}/{tf}"):
        r = json.loads(line)
        out[r[key]] = r["text"]
    return out


def best_form_scores(task, sf):
    man = json.load(open(f"{MD}/{task}_forms_manifest.json"))
    d = json.load(open(f"{MD}/{sf}"))
    best = {}
    for e in man:
        m = e["metric_id"]
        if m not in best or e.get("mi_form", 0) > best[m][1]:
            best[m] = (e["form_idx"], e.get("mi_form", 0))
    return d["post_ids"], {m: d["scores"][f"{m}__{fi}"]
                           for m, (fi, _) in best.items() if f"{m}__{fi}" in d["scores"]}


def plain_scores(sf):
    d = json.load(open(f"{MD}/{sf}"))
    return d["post_ids"], d["scores"]


def ymap_load(yf):
    raw = json.load(open(f"{MD}/{yf}"))
    k = next(iter(raw))
    out = defaultdict(set)
    if re.fullmatch(r"a\d+", k):
        for m, docs in raw.items():
            for d in docs:
                out[d].add(m)
    else:
        for d, ms in raw.items():
            out[d] = set(ms)
    return out


TASKS = {
    "crx": dict(load=lambda: plain_scores("crx_p_scores.json"), yf="crx_y_pos.json",
                tf=("crx_pr_texts.jsonl", "source_id")),
    "cw": dict(load=lambda: best_form_scores("cw", "cw_scores_g4.json"), yf="variant_ypos_cw.json",
               tf=("cw_story_texts.jsonl", "post_id")),
    "pr": dict(load=lambda: best_form_scores("pr", "pr_scores_g4.json"), yf="variant_ypos_pr.json",
               tf=("pressrel_score_texts.jsonl", "source_id")),
    "humor": dict(load=lambda: best_form_scores("humor", "humor_scores_g4.json"), yf="humor_ypos.json",
                  tf=("humor_score_texts.jsonl", "source_id")),
    "peer": dict(load=lambda: plain_scores("peer_p_scores.json"), yf="peer_y_pos.json",
                 tf=("peer_paper_texts.jsonl", "paper_id")),
}

name_of = {}
for task in TASKS:
    try:
        for e in json.load(open(f"{MD}/{task}_forms_manifest.json")):
            if e["form_idx"] == 0:
                name_of[(task, e["metric_id"])] = e["rubric"]
    except Exception:
        pass

existing = json.load(open(f"{MD}/t3_items_v2.json"))
taken = {(it.get("task", "peer"), it["metric"], it["doc"]) for it in existing}
v2items = []

# (A) re-arbitrate existing extension items with proper construct names + definitions
CON = {t: construct_of(t) for t in ("cw", "pr", "humor", "crx", "peer")}
for it in existing:
    t = it.get("task", "peer")
    if t in ("cw", "pr", "humor", "crx") and it["stratum"] != "anchor":
        nm, d = CON[t].get(it["metric"], (it["name"], ""))
        v2items.append({**it, "name": nm, "desc": d, "wave": "v2redo"})

# (B) new samples to reach ~400/task for extension tasks (with defs)
have = Counter(it.get("task", "peer") for it in existing if it["stratum"] != "anchor")
for task in ("crx", "cw", "pr", "humor"):
    need = 400 - have.get(task, 0)
    if need <= 0:
        continue
    cfg = TASKS[task]
    ids, S = cfg["load"]()
    ym = ymap_load(cfg["yf"])
    texts = load_texts(*cfg["tf"])
    stats = []
    for mid, ps in S.items():
        pa = np.asarray(ps, float)
        rows = [(ids[i], pa[i], 1 if mid in ym.get(ids[i], ()) else 0)
                for i in range(len(ids)) if np.isfinite(pa[i]) and ids[i] in texts]
        n1 = sum(r[2] for r in rows)
        if n1 >= 10:
            stats.append((n1, mid, rows))
    stats.sort(reverse=True)
    got_task = 0
    for n1, mid, rows in stats[:12]:
        if got_task >= need:
            break
        qs = np.quantile([r[1] for r in rows], [1 / 3, 2 / 3])
        strata = defaultdict(list)
        for doc, sc, yv in rows:
            tq = 0 if sc <= qs[0] else (1 if sc <= qs[1] else 2)
            strata[(yv, tq)].append(doc)
        for (yv, tq), docs in sorted(strata.items()):
            take = 3 if yv == 1 else 2
            got = 0
            for d in sorted(docs, key=lambda d: h(f"w2:{task}:{mid}:{d}")):
                if got >= take or got_task >= need:
                    break
                if (task, mid, d) in taken:
                    continue
                nm, dd_ = CON[task].get(mid, (mid, ""))
                v2items.append({"item_type": "sample", "task": task, "metric": mid, "name": nm,
                                "desc": dd_, "doc": d,
                                "stratum": f"y{yv}_q{tq}", "wave": "v2new"})
                taken.add((task, mid, d))
                got += 1
                got_task += 1

# (C) peer y=0 top-up: 500 additional mention-negatives, score-stratified
cfg = TASKS["peer"]
ids, S = cfg["load"]()
ym = ymap_load(cfg["yf"])
texts = load_texts(*cfg["tf"])
bank = json.load(open("/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful/banks/peer-review.json"))["metrics"]
pdesc = {f"a{i}": (m.get("description", "")[:400] if isinstance(m, dict) else "") for i, m in enumerate(bank)}
pname = {f"a{i}": (m["name"] if isinstance(m, dict) else str(m)) for i, m in enumerate(bank)}
stats = []
for mid, ps in S.items():
    pa = np.asarray(ps, float)
    rows = [(ids[i], pa[i]) for i in range(len(ids))
            if np.isfinite(pa[i]) and ids[i] in texts and mid not in ym.get(ids[i], ())]
    if mid in pname and len(rows) > 100:
        stats.append((mid, rows))
got = 0
per_metric = 500 // max(1, len(stats[:16])) + 1
for mid, rows in stats[:16]:
    qs = np.quantile([r[1] for r in rows], [1 / 3, 2 / 3])
    strata = defaultdict(list)
    for doc, sc in rows:
        tq = 0 if sc <= qs[0] else (1 if sc <= qs[1] else 2)
        strata[tq].append(doc)
    for tq, docs in sorted(strata.items()):
        for d in sorted(docs, key=lambda d: h(f"negtop:{mid}:{d}"))[:per_metric // 3 + 1]:
            if got >= 500:
                break
            if ("peer", mid, d) in taken:
                continue
            v2items.append({"item_type": "sample", "task": "peer", "metric": mid,
                            "name": pname[mid], "desc": pdesc[mid], "doc": d,
                            "stratum": f"y0_q{tq}", "wave": "v2negtop"})
            taken.add(("peer", mid, d))
            got += 1

# (D) fresh sealed anchors for wave-2 certification
key = json.load(open(f"{MD}/t3_anchor_key_SEALED.json"))
key = [k for k in key if not str(k.get("metric", "")).endswith("-w2")]   # rebuild-safe dedupe
MECH = [("MECH-PCT", "The text explicitly contains at least one numerical percentage (a number followed by % or the word percent)",
         lambda t: bool(re.search(r"\d+(\.\d+)?\s*(%|percent)", t))),
        ("MECH-Q", "The text contains at least one direct question ending with a question mark",
         lambda t: "?" in t)]
for task in ("peer", "cw", "pr", "crx"):
    texts = load_texts(*TASKS[task]["tf"])
    pool = sorted(texts, key=lambda d: h(f"anc2:{task}:{d}"))
    for aid_, desc, fn in MECH:
        pg = ng = 0
        for d in pool:
            tr = fn(texts[d][:12000])
            if tr and pg < 2:
                v2items.append({"item_type": "sample", "task": task, "metric": aid_ + "-w2",
                                "name": desc, "desc": "", "doc": d, "stratum": "anchor", "wave": "v2"})
                key.append({"task": task, "metric": aid_ + "-w2", "doc": d, "truth": 1})
                pg += 1
            elif not tr and ng < 2:
                v2items.append({"item_type": "sample", "task": task, "metric": aid_ + "-w2",
                                "name": desc, "desc": "", "doc": d, "stratum": "anchor", "wave": "v2"})
                key.append({"task": task, "metric": aid_ + "-w2", "doc": d, "truth": 0})
                ng += 1
            if pg >= 2 and ng >= 2:
                break

rng = np.random.default_rng(17)
v2items = [v2items[i] for i in rng.permutation(len(v2items))]
for i, it in enumerate(v2items):
    it["item_id_v1"] = it.get("item_id")
    it["item_id"] = "T3W2-%04d" % i
json.dump(v2items, open(f"{MD}/t3_items_wave2.json", "w"), indent=1)
json.dump(key, open(f"{MD}/t3_anchor_key_SEALED.json", "w"), indent=1)

alltexts = {}
for task in TASKS:
    tx = load_texts(*TASKS[task]["tf"])
    for it in v2items:
        if it["task"] == task:
            alltexts[(task, it["doc"])] = tx[it["doc"]]
import os
os.makedirs(f"{MD}/t3_batches_w2", exist_ok=True)
B = 6
nb = 0
for b in range(0, len(v2items), B):
    batch = [{"item_id": it["item_id"], "criterion": it["name"], "definition": it["desc"],
              "document": alltexts[(it["task"], it["doc"])][:12000]} for it in v2items[b:b + B]]
    json.dump(batch, open(f"{MD}/t3_batches_w2/batch_%03d.json" % (nb), "w"), indent=0)
    nb += 1
withdef = sum(1 for it in v2items if it["desc"])
print("wave-2 items:", len(v2items), dict(Counter(it["wave"] for it in v2items)))
print("per task:", dict(Counter(it["task"] for it in v2items)),
      "| with definitions: %d/%d" % (withdef, len(v2items)), "| batches:", nb)
