#!/usr/bin/env python3
"""Species clustering of a round's sealed-fleet proposal pool (both tracks).

Two-stage instrument, per the missing-mass note's PART-4 recommendation 2:
  * embedding SHORTLIST only, and only WITHIN one register (fleet proposal vs fleet
    proposal, both written by the same kind of author about the same corpus) -- this
    is exactly the case where the pilot's tau band has dynamic range (max cosine
    inside the fleet was .84-.88, well above the band, vs .72 across registers);
  * IDENTITY decided by a blind pairwise judge pass over the shortlist, with an
    authored anchor battery. Cosine never decides.

`build` writes the blind adjudication prompt; `finalize` folds two judges' verdicts
into single-linkage species and writes roundN_species_<track>.json.

Usage:
  python fleet_species.py build    --round 1 --track a
  python fleet_species.py finalize --round 1 --track a
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "fleet_embed_cache.npz"
SHORTLIST_TAU = 0.78
TOP_K = 6

ANCHORS = [
    {"label": "SAME",
     "X": {"name": "Quantifies the rule's cost using the commenter's own data",
           "instruction": "Score 10 when the comment supplies its own numeric estimate of "
                          "compliance cost with the basis stated; 0 when no numbers appear."},
     "Y": {"name": "Supplies own dollar estimates of compliance burden",
           "instruction": "Score 10 when the submission puts forward concrete monetary burden "
                          "figures it derived itself and says where they come from; 0 when it "
                          "offers no figures."}},
    {"label": "SAME",
     "X": {"name": "Names a specific implementable alternative",
           "instruction": "Score 10 when the comment sets out a concrete different design the "
                          "agency could adopt, specific enough to write into the rule; 0 when it "
                          "only objects."},
     "Y": {"name": "Offers an actionable substitute regulatory design",
           "instruction": "Score 10 when an alternative approach is described concretely enough "
                          "to be implemented; 0 when the comment proposes nothing."}},
    {"label": "DIFFERENT",
     "X": {"name": "Cites CFR provisions by part and section",
           "instruction": "Score 10 for explicit part/section CFR citations identifying the "
                          "provisions addressed; 0 for none."},
     "Y": {"name": "Argues the rule exceeds statutory authority",
           "instruction": "Score 10 for a developed legal argument that the statute does not "
                          "authorise the proposal; 0 when no authority argument is made."}},
    {"label": "DIFFERENT",
     "X": {"name": "Reports first-hand operational experience",
           "instruction": "Score 10 when the commenter describes what they themselves do in the "
                          "regulated activity and how the rule would change it; 0 when no "
                          "first-hand practice is described."},
     "Y": {"name": "Expresses intense personal feeling about the proposal",
           "instruction": "Score 10 for strongly-worded approval or outrage; 0 for flat neutral "
                          "register."}},
]

INSTRUCTION = """You are auditing a pool of candidate scoring criteria written by several
independent analysts for judging PUBLIC COMMENTS submitted to United States federal
agencies on proposed rules.

For each PAIR below, decide ONE question:

  Would an independent judge, scoring public comments against criterion X and against
  criterion Y, be measuring THE SAME UNDERLYING CONCEPT?

Answer SAME only if the two would produce essentially interchangeable scores because
they name one concept in different words. Answer DIFFERENT if a comment could plausibly
score high on one and low on the other -- including when the two are closely related,
overlapping, or members of the same family. Related is not the same.

Emit exactly one JSON object and nothing else:

{"verdicts": [
  {"pair_id": "FP001", "verdict": "SAME" or "DIFFERENT", "confidence": "high"/"medium"/"low",
   "reason": "<one sentence>"},
  ... one entry for EVERY pair ...
]}
"""


def embed(texts):
    import torch
    from transformers import AutoModel, AutoTokenizer

    cached = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cached = {str(k): v for k, v in zip(z["keys"], z["vecs"])}
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    need = [t for t, k in zip(texts, keys) if k not in cached]
    if need:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        mod = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(dev).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(need), 32):
                b = tok(need[i:i + 32], padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(dev)
                h = mod(**b).last_hidden_state[:, 0]
                h = torch.nn.functional.normalize(h, dim=-1)
                out.append(h.cpu().numpy())
        out = np.vstack(out)
        for t, v in zip(need, out):
            cached[hashlib.sha1(t.encode()).hexdigest()] = v
        np.savez_compressed(CACHE, keys=np.array(list(cached), dtype=object),
                            vecs=np.array(list(cached.values())))
    return np.array([cached[k] for k in keys])


def pool_for(r, track, pool_file=None):
    d = json.loads((HERE / (pool_file or f"fleet_r{r}.json")).read_text())
    return [p for p in d["proposals"] if p["track"] == track]


def cmd_build(a):
    pool = pool_for(a.round, a.track, getattr(a, "pool_file", None))
    texts = [f"{p['name']}. {p['instruction']}" for p in pool]
    E = embed(texts)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    m = len(pool)
    cand = set()
    for i in range(m):
        for j in np.argsort(-S[i])[:TOP_K]:
            if j > i and S[i, j] >= SHORTLIST_TAU:
                cand.add((i, int(j), round(float(S[i, j]), 4)))
    cand = sorted(cand, key=lambda t: -t[2])

    items = []
    for n, (i, j, s) in enumerate(cand):
        flip = int(hashlib.sha256(f"fp|{a.round}|{a.track}|{i}|{j}".encode()).hexdigest(), 16) % 2
        x, yq = (i, j) if not flip else (j, i)
        items.append({"kind": "real", "i": i, "j": j, "cos": s,
                      "X": {"name": pool[x]["name"], "instruction": pool[x]["instruction"]},
                      "Y": {"name": pool[yq]["name"], "instruction": pool[yq]["instruction"]}})
    for n, anc in enumerate(ANCHORS):
        flip = int(hashlib.sha256(f"fanchor|{a.round}|{a.track}|{n}".encode()).hexdigest(), 16) % 2
        X, Y = (anc["X"], anc["Y"]) if not flip else (anc["Y"], anc["X"])
        items.append({"kind": "anchor", "anchor_label": anc["label"], "X": X, "Y": Y})

    items.sort(key=lambda p: hashlib.sha256(
        f"fs|{a.round}|{a.track}|{p['X']['name']}|{p['Y']['name']}".encode()).hexdigest())
    for k, it in enumerate(items):
        it["shown_id"] = f"FP{k + 1:03d}"

    body = INSTRUCTION + "\n\n" + "\n\n".join(
        f"--- PAIR {it['shown_id']} ---\n"
        f"X NAME: {it['X']['name']}\nX INSTRUCTION: {it['X']['instruction']}\n"
        f"Y NAME: {it['Y']['name']}\nY INSTRUCTION: {it['Y']['instruction']}"
        for it in items) + "\n"

    (HERE / f"round{a.round}_species_{a.track}{a.suffix}_prompt.txt").write_text(body)
    (HERE / f"round{a.round}_species_{a.track}{a.suffix}_key.json").write_text(json.dumps(
        {"n_pool": m, "shortlist_tau": SHORTLIST_TAU,
         "cos_max_offdiag": float(S.max()),
         "items": [{k: v for k, v in it.items() if k not in ("X", "Y")} for it in items]}, indent=1))
    print(f"r{a.round}/{a.track}: pool={m} shortlist={len(cand)} (+{len(ANCHORS)} anchors) "
          f"max_cos={S.max():.3f} -> round{a.round}_species_{a.track}_prompt.txt")


def cmd_finalize(a):
    pool = pool_for(a.round, a.track, getattr(a, "pool_file", None))
    m = len(pool)
    key = json.loads((HERE / f"round{a.round}_species_{a.track}{a.suffix}_key.json").read_text())
    items = {it["shown_id"]: it for it in key["items"]}

    def load(p):
        d = json.loads((HERE / p).read_text())
        return {v["pair_id"]: v for v in d["verdicts"]}

    j1 = load(f"round{a.round}_species_{a.track}{a.suffix}_judge1.json")
    j2 = load(f"round{a.round}_species_{a.track}{a.suffix}_judge2.json")

    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    anchors, agree, n_real = [], 0, 0
    edges = []
    for sid, it in items.items():
        v1, v2 = j1.get(sid), j2.get(sid)
        if not v1 or not v2:
            continue
        same = v1["verdict"] == "SAME" and v2["verdict"] == "SAME"
        if it["kind"] == "anchor":
            anchors.append({"shown_id": sid, "label": it["anchor_label"],
                            "j1": v1["verdict"], "j2": v2["verdict"],
                            "j1_correct": v1["verdict"] == it["anchor_label"],
                            "j2_correct": v2["verdict"] == it["anchor_label"]})
            continue
        n_real += 1
        agree += int(v1["verdict"] == v2["verdict"])
        if same:
            edges.append({"i": it["i"], "j": it["j"], "cos": it["cos"]})
            ra, rb = find(it["i"]), find(it["j"])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    clus = defaultdict(list)
    for i in range(m):
        clus[find(i)].append(i)

    species = []
    for root, members in sorted(clus.items()):
        props = [pool[i] for i in members]
        proposers = sorted({p["proposer"] for p in props})
        families = sorted({p["family"] for p in props})
        # REPRESENTATIVE: stable-hash pick among members (never a judgement call)
        rep_i = min(members, key=lambda i: hashlib.sha256(
            f"rep|{a.round}|{a.track}|{pool[i]['pid']}".encode()).hexdigest())
        species.append({
            "species_id": f"S{len(species) + 1:03d}",
            "members": [pool[i]["pid"] for i in members],
            "n_members": len(members),
            "n_proposers": len(proposers),
            "proposers": proposers,
            "families": families,
            "n_families": len(families),
            "rep_pid": pool[rep_i]["pid"],
            "name": pool[rep_i]["name"],
            "instruction": pool[rep_i]["instruction"],
            "rationale": pool[rep_i]["rationale"],
            "upstream_parent": pool[rep_i].get("upstream_parent"),
            "mixed": pool[rep_i].get("mixed"),
            "all_names": [pool[i]["name"] for i in members],
        })

    out = {
        "round": a.round, "track": a.track,
        "n_proposals": m, "n_species": len(species),
        "n_shortlist_pairs": n_real, "n_merge_edges": len(edges),
        "judge_agreement": agree / max(1, n_real),
        "anchor_battery": {"detail": anchors,
                           "j1_score": f"{sum(x['j1_correct'] for x in anchors)}/{len(anchors)}",
                           "j2_score": f"{sum(x['j2_correct'] for x in anchors)}/{len(anchors)}"},
        "cos_max_offdiag": key["cos_max_offdiag"],
        "species": species,
    }
    (HERE / f"round{a.round}_species_{a.track}{a.suffix}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("species", "anchor_battery")}, indent=1))
    print("anchors", out["anchor_battery"]["j1_score"], out["anchor_battery"]["j2_score"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "finalize"):
        s = sub.add_parser(c)
        s.add_argument("--round", type=int, required=True)
        s.add_argument("--track", required=True, choices=["a", "b"])
        s.add_argument("--pool-file", dest="pool_file", default=None)
        s.add_argument("--suffix", default="")
    a = ap.parse_args()
    {"build": cmd_build, "finalize": cmd_finalize}[a.cmd](a)
