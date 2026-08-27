"""EXP-MP-1c S1 (prereg 74d99a18c). Runs on sk3. 17 peer metrics, 12-candidate family,
labels floor>=5, critic prompts for the 3 new metrics."""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

MD = Path("/lfs/skampere3/0/alexspan/mention_auc")
BANK = "/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful/banks/peer-review.json"
HIER = "/lfs/skampere3/0/alexspan/norm-research/outputs/hierarchy/peer-review_general_r2_expanded.json"
H = lambda s: hashlib.md5(s.encode()).hexdigest()
_norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
TARGETS = ["a47", "a11", "a65", "a18", "a77", "a34", "a6", "a68", "a10", "a48",
           "a53", "a1", "a73", "a55", "a66", "a41", "a80"]

bank = json.load(open(BANK))["metrics"]
groups = json.load(open(HIER))["merged_groups"]
g_by_norm = {_norm(g["merged_name"]): g for g in groups}
corpus = set(json.load(open(MD / "mo_peer_corpus8b.json"))["post_ids"])
pos_rev, negmix = defaultdict(set), set()
paper_rev_mentions = defaultdict(lambda: defaultdict(int))
paper_aid_any = defaultdict(set)
name2aid = {_norm(m["name"]): f"a{i}" for i, m in enumerate(bank)}
for line in open(MD / "mention_join_peer_20260716.jsonl"):
    r = json.loads(line)
    aid = name2aid.get(_norm(r.get("choice", "")))
    mm = re.match(r"(.+)_r(\d+)$", r["source_id"])
    if not mm:
        continue
    paper, rev = mm.group(1), int(mm.group(2))
    if paper not in corpus:
        continue
    if aid is None:
        continue
    paper_rev_mentions[paper][rev] += 1
    paper_aid_any[paper].add(aid)
    if str(r.get("polarity", "")).lower() == "pos":
        pos_rev[(aid, paper)].add(rev)
    else:
        negmix.add((aid, paper))
attentive = {p for p, revs in paper_rev_mentions.items()
             if len(revs) >= 2 and all(v >= 1 for v in revs.values())
             and sum(revs.values()) >= 5}
labels = {}
for aid in TARGETS:
    P = sorted(p for (a, p), revs in pos_rev.items()
               if a == aid and len(revs) >= 2 and (a, p) not in negmix)
    S1 = sorted(p for (a, p), revs in pos_rev.items()
                if a == aid and len(revs) == 1 and (a, p) not in negmix
                and p not in set(P))[:200]
    Nn = sorted((p for p in attentive if aid not in paper_aid_any[p]),
                key=lambda p: H(f"mp3neg:{aid}:{p}"))[:600]
    labels[aid] = {"pos": P, "single": S1, "neg": Nn}
    print(f"{aid}: pos {len(P)} single {len(S1)} neg {len(Nn)}")
json.dump(labels, open(MD / "mp5_labels.json", "w"), indent=0)

cands, man = [], []
for aid in TARGETS:
    bm = bank[int(aid[1:])]
    c0 = f"{bm['name']}: {str(bm.get('description', ''))[:500]}"
    g = g_by_norm.get(_norm(bm["name"]))
    leaves = []
    if g:
        seen = set()
        for lf in g["all_leaves"]:
            n = lf["name"].strip()
            if 10 < len(n) < 120 and _norm(n) not in seen:
                seen.add(_norm(n)); leaves.append(n)
    leaves.sort(key=lambda n: H(f"mp2leaf:{aid}:{n}"))
    L = leaves

    def ck(sub):
        return " Key checks: " + " ".join(f"({j+1}) {u}" for j, u in enumerate(sub))

    def ev(sub):
        return "Evaluate: " + " ".join(f"({j+1}) {u}" for j, u in enumerate(sub))

    fam = {0: c0,
           1: (f"{g['merged_name']}: {str(g.get('merged_description',''))[:500]}" if g else c0),
           2: c0 + ck(L[:2]) if len(L) >= 2 else c0,
           3: c0 + ck(L[:4]) if len(L) >= 4 else c0,
           4: c0 + ck(L[:8]) if len(L) >= 8 else c0,
           5: ev(L[:4]) if len(L) >= 4 else c0,
           6: ev(L[:8]) if len(L) >= 8 else c0,
           7: L[0] if L else c0,
           8: L[1] if len(L) >= 2 else c0,
           9: c0 + ck(L[4:6]) if len(L) >= 6 else c0,
           10: ev(L[4:8]) if len(L) >= 8 else c0,
           11: str(bm["name"])}
    cands.append({"metric": aid, "name": bm["name"], "n_leaves": len(L),
                  "candidates": {f"C{j}": fam[j] for j in range(12)}})
    for j in range(12):
        man.append({"metric_id": aid, "form_idx": 700 + j, "rubric": fam[j]})
json.dump(cands, open(MD / "mp5_candidates.json", "w"), indent=1)
json.dump(man, open(MD / "mp5_peer_manifest.json", "w"), indent=0)
print(f"manifest {len(man)} rubrics")

pids = json.load(open(MD / "ocl_llama8b_peer_probes.json"))["post_ids"]
ptx = {}
for line in open(MD / "peer_probe_texts.jsonl"):
    r = json.loads(line)
    ptx[r["probe_id"]] = r["text"]
rows = []
for aid in ("a66", "a41", "a80"):
    bm = bank[int(aid[1:])]
    c0 = f"{bm['name']}: {str(bm.get('description', ''))[:500]}"
    for d in sorted(pids, key=lambda d: H(f"mp2crit:{aid}:{d}"))[:150]:
        rows.append({"aspect_id": aid, "datapoint_id": d,
                     "prompt": f"Criterion: {c0}\n\nText:\n{ptx[d][:5000]}\n\n"
                               f"Rate how strongly the text satisfies the criterion, "
                               f"0 (not at all) to 10 (fully). Reply with ONLY the integer."})
with open(MD / "mp5_critic_prompts.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"critic prompts (new metrics): {len(rows)}")
