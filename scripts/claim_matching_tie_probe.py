#!/usr/bin/env python3
"""Tie-stratum localization probe: for claims where NEITHER fetched span shows disclosure (the 61%
"tie" stratum of the GLM-judged subset) — does disclosing text exist ELSEWHERE in the gold reference
document? Separates "retrieval/localization fetched the wrong passage" (fixable, upstream) from
"reference genuinely doesn't disclose" (label noise). Also runs the 20-claim DISAGREE stratum
(filler>gold): if a better gold passage exists elsewhere, the gold DOC was right and the SPAN was wrong.

Method: gold docs' full detailed descriptions (cand_doc_detaildesc.jsonl, 111/111 coverage) ->
paragraph-pack -> bge-m3 (CPU, offline snapshot) ranks paragraphs vs the claim element -> top-3
non-fetched paragraphs per claim -> GLM strict disclosure verdict (+ blinded pos/neg anchors per the
anchor-test discipline). ~350 GLM calls total.

  python scripts/claim_matching_tie_probe.py
"""
import json, re, time, hashlib, collections, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

BASE = "/lfs/skampere3/0/alexspan/norm-research"
TARGETS = f"{BASE}/outputs/claim_matching/tie_probe_targets.json"
TESTBED = f"{BASE}/datasets/claim-matching/testbed/pair_testbed.jsonl"
DETAIL = f"{BASE}/datasets/patents/processed/cand_doc_detaildesc.jsonl"
GLMH = f"{BASE}/outputs/claim_matching/scores_glm_holistic.jsonl"
BGE = "/lfs/skampere3/0/shared_hf_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
OUTDIR = f"{BASE}/outputs/claim_matching"
KEY_FILE = "/lfs/skampere3/0/alexspan/.z-ai-api-key-alexander-spangher.txt"
KEY_FILE2 = "/lfs/skampere3/0/alexspan/.z-ai-api-key.txt"
WORD = re.compile(r"[a-z]{3,}")

SYS = ("You are a strict patent-disclosure judge. Decide whether the PASSAGE discloses the CLAIM "
       "ELEMENT in substance. Surface word overlap is NOT disclosure; the passage must actually "
       "describe the claimed feature or its clear equivalent.")
_BOOL = re.compile(r'"discloses"\s*:\s*(true|false)', re.I)


def toks(s):
    return set(WORD.findall((s or "").lower()))


def glm(user, model="glm-4.7"):
    body = json.dumps({"model": model, "max_tokens": 60, "temperature": 0.0,
                       "system": SYS, "messages": [{"role": "user", "content": user}]}).encode()
    for att in range(8):
        try:
            kf = [KEY_FILE, KEY_FILE2][att % 2]
            key = open(kf).read().strip()
            req = urllib.request.Request("https://api.z.ai/api/anthropic/v1/messages", data=body,
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                o = json.loads(r.read())
            return "".join(b.get("text", "") for b in o.get("content", []))
        except Exception:
            time.sleep(min(40, 4 * (att + 1)))
    return ""


def parse_bool(raw):
    m = _BOOL.search(raw or "")
    return None if not m else m.group(1).lower() == "true"


def paras_of(text, lo=300, hi=1400):
    """pack description lines into ~lo..hi-char paragraphs."""
    out, cur = [], ""
    for piece in re.split(r"\n+", text):
        piece = piece.strip()
        if not piece:
            continue
        if len(cur) + len(piece) + 1 <= hi:
            cur = (cur + " " + piece).strip()
        else:
            if len(cur) >= lo:
                out.append(cur); cur = piece
            else:  # cur too small to stand alone; hard-append then flush
                cur = (cur + " " + piece).strip()
                out.append(cur[:hi]); cur = cur[hi:]
    if len(cur) >= 80:
        out.append(cur)
    return out


def main():
    t = json.load(open(TARGETS))
    tie, dis = t["tie"], t["disagree"]
    golddoc, elem = t["golddoc"], t["elem"]
    stratum = {u: "tie" for u in tie}; stratum.update({u: "disagree" for u in dis})
    uids = [u for u in stratum if u in golddoc]

    # fetched gold spans (to exclude "elsewhere" paragraphs that are just the same passage)
    fetched = {}
    for ln in open(TESTBED):
        r = json.loads(ln)
        if r["uid"] in stratum and r["y"] == 1:
            fetched[r["uid"]] = r["span"]

    # gold docs' full descriptions
    docs = set(golddoc.values())
    dtext = {}
    for ln in open(DETAIL):
        if not any(d in ln[:80] for d in docs):
            continue
        r = json.loads(ln)
        pid = str(r.get("pgpub_id"))
        if pid in docs and r.get("description_text"):
            dtext[pid] = r["description_text"]
    print(f"[probe] {len(uids)} claims ({len(tie)} tie / {len(dis)} disagree); "
          f"docs with text {len(dtext)}/{len(docs)}", flush=True)

    # paragraph-pack + fetched-span exclusion flags
    dparas = {pid: paras_of(tx) for pid, tx in dtext.items()}
    print(f"[probe] {sum(len(v) for v in dparas.values())} paragraphs total "
          f"(med/doc {int(np.median([len(v) for v in dparas.values()]))})", flush=True)

    # embed (CPU, offline snapshot — GPUs are contended by the 27b rung)
    os.environ["HF_HUB_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(BGE, device="cpu"); m.max_seq_length = 512
    flat, owner = [], []
    for pid, ps in dparas.items():
        for p in ps:
            flat.append(p); owner.append(pid)
    E = m.encode([elem[u][:512] for u in uids], batch_size=16, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=False)
    P = m.encode([p[:512] for p in flat], batch_size=16, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=False)
    print("[probe] embeddings done", flush=True)

    byd = collections.defaultdict(list)
    for i, pid in enumerate(owner):
        byd[pid].append(i)

    # top-3 non-fetched paragraphs per claim
    jobs = []  # (uid, stratum, para_text, cos, kind)
    for k, u in enumerate(uids):
        pid = golddoc[u]
        idx = byd.get(pid, [])
        if not idx:
            continue
        ftk = toks(fetched.get(u, ""))
        cands = []
        for i in idx:
            ptk = toks(flat[i])
            cont = len(ftk & ptk) / max(1, len(ftk)) if ftk else 0.0
            if cont >= 0.7:      # this "paragraph" is (mostly) the already-fetched span
                continue
            cands.append((float(E[k] @ P[i]), i))
        cands.sort(reverse=True)
        for cos, i in cands[:3]:
            jobs.append({"uid": u, "stratum": stratum[u], "para": flat[i], "cos": cos,
                         "kind": "probe"})

    # blinded anchors: pos = STRICT-clean gold spans (GLM already rated them disclosing);
    # neg = a paragraph from a DIFFERENT claim's doc (content mismatch by construction)
    glmh = collections.defaultdict(dict)
    for ln in open(GLMH):
        r = json.loads(ln)
        if r["score"] is not None:
            glmh[r["uid"]][r["y"]] = r["score"]
    strictu = [u for u, d in glmh.items() if d.get(1, 0) >= 2 and d.get(0, 9) == 0][:10]
    sspan = {}
    for ln in open(TESTBED):
        r = json.loads(ln)
        if r["uid"] in strictu and r["y"] == 1:
            sspan[r["uid"]] = r["span"]
    # anchor elements come from the testbed rows
    aelem = {}
    for ln in open(TESTBED):
        r = json.loads(ln)
        if r["uid"] in strictu and r["y"] == 1:
            aelem[r["uid"]] = r["element"]
    for u in strictu:
        if u in sspan and u in aelem:
            jobs.append({"uid": u, "stratum": "anchor_pos", "para": sspan[u], "cos": 1.0,
                         "kind": "anchor", "_elem": aelem[u]})
    rng = np.random.RandomState(7)
    for j in range(10):
        u = uids[int(rng.randint(len(uids)))]
        other = uids[int(rng.randint(len(uids)))]
        if golddoc[other] == golddoc[u]:
            continue
        pool = byd.get(golddoc[other], [])
        if not pool:
            continue
        jobs.append({"uid": u, "stratum": "anchor_neg", "para": flat[pool[int(rng.randint(len(pool)))]],
                     "cos": 0.0, "kind": "anchor"})

    print(f"[probe] {len(jobs)} GLM calls ({sum(j['kind']=='probe' for j in jobs)} probe + "
          f"{sum(j['kind']=='anchor' for j in jobs)} anchors)", flush=True)

    def work(j):
        el = j.get("_elem") or elem.get(j["uid"], "")
        user = (f"CLAIM ELEMENT:\n{el[:800]}\n\nPASSAGE (from a prior-art document):\n"
                f"{j['para'][:1600]}\n\nDoes the passage disclose this claim element in substance? "
                'Reply ONE JSON: {"discloses": true/false, "quote": "<=15 words of the disclosing text or empty"}.')
        v = parse_bool(glm(user))
        return {**{k: j[k] for k in ("uid", "stratum", "cos", "kind")},
                "para": j["para"][:400], "discloses": v}

    out = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, res in enumerate(ex.map(work, jobs)):
            out.append(res)
            if (i + 1) % 50 == 0:
                print(f"[probe] {i+1}/{len(jobs)}", flush=True)
    with open(f"{OUTDIR}/tie_probe_results.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")

    # readout
    ok = [r for r in out if r["discloses"] is not None]
    print(f"\n[coverage] parsed {len(ok)}/{len(out)}", flush=True)
    for s in ("anchor_pos", "anchor_neg"):
        v = [r["discloses"] for r in ok if r["stratum"] == s]
        if v:
            print(f"[anchor] {s}: discloses-rate {np.mean(v):.2f} (n={len(v)})", flush=True)
    summ = {}
    for s in ("tie", "disagree"):
        byu = collections.defaultdict(list)
        for r in ok:
            if r["stratum"] == s:
                byu[r["uid"]].append(bool(r["discloses"]))
        n = len(byu)
        hit = sum(any(v) for v in byu.values())
        summ[s] = {"n_claims": n, "elsewhere_hit": hit, "rate": hit / max(1, n)}
        print(f"[{s.upper()}] {hit}/{n} claims have >=1 GLM-confirmed disclosing paragraph "
              f"ELSEWHERE in the gold doc ({hit/max(1,n):.1%}) -> wrong-passage-fetched rate",
              flush=True)
    json.dump(summ, open(f"{OUTDIR}/tie_probe_summary.json", "w"), indent=1)
    print("TIE_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
