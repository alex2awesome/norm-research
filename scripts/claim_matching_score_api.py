#!/usr/bin/env python3
"""No-GPU claim-matching scoring via GLM (z.ai Anthropic endpoint) — for a quick first VAT signal
and a frontier ceiling anchor when local GPUs are saturated. Used SPARINGLY (GLM quota is binding):
default probe is small. Same output schema as the vLLM scorer so recovery.py reads both.

  python scripts/claim_matching_score_api.py --tag glm --model glm-4.7 --n-claims 120 --max-metrics 0
"""
import argparse, json, re, hashlib, os, time, urllib.request, collections
from concurrent.futures import ThreadPoolExecutor

BASE = "/lfs/skampere3/0/alexspan/norm-research"
BANK = f"{BASE}/datasets/claim-matching/claim_matching_bank.jsonl"
TESTBED = f"{BASE}/datasets/claim-matching/testbed/pair_testbed.jsonl"
OUTDIR = f"{BASE}/outputs/claim_matching"
KEY_FILE = "/lfs/skampere3/0/alexspan/.z-ai-api-key-alexander-spangher.txt"
KEY_FILE2 = "/lfs/skampere3/0/alexspan/.z-ai-api-key.txt"

SYS = ("You compare a CLAIM to a REFERENCE passage using ONE specific matching CRITERION. Judge only "
       "how well the reference satisfies that criterion for that claim. Be strict: surface word-"
       "overlap is not a match; the reference must satisfy the criterion in substance.")
_SCORE = re.compile(r'"?(?:match|score|strength)"?\s*[:=]\s*([0-4])')
_BARE = re.compile(r"\b([0-4])\b")


def parse_score(raw):
    m = _SCORE.search(raw or "")
    if m:
        return int(m.group(1))
    m = _BARE.search((raw or "")[:60])
    return int(m.group(1)) if m else None


def glm(user, model, key_file=KEY_FILE):
    keyfiles = [key_file] + [k for k in (KEY_FILE2, KEY_FILE) if k != key_file]
    body = json.dumps({"model": model, "max_tokens": 40, "temperature": 0.0,
                       "system": SYS, "messages": [{"role": "user", "content": user}]}).encode()
    last = None
    for att in range(8):
        try:
            key = open(keyfiles[att % len(keyfiles)]).read().strip()
            req = urllib.request.Request("https://api.z.ai/api/anthropic/v1/messages", data=body,
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                o = json.loads(r.read())
            return "".join(b.get("text", "") for b in o.get("content", []))
        except Exception as e:
            last = e; time.sleep(min(40, 4 * (att + 1)))
    return ""


def build_prompt(m, e, s):
    crit = f"CRITERION: {m['name']}\n{m['description']}"
    if m.get("guidance"):
        crit += f"\nGuidance: {m['guidance'][:400]}"
    return (f"{crit}\n\nCLAIM (a patent claim element):\n{e[:600]}\n\nREFERENCE passage:\n{s[:1200]}\n\n"
            "Under THIS criterion only, rate how strongly the reference matches/supports/discloses the "
            "claim:\n0=not at all,1=barely,2=partial,3=substantial,4=fully.\n"
            'Reply ONE JSON: {"match": 0-4}.')


def probe(n_claims):
    byu = collections.defaultdict(list)
    for ln in open(TESTBED):
        r = json.loads(ln); byu[r["uid"]].append(r)
    uids = [u for u, v in byu.items() if len(v) == 2 and {x["y"] for x in v} == {0, 1}]
    uids.sort(key=lambda u: hashlib.md5(f"probe::{u}".encode()).hexdigest())
    return [p for u in uids[:n_claims] for p in byu[u]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="glm")
    ap.add_argument("--model", default="glm-4.7")
    ap.add_argument("--n-claims", type=int, default=120, dest="n_claims")
    ap.add_argument("--max-metrics", type=int, default=0, dest="max_metrics")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bank-file", default=BANK, dest="bank_file")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    bank = [json.loads(l) for l in open(a.bank_file)]
    if a.max_metrics:
        bank = bank[:a.max_metrics]
    pairs = probe(a.n_claims)
    jobs = [(m, p) for m in bank for p in pairs]
    print(f"[api:{a.tag}] {len(bank)} metrics x {len(pairs)} pairs = {len(jobs)} GLM calls "
          f"(model {a.model})", flush=True)

    def work(job):
        m, p = job
        sc = parse_score(glm(build_prompt(m, p["element"], p["span"]), a.model))
        return {"metric_id": m["metric_id"], "domain": m["domain"], "uid": p["uid"],
                "y": p["y"], "score": sc}

    out, done = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(work, jobs):
            out.append(res); done += 1
            if done % 500 == 0:
                print(f"[api:{a.tag}] {done}/{len(jobs)}", flush=True)
    with open(f"{OUTDIR}/scores_{a.tag}.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    cov = sum(r["score"] is not None for r in out) / len(out)
    print(f"[api:{a.tag}] done, coverage {cov:.3f} -> {OUTDIR}/scores_{a.tag}.jsonl", flush=True)
    print(f"SCORE_{a.tag}_DONE", flush=True)


if __name__ == "__main__":
    main()
